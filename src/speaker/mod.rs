//! 说话人分离（cam++ 嵌入 + 谱聚类 + 后处理 + 分配）。
//!
//! 算法契约（冻结；实现任务不得偏离）：
//! * 全部逻辑移植自 FunASR `funasr/models/campplus/`（`utils.py` / `cluster_backend.py`）
//!   与 `auto_model.py` 的 `inference_with_vad` 说话人分支；上游 full 引擎可见
//!   `reference/funasr-subtitle/backend/app/engine/funasr_full_engine.py`。
//! * golden 基准：`tests/golden/diar/diar_two_speakers/`（由
//!   `reference/dump_golden_diar.py` 生成；**重新生成 golden 必须先改脚本并说明理由**）。
//!
//! 切窗（sv_chunk，对应 `utils.sv_chunk`）：
//! * 每 VAD 段独立切窗：窗长 `CHUNK_LEN = 1.5s = 24000` 采样、窗移 `CHUNK_SHIFT = 0.75s = 12000`。
//! * 尾部窗不足窗长时**前移窗口使其贴段尾**（`chunk_st = max(0, chunk_ed - CHUNK_LEN)`），
//!   再对采样**补零到满窗**；`chunk_ed <= last_chunk_ed` 时终止。
//! * 时间戳为**秒**（`chunk_* / 16000 + seg_st`，seg_st = 段起点毫秒 / 1000）。
//!
//! 特征（对应 `utils.extract_feature`；fbank 参数与 ASR 前端同族但**入口不同**）：
//! * `crate::frontend` 无 LFR(lfr_m=lfr_n=1)、无 CMVN 的 80 维 fbank（dither=0），
//!   再对每个 chunk 做**按维度的时间均值减**（utterance mean-subtract），f32。
//! * golden 喂给 knf 的是 `i16 / 32768` 归一化采样且用 knf 默认的 **povey 窗**
//!   （`reference/diar_common.chunk_fbank` 未设 `window_type`）；SenseVoice/Paraformer
//!   前端则是原始 i16 幅度 + hamming（`config.yaml` 显式指定）。嵌入链路必须走
//!   `extract_features_f32(.., WindowType::Povey)`，不能用 `extract_features`，
//!   否则相似度掉两个量级（实测仅窗函数一项余弦即跌到 0.958，见 `embed.rs` 头注释）。
//!
//! 嵌入（`embed.rs`）：
//! * cam++ ONNX（`SPK_MODEL.direct` 直链；输入 `fbank`(1,T,80) f32，输出 (1,192) f32）。
//! * 与 golden 比对用**容差**（ORT/数值序差异）：逐 chunk 余弦 ≥ 1 - 1e-4。
//!
//! 聚类（`cluster.rs`，对应 `cluster_backend.ClusterBackend.forward` + `SpectralCluster`）：
//! * N < 20 个嵌入 → 全 0 标签（上游原行为，短音频单说话人）。
//! * N < 2048 → 谱聚类：余弦相似度 → p 剪枝（pval=0.022；N*pval < 6 时 pval = 6/N；
//!   每行保留最大的 `N - int((1-pval)*N)` 个，其余清零）→ 0.5*(A+Aᵀ) 对称化 →
//!   对角清零 → 未归一化 Laplacian L = D - A（D = 行绝对值和的对角阵）→
//!   取前 min(16, N) 个最小特征值/向量（升序）→ 自动模式取
//!   `argmax(diff(λ[0..16])) + 1` 个说话人（oracle 模式用 `preset_spk_num`）→
//!   对特征向量做定种子 k-means（见下）→ 自动模式再做 merge_by_cos(0.78)。
//! * N ≥ 2048 → 上游用 umap+hdbscan（不移植）；实现为回退谱聚类（算法对 N≤2048 已验证，
//!   更大 N 时谱分解变慢但结果仍正确），并在文档注释中说明。
//! * **定种子 k-means 契约**（上游 sklearn 无种子不可复现，golden 采用下列确定性算法，
//!   Rust 必须与 golden 的 numpy 参考实现逐次一致）：n_init=10；每次 init 用
//!   **MT19937(种子 0)**（标准 32 位 MT19937，双精度取 `(a*67108864+b)/9007199254740992`，
//!   a=高 27 位? 见 golden 脚本 `mt19937_double`）；k-means++ 初始化
//!   （`randint(n)=floor(u*n)`；按 `d²` 概率选点，`choice=cumsum(d²) 首个 ≥ u*sum(d²)` 的下标）；
//!   Lloyd 迭代 ≤ 300 轮，标签不变即停；10 轮中取 inertia 最小（平局取先达）。
//!
//! 后处理（对应 `utils.postprocess`，`diarize.rs`）：
//! * correct_labels：按**首次出现**重编号为 0..k。
//! * merge_seque：相邻同说话人且 `cur.start <= prev.end` 时合并（end 取大）。
//! * 重叠处理：上一条 end > 当前 start + 1e-4 时，两者都置为中点。
//! * smooth(mindur=0.7)：时间**先** round 两位小数（**half-even**，Python round 语义，
//!   不是 Rust f64::round 的 half-away），时长 < 0.7s 的段继承最近邻居（首尾单邻居，
//!   距离相等取左侧），再 merge_seque。
//!
//! 分配（对应 `utils.distribute_spk`，`diarize.rs`）：
//! * 每 VAD 段（毫秒）对各说话人时间线（秒×1000）求重叠和，取最大者；
//!   无重叠 → 0；并列取时间线中先出现的说话人。

pub mod cluster;
pub mod diarize;
pub mod embed;

/// sv_chunk 窗长（1.5s @16k）。
pub const CHUNK_LEN: usize = 24_000;
/// sv_chunk 窗移（0.75s @16k）。
pub const CHUNK_SHIFT: usize = 12_000;

/// 谱聚类参数（对应 FunASR `SpectralCluster` / `ClusterBackend` 默认值）。
pub const MIN_NUM_SPKS: usize = 1;
/// `max_num_spks`：eigengap 搜索上界（特征值取前 `MAX_NUM_SPKS + 1` 个）。
pub const MAX_NUM_SPKS: usize = 15;
/// p 剪枝比例（N*pval < 6 时抬到 6/N）。
pub const PVAL: f64 = 0.022;
/// 自动模式下的说话人合并阈值（余弦）。
pub const MERGE_THR: f64 = 0.78;
/// smooth 的最短段时长（秒）。
pub const SMOOTH_MIN_DUR: f64 = 0.7;
/// 少于该数量的嵌入直接判定单说话人（FunASR 原行为）。
pub const MIN_CHUNKS_FOR_CLUSTER: usize = 20;
/// 谱聚类实现验证过的规模上界（N ≥ 时上游用 umap+hdbscan，本实现回退谱聚类）。
pub const SPECTRAL_MAX_N: usize = 2048;
