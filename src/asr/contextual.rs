//! T11：ContextualParaformer（热词增强版 Paraformer）。
//!
//! 移植来源 `reference/funasr_onnx/paraformer_bin.py::ContextualParaformer`：
//! `proc_hotword` → `eb_infer` → 取 `bias_embed` → `bb_infer(feats, feats_len, bias_embed)`，
//! 解码与后处理与 `Paraformer` 完全相同（Python 侧 `decode/decode_one` 是直接复制的）。
//! `funasr_engine.py` 额外打的 `language = "zh"` 补丁在这里不必存在：本移植对 `en-bpe`
//! 之外的模型统一走 `sentence_postprocess`（`en-bpe` 在加载期直接拒绝）。
//!
//! 两个模型文件：`model_quant.onnx`（bb，声学模型）与 `model_eb_quant.onnx`（eb，热词
//! embedding；仓库里只提供 `model_eb.onnx`，由 `ModelManager` 的 `local_copies` 复制改名）。

use std::path::Path;

use ort::session::Session;
use ort::value::{DynValue, Value};

use crate::asr::paraformer::{BiasEmbed, ParaformerCore, TokenList, build_session, sentence_text};
use crate::asr::{AsrChunkOutput, AsrEngine};
use crate::error::{Error, Result};
use crate::types::{EngineKind, JobOptions};

/// 热词 embedding 模型（`quantize=True` 时的文件名）。
const EB_MODEL_FILE: &str = "model_eb_quant.onnx";
/// `pad_list(hotwords, pad_value=0, max_len=10)` 的列数（`proc_hotword` 硬编码）。
const HOTWORD_MAX_LEN: usize = 10;
/// `proc_hotword` 里 OOV 字符固定换成 8403（源码里的字面量，不是 `<unk>` 的 id）。
const HOTWORD_OOV_ID: i32 = 8403;
/// 热词矩阵最后补的 `<s>` 行（`hotword_int.append(np.array([1]))`）。
const HOTWORD_SOS_ID: i32 = 1;

/// `proc_hotword` 的输出：N×10 的 token id 矩阵 + 每行真实长度（长度最后多一个 0）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HotwordBatch {
    pub ids: Vec<Vec<i32>>,
    pub lengths: Vec<i32>,
}

/// 移植 `ContextualParaformer.proc_hotword`。
///
/// 逐行对照源码：`hotwords.split(" ")`（空串会得到 `[""]`，长度 -1，与 golden 的
/// `hotword_lengths` 一致）→ 每行逐字符查 `vocab`（缺失落 8403）→ 追加 `<s>` 行 →
/// 用 0 右补齐到 10 列；`hotwords_length = [len(word) - 1, ..., 0]`。
pub fn proc_hotword(tokens: &TokenList, hotwords: &str) -> Result<HotwordBatch> {
    let words: Vec<&str> = hotwords.split(' ').collect();
    let mut lengths: Vec<i32> = words
        .iter()
        .map(|word| word.chars().count() as i32 - 1)
        .collect();
    lengths.push(0);

    let mut rows: Vec<Vec<i32>> = Vec::with_capacity(words.len() + 1);
    for word in &words {
        let mut row: Vec<i32> = Vec::with_capacity(word.chars().count());
        for c in word.chars() {
            let token = c.to_string();
            // Python 的 `self.vocab[c]` 找不到就写死 8403，不走 TokenIDConverter 的 unk_id
            let id = match tokens.id_of(&token) {
                Some(idx) => idx as i32,
                None => HOTWORD_OOV_ID,
            };
            row.push(id);
        }
        if row.len() > HOTWORD_MAX_LEN {
            return Err(Error::InvalidInput(format!(
                "热词 {word:?} 有 {} 个字符，超过 pad_list 的 max_len={HOTWORD_MAX_LEN}",
                row.len()
            )));
        }
        rows.push(row);
    }
    rows.push(vec![HOTWORD_SOS_ID]);

    let ids = rows
        .into_iter()
        .map(|mut row| {
            row.resize(HOTWORD_MAX_LEN, 0);
            row
        })
        .collect();
    Ok(HotwordBatch { ids, lengths })
}

/// 热词 embedding 模型（`model_eb_quant.onnx`）。
struct EbSession {
    session: Session,
    /// 模型声明的输入名。本模型只有 1 个（`hotword`）；Python 的 `zip` 会把第二个值
    /// （`hotwords_length`）丢掉，长度只用于后面的 `bias_embed` 取行。
    input_names: Vec<String>,
}

impl EbSession {
    fn load(model_dir: &Path, threads: usize) -> Result<Self> {
        let path = model_dir.join(EB_MODEL_FILE);
        if !path.is_file() {
            return Err(Error::Model(format!(
                "热词 embedding 模型不存在: {}",
                path.display()
            )));
        }
        let session = build_session(&path, threads)?;
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|input| input.name().to_string())
            .collect();
        if input_names.is_empty() {
            return Err(Error::Model(format!(
                "热词 embedding 模型 {} 没有输入",
                path.display()
            )));
        }
        if session.outputs().is_empty() {
            return Err(Error::Model(format!(
                "热词 embedding 模型 {} 没有输出",
                path.display()
            )));
        }
        Ok(Self {
            session,
            input_names,
        })
    }

    /// `eb_infer` + `bias_embed = bias_embed.transpose(1, 0, 2)[arange(N), lengths]`。
    ///
    /// 模型的 `hw_embed` 输出形状是 `[10, N, dim]`（已按 `[位置, 热词行, 维度]` 排好），
    /// Python 侧再转置成 `[N, 10, dim]` 后用 `lengths` 取行，等价于
    /// `result[i] = hw_embed[lengths[i], i, :]`（长度按 Python 负索引语义回绕）。
    fn bias_embed(&mut self, batch: &HotwordBatch) -> Result<BiasEmbed> {
        let rows = batch.ids.len();
        let mut flat: Vec<i32> = Vec::with_capacity(rows * HOTWORD_MAX_LEN);
        for row in &batch.ids {
            flat.extend_from_slice(row);
        }
        let hotword = Value::from_array((vec![rows as i64, HOTWORD_MAX_LEN as i64], flat))
            .map_err(|e| Error::Inference(format!("构造 hotword 张量失败: {e}")))?
            .into_dyn();
        let lengths = Value::from_array((vec![batch.lengths.len() as i64], batch.lengths.clone()))
            .map_err(|e| Error::Inference(format!("构造 hotwords_length 张量失败: {e}")))?
            .into_dyn();

        // `zip(get_input_names(), [hotwords, hotwords_length])`：模型声明几个输入就喂几个
        let inputs: Vec<(String, DynValue)> = self
            .input_names
            .iter()
            .cloned()
            .zip([hotword, lengths])
            .collect();
        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| Error::Inference(format!("热词 embedding 推理失败: {e}")))?;
        let (shape, raw) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Inference(format!("读取 hw_embed 失败: {e}")))?;
        let dims: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
        if dims.len() != 3 || dims[0] == 0 || dims[1] == 0 || dims[2] == 0 {
            return Err(Error::Inference(format!("hw_embed 维度异常: {dims:?}")));
        }
        let (positions, n, dim) = (dims[0], dims[1], dims[2]);
        if n != batch.lengths.len() {
            return Err(Error::Inference(format!(
                "hw_embed 的热词行数 {n} 与 hotwords_length 长度 {} 不一致",
                batch.lengths.len()
            )));
        }
        if raw.len() < positions * n * dim {
            return Err(Error::Inference(format!(
                "hw_embed 数据大小异常: {} < {}",
                raw.len(),
                positions * n * dim
            )));
        }

        let mut data: Vec<f32> = Vec::with_capacity(n * dim);
        for (i, len) in batch.lengths.iter().enumerate() {
            let mut pos = *len as isize;
            if pos < 0 {
                pos += positions as isize;
            }
            if pos < 0 || pos >= positions as isize {
                return Err(Error::Inference(format!(
                    "热词长度 {len} 超出发音位置范围 0..{positions}"
                )));
            }
            let start = pos as usize * n * dim + i * dim;
            data.extend_from_slice(&raw[start..start + dim]);
        }
        Ok(BiasEmbed { data, rows: n, dim })
    }
}

/// ContextualParaformer（热词增强）。
pub struct ContextualParaformer {
    core: ParaformerCore,
    eb: EbSession,
}

impl ContextualParaformer {
    pub fn new(model_dir: &Path, threads: usize) -> Result<Self> {
        // 先加载 bb（词表/前端/CMVN 都在 core 里），再加载 eb
        Ok(Self {
            core: ParaformerCore::load(model_dir, threads)?,
            eb: EbSession::load(model_dir, threads)?,
        })
    }

    /// 实例词表上的 `proc_hotword`（等价于 Python 的 `self.proc_hotword`）。
    pub fn proc_hotword(&self, hotwords: &str) -> Result<HotwordBatch> {
        proc_hotword(self.core.tokens(), hotwords)
    }
}

impl AsrEngine for ContextualParaformer {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::Contextual
    }

    fn transcribe_chunk(
        &mut self,
        chunk: &[i16],
        sample_rate: u32,
        job: &JobOptions,
    ) -> Result<AsrChunkOutput> {
        // `_asr_paraformer` 传的是 `options.hotwords or ""`（空串也照常走 proc_hotword，
        // golden 的 without_hotwords 就是这条路径）
        let batch = self.proc_hotword(&job.hotwords)?;
        let bias = self.eb.bias_embed(&batch)?;
        let tokens = self
            .core
            .transcribe_tokens(chunk, sample_rate, Some(&bias))?;
        Ok(AsrChunkOutput::Plain(sentence_text(&tokens)?))
    }
}
