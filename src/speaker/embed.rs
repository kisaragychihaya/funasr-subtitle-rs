//! CAM++ 说话人嵌入（ONNX）。契约见 `super` 模块文档。
//!
//! 移植来源（逐行对应）：
//! * 特征：`campplus/utils.py::extract_feature` 的 `WavFrontend.fbank` → `pad_feats` 的
//!   utterance 均值减；golden 侧同一流程见 `reference/diar_common.chunk_fbank`。该函数喂的是
//!   `i16 / 32768` 归一化后的采样，且**没有**设置 knf 的 `window_type`（knf 默认 `povey`），
//!   所以这里用 [`extract_features_f32`] + [`WindowType::Povey`]，而不是 SenseVoice 那条
//!   `hamming` + 原始 i16 幅度的入口（两者差异远大于 golden 容差：仅窗函数一项余弦就掉到
//!   0.958，再加采样缩放会掉到 0.812）。
//! * 推理：golden `diar_common.embed` = `session.run(None, {"fbank": feat[None, ...]})[0]`，
//!   输入 (1,T,80) f32、输出 (1,192) f32。
//!
//! 均值减按 float32 做（numpy `mat.mean(axis=0)` 对 float32 输入也返回 float32）；
//! 顺序累加与 numpy 的成对求和差 ULP 级，被 golden 的比对容差（余弦 ≥ 1-1e-4）覆盖。

use std::path::Path;

use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;

use crate::error::{Error, Result};
use crate::frontend::{FrontendConfig, WindowType, extract_features_f32};

/// 模型文件名（`SPK_MODEL.direct` 直链下载到模型目录的文件名）。
const MODEL_FILE: &str = "campplus.onnx";
/// ONNX 输入名（与 golden 侧同一个名字）。
const INPUT_NAME: &str = "fbank";
/// ONNX 输出名。
const OUTPUT_NAME: &str = "embedding";
/// 前端采样率（cam++ 只按 16k 建模）。
const SAMPLE_RATE: u32 = 16_000;
/// s16 → f32 的归一化分母（golden `pcm_to_float`：`pcm / 32768.0`）。
const PCM_SCALE: f32 = 32_768.0;
/// 嵌入维数（cam++ 的 192 维 x-vector）。
const EMBED_DIM: usize = 192;

/// cam++ 前端参数（`extract_feature` 的 `frontend_conf`：n_mels=80、25/10ms、preemph 0.97、
/// dither=0、无 LFR——与 SenseVoice 的差别只在 lfr；窗函数见 [`WindowType::Povey`]）。
fn campplus_frontend() -> FrontendConfig {
    FrontendConfig {
        n_mels: 80,
        frame_length_ms: 25.0,
        frame_shift_ms: 10.0,
        preemphasis: 0.97,
        dither: 0.0,
        lfr_m: 1,
        lfr_n: 1,
    }
}

/// CAM++ 嵌入器：补零到满窗的 16k 单声道 i16 chunk → 192 维嵌入。
///
/// 内部流水线：knf 语义 fbank(80, dither=0)（复用 `crate::frontend`，
/// lfr_m=lfr_n=1、无 CMVN）→ 按维度时间均值减（f32）→ ONNX 推理。
pub struct SpeakerEmbedder {
    session: Session,
    /// 前端参数（fbank 部分；LFR/CMVN 都不用）。
    cfg: FrontendConfig,
}

impl SpeakerEmbedder {
    /// 加载 `<model_dir>/campplus.onnx`（`SPK_MODEL.direct` 下载到的目录）。
    pub fn new(model_dir: &Path, threads: usize) -> Result<Self> {
        let model_file = model_dir.join(MODEL_FILE);
        if !model_file.is_file() {
            return Err(Error::Model(format!(
                "CAM++ 模型文件不存在: {}",
                model_file.display()
            )));
        }
        // 环境共享自 crate::ort_util（ort 每进程只允许一个 Environment）
        let session = crate::ort_util::build_session(&model_file, threads)?;

        // 输入/输出名是契约（golden 侧按同一对名字喂/取），建会话时就核对，
        // 免得推理时才发现接错。
        if session.inputs().len() != 1 || session.inputs()[0].name() != INPUT_NAME {
            let names: Vec<&str> = session
                .inputs()
                .iter()
                .map(|outlet| outlet.name())
                .collect();
            return Err(Error::Model(format!(
                "CAM++ 模型输入 {names:?}，期望单个 {INPUT_NAME:?}"
            )));
        }
        if !session
            .outputs()
            .iter()
            .any(|outlet| outlet.name() == OUTPUT_NAME)
        {
            let names: Vec<&str> = session
                .outputs()
                .iter()
                .map(|outlet| outlet.name())
                .collect();
            return Err(Error::Model(format!(
                "CAM++ 模型输出 {names:?}，期望含 {OUTPUT_NAME:?}"
            )));
        }

        Ok(Self {
            session,
            cfg: campplus_frontend(),
        })
    }

    /// 单个满窗 chunk（`CHUNK_LEN` 采样，可更短——内部不补零，由调用方按 sv_chunk 语义补）→ 192 维。
    pub fn embed(&mut self, samples: &[i16]) -> Result<Vec<f32>> {
        // 归一化到 [-1, 1)（golden `pcm_to_float`）后提特征：fbank(80, povey) → 均值减
        let scaled: Vec<f32> = samples.iter().map(|v| f32::from(*v) / PCM_SCALE).collect();
        let features =
            extract_features_f32(&scaled, SAMPLE_RATE, &self.cfg, None, WindowType::Povey)?;
        let feats = mean_subtract(&features.feats);
        let (rows, cols) = feats.dim();
        if rows == 0 || cols == 0 {
            return Err(Error::Inference(format!(
                "CAM++ 输入特征为空: {rows}x{cols}"
            )));
        }

        let mut flat = Vec::with_capacity(rows * cols);
        flat.extend(feats.iter().copied());
        let input = Value::from_array((vec![1_i64, rows as i64, cols as i64], flat))
            .map_err(|e| Error::Inference(format!("构造 fbank 张量失败: {e}")))?
            .into_dyn();

        let outputs = self
            .session
            .run(vec![(INPUT_NAME.to_string(), input)])
            .map_err(|e| Error::Inference(format!("CAM++ 推理失败: {e}")))?;

        let Some(output) = outputs.get(OUTPUT_NAME) else {
            return Err(Error::Inference(format!(
                "CAM++ 推理结果缺少输出 {OUTPUT_NAME:?}"
            )));
        };
        let (shape, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Inference(format!("读取 embedding 失败: {e}")))?;
        if data.len() != EMBED_DIM {
            return Err(Error::Inference(format!(
                "CAM++ 输出 {shape:?} 的元素数 {}，期望 {EMBED_DIM}",
                data.len()
            )));
        }
        Ok(data.to_vec())
    }
}

/// `mat - mat.mean(axis=0, keepdims=True)`：按列（mel 维）减时间均值。
///
/// numpy 侧 mat 是 float32，均值也用 float32 累加后相减，这里保持一致
/// （列和 / 行数；与 numpy 的成对求和差 ULP 级，见模块文档）。
fn mean_subtract(feats: &Array2<f32>) -> Array2<f32> {
    let (rows, cols) = feats.dim();
    let mut out = feats.to_owned();
    if rows == 0 || cols == 0 {
        return out;
    }
    let mut mean = vec![0.0f32; cols];
    for row in feats.rows() {
        for (slot, value) in mean.iter_mut().zip(row.iter()) {
            *slot += *value;
        }
    }
    let denom = rows as f32;
    for slot in mean.iter_mut() {
        *slot /= denom;
    }
    for mut row in out.rows_mut() {
        for (slot, mu) in row.iter_mut().zip(mean.iter()) {
            *slot -= *mu;
        }
    }
    out
}
