//! T6：声学特征前端（fbank + LFR + CMVN）。
//! 移植来源：funasr-onnx vendored `sensevoice_bin.py` / `paraformer_bin.py` 的 extract_feat，
//! 以及 FunASR kaldi 式 fbank 实现。参数一律以 vendored 源码为准，禁止猜测。

use std::path::Path;

use ndarray::Array2;

use crate::error::Result;

/// 前端配置（各字段取值由 T6 依据 vendored 源码确定；调用方使用下方预设函数）。
#[derive(Debug, Clone)]
pub struct FrontendConfig {
    pub n_mels: usize,
    pub frame_length_ms: f32,
    pub frame_shift_ms: f32,
    /// 预加重系数；0.0 = 关闭。
    pub preemphasis: f32,
    pub dither: f32,
    /// LFR 叠帧参数（m=叠帧数, n=跳帧数）。
    pub lfr_m: usize,
    pub lfr_n: usize,
}

/// CMVN 统计量（读自模型目录 am.mvn）。
#[derive(Debug, Clone)]
pub struct Cmvn {
    pub mean: Vec<f32>,
    pub istd: Vec<f32>,
}

/// 提取完成的特征：feats 形状 (T, n_mels * lfr_m)，feats_len = T。
#[derive(Debug, Clone)]
pub struct Features {
    pub feats: Array2<f32>,
    pub feats_len: usize,
}

/// 读 am.mvn（Kaldi 文本格式 mean/istd）。
pub fn load_cmvn(path: &Path) -> Result<Cmvn> {
    let _ = path;
    todo!("T6")
}

/// 提取特征：fbank(80维) → LFR 叠帧 → CMVN（顺序以 vendored 源码为准）。
/// wave 为 s16 原始采样；数值须与 golden feats 对齐（atol 1e-4）。
pub fn extract_features(
    wave: &[i16],
    sample_rate: u32,
    cfg: &FrontendConfig,
    cmvn: Option<&Cmvn>,
) -> Result<Features> {
    let _ = (wave, sample_rate, cfg, cmvn);
    todo!("T6")
}

/// SenseVoiceSmall 前端参数预设（以 vendored sensevoice_bin.py 为准）。
pub fn sensevoice_frontend() -> FrontendConfig {
    todo!("T6")
}

/// Paraformer-large 前端参数预设（以 vendored paraformer_bin.py 为准）。
pub fn paraformer_frontend() -> FrontendConfig {
    todo!("T6")
}
