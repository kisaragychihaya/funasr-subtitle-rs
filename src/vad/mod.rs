//! T7：FSMN-VAD。移植来源：funasr-onnx vendored `vad_bin.py`。
//! 分块推理、状态 cache、段合并阈值须逐行对照源码移植。

use std::path::Path;

use crate::error::Result;

/// FSMN-VAD 离线语音活动检测。
pub struct Vad {
    // T7：ort session、前端配置、CMVN、合并阈值参数等
}

impl Vad {
    /// 从模型目录加载（model_quant.onnx + am.mvn）。
    pub fn new(model_dir: &Path, threads: usize) -> Result<Self> {
        let _ = (model_dir, threads);
        todo!("T7")
    }

    /// 对整段音频做 VAD，返回语音段时间 [(start_ms, end_ms), ...]。
    pub fn infer(&mut self, wave: &[i16], sample_rate: u32) -> Result<Vec<(i64, i64)>> {
        let _ = (wave, sample_rate);
        todo!("T7")
    }
}
