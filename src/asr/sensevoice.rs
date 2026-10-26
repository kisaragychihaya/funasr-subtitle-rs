//! T8：SenseVoiceSmall。移植来源：funasr-subtitle `engine/funasr_engine.py` 的
//! _sensevoice_timed / _ctc_greedy_with_frames，以及 vendored `sensevoice_bin.py` 的
//! extract_feat / infer / _get_lid / _get_tnid。

use std::path::Path;

use crate::asr::{AsrChunkOutput, AsrEngine};
use crate::error::Result;
use crate::types::{EngineKind, JobOptions};

/// SenseVoiceSmall 引擎（量化 ONNX，纯 CPU）。
pub struct SenseVoice {
    // T8：ort session、sentencepiece piece 表、前端配置、CMVN、lid/tnid 映射
}

impl SenseVoice {
    pub fn new(model_dir: &Path, threads: usize) -> Result<Self> {
        let _ = (model_dir, threads);
        todo!("T8")
    }
}

impl AsrEngine for SenseVoice {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::SenseVoice
    }

    fn transcribe_chunk(
        &mut self,
        chunk: &[i16],
        sample_rate: u32,
        job: &JobOptions,
    ) -> Result<AsrChunkOutput> {
        let _ = (chunk, sample_rate, job);
        todo!("T8")
    }
}
