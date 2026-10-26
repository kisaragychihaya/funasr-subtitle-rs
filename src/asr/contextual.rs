//! T11：Paraformer contextual（热词版）。

use std::path::Path;

use crate::asr::{AsrChunkOutput, AsrEngine};
use crate::error::Result;
use crate::types::{EngineKind, JobOptions};

/// ContextualParaformer（热词增强）。
pub struct ContextualParaformer {
    // T11：ort session（含 model_eb_quant.onnx 热词 embedding）、词表、热词打分参数
}

impl ContextualParaformer {
    pub fn new(model_dir: &Path, threads: usize) -> Result<Self> {
        let _ = (model_dir, threads);
        todo!("T11")
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
        let _ = (chunk, sample_rate, job);
        todo!("T11")
    }
}
