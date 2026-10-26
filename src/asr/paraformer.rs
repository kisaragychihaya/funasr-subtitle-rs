//! T11：Paraformer-zh。

use std::path::Path;

use crate::asr::{AsrChunkOutput, AsrEngine};
use crate::error::Result;
use crate::types::{EngineKind, JobOptions};

/// Paraformer-large（中文）引擎。
pub struct Paraformer {
    // T11：ort session、词表、前端配置、CMVN
}

impl Paraformer {
    pub fn new(model_dir: &Path, threads: usize) -> Result<Self> {
        let _ = (model_dir, threads);
        todo!("T11")
    }
}

impl AsrEngine for Paraformer {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::Paraformer
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
