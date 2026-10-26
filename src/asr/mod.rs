//! ASR 引擎抽象与三种实现（T8 SenseVoice / T11 Paraformer+热词）。

pub mod contextual;
pub mod paraformer;
pub mod sensevoice;

pub use contextual::ContextualParaformer;
pub use paraformer::Paraformer;
pub use sensevoice::SenseVoice;

use crate::error::Result;
use crate::types::{EngineKind, JobOptions};
use crate::types::TimedPiece;

/// 一个 VAD 语音段的识别输出。
pub enum AsrChunkOutput {
    /// 带真实 CTC 时间的文本片（SenseVoice 路径）。
    Timed(Vec<TimedPiece>),
    /// 整句文本（Paraformer / Contextual，时间由 pipeline 比例分配）。
    Plain(String),
}

/// ASR 引擎统一接口。
pub trait AsrEngine {
    fn engine_kind(&self) -> EngineKind;

    /// 识别一个语音段（16kHz s16 PCM）。
    fn transcribe_chunk(
        &mut self,
        chunk: &[i16],
        sample_rate: u32,
        job: &JobOptions,
    ) -> Result<AsrChunkOutput>;
}
