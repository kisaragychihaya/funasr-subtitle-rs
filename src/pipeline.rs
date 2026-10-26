//! T10：转写管线编排。复刻 funasr-subtitle `engine/funasr_engine.py` 的
//! FunasrEngine.transcribe 全编排（VAD → 逐段 ASR → 标点三模式 → 切句 → 单调化）。

use std::path::PathBuf;

use crate::error::Result;
use crate::media::Audio;
use crate::types::{JobOptions, Segment};

/// 引擎构造选项。
pub struct EngineOptions {
    /// 模型 id（registry 中的稳定 id，默认 DEFAULT_MODEL_ID）。
    pub model_id: String,
    pub models_root: PathBuf,
    pub num_threads: usize,
}

/// 转写引擎（懒加载：首次 transcribe 时才下载/加载模型）。
pub struct Engine {
    // T10：ModelManager、Vad、AsrEngine、Punctuator、ModelSpec
}

/// 进度回调：stage ∈ vad/asr/punc/assembling，percent ∈ [0,100]。
pub type ProgressFn<'a> = &'a dyn Fn(&str, f32);

impl Engine {
    pub fn new(opts: EngineOptions) -> Result<Self> {
        let _ = opts;
        todo!("T10")
    }

    /// 整段转写。progress 回调 (stage, percent)。
    pub fn transcribe(
        &mut self,
        audio: &Audio,
        job: &JobOptions,
        progress: Option<ProgressFn<'_>>,
    ) -> Result<Vec<Segment>> {
        let _ = (audio, job, progress);
        todo!("T10")
    }
}
