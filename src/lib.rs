//! funasr-subtitle-rs：FunASR ONNX 模型（SenseVoice / Paraformer）离线字幕生成，纯 CPU。
//! 本项目是 rockbenben/funasr-subtitle 的 Rust CLI 移植；`types`/`error` 为冻结契约，实现任务不得修改。

pub mod error;
pub mod types;

pub mod asr;
pub mod export;
pub mod frontend;
pub mod media;
pub mod models;
pub mod ort_util;
pub mod pipeline;
pub mod punc;
pub mod text;
pub mod vad;

pub use error::{Error, Result};
pub use types::{EngineKind, JobOptions, Language, PunctMode, Segment, TimedPiece};
