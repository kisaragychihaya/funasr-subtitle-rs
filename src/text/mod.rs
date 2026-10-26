//! T1：字幕切分与文本规范化。
//! 移植自 funasr-subtitle `backend/app/subtitle/segmentation.py`，须逐行对照行为移植。

pub mod normalize;
pub mod segmentation;

pub use segmentation::{
    enforce_monotonic, segment_span, segment_timed, split_sentences, strip_terminal_punct,
};
