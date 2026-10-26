//! 切句核心逻辑。移植来源：funasr-subtitle `backend/app/subtitle/segmentation.py`。
//! 显示宽度规则：CJK=1，拉丁/数字/其它非空白=0.5，空白不计。

use crate::types::{Segment, TimedPiece};

/// 移植 `split_sentences`：切成自然语句；过长的句子在次级标点处贪心切分到 <= max_chars。
/// max_chars=0 表示不限制宽度。
pub fn split_sentences(text: &str, max_chars: usize) -> Vec<String> {
    todo!("T1: 移植 segmentation.py split_sentences")
}

/// 移植 `segment_timed`：用真实 token 时间切句（SenseVoice CTC 时间戳路径）。
/// 每句 start=首 token 时间，end=下一句首 token 时间（末句=span_end_ms）。
pub fn segment_timed(
    pieces: &[TimedPiece],
    span_end_ms: i64,
    max_chars: usize,
    max_cue_ms: i64,
) -> Vec<Segment> {
    todo!("T1: 移植 segmentation.py segment_timed")
}

/// 移植 `segment_span`：把一个 VAD 语音段的文本切成逐句字幕，按显示宽度比例分配时间。
pub fn segment_span(text: &str, start_ms: i64, end_ms: i64, max_chars: usize) -> Vec<Segment> {
    todo!("T1: 移植 segmentation.py segment_span")
}

/// 移植 `enforce_monotonic`：保证 cue 单调不重叠（start 不早于上一条 end，end > start）。
pub fn enforce_monotonic(segments: &mut [Segment]) {
    todo!("T1: 移植 segmentation.py enforce_monotonic")
}

/// 移植 `strip_terminal_punct`：删句末终止标点（保留逗号/冒号/小数点/缩写内点）。
/// 英文句号仅当其后是空格/结尾时才算句末删除。
pub fn strip_terminal_punct(text: &str) -> String {
    todo!("T1: 移植 segmentation.py strip_terminal_punct")
}
