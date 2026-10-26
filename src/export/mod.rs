//! T2：字幕导出格式（SRT / VTT / TXT / JSON）。
//! 移植自 funasr-subtitle `backend/app/subtitle/formatters.py`，内部组织可自行实现。

use crate::types::Segment;

/// 导出 SRT 文本（含序号、HH:MM:SS,mmm 时间码、空行分隔）。
pub fn to_srt(segments: &[Segment]) -> String {
    todo!("T2: 移植 formatters.py SRT 导出")
}

/// 导出 WebVTT 文本（WEBVTT 头、HH:MM:SS.mmm 时间码）。
pub fn to_vtt(segments: &[Segment]) -> String {
    todo!("T2: 移植 formatters.py VTT 导出")
}

/// 导出纯文本（每句一行）。
pub fn to_txt(segments: &[Segment]) -> String {
    todo!("T2: 移植 formatters.py TXT 导出")
}

/// 导出 JSON 数组（字段与原项目 schemas 一致：start_ms/end_ms/text）。
pub fn to_json(segments: &[Segment]) -> String {
    todo!("T2: 移植 formatters.py JSON 导出")
}
