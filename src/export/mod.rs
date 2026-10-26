//! T2：字幕导出格式（SRT / VTT / TXT / JSON）。
//! 移植自 funasr-subtitle `backend/app/subtitle/formatters.py`，内部组织可自行实现。
//!
//! 与 Python 实现逐字节对齐的要点：
//! - 时间码 `HH:MM:SS,mmm`（SRT）/ `HH:MM:SS.mmm`（VTT），小时两位起、不进位截断；
//! - 毫秒转时间码只做 `divmod`，不四舍五入；负值一律钳到 0；
//! - cue 文本先折叠「含换行的空白段」为单空格再 strip（SRT/VTT/TXT），JSON 只 strip；
//! - SRT、VTT 末尾各有一个换行，TXT 无末尾换行，JSON 是 `indent=2` 多行数组且无末尾换行；
//! - JSON 字段 `start_ms` / `end_ms` / `text`，非 ASCII 原样输出（`ensure_ascii=False`）。

use crate::text::normalize::is_py_space;
use crate::types::Segment;

const HOUR_MS: i64 = 3_600_000;
const MINUTE_MS: i64 = 60_000;
const SECOND_MS: i64 = 1_000;

/// 导出 SRT 文本（含序号、HH:MM:SS,mmm 时间码、空行分隔）。
pub fn to_srt(segments: &[Segment]) -> String {
    let mut blocks: Vec<String> = Vec::with_capacity(segments.len());
    for (i, seg) in segments.iter().enumerate() {
        let start = fmt_ts(seg.start_ms, ',');
        let end = fmt_ts(seg.end_ms, ',');
        let text = cue_text(seg);
        let index = i + 1;
        blocks.push(format!("{index}\n{start} --> {end}\n{text}\n"));
    }
    blocks.join("\n")
}

/// 导出 WebVTT 文本（WEBVTT 头、HH:MM:SS.mmm 时间码）。
pub fn to_vtt(segments: &[Segment]) -> String {
    let mut out: Vec<String> = vec![String::from("WEBVTT"), String::new()];
    for seg in segments {
        let start = fmt_ts(seg.start_ms, '.');
        let end = fmt_ts(seg.end_ms, '.');
        out.push(format!("{start} --> {end}"));
        out.push(cue_text(seg));
        out.push(String::new());
    }
    out.join("\n")
}

/// 导出纯文本（每句一行）。
pub fn to_txt(segments: &[Segment]) -> String {
    let lines: Vec<String> = segments.iter().map(cue_text).collect();
    lines.join("\n")
}

/// 导出 JSON 数组（字段与原项目 schemas 一致：start_ms/end_ms/text[/speaker]）。
///
/// `speaker` 仅在存在时输出，键序对齐原实现
/// `{**base, **({"speaker": seg.speaker} if seg.speaker else {})}`（在 text 之后）；
/// 文本沿用 `json.dumps(..., ensure_ascii=False, indent=2)` 的排布与转义规则。
pub fn to_json(segments: &[Segment]) -> String {
    if segments.is_empty() {
        return String::from("[]");
    }
    let mut out = String::from("[\n");
    for (i, seg) in segments.iter().enumerate() {
        if i > 0 {
            out.push_str(",\n");
        }
        out.push_str("  {\n");
        out.push_str(&format!("    \"start_ms\": {},\n", seg.start_ms));
        out.push_str(&format!("    \"end_ms\": {},\n", seg.end_ms));
        out.push_str("    \"text\": \"");
        push_json_string(&mut out, py_strip(&seg.text));
        out.push('\"');
        if let Some(spk) = &seg.speaker
            && !spk.is_empty()
        {
            out.push_str(",\n    \"speaker\": \"");
            push_json_string(&mut out, spk);
            out.push('\"');
        }
        out.push_str("\n  }");
    }
    out.push_str("\n]");
    out
}

/// 毫秒 -> `HH:MM:SS<sep>mmm`；负数与 0 同样输出 `00:00:00<sep>000`。
fn fmt_ts(ms: i64, sep: char) -> String {
    let ms = ms.max(0);
    let h = ms / HOUR_MS;
    let m = ms % HOUR_MS / MINUTE_MS;
    let s = ms % MINUTE_MS / SECOND_MS;
    let milli = ms % SECOND_MS;
    format!("{h:02}:{m:02}:{s:02}{sep}{milli:03}")
}

/// cue 的可见文本：对应 Python `_line_text`（speaker 存在时输出 `[spk0] 文本`）。
fn cue_text(seg: &Segment) -> String {
    let text = py_strip(&fold_newline_runs(&seg.text)).to_string();
    match &seg.speaker {
        Some(spk) if !spk.is_empty() => format!("[{spk}] {text}"),
        _ => text,
    }
}

/// Python `str.strip()`（无参版本）；空白判定统一走 `crate::text::normalize::is_py_space`。
fn py_strip(text: &str) -> &str {
    text.trim_matches(is_py_space)
}

/// 等价于 `re.compile(r"\s*[\r\n]+\s*").sub(" ", text)`：
/// 把「含 \r 或 \n 的极大空白段」折叠为一个空格，其余空白原样保留。
fn fold_newline_runs(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut run: Option<(usize, bool)> = None;
    for (idx, ch) in text.char_indices() {
        if is_py_space(ch) {
            let (_, has_newline) = run.get_or_insert((idx, false));
            *has_newline |= ch == '\r' || ch == '\n';
            continue;
        }
        match run.take() {
            Some((_, true)) => out.push(' '),
            Some((start, false)) => out.push_str(&text[start..idx]),
            None => {}
        }
        out.push(ch);
    }
    match run {
        Some((_, true)) => out.push(' '),
        Some((start, false)) => out.push_str(&text[start..]),
        None => {}
    }
    out
}

/// 按 Python `json.dumps(ensure_ascii=False)` 的规则写入 JSON 字符串体（不含两侧引号）。
fn push_json_string(out: &mut String, text: &str) {
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{09}' => out.push_str("\\t"),
            '\u{0a}' => out.push_str("\\n"),
            '\u{0c}' => out.push_str("\\f"),
            '\u{0d}' => out.push_str("\\r"),
            c if u32::from(c) < 0x20 => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => out.push(c),
        }
    }
}
