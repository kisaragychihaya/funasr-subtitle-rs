//! 文本规范化辅助。移植来源：funasr-subtitle `backend/app/engine/funasr_engine.py`
//! 中的标点/语言判断辅助函数（_TERMINAL_CHARS/_PUNCT_STRIP/_CJK_RE/_FW2HW/_SPECIAL_TOKEN_RE）。

use crate::types::TimedPiece;

// ---------------------------------------------------------------------------
// Python 语义助手：`str.isspace()` / `str.strip()` / `[一-鿿぀-ヿ가-힯]`
// ---------------------------------------------------------------------------

/// Python `str.isspace()` 的判定集合：Unicode White_Space 属性 ∪ U+001C..U+001F。
///
/// 注意 CPython 的 `str.isspace()` 对 U+001C–U+001F（bidi 控制符）也返回 True，
/// 必须包含；export 模块共用本实现，勿各自复制。
/// 显式列出码点而不复用 `char::is_whitespace`，以固定语义（Python 侧所有 `isspace`/
/// `strip`/`\s+` 替换点都以此为准）。
pub fn is_py_space(c: char) -> bool {
    matches!(
        c,
        '\t'..='\r'
            | ' '
            | '\u{1c}'..='\u{1f}'
            | '\u{85}'
            | '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
    )
}

/// Python `str.strip()`（按 `is_py_space` 判定），返回原文切片。
pub(crate) fn py_trim(s: &str) -> &str {
    py_trim_end(py_trim_start(s))
}

/// Python `str.lstrip()`。
pub(crate) fn py_trim_start(s: &str) -> &str {
    match s.find(|c: char| !is_py_space(c)) {
        Some(i) => &s[i..],
        None => "",
    }
}

/// Python `str.rstrip()`。
pub(crate) fn py_trim_end(s: &str) -> &str {
    let mut end = 0;
    for (i, c) in s.char_indices() {
        if !is_py_space(c) {
            end = i + c.len_utf8();
        }
    }
    &s[..end]
}

/// 中日韩表意字符：Python `[一-鿿぀-ヿ가-힯]`
/// = U+4E00..U+9FFF / U+3040..U+30FF / U+AC00..U+D7AF。
pub(crate) fn is_cjk_char(c: char) -> bool {
    matches!(
        c,
        '\u{4e00}'..='\u{9fff}' | '\u{3040}'..='\u{30ff}' | '\u{ac00}'..='\u{d7af}'
    )
}

/// 全角标点。Python `_FW2HW = {"，": ", ", "。": ". ", "！": "! ", "？": "? ",
/// "；": "; ", "：": ": ", "、": ", "}`。
fn fw2hw(c: char) -> Option<&'static str> {
    match c {
        '，' => Some(", "),
        '。' => Some(". "),
        '！' => Some("! "),
        '？' => Some("? "),
        '；' => Some("; "),
        '：' => Some(": "),
        '、' => Some(", "),
        _ => None,
    }
}

/// Python `_TERMINAL_CHARS = set("。！？!?….;")`（含半角 `.`/`;`：normalize 后英文句末用半角）。
fn is_terminal_char(c: char) -> bool {
    matches!(c, '。' | '！' | '？' | '!' | '?' | '…' | '.' | ';')
}

/// Python `_STRIP_TERMINALS = frozenset("。！？!?…；;")`（不含 `.`，句点另行按位置判断）。
fn is_strip_terminal_char(c: char) -> bool {
    matches!(c, '。' | '！' | '？' | '!' | '?' | '…' | '；' | ';')
}

/// 是否含中日韩表意字符（`_CJK_RE.search`）。
pub fn has_cjk(text: &str) -> bool {
    text.chars().any(is_cjk_char)
}

/// 是否含句末终止标点（`_has_terminal`）。
pub fn has_terminal(text: &str) -> bool {
    text.chars().any(is_terminal_char)
}

/// 纯拉丁文本的全角标点转半角并规整空格（`_normalize_latin_punct`）。
pub fn normalize_latin_punct(text: &str) -> String {
    if has_cjk(text) {
        return text.to_string();
    }
    // Python 侧是按顺序对每个全角键做 str.replace：替换串里不含任何全角键，
    // 因此等价于逐字符映射。
    let mut mapped = String::with_capacity(text.len() + 8);
    for c in text.chars() {
        match fw2hw(c) {
            Some(hw) => mapped.push_str(hw),
            None => mapped.push(c),
        }
    }
    let mut out = String::with_capacity(mapped.len());
    let mut in_space = false;
    for c in mapped.chars() {
        if is_py_space(c) {
            if !in_space {
                out.push(' ');
                in_space = true;
            }
        } else {
            out.push(c);
            in_space = false;
        }
    }
    py_trim(&out).to_string()
}

/// 剥离 SenseVoice 特殊标记 `<|...|>`（`_SPECIAL_TOKEN_RE` = `<\|[^|>]*\|>`）。
pub fn strip_special_tokens(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < n {
        if chars[i] == '<' && i + 1 < n && chars[i + 1] == '|' {
            let mut j = i + 2;
            while j < n && chars[j] != '|' && chars[j] != '>' {
                j += 1;
            }
            if j + 1 < n && chars[j] == '|' && chars[j + 1] == '>' {
                i = j + 2;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// 移植 `_strip_terminal_timed`：工作在带时间的逐 token 上，删句末终止标点（保留逗号/小数/缩写内点）。
pub fn strip_terminal_timed(timed: &[TimedPiece]) -> Vec<TimedPiece> {
    let chars: Vec<(char, i64)> = timed
        .iter()
        .flat_map(|p| p.text.chars().map(move |c| (c, p.time_ms)))
        .collect();
    let n = chars.len();
    let mut out: Vec<TimedPiece> = Vec::with_capacity(n);
    for (i, (ch, t)) in chars.iter().enumerate() {
        if is_strip_terminal_char(*ch) {
            continue;
        }
        if *ch == '.' {
            let after = if i + 1 < n {
                Some(chars[i + 1].0)
            } else {
                None
            };
            if after.is_none_or(is_py_space) {
                continue;
            }
        }
        out.push(TimedPiece {
            text: ch.to_string(),
            time_ms: *t,
        });
    }
    out
}
