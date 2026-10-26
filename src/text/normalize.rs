//! 文本规范化辅助。移植来源：funasr-subtitle `backend/app/engine/funasr_engine.py`
//! 中的标点/语言判断辅助函数（_TERMINAL_CHARS/_PUNCT_STRIP/_CJK_RE/_FW2HW/_SPECIAL_TOKEN_RE）。

use crate::types::TimedPiece;

/// 是否含中日韩表意字符（`_CJK_RE.search`）。
pub fn has_cjk(text: &str) -> bool {
    todo!("T1: 移植 funasr_engine.py _CJK_RE")
}

/// 是否含句末终止标点（`_has_terminal`）。
pub fn has_terminal(text: &str) -> bool {
    todo!("T1: 移植 funasr_engine.py _has_terminal")
}

/// 纯拉丁文本的全角标点转半角并规整空格（`_normalize_latin_punct`）。
pub fn normalize_latin_punct(text: &str) -> String {
    todo!("T1: 移植 funasr_engine.py _normalize_latin_punct")
}

/// 剥离 SenseVoice 特殊标记 `<|...|>`（`_SPECIAL_TOKEN_RE`）。
pub fn strip_special_tokens(text: &str) -> String {
    todo!("T1: 移植 funasr_engine.py _SPECIAL_TOKEN_RE")
}

/// 移植 `_strip_terminal_timed`：工作在带时间的逐 token 上，删句末终止标点（保留逗号/小数/缩写内点）。
pub fn strip_terminal_timed(timed: &[TimedPiece]) -> Vec<TimedPiece> {
    todo!("T1: 移植 funasr_engine.py _strip_terminal_timed")
}
