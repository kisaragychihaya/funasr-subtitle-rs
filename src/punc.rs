//! T9：CT-Transformer 标点。移植来源：funasr-subtitle `engine/funasr_engine.py` 的
//! _punctuate/_punc_once/_apply_punct_to_timed 与 vendored `punc_bin.py`。

use std::path::Path;

use crate::error::Result;
use crate::types::TimedPiece;

/// ct-punc 标点器（量化 ONNX）。punctuate 内含 110 字分块逻辑（PUNC_CHUNK）。
pub struct Punctuator {
    // T9：ort session、token 映射、分块参数
}

impl Punctuator {
    pub fn new(model_dir: &Path, threads: usize) -> Result<Self> {
        let _ = (model_dir, threads);
        todo!("T9")
    }

    /// 给文本补标点；失败时应返回原文（标点不致命），由调用方决定是否容错。
    pub fn punctuate(&mut self, text: &str) -> Result<String> {
        let _ = text;
        todo!("T9")
    }
}

/// 移植 `_apply_punct_to_timed`：把补标点后的文本对齐回带时间的 token，
/// 保留真实 CTC 时间；失配时有界向前重同步（窗口 8），新插入标点用上一字符时间。
pub fn align_punct_to_timed(timed: &[TimedPiece], punct_text: &str) -> Vec<TimedPiece> {
    let _ = (timed, punct_text);
    todo!("T9")
}
