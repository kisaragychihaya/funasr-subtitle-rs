//! ffmpeg 解码。对应 funasr-subtitle `backend/app/media/decode.py`。

use std::path::Path;

use crate::error::Result;

/// 解码卡死判定秒数（原项目 FUNASR_SUBTITLE_STALL_TIMEOUT 默认 90）。
pub const DEFAULT_STALL_TIMEOUT_SECS: u64 = 90;

/// 16kHz 单声道 PCM。samples 为 s16 原始采样，与 Python 版送入引擎的 int16 数组一致。
#[derive(Debug, Clone)]
pub struct Audio {
    pub samples: Vec<i16>,
    pub sample_rate: u32,
}

/// 调 ffmpeg 把任意音视频解码为 16kHz 单声道 PCM（流式读入预分配缓冲区）。
/// ffmpeg 不在 PATH 时返回明确报错。
pub fn decode_16k_mono(path: &Path) -> Result<Audio> {
    todo!("T3: 移植 decode.py 的 ffmpeg 调用与读流逻辑")
}
