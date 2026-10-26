//! T3：媒体解码。调外部 ffmpeg 子进程输出 16kHz 单声道 s16le PCM。

mod decode;

pub use decode::{decode_16k_mono, Audio, DEFAULT_STALL_TIMEOUT_SECS};
