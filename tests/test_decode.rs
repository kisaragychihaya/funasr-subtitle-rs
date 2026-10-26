//! T3 验收：ffmpeg 解码路径（真实文件 / 不存在的文件 / 非音视频文件）。

use std::path::{Path, PathBuf};

use funasr_subtitle_rs::media::decode_16k_mono;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn le_u16(bytes: &[u8], at: usize) -> Option<u16> {
    let raw: [u8; 2] = bytes.get(at..at + 2)?.try_into().ok()?;
    Some(u16::from_le_bytes(raw))
}

fn le_u32(bytes: &[u8], at: usize) -> Option<u32> {
    let raw: [u8; 4] = bytes.get(at..at + 4)?.try_into().ok()?;
    Some(u32::from_le_bytes(raw))
}

/// 从 wav 头推源文件时长（秒）：data 字节数 / (采样率 * 声道 * 位深 / 8)。
/// 测试用真实 fixture 自适应，不写死时长。
fn wav_duration_secs(bytes: &[u8]) -> Option<f64> {
    if bytes.get(0..4)? != b"RIFF" || bytes.get(8..12)? != b"WAVE" {
        return None;
    }
    let mut pos = 12usize;
    let mut byte_rate: Option<f64> = None;
    let mut data_len: Option<u64> = None;
    while pos + 8 <= bytes.len() {
        let id = bytes.get(pos..pos + 4)?;
        let size = u32::from_le_bytes(bytes.get(pos + 4..pos + 8)?.try_into().ok()?) as usize;
        let body = pos + 8;
        match id {
            b"fmt " => {
                let channels = f64::from(le_u16(bytes, body + 2)?);
                let sample_rate = f64::from(le_u32(bytes, body + 4)?);
                let bits = f64::from(le_u16(bytes, body + 14)?);
                if channels <= 0.0 || sample_rate <= 0.0 || bits <= 0.0 {
                    return None;
                }
                byte_rate = Some(sample_rate * channels * bits / 8.0);
            }
            b"data" => {
                let available = (bytes.len() - body) as u64;
                data_len = Some((size as u64).min(available));
            }
            _ => {}
        }
        pos = body + size + (size & 1); // chunk 按偶数字节对齐
    }
    let rate = byte_rate?;
    let len = data_len?;
    if rate <= 0.0 {
        return None;
    }
    Some(len as f64 / rate)
}

#[test]
fn decodes_zh_wav_to_16k_mono() {
    let path = fixture("sample_zh.wav");
    let audio = match decode_16k_mono(&path) {
        Ok(audio) => audio,
        Err(err) => panic!("解码 {} 失败：{err}", path.display()),
    };

    assert_eq!(audio.sample_rate, 16000, "采样率必须是 16kHz");
    assert!(!audio.samples.is_empty(), "解码结果不能是空音频");
    assert!(
        audio.samples.iter().any(|&s| s != 0),
        "解码结果全是静音，说明没读到真实 PCM"
    );

    let raw = match std::fs::read(&path) {
        Ok(raw) => raw,
        Err(err) => panic!("读取 {} 失败：{err}", path.display()),
    };
    let secs = match wav_duration_secs(&raw) {
        Some(secs) => secs,
        None => panic!("无法从 wav 头解析时长：{}", path.display()),
    };
    let expected = secs * 16000.0;
    let actual = audio.samples.len() as f64;
    let deviation = (actual - expected).abs() / expected;
    assert!(
        deviation <= 0.05,
        "采样点数 {actual:.0} 与 wav 头推出的 {expected:.0}（{secs:.3}s @16k）偏差 {:.2}%，超过 5%",
        deviation * 100.0
    );
}

#[test]
fn missing_path_returns_err() {
    let dir = match tempfile::tempdir() {
        Ok(dir) => dir,
        Err(err) => panic!("创建临时目录失败：{err}"),
    };
    let path = dir.path().join("no_such_file.wav");

    match decode_16k_mono(&path) {
        Ok(audio) => panic!("不存在的路径却解码成功（{} 个采样）", audio.samples.len()),
        Err(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("ffmpeg") || msg.contains("No such file") || msg.contains("文件"),
                "错误信息应说明原因（ffmpeg stderr 或文件不存在）：{msg}"
            );
        }
    }
}

#[test]
fn non_media_file_returns_err() {
    let dir = match tempfile::tempdir() {
        Ok(dir) => dir,
        Err(err) => panic!("创建临时目录失败：{err}"),
    };
    let path = dir.path().join("garbage.bin");
    if let Err(err) = std::fs::write(&path, b"this is definitely not audio or video") {
        panic!("写测试文件失败：{err}");
    }

    match decode_16k_mono(&path) {
        Ok(audio) => panic!("非音视频文件却解码成功（{} 个采样）", audio.samples.len()),
        Err(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("ffmpeg"),
                "错误信息应带上 ffmpeg 的失败原因：{msg}"
            );
        }
    }
}
