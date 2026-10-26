//! ffmpeg 解码。对应 funasr-subtitle `backend/app/media/decode.py`。
//!
//! 流程与 Python 版一致：ffmpeg 把原始 PCM 写到 stdout，主线程边收边攒；
//! stderr 全程抽干（`-progress pipe:2` 的进度行也走 stderr，不读会把 ffmpeg 堵死），
//! 非零退出时把末尾若干行拼进错误信息；看门狗按「连续多久没收到任何字节」判卡死。
//!
//! 与 Python 版唯一的有意差异：管道采样格式用 `s16le`（Python 用 `f32le`，送引擎前再
//! 转 int16），因为本模块的 [`Audio`] 冻结为 `Vec<i16>`，直接取 s16 可省一次全量转换。

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read};
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::path::Path;
use std::process::{Child, ChildStderr, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::error::{Error, Result};

/// 解码卡死判定秒数（原项目 FUNASR_SUBTITLE_STALL_TIMEOUT 默认 90）。
pub const DEFAULT_STALL_TIMEOUT_SECS: u64 = 90;

/// 目标采样率：16kHz（Python `TARGET_SR`）。
const TARGET_SR: u32 = 16_000;
/// 每次从 ffmpeg stdout 取的数据块（Python `_READ_CHUNK`：1MB）。
const READ_CHUNK: usize = 1 << 20;
/// 出错时最多保留的 stderr 行数（Python `_ERR_CAP`）。
const STDERR_TAIL_LINES: usize = 50;
/// 拼进错误信息的 stderr 字符上限（Python `msg[:400]`）。
const STDERR_TAIL_CHARS: usize = 400;
/// ffmpeg 可执行文件名，跟 Python 版一样从 PATH 找。
const FFMPEG_BIN: &str = "ffmpeg";
/// Windows 下隐藏子进程控制台窗口（Python `_NO_WINDOW` = CREATE_NO_WINDOW）。
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// `-progress pipe:2` 写出的进度行前缀；这些行不算错误内容（Python 同款过滤表）。
const PROGRESS_PREFIXES: [&str; 10] = [
    "out_time",
    "total_size",
    "bitrate",
    "speed",
    "frame=",
    "fps=",
    "stream_",
    "progress=",
    "dup_frames",
    "drop_frames",
];

/// 16kHz 单声道 PCM。samples 为 s16 原始采样，与 Python 版送入引擎的 int16 数组一致。
#[derive(Debug, Clone)]
pub struct Audio {
    pub samples: Vec<i16>,
    pub sample_rate: u32,
}

/// 读线程送给主线程的一块 PCM（已按小端 s16 组装好）。
enum Chunk {
    Data(Vec<i16>),
    Eof,
    Failed(String),
}

/// 调 ffmpeg 把任意音视频解码为 16kHz 单声道 PCM（流式读入）。
/// ffmpeg 不在 PATH 时返回明确报错。
pub fn decode_16k_mono(path: &Path) -> Result<Audio> {
    let (mut proc, rx) = Proc::spawn(path)?;

    let stall = Duration::from_secs(DEFAULT_STALL_TIMEOUT_SECS);
    let mut samples: Vec<i16> = Vec::new();
    let mut stalled = false;
    let mut read_failure: Option<String> = None;

    // 每收到一块数据就重新计时：窗口内一个字节都没到才算卡死。
    loop {
        match rx.recv_timeout(stall) {
            Ok(Chunk::Data(chunk)) => samples.extend_from_slice(&chunk),
            Ok(Chunk::Eof) => break,
            Ok(Chunk::Failed(err)) => {
                read_failure = Some(err);
                break;
            }
            Err(RecvTimeoutError::Timeout) => {
                stalled = true;
                break;
            }
            // 读线程没发 Eof 就退出（理论上不会）——按 EOF 处理，交给退出码判定。
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    if let Some(err) = read_failure {
        proc.terminate();
        let tail = proc.take_stderr_tail();
        let mut msg = format!("读取 ffmpeg 输出失败：{err}");
        if !tail.is_empty() {
            msg.push_str("；ffmpeg: ");
            msg.push_str(&tail);
        }
        return Err(Error::Decode(msg));
    }
    if stalled {
        proc.terminate();
        return Err(Error::Decode(
            "解码卡住已超时中断：文件可能损坏或编码异常".to_string(),
        ));
    }

    let status = proc.wait()?;
    let tail = proc.take_stderr_tail();
    let code = status.code().unwrap_or(-1);
    if code != 0 {
        let mut msg = format!("ffmpeg 解码失败（退出码 {code}）");
        if !tail.is_empty() {
            msg.push('：');
            msg.push_str(&tail);
        }
        return Err(Error::Decode(msg));
    }
    if samples.is_empty() {
        return Err(Error::Decode(
            "解码出来的音频是空的（可能没有有效音轨）".to_string(),
        ));
    }
    Ok(Audio {
        samples,
        sample_rate: TARGET_SR,
    })
}

/// 子进程 + 两个抽水线程的统一回收点：任何路径（正常/超时/读错误/提前 `?` 返回）
/// 都会在 `Drop` 里 kill + wait，不留孤儿 ffmpeg 或僵尸进程。
struct Proc {
    child: Option<Child>,
    reader: Option<JoinHandle<()>>,
    stderr: Option<JoinHandle<Vec<String>>>,
}

impl Proc {
    /// 起 ffmpeg 并挂上 stderr/stdout 两个抽水线程；返回 (进程句柄, 数据通道)。
    fn spawn(path: &Path) -> Result<(Self, Receiver<Chunk>)> {
        let mut cmd = Command::new(FFMPEG_BIN);
        cmd.arg("-nostdin")
            .arg("-hide_banner")
            .arg("-loglevel")
            .arg("error")
            .arg("-i")
            .arg(path)
            .arg("-vn") // 丢弃视频
            .arg("-ac")
            .arg("1") // 单声道
            .arg("-ar")
            .arg(TARGET_SR.to_string()) // 16k
            .arg("-f")
            .arg("s16le") // 原始 int16 little-endian
            .arg("-progress")
            .arg("pipe:2") // 进度写 stderr
            .arg("pipe:1") // PCM 写 stdout
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        cmd.creation_flags(CREATE_NO_WINDOW);

        let mut child = cmd.spawn().map_err(|err| {
            if err.kind() == std::io::ErrorKind::NotFound {
                Error::Decode("找不到 ffmpeg，请安装并把 ffmpeg 加入 PATH".to_string())
            } else {
                Error::Decode(format!("启动 ffmpeg 失败：{err}"))
            }
        })?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::Decode("无法接管 ffmpeg stdout".to_string()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| Error::Decode("无法接管 ffmpeg stderr".to_string()))?;

        let (tx, rx) = channel();
        let reader = thread::spawn(move || pump_stdout(stdout, &tx));
        let stderr_thread = thread::spawn(move || pump_stderr(stderr));

        Ok((
            Self {
                child: Some(child),
                reader: Some(reader),
                stderr: Some(stderr_thread),
            },
            rx,
        ))
    }

    /// 正常路径：等进程自己结束并回收（stdout 已 EOF，不会挂住）。
    fn wait(&mut self) -> Result<ExitStatus> {
        let Some(mut child) = self.child.take() else {
            return Err(Error::Decode("ffmpeg 进程已被回收".to_string()));
        };
        match child.wait() {
            Ok(status) => {
                self.join_reader();
                Ok(status)
            }
            Err(err) => {
                // 收回句柄，交给 Drop 的 kill + wait 兜底，避免读线程永远等不到 EOF。
                self.child = Some(child);
                Err(Error::Io(err))
            }
        }
    }

    /// 异常路径：杀掉再 wait（Windows 上 kill 也要 wait 才能收干净），并等抽水线程退出。
    fn terminate(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.join_reader();
    }

    fn join_reader(&mut self) {
        if let Some(handle) = self.reader.take() {
            let _ = handle.join();
        }
    }

    /// 收下 stderr 线程的产物，拼成一行、截到 400 字符（窗口关掉后线程必然已退出）。
    fn take_stderr_tail(&mut self) -> String {
        let lines = match self.stderr.take() {
            Some(handle) => handle.join().unwrap_or_default(),
            None => Vec::new(),
        };
        let joined = lines.join(" ");
        let mut tail = String::new();
        for ch in joined.chars().take(STDERR_TAIL_CHARS) {
            tail.push(ch);
        }
        tail
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        if self.child.is_some() {
            self.terminate();
        }
        self.join_reader();
        if let Some(handle) = self.stderr.take() {
            let _ = handle.join();
        }
    }
}

/// 分块读 ffmpeg stdout，按小端 s16 组装后经通道送出；EOF / 出错各发一条终止消息。
fn pump_stdout(mut stdout: ChildStdout, tx: &Sender<Chunk>) {
    let mut buf = vec![0u8; READ_CHUNK];
    let mut carry: Option<u8> = None;
    loop {
        match stdout.read(&mut buf) {
            Ok(0) => {
                let _ = tx.send(Chunk::Eof);
                return;
            }
            Ok(n) => {
                let mut chunk = Vec::with_capacity(n / 2 + 1);
                append_le_i16(&mut chunk, &mut carry, &buf[..n]);
                if tx.send(Chunk::Data(chunk)).is_err() {
                    return; // 主线程已经走了，无需继续读
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(err) => {
                let _ = tx.send(Chunk::Failed(err.to_string()));
                return;
            }
        }
    }
}

/// 把字节流按小端 16 位切成 i16；块边界上的半个样本用 `carry` 带到下一块。
/// 末尾残留的孤字节直接丢弃（Python `to_audio` 同样截到样本整数倍）。
fn append_le_i16(dst: &mut Vec<i16>, carry: &mut Option<u8>, data: &[u8]) {
    let mut rest = data;
    if let Some(low) = carry.take() {
        match rest.split_first() {
            Some((&high, tail)) => {
                dst.push(i16::from_le_bytes([low, high]));
                rest = tail;
            }
            None => {
                *carry = Some(low);
                return;
            }
        }
    }
    let (pairs, remainder) = rest.as_chunks::<2>();
    for pair in pairs {
        dst.push(i16::from_le_bytes(*pair));
    }
    if let Some(&byte) = remainder.first() {
        *carry = Some(byte);
    }
}

/// 读完 stderr，只留下最后 [`STDERR_TAIL_LINES`] 行真正的内容行。
fn pump_stderr(stderr: ChildStderr) -> Vec<String> {
    let mut kept: VecDeque<String> = VecDeque::new();
    let mut reader = BufReader::new(stderr);
    let mut raw: Vec<u8> = Vec::new();
    loop {
        raw.clear();
        match reader.read_until(b'\n', &mut raw) {
            Ok(0) => break,
            Ok(_) => {}
            Err(err) => {
                if err.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                break;
            }
        }
        let text = String::from_utf8_lossy(&raw);
        let line = text.trim();
        if line.is_empty() || is_progress_line(line) {
            continue;
        }
        kept.push_back(line.to_string());
        if kept.len() > STDERR_TAIL_LINES {
            kept.pop_front();
        }
    }
    kept.into_iter().collect()
}

fn is_progress_line(line: &str) -> bool {
    PROGRESS_PREFIXES.iter().any(|p| line.starts_with(p))
}
