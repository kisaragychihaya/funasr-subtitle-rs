//! CLI 入口：`funasr-subtitle-rs <INPUT> [-o OUTPUT] ...`。
//!
//! 流程与参考实现的任务管线一致：ffmpeg 解码 → [`Engine::transcribe`] → 按格式导出 →
//! 写文件（UTF-8，不加 BOM）。日志/进度走 stderr，`-o -` 时字幕正文走 stdout，
//! 因此 `funasr-subtitle-rs in.mp4 -o - > out.srt` 可以直接重定向。

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use funasr_subtitle_rs::export::{to_json, to_srt, to_txt, to_vtt};
use funasr_subtitle_rs::media::decode_16k_mono;
use funasr_subtitle_rs::models::DEFAULT_MODEL_ID;
use funasr_subtitle_rs::models::manager::default_models_root;
use funasr_subtitle_rs::pipeline::{
    Engine, EngineOptions, language_from_code, punct_mode_from_code, resolve_language,
    resolve_punct_mode,
};
use funasr_subtitle_rs::types::{JobOptions, Segment};

/// 命令行参数（默认值与原项目 config.py 一致）。
#[derive(Debug, Parser)]
#[command(
    name = "funasr-subtitle-rs",
    version,
    about = "本地离线音视频转字幕（FunASR ONNX，纯 CPU）"
)]
struct Cli {
    /// 输入音频/视频文件
    input: PathBuf,

    /// 输出文件；缺省为同名改后缀，`-` 表示输出到 stdout
    #[arg(short = 'o', long = "output")]
    output: Option<String>,

    /// ASR 模型 id（见 `models::ASR_MODELS`）
    #[arg(long = "model", default_value = DEFAULT_MODEL_ID)]
    model: String,

    /// 语言：auto / zh / yue / en / ja / ko
    #[arg(long = "language", default_value = "auto")]
    language: String,

    /// 标点模式：auto（中文加、英文按停顿）/ on（都加）/ off（都不加）
    #[arg(long = "punct", default_value = "auto")]
    punct: String,

    /// 每行最大显示宽度，0 = 不限制
    #[arg(long = "max-chars", default_value_t = 30)]
    max_chars: usize,

    /// 单条字幕最长时长（毫秒）
    #[arg(long = "max-cue-ms", default_value_t = 7000)]
    max_cue_ms: i64,

    /// 热词，空格分隔（仅热词版模型生效）
    #[arg(long = "hotwords", default_value = "")]
    hotwords: String,

    /// onnxruntime 线程数
    #[arg(long = "threads", default_value_t = 4)]
    threads: usize,

    /// 输出格式：srt / vtt / txt / json
    #[arg(long = "format", default_value = "srt")]
    format: String,

    /// 模型缓存目录（缺省读环境变量 FUNASR_SUBTITLE_MODELS_ROOT，再退回系统缓存目录）
    #[arg(long = "models-dir")]
    models_dir: Option<PathBuf>,
}

/// 导出格式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Srt,
    Vtt,
    Txt,
    Json,
}

impl Format {
    fn from_code(code: &str) -> Option<Self> {
        match code {
            "srt" => Some(Self::Srt),
            "vtt" => Some(Self::Vtt),
            "txt" => Some(Self::Txt),
            "json" => Some(Self::Json),
            _ => None,
        }
    }

    /// 默认输出文件的后缀。
    fn ext(self) -> &'static str {
        match self {
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::Txt => "txt",
            Self::Json => "json",
        }
    }

    fn render(self, segments: &[Segment]) -> String {
        match self {
            Self::Srt => to_srt(segments),
            Self::Vtt => to_vtt(segments),
            Self::Txt => to_txt(segments),
            Self::Json => to_json(segments),
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("错误：{err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    let format = Format::from_code(&cli.format).ok_or_else(|| {
        anyhow!(
            "未知输出格式 `{}`（可选 srt / vtt / txt / json）",
            cli.format
        )
    })?;
    let language = resolve_language(&cli.language);
    if language_from_code(&cli.language).is_none() {
        eprintln!("提示：未知语言 `{}`，按 auto 处理", cli.language);
    }
    let punct = resolve_punct_mode(&cli.punct);
    if punct_mode_from_code(&cli.punct).is_none() {
        eprintln!("提示：未知标点模式 `{}`，按 auto 处理", cli.punct);
    }

    if !cli.input.is_file() {
        bail!("输入文件不存在：{}", cli.input.display());
    }

    // 1) 解码：ffmpeg -> 16kHz 单声道 s16
    let bar = progress_bar()?;
    bar.set_message("解码");
    let audio = decode_16k_mono(&cli.input)
        .with_context(|| format!("解码 {} 失败", cli.input.display()))?;
    if audio.samples.is_empty() {
        bail!(
            "{} 里没有可用的音频采样（文件为空或格式不受支持）",
            cli.input.display()
        );
    }

    // 2) 转写
    let models_root = cli.models_dir.clone().unwrap_or_else(|| {
        std::env::var("FUNASR_SUBTITLE_MODELS_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| default_models_root())
    });
    let mut engine = Engine::new(EngineOptions {
        model_id: cli.model.clone(),
        models_root,
        num_threads: cli.threads,
    })
    .with_context(|| format!("初始化引擎失败（模型 {}）", cli.model))?;

    let job = JobOptions {
        language,
        max_chars: cli.max_chars,
        max_cue_ms: cli.max_cue_ms,
        punct,
        hotwords: cli.hotwords.clone(),
    };

    let on_progress = |stage: &str, pct: f32| {
        bar.set_message(stage_label(stage));
        bar.set_position(pct.clamp(0.0, 100.0) as u64);
    };
    let segments = engine
        .transcribe(&audio, &job, Some(&on_progress))
        .context("转写失败")?;
    bar.finish_and_clear();

    // 3) 导出 + 落盘
    let text = format.render(&segments);
    match output_target(&cli, format) {
        None => {
            let mut stdout = io::stdout().lock();
            stdout
                .write_all(text.as_bytes())
                .context("写入 stdout 失败")?;
            stdout.flush().context("刷新 stdout 失败")?;
        }
        Some(path) => {
            fs::write(&path, text.as_bytes())
                .with_context(|| format!("写入 {} 失败", path.display()))?;
            eprintln!("已写入 {}（{} 条字幕）", path.display(), segments.len());
        }
    }
    Ok(())
}

/// 进度条（非终端时 indicatif 自动不绘制，不会污染重定向的输出）。
fn progress_bar() -> Result<ProgressBar> {
    let bar = ProgressBar::new(100);
    let style = ProgressStyle::with_template("{msg:>8} {bar:28} {percent:>3}%")
        .map_err(|e| anyhow!("初始化进度条失败：{e}"))?;
    bar.set_style(style);
    Ok(bar)
}

/// stage 名 -> 进度条上显示的中文说明。
fn stage_label(stage: &str) -> &'static str {
    match stage {
        "vad" => "切段",
        "asr" => "识别",
        "punc" => "标点",
        "assembling" => "整理",
        _ => "处理",
    }
}

/// `-o -` 或未指定输出时返回目标路径；`None` 表示写 stdout。
fn output_target(cli: &Cli, format: Format) -> Option<PathBuf> {
    match cli.output.as_deref() {
        Some("-") => None,
        Some(path) => Some(PathBuf::from(path)),
        None => Some(default_output_path(&cli.input, format.ext())),
    }
}

/// 缺省输出路径：输入文件同名改后缀（`talk.mp4` -> `talk.srt`）。
fn default_output_path(input: &Path, ext: &str) -> PathBuf {
    input.with_extension(ext)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_output_path_replaces_extension() {
        assert_eq!(
            default_output_path(Path::new("a/b/talk.mp4"), "srt"),
            PathBuf::from("a/b/talk.srt")
        );
        assert_eq!(
            default_output_path(Path::new("talk"), "json"),
            PathBuf::from("talk.json")
        );
    }

    #[test]
    fn format_codes_and_extensions() {
        assert_eq!(Format::from_code("vtt"), Some(Format::Vtt));
        assert_eq!(Format::from_code("SRT"), None);
        assert_eq!(Format::Srt.ext(), "srt");
        assert_eq!(Format::Json.ext(), "json");
    }
}
