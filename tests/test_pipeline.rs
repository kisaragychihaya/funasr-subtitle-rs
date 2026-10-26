//! T10 验收测试：端到端编排（VAD → ASR → 标点三模式 → 切句 → SRT）。
//!
//! 分两类：
//!
//! 1) **纯逻辑**（不需要模型，随 `cargo test` 常跑）：标点三分支选择、英文 bare 文本构造、
//!    语言代码合法性回退、`max_chars=0` 不限宽。
//! 2) **端到端**（`#[ignore]`，需要真实模型）：整条管线 `decode_16k_mono` →
//!    [`Engine::transcribe`] → `to_srt`，与参考实现逐字节比对：
//!    * `tests/golden/sensevoice/{fixture}/output.srt` —— punct=auto（golden 直出）；
//!    * `reference/testdata/e2e_punct_{on,off}/{fixture}.srt` —— on/off 两种模式，
//!      由 `reference/testdata/gen_punct_modes.py`（参考引擎直跑、fbank dither 钉 0）生成；
//!    * `tests/golden/paraformer/{fixture}/segments.json` —— 由 golden 分段渲染 SRT 后比对；
//!    * `tests/golden/contextual/sample_zh/segments.json` —— 热词版 + `--hotwords` 同上。
//!
//! 运行（模型根优先取 `FUNASR_SUBTITLE_MODELS_ROOT`，未设则用默认缓存目录，
//! 缺模型时由 `ModelManager` 自动下载）：
//!
//! ```text
//! set FUNASR_SUBTITLE_MODELS_ROOT=C:\path\to\models
//! cargo test --test test_pipeline -- --ignored --nocapture
//! ```

use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use funasr_subtitle_rs::export::to_srt;
use funasr_subtitle_rs::media::decode_16k_mono;
use funasr_subtitle_rs::models::manager::default_models_root;
use funasr_subtitle_rs::models::{
    DEFAULT_MODEL_ID, PARAFORMER_ZH, PARAFORMER_ZH_HOTWORD, SENSEVOICE,
};
use funasr_subtitle_rs::pipeline::{
    Engine, EngineOptions, PunctAction, bare_timed, language_from_code, punct_action,
    punct_mode_from_code, resolve_language, resolve_punct_mode,
};
use funasr_subtitle_rs::text::segment_timed;
use funasr_subtitle_rs::types::{JobOptions, Language, PunctMode, Segment, TimedPiece};

const FIXTURES: [&str; 3] = ["sample_zh", "sample_long", "sample_en_runon"];
/// `dump_golden.py` / `gen_punct_modes.py` 用的热词（contextual golden 同源）。
const HOTWORDS: &str = "机器学习 深度学习";
const MAX_CHARS: usize = 30;
const MAX_CUE_MS: i64 = 7000;

// ---------------------------------------------------------------------------
// 通用小工具
// ---------------------------------------------------------------------------

fn manifest_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn fixture_wav(name: &str) -> PathBuf {
    manifest_path("tests/fixtures").join(format!("{name}.wav"))
}

/// 模型根：优先 `FUNASR_SUBTITLE_MODELS_ROOT`，否则默认缓存目录。
fn models_root() -> PathBuf {
    std::env::var("FUNASR_SUBTITLE_MODELS_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_models_root())
}

fn new_engine(model_id: &str) -> Engine {
    Engine::new(EngineOptions {
        model_id: model_id.to_string(),
        models_root: models_root(),
        num_threads: 4,
    })
    .unwrap_or_else(|e| {
        panic!(
            "初始化引擎 {model_id} 失败（模型根 {}）: {e}",
            models_root().display()
        )
    })
}

fn job(opts: PunctMode) -> JobOptions {
    JobOptions {
        language: Language::Auto,
        max_chars: MAX_CHARS,
        max_cue_ms: MAX_CUE_MS,
        punct: opts,
        hotwords: String::new(),
        diarization: false,
        preset_spk_num: None,
    }
}

/// 解码 + 转写，返回 (SRT 文本, 分段)。
fn transcribe_srt(engine: &mut Engine, wav: &Path, job: &JobOptions) -> (String, Vec<Segment>) {
    let audio = decode_16k_mono(wav).unwrap_or_else(|e| panic!("解码 {} 失败: {e}", wav.display()));
    let segments = engine
        .transcribe(&audio, job, None)
        .unwrap_or_else(|e| panic!("转写 {} 失败: {e}", wav.display()));
    (to_srt(&segments), segments)
}

/// 逐字节比对 SRT；不一致时打印首个差异行与完整实际输出。
fn assert_srt_eq(got: &str, want_bytes: &[u8], label: &str) {
    if got.as_bytes() == want_bytes {
        return;
    }
    let want = String::from_utf8_lossy(want_bytes);
    let mut diffs = String::new();
    for (i, (g, w)) in got.lines().zip(want.lines()).enumerate() {
        if g != w {
            diffs.push_str(&format!("  第 {} 行: got {g:?} vs want {w:?}\n", i + 1));
        }
    }
    panic!(
        "{label}: SRT 不一致（got {} 字节 / {} 行, want {} 字节 / {} 行）\n{diffs}\
         ---- 实际输出 ----\n{got}\n",
        got.len(),
        got.lines().count(),
        want_bytes.len(),
        want.lines().count()
    );
}

/// 读 golden 里的期望文件；缺失时给出复现命令而不是含糊失败。
fn read_expected(path: &Path, how: &str) -> Vec<u8> {
    fs::read(path)
        .unwrap_or_else(|e| panic!("读不到期望文件 {}（{e}）；先运行：{how}", path.display()))
}

// ---------------------------------------------------------------------------
// 1) 纯逻辑：标点策略 / bare 构造 / 语言合法性 / max_chars=0
// ---------------------------------------------------------------------------

#[test]
fn punct_action_covers_the_three_branches() {
    // add_punct && !is_cjk -> 原生拉丁标点优先（与原生有没有句末标点无关）
    assert_eq!(punct_action(true, false, true), Some(PunctAction::Latin));
    assert_eq!(punct_action(true, false, false), Some(PunctAction::Latin));
    // add_punct && is_cjk && 原生无句末标点 -> ct-punc 补齐
    assert_eq!(punct_action(true, true, false), Some(PunctAction::FillCjk));
    // add_punct && is_cjk && 原生已有句末标点 -> 什么都不做
    assert_eq!(punct_action(true, true, true), None);
    // !add_punct -> 删句末终止标点（按停顿切）
    assert_eq!(punct_action(false, true, true), Some(PunctAction::Strip));
    assert_eq!(punct_action(false, false, false), Some(PunctAction::Strip));
}

#[test]
fn bare_timed_strips_punct_strip_chars_only() {
    // _PUNCT_STRIP = 。，！？!?…；;：:、,. —— 只摘这些，其余字符与时间原样保留
    let timed = vec![
        TimedPiece {
            text: "Hi".to_string(),
            time_ms: 100,
        },
        TimedPiece {
            text: ",".to_string(),
            time_ms: 150,
        },
        TimedPiece {
            text: " there".to_string(),
            time_ms: 200,
        },
        TimedPiece {
            text: "。".to_string(),
            time_ms: 250,
        },
        TimedPiece {
            text: "x=y".to_string(),
            time_ms: 300,
        },
    ];
    let bare = bare_timed(&timed);
    let text: String = bare.iter().map(|p| p.text.as_str()).collect();
    assert_eq!(text, "Hi therex=y");
    assert_eq!(bare[0].time_ms, 100);
    assert_eq!(bare[1].time_ms, 200);
    assert_eq!(bare[2].time_ms, 300);
    // 纯标点片被整片丢弃，不会留下空 piece
    assert!(bare.iter().all(|p| !p.text.is_empty()));
}

#[test]
fn language_codes_are_validated_and_fall_back_to_auto() {
    assert_eq!(language_from_code("auto"), Some(Language::Auto));
    assert_eq!(language_from_code("zh"), Some(Language::Zh));
    assert_eq!(language_from_code("yue"), Some(Language::Yue));
    assert_eq!(language_from_code("en"), Some(Language::En));
    assert_eq!(language_from_code("ja"), Some(Language::Ja));
    assert_eq!(language_from_code("ko"), Some(Language::Ko));
    // 非法代码 -> None（CLI 侧提示），管线侧回退 auto
    for bad in ["", "ZH", "cn", "fr", "auto "] {
        assert_eq!(language_from_code(bad), None, "{bad:?} 不该被接受");
        assert_eq!(resolve_language(bad), Language::Auto);
    }
    assert_eq!(resolve_language("ja"), Language::Ja);
}

#[test]
fn punct_mode_codes_are_validated_and_fall_back_to_auto() {
    assert_eq!(punct_mode_from_code("auto"), Some(PunctMode::Auto));
    assert_eq!(punct_mode_from_code("on"), Some(PunctMode::On));
    assert_eq!(punct_mode_from_code("off"), Some(PunctMode::Off));
    assert_eq!(punct_mode_from_code("ON"), None);
    assert_eq!(resolve_punct_mode("yes"), PunctMode::Auto);
    assert_eq!(resolve_punct_mode("off"), PunctMode::Off);
}

#[test]
fn max_chars_zero_means_unlimited() {
    // 无标点的长中文句：max_chars=0 必须只有一条（0 是合法值，不能回退默认 30）
    let text = "这是一个很长很长的句子用来验证零值不限制行宽的行为";
    let timed: Vec<TimedPiece> = text
        .chars()
        .enumerate()
        .map(|(i, c)| TimedPiece {
            text: c.to_string(),
            time_ms: i as i64 * 100,
        })
        .collect();

    let unlimited = segment_timed(&timed, 5000, 0, MAX_CUE_MS);
    assert_eq!(unlimited.len(), 1, "max_chars=0 不应切行: {unlimited:?}");
    assert_eq!(unlimited[0].start_ms, 0);
    assert_eq!(unlimited[0].end_ms, 5000);

    let limited = segment_timed(&timed, 5000, 5, MAX_CUE_MS);
    assert!(limited.len() > 1, "max_chars=5 应切成多行: {limited:?}");
}

// ---------------------------------------------------------------------------
// 2) 端到端：SenseVoice × 3 fixture × punct 三模式
// ---------------------------------------------------------------------------

#[test]
#[ignore = "需要 SenseVoice + VAD + ct-punc 模型；FUNASR_SUBTITLE_MODELS_ROOT 指向模型根"]
fn sensevoice_punct_auto_matches_golden_srt() {
    let mut engine = new_engine(DEFAULT_MODEL_ID);
    let stages = RefCell::new(Vec::<String>::new());
    let on_progress = |stage: &str, pct: f32| {
        assert!((0.0..=100.0).contains(&pct), "percent 越界: {pct}");
        let mut list = stages.borrow_mut();
        if list.last().map(String::as_str) != Some(stage) {
            list.push(stage.to_string());
        }
    };

    for name in FIXTURES {
        let wav = fixture_wav(name);
        let audio =
            decode_16k_mono(&wav).unwrap_or_else(|e| panic!("解码 {} 失败: {e}", wav.display()));
        let segments = engine
            .transcribe(&audio, &job(PunctMode::Auto), Some(&on_progress))
            .unwrap_or_else(|e| panic!("转写 {name} 失败: {e}"));
        let got = to_srt(&segments);
        let want_path = manifest_path(&format!("tests/golden/sensevoice/{name}/output.srt"));
        let want = read_expected(&want_path, "reference/dump_golden.py");
        assert_srt_eq(&got, &want, &format!("sensevoice/{name}/punct=auto"));
        println!("[auto] {name}: {} 条字幕（golden 一致）", segments.len());
    }

    let seen = stages.borrow();
    for stage in ["vad", "asr", "assembling"] {
        assert!(
            seen.iter().any(|s| s == stage),
            "progress 回调没收到 stage {stage}: {seen:?}"
        );
    }
}

#[test]
#[ignore = "需要 SenseVoice + VAD + ct-punc 模型；FUNASR_SUBTITLE_MODELS_ROOT 指向模型根"]
fn sensevoice_punct_on_matches_reference() {
    let mut engine = new_engine(DEFAULT_MODEL_ID);
    for name in FIXTURES {
        let (got, segments) = transcribe_srt(&mut engine, &fixture_wav(name), &job(PunctMode::On));
        let want_path = manifest_path(&format!("reference/testdata/e2e_punct_on/{name}.srt"));
        let want = read_expected(&want_path, "reference/testdata/gen_punct_modes.py");
        assert_srt_eq(&got, &want, &format!("sensevoice/{name}/punct=on"));
        println!("[on] {name}: {} 条字幕（参考实现一致）", segments.len());
    }
}

#[test]
#[ignore = "需要 SenseVoice + VAD + ct-punc 模型；FUNASR_SUBTITLE_MODELS_ROOT 指向模型根"]
fn sensevoice_punct_off_matches_reference() {
    let mut engine = new_engine(DEFAULT_MODEL_ID);
    for name in FIXTURES {
        let (got, segments) = transcribe_srt(&mut engine, &fixture_wav(name), &job(PunctMode::Off));
        let want_path = manifest_path(&format!("reference/testdata/e2e_punct_off/{name}.srt"));
        let want = read_expected(&want_path, "reference/testdata/gen_punct_modes.py");
        assert_srt_eq(&got, &want, &format!("sensevoice/{name}/punct=off"));
        println!("[off] {name}: {} 条字幕（参考实现一致）", segments.len());
    }
}

// ---------------------------------------------------------------------------
// 3) 端到端：Paraformer / Contextual
// ---------------------------------------------------------------------------

/// golden `segments.json` 里的期望分段。
fn golden_segments(rel: &str) -> Vec<Segment> {
    let path = manifest_path(rel);
    let raw =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("读取 {} 失败: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("解析 {} 失败: {e}", path.display()))
}

#[test]
#[ignore = "需要 Paraformer + VAD + ct-punc 模型；FUNASR_SUBTITLE_MODELS_ROOT 指向模型根"]
fn paraformer_matches_golden_segments() {
    let mut engine = new_engine(PARAFORMER_ZH.id);
    for name in ["sample_zh", "sample_long"] {
        let (got, segments) =
            transcribe_srt(&mut engine, &fixture_wav(name), &job(PunctMode::Auto));
        let golden = golden_segments(&format!("tests/golden/paraformer/{name}/segments.json"));
        assert_eq!(
            segments, golden,
            "paraformer/{name}: 分段与 golden segments.json 不一致"
        );
        // 再按「golden 渲染的 SRT」比一次，锁住导出侧
        assert_srt_eq(
            &got,
            to_srt(&golden).as_bytes(),
            &format!("paraformer/{name}/segments.json → SRT"),
        );
        println!(
            "[paraformer] {name}: {} 条字幕（golden 一致）",
            segments.len()
        );
    }
}

#[test]
#[ignore = "需要热词版 Paraformer（900MB）+ VAD + ct-punc 模型"]
fn contextual_hotwords_matches_golden_segments() {
    let mut engine = new_engine(PARAFORMER_ZH_HOTWORD.id);
    let mut options = job(PunctMode::Auto);
    options.hotwords = HOTWORDS.to_string();
    let (got, segments) = transcribe_srt(&mut engine, &fixture_wav("sample_zh"), &options);
    assert!(!segments.is_empty(), "热词版转写不应为空");
    let golden = golden_segments("tests/golden/contextual/sample_zh/segments.json");
    assert_eq!(
        segments, golden,
        "contextual/sample_zh（热词 {HOTWORDS}）: 分段与 golden 不一致"
    );
    assert_srt_eq(
        &got,
        to_srt(&golden).as_bytes(),
        "contextual/sample_zh/segments.json → SRT",
    );
    println!(
        "[contextual] sample_zh（热词 {HOTWORDS}）: {} 条字幕（golden 一致）",
        segments.len()
    );
}

// ---------------------------------------------------------------------------
// 4) RTF（实时率）报告
// ---------------------------------------------------------------------------

#[test]
#[ignore = "需要 SenseVoice + VAD + ct-punc 模型；RTF 建议用 --release 跑"]
fn sensevoice_rtf_report() {
    let mut engine = new_engine(DEFAULT_MODEL_ID);
    // 预热：把模型加载（懒加载）与首次 session 初始化排除在计时之外
    let warm = fixture_wav("sample_zh");
    let warm_audio = decode_16k_mono(&warm).unwrap_or_else(|e| panic!("解码失败: {e}"));
    let slice = funasr_subtitle_rs::media::Audio {
        samples: warm_audio.samples[..warm_audio.samples.len().min(32_000)].to_vec(),
        sample_rate: warm_audio.sample_rate,
    };
    let _ = engine.transcribe(&slice, &job(PunctMode::Auto), None);

    println!("[RTF] 模型 {}（16k 单声道，纯 CPU）", SENSEVOICE.id);
    let mut total_audio = 0.0f64;
    let mut total_cost = 0.0f64;
    for name in FIXTURES {
        let audio =
            decode_16k_mono(&fixture_wav(name)).unwrap_or_else(|e| panic!("解码 {name} 失败: {e}"));
        let audio_secs = audio.samples.len() as f64 / f64::from(audio.sample_rate);
        let started = Instant::now();
        let segments = engine
            .transcribe(&audio, &job(PunctMode::Auto), None)
            .unwrap_or_else(|e| panic!("转写 {name} 失败: {e}"));
        let cost = started.elapsed().as_secs_f64();
        total_audio += audio_secs;
        total_cost += cost;
        println!(
            "  {name:<16} 音频 {:>6.2}s  耗时 {:>6.2}s  RTF {:.3}（{} 条字幕）",
            audio_secs,
            cost,
            cost / audio_secs,
            segments.len()
        );
    }
    println!(
        "  合计              音频 {total_audio:>6.2}s  耗时 {total_cost:>6.2}s  RTF {:.3}",
        total_cost / total_audio
    );
}

// ---------------------------------------------------------------------------
// 5) CLI 冒烟
// ---------------------------------------------------------------------------

#[test]
#[ignore = "需要 SenseVoice + VAD + ct-punc 模型；走真实 CLI 子进程"]
fn cli_stdout_srt_smoke() {
    let output = Command::new(env!("CARGO_BIN_EXE_funasr-subtitle-rs"))
        .arg(fixture_wav("sample_zh"))
        .arg("-o")
        .arg("-")
        .arg("--format")
        .arg("srt")
        .arg("--models-dir")
        .arg(models_root())
        .output()
        .expect("启动 CLI 失败");
    assert!(
        output.status.success(),
        "CLI 退出码非零: {:?}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout 不是合法 UTF-8");
    assert!(!stdout.trim().is_empty(), "CLI 没有输出字幕");
    assert!(stdout.starts_with("1\n"), "SRT 不以序号 1 开头:\n{stdout}");
    assert!(stdout.contains(" --> "), "缺少 SRT 时间码行:\n{stdout}");
    assert!(stdout.ends_with('\n'), "SRT 应以换行结尾:\n{stdout:?}");
    assert!(!stdout.contains('\u{feff}'), "SRT 不应带 BOM");
    println!("[cli] sample_zh -o - --format srt:\n{stdout}");
}
