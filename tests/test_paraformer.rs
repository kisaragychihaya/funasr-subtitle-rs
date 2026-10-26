//! T11 验收测试：Paraformer-zh 与 ContextualParaformer（热词版）。
//!
//! golden 全部来自 `reference/dump_golden.py` 的 `stage_paraformer` / `stage_contextual` /
//! `stage_hotword_case`（参考实现直跑）：
//! * `tests/golden/paraformer/{fixture}/` 与 `tests/golden/contextual/sample_zh/`：
//!   `preds.json`（逐 VAD 段的模型输出文本 + `sample_range`）、`text.txt`（整段转写，
//!   逐段一行）、`segments.json`（标点/切句后的最终字幕，属 T9/T12 范围，这里不比对）；
//! * `tests/golden/contextual/hotword_case.json`：`proc_hotword("机器学习 深度学习")` 的
//!   输出（`hotword_token_ids` N×10 + `hotword_lengths`）与带/不带热词的识别结果；
//! * `tests/golden/input/{fixture}/input.s16bin`：送进引擎的 16k 单声道 s16le PCM。
//!
//! 覆盖：
//! 1) 纯逻辑：`tokens.txt`/`tokens.json` 词表解析、`sentence_postprocess` 的边界用例
//!    （中文/英文/混合/@/缩写/*标点 token*）、`decode_frames`（argmax/blank/截断）、
//!    `config.yaml` 解析——期望值全部由参考实现逐例跑出；
//! 2) 热词分词对拍（纯离线，只用本地 `tokens.json`，不做推理）：`proc_hotword` 的
//!    ids/lengths 与 `hotword_case.json` 全等；
//! 3) `#[ignore]` 端到端：`Paraformer::new`/`ContextualParaformer::new` +
//!    `transcribe_chunk` 跑 golden 的每个 VAD 段，`AsrChunkOutput::Plain` 的文本与
//!    `preds.json` 逐段全等、按行拼回后与 `text.txt` **byte-equal**；contextual 再验证
//!    hotword_case 音频带/不带热词都能跑通。
//!
//! 2)、3) 需要模型目录里的 `tokens.json`；模型没缓存时 3) 会走 `ModelManager` 下载
//! （约 240MB / 900MB）。手动运行：
//! `cargo test --test test_paraformer -- --ignored`（`FUNASR_SUBTITLE_MODELS_ROOT` 可指向
//! 已有的模型缓存根目录）。

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use funasr_subtitle_rs::asr::contextual::{ContextualParaformer, proc_hotword};
use funasr_subtitle_rs::asr::paraformer::{
    TokenList, decode_frames, load_model_config, parse_model_config, sentence_postprocess,
};
use funasr_subtitle_rs::asr::{AsrChunkOutput, AsrEngine};
use funasr_subtitle_rs::frontend::paraformer_frontend;
use funasr_subtitle_rs::models::manager::{ModelManager, default_models_root, has_onnx, model_dir};
use funasr_subtitle_rs::models::{ModelSpec, PARAFORMER_ZH, PARAFORMER_ZH_HOTWORD};
use funasr_subtitle_rs::types::JobOptions;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const SR: u32 = 16_000;
/// `dump_golden.py` 的 HOTWORDS（contextual 的 golden 与 hotword_case 都用它）。
const HOTWORDS: &str = "机器学习 深度学习";
/// 两个中文模型的 config.yaml 里 `model_conf.predictor_bias` 都是 1。
const PREDICTOR_BIAS: usize = 1;

fn golden(rel: impl AsRef<Path>) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join(rel)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

fn read_s16(path: &Path) -> std::io::Result<Vec<i16>> {
    let bytes = fs::read(path)?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_le_bytes(*c))
        .collect())
}

// ---------------------------------------------------------------------------
// 纯逻辑：词表
// ---------------------------------------------------------------------------

#[test]
fn tokens_txt_is_one_token_per_line() -> TestResult {
    let dir = tempfile::tempdir()?;
    fs::write(dir.path().join("tokens.txt"), "今\n天\n，\n<unk>\n")?;
    let list = TokenList::load(dir.path())?;

    assert_eq!(list.len(), 4, "四行 = 四个 token");
    let got: Vec<&str> = list.tokens().iter().map(String::as_str).collect();
    assert_eq!(got, ["今", "天", "，", "<unk>"]);
    assert_eq!(list.id_of("天"), Some(1));
    assert_eq!(list.unk_id(), 3, "unk 取最后一项");
    assert_eq!(
        list.ids2tokens(&[0, 3])?,
        vec!["今".to_string(), "<unk>".to_string()]
    );
    assert!(list.ids2tokens(&[4]).is_err(), "越界 id 应报错");
    assert!(list.ids2tokens(&[-1]).is_err(), "负 id 应报错");
    assert_eq!(list.id_of("不在词表"), None);
    Ok(())
}

#[test]
fn tokens_json_wins_over_txt_and_missing_is_error() -> TestResult {
    let dir = tempfile::tempdir()?;
    fs::write(dir.path().join("tokens.json"), r#"["a","b","<unk>"]"#)?;
    fs::write(dir.path().join("tokens.txt"), "x\ny\n<unk>\n")?;
    let list = TokenList::load(dir.path())?;
    assert_eq!(list.len(), 3, "两者同时存在时以 tokens.json 为准");
    assert_eq!(list.id_of("a"), Some(0));
    assert_eq!(list.id_of("x"), None);

    let empty = tempfile::tempdir()?;
    assert!(TokenList::load(empty.path()).is_err(), "没有词表应报错");
    Ok(())
}

#[test]
fn token_list_keeps_last_duplicate_and_rejects_empty() -> TestResult {
    let list = TokenList::new(vec![
        "a".to_string(),
        "b".to_string(),
        "a".to_string(),
        "<unk>".to_string(),
    ])?;
    assert_eq!(
        list.id_of("a"),
        Some(2),
        "dict 语义：重复 token 取最后一个下标"
    );
    assert_eq!(list.unk_id(), 3);
    assert!(TokenList::new(Vec::new()).is_err(), "空词表应报错");
    Ok(())
}

// ---------------------------------------------------------------------------
// 纯逻辑：sentence_postprocess
// ---------------------------------------------------------------------------

fn sp(tokens: &[&str]) -> Result<String, Box<dyn std::error::Error>> {
    let owned: Vec<String> = tokens.iter().map(|t| (*t).to_string()).collect();
    Ok(sentence_postprocess(&owned)?)
}

#[test]
fn sentence_postprocess_matches_reference_cases() -> TestResult {
    // 期望值由参考实现 postprocess_utils.sentence_postprocess 逐例跑出
    let cases: &[(&[&str], &str)] = &[
        // 全中文：词内空格去掉后直接拼
        (
            &["今", "天", "天", "气", "非", "常", "好"],
            "今天天气非常好",
        ),
        (&["今 天", "天  气"], "今天天气"),
        (&["我 ", "们"], "我们"),
        (&["<s>", "今", "天", "</s>", "<unk>"], "今天"),
        // 数字在 isChinese 里也算「中文」，与汉字一起直接拼
        (&["今", "1", "2"], "今12"),
        (&["1", "2", "3"], "123"),
        (&["我", "们", "10", "个", "人"], "我们10个人"),
        (&["今", "天", "12", "3"], "今天123"),
        // 全字母：@@ 续接 + 词间补空格（最终 strip 掉）
        (&["and@@", "the@@", "test"], "andthetest"),
        (&["and@@"], ""),
        (&["a", "b"], "AB"),
        (&["a", "b", "c", "d", "e"], "ABCDE"),
        // 混合：中文吞掉前一个英文空格
        (&["我", "们", "and@@", "test"], "我们andtest"),
        (&["我", "and@@", "test", "好"], "我andtest好"),
        (&["test@@", "test", "我"], "testtest我"),
        (&["今", "天", "A@@", "B"], "今天AB"),
        (&["今", "a", "b"], "今AB"),
        (&["今", "天", "a"], "今天a"),
        (&[], ""),
    ];
    for &(tokens, want) in cases {
        assert_eq!(sp(tokens)?, want, "tokens = {tokens:?}");
    }
    Ok(())
}

#[test]
fn sentence_postprocess_rejects_punct_and_blank_tokens() -> TestResult {
    // 中文模型不产标点（builtin_punc=false），参考实现对标点/纯空白 token 抛 ValueError
    for tokens in [
        &["今", "天", "，", "好"][..],
        &["今", "天", "。"][..],
        &["a", " ", "b"][..],
        &[" ", "a"][..],
    ] {
        assert!(
            sp(tokens).is_err(),
            "tokens = {tokens:?} 应报 invalid character"
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 纯逻辑：decode_frames（decode_one 的 argmax/过滤/截断）
// ---------------------------------------------------------------------------

#[test]
fn decode_frames_argmax_filter_and_truncation() -> TestResult {
    // 3 帧 × 3 词表：帧 0 → 1，帧 1 → 0（blank，丢弃），帧 2 → 2（eos，丢弃）
    let blank_and_eos = [0.1f32, 0.9, 0.2, 0.9, 0.1, 0.1, 0.1, 0.1, 0.5];
    assert_eq!(decode_frames(&blank_and_eos, 3, 3, 0)?, vec![1]);

    // np.argmax 平局取第一个
    assert_eq!(decode_frames(&[0.0f32, 1.0, 1.0], 3, 3, 0)?, vec![1]);

    // 两帧都判成 id 3（词表 4 维）
    let two_frames = [0.0f32, 0.1, 0.2, 9.0, 0.0, 0.1, 0.2, 9.0];
    assert_eq!(decode_frames(&two_frames, 4, 2, 0)?, vec![3, 3]);
    // token[: valid_token_num - pred_bias]
    assert_eq!(decode_frames(&two_frames, 4, 1, 1)?, Vec::<i64>::new());
    // 负数按 Python 负索引语义：token[:0-1] = token[:-1] → 去掉最后一个
    assert_eq!(decode_frames(&two_frames, 4, 0, 1)?, vec![3]);
    // 超过实际 token 数时只取到末尾
    assert_eq!(decode_frames(&two_frames, 4, 5, 0)?, vec![3, 3]);
    assert!(
        decode_frames(&two_frames, 0, 2, 0).is_err(),
        "词表为 0 应报错"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 纯逻辑：config.yaml
// ---------------------------------------------------------------------------

#[test]
fn parse_model_config_reads_frontend_and_predictor_bias() -> TestResult {
    // 与真实 config.yaml 同构的片段（含列表项与注释不该被误读）
    let text = "\
model: Paraformer
model_conf:
    ctc_weight: 0.0
    predictor_bias: 1
frontend: WavFrontend
frontend_conf:
    fs: 16000
    window: hamming
    n_mels: 80
    frame_length: 25
    frame_shift: 10
    lfr_m: 7
    lfr_n: 6
specaug_conf:
    apply_time_warp: false
    freq_mask_width_range:
    - 0
    - 30
# 末尾注释
";
    let cfg = parse_model_config(text)?;
    let preset = paraformer_frontend();
    assert_eq!(cfg.predictor_bias, PREDICTOR_BIAS);
    assert_eq!(cfg.language, None, "两个中文模型都没有 lang 键");
    assert_eq!(cfg.frontend.n_mels, preset.n_mels);
    assert_eq!(cfg.frontend.frame_length_ms, preset.frame_length_ms);
    assert_eq!(cfg.frontend.frame_shift_ms, preset.frame_shift_ms);
    assert_eq!((cfg.frontend.lfr_m, cfg.frontend.lfr_n), (7, 6));
    assert_eq!(cfg.frontend.dither, 0.0, "golden 把 dither 钉成 0");

    // 缺 predictor_bias → 0（Python 的默认分支）；缺整个 frontend_conf → 预设值
    let no_bias = parse_model_config("model_conf:\n    sampling_ratio: 0.75\n")?;
    assert_eq!(no_bias.predictor_bias, 0);
    assert_eq!(no_bias.frontend.n_mels, preset.n_mels);

    // 与前端能力冲突的配置显式报错，而不是静默跑歪
    assert!(parse_model_config("frontend_conf:\n    fs: 8000\n").is_err());
    assert!(parse_model_config("frontend_conf:\n    window: hann\n").is_err());
    assert!(parse_model_config("model_conf:\n    predictor_bias: 一\n").is_err());
    Ok(())
}

// ---------------------------------------------------------------------------
// 热词分词对拍（纯离线：只要本地 tokens.json，不加载 ONNX）
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct HotwordRun {
    hotwords: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct HotwordCase {
    with_hotwords: HotwordRun,
    without_hotwords: HotwordRun,
    hotword_token_ids: Vec<Vec<i32>>,
    hotword_lengths: Vec<i32>,
}

#[test]
fn proc_hotword_matches_golden_token_ids() -> TestResult {
    let case: HotwordCase = read_json(&golden("contextual/hotword_case.json"))?;
    assert_eq!(case.with_hotwords.hotwords, HOTWORDS);

    let Some(dir) = cached_tokens_dir(&PARAFORMER_ZH_HOTWORD) else {
        println!(
            "跳过：{} 未缓存（tokens.json 缺失），无法离线对拍 proc_hotword",
            PARAFORMER_ZH_HOTWORD.id
        );
        return Ok(());
    };
    let tokens = TokenList::load(&dir)?;
    let batch = proc_hotword(&tokens, &case.with_hotwords.hotwords)?;
    assert_eq!(
        batch.ids, case.hotword_token_ids,
        "hotword_token_ids 应与 golden 全等"
    );
    assert_eq!(
        batch.lengths, case.hotword_lengths,
        "hotword_lengths 应与 golden 全等"
    );

    // 空热词走的是 `"".split(" ")` → [""]，长度 -1，再补 `<s>` 行（golden 的
    // without_hotwords 就是这条路径）
    let empty = proc_hotword(&tokens, &case.without_hotwords.hotwords)?;
    assert_eq!(empty.lengths, vec![-1, 0]);
    assert_eq!(empty.ids.len(), 2);
    let sos_row: Vec<i32> = empty.ids[1].clone();
    assert_eq!(&sos_row[..2], &[1, 0], "第二行是 <s> 后补 0");
    assert!(sos_row[2..].iter().all(|&v| v == 0));

    // 超过 pad_list 的 max_len=10 时参考实现会因 numpy 广播失败而抛错
    let too_long = "一二三四五六七八九十一";
    assert!(proc_hotword(&tokens, too_long).is_err());
    Ok(())
}

// ---------------------------------------------------------------------------
// 真实模型（#[ignore]）
// ---------------------------------------------------------------------------

/// golden 的 `preds.json`：逐 VAD 段的输出文本 + 该段在 input.s16bin 里的采样区间。
#[derive(Debug, Deserialize)]
struct PredSpan {
    sample_range: [usize; 2],
    text: String,
}

#[derive(Debug, Deserialize)]
struct HotwordAudio {
    sentence: String,
}

fn models_root() -> PathBuf {
    std::env::var_os("FUNASR_SUBTITLE_MODELS_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(default_models_root)
}

/// 已缓存的模型目录（只看词表是否就位：热词分词对拍不需要 ONNX 权重）。
fn cached_tokens_dir(spec: &ModelSpec) -> Option<PathBuf> {
    let dir = model_dir(&models_root(), spec);
    dir.join("tokens.json").is_file().then_some(dir)
}

/// 定位模型目录：优先用缓存，缺失时用 `ModelManager::ensure_model` 下载到同一个模型根
/// （`FUNASR_SUBTITLE_MODELS_ROOT` 或默认缓存），第二次运行即可复用。
fn ensure_spec_dir(
    spec: &ModelSpec,
    required: &[&str],
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let root = models_root();
    let cached = model_dir(&root, spec);
    if has_onnx(&cached) && required.iter().all(|name| cached.join(name).is_file()) {
        return Ok(cached);
    }
    Ok(ModelManager::new(root).ensure_model(spec)?)
}

/// 整段文本：`AsrChunkOutput::Plain`（Paraformer/Contextual 的唯一输出形态）。
fn plain_text(
    engine: &mut dyn AsrEngine,
    chunk: &[i16],
    job: &JobOptions,
) -> Result<String, Box<dyn std::error::Error>> {
    let AsrChunkOutput::Plain(text) = engine.transcribe_chunk(chunk, SR, job)? else {
        panic!("Paraformer/Contextual 应返回 AsrChunkOutput::Plain");
    };
    Ok(text)
}

/// 逐 VAD 段对拍：`stage_dir` 是 golden 里该引擎的输出目录（含 `preds.json`），
/// `input_fixture` 是对应的 `tests/golden/input/<name>/input.s16bin`。返回 (逐段文本, 失败说明)。
fn compare_spans(
    engine: &mut dyn AsrEngine,
    stage_dir: &str,
    input_fixture: &str,
    job: &JobOptions,
) -> Result<(Vec<String>, Vec<String>), Box<dyn std::error::Error>> {
    let wave = read_s16(&golden(format!("input/{input_fixture}/input.s16bin")))?;
    let spans: Vec<PredSpan> = read_json(&golden(format!("{stage_dir}/preds.json")))?;
    assert!(!spans.is_empty(), "{stage_dir} 的 preds.json 为空");

    let mut texts = Vec::with_capacity(spans.len());
    let mut failures = Vec::new();
    for (idx, span) in spans.iter().enumerate() {
        let (start, end) = (span.sample_range[0], span.sample_range[1]);
        assert!(
            start < end && end <= wave.len(),
            "{stage_dir} span #{idx} 的 sample_range {start}..{end} 越界（共 {} 采样）",
            wave.len()
        );
        let got = plain_text(engine, &wave[start..end], job)?;
        let equal = got.as_bytes() == span.text.as_bytes();
        println!(
            "{stage_dir} span #{idx} [{start}..{end}]: {}\n  want = {:?}\n  got  = {:?}",
            if equal { "byte-equal" } else { "MISMATCH" },
            span.text,
            got
        );
        if !equal {
            failures.push(format!(
                "{stage_dir} span #{idx}: want {:?} got {:?}",
                span.text, got
            ));
        }
        texts.push(got);
    }
    Ok((texts, failures))
}

fn assert_text_file_byte_equal(
    fixture_dir: &str,
    texts: &[String],
    failures: &mut Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let want = fs::read_to_string(golden(format!("{fixture_dir}/text.txt")))?;
    let got = texts.join("\n") + "\n";
    let equal = got.as_bytes() == want.as_bytes();
    println!(
        "{fixture_dir}/text.txt 整段拼接: {}（{} 字节 / {} 字节）",
        if equal { "byte-equal" } else { "MISMATCH" },
        got.len(),
        want.len()
    );
    if !equal {
        failures.push(format!(
            "{fixture_dir}/text.txt: want {:?} got {:?}",
            want, got
        ));
    }
    Ok(())
}

#[test]
#[ignore = "需要 Paraformer-zh 模型（约 240MB）；手动运行：cargo test --test test_paraformer -- --ignored"]
fn paraformer_text_matches_golden() -> TestResult {
    let dir = ensure_spec_dir(
        &PARAFORMER_ZH,
        &["model_quant.onnx", "tokens.json", "am.mvn", "config.yaml"],
    )?;
    println!("Paraformer 模型目录: {}", dir.display());

    // config.yaml 的实测值与移植里用的常量对得上（前端参数由 T6 验收）
    let cfg = load_model_config(&dir.join("config.yaml"))?;
    let preset = paraformer_frontend();
    assert_eq!(cfg.predictor_bias, PREDICTOR_BIAS);
    assert_eq!(cfg.language, None);
    assert_eq!(cfg.frontend.n_mels, preset.n_mels);
    assert_eq!(cfg.frontend.frame_length_ms, preset.frame_length_ms);
    assert_eq!(cfg.frontend.frame_shift_ms, preset.frame_shift_ms);
    assert_eq!((cfg.frontend.lfr_m, cfg.frontend.lfr_n), (7, 6));

    let mut engine = funasr_subtitle_rs::asr::Paraformer::new(&dir, 4)?;
    let job = JobOptions::default();
    let mut failures = Vec::new();
    for fixture in ["sample_zh", "sample_long"] {
        let stage = format!("paraformer/{fixture}");
        let (texts, mut bad) = compare_spans(&mut engine, &stage, fixture, &job)?;
        assert_text_file_byte_equal(&stage, &texts, &mut bad)?;
        failures.append(&mut bad);
    }
    if !failures.is_empty() {
        return Err(format!(
            "Paraformer 有 {} 处与 golden 不一致:\n{}",
            failures.len(),
            failures.join("\n")
        )
        .into());
    }
    println!("Paraformer：2 个 fixture 的整段文本全部 byte-equal");
    Ok(())
}

#[test]
#[ignore = "需要 Paraformer-zh 热词版模型（约 900MB，含 model_eb）；手动运行：cargo test --test test_paraformer -- --ignored"]
fn contextual_text_matches_golden_and_hotword_case() -> TestResult {
    let dir = ensure_spec_dir(
        &PARAFORMER_ZH_HOTWORD,
        &[
            "model_quant.onnx",
            "model_eb_quant.onnx",
            "tokens.json",
            "am.mvn",
            "config.yaml",
        ],
    )?;
    println!("Contextual 模型目录: {}", dir.display());

    let mut engine = ContextualParaformer::new(&dir, 4)?;
    let case: HotwordCase = read_json(&golden("contextual/hotword_case.json"))?;

    // 1) 实例上的 proc_hotword 与 golden 全等（热词分词不依赖推理）
    let batch = engine.proc_hotword(&case.with_hotwords.hotwords)?;
    assert_eq!(batch.ids, case.hotword_token_ids);
    assert_eq!(batch.lengths, case.hotword_lengths);

    // 2) sample_zh 逐段对拍（golden 的 contextual 阶段就是用 HOTWORDS 跑的）
    let job = JobOptions {
        hotwords: case.with_hotwords.hotwords.clone(),
        ..JobOptions::default()
    };
    let mut failures = Vec::new();
    let (texts, mut bad) = compare_spans(&mut engine, "contextual/sample_zh", "sample_zh", &job)?;
    assert_text_file_byte_equal("contextual/sample_zh", &texts, &mut bad)?;
    failures.append(&mut bad);

    // 3) hotword_case：带热词/不带热词都能跑通，且文本一致（golden: outputs_identical）
    let wave = read_s16(&golden("contextual/hotword_case.s16bin"))?;
    let audio: HotwordAudio = read_json(&golden("contextual/hotword_case_audio.json"))?;
    let with_hot = plain_text(&mut engine, &wave, &job)?;
    let cold_job = JobOptions {
        hotwords: case.without_hotwords.hotwords.clone(),
        ..job.clone()
    };
    let without_hot = plain_text(&mut engine, &wave, &cold_job)?;
    println!("hotword_case 带热词:   {with_hot:?}");
    println!("hotword_case 不带热词: {without_hot:?}");
    println!(
        "golden 带热词（标点后）: {:?} / 原句: {:?}",
        case.with_hotwords.text, audio.sentence
    );
    if with_hot.as_bytes() != audio.sentence.as_bytes() {
        failures.push(format!(
            "hotword_case: want {:?} got {:?}",
            audio.sentence, with_hot
        ));
    }
    if with_hot != without_hot {
        failures.push(format!(
            "hotword_case 带/不带热词结果不一致: {with_hot:?} vs {without_hot:?}"
        ));
    }
    if !failures.is_empty() {
        return Err(format!(
            "Contextual 有 {} 处与 golden 不一致:\n{}",
            failures.len(),
            failures.join("\n")
        )
        .into());
    }
    println!("Contextual：sample_zh 整段文本 byte-equal，hotword_case 带/不带热词跑通且一致");
    Ok(())
}
