//! T8 验收测试：SenseVoiceSmall（`src/asr/sensevoice.rs`）。
//!
//! golden 全部来自 `reference/dump_golden.py` 的 `stage_sensevoice`（参考实现直跑）：
//! `tests/golden/sensevoice/{fixture}/spans/span_NNN/` 下有
//! * `feats.f32bin` + `feats.json` —— 推理输入特征（前端 T6 已单独验收）；
//! * `span.json` —— VAD 段 [start_ms,end_ms] 与 `sample_range`（从 input.s16bin 切片）；
//! * `logits.f32bin` + `logits.json` + `enc_len.json` —— 模型输出（[enc_len, vocab]）；
//! * `tokens.json` —— CTC 贪心结果（token_id/frame/piece/text/special/ms）。
//!
//! 覆盖：
//! 1) tokenizer：`SpModel`（protobuf 最小解析）按 golden 里出现的 token_id 校验
//!    `id_to_piece` 与 golden `piece` 逐条一致；
//! 2) CTC 纯逻辑：golden logits 喂 `ctc_greedy_with_frames`，输出 (token_id, frame)
//!    与 tokens.json 逐元素全等（含 frame）；
//! 3) `#[ignore]` 端到端：`SenseVoice::new` + `transcribe_chunk` 跑全部 fixture/span，
//!    TimedPiece 的 text 拼接与逐 token 绝对 ms 与 tokens.json 全等。
//!
//! 1)、2) 不需要模型，直接读 golden；1) 里的 `id_to_piece` 校验需要模型目录里的
//! `.bpe.model`（和模型一起下载），模型没缓存时打印说明后跳过（不静默判过：
//! 跳过时会明确打印「跳过」）。3) 需要 235MB 模型，手动运行：
//! `cargo test --test test_sensevoice -- --ignored`。

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use funasr_subtitle_rs::asr::sensevoice::{SenseVoice, SpModel, ctc_greedy_with_frames};
use funasr_subtitle_rs::asr::{AsrChunkOutput, AsrEngine};
use funasr_subtitle_rs::models::SENSEVOICE;
use funasr_subtitle_rs::models::manager::{ModelManager, default_models_root, model_dir};
use funasr_subtitle_rs::types::{JobOptions, TimedPiece};

const FIXTURES: [&str; 3] = ["sample_zh", "sample_long", "sample_en_runon"];
/// 与 `src/asr/sensevoice.rs` 的 `BPE_FILE` 一致（该常量不对外导出）。
const BPE_FILE: &str = "chn_jpn_yue_eng_ko_spectok.bpe.model";
const SR: u32 = 16_000;
/// CTC blank id（`SenseVoiceSmall.blank_id`）。
const BLANK_ID: usize = 0;

// ---------------------------------------------------------------------------
// golden 数据结构
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct GoldenToken {
    token_id: usize,
    frame: usize,
    piece: String,
    text: String,
    special: bool,
    ms: i64,
}

#[derive(Debug, Deserialize)]
struct ShapeFile {
    shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct SpanMeta {
    start_ms: i64,
    end_ms: i64,
    sample_range: [usize; 2],
    feats_len: usize,
    enc_len: usize,
}

// ---------------------------------------------------------------------------
// 通用小工具
// ---------------------------------------------------------------------------

fn manifest_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn golden_dir(fixture: &str) -> PathBuf {
    manifest_path("tests/golden/sensevoice").join(fixture)
}

fn span_dir(fixture: &str, span: usize) -> PathBuf {
    golden_dir(fixture)
        .join("spans")
        .join(format!("span_{span:03}"))
}

fn input_path(fixture: &str) -> PathBuf {
    manifest_path("tests/golden/input")
        .join(fixture)
        .join("input.s16bin")
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> T {
    let raw =
        fs::read_to_string(path).unwrap_or_else(|e| panic!("读取 {} 失败: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("解析 {} 失败: {e}", path.display()))
}

fn read_s16(path: &Path) -> Vec<i16> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("读取 {} 失败: {e}", path.display()));
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_le_bytes(*c))
        .collect()
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("读取 {} 失败: {e}", path.display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// 一个 fixture 下 golden 记录的全部 span 下标（按目录探测，缺 span.json 即停）。
fn span_indices(fixture: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for idx in 0.. {
        if !span_dir(fixture, idx).join("span.json").is_file() {
            break;
        }
        out.push(idx);
    }
    out
}

/// 模型根目录：默认 `default_models_root()`；`FUNASR_SUBTITLE_MODELS_ROOT` 可覆盖。
fn models_root() -> PathBuf {
    std::env::var_os("FUNASR_SUBTITLE_MODELS_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(default_models_root)
}

/// 已缓存且带 `.bpe.model` 的 SenseVoice 模型目录；没有就是 None（不触发下载）。
fn cached_model_dir() -> Option<PathBuf> {
    let dir = model_dir(&models_root(), &SENSEVOICE);
    if dir.join(BPE_FILE).is_file() {
        Some(dir)
    } else {
        None
    }
}

/// 端到端用：优先用缓存，缺失时下载到临时目录（约 235MB，需要联网）。
fn sensevoice_model_dir(tmp: &mut Option<tempfile::TempDir>) -> PathBuf {
    if let Some(dir) = cached_model_dir() {
        return dir;
    }
    let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("创建临时模型目录失败: {e}"));
    let path = ModelManager::new(dir.path().to_path_buf())
        .ensure_model(&SENSEVOICE)
        .unwrap_or_else(|e| panic!("下载 SenseVoice 模型失败（需要联网，约 235MB）: {e}"));
    *tmp = Some(dir);
    path
}

// ---------------------------------------------------------------------------
// 1) tokenizer：id → piece 与 golden 逐条一致
// ---------------------------------------------------------------------------

#[test]
fn tokenizer_id_to_piece_matches_golden_pieces() {
    let Some(dir) = cached_model_dir() else {
        println!(
            "跳过 id_to_piece 对拍：{} 下没有缓存 SenseVoice 模型（先跑 --ignored 端到端测试下载）",
            model_dir(&models_root(), &SENSEVOICE).display()
        );
        return;
    };
    let bpe_path = dir.join(BPE_FILE);
    let sp = SpModel::load(&bpe_path)
        .unwrap_or_else(|e| panic!("加载 {} 失败: {e}", bpe_path.display()));

    let mut checked = 0usize;
    let mut mismatches = Vec::new();
    for fixture in FIXTURES {
        for span in span_indices(fixture) {
            let tokens: Vec<GoldenToken> = read_json(&span_dir(fixture, span).join("tokens.json"));
            for tok in &tokens {
                let got = sp.id_to_piece(tok.token_id);
                match got {
                    Ok(piece) if piece == tok.piece => checked += 1,
                    Ok(piece) => mismatches.push(format!(
                        "{fixture}/span_{span:03}: id {} → {piece:?}，golden {:?}",
                        tok.token_id, tok.piece
                    )),
                    Err(e) => mismatches.push(format!(
                        "{fixture}/span_{span:03}: id {} 取 piece 失败: {e}",
                        tok.token_id
                    )),
                }
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "id_to_piece 与 golden piece 不一致（{} 条）：\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
    assert!(checked > 0, "没有可对拍的 token");
    println!(
        "id_to_piece 对拍通过：{checked} 个 token，词表 {} 个 piece",
        sp.len()
    );
    // 词表必须覆盖 golden 里出现过的全部 id
    let max_id = sp.len() - 1;
    assert!(
        max_id >= 25_000,
        "词表只有 {} 个 piece，无法覆盖 SenseVoice 的 id",
        sp.len()
    );
}

// ---------------------------------------------------------------------------
// 2) CTC 贪心：golden logits → (token_id, frame) 与 tokens.json 全等
// ---------------------------------------------------------------------------

#[test]
fn ctc_greedy_with_frames_matches_golden_tokens() {
    let mut total = 0usize;
    for fixture in FIXTURES {
        for span in span_indices(fixture) {
            let dir = span_dir(fixture, span);
            let shape: ShapeFile = read_json(&dir.join("logits.json"));
            assert_eq!(
                shape.shape.len(),
                2,
                "{fixture}/span_{span:03}: logits 维度异常"
            );
            let (enc_rows, vocab) = (shape.shape[0], shape.shape[1]);
            let enc_len: Vec<usize> = read_json(&dir.join("enc_len.json"));
            let valid_len = *enc_len
                .first()
                .unwrap_or_else(|| panic!("{fixture}/span_{span:03}: enc_len.json 为空"));
            let logits = read_f32(&dir.join("logits.f32bin"));
            assert_eq!(
                logits.len(),
                enc_rows * vocab,
                "{fixture}/span_{span:03}: logits.f32bin 大小与 logits.json 不符"
            );
            assert_eq!(
                valid_len, enc_rows,
                "{fixture}/span_{span:03}: enc_len 与 logits 行数不一致（golden 写了 [0:enc_len]）"
            );
            assert_eq!(
                vocab, 25_055,
                "{fixture}/span_{span:03}: 词表大小与 SenseVoice 不符"
            );

            let got = ctc_greedy_with_frames(&logits, vocab, valid_len, BLANK_ID)
                .unwrap_or_else(|e| panic!("{fixture}/span_{span:03}: CTC 解码失败: {e}"));
            let tokens: Vec<GoldenToken> = read_json(&dir.join("tokens.json"));
            let want: Vec<(usize, usize)> = tokens.iter().map(|t| (t.token_id, t.frame)).collect();

            if got != want {
                let mut lines = Vec::new();
                for i in 0..got.len().max(want.len()) {
                    let g = got.get(i);
                    let w = want.get(i);
                    if g != w {
                        lines.push(format!("  #{i}: got {g:?} vs golden {w:?}"));
                        if lines.len() >= 20 {
                            break;
                        }
                    }
                }
                panic!(
                    "{fixture}/span_{span:03}: CTC 序列不一致（got {} 个，golden {} 个）\n{}",
                    got.len(),
                    want.len(),
                    lines.join("\n")
                );
            }
            total += want.len();
            println!(
                "{fixture}/span_{span:03}: {} tokens 全等（enc_len={valid_len}, vocab={vocab}）",
                want.len()
            );
        }
    }
    assert!(total > 0, "没有可对拍的 span");
    println!("CTC 对拍通过：{total} 个 token 的 (id, frame) 全等");
}

// ---------------------------------------------------------------------------
// 3) 端到端：SenseVoice::new + transcribe_chunk（#[ignore]，需 235MB 模型）
// ---------------------------------------------------------------------------

#[test]
#[ignore = "需要 SenseVoiceSmall 模型（约 235MB）；手动运行：cargo test --test test_sensevoice -- --ignored"]
fn test_sensevoice_chunk_matches_golden_tokens() {
    let mut tmp = None;
    let dir = sensevoice_model_dir(&mut tmp);
    println!("SenseVoice 模型目录: {}", dir.display());
    let mut engine = SenseVoice::new(&dir, 4).unwrap_or_else(|e| panic!("加载模型失败: {e}"));
    assert_eq!(
        engine.engine_kind(),
        funasr_subtitle_rs::types::EngineKind::SenseVoice
    );

    // golden 全部来自 language=auto + textnorm=withitn
    let job = JobOptions::default();
    let mut spans_run = 0usize;
    let mut tokens_checked = 0usize;
    // 第一个 span 的 (fixture, span, chunk, pieces)，用于同进程第二个实例的复跑比对
    let mut first: Option<(&'static str, usize, Vec<i16>, Vec<TimedPiece>)> = None;
    for fixture in FIXTURES {
        let wave = read_s16(&input_path(fixture));
        for span in span_indices(fixture) {
            let sdir = span_dir(fixture, span);
            let meta: SpanMeta = read_json(&sdir.join("span.json"));
            let [a, b] = meta.sample_range;
            assert!(
                b <= wave.len(),
                "{fixture}/span_{span:03}: sample_range {:?} 超出音频长度 {}",
                meta.sample_range,
                wave.len()
            );
            let chunk = &wave[a..b];
            let out = engine
                .transcribe_chunk(chunk, SR, &job)
                .unwrap_or_else(|e| panic!("{fixture}/span_{span:03}: 推理失败: {e}"));
            let pieces = match out {
                AsrChunkOutput::Timed(p) => p,
                AsrChunkOutput::Plain(_) => {
                    panic!("{fixture}/span_{span:03}: SenseVoice 应返回 Timed")
                }
            };

            let tokens: Vec<GoldenToken> = read_json(&sdir.join("tokens.json"));
            // golden：TimedPiece 序列 = 非 special 且 text 非空的 token
            let want: Vec<TimedPiece> = tokens
                .iter()
                .filter(|t| !t.special && !t.text.is_empty())
                .map(|t| TimedPiece {
                    text: t.text.clone(),
                    time_ms: t.ms,
                })
                .collect();

            let got_text: String = pieces.iter().map(|p| p.text.as_str()).collect();
            let want_text: String = want.iter().map(|p| p.text.as_str()).collect();
            assert_eq!(
                got_text,
                want_text,
                "{fixture}/span_{span:03}: 文本拼接不一致（got {} 个 piece，golden {} 个）",
                pieces.len(),
                want.len()
            );
            assert_eq!(
                pieces.len(),
                want.len(),
                "{fixture}/span_{span:03}: TimedPiece 个数不一致"
            );
            // 引擎产出段内相对毫秒，加上 span 起点才是 golden 的绝对时间
            let mut mismatches = Vec::new();
            for (i, (got, want)) in pieces.iter().zip(want.iter()).enumerate() {
                let abs = got.time_ms + meta.start_ms;
                if got.text != want.text || abs != want.time_ms {
                    mismatches.push(format!(
                        "  #{i}: got ({:?}, {}ms→abs {}ms) vs golden ({:?}, {}ms)",
                        got.text, got.time_ms, abs, want.text, want.time_ms
                    ));
                }
            }
            assert!(
                mismatches.is_empty(),
                "{fixture}/span_{span:03}: TimedPiece 不一致（{} 条）\n{}",
                mismatches.len(),
                mismatches.join("\n")
            );

            // 顺带核对 golden 记录的分段元信息与推理输入规模
            println!(
                "{fixture}/span_{span:03}: [{}..{}]ms, {} 采样, feats_len={}, enc_len={}, {} pieces",
                meta.start_ms,
                meta.end_ms,
                b - a,
                meta.feats_len,
                meta.enc_len,
                pieces.len()
            );
            spans_run += 1;
            tokens_checked += want.len();
            if first.as_ref().is_none() {
                first = Some((fixture, span, chunk.to_vec(), pieces.clone()));
            }
        }
    }
    assert!(spans_run > 0, "没有跑到任何 span");
    println!(
        "端到端对拍通过：{spans_run} 个 span，{tokens_checked} 个 TimedPiece 的 text 与 ms 全等"
    );

    // 同进程第二个实例：两处各自建 Environment 会直接失败（ort 只允许一个环境），
    // 这里锁住「环境共享自 crate::ort_util」这个集成约束。
    let (fixture, span, chunk, pieces) = first.unwrap_or_else(|| panic!("没有可复跑的 span"));
    let mut second =
        SenseVoice::new(&dir, 4).unwrap_or_else(|e| panic!("同进程第二个实例加载失败: {e}"));
    let out2 = second
        .transcribe_chunk(&chunk, SR, &job)
        .unwrap_or_else(|e| panic!("第二个实例推理失败: {e}"));
    let pieces2 = match out2 {
        AsrChunkOutput::Timed(p) => p,
        AsrChunkOutput::Plain(_) => panic!("第二个实例应返回 Timed"),
    };
    assert_eq!(
        pieces2, pieces,
        "{fixture}/span_{span:03}: 同进程第二个实例的输出应与第一个逐元素一致"
    );
    println!(
        "同进程双实例共存通过：{fixture}/span_{span:03} 两次输出逐元素一致（{} pieces）",
        pieces2.len()
    );
}
