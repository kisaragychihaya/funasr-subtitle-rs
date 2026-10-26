//! T9 验收测试：CT-Transformer 标点（`src/punc.rs`）。
//!
//! 1) 纯逻辑：`plan_chunks` 与 `tests/golden/punc/chunk_plan.json` 对拍 + 边界用例；
//!    `align_punct_to_timed` 与 `reference/testdata/punc_align_cases.json` 逐条全等比对
//!    （对拍数据由 Python 参考实现 `_apply_punct_to_timed` 生成）。
//! 2) `#[ignore]` 真实模型：`Punctuator::new` + `punctuate` 跑
//!    `tests/golden/punc/cases.json`，输出逐字节一致。

use std::fs;
use std::path::{Path, PathBuf};

use funasr_subtitle_rs::models::PUNC_MODEL;
use funasr_subtitle_rs::models::manager::{ModelManager, default_models_root, has_onnx, model_dir};
use funasr_subtitle_rs::punc::{
    PUNC_CHUNK, Punctuator, align_punct_to_timed, code_mix_split_words, plan_chunks,
};
use funasr_subtitle_rs::types::TimedPiece;

// ---------------------------------------------------------------------------
// 通用小工具
// ---------------------------------------------------------------------------

fn manifest_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn read_json<T: serde::de::DeserializeOwned>(rel: &str) -> T {
    let path = manifest_path(rel);
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("读取 {} 失败: {e}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("解析 {} 失败: {e}", path.display()))
}

fn timed(pairs: &[(&str, i64)]) -> Vec<TimedPiece> {
    pairs
        .iter()
        .map(|(text, ms)| TimedPiece {
            text: (*text).to_string(),
            time_ms: *ms,
        })
        .collect()
}

fn align_pairs(out: &[TimedPiece]) -> Vec<(String, i64)> {
    out.iter()
        .map(|p| (p.text.clone(), p.time_ms))
        .collect()
}

/// 逐字拼成一个字符串（align 的输出是一字符一个 TimedPiece）。
fn joined(out: &[TimedPiece]) -> String {
    out.iter().map(|p| p.text.as_str()).collect()
}

// ---------------------------------------------------------------------------
// 分块逻辑：与 golden chunk_plan.json 对拍
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct PuncCase {
    input: String,
    output: String,
}

#[derive(serde::Deserialize)]
struct ChunkPlanFile {
    punc_chunk: usize,
    single_call_limit: usize,
    cases: Vec<ChunkPlanCase>,
}

#[derive(serde::Deserialize)]
struct ChunkPlanCase {
    input_len: usize,
    chunked: bool,
    chunks: Vec<String>,
}

#[test]
fn test_chunk_plan_constants_match_config() {
    let golden: ChunkPlanFile = read_json("tests/golden/punc/chunk_plan.json");
    assert_eq!(golden.punc_chunk, PUNC_CHUNK, "PUNC_CHUNK 必须等于 config.py 默认值");
    assert_eq!(
        golden.single_call_limit,
        PUNC_CHUNK + 30,
        "单次调用阈值必须是 PUNC_CHUNK + 30"
    );
}

#[test]
fn test_chunk_plan_matches_golden() {
    let golden: ChunkPlanFile = read_json("tests/golden/punc/chunk_plan.json");
    let cases: Vec<PuncCase> = read_json("tests/golden/punc/cases.json");
    assert_eq!(
        golden.cases.len(),
        cases.len(),
        "chunk_plan.json 与 cases.json 用例数不一致"
    );
    for (i, (expected, case)) in golden.cases.iter().zip(cases.iter()).enumerate() {
        assert_eq!(
            case.input.chars().count(),
            expected.input_len,
            "case #{i} 字符数与 golden 记录的 input_len 不一致"
        );
        let plan = plan_chunks(&case.input);
        assert_eq!(plan.chunked, expected.chunked, "case #{i} chunked 判定不一致");
        assert_eq!(plan.chunks, expected.chunks, "case #{i} 分块结果不一致");
    }
}

#[test]
fn test_chunk_plan_long_text_is_split_at_punc_chunk() {
    // golden 里的 230 字长文本：110 / 110 / 10 三块（纯 CJK 无空格可回退）
    let text: String = "今天我们来讲一下语音识别技术的基本原理".chars().cycle().take(230).collect();
    let plan = plan_chunks(&text);
    assert!(plan.chunked);
    let lens: Vec<usize> = plan.chunks.iter().map(|c| c.chars().count()).collect();
    assert_eq!(lens, vec![PUNC_CHUNK, PUNC_CHUNK, 230 - 2 * PUNC_CHUNK]);
    assert_eq!(plan.chunks.concat(), text);
}

// ---------------------------------------------------------------------------
// 分块逻辑：边界用例（110 / 140 / 141 字、空格回退、多空格、块内全空白）
// ---------------------------------------------------------------------------

#[test]
fn test_chunk_plan_boundary_lengths() {
    // ≤ 140 字：单次调用，chunks 就是原文
    for len in [1usize, PUNC_CHUNK, PUNC_CHUNK + 30] {
        let text = "a".repeat(len);
        let plan = plan_chunks(&text);
        assert!(!plan.chunked, "len={len} 应该走单次调用");
        assert_eq!(plan.chunks, vec![text]);
    }
    // 141 字：分块
    let text = "a".repeat(PUNC_CHUNK + 31);
    let plan = plan_chunks(&text);
    assert!(plan.chunked, "len=141 应该走分块");
    assert_eq!(
        plan.chunks,
        vec!["a".repeat(PUNC_CHUNK), "a".repeat(31)],
        "纯 ASCII 无空格时应按长度硬切"
    );
}

#[test]
fn test_chunk_plan_falls_back_to_last_space() {
    // 第 105 个字符是空格（下标 105，块窗口 [0,110) 内最后一个空格）
    let text = format!("{}{}{}", "x".repeat(105), " ", "y".repeat(40));
    assert_eq!(text.chars().count(), 146);
    let plan = plan_chunks(&text);
    assert!(plan.chunked);
    assert_eq!(
        plan.chunks,
        vec!["x".repeat(105), "y".repeat(40)],
        "应在块内最后一个空格处回退并逐块 strip"
    );
}

#[test]
fn test_chunk_plan_space_at_chunk_start_does_not_move_cut() {
    // Python 的 `if sp > i: end = sp`：空格正好在块首（i）时不生效，仍按 110 硬切
    let text = format!("{}{}", " ", "z".repeat(140));
    assert_eq!(text.chars().count(), 141);
    let plan = plan_chunks(&text);
    assert!(plan.chunked);
    assert_eq!(
        plan.chunks,
        vec!["z".repeat(109), "z".repeat(31)],
        "块首空格不应成为切点，且块内 strip 吃掉它"
    );
}

#[test]
fn test_chunk_plan_multiple_spaces_and_blank_chunk_skipped() {
    // w*110 + 20 个空格 + w*20：中间那块 strip 后为空 -> 丢弃；join 只加一个空格
    let text = format!("{}{}{}", "w".repeat(110), " ".repeat(20), "w".repeat(20));
    assert_eq!(text.chars().count(), 150);
    let plan = plan_chunks(&text);
    assert!(plan.chunked);
    assert_eq!(plan.chunks, vec!["w".repeat(110), "w".repeat(20)]);
    assert_eq!(plan.chunks.join(" "), format!("{}{}{}", "w".repeat(110), " ", "w".repeat(20)));
}

#[test]
fn test_chunk_plan_only_ascii_space_is_a_cut_candidate() {
    // rfind(" ") 只认半角空格：全角空格 / 制表符都不作为回退点，切点仍是 110
    let text = format!("{}{}{}", "x".repeat(100), "\u{3000}", "y".repeat(40));
    assert_eq!(text.chars().count(), 141);
    let plan = plan_chunks(&text);
    assert!(plan.chunked);
    assert_eq!(
        plan.chunks,
        vec![
            format!("{}{}{}", "x".repeat(100), "\u{3000}", "y".repeat(9)),
            "y".repeat(31)
        ],
        "全角空格不是回退点，切点仍是 110"
    );

    // 块内有半角空格时才会回退（倒数第二个字符是半角空格 -> 切在它上面）
    let text = format!("{}{}{}", "x".repeat(105), " ", "y".repeat(40));
    let plan = plan_chunks(&text);
    assert_eq!(plan.chunks, vec!["x".repeat(105), "y".repeat(40)]);
}

// ---------------------------------------------------------------------------
// 逐字对齐：与 Python 参考实现对拍
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct AlignFile {
    count: usize,
    cases: Vec<AlignCase>,
}

#[derive(serde::Deserialize)]
struct AlignCase {
    timed: Vec<(String, i64)>,
    punct: String,
    expected: Vec<(String, i64)>,
}

#[test]
fn test_align_matches_python_reference() {
    let path = manifest_path("reference/testdata/punc_align_cases.json");
    let raw = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "读取对拍数据 {} 失败: {e}\n\
             生成方式：用 Python 参考实现 reference/py-src/funasr_engine.py 的 \
             `_apply_punct_to_timed`（AST 抽函数原样执行）随机造 TimedPiece 序列 + punct_text，\
             写出 {{\"count\": n, \"cases\": [{{\"timed\": [[piece, ms], ...], \
             \"punct\": \"...\", \"expected\": [[char, ms], ...]}}]}}。\
             本仓库生成脚本见 target/tmp/gen_punc_align.py（不进仓库）。",
            path.display()
        )
    });
    let file: AlignFile = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("解析 {} 失败: {e}", path.display()));
    assert!(file.count >= 2000, "对拍用例数 {} 少于 2000", file.count);
    assert_eq!(file.count, file.cases.len());

    let mut mismatched = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for (i, case) in file.cases.iter().enumerate() {
        let input: Vec<TimedPiece> = case
            .timed
            .iter()
            .map(|(text, ms)| TimedPiece {
                text: text.clone(),
                time_ms: *ms,
            })
            .collect();
        let got = align_pairs(&align_punct_to_timed(&input, &case.punct));
        if got != case.expected {
            mismatched += 1;
            if failures.len() < 5 {
                failures.push(format!(
                    "case #{i}\n  timed   = {:?}\n  punct   = {:?}\n  want    = {:?}\n  got     = {:?}",
                    case.timed, case.punct, case.expected, got
                ));
            }
        }
    }
    assert!(
        mismatched == 0,
        "{mismatched} / {} 条用例与 Python 参考实现不一致，前几条：\n{}",
        file.count,
        failures.join("\n")
    );
}

#[test]
fn test_align_basic_behaviors() {
    // 新插入标点用上一个字符的时间
    let out = align_punct_to_timed(&timed(&[("你好", 0), ("世界", 100)]), "你好，世界。");
    assert_eq!(joined(&out), "你好，世界。");
    assert_eq!(
        align_pairs(&out),
        vec![
            ("你".to_string(), 0),
            ("好".to_string(), 0),
            ("，".to_string(), 0),
            ("世".to_string(), 100),
            ("界".to_string(), 100),
            ("。".to_string(), 100),
        ]
    );

    // 空格用上一个时间（不再是比例分配）
    let out = align_punct_to_timed(&timed(&[("ab", 5), ("cd", 9)]), "ab cd");
    assert_eq!(align_pairs(&out), vec![
        ("a".to_string(), 5),
        ("b".to_string(), 5),
        (" ".to_string(), 5),
        ("c".to_string(), 9),
        ("d".to_string(), 9),
    ]);

    // 空 timed：全部用 last_t = 0
    assert_eq!(align_pairs(&align_punct_to_timed(&[], "。")), vec![("。".to_string(), 0)]);
    assert!(align_punct_to_timed(&timed(&[("abc", 1)]), "").is_empty());
}

#[test]
fn test_align_case_insensitive_and_resync_window() {
    // 大小写失配仍对上，时间不塌
    let out = align_punct_to_timed(&timed(&[("HELLO", 10), ("world", 20)]), "hello world");
    assert_eq!(
        align_pairs(&out),
        vec![
            ("h".to_string(), 10),
            ("e".to_string(), 10),
            ("l".to_string(), 10),
            ("l".to_string(), 10),
            ("o".to_string(), 10),
            (" ".to_string(), 10),
            ("w".to_string(), 20),
            ("o".to_string(), 20),
            ("r".to_string(), 20),
            ("l".to_string(), 20),
            ("d".to_string(), 20),
        ]
    );

    // 8 个多余字符仍在窗口内 -> 之后重新对齐到真实时间
    let out = align_punct_to_timed(&timed(&[("ab", 1), ("cd", 2)]), "xxab，，cd");
    let times: Vec<i64> = out.iter().map(|p| p.time_ms).collect();
    assert_eq!(times, vec![1, 1, 1, 1, 1, 1, 2, 2], "窗口内应重同步回 cd 的真实时间");

    // 超出 8 字符窗口 -> 该字符落回 last_t，后续按原序继续
    let out = align_punct_to_timed(&timed(&[("ab", 1), ("cd", 2)]), "ab123456789cd");
    let tail: Vec<i64> = out.iter().skip(2).take(9).map(|p| p.time_ms).collect();
    assert_eq!(tail, vec![1; 9], "超出窗口的字符应用上一字符时间");
    assert_eq!(
        out.last().map(|p| p.time_ms),
        Some(2),
        "后续字符仍应回到真实时间"
    );
}

// ---------------------------------------------------------------------------
// 分词：code_mix_split_words
// ---------------------------------------------------------------------------

#[test]
fn test_code_mix_split_words() {
    assert_eq!(code_mix_split_words(""), Vec::<String>::new());
    assert_eq!(code_mix_split_words("   "), Vec::<String>::new());
    assert_eq!(
        code_mix_split_words("hello world"),
        vec!["hello".to_string(), "world".to_string()]
    );
    // 中文逐字、ASCII 连续成词
    assert_eq!(
        code_mix_split_words("今天CTC好用"),
        vec!["今".to_string(), "天".to_string(), "CTC".to_string(), "好".to_string(), "用".to_string()]
    );
    // 任意空白（含全角空格 / 制表）都是分隔符
    assert_eq!(
        code_mix_split_words("a\u{3000}b\tc"),
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );
}

// ---------------------------------------------------------------------------
// 真实模型对拍（#[ignore]）
// ---------------------------------------------------------------------------

/// 模型根目录：默认 `default_models_root()`；`FUNASR_SUBTITLE_MODELS_ROOT` 可覆盖
/// （把它指向空目录即可强制走下载路径）。
fn models_root() -> PathBuf {
    std::env::var_os("FUNASR_SUBTITLE_MODELS_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(default_models_root)
}

/// 定位 ct-punc 模型目录：优先用已下载好的缓存，缺失时下载到临时目录（约 290MB）。
fn punc_model_dir(tmp: &mut Option<tempfile::TempDir>) -> PathBuf {
    let cached = model_dir(&models_root(), &PUNC_MODEL);
    if has_onnx(&cached) {
        return cached;
    }
    let dir = tempfile::tempdir().expect("创建临时模型目录失败");
    let path = ModelManager::new(dir.path().to_path_buf())
        .ensure_model(&PUNC_MODEL)
        .expect("下载 ct-punc 模型失败（该分支需要联网，约 290MB）");
    *tmp = Some(dir);
    path
}

#[test]
#[ignore = "需要 ct-punc 模型（约 290MB）；手动运行：cargo test --test test_punc -- --ignored"]
fn test_punc_real_model_matches_golden_cases() {
    let mut tmp = None;
    let model_dir = punc_model_dir(&mut tmp);
    println!("ct-punc 模型目录: {}", model_dir.display());
    let mut punc = Punctuator::new(&model_dir, 4).expect("加载 ct-punc 模型失败");
    let cases: Vec<PuncCase> = read_json("tests/golden/punc/cases.json");
    let mut mismatches = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let got = punc
            .punctuate(&case.input)
            .unwrap_or_else(|e| panic!("case #{i} 推理失败: {e}"));
        let equal = got == case.output;
        println!(
            "case #{i} (len={}): {}\n  input = {}\n  want  = {}\n  got   = {}",
            case.input.chars().count(),
            if equal { "byte-equal" } else { "MISMATCH" },
            case.input,
            case.output,
            got
        );
        if !equal {
            mismatches.push(i);
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} / {} 个用例与 golden 不一致（case 序号 {:?}）",
        mismatches.len(),
        cases.len(),
        mismatches
    );
}
