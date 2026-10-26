//! 移植自 funasr-subtitle `backend/tests/test_segmentation.py`：用例名与断言语义一一对应，
//! 构造数据照抄；另附 normalize 的最小用例（has_cjk / normalize_latin_punct /
//! strip_terminal_punct / strip_terminal_timed）。

use funasr_subtitle_rs::text::normalize::{
    has_cjk, has_terminal, normalize_latin_punct, strip_special_tokens, strip_terminal_timed,
};
use funasr_subtitle_rs::text::segmentation::{
    enforce_monotonic, segment_span, segment_timed, split_sentences, strip_terminal_punct,
};
use funasr_subtitle_rs::types::{Segment, TimedPiece};

/// Python 侧 `MAX_CUE_MS`（config.py 默认 7000）：Rust 侧由调用方显式传入。
const MAX_CUE_MS: i64 = 7000;

fn timed(pairs: &[(&str, i64)]) -> Vec<TimedPiece> {
    pairs
        .iter()
        .map(|(text, ms)| TimedPiece {
            text: (*text).to_string(),
            time_ms: *ms,
        })
        .collect()
}

fn texts(segs: &[Segment]) -> Vec<&str> {
    segs.iter().map(|s| s.text.as_str()).collect()
}

// ---------------------------------------------------------------------------
// segmentation.py 用例逐条移植
// ---------------------------------------------------------------------------

#[test]
fn test_strip_terminal_punct_keeps_commas_and_decimals() {
    // 删句末终止标点，保留逗号、小数点、缩写内点
    assert_eq!(
        strip_terminal_punct("Hello, world. How are you?"),
        "Hello, world How are you"
    );
    assert_eq!(strip_terminal_punct("它花了 3.14 元。"), "它花了 3.14 元"); // 小数点保留、句末。删
    assert_eq!(strip_terminal_punct("the U.S. economy"), "the U.S economy"); // 句末位以外的点保留
    assert_eq!(strip_terminal_punct("a; b; c."), "a b c"); // 分号/句末删
    assert_eq!(strip_terminal_punct("no terminal here"), "no terminal here");
}

#[test]
fn test_enforce_monotonic_fixes_overlap_and_order() {
    let mut segs = vec![
        Segment {
            start_ms: 1000,
            end_ms: 1001,
            text: "Hi.".to_string(),
            speaker: None,
        },
        Segment {
            start_ms: 900,
            end_ms: 2000,
            text: "Yo.".to_string(),
            speaker: None,
        }, // 起点早于上一条终点
    ];
    enforce_monotonic(&mut segs);
    assert!(segs[0].end_ms <= segs[1].start_ms); // 不重叠
    assert!(segs.iter().all(|s| s.end_ms > s.start_ms)); // 每条非零时长
    assert_eq!(segs[1].start_ms, 1001); // 被钳到上一条 end
}

#[test]
fn test_timed_no_limit_keeps_clauses_together() {
    // max_chars=0：只在句末标点断句，逗号不断
    let pieces = timed(&[
        ("今", 0),
        ("天", 50),
        ("，", 100),
        ("好", 150),
        ("。", 200),
        ("走", 300),
        ("吧", 350),
        ("。", 400),
    ]);
    let segs = segment_timed(&pieces, 500, 0, MAX_CUE_MS);
    assert_eq!(texts(&segs), ["今天，好。", "走吧。"]);
    assert_eq!(segs[0].start_ms, 0);
    assert_eq!(segs[segs.len() - 1].end_ms, 500);
}

#[test]
fn test_timed_with_limit_splits_at_comma() {
    let pieces = timed(&[("今", 0), ("天", 50), ("，", 100), ("好", 150), ("。", 200)]);
    let segs = segment_timed(&pieces, 300, 2, MAX_CUE_MS);
    assert_eq!(segs.len(), 2); // 超过 2 字后在逗号断
}

#[test]
fn test_timed_preserves_english_spaces() {
    // piece 带词边界空格（来自 sentencepiece ▁ -> space），不应被吃掉
    let pieces = timed(&[("Today", 0), (" the", 50), (" weather", 100), (".", 150)]);
    let segs = segment_timed(&pieces, 200, 0, MAX_CUE_MS);
    assert_eq!(segs[0].text, "Today the weather.");
}

#[test]
fn test_timed_english_period_space_splits() {
    let pieces = timed(&[
        ("Hi", 0),
        (".", 100),
        (" Go", 200),
        (" on", 300),
        (".", 400),
    ]);
    let segs = segment_timed(&pieces, 500, 0, MAX_CUE_MS);
    assert_eq!(texts(&segs), ["Hi.", "Go on."]);
}

#[test]
fn test_split_does_not_break_english_abbreviations() {
    // e.g. / U.S. / Inc. 里的句点不应当作句末切句
    assert_eq!(
        split_sentences("the U.S. economy grew.", 0),
        vec!["the U.S. economy grew."]
    );
    assert_eq!(
        split_sentences("see e.g. this example.", 0),
        vec!["see e.g. this example."]
    );
    assert_eq!(
        split_sentences("Acme Inc. shipped it.", 0),
        vec!["Acme Inc. shipped it."]
    );
    // 真正的句末仍要切
    assert_eq!(
        split_sentences("It works. We ship it.", 0),
        vec!["It works.", "We ship it."]
    );
}

#[test]
fn test_split_long_cjk_latin_mix_no_oversized_line() {
    // CJK 前缀 + 无空格长拉丁串：不应整体留成一条远超宽度的行
    let s = format!(
        "{}{}{}",
        "中".repeat(5),
        "abcdefghijklmnopqrstuvwxyz0123",
        "文".repeat(5)
    );
    let pieces = split_sentences(&s, 20);
    // 至少切成 2 段，且不是「单段全塞」
    assert!(pieces.len() >= 2);
    // 拉丁长词本身不拆，但 CJK 边界要利用上，首段应是纯 CJK 前缀
    assert_eq!(pieces[0], "中".repeat(5));
}

#[test]
fn test_single_letter_period_splits_when_next_lowercase() {
    // 句末单字母 + 下一词小写 = 真正句末，应切
    assert_eq!(
        split_sentences("the grade was a. then class ended.", 0),
        vec!["the grade was a.", "then class ended."]
    );
    // 单字母 + 下一词大写(像首字母/名字) = 不切
    assert_eq!(
        split_sentences("A. Smith arrived late.", 0),
        vec!["A. Smith arrived late."]
    );
}

#[test]
fn test_timed_abbreviation_not_split() {
    let pieces = timed(&[
        ("the", 0),
        (" U", 50),
        (".S", 100),
        (".", 150),
        (" economy", 200),
        (" grew", 250),
        (".", 300),
    ]);
    let segs = segment_timed(&pieces, 400, 0, MAX_CUE_MS);
    assert_eq!(texts(&segs), ["the U.S. economy grew."]);
}

#[test]
fn test_sentence_final_abbrev_still_splits_when_next_capitalized() {
    // 句末缩写 + 下一句首字母大写 -> 应当切句（不能因为 etc./Inc. 就并成一句）
    assert_eq!(
        split_sentences("We use it etc. They agree.", 0),
        vec!["We use it etc.", "They agree."]
    );
    assert_eq!(
        split_sentences("Made by Acme Inc. They ship.", 0),
        vec!["Made by Acme Inc.", "They ship."]
    );
    // 但缩写后接小写延续 -> 不切（e.g./U.S./延续）
    assert_eq!(
        split_sentences("the U.S. economy grew fast.", 0),
        vec!["the U.S. economy grew fast."]
    );
    assert_eq!(
        split_sentences("see e.g. this and that.", 0),
        vec!["see e.g. this and that."]
    );
}

#[test]
fn test_no_orphan_punctuation() {
    // 纯标点碎片应并入上一句，不单独成行
    assert_eq!(split_sentences("好几段。", 4), vec!["好几段。"]);
    let segs = segment_timed(
        &timed(&[("好", 0), ("几", 50), ("段", 100), ("。", 150)]),
        200,
        2,
        MAX_CUE_MS,
    );
    assert!(segs.iter().all(|s| s.text.trim() != "。"));
    assert_eq!(
        segs.iter().map(|s| s.text.as_str()).collect::<String>(),
        "好几段。"
    );
}

#[test]
fn test_split_long_english_no_punct_keeps_words() {
    // 无标点的英文长句：按显示宽度在空格处断，绝不拆词
    let parts = split_sentences("the quick brown fox jumps over the lazy dog", 8);
    assert!(parts.len() >= 2);
    assert_eq!(
        parts.join(" ").split_whitespace().collect::<Vec<_>>(),
        "the quick brown fox jumps over the lazy dog"
            .split_whitespace()
            .collect::<Vec<_>>()
    ); // 单词完整
}

#[test]
fn test_width_english_longer_than_cjk() {
    // 同一 max_chars 下，拉丁每行容纳的字符数约为 CJK 的 2 倍（显示宽度）
    let en = split_sentences(
        "alpha bravo charlie delta echo foxtrot golf hotel india",
        10,
    );
    // 10 宽 ≈ 20 拉丁字符/行；每行可见字符数应明显超过 10
    let widest = en
        .iter()
        .map(|p| p.replace(' ', "").chars().count())
        .max()
        .unwrap_or(0);
    assert!(widest > 12);
}

#[test]
fn test_split_long_unbreakable_word_not_cut() {
    // 超长拉丁词且无空格：不从中间切（宁可超长）
    assert_eq!(
        split_sentences("supercalifragilisticexpialidocious", 8),
        vec!["supercalifragilisticexpialidocious"]
    );
}

#[test]
fn test_split_long_cjk_no_punct_char_cut() {
    let parts = split_sentences("一二三四五六七八九十", 4);
    assert_eq!(parts, vec!["一二三四", "五六七八", "九十"]);
}

#[test]
fn test_timed_english_cap_breaks_at_word_boundary() {
    let pieces = timed(&[("the", 0), (" quick", 50), (" brown", 100), (" fox", 150)]);
    let segs = segment_timed(&pieces, 200, 8, MAX_CUE_MS);
    for s in &segs {
        // 不拆词
        for w in s.text.split_whitespace() {
            assert!(["the", "quick", "brown", "fox"].contains(&w));
        }
    }
}

#[test]
fn test_timed_charlevel_no_empty_terminal_bug() {
    // 字符级 pieces（含纯空白）：空白 piece 的 last="" 不能被当作句末标点
    // （"" in 任意串都为 True 的坑）；应只在 "." 处断，且第二句用真实时间。
    let pieces = timed(&[
        ("H", 100),
        ("i", 100),
        (".", 100),
        (" ", 100),
        ("G", 900),
        ("o", 900),
    ]);
    let segs = segment_timed(&pieces, 1000, 0, MAX_CUE_MS);
    assert_eq!(texts(&segs), ["Hi.", "Go"]);
    assert_eq!(segs[1].start_ms, 900); // 真实时间（非前一 piece/比例）
}

#[test]
fn test_timed_decimal_not_split() {
    // '.' 后无空格（小数/缩写）不断句
    let pieces = timed(&[("3", 0), (".", 50), ("14", 100), (" pies", 150)]);
    let segs = segment_timed(&pieces, 200, 0, MAX_CUE_MS);
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].text, "3.14 pies");
}

#[test]
fn test_split_at_sentence_punctuation() {
    let parts = split_sentences("今天天气非常好，我们去公园吧。这是一个测试。", 0);
    assert_eq!(
        parts,
        vec!["今天天气非常好，我们去公园吧。", "这是一个测试。"]
    );
}

#[test]
fn test_split_long_sentence_at_comma() {
    let s = "甲乙丙丁戊己庚辛，壬癸子丑寅卯辰巳午未申酉戌亥再加几个字凑长一点继续";
    let parts = split_sentences(s, 12);
    assert!(parts.len() >= 2);
    // 容许标点略超
    assert!(
        parts
            .iter()
            .all(|p| p.replace(' ', "").chars().count() <= 12 + 2)
    );
}

#[test]
fn test_segment_span_allocates_time_proportionally() {
    let segs = segment_span("今天天气非常好。这是一个本地语音转写的测试。", 0, 10000, 0);
    assert_eq!(segs.len(), 2);
    // 连续、覆盖整段
    assert_eq!(segs[0].start_ms, 0);
    assert_eq!(segs[segs.len() - 1].end_ms, 10000);
    assert_eq!(segs[0].end_ms, segs[1].start_ms);
    // 第二句更长 -> 占用更多时间
    assert!((segs[1].end_ms - segs[1].start_ms) > (segs[0].end_ms - segs[0].start_ms));
}

#[test]
fn test_single_sentence_keeps_span() {
    let segs = segment_span("就一句话", 1000, 3000, 20);
    assert_eq!(segs.len(), 1);
    assert_eq!((segs[0].start_ms, segs[0].end_ms), (1000, 3000));
}

#[test]
fn test_empty_text() {
    assert_eq!(segment_span("   ", 0, 1000, 20), Vec::<Segment>::new());
}

// ---------------------------------------------------------------------------
// normalize.rs：最小用例
// ---------------------------------------------------------------------------

#[test]
fn test_has_cjk_mixed() {
    assert!(has_cjk("今天 weather")); // 中文
    assert!(has_cjk("カタカナ")); // 日文片假名
    assert!(has_cjk("한국어")); // 韩文
    assert!(!has_cjk("pure latin text, 123."));
    assert!(!has_cjk("“quoted” — no ideographs"));
    assert!(!has_cjk(""));
}

#[test]
fn test_has_terminal() {
    assert!(has_terminal("你好。"));
    assert!(has_terminal("hi."));
    assert!(has_terminal("why?"));
    assert!(!has_terminal("hello, world"));
    assert!(!has_terminal(""));
}

#[test]
fn test_normalize_latin_punct_fullwidth_to_halfwidth() {
    // 全角 -> 半角（后跟空格）
    assert_eq!(normalize_latin_punct("hello，world。"), "hello, world.");
    assert_eq!(
        normalize_latin_punct("a！b？c；d：e、f"),
        "a! b? c; d: e, f"
    );
    // 空白规整：多空白折成一个空格并 strip
    assert_eq!(
        normalize_latin_punct("  multiple \t spaces\n here  "),
        "multiple spaces here"
    );
    // 含 CJK 时原样返回（不做任何替换/规整）
    assert_eq!(normalize_latin_punct("中文，标点。"), "中文，标点。");
}

#[test]
fn test_strip_special_tokens() {
    assert_eq!(strip_special_tokens("<|zh|><|NEUTRAL|>你好"), "你好");
    assert_eq!(
        strip_special_tokens("<|en|>hello <|HAPPY|>world"),
        "hello world"
    );
    // 未闭合不匹配
    assert_eq!(strip_special_tokens("<|broken text"), "<|broken text");
    // 内部含 '|' 或 '>' 不算标记（[^|>]* 语义）
    assert_eq!(strip_special_tokens("<|a>b|>x"), "<|a>b|>x");
    assert_eq!(strip_special_tokens("no tokens"), "no tokens");
}

#[test]
fn test_strip_terminal_punct_decimal_and_abbrev_dots_kept() {
    // 小数点、缩写内点保留；句末句点删除
    assert_eq!(strip_terminal_punct("3.14"), "3.14");
    assert_eq!(strip_terminal_punct("U.S."), "U.S");
    assert_eq!(strip_terminal_punct("hello."), "hello");
}

#[test]
fn test_strip_terminal_timed_same_rules_with_times() {
    // "3.14 元。" -> 小数点保留、句末。删；每个字符沿用原 token 时间
    let input = timed(&[("3", 0), (".", 50), ("14", 100), (" 元", 150), ("。", 200)]);
    let out = strip_terminal_timed(&input);
    assert_eq!(
        out.iter().map(|p| p.text.as_str()).collect::<String>(),
        "3.14 元"
    );
    assert_eq!(
        out.iter().map(|p| p.time_ms).collect::<Vec<_>>(),
        vec![0, 50, 100, 100, 150, 150]
    ); // '.' 保留在 50；"1"/"4"@100；" "@150、"元"@150

    // "U.S." -> 内部点保留、句末点删除
    let out = strip_terminal_timed(&timed(&[("U", 0), (".", 10), ("S", 20), (".", 30)]));
    assert_eq!(
        out.iter().map(|p| p.text.as_str()).collect::<String>(),
        "U.S"
    );
    assert_eq!(
        out.iter().map(|p| p.time_ms).collect::<Vec<_>>(),
        vec![0, 10, 20]
    );

    // "hello." -> 句点删除
    let out = strip_terminal_timed(&timed(&[("hello", 0), (".", 100)]));
    assert_eq!(
        out.iter().map(|p| p.text.as_str()).collect::<String>(),
        "hello"
    );
}

/// 契约统一后（CPython isspace 含 U+001C–U+001F）的回归锁死：
/// bidi 控制符必须被当作空白处理，且 text/export 两侧共用同一实现。
#[test]
fn test_is_py_space_covers_bidi_controls() {
    use funasr_subtitle_rs::text::normalize::is_py_space;
    assert!(
        is_py_space('\u{1c}')
            && is_py_space('\u{1d}')
            && is_py_space('\u{1e}')
            && is_py_space('\u{1f}')
    );
    assert!(!is_py_space('\u{200b}')); // ZWSP 不是空白
    assert!(is_py_space('\u{85}') && is_py_space('\u{a0}'));
}
