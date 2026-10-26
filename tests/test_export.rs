//! T2 验收测试：SRT / VTT / TXT / JSON 导出。
//!
//! 所有期望值均由参考实现 `backend/app/subtitle/formatters.py` 实际运行产出。

use funasr_subtitle_rs::export::{to_json, to_srt, to_txt, to_vtt};
use funasr_subtitle_rs::types::Segment;

fn seg(start_ms: i64, end_ms: i64, text: &str) -> Segment {
    Segment {
        start_ms,
        end_ms,
        text: text.to_string(),
        speaker: None,
    }
}

#[test]
fn empty_list_exact_output() {
    let empty: Vec<Segment> = Vec::new();
    assert_eq!(to_srt(&empty), "");
    assert_eq!(to_vtt(&empty), "WEBVTT\n");
    assert_eq!(to_txt(&empty), "");
    assert_eq!(to_json(&empty), "[]");
}

#[test]
fn single_cue_normal_time() {
    let segs = [seg(0, 1500, "Hello world")];
    assert_eq!(
        to_srt(&segs),
        "1\n00:00:00,000 --> 00:00:01,500\nHello world\n"
    );
    assert_eq!(
        to_vtt(&segs),
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello world\n"
    );
    assert_eq!(to_txt(&segs), "Hello world");
    assert_eq!(
        to_json(&segs),
        "[\n  {\n    \"start_ms\": 0,\n    \"end_ms\": 1500,\n    \"text\": \"Hello world\"\n  }\n]"
    );
}

#[test]
fn millisecond_carry_not_rounded() {
    let segs = [seg(0, 999, "a"), seg(999, 1000, "b"), seg(999, 1999, "c")];
    assert_eq!(
        to_srt(&segs),
        "1\n00:00:00,000 --> 00:00:00,999\na\n\n\
         2\n00:00:00,999 --> 00:00:01,000\nb\n\n\
         3\n00:00:00,999 --> 00:00:01,999\nc\n"
    );
    assert_eq!(
        to_vtt(&segs),
        "WEBVTT\n\n00:00:00.000 --> 00:00:00.999\na\n\n\
         00:00:00.999 --> 00:00:01.000\nb\n\n\
         00:00:00.999 --> 00:00:01.999\nc\n"
    );
    assert_eq!(to_txt(&segs), "a\nb\nc");
    assert_eq!(
        to_json(&segs),
        "[\n  {\n    \"start_ms\": 0,\n    \"end_ms\": 999,\n    \"text\": \"a\"\n  },\n  \
         {\n    \"start_ms\": 999,\n    \"end_ms\": 1000,\n    \"text\": \"b\"\n  },\n  \
         {\n    \"start_ms\": 999,\n    \"end_ms\": 1999,\n    \"text\": \"c\"\n  }\n]"
    );
}

#[test]
fn minute_and_hour_carry() {
    let segs = [seg(59_999, 60_000, "a"), seg(3_599_999, 3_600_000, "b")];
    assert_eq!(
        to_srt(&segs),
        "1\n00:00:59,999 --> 00:01:00,000\na\n\n2\n00:59:59,999 --> 01:00:00,000\nb\n"
    );
    assert_eq!(
        to_vtt(&segs),
        "WEBVTT\n\n00:00:59.999 --> 00:01:00.000\na\n\n00:59:59.999 --> 01:00:00.000\nb\n"
    );
}

#[test]
fn cross_hour_timestamp() {
    let segs = [seg(3_661_000, 3_662_500, "cross hour")];
    assert_eq!(
        to_srt(&segs),
        "1\n01:01:01,000 --> 01:01:02,500\ncross hour\n"
    );
    assert_eq!(
        to_vtt(&segs),
        "WEBVTT\n\n01:01:01.000 --> 01:01:02.500\ncross hour\n"
    );
    assert_eq!(to_txt(&segs), "cross hour");
}

#[test]
fn hour_field_wider_than_two_digits() {
    let segs = [seg(360_000_000, 360_001_234, "wide")];
    assert_eq!(to_srt(&segs), "1\n100:00:00,000 --> 100:00:01,234\nwide\n");
    assert_eq!(
        to_vtt(&segs),
        "WEBVTT\n\n100:00:00.000 --> 100:00:01.234\nwide\n"
    );
}

#[test]
fn multiple_cues_numbering_and_blank_line_separation() {
    let segs = [
        seg(0, 1000, "one"),
        seg(1000, 2000, "two"),
        seg(2000, 3000, "three"),
    ];
    let srt = to_srt(&segs);
    assert_eq!(
        srt,
        "1\n00:00:00,000 --> 00:00:01,000\none\n\n\
         2\n00:00:01,000 --> 00:00:02,000\ntwo\n\n\
         3\n00:00:02,000 --> 00:00:03,000\nthree\n"
    );
    // 序号从 1 开始，块之间恰好一个空行，文件末尾只有一个换行。
    assert!(srt.starts_with("1\n"));
    assert_eq!(srt.matches("\n\n").count(), 2);
    assert!(!srt.ends_with("\n\n"));

    let vtt = to_vtt(&segs);
    assert_eq!(
        vtt,
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\none\n\n\
         00:00:01.000 --> 00:00:02.000\ntwo\n\n\
         00:00:02.000 --> 00:00:03.000\nthree\n"
    );
    assert!(vtt.starts_with("WEBVTT\n\n"));
    assert!(!vtt.ends_with("\n\n"));

    assert_eq!(to_txt(&segs), "one\ntwo\nthree");
    assert_eq!(
        to_json(&segs),
        "[\n  {\n    \"start_ms\": 0,\n    \"end_ms\": 1000,\n    \"text\": \"one\"\n  },\n  \
         {\n    \"start_ms\": 1000,\n    \"end_ms\": 2000,\n    \"text\": \"two\"\n  },\n  \
         {\n    \"start_ms\": 2000,\n    \"end_ms\": 3000,\n    \"text\": \"three\"\n  }\n]"
    );
}

#[test]
fn txt_one_line_per_segment_without_trailing_newline() {
    let segs = [seg(0, 1000, "first"), seg(1000, 2000, "second")];
    assert_eq!(to_txt(&segs), "first\nsecond");
}

#[test]
fn json_field_names_and_chinese_content() {
    let segs = [seg(0, 1500, "你好，世界。")];
    assert_eq!(
        to_json(&segs),
        "[\n  {\n    \"start_ms\": 0,\n    \"end_ms\": 1500,\n    \"text\": \"你好，世界。\"\n  }\n]"
    );
    assert_eq!(to_txt(&segs), "你好，世界。");

    let value: serde_json::Value = serde_json::from_str(&to_json(&segs)).unwrap_or_default();
    let first = value.get(0).cloned().unwrap_or_default();
    let keys: Vec<&str> = first
        .as_object()
        .map(|obj| obj.keys().map(String::as_str).collect())
        .unwrap_or_default();
    assert_eq!(keys, ["end_ms", "start_ms", "text"]);
}

#[test]
fn json_escapes_quotes_and_backslashes() {
    let segs = [
        seg(0, 1000, "he said \"hi\" \\ back\\slash"),
        seg(1000, 2000, "tab\there\nnewline"),
    ];
    assert_eq!(
        to_json(&segs),
        "[\n  {\n    \"start_ms\": 0,\n    \"end_ms\": 1000,\n    \
         \"text\": \"he said \\\"hi\\\" \\\\ back\\\\slash\"\n  },\n  \
         {\n    \"start_ms\": 1000,\n    \"end_ms\": 2000,\n    \
         \"text\": \"tab\\there\\nnewline\"\n  }\n]"
    );
    // SRT/VTT/TXT 里的引号与反斜杠原样输出；换行被折叠成空格。
    assert_eq!(
        to_srt(&segs),
        "1\n00:00:00,000 --> 00:00:01,000\nhe said \"hi\" \\ back\\slash\n\n\
         2\n00:00:01,000 --> 00:00:02,000\ntab\there newline\n"
    );
    assert_eq!(
        to_txt(&segs),
        "he said \"hi\" \\ back\\slash\ntab\there newline"
    );
}

#[test]
fn json_escapes_control_chars_but_not_del() {
    let segs = [seg(0, 1000, "ctl\u{1}\u{1f}\u{7f} end")];
    assert_eq!(
        to_json(&segs),
        "[\n  {\n    \"start_ms\": 0,\n    \"end_ms\": 1000,\n    \
         \"text\": \"ctl\\u0001\\u001f\u{7f} end\"\n  }\n]"
    );
    assert_eq!(
        to_srt(&segs),
        "1\n00:00:00,000 --> 00:00:01,000\nctl\u{1}\u{1f}\u{7f} end\n"
    );
}

#[test]
fn json_keeps_non_ascii_raw() {
    let segs = [seg(0, 1000, "emoji 😀 中文")];
    assert_eq!(
        to_json(&segs),
        "[\n  {\n    \"start_ms\": 0,\n    \"end_ms\": 1000,\n    \"text\": \"emoji 😀 中文\"\n  }\n]"
    );
}

#[test]
fn text_is_trimmed_in_all_formats() {
    let segs = [seg(0, 1000, "   padded text   ")];
    assert_eq!(
        to_srt(&segs),
        "1\n00:00:00,000 --> 00:00:01,000\npadded text\n"
    );
    assert_eq!(to_txt(&segs), "padded text");
    assert_eq!(
        to_json(&segs),
        "[\n  {\n    \"start_ms\": 0,\n    \"end_ms\": 1000,\n    \"text\": \"padded text\"\n  }\n]"
    );

    // Python 的 str.isspace() 也包含 NBSP 与 U+001C..U+001F。
    let exotic = [
        seg(0, 1000, "\u{a0}a b\u{a0}"),
        seg(1000, 2000, "\u{1c}a\u{1c}"),
    ];
    assert_eq!(to_txt(&exotic), "a b\na");
}

#[test]
fn srt_vtt_txt_fold_newlines_json_keeps_them() {
    let segs = [
        seg(0, 1000, "a\n\nb"),
        seg(1000, 2000, "a \r\n b"),
        seg(2000, 3000, "  line1\nline2  "),
    ];
    assert_eq!(to_txt(&segs), "a b\na b\nline1 line2");
    assert_eq!(
        to_srt(&segs),
        "1\n00:00:00,000 --> 00:00:01,000\na b\n\n\
         2\n00:00:01,000 --> 00:00:02,000\na b\n\n\
         3\n00:00:02,000 --> 00:00:03,000\nline1 line2\n"
    );
    assert_eq!(
        to_json(&segs),
        "[\n  {\n    \"start_ms\": 0,\n    \"end_ms\": 1000,\n    \"text\": \"a\\n\\nb\"\n  },\n  \
         {\n    \"start_ms\": 1000,\n    \"end_ms\": 2000,\n    \"text\": \"a \\r\\n b\"\n  },\n  \
         {\n    \"start_ms\": 2000,\n    \"end_ms\": 3000,\n    \"text\": \"line1\\nline2\"\n  }\n]"
    );
}

#[test]
fn leading_and_trailing_newlines_are_dropped_after_folding() {
    let segs = [seg(0, 1000, "\n\nlead and trail\n\n")];
    assert_eq!(
        to_srt(&segs),
        "1\n00:00:00,000 --> 00:00:01,000\nlead and trail\n"
    );
    assert_eq!(
        to_vtt(&segs),
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nlead and trail\n"
    );
    assert_eq!(to_txt(&segs), "lead and trail");
    assert_eq!(
        to_json(&segs),
        "[\n  {\n    \"start_ms\": 0,\n    \"end_ms\": 1000,\n    \"text\": \"lead and trail\"\n  }\n]"
    );
}

#[test]
fn negative_times_clamp_to_zero() {
    let segs = [seg(-500, -100, "neg")];
    assert_eq!(to_srt(&segs), "1\n00:00:00,000 --> 00:00:00,000\nneg\n");
    assert_eq!(
        to_vtt(&segs),
        "WEBVTT\n\n00:00:00.000 --> 00:00:00.000\nneg\n"
    );
    assert_eq!(
        to_json(&segs),
        "[\n  {\n    \"start_ms\": -500,\n    \"end_ms\": -100,\n    \"text\": \"neg\"\n  }\n]"
    );
}

#[test]
fn empty_cue_text_keeps_structure() {
    let segs = [seg(0, 1000, ""), seg(1000, 2000, "   ")];
    assert_eq!(
        to_srt(&segs),
        "1\n00:00:00,000 --> 00:00:01,000\n\n\n2\n00:00:01,000 --> 00:00:02,000\n\n"
    );
    assert_eq!(
        to_vtt(&segs),
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n\n\n00:00:01.000 --> 00:00:02.000\n\n"
    );
    assert_eq!(to_txt(&segs), "\n");
    assert_eq!(
        to_json(&segs),
        "[\n  {\n    \"start_ms\": 0,\n    \"end_ms\": 1000,\n    \"text\": \"\"\n  },\n  \
         {\n    \"start_ms\": 1000,\n    \"end_ms\": 2000,\n    \"text\": \"\"\n  }\n]"
    );
}
