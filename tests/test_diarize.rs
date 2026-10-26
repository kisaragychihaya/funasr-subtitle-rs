//! W4 验收测试：说话人分离的切窗 / 后处理 / 分配（`src/speaker/diarize.rs`）。
//!
//! 两类用例：
//!
//! 1) **golden 对拍**（不需要模型，随 `cargo test` 常跑）：读
//!    `tests/golden/diar/diar_two_speakers/`（`reference/dump_golden_diar.py` 生成）：
//!    * `sv_chunk` × `chunks.json` + `input.s16bin`：窗数、每窗采样数、起止时间；
//!    * `postprocess` × `cluster.json` 的 labels → `sv_output.{auto,oracle2}.json`（**全等**，
//!      时间已在 smooth 内 round 两位）；
//!    * `distribute_spk` × `span_labels.json` → `span_speakers`。
//! 2) **手写小用例**：merge_seque 相邻合并、重叠对半、round-half-even（Python `round(x, 2)`
//!    语义）、`< 0.7s` 段继承邻居、并列说话人取时间线中先出现者。

use std::fs;
use std::path::{Path, PathBuf};

use funasr_subtitle_rs::speaker::CHUNK_LEN;
use funasr_subtitle_rs::speaker::diarize::{
    SpkChunk, SpkSegment, distribute_spk, postprocess, sv_chunk,
};
use serde_json::Value;

const GOLDEN: &str = "tests/golden/diar/diar_two_speakers";

// ---------------------------------------------------------------------------
// 通用小工具
// ---------------------------------------------------------------------------

fn manifest_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn read_json(rel: &str) -> Value {
    let path = manifest_path(rel);
    let raw =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("读取 {} 失败: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("解析 {} 失败: {e}", path.display()))
}

/// 读 16k 单声道 s16le 原始采样（golden 的 `input.s16bin`）。
fn read_s16(rel: &str) -> Vec<i16> {
    let path = manifest_path(rel);
    let raw = fs::read(&path).unwrap_or_else(|e| panic!("读取 {} 失败: {e}", path.display()));
    raw.as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_le_bytes(*c))
        .collect()
}

/// `[[start_ms, end_ms], ...]`。
fn spans_of(v: &Value) -> Vec<(i64, i64)> {
    v.as_array()
        .unwrap_or_else(|| panic!("期望数组: {v}"))
        .iter()
        .map(|pair| {
            assert!(pair.is_array(), "期望二元组: {pair}");
            (int_at(pair, 0), int_at(pair, 1))
        })
        .collect()
}

fn int_at(v: &Value, i: usize) -> i64 {
    v.get(i)
        .and_then(Value::as_i64)
        .unwrap_or_else(|| panic!("第 {i} 项不是整数: {v}"))
}

fn f64_at(v: &Value, i: usize) -> f64 {
    v.get(i)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("第 {i} 项不是数字: {v}"))
}

/// `[[start_s, end_s], ...]`。
fn chunk_times(v: &Value) -> Vec<(f64, f64)> {
    v.as_array()
        .unwrap_or_else(|| panic!("期望数组: {v}"))
        .iter()
        .map(|pair| (f64_at(pair, 0), f64_at(pair, 1)))
        .collect()
}

/// `[[start_s, end_s, spk], ...]` -> 说话人时间线。
fn segments_of(v: &Value) -> Vec<SpkSegment> {
    v.as_array()
        .unwrap_or_else(|| panic!("期望数组: {v}"))
        .iter()
        .map(|triple| SpkSegment {
            start_s: f64_at(triple, 0),
            end_s: f64_at(triple, 1),
            speaker: f64_at(triple, 2) as usize,
        })
        .collect()
}

fn labels_of(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap_or_else(|| panic!("期望数组: {v}"))
        .iter()
        .map(|x| x.as_u64().unwrap_or_else(|| panic!("标签不是整数: {x}")) as usize)
        .collect()
}

/// 用 golden 的窗时间重建 `SpkChunk`：postprocess 只用时间，采样填空满窗。
fn chunks_at(times: &[(f64, f64)]) -> Vec<SpkChunk> {
    times
        .iter()
        .map(|&(start_s, end_s)| SpkChunk {
            start_s,
            end_s,
            samples: vec![0i16; CHUNK_LEN],
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 1) golden 对拍
// ---------------------------------------------------------------------------

#[test]
fn sv_chunk_matches_golden_windows() {
    let golden = read_json(&format!("{GOLDEN}/chunks.json"));
    let spans = spans_of(&golden["vad_spans"]);
    let want = chunk_times(&golden["chunks"]);
    let samples = read_s16(&format!("{GOLDEN}/input.s16bin"));

    assert_eq!(golden["chunk_len_samples"].as_u64(), Some(CHUNK_LEN as u64));
    assert_eq!(
        golden["chunk_shift_samples"].as_u64(),
        Some(funasr_subtitle_rs::speaker::CHUNK_SHIFT as u64)
    );

    let got = sv_chunk(&spans, &samples);
    println!(
        "{} 个 VAD 段 -> {} 个窗（golden {} 个）",
        spans.len(),
        got.len(),
        want.len()
    );
    assert_eq!(got.len(), 70, "golden 共 70 个窗");
    assert_eq!(got.len(), want.len(), "窗数与 golden 不一致");

    let mut diffs = Vec::new();
    for (i, (chunk, &(want_start, want_end))) in got.iter().zip(&want).enumerate() {
        if chunk.samples.len() != CHUNK_LEN {
            diffs.push(format!(
                "第 {i} 窗采样数 {} != {CHUNK_LEN}",
                chunk.samples.len()
            ));
        }
        if (chunk.start_s - want_start).abs() > 1e-6 || (chunk.end_s - want_end).abs() > 1e-6 {
            diffs.push(format!(
                "第 {i} 窗 [{}, {}] != golden [{want_start}, {want_end}]",
                chunk.start_s, chunk.end_s
            ));
        }
    }
    assert!(
        diffs.is_empty(),
        "sv_chunk 与 golden 不一致:\n{}",
        diffs.join("\n")
    );
}

#[test]
fn postprocess_matches_golden_timelines() {
    let chunks = chunks_at(&chunk_times(
        &read_json(&format!("{GOLDEN}/chunks.json"))["chunks"],
    ));
    let cluster = read_json(&format!("{GOLDEN}/cluster.json"));

    for (labels_key, out_file) in [
        ("labels_auto", "sv_output.auto.json"),
        ("labels_oracle2", "sv_output.oracle2.json"),
    ] {
        let labels = labels_of(&cluster[labels_key]);
        assert_eq!(labels.len(), chunks.len(), "{labels_key} 与窗数不等");
        let want = segments_of(&read_json(&format!("{GOLDEN}/{out_file}")));
        let got = postprocess(&chunks, &labels);
        assert_eq!(
            got, want,
            "postprocess（{labels_key}）与 golden {out_file} 不一致"
        );
        println!("{labels_key}: {} 段说话人时间线（golden 一致）", got.len());
    }
}

#[test]
fn distribute_spk_matches_golden_span_labels() {
    let golden = read_json(&format!("{GOLDEN}/span_labels.json"));
    let spans = spans_of(&golden["vad_spans"]);
    let timeline = segments_of(&golden["sv_output"]);
    let want = labels_of(&golden["span_speakers"]);

    let got = distribute_spk(&spans, &timeline);
    assert_eq!(got, want, "distribute_spk 与 golden span_speakers 不一致");
    println!("{} 个 VAD 段 -> {got:?}", spans.len());
}

// ---------------------------------------------------------------------------
// 2) 手写小用例：切窗
// ---------------------------------------------------------------------------

#[test]
fn sv_chunk_windows_follow_shift_and_tail_shift() {
    // 段 0..2.5s = 40000 采样：窗 0（0~1.5s）、窗 12000（0.75~2.25s）、
    // 窗 24000（尾部 36000 贴到 40000 -> 1.0~2.5s），之后 chunk_ed <= last 终止。
    let samples: Vec<i16> = (0..40_000).map(|i| (i % 32_767) as i16).collect();
    let chunks = sv_chunk(&[(0, 2500)], &samples);
    let times: Vec<(f64, f64)> = chunks.iter().map(|c| (c.start_s, c.end_s)).collect();
    assert_eq!(times, vec![(0.0, 1.5), (0.75, 2.25), (1.0, 2.5)]);

    // 采样内容就是对应下标区间（尾部窗已前移，不是补零）
    for chunk in &chunks {
        assert_eq!(chunk.samples.len(), CHUNK_LEN);
        let st = (chunk.start_s * 16_000.0) as usize;
        let ed = (chunk.end_s * 16_000.0) as usize;
        assert_eq!(chunk.samples[0], (st % 32_767) as i16);
        assert_eq!(chunk.samples[CHUNK_LEN - 1], ((ed - 1) % 32_767) as i16);
    }
}

#[test]
fn sv_chunk_pads_short_span_with_zeros() {
    // 100ms 的极短段：只有一窗，尾部补零到满窗
    let samples = vec![123i16; 1600];
    let chunks = sv_chunk(&[(0, 100)], &samples);
    assert_eq!(chunks.len(), 1);
    assert_eq!((chunks[0].start_s, chunks[0].end_s), (0.0, 0.1));
    assert_eq!(chunks[0].samples.len(), CHUNK_LEN);
    assert!(chunks[0].samples[..1600].iter().all(|&s| s == 123));
    assert!(chunks[0].samples[1600..].iter().all(|&s| s == 0));
}

#[test]
fn sv_chunk_keeps_segment_start_offset_and_handles_empty() {
    // 每段独立切窗，时间戳加段起点秒数
    let samples = vec![0i16; 24_000];
    let chunks = sv_chunk(&[(0, 400), (1000, 1400)], &samples);
    let times: Vec<(f64, f64)> = chunks.iter().map(|c| (c.start_s, c.end_s)).collect();
    assert_eq!(times, vec![(0.0, 0.4), (1.0, 1.4)]);
    // 空段（ed <= st）与空 spans 都不产出窗，也不 panic
    assert!(sv_chunk(&[(500, 500)], &samples).is_empty());
    assert!(sv_chunk(&[], &samples).is_empty());
    // 段尾越过实际采样长度时按实际长度截断（`min(len)`）
    let beyond = sv_chunk(&[(0, 10_000)], &samples);
    assert_eq!(beyond.len(), 1);
    assert_eq!((beyond[0].start_s, beyond[0].end_s), (0.0, 1.5));
}

// ---------------------------------------------------------------------------
// 3) 手写小用例：merge_seque / 重叠对半 / round-half-even / 继承
// ---------------------------------------------------------------------------

#[test]
fn merge_seque_joins_adjacent_same_speaker() {
    // 相邻同人且 cur.start <= prev.end -> 合并（end 取大）；不同人/有空隙则分开。
    // 每段都 >= 0.7s，避免 smooth 的继承分支干扰。
    let chunks = chunks_at(&[(0.0, 1.0), (1.0, 2.0), (3.0, 4.0), (4.0, 5.0)]);
    let got = postprocess(&chunks, &[0, 0, 1, 0]);
    assert_eq!(
        got,
        vec![
            SpkSegment {
                start_s: 0.0,
                end_s: 2.0,
                speaker: 0
            },
            SpkSegment {
                start_s: 3.0,
                end_s: 4.0,
                speaker: 1
            },
            SpkSegment {
                start_s: 4.0,
                end_s: 5.0,
                speaker: 0
            },
        ]
    );
}

#[test]
fn overlapping_pair_is_split_at_midpoint() {
    // [0,5,A][3,8,B] -> 两者都置中点 4 -> [0,4,A][4,8,B]
    let chunks = chunks_at(&[(0.0, 5.0), (3.0, 8.0)]);
    let got = postprocess(&chunks, &[0, 1]);
    assert_eq!(
        got,
        vec![
            SpkSegment {
                start_s: 0.0,
                end_s: 4.0,
                speaker: 0
            },
            SpkSegment {
                start_s: 4.0,
                end_s: 8.0,
                speaker: 1
            },
        ]
    );
}

#[test]
fn round_is_python_half_even() {
    // Python round(x, 2) = half-even（不是 Rust f64::round 的 half-away）：
    //   0.125 -> 0.12（2 偶），0.375 -> 0.38（8 偶）
    // 段时长都 >= 0.7s，排除 smooth 的继承分支，只观察舍入结果。
    let down = postprocess(&chunks_at(&[(0.125, 0.875), (1.0, 2.0)]), &[0, 1]);
    assert_eq!(down[0].start_s, 0.12, "0.125 应 half-even 到 0.12");
    assert_eq!(down[0].end_s, 0.88, "0.875 应 half-even 到 0.88");

    let up = postprocess(&chunks_at(&[(0.0, 0.375), (0.5, 2.0)]), &[0, 1]);
    assert_eq!(up[0].end_s, 0.38, "0.375 应 half-even 到 0.38");
}

#[test]
fn short_segment_inherits_neighbour_speaker() {
    // (1) 首段 < 0.7s -> 继承下一段
    let head = postprocess(&chunks_at(&[(0.0, 0.375), (0.5, 2.0)]), &[0, 1]);
    assert!(
        head.iter().all(|s| s.speaker == 1),
        "首段应继承下一段的说话人: {head:?}"
    );
    assert_eq!((head[0].start_s, head[0].end_s), (0.0, 0.38));

    // (2) 末段 < 0.7s -> 继承上一段
    let tail = postprocess(&chunks_at(&[(0.0, 1.0), (1.0, 1.2)]), &[0, 1]);
    assert_eq!(
        tail,
        vec![SpkSegment {
            start_s: 0.0,
            end_s: 1.2,
            speaker: 0
        }],
        "末段应继承上一段后合并"
    );

    // (3) 中间段 < 0.7s、左右距离相等 -> 取左侧，随后因首尾相接被合并
    let middle = postprocess(
        &chunks_at(&[(0.0, 1.0), (1.0, 1.5), (1.5, 2.5)]),
        &[0, 1, 0],
    );
    assert_eq!(
        middle,
        vec![SpkSegment {
            start_s: 0.0,
            end_s: 2.5,
            speaker: 0
        }]
    );

    // (4) 中间段距右侧更近 -> 取右侧
    let middle_right = postprocess(
        &chunks_at(&[(0.0, 1.0), (1.6, 1.8), (1.9, 2.9)]),
        &[0, 1, 2],
    );
    assert_eq!(middle_right[1].speaker, 2, "{middle_right:?}");
}

#[test]
fn correct_labels_renumbers_by_first_appearance() {
    // 原始标签 [5,5,3,5] -> [0,0,1,0]（先出现的 5 变 0）
    let chunks = chunks_at(&[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]);
    let got = postprocess(&chunks, &[5, 5, 3, 5]);
    assert_eq!(
        got,
        vec![
            SpkSegment {
                start_s: 0.0,
                end_s: 2.0,
                speaker: 0
            },
            SpkSegment {
                start_s: 2.0,
                end_s: 3.0,
                speaker: 1
            },
            SpkSegment {
                start_s: 3.0,
                end_s: 4.0,
                speaker: 0
            },
        ]
    );
}

#[test]
#[should_panic(expected = "必须等长")]
fn postprocess_rejects_length_mismatch() {
    let chunks = chunks_at(&[(0.0, 1.0)]);
    let _ = postprocess(&chunks, &[0, 1]);
}

// ---------------------------------------------------------------------------
// 4) 手写小用例：distribute_spk
// ---------------------------------------------------------------------------

#[test]
fn distribute_spk_picks_max_overlap_and_ties_by_first_in_timeline() {
    let timeline = vec![
        SpkSegment {
            start_s: 0.0,
            end_s: 0.5,
            speaker: 1,
        },
        SpkSegment {
            start_s: 0.5,
            end_s: 1.0,
            speaker: 0,
        },
    ];
    // 各占一半 -> 并列，取时间线中先出现者（说话人 1）
    assert_eq!(distribute_spk(&[(0, 1000)], &timeline), vec![1]);
    // 左半多 -> 说话人 1；右半多 -> 说话人 0
    assert_eq!(distribute_spk(&[(0, 600)], &timeline), vec![1]);
    assert_eq!(distribute_spk(&[(400, 1000)], &timeline), vec![0]);
    // 完全无重叠 -> 0
    assert_eq!(distribute_spk(&[(2000, 3000)], &timeline), vec![0]);
    // 空时间线 -> 全 0
    assert_eq!(distribute_spk(&[(0, 1000)], &[]), vec![0]);
}
