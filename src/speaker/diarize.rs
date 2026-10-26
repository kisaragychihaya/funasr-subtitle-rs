//! 切窗 / 后处理 / 分配。契约见 `super` 模块文档。

use std::collections::HashMap;

use super::{CHUNK_LEN, CHUNK_SHIFT, SMOOTH_MIN_DUR};

/// 采样率（`SAMPLE_RATE`，秒 ↔ 采样换算用）。
const SAMPLE_RATE: f64 = 16_000.0;
/// 重叠判定的容差（`is_overlapped`：`prev.end > cur.start + 1e-4`）。
const OVERLAP_EPS: f64 = 1e-4;

/// 一个 sv_chunk 切窗：秒级时间戳 + 满窗采样（不足已补零）。
pub struct SpkChunk {
    pub start_s: f64,
    pub end_s: f64,
    /// 满窗 `CHUNK_LEN` 采样（i16；sv_chunk 语义补零）。
    pub samples: Vec<i16>,
}

/// 后处理后的说话人时间线（秒，两位小数已在 smooth 内完成）。
#[derive(Debug, Clone, PartialEq)]
pub struct SpkSegment {
    pub start_s: f64,
    pub end_s: f64,
    pub speaker: usize,
}

/// `int(ms / 1000 * 16000)`（毫秒 -> 采样下标，向下取整；`ms <= 0` -> 0）。
fn samples_at_ms(ms: i64) -> usize {
    if ms <= 0 {
        return 0;
    }
    (ms as f64 / 1000.0 * SAMPLE_RATE) as usize
}

/// `utils.sv_chunk` 逐行移植：每 VAD 段独立切窗（窗长 `CHUNK_LEN`、窗移 `CHUNK_SHIFT`，
/// 尾窗前移贴段尾 + 补零）。采样下标用 `floor(ms / 1000 * 16000)`（与 pipeline 一致）。
pub fn sv_chunk(spans_ms: &[(i64, i64)], samples: &[i16]) -> Vec<SpkChunk> {
    let mut chunks: Vec<SpkChunk> = Vec::new();
    for &(st_ms, ed_ms) in spans_ms {
        let seg_st = st_ms as f64 / 1000.0;
        // 对应 `data = wav[int(st_ms/1000*SR) : int(ed_ms/1000*SR)]`
        let a = samples_at_ms(st_ms).min(samples.len());
        let b = samples_at_ms(ed_ms).min(samples.len());
        let data = if b > a { &samples[a..b] } else { &[] };

        let mut last_chunk_ed = 0usize;
        // 对应 `for chunk_st in range(0, len(data), CHUNK_SHIFT)`
        // （循环变量取自 range，函数内对 chunk_st 的重新赋值只影响本轮的取值）
        let mut pos = 0usize;
        while pos < data.len() {
            let chunk_ed = (pos + CHUNK_LEN).min(data.len());
            if chunk_ed <= last_chunk_ed {
                break;
            }
            last_chunk_ed = chunk_ed;
            // 尾窗前移贴段尾：`chunk_st = max(0, chunk_ed - CHUNK_LEN)`
            let chunk_st = chunk_ed.saturating_sub(CHUNK_LEN);
            let mut window = vec![0i16; CHUNK_LEN];
            let taken = &data[chunk_st..chunk_ed];
            window[..taken.len()].copy_from_slice(taken); // `np.pad(..., "constant")`
            chunks.push(SpkChunk {
                start_s: chunk_st as f64 / SAMPLE_RATE + seg_st,
                end_s: chunk_ed as f64 / SAMPLE_RATE + seg_st,
                samples: window,
            });
            pos += CHUNK_SHIFT;
        }
    }
    chunks
}

/// `utils.postprocess` 逐行移植（correct_labels → merge_seque → 重叠对半 →
/// smooth(0.7) + round-half-even）。`labels` 与 `chunks` 等长。
pub fn postprocess(chunks: &[SpkChunk], labels: &[usize]) -> Vec<SpkSegment> {
    assert_eq!(
        chunks.len(),
        labels.len(),
        "postprocess: chunks 与 labels 必须等长（{} vs {}）",
        chunks.len(),
        labels.len()
    );
    let labels = correct_labels(labels);
    let res: Vec<SpkSegment> = chunks
        .iter()
        .zip(&labels)
        .map(|(chunk, &speaker)| SpkSegment {
            start_s: chunk.start_s,
            end_s: chunk.end_s,
            speaker,
        })
        .collect();
    let mut res = merge_seque(res);
    // is_overlapped -> 相邻两段都置中点
    for i in 1..res.len() {
        let prev_end = res[i - 1].end_s;
        let cur_start = res[i].start_s;
        if prev_end > cur_start + OVERLAP_EPS {
            let p = (cur_start + prev_end) / 2.0;
            res[i].start_s = p;
            res[i - 1].end_s = p;
        }
    }
    smooth(res)
}

/// `utils.distribute_spk` 逐行移植：VAD 段（毫秒）按最大总重叠分配说话人；
/// 无重叠 → 0；并列取时间线中先出现者。
pub fn distribute_spk(spans_ms: &[(i64, i64)], timeline: &[SpkSegment]) -> Vec<usize> {
    // `sd_ms = [(st * 1000, ed * 1000, spk) ...]`
    let sd: Vec<(f64, f64, usize)> = timeline
        .iter()
        .map(|seg| (seg.start_s * 1000.0, seg.end_s * 1000.0, seg.speaker))
        .collect();
    let mut out: Vec<usize> = Vec::with_capacity(spans_ms.len());
    for &(s_st, s_ed) in spans_ms {
        let (s_st, s_ed) = (s_st as f64, s_ed as f64);
        // 按时间线顺序累加每人的重叠和（插入顺序 = Python dict 的键序）
        let mut overlaps: Vec<(usize, f64)> = Vec::new();
        for &(st, ed, spk) in &sd {
            let ov = (s_ed.min(ed) - s_st.max(st)).max(0.0);
            if ov > 0.0 {
                match overlaps.iter_mut().find(|(s, _)| *s == spk) {
                    Some(entry) => entry.1 += ov,
                    None => overlaps.push((spk, ov)),
                }
            }
        }
        // 严格大于才更新 -> 并列取先出现者；无重叠保持默认 0
        let mut best = 0usize;
        let mut best_ov = -1.0f64;
        for &(spk, ov) in &overlaps {
            if ov > best_ov {
                best_ov = ov;
                best = spk;
            }
        }
        out.push(best);
    }
    out
}

// ---------------------------------------------------------------------------
// 后处理辅助（对应 utils.correct_labels / merge_seque / smooth）
// ---------------------------------------------------------------------------

/// `utils.correct_labels`：按**首次出现**重编号为 0..k。
fn correct_labels(labels: &[usize]) -> Vec<usize> {
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut out = Vec::with_capacity(labels.len());
    for &label in labels {
        let next = remap.len();
        out.push(*remap.entry(label).or_insert(next));
    }
    out
}

/// `utils.merge_seque`：相邻同说话人且 `cur.start <= prev.end` 时合并（end 取大）。
fn merge_seque(res: Vec<SpkSegment>) -> Vec<SpkSegment> {
    let mut out: Vec<SpkSegment> = Vec::with_capacity(res.len());
    for seg in res {
        match out.last_mut() {
            Some(prev) if prev.speaker == seg.speaker && seg.start_s <= prev.end_s => {
                prev.end_s = seg.end_s;
            }
            _ => out.push(seg),
        }
    }
    out
}

/// `utils.smooth`：时间先 round 两位（half-even），时长 < `SMOOTH_MIN_DUR` 的段继承邻居，
/// 最后再 merge_seque 一次。
fn smooth(res: Vec<SpkSegment>) -> Vec<SpkSegment> {
    if res.len() < 2 {
        return res;
    }
    let mut res = res;
    let n = res.len();
    for i in 0..n {
        res[i].start_s = round2(res[i].start_s);
        res[i].end_s = round2(res[i].end_s);
        if res[i].end_s - res[i].start_s < SMOOTH_MIN_DUR {
            let inherited = if i == 0 {
                res[i + 1].speaker
            } else if i == n - 1 {
                res[i - 1].speaker
            } else if res[i].start_s - res[i - 1].end_s <= res[i + 1].start_s - res[i].end_s {
                // 距离相等取左侧
                res[i - 1].speaker
            } else {
                res[i + 1].speaker
            };
            res[i].speaker = inherited;
        }
    }
    merge_seque(res)
}

/// Python `round(x, 2)`（half-even）：十进制 2 位**正确舍入**。
///
/// `format!("{x:.2}")` 与 Python 的 float repr 一样按二进制精确值做正确舍入、平局取偶，
/// 所以「格式化再解析」即可逐位对齐；不能用 `(x * 100.0).round() / 100.0`
/// （那是 half-away 且有乘除误差）。
fn round2(x: f64) -> f64 {
    format!("{x:.2}")
        .parse::<f64>()
        .expect("两位小数的格式化结果必然可解析")
}
