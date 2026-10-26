//! 切句核心逻辑。移植来源：funasr-subtitle `backend/app/subtitle/segmentation.py`。
//! 显示宽度规则：CJK=1，拉丁/数字/其它非空白=0.5，空白不计。

use crate::types::{Segment, TimedPiece};

use super::normalize::{is_cjk_char, is_py_space, py_trim, py_trim_end, py_trim_start};

// ---------------------------------------------------------------------------
// 常量与字符判定（与 Python 侧常量逐字对应）
// ---------------------------------------------------------------------------

/// `_TERMINALS = "。！？!?…"`：句子终止标点（在其后切句，标点跟随前句）。
fn is_terminal(ch: char) -> bool {
    matches!(ch, '。' | '！' | '？' | '!' | '?' | '…')
}

/// `_SOFT = "，,、；;：:"`：次级停顿标点（仅当一句过长时在此进一步切分）。
fn is_soft(ch: char) -> bool {
    matches!(ch, '，' | ',' | '、' | '；' | ';' | '：' | ':')
}

/// `_STRIP_TERMINALS = frozenset("。！？!?…；;")`（不含 `.`，句点按位置判断）。
fn is_strip_terminal(ch: char) -> bool {
    matches!(ch, '。' | '！' | '？' | '!' | '?' | '…' | '；' | ';')
}

/// 头衔/引用类缩写：几乎不作句末（其后多为人名/延续），恒不切。
const TITLE_ABBREVS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "st", "sr", "jr", "messrs", "mt", "rev", "hon", "capt", "col",
    "gen", "sgt", "lt", "gov", "sen", "vs", "no", "vol", "fig", "dept", "approx", "ave", "blvd",
    "est", "ph",
];

/// 既可句中也常作句末的缩写：仅当其后是小写延续时才不切（大写=新句，照切）。
const TERMINAL_ABBREVS: &[&str] = &["inc", "ltd", "co", "corp", "etc"];

/// `_is_punct_only`：没有实义字符（`_WORD = [\w一-鿿぀-ヿ가-힯]`）的片段视为「纯标点」。
fn is_punct_only(s: &str) -> bool {
    !s.chars().any(is_word_char)
}

/// Python `\w`（Unicode 词字符：字母/数字/下划线）并 CJK 范围。
fn is_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_' || is_cjk_char(ch)
}

/// `_visible_len`：CJK 计 1，拉丁/数字/其它非空白计 0.5（空白不计）。
fn visible_len(s: &str) -> f64 {
    let mut w = 0.0;
    for ch in s.chars() {
        if is_py_space(ch) {
            continue;
        }
        w += if is_cjk_char(ch) { 1.0 } else { 0.5 };
    }
    w
}

/// 复刻 Python `round(x)`（单参数）：IEEE 754 round-half-to-even。
///
/// Python 的 `round` 对浮点做「精确值的 half-to-even」舍入，Rust `f64::round` 是
/// half-away-from-zero，两者在 `.5` 上不同（`round(0.5)=0`、`round(2.5)=2`，
/// 负数 `round(-2.5)=-2`）。`round_ties_even` 与之等价。
fn py_round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// 复刻 `_ABBREV_TAIL = re.compile(r"([A-Za-z][A-Za-z.]*)\.$")` 的 `search` + `group(1)`。
///
/// 回溯到「仅由 `[A-Za-z.]` 组成的后缀」的起点，跳过前导句点后要求是 ASCII 字母；
/// 因 `$` 锚定结尾，group(1) 即开头到末尾句点之间的部分。
fn abbrev_tail_word(s: &str) -> Option<String> {
    if !s.ends_with('.') {
        return None;
    }
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut start = n - 1;
    while start > 0 && is_abbrev_char(chars[start - 1]) {
        start -= 1;
    }
    let mut q = start;
    while q < n && chars[q] == '.' {
        q += 1;
    }
    if q >= n || !chars[q].is_ascii_alphabetic() {
        return None;
    }
    Some(chars[q..n - 1].iter().collect())
}

fn is_abbrev_char(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch == '.'
}

/// `_ends_with_abbrev`：text 以「疑似缩写 + 句点」结尾且不应在此切句时返回 true。
/// 只看末尾 ~40 字符，避免对超长无空格串做 O(n^2) 回溯。
fn ends_with_abbrev(text: &str, next_is_lower: bool) -> bool {
    let chars: Vec<char> = text.chars().collect();
    let from = chars.len().saturating_sub(40);
    let tail_owned: String = chars[from..].iter().collect();
    let tail = py_trim_end(&tail_owned);
    let Some(word) = abbrev_tail_word(tail) else {
        return false;
    };
    if word.contains('.') {
        return true; // 含内部点：U.S / e.g / i.e
    }
    if word.chars().count() == 1 {
        // 单字母：仅当下一词大写(像 J. R. / A. Smith)才不切；小写=真正句末，照切。
        return !next_is_lower;
    }
    let w = word.to_ascii_lowercase();
    if TITLE_ABBREVS.contains(&w.as_str()) {
        return true;
    }
    if TERMINAL_ABBREVS.contains(&w.as_str()) {
        return next_is_lower; // 下一句小写=延续(不切)；大写=新句(切)
    }
    false
}

// ---------------------------------------------------------------------------
// 公共 API
// ---------------------------------------------------------------------------

/// 移植 `strip_terminal_punct`：删句末终止标点（保留逗号/冒号/小数点/缩写内点）。
/// 英文句号仅当其后是空格/结尾时才算句末删除。
pub fn strip_terminal_punct(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(text.len());
    for (i, ch) in chars.iter().enumerate() {
        if is_strip_terminal(*ch) {
            continue;
        }
        if *ch == '.' {
            let after = if i + 1 < n { Some(chars[i + 1]) } else { None };
            if after.is_none_or(is_py_space) {
                continue;
            }
        }
        out.push(*ch);
    }
    out
}

/// 移植 `split_sentences`：切成自然语句；过长的句子在次级标点处贪心切分到 <= max_chars。
/// max_chars=0 表示不限制宽度。
pub fn split_sentences(text: &str, max_chars: usize) -> Vec<String> {
    let text = py_trim(text);
    if text.is_empty() {
        return Vec::new();
    }
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();

    // 1) 在终止标点后切句（保留标点）；英文 '.'/';' 仅当其后空格/结尾才算句末
    let mut sentences: Vec<String> = Vec::new();
    let mut buf = String::new();
    for (i, ch) in chars.iter().enumerate() {
        buf.push(*ch);
        let nxt = chars.get(i + 1).copied();
        let soft = if (*ch == '.' || *ch == ';') && nxt.is_none_or(is_py_space) {
            // 下一个有意义字符（跳过空白）是否小写，用于判断缩写后是延续还是新句
            let mut j = i + 1;
            while j < n && is_py_space(chars[j]) {
                j += 1;
            }
            let next_is_lower = j < n && chars[j].is_lowercase();
            !ends_with_abbrev(&buf, next_is_lower)
        } else {
            false
        };
        if is_terminal(*ch) || soft {
            let s = py_trim(&buf);
            if !s.is_empty() {
                sentences.push(s.to_string());
            }
            buf.clear();
        }
    }
    let tail = py_trim(&buf);
    if !tail.is_empty() {
        sentences.push(tail.to_string());
    }

    // 2) 过长句子在次级标点处再切（尽量靠近 max_chars），仍过长则硬切
    if max_chars > 0 {
        let mut split: Vec<String> = Vec::new();
        for s in &sentences {
            split.extend(split_long(s, max_chars));
        }
        sentences = split;
    }

    // 3) 纯标点的碎片并入上一句，避免「。」单独成行
    let mut merged: Vec<String> = Vec::new();
    for s in sentences {
        if let Some(last) = merged.last_mut()
            && is_punct_only(&s)
        {
            last.push_str(&s);
            continue;
        }
        merged.push(s);
    }
    merged
}

/// 把过长的一句按显示宽度 max_chars 贪心切；断点优先 次级标点 > 空格 > CJK 字。
fn split_long(s: &str, max_chars: usize) -> Vec<String> {
    let limit = max_chars as f64;
    if visible_len(s) <= limit {
        return vec![s.to_string()];
    }
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut pieces: Vec<String> = Vec::new();
    let mut start = 0usize;
    while start < n {
        while start < n && chars[start] == ' ' {
            // 跳过前导空格
            start += 1;
        }
        if start >= n {
            break;
        }
        // 从 start 累积显示宽度到 max_chars，得到窗口末位 end
        let mut end = start;
        let mut w = 0.0;
        while end < n && w < limit {
            if !is_py_space(chars[end]) {
                w += if is_cjk_char(chars[end]) { 1.0 } else { 0.5 };
            }
            end += 1;
        }
        if end >= n {
            let tail = py_trim(&slice_to_string(&chars[start..n])).to_string();
            if !tail.is_empty() {
                pieces.push(tail);
            }
            break;
        }
        // 切点优先级：窗口内最后的 次级标点 > 空格(词边界) > CJK 字边界 > 拉丁扩到下一空格
        let window = &chars[start..end];
        let mut cut = window
            .iter()
            .rposition(|c| is_soft(*c))
            .map(|i| start + i + 1);
        if cut.is_none() {
            cut = window
                .iter()
                .rposition(|c| *c == ' ')
                .map(|i| start + i + 1); // 空格后切，不拆英文单词
        }
        if cut.is_none() && is_cjk_char(chars[end - 1]) {
            cut = Some(end); // CJK 可按字断
        }
        if cut.is_none() {
            // 窗口末位是拉丁、且无标点/空格可断：退回窗口内最后一个 CJK 字处切
            cut = window
                .iter()
                .rposition(|c| is_cjk_char(*c))
                .map(|i| start + i + 1);
        }
        let cut = match cut {
            Some(c) if c > start => c,
            // 拉丁长词无空格：扩到下一个空格，避免拆词
            _ => match chars[end..].iter().position(|c| *c == ' ') {
                Some(i) => end + i + 1,
                None => n,
            },
        };
        let piece = py_trim(&slice_to_string(&chars[start..cut])).to_string();
        if !piece.is_empty() {
            pieces.push(piece);
        }
        start = cut;
    }
    if pieces.is_empty() {
        vec![s.to_string()]
    } else {
        pieces
    }
}

fn slice_to_string(chars: &[char]) -> String {
    chars.iter().collect()
}

/// 移植 `segment_timed`：用真实 token 时间切句（SenseVoice CTC 时间戳路径）。
/// 每句 start=首 token 时间，end=下一句首 token 时间（末句=span_end_ms）。
///
/// `max_cue_ms` 对应 Python 侧从 config 导入的 `MAX_CUE_MS`（单条字幕最长时长）。
pub fn segment_timed(
    pieces: &[TimedPiece],
    span_end_ms: i64,
    max_chars: usize,
    max_cue_ms: i64,
) -> Vec<Segment> {
    let pieces: Vec<(&str, i64)> = pieces
        .iter()
        .filter(|p| !p.text.is_empty())
        .map(|p| (p.text.as_str(), p.time_ms))
        .collect();
    if pieces.is_empty() {
        return Vec::new();
    }

    let mut sentences: Vec<(String, i64)> = Vec::new(); // (text, start_ms)
    let mut buf = String::new();
    let mut buf_start: Option<i64> = None;
    let mut vis = 0.0;
    let limit = max_chars as f64;
    for (idx, (text, ms)) in pieces.iter().enumerate() {
        let (text, ms) = (*text, *ms);
        if buf_start.is_none() && !py_trim(text).is_empty() {
            // 起始以首个非空白 token 的时间为准
            buf_start = Some(ms);
        }
        buf.push_str(text);
        vis += visible_len(text);
        let stripped = py_trim(text);
        let last = stripped.chars().last();
        let nxt = pieces.get(idx + 1).map(|p| p.0);
        let next_starts_space = nxt.is_none_or(|s| s.chars().next().is_some_and(is_py_space));
        // 下一片首个有意义字符（Python 侧 _nxt_sig / nxt_head 为同一表达式）
        let nxt_head = nxt.and_then(|s| py_trim_start(s).chars().next());
        // 句末标点恒断；英文 '.'/';' 仅当其后是空格/结尾时才断（避开小数、缩写）。
        // last 可能为 None（纯空白 piece），必须先判非空再比标点集合。
        let hard = last.is_some_and(is_terminal);
        let soft_period = last.is_some_and(|c| c == '.' || c == ';')
            && next_starts_space
            && !ends_with_abbrev(&buf, nxt_head.is_some_and(|c| c.is_lowercase()));
        // 超过 max_chars 时在「安全边界」断：次级标点 / 词边界(下一片以空格起) / CJK 字。
        // 拉丁词中途（last 是字母、下一片非空格）不断，避免拆词；
        // 若下一片是标点，则不在 CJK 字处断（让标点跟在本行末）。
        let next_is_punct = nxt_head.is_some_and(|c| is_soft(c) || is_terminal(c));
        let boundary = last.is_some_and(is_soft)
            || next_starts_space
            || (last.is_some_and(is_cjk_char) && !next_is_punct);
        let too_long = max_chars > 0 && vis >= limit && boundary;
        // 单条时长超上限也在安全边界断（专业规范 <=7s）
        let over_dur = buf_start.is_some_and(|st| (ms - st) >= max_cue_ms) && boundary;
        if hard || soft_period || too_long || over_dur {
            let s = py_trim(&buf);
            if !s.is_empty() {
                // Python 侧此处可能把 buf_start 的 None 存入列表（随后 int(None) 会报错），
                // 这里取 0 以避免 Rust 侧出现不可能的时间值。
                sentences.push((s.to_string(), buf_start.unwrap_or(0)));
            }
            buf.clear();
            buf_start = None;
            vis = 0.0;
        }
    }
    let trailing = py_trim(&buf);
    if !trailing.is_empty() {
        sentences.push((trailing.to_string(), buf_start.unwrap_or(0)));
    }

    // 纯标点碎片并入上一句（保留上一句的 start），避免「。」单独成行
    let mut merged: Vec<(String, i64)> = Vec::new();
    for (txt, st) in sentences {
        if let Some(last) = merged.last_mut()
            && is_punct_only(&txt)
        {
            last.0.push_str(&txt);
            continue;
        }
        merged.push((txt, st));
    }

    let mut segs: Vec<Segment> = Vec::with_capacity(merged.len());
    for (i, (txt, st)) in merged.iter().enumerate() {
        let mut en = merged.get(i + 1).map_or(span_end_ms, |p| p.1);
        if en <= *st {
            en = st + 1;
        }
        segs.push(Segment {
            start_ms: *st,
            end_ms: en,
            text: txt.clone(),
            speaker: None,
        });
    }
    segs
}

/// 移植 `enforce_monotonic`：保证 cue 单调不重叠（start 不早于上一条 end，end > start）。
pub fn enforce_monotonic(segments: &mut [Segment]) {
    let mut prev_end: Option<i64> = None;
    for s in segments.iter_mut() {
        if let Some(pe) = prev_end
            && s.start_ms < pe
        {
            s.start_ms = pe;
        }
        if s.end_ms <= s.start_ms {
            s.end_ms = s.start_ms + 1;
        }
        prev_end = Some(s.end_ms);
    }
}

/// 移植 `segment_span`：把一个 VAD 语音段的文本切成逐句字幕，按显示宽度比例分配时间。
pub fn segment_span(text: &str, start_ms: i64, end_ms: i64, max_chars: usize) -> Vec<Segment> {
    let pieces = split_sentences(text, max_chars);
    if pieces.is_empty() {
        return Vec::new();
    }
    if pieces.len() == 1 {
        return vec![Segment {
            start_ms,
            end_ms,
            text: pieces[0].clone(),
            speaker: None,
        }];
    }

    let weights: Vec<f64> = pieces.iter().map(|p| visible_len(p).max(1.0)).collect();
    let total: f64 = weights.iter().sum();
    let span = (end_ms - start_ms).max(0) as f64;

    let mut segs: Vec<Segment> = Vec::with_capacity(pieces.len());
    let mut acc = 0.0;
    for (p, w) in pieces.iter().zip(weights.iter()) {
        let s = start_ms + py_round(span * acc / total);
        acc += w;
        let mut e = start_ms + py_round(span * acc / total);
        if e <= s {
            e = s + 1;
        }
        segs.push(Segment {
            start_ms: s,
            end_ms: e,
            text: p.clone(),
            speaker: None,
        });
    }
    // 末段对齐到 end_ms
    if let Some(last) = segs.last_mut() {
        last.end_ms = (last.start_ms + 1).max(end_ms);
    }
    segs
}
