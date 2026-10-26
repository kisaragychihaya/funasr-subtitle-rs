//! T9：CT-Transformer 标点。移植来源：funasr-subtitle `engine/funasr_engine.py` 的
//! _punctuate/_punc_once/_apply_punct_to_timed 与 vendored `punc_bin.py`。
//!
//! 移植范围与对应关系（逐行对照，不做「修正」）：
//! - `CtPunc::call`   ← `CT_Transformer.__call__`（code_mix_split_words → tokens2ids →
//!   split_to_mini_sentence(20) → 逐片推理 argmax → 找句末/逗号 cache 边界 → 拼回文本 → 末尾补句号）
//! - `Punctuator::punctuate` ← `FunasrEngine._punctuate`（≤140 字单次；否则按 110 字分块、
//!   块内在半角空格处回退、逐块标点后用单个空格 join）
//! - `align_punct_to_timed` ← `_apply_punct_to_timed`（大小写不敏感匹配、8 字符有界重同步）
//!
//! 上游怪癖全部保留：给已含句号的句子再补逗号（`今天天气很好。` → `今天天气很好，。`）、
//! 英文用全角句号（ct-punc 训练数据决定）、分块处插入 `" "` 把一句话切断。
//! 与 Python 的唯一差异：模型目录若带 `jieba_usr_dict`（jieba 分词路径）时退化为
//! `code_mix_split_words`；注册表里的 ct-punc 模型没有该文件，故对 golden 零偏差。
//!
//! 错误语义：Python `_punctuate` 用 `except` 把任何异常兜成「返回原文」；Rust 版把
//! 模型加载/推理错误按 `Result` 上抛（由上层决定容错），只有「输入没有可标点的词」
//! （Python 侧 IndexError 后兜成原文）等价地返回原文/空串。

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;

use crate::error::{Error, Result};
use crate::text::normalize::{is_py_space, py_trim};
use crate::types::TimedPiece;

/// ct-punc 对超长输入只在结尾加标点，故按此长度分块（`config.py` 的 `PUNC_CHUNK` 默认值）。
pub const PUNC_CHUNK: usize = 110;
/// `len(text) <= PUNC_CHUNK + 30` 走单次调用（`_punctuate` 的阈值）。
const PUNC_CHUNK_SLACK: usize = 30;
/// `CT_Transformer.__call__` 的 `split_size`：每次推理最多多少个「词」。
const SPLIT_SIZE: usize = 20;
/// 单片词数超过它才允许在逗号处强制截断（`cache_pop_trigger_limit`）。
const CACHE_POP_TRIGGER_LIMIT: usize = 200;
/// 对齐失配时的有界前向重同步窗口（`_apply_punct_to_timed`）。
const RESYNC_WINDOW: usize = 8;
/// 量化模型文件名（funasr 侧固定 `quantize=True`）。
const MODEL_FILE: &str = "model_quant.onnx";

/// ct-punc 标点器（量化 ONNX）。punctuate 内含 110 字分块逻辑（PUNC_CHUNK）。
pub struct Punctuator {
    ct: CtPunc,
}

impl Punctuator {
    /// 从模型目录加载（`model_quant.onnx` + `config.yaml` + `tokens.json`）。
    pub fn new(model_dir: &Path, threads: usize) -> Result<Self> {
        Ok(Self {
            ct: CtPunc::load(model_dir, threads)?,
        })
    }

    /// 给文本补标点；失败时应返回原文（标点不致命），由调用方决定是否容错。
    pub fn punctuate(&mut self, text: &str) -> Result<String> {
        let plan = plan_chunks(text);
        if !plan.chunked {
            return self.punc_once(text);
        }
        let mut parts = Vec::with_capacity(plan.chunks.len());
        for chunk in &plan.chunks {
            parts.push(self.punc_once(chunk)?);
        }
        Ok(parts.join(" "))
    }

    /// `_punc_once`：一次 `CT_Transformer.__call__`，取返回的第一个元素。
    ///
    /// 空/纯空白输入在 Python 侧会在 `__call__` 里抛 IndexError，被 `_punctuate` 的
    /// `except` 兜成原文；这里等价地返回原文。
    fn punc_once(&mut self, text: &str) -> Result<String> {
        if code_mix_split_words(text).is_empty() {
            return Ok(text.to_string());
        }
        self.ct.call(text)
    }
}

/// 一次 `_punctuate` 调用要跑的块计划。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkPlan {
    /// 是否走分块路径（`len(text) > PUNC_CHUNK + 30`）。
    pub chunked: bool,
    /// 逐块要标点的文本（未分块时就是原文本身）。
    pub chunks: Vec<String>,
}

/// 移植 `FunasrEngine._punctuate` 的分块逻辑：短文本单次调用；长文本按 110 字切块，
/// 切点先回退到块内最后一个半角空格，块 `strip()` 后逐块标点，块间用单个空格 join。
pub fn plan_chunks(text: &str) -> ChunkPlan {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    if n <= PUNC_CHUNK + PUNC_CHUNK_SLACK {
        return ChunkPlan {
            chunked: false,
            chunks: vec![text.to_string()],
        };
    }
    let mut chunks = Vec::new();
    let mut i = 0usize;
    while i < n {
        let mut end = (i + PUNC_CHUNK).min(n);
        if end < n {
            // Python `text.rfind(" ", i, end)`：只认半角空格，且切点必须严格在 i 之后
            if let Some(sp) = (i + 1..end).rev().find(|&k| chars[k] == ' ') {
                end = sp;
            }
        }
        let chunk: String = chars[i..end].iter().collect();
        let chunk = py_trim(&chunk);
        if !chunk.is_empty() {
            chunks.push(chunk.to_string());
        }
        i = end;
    }
    ChunkPlan {
        chunked: true,
        chunks,
    }
}

/// 移植 `code_mix_split_words`：按空白切段，段内连续 ASCII 收成一个「词」、非 ASCII 字符
/// 各自成词。返回的每个词就是 ct-punc 的一个输入 token。
pub fn code_mix_split_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    for seg in text.split(is_py_space).filter(|s| !s.is_empty()) {
        let mut current = String::new();
        for c in seg.chars() {
            if c.is_ascii() {
                current.push(c);
            } else {
                if !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                }
                words.push(c.to_string());
            }
        }
        if !current.is_empty() {
            words.push(current);
        }
    }
    words
}

/// 移植 `split_to_mini_sentence`：每 `word_limit` 个词一组（末组为余数）。
fn split_to_mini_sentence<T: Clone>(words: &[T], word_limit: usize) -> Vec<Vec<T>> {
    if words.len() <= word_limit {
        return vec![words.to_vec()];
    }
    words.chunks(word_limit).map(<[T]>::to_vec).collect()
}

/// 移植 `_apply_punct_to_timed`：把补标点后的文本对齐回带时间的 token，
/// 保留真实 CTC 时间；失配时有界向前重同步（窗口 8），新插入标点用上一字符时间。
pub fn align_punct_to_timed(timed: &[TimedPiece], punct_text: &str) -> Vec<TimedPiece> {
    // 原始逐字 + 时间
    let ct: Vec<(char, i64)> = timed
        .iter()
        .flat_map(|piece| piece.text.chars().map(move |c| (c, piece.time_ms)))
        .collect();
    let mut out: Vec<TimedPiece> = Vec::with_capacity(punct_text.chars().count());
    let mut p = 0usize;
    let mut last_t = timed.first().map_or(0, |t| t.time_ms);
    for c in punct_text.chars() {
        if is_py_space(c) {
            out.push(TimedPiece {
                text: c.to_string(),
                time_ms: last_t,
            });
            continue;
        }
        while p < ct.len() && is_py_space(ct[p].0) {
            p += 1;
        }
        // 大小写不敏感匹配：ct-punc 兜底可能改大小写，严格比对会让指针卡住、
        // 其后所有 token 时间塌到 last_t。
        if p < ct.len() && char_eq_ignore_case(ct[p].0, c) {
            last_t = ct[p].1;
            out.push(TimedPiece {
                text: c.to_string(),
                time_ms: last_t,
            });
            p += 1;
            continue;
        }
        // 失配时在 p..p+8 内找第一个非空白且大小写不敏感相等的字符（有界重同步）
        let found = ct[p..(p + RESYNC_WINDOW).min(ct.len())]
            .iter()
            .position(|item| !is_py_space(item.0) && char_eq_ignore_case(item.0, c));
        if let Some(offset) = found {
            last_t = ct[p + offset].1;
            p += offset + 1;
        }
        // 没找到就是新插入的标点：沿用上一个字符的时间
        out.push(TimedPiece {
            text: c.to_string(),
            time_ms: last_t,
        });
    }
    out
}

/// Python `a.lower() == b.lower()`（单个字符；`to_lowercase` 可能展开成多个字符）。
fn char_eq_ignore_case(a: char, b: char) -> bool {
    a == b || a.to_lowercase().eq(b.to_lowercase())
}

/// 进程级 ONNX Runtime 环境（ort 每进程只允许一个 Environment）。
fn ort_environment() -> Result<Environment> {
    static ENV: OnceLock<Environment> = OnceLock::new();
    if let Some(env) = ENV.get() {
        return Ok(env.clone());
    }
    let env = ort::init()
        .with_name("funasr-subtitle-rs")
        .build()
        .map_err(|e| Error::Model(format!("初始化 ONNX Runtime 失败: {e}")))?;
    // 并发首建时以先到者为准（第二个 Environment 建不出来，丢弃即可）
    let _ = ENV.set(env);
    ENV.get()
        .cloned()
        .ok_or_else(|| Error::Model("初始化 ONNX Runtime 失败".to_string()))
}

/// CT-Transformer 本体（`punc_bin.py` 的 `CT_Transformer`）。
struct CtPunc {
    session: Session,
    token2id: HashMap<String, i64>,
    unk_id: i64,
    punc_list: Vec<String>,
    period: usize,
}

impl CtPunc {
    fn load(model_dir: &Path, threads: usize) -> Result<Self> {
        let model_file = model_dir.join(MODEL_FILE);
        if !model_file.is_file() {
            return Err(Error::Model(format!(
                "标点模型文件不存在: {}",
                model_file.display()
            )));
        }
        let tokens = read_tokens(&model_dir.join("tokens.json"))?;
        let (punc_list, period) = read_punc_list(&model_dir.join("config.yaml"))?;

        let env = ort_environment()?;
        let builder = Session::builder(&env)
            .map_err(|e| Error::Model(format!("创建 ONNX Session 失败: {e}")))?;
        let mut builder = builder
            .with_intra_threads(threads)
            .map_err(|e| Error::Model(format!("设置 intra_op 线程数失败: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| Error::Model(format!("设置图优化级别失败: {e}")))?
            // 对齐 funasr 的 log_severity_level=4（只保留 FATAL）
            .with_log_level(ort::logging::LogLevel::Fatal)
            .map_err(|e| Error::Model(format!("设置日志级别失败: {e}")))?;
        let session = builder
            .commit_from_file(&model_file)
            .map_err(|e| Error::Model(format!("加载标点模型 {} 失败: {e}", model_file.display())))?;

        // 词表最后一项是 `<unk>`（`TokenIDConverter` 的 unk_symbol），映射要按最终 map 取，
        // 保证和 Python 的 dict 语义一致（重复项取最后一个下标）
        let unk_symbol = tokens
            .last()
            .cloned()
            .ok_or_else(|| Error::Model("tokens.json 为空".to_string()))?;
        let token2id: HashMap<String, i64> = tokens
            .into_iter()
            .enumerate()
            .map(|(idx, tok)| (tok, idx as i64))
            .collect();
        let unk_id = token2id.get(&unk_symbol).copied().ok_or_else(|| {
            Error::Model(format!("tokens.json 里找不到 unk 符号 {unk_symbol}"))
        })?;

        Ok(Self {
            session,
            token2id,
            unk_id,
            punc_list,
            period,
        })
    }

    /// `CT_Transformer.__call__`：返回补好标点的文本（Python 侧第二个返回值 punc_ids 不需要）。
    fn call(&mut self, text: &str) -> Result<String> {
        let split_text = code_mix_split_words(text);
        if split_text.is_empty() {
            return Ok(String::new());
        }
        let split_ids = self.tokens2ids(&split_text);
        let mini_sentences = split_to_mini_sentence(&split_text, SPLIT_SIZE);
        let mini_ids = split_to_mini_sentence(&split_ids, SPLIT_SIZE);

        let mut cache_sent: Vec<String> = Vec::new();
        let mut cache_sent_id: Vec<i32> = Vec::new();
        let mut new_mini_sentence = String::new();
        let last = mini_sentences.len() - 1;

        for (idx, (sentence, ids)) in mini_sentences.iter().zip(mini_ids.iter()).enumerate() {
            let mut mini_sentence = cache_sent.clone();
            mini_sentence.extend(sentence.iter().cloned());
            let mut mini_sentence_id = cache_sent_id.clone();
            mini_sentence_id.extend(ids.iter().copied());

            let mut punctuations = self.infer_argmax(&mini_sentence_id)?;
            if punctuations.len() != mini_sentence.len() {
                return Err(Error::Inference(format!(
                    "ct-punc 输出长度 {} 与输入词数 {} 不一致",
                    punctuations.len(),
                    mini_sentence.len()
                )));
            }

            // 找最后一个句末标点，把它之前的都吐出去，剩下的留作下一次推理的 cache
            if idx < last {
                let mut sentence_end: i64 = -1;
                let mut last_comma: i64 = -1;
                let mut i = punctuations.len() as i64 - 2;
                while i > 1 {
                    let p = self.punc_at(punctuations[i as usize]);
                    if p == "。" || p == "？" {
                        sentence_end = i;
                        break;
                    }
                    if last_comma < 0 && p == "，" {
                        last_comma = i;
                    }
                    i -= 1;
                }
                if sentence_end < 0
                    && mini_sentence.len() > CACHE_POP_TRIGGER_LIMIT
                    && last_comma >= 0
                {
                    // 句子太长：在逗号处强制截断
                    sentence_end = last_comma;
                    punctuations[sentence_end as usize] = self.period;
                }
                let cut = (sentence_end + 1) as usize;
                cache_sent = mini_sentence[cut..].to_vec();
                cache_sent_id = mini_sentence_id[cut..].to_vec();
                mini_sentence.truncate(cut);
                punctuations.truncate(cut);
            }

            for i in 0..mini_sentence.len() {
                // 两个「词」都以 ASCII 开头时补一个空格（英文词间空格靠这里还原）
                if i > 0
                    && first_char_is_ascii(&mini_sentence[i])
                    && first_char_is_ascii(&mini_sentence[i - 1])
                {
                    mini_sentence[i].insert(0, ' ');
                }
                new_mini_sentence.push_str(&mini_sentence[i]);
                let p = self.punc_at(punctuations[i]);
                if p != "_" {
                    new_mini_sentence.push_str(p);
                }
            }
        }

        // 句末补句号：末尾是逗号/顿号则替换，否则在缺句末标点时追加
        let mut out = new_mini_sentence.clone();
        if let Some(tail) = new_mini_sentence.chars().next_back() {
            if tail == '，' || tail == '、' {
                out.truncate(new_mini_sentence.len() - tail.len_utf8());
                out.push('。');
            } else if tail != '。' && tail != '？' {
                out.push('。');
            }
        }
        Ok(out)
    }

    /// `TokenIDConverter.tokens2ids`：词表里没有的一律映射到 `<unk>`（最后一个 token）。
    fn tokens2ids(&self, tokens: &[String]) -> Vec<i32> {
        tokens
            .iter()
            .map(|t| self.token2id.get(t).copied().unwrap_or(self.unk_id) as i32)
            .collect()
    }

    /// `punc_list[id]`；越界（不该发生）返回空串，等价于「不加标点」。
    fn punc_at(&self, id: usize) -> &str {
        self.punc_list.get(id).map_or("", String::as_str)
    }

    /// 一次推理：输入 `[1, seq]` 的 int32 token id，输出 `[1, T, num_punc]` 的 logits，
    /// 逐帧 argmax 得到标点 id（`np.argmax` 取第一个最大值）。
    fn infer_argmax(&mut self, ids: &[i32]) -> Result<Vec<usize>> {
        let seq_len = ids.len();
        if seq_len == 0 {
            return Err(Error::Inference("ct-punc 输入为空".to_string()));
        }
        let text = Value::from_array((vec![1_i64, seq_len as i64], ids.to_vec()))
            .map_err(|e| Error::Inference(format!("构造 token 输入张量失败: {e}")))?;
        let text_lengths = Value::from_array((vec![1_i64], vec![seq_len as i32]))
            .map_err(|e| Error::Inference(format!("构造 text_lengths 张量失败: {e}")))?;
        let outputs = self
            .session
            .run(vec![("inputs", text), ("text_lengths", text_lengths)])
            .map_err(|e| Error::Inference(format!("ct-punc 推理失败: {e}")))?;

        let (shape, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Inference(format!("读取 ct-punc 输出失败: {e}")))?;
        let dims: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
        if dims.len() != 3 || dims[0] == 0 || dims[2] == 0 {
            return Err(Error::Inference(format!(
                "ct-punc 输出维度异常: {dims:?}"
            )));
        }
        let (frames, classes) = (dims[1], dims[2]);
        if logits.len() < frames * classes {
            return Err(Error::Inference(format!(
                "ct-punc 输出大小异常: {} < {}",
                logits.len(),
                frames * classes
            )));
        }
        let mut out = Vec::with_capacity(frames);
        for frame in 0..frames {
            let row = &logits[frame * classes..(frame + 1) * classes];
            let mut best = 0usize;
            for (cls, value) in row.iter().enumerate().skip(1) {
                if *value > row[best] {
                    best = cls;
                }
            }
            out.push(best);
        }
        Ok(out)
    }
}

/// 词的首字符是否 ASCII（Python `len(word[0].encode()) == 1`）。
fn first_char_is_ascii(word: &str) -> bool {
    word.chars().next().is_some_and(|c| c.is_ascii())
}

/// 读 `tokens.json`（模型词表，一行 JSON 字符串数组）。
fn read_tokens(path: &Path) -> Result<Vec<String>> {
    let raw = fs::read_to_string(path)
        .map_err(|e| Error::Model(format!("读取 {} 失败: {e}", path.display())))?;
    serde_json::from_str(&raw)
        .map_err(|e| Error::Model(format!("解析 {} 失败: {e}", path.display())))
}

/// 读 `config.yaml` 的 `model_conf.punc_list`，并按 `CT_Transformer.__init__` 的做法
/// 把半角 `,`/`?` 换成全角、同时记下 `。` 的下标（`self.period`）。
fn read_punc_list(path: &Path) -> Result<(Vec<String>, usize)> {
    let raw = fs::read_to_string(path)
        .map_err(|e| Error::Model(format!("读取 {} 失败: {e}", path.display())))?;
    let mut punc_list = parse_punc_list(&raw).ok_or_else(|| {
        Error::Model(format!(
            "{} 里没有可用的 model_conf.punc_list",
            path.display()
        ))
    })?;
    let mut period = 0usize;
    for (idx, punc) in punc_list.iter_mut().enumerate() {
        match punc.as_str() {
            "," => *punc = "，".to_string(),
            "?" => *punc = "？".to_string(),
            "。" => period = idx,
            _ => {}
        }
    }
    Ok((punc_list, period))
}

/// 最小 YAML 解析：只取 `punc_list:` 之后的缩进列表项（够用且不引入 yaml 依赖）。
fn parse_punc_list(config: &str) -> Option<Vec<String>> {
    let mut items = Vec::new();
    let mut in_list = false;
    for line in config.lines() {
        let trimmed = line.trim();
        if !in_list {
            in_list = trimmed == "punc_list:";
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix('-') {
            if !rest.is_empty() && !rest.starts_with(' ') {
                break; // 形如 `-foo`：不是列表项
            }
            items.push(unquote(rest.trim()));
        } else if !trimmed.is_empty() && !trimmed.starts_with('#') {
            break; // 列表结束（下一个配置项）
        }
    }
    if items.is_empty() { None } else { Some(items) }
}

/// 去掉 YAML 标量两侧的成对引号。
fn unquote(value: &str) -> String {
    let bytes = value.as_bytes();
    if bytes.len() >= 2 {
        let first = bytes[0];
        let last = bytes[bytes.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return value[1..value.len() - 1].to_string();
        }
    }
    value.to_string()
}
