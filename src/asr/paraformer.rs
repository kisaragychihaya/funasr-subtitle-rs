//! T11：Paraformer-zh（非自回归中文 ASR）与热词版（`contextual.rs`）的公共骨架。
//!
//! 移植来源（顺序、边界一律照抄源码，不做「修正」）：
//! * `reference/funasr_onnx/paraformer_bin.py`：`__init__`（模型/词表/前端/CMVN/predictor_bias）、
//!   `extract_feat` → `infer` → `decode/decode_one` 的链路、`__call__` 里的 preds 提取；
//! * `reference/funasr_onnx/utils/postprocess_utils.py`：`sentence_postprocess`，含
//!   `isChinese` / `isAllChinese` / `isAllAlpha` / `abbr_dispose` 的逐个怪癖（见下文注释）；
//! * `reference/funasr_onnx/utils/utils.py`：`TokenIDConverter`（词表 + `ids2tokens`）；
//! * `reference/py-src/funasr_engine.py`：`_asr_paraformer`（`preds[0]` 再 `strip()`），以及
//!   contextual 的 `language = "zh"` 补丁——后者等价于「非 `en-bpe` 一律走
//!   `sentence_postprocess`」，本移植直接在加载期拒绝 `en-bpe` 模型。
//!
//! 与 Python 的两处有意差异（都不影响 golden）：
//! 1. 推理失败时 Python 的 `except ONNXRuntimeError` 会把本段当静音、最终返回空串；这里按
//!    `Result` 上抛，由上层决定容错（与 `punc.rs` 的错误语义一致），不静默吞错。
//! 2. 参考实现只读 `tokens.json`；这里额外支持任务书要求的 `tokens.txt`（一行一个 token），
//!    两者同时存在时以 `tokens.json` 为准。
//!
//! golden 对拍的量化输入是 `model_quant.onnx`（`funasr_engine.py` 对三个 ASR 都传
//! `quantize=True`），前端参数读自模型目录的 `config.yaml`（两个中文模型与 T6 的
//! `paraformer_frontend()` 逐字段一致，`am.mvn` 也与 SenseVoice 逐字节相同）。

use std::collections::HashMap;
use std::path::Path;

use ort::session::Session;
use ort::value::{DynValue, Value};

use crate::asr::{AsrChunkOutput, AsrEngine};
use crate::error::{Error, Result};
use crate::frontend::{Cmvn, FrontendConfig, extract_features, load_cmvn, paraformer_frontend};
use crate::text::normalize::py_trim;
use crate::types::{EngineKind, JobOptions};

/// 量化权重文件名（`funasr_engine.py` 固定 `quantize=True`）。
const MODEL_FILE: &str = "model_quant.onnx";
/// 词表（参考实现唯一读取的词表文件，JSON 字符串数组）。
const TOKENS_JSON: &str = "tokens.json";
/// 一行一个 token 的词表（任务书要求支持；参考实现没有这条路径）。
const TOKENS_TXT: &str = "tokens.txt";
const CONFIG_FILE: &str = "config.yaml";
const CMVN_FILE: &str = "am.mvn";
/// `<blank>` 的 id（`decode_one` 里 `filter(lambda x: x not in (0, 2))`）。
const BLANK_ID: i64 = 0;
/// `</s>` 的 id（同上被过滤掉）。
const EOS_ID: i64 = 2;

/// 词表里 `<blank>` / `<s>` / `</s>` 之后就是模型词表本体；输入只允许 2（Paraformer）或
/// 3（Contextual，多一个 `bias_embed`）个。
const INPUT_MIN: usize = 2;
const INPUT_MAX: usize = 3;
/// 至少要有 `logits` 与 `token_num` 两个输出（BiCifParaformer 还会多 `us_alphas/us_peaks`，
/// 本移植不使用时间戳路径）。
const OUTPUT_MIN: usize = 2;
/// `WavFrontend` 未在 config 里写时的 Kaldi 默认预加重系数。
const PREEMPHASIS: f32 = 0.97;
/// golden 把 dither 钉成 0（见 `tests/golden/index.json`）；参考实现默认 1.0 时输出不可复现。
const DITHER: f32 = 0.0;

// ---------------------------------------------------------------- 词表

/// `tokens.json` / `tokens.txt` 词表（`utils.TokenIDConverter`）。
#[derive(Debug, Clone)]
pub struct TokenList {
    tokens: Vec<String>,
    token2id: HashMap<String, usize>,
    unk_id: usize,
}

impl TokenList {
    /// 读词表：先 `tokens.json`（Python 侧唯一路径），再 `tokens.txt`（一行一个 token）。
    pub fn load(model_dir: &Path) -> Result<Self> {
        let json = model_dir.join(TOKENS_JSON);
        if json.is_file() {
            let raw = std::fs::read_to_string(&json)
                .map_err(|e| Error::Model(format!("读取 {} 失败: {e}", json.display())))?;
            let tokens: Vec<String> = serde_json::from_str(&raw)
                .map_err(|e| Error::Model(format!("解析 {} 失败: {e}", json.display())))?;
            return Self::new(tokens);
        }
        let txt = model_dir.join(TOKENS_TXT);
        if txt.is_file() {
            let raw = std::fs::read_to_string(&txt)
                .map_err(|e| Error::Model(format!("读取 {} 失败: {e}", txt.display())))?;
            let tokens: Vec<String> = raw.lines().map(str::to_string).collect();
            return Self::new(tokens);
        }
        Err(Error::Model(format!(
            "{} 下缺少词表（{TOKENS_JSON} 或 {TOKENS_TXT}）",
            model_dir.display()
        )))
    }

    /// `TokenIDConverter.__init__`：`unk_symbol` 取最后一项；`token2id` 是 dict，
    /// 重复 token 取最后一个下标（`{v: i for i, v in enumerate(token_list)}`）。
    pub fn new(tokens: Vec<String>) -> Result<Self> {
        let unk_symbol = tokens
            .last()
            .cloned()
            .ok_or_else(|| Error::Model("词表为空".to_string()))?;
        let mut token2id = HashMap::with_capacity(tokens.len());
        for (idx, token) in tokens.iter().enumerate() {
            token2id.insert(token.clone(), idx);
        }
        let unk_id = token2id
            .get(&unk_symbol)
            .copied()
            .ok_or_else(|| Error::Model(format!("词表里找不到 unk 符号 {unk_symbol:?}")))?;
        Ok(Self {
            tokens,
            token2id,
            unk_id,
        })
    }

    /// 词表项（`token_list`）。
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    /// 词表大小。
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// 词表是否为空（`TokenList::new` 会拒绝空表，这里只是随 `len` 一起提供）。
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// `<unk>` 的下标（`TokenIDConverter.unk_id`）。
    pub fn unk_id(&self) -> usize {
        self.unk_id
    }

    /// token → id（`token2id`）；不存在返回 `None`（调用方按 `tokens2ids` 语义落到 unk）。
    pub fn id_of(&self, token: &str) -> Option<usize> {
        self.token2id.get(token).copied()
    }

    /// `TokenIDConverter.ids2tokens`：越界下标在 Python 侧是 IndexError，这里上抛。
    pub fn ids2tokens(&self, ids: &[i64]) -> Result<Vec<String>> {
        ids.iter()
            .map(|&id| {
                if id < 0 {
                    return Err(Error::Inference(format!("token id 为负: {id}")));
                }
                self.tokens.get(id as usize).cloned().ok_or_else(|| {
                    Error::Inference(format!("token id {id} 超出词表大小 {}", self.tokens.len()))
                })
            })
            .collect()
    }
}

// ---------------------------------------------------------------- config.yaml

/// `config.yaml` 里本移植用到的字段（`read_yaml` 在 Python 侧是整份 YAML，这里只取需要的）。
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// 前端参数（`WavFrontend(cmvn_file=..., **config["frontend_conf"])`）。
    pub frontend: FrontendConfig,
    /// `model_conf.predictor_bias`（缺省 0，与 `Paraformer.__init__` 一致）。
    pub predictor_bias: usize,
    /// 顶层 `lang`（两个中文模型都没有这个键 → `None`，走 `sentence_postprocess`）。
    pub language: Option<String>,
}

/// 读模型目录的 `config.yaml`。
pub fn load_model_config(path: &Path) -> Result<ModelConfig> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| Error::Model(format!("读取 {} 失败: {e}", path.display())))?;
    parse_model_config(&raw)
}

/// 最小 YAML 解析：只认 `frontend_conf`/`model_conf` 两层的 `key: value` 标量与顶层 `lang`。
///
/// 参考实现用完整 YAML（yaml.Loader），但这两个模型目录里我们只依赖这几个标量，
/// 引入 YAML 依赖不值得（项目禁止新增依赖）。缺少某个键时落到 `paraformer_frontend()`
/// 预设值；写了却与前端能力不符（非 16k、非 hamming）时显式报错而不是静默跑歪。
pub fn parse_model_config(text: &str) -> Result<ModelConfig> {
    let mut top: HashMap<String, String> = HashMap::new();
    let mut blocks: HashMap<String, HashMap<String, String>> = HashMap::new();
    let mut section: Option<String> = None;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        let Some((key, value)) = trimmed.split_once(':') else {
            continue; // 列表项（如 `- 0`）与其它非映射行
        };
        let key = key.trim().to_string();
        let value = unquote(value.trim());
        if indent == 0 {
            if value.is_empty() {
                section = Some(key);
            } else {
                section = None;
                top.insert(key, value);
            }
        } else if let Some(sec) = &section
            && !value.is_empty()
        {
            blocks.entry(sec.clone()).or_default().insert(key, value);
        }
    }

    let preset = paraformer_frontend();
    let fe = blocks.get("frontend_conf");
    let fs = opt_usize(fe, "fs")?.unwrap_or(16_000);
    if fs != 16_000 {
        return Err(Error::Frontend(format!(
            "前端只支持 16000 Hz，config.yaml 写的是 {fs} Hz"
        )));
    }
    if let Some(window) = fe.and_then(|m| m.get("window"))
        && window != "hamming"
    {
        return Err(Error::Frontend(format!(
            "fbank 窗口只实现了 hamming，config.yaml 写的是 {window:?}"
        )));
    }
    let frontend = FrontendConfig {
        n_mels: opt_usize(fe, "n_mels")?.unwrap_or(preset.n_mels),
        frame_length_ms: opt_usize(fe, "frame_length")?.unwrap_or(preset.frame_length_ms as usize)
            as f32,
        frame_shift_ms: opt_usize(fe, "frame_shift")?.unwrap_or(preset.frame_shift_ms as usize)
            as f32,
        preemphasis: PREEMPHASIS,
        dither: DITHER,
        lfr_m: opt_usize(fe, "lfr_m")?.unwrap_or(preset.lfr_m),
        lfr_n: opt_usize(fe, "lfr_n")?.unwrap_or(preset.lfr_n),
    };
    let predictor_bias = opt_usize(blocks.get("model_conf"), "predictor_bias")?.unwrap_or(0);
    let language = top.get("lang").and_then(|v| {
        if v.is_empty() || v == "None" || v == "null" || v == "~" {
            None
        } else {
            Some(v.clone())
        }
    });
    Ok(ModelConfig {
        frontend,
        predictor_bias,
        language,
    })
}

/// 取某个块里的整数字段；键不存在返回 `None`，存在但解析失败上抛。
fn opt_usize(block: Option<&HashMap<String, String>>, key: &str) -> Result<Option<usize>> {
    match block.and_then(|m| m.get(key)) {
        None => Ok(None),
        Some(raw) => raw
            .trim_matches('"')
            .trim_matches('\'')
            .parse::<usize>()
            .map(Some)
            .map_err(|e| Error::Model(format!("config.yaml 的 {key}={raw:?} 不是整数: {e}"))),
    }
}

/// 去掉 YAML 标量两侧的成对引号。
fn unquote(value: &str) -> String {
    let bytes = value.as_bytes();
    if bytes.len() >= 2 {
        let (first, last) = (bytes[0], bytes[bytes.len() - 1]);
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return value[1..value.len() - 1].to_string();
        }
    }
    value.to_string()
}

// ---------------------------------------------------------------- 解码（decode_one）

/// `Paraformer.decode_one` 的纯逻辑部分：逐帧 argmax → 去 blank/eos →
/// 截断到 `valid_token_num - pred_bias`。
///
/// Python 先补 `[1] + yseq + [2]` 再切 `[1:-1]`，两头正好抵消，这里等价地不补；
/// `np.argmax` 取**第一个**最大值，因此用严格大于比较。
pub fn decode_frames(
    scores: &[f32],
    vocab: usize,
    valid_token_num: i64,
    pred_bias: usize,
) -> Result<Vec<i64>> {
    if vocab == 0 {
        return Err(Error::Inference("词表大小为 0".to_string()));
    }
    let frames = scores.len() / vocab;
    let mut ids: Vec<i64> = Vec::with_capacity(frames);
    for frame in 0..frames {
        let row = &scores[frame * vocab..(frame + 1) * vocab];
        let mut best = 0usize;
        for (cls, value) in row.iter().enumerate().skip(1) {
            if *value > row[best] {
                best = cls;
            }
        }
        let id = best as i64;
        if id != BLANK_ID && id != EOS_ID {
            ids.push(id);
        }
    }
    // Python `token[: valid_token_num - pred_bias]`：负数按负索引语义从头截掉
    let keep = valid_token_num - pred_bias as i64;
    let len = ids.len() as i64;
    let keep = if keep >= 0 {
        keep.min(len)
    } else {
        (len + keep).max(0)
    };
    ids.truncate(keep as usize);
    Ok(ids)
}

// ---------------------------------------------------------------- 后处理（sentence_postprocess）

/// `isAllChinese`/`isAllAlpha` 里对每个词做的清洗（`replace(" ")` → `replace("</s>")`
/// → `replace("<s>")`）。
fn clean_word(word: &str) -> String {
    word.replace(' ', "").replace("</s>", "").replace("<s>", "")
}

/// Python 字符串比较（按码点）的 `a <= b`；`isChinese` 对多字符词也照 Python 的
/// 整串字典序比较，所以不能简化成「单字符判断」。
fn py_le(a: &str, b: &str) -> bool {
    let mut left = a.chars();
    let mut right = b.chars();
    loop {
        match (left.next(), right.next()) {
            (None, _) => return true,
            (Some(_), None) => return false,
            (Some(x), Some(y)) => {
                if x < y {
                    return true;
                }
                if x > y {
                    return false;
                }
            }
        }
    }
}

/// `isChinese`：`"\u4e00" <= ch <= "\u9fff"` 或 `"0" <= ch <= "9"`（数字也算「中文」）。
fn is_chinese(word: &str) -> bool {
    (py_le("\u{4e00}", word) && py_le(word, "\u{9fff}")) || (py_le("0", word) && py_le(word, "9"))
}

/// Python `str.isalpha()`（Unicode 字母，空串为 False）。
fn py_isalpha(word: &str) -> bool {
    !word.is_empty() && word.chars().all(char::is_alphabetic)
}

/// `isAllChinese`：空列表为 False；每个词清洗后都要满足 `isChinese`。
fn is_all_chinese(words: &[&str]) -> bool {
    if words.is_empty() {
        return false;
    }
    words.iter().all(|w| is_chinese(&clean_word(w)))
}

/// `isAllAlpha`：空列表为 False；清洗后 `isalpha()` 为假只允许单引号，
/// `isalpha()` 为真则不能是中文（CJK 在 Python 里也是 alpha）。
fn is_all_alpha(words: &[&str]) -> bool {
    if words.is_empty() {
        return false;
    }
    words.iter().all(|w| {
        let cur = clean_word(w);
        if py_isalpha(&cur) {
            !is_chinese(&cur)
        } else {
            cur == "'"
        }
    })
}

/// 单个词版本的 `isAllChinese([word])`。
fn is_chinese_word(word: &str) -> bool {
    is_chinese(&clean_word(word))
}

/// 单个词版本的 `isAllAlpha([word])`。
fn is_alpha_word(word: &str) -> bool {
    let cur = clean_word(word);
    if py_isalpha(&cur) {
        !is_chinese(&cur)
    } else {
        cur == "'"
    }
}

/// 单个 ASCII 字母（Python `len(w) == 1 and w.encode("utf-8").isalpha()`：
/// `bytes.isalpha` 只认 ASCII 字母）。
fn is_single_ascii_letter(word: &str) -> bool {
    let mut chars = word.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => c.is_ascii_alphabetic(),
        _ => false,
    }
}

/// 移植 `abbr_dispose`（`time_stamp=None` 分支）：把「单个字母 + 空格 + 单个字母」串成缩写，
/// 逐字母大写化。
///
/// 两遍扫描与 Python 完全相同，包括两处反直觉行为：第一遍的 `last_num` 只影响第一遍自身的
/// 跳过判断；第二遍重新从 -1 开始，遇到 `abbr_begin` 就把空格吞掉、只把字母大写后拼接。
fn abbr_dispose(words: &[String]) -> Vec<String> {
    let n = words.len();
    let mut abbr_begin: Vec<usize> = Vec::new();
    let mut abbr_end: Vec<usize> = Vec::new();

    // 第一遍：找缩写区间（Python 的 for 循环里改 num 不影响循环变量，只有内层 while 走位）
    let mut last_num: isize = -1;
    for num in 0..n {
        if num as isize <= last_num {
            continue;
        }
        if !is_single_ascii_letter(&words[num]) {
            continue;
        }
        if num + 1 < n
            && words[num + 1] == " "
            && num + 2 < n
            && is_single_ascii_letter(&words[num + 2])
        {
            abbr_begin.push(num);
            let mut cursor = num + 2;
            abbr_end.push(cursor);
            loop {
                cursor += 1;
                if cursor < n && words[cursor] == " " {
                    cursor += 1;
                    if cursor < n && is_single_ascii_letter(&words[cursor]) {
                        abbr_end.pop();
                        abbr_end.push(cursor);
                        last_num = cursor as isize;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
    }

    // 第二遍：按区间拼接（区间内只保留字母并大写，空格丢弃）
    let mut out: Vec<String> = Vec::with_capacity(n);
    let mut last_num: isize = -1;
    let mut num = 0usize;
    while num < n {
        if num as isize <= last_num {
            num += 1;
            continue;
        }
        if abbr_begin.contains(&num) {
            out.push(words[num].to_uppercase());
            num += 1;
            while num < n {
                if abbr_end.contains(&num) {
                    out.push(words[num].to_uppercase());
                    last_num = num as isize;
                    break;
                }
                if is_single_ascii_letter(&words[num]) {
                    out.push(words[num].to_uppercase());
                }
                num += 1;
            }
            num += 1;
        } else {
            out.push(words[num].clone());
            num += 1;
        }
    }
    out
}

/// 移植 `sentence_postprocess`（`time_stamp=None` 分支），返回 Python 的第一个返回值
/// （整句 `sentence`；`real_word_lists` 在引擎里没用）。
///
/// 三条分支：全中文（去掉词内空格直接拼）、全字母（`@@` 续接，词间补空格）、混合
/// （中文打断前一个英文空格；`@@` 续接；遇到非法字符 Python 抛 ValueError，这里上抛）。
pub fn sentence_postprocess(words: &[String]) -> Result<String> {
    // wash words lists：丢掉句界符号
    let middle_lists: Vec<&str> = words
        .iter()
        .map(String::as_str)
        .filter(|w| !matches!(*w, "<s>" | "</s>" | "<unk>"))
        .collect();

    let mut word_lists: Vec<String> = Vec::new();
    if is_all_chinese(&middle_lists) {
        for ch in &middle_lists {
            word_lists.push(ch.replace(' ', ""));
        }
    } else if is_all_alpha(&middle_lists) {
        let mut word_item = String::new();
        for ch in &middle_lists {
            if ch.contains("@@") {
                word_item.push_str(&ch.replace("@@", ""));
            } else {
                word_item.push_str(ch);
                word_lists.push(word_item.clone());
                word_lists.push(" ".to_string());
                word_item.clear();
            }
        }
    } else {
        let mut word_item = String::new();
        let mut alpha_blank = false;
        for ch in &middle_lists {
            if is_chinese_word(ch) {
                // 中文紧贴前一个英文词：把刚补上的空格吞掉
                if alpha_blank {
                    word_lists.pop();
                }
                word_lists.push((*ch).to_string());
                alpha_blank = false;
            } else if ch.contains("@@") {
                word_item.push_str(&ch.replace("@@", ""));
                alpha_blank = false;
            } else if is_alpha_word(ch) {
                word_item.push_str(ch);
                word_lists.push(word_item.clone());
                word_lists.push(" ".to_string());
                word_item.clear();
                alpha_blank = true;
            } else {
                return Err(Error::Inference(format!("invalid character: {ch}")));
            }
        }
    }

    let word_lists = abbr_dispose(&word_lists);
    let joined: String = word_lists.concat();
    Ok(py_trim(&joined).to_string())
}

/// token 序列 → 引擎整句文本：`_asr_paraformer` = `sentence_postprocess(...)[0].strip()`。
pub fn sentence_text(tokens: &[String]) -> Result<String> {
    Ok(py_trim(&sentence_postprocess(tokens)?).to_string())
}

// ---------------------------------------------------------------- ONNX Runtime

/// 建 session：线程数、图优化、日志级别对齐 `OrtInferSession`。
/// 环境共享自 `crate::ort_util`（ort 每进程只允许一个 `Environment`）。
pub(crate) fn build_session(model_file: &Path, threads: usize) -> Result<Session> {
    crate::ort_util::build_session(model_file, threads)
}

// ---------------------------------------------------------------- 引擎骨架

/// 热词偏置矩阵（`bias_embed`），按 `[1, rows, dim]` 喂进 bb 模型。
pub(crate) struct BiasEmbed {
    pub(crate) data: Vec<f32>,
    pub(crate) rows: usize,
    pub(crate) dim: usize,
}

/// Paraformer / ContextualParaformer 共用的模型部分（词表、前端、CMVN、bb session）。
pub(crate) struct ParaformerCore {
    session: Session,
    tokens: TokenList,
    frontend: FrontendConfig,
    cmvn: Cmvn,
    /// 模型声明的输入名：Python `zip(get_input_names(), [...])` 就是按声明顺序喂的。
    input_names: Vec<String>,
    pred_bias: usize,
}

impl ParaformerCore {
    pub(crate) fn load(model_dir: &Path, threads: usize) -> Result<Self> {
        let model_file = model_dir.join(MODEL_FILE);
        if !model_file.is_file() {
            return Err(Error::Model(format!(
                "Paraformer 模型文件不存在: {}",
                model_file.display()
            )));
        }
        let tokens = TokenList::load(model_dir)?;
        let cmvn = load_cmvn(&model_dir.join(CMVN_FILE))?;
        let cfg = load_model_config(&model_dir.join(CONFIG_FILE))?;
        if cfg.language.as_deref() == Some("en-bpe") {
            return Err(Error::Model(
                "不支持 lang=en-bpe 的模型：参考实现会走 sentencepiece 后处理".to_string(),
            ));
        }
        let session = build_session(&model_file, threads)?;
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|input| input.name().to_string())
            .collect();
        if input_names.len() < INPUT_MIN || input_names.len() > INPUT_MAX {
            return Err(Error::Model(format!(
                "Paraformer 模型输入个数 {}，期望 {INPUT_MIN}..={INPUT_MAX}（speech/speech_lengths[/bias_embed]）",
                input_names.len()
            )));
        }
        if session.outputs().len() < OUTPUT_MIN {
            return Err(Error::Model(format!(
                "Paraformer 模型输出个数 {}，期望至少 {OUTPUT_MIN}（logits/token_num）",
                session.outputs().len()
            )));
        }
        Ok(Self {
            session,
            tokens,
            frontend: cfg.frontend,
            cmvn,
            input_names,
            pred_bias: cfg.predictor_bias,
        })
    }

    pub(crate) fn tokens(&self) -> &TokenList {
        &self.tokens
    }

    /// `extract_feat` → `infer` → `decode`：返回解码后的 token 序列。
    ///
    /// `bias` 只有 Contextual 传；输入个数与模型声明不符时显式报错（Python 的 `zip`
    /// 会把多余的值静默丢掉，正是 `model_eb` 只有 1 个输入时丢掉长度的原因）。
    pub(crate) fn transcribe_tokens(
        &mut self,
        chunk: &[i16],
        sample_rate: u32,
        bias: Option<&BiasEmbed>,
    ) -> Result<Vec<String>> {
        let feats = extract_features(chunk, sample_rate, &self.frontend, Some(&self.cmvn))?;
        let (rows, cols) = feats.feats.dim();
        if rows == 0 || cols == 0 {
            return Err(Error::Inference(format!(
                "Paraformer 输入特征为空: {rows}x{cols}"
            )));
        }
        let mut flat = Vec::with_capacity(rows * cols);
        flat.extend(feats.feats.iter().copied());

        let speech = Value::from_array((vec![1_i64, rows as i64, cols as i64], flat))
            .map_err(|e| Error::Inference(format!("构造 speech 张量失败: {e}")))?
            .into_dyn();
        let speech_lengths = Value::from_array((vec![1_i64], vec![rows as i32]))
            .map_err(|e| Error::Inference(format!("构造 speech_lengths 张量失败: {e}")))?
            .into_dyn();
        let mut values: Vec<DynValue> = vec![speech, speech_lengths];
        if let Some(bias) = bias {
            values.push(
                Value::from_array((
                    vec![1_i64, bias.rows as i64, bias.dim as i64],
                    bias.data.clone(),
                ))
                .map_err(|e| Error::Inference(format!("构造 bias_embed 张量失败: {e}")))?
                .into_dyn(),
            );
        }
        if values.len() != self.input_names.len() {
            return Err(Error::Inference(format!(
                "Paraformer 模型声明了 {} 个输入（{}），本次准备了 {} 个",
                self.input_names.len(),
                self.input_names.join(", "),
                values.len()
            )));
        }
        let inputs: Vec<(String, DynValue)> =
            self.input_names.iter().cloned().zip(values).collect();
        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| Error::Inference(format!("Paraformer 推理失败: {e}")))?;
        if outputs.len() < OUTPUT_MIN {
            return Err(Error::Inference(format!(
                "Paraformer 输出个数 {}，期望至少 {OUTPUT_MIN}",
                outputs.len()
            )));
        }

        // outputs[1] = token_num（可能 int32 也可能 int64，取第一个元素）
        let valid_token_num = match outputs[1].try_extract_scalar::<i32>() {
            Ok(v) => v as i64,
            Err(_) => outputs[1]
                .try_extract_scalar::<i64>()
                .map_err(|e| Error::Inference(format!("读取 token_num 失败: {e}")))?,
        };

        // outputs[0] = logits，形状 [1, T, vocab]
        let (shape, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Inference(format!("读取 logits 失败: {e}")))?;
        let dims: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
        if dims.len() != 3 || dims[0] == 0 || dims[1] == 0 || dims[2] == 0 {
            return Err(Error::Inference(format!(
                "Paraformer 输出维度异常: {dims:?}"
            )));
        }
        let (frames, vocab) = (dims[1], dims[2]);
        if logits.len() < frames * vocab {
            return Err(Error::Inference(format!(
                "Paraformer 输出大小异常: {} < {}",
                logits.len(),
                frames * vocab
            )));
        }
        let ids = decode_frames(
            &logits[..frames * vocab],
            vocab,
            valid_token_num,
            self.pred_bias,
        )?;
        self.tokens.ids2tokens(&ids)
    }
}

/// Paraformer-large（中文）引擎。
pub struct Paraformer {
    core: ParaformerCore,
}

impl Paraformer {
    pub fn new(model_dir: &Path, threads: usize) -> Result<Self> {
        Ok(Self {
            core: ParaformerCore::load(model_dir, threads)?,
        })
    }
}

impl AsrEngine for Paraformer {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::Paraformer
    }

    fn transcribe_chunk(
        &mut self,
        chunk: &[i16],
        sample_rate: u32,
        _job: &JobOptions,
    ) -> Result<AsrChunkOutput> {
        // 参考实现的 Paraformer.__call__ 不接热词，job.hotwords 在这里不生效。
        let tokens = self.core.transcribe_tokens(chunk, sample_rate, None)?;
        Ok(AsrChunkOutput::Plain(sentence_text(&tokens)?))
    }
}
