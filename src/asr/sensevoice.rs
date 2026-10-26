//! T8：SenseVoiceSmall。移植来源：funasr-subtitle `engine/funasr_engine.py` 的
//! _sensevoice_timed / _ctc_greedy_with_frames，以及 vendored `sensevoice_bin.py` 的
//! extract_feat / infer / _get_lid / _get_tnid。
//!
//! 对应关系（逐行对照源码，不做「修正」）：
//! - `SenseVoice::new` ← `SenseVoiceSmall.__init__`：模型目录里的 `model_quant.onnx`
//!   （funasr 侧固定 `quantize=True`）、`am.mvn`（CMVN）、
//!   `chn_jpn_yue_eng_ko_spectok.bpe.model`（sentencepiece 词表）。
//! - `SenseVoice::transcribe_chunk` ← `FunasrEngine._sensevoice_timed`：
//!   `extract_feat(chunk)` → `infer(feats, feats_len, lid, tn)` →
//!   `_ctc_greedy_with_frames` → `sp.id_to_piece` + `▁`→空格 + 跳过 `<|...|>` →
//!   `(frame - 4) / (T - 4)` 线性映射到段内毫秒。
//! - `SpModel` ← `SentencePieceProcessor.id_to_piece` 的只读子集：直接解析 `.bpe.model`
//!   （protobuf `ModelProto`）的 `pieces`，下标即 id（最小 protobuf 解析，不引依赖）。
//!
//! CMVN：参考实现的统计量是 **float64**（`frontend.py::load_cmvn` 的 `np.array(list(map(float, ...)))`），
//! `apply_cmvn` 也在 float64 下算完、由 `pad_feats` 统一落 float32。`frontend::Cmvn` 把统计量存成
//! f32，会让 feats 与 golden 差 1 ULP（max 4.77e-7），恰好翻转 `sample_en_runon` 上一个近 tie
//! 的 CTC 帧。因此本模块按源码口径用 `Cmvn64`（f64）自行应用 CMVN；`frontend::Cmvn` 改为 f64 后
//! 这一段即可删掉。
//!
//! 时间口径：Python 侧是 `int(span_start_ms + rel*span_ms)`（绝对毫秒）。`transcribe_chunk`
//! 只拿得到一个语音段、没有绝对偏移，因此这里产出**段内相对毫秒**（段首 = 0），由 pipeline
//! 加上 VAD span 起点。由于 `span_start_ms` 是整数且 `rel*span_ms >= 0`，
//! `int(start + x) == start + int(x)`，与 Python 的绝对时间逐值等价。

use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use ndarray::Array2;
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;

use crate::asr::{AsrChunkOutput, AsrEngine};
use crate::error::{Error, Result};
use crate::frontend::{FrontendConfig, extract_features, sensevoice_frontend};
use crate::types::{EngineKind, JobOptions, Language, TimedPiece};

/// 量化模型文件名（funasr 侧固定 `quantize=True`）。
const MODEL_FILE: &str = "model_quant.onnx";
/// CMVN 统计量文件名（`config.yaml` 的 `frontend_conf.cmvn_file`）。
const CMVN_FILE: &str = "am.mvn";
/// sentencepiece 词表文件名（`SentencepiecesTokenizer(bpemodel=...)`）。
const BPE_FILE: &str = "chn_jpn_yue_eng_ko_spectok.bpe.model";
/// CTC blank id（`self.blank_id = 0`）。
pub const BLANK_ID: usize = 0;
/// SenseVoice 编码器前置 4 个特殊 token（语言/情感/事件/ITN），其后才是语音帧。
pub const SV_PREFIX_FRAMES: usize = 4;
/// ONNX 模型的输入个数（speech / speech_lengths / language / textnorm）。
const INPUT_COUNT: usize = 4;
/// ONNX 模型的输出个数（ctc_logits / ctc_lengths）。
const OUTPUT_COUNT: usize = 2;
/// `.bpe.model` 里至少要有几个 piece（`<unk>`/`<s>`/`</s>` + 至少一个词）。
const MIN_PIECES: usize = 3;

/// `SenseVoiceSmall.lid_dict`：语言标签 → lid 整数。
pub const LID_DICT: &[(&str, i32)] = &[
    ("auto", 0),
    ("zh", 3),
    ("en", 4),
    ("yue", 7),
    ("ja", 11),
    ("ko", 12),
    ("nospeech", 13),
];

/// `SenseVoiceSmall.textnorm_dict`：文本正则标签 → 整数。
pub const TEXTNORM_DICT: &[(&str, i32)] = &[("withitn", 14), ("woitn", 15)];

/// `_get_lid`：语言标签查表，未知标签报错（Python 抛 ValueError）。
pub fn lid_for_name(name: &str) -> Result<i32> {
    LID_DICT
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, id)| *id)
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "语言 {name} 不在 {:?} 中",
                LID_DICT.iter().map(|(k, _)| *k).collect::<Vec<_>>()
            ))
        })
}

/// `_get_tnid`：文本正则标签查表，未知标签报错（Python 抛 ValueError）。
pub fn textnorm_for_name(name: &str) -> Result<i32> {
    TEXTNORM_DICT
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, id)| *id)
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "文本正则 {name} 不在 {:?} 中",
                TEXTNORM_DICT.iter().map(|(k, _)| *k).collect::<Vec<_>>()
            ))
        })
}

/// `JobOptions.language` → `_get_lid` 的入参名字（六种语言两者取值范围一致）。
fn language_name(language: Language) -> &'static str {
    match language {
        Language::Auto => "auto",
        Language::Zh => "zh",
        Language::Yue => "yue",
        Language::En => "en",
        Language::Ja => "ja",
        Language::Ko => "ko",
    }
}

/// 语言枚举 → lid 整数（`_get_lid`）。
pub fn lid_for(language: Language) -> Result<i32> {
    lid_for_name(language_name(language))
}

/// `_SPECIAL_TOKEN_RE.fullmatch`：`<\|[^|>]*\|>` 是否整串匹配（`<|zh|>`、`<|NEUTRAL|>` 等）。
pub fn is_special_token(piece: &str) -> bool {
    let Some(rest) = piece.strip_prefix("<|") else {
        return false;
    };
    let Some(inner) = rest.strip_suffix("|>") else {
        return false;
    };
    !inner.contains(['|', '>'])
}

/// `_ctc_greedy_with_frames`：CTC 贪心解码，保留每个 token 首次出现的帧号。
///
/// `logits` 是行主序的二维 logits（每帧 `vocab` 个值），`valid_len` 是有效帧数
/// （Python `ctc_logits[:valid_len]`，超出实际行数时 numpy 静默取全部行）。
/// 逐帧 argmax 取**第一个**最大值（numpy 语义），再按 `prev = blank_id` 起始去重。
pub fn ctc_greedy_with_frames(
    logits: &[f32],
    vocab: usize,
    valid_len: usize,
    blank_id: usize,
) -> Result<Vec<(usize, usize)>> {
    if vocab == 0 {
        return Err(Error::Inference("CTC 词表大小为 0".to_string()));
    }
    if blank_id >= vocab {
        return Err(Error::Inference(format!(
            "blank_id {blank_id} 超出词表大小 {vocab}"
        )));
    }
    if !logits.len().is_multiple_of(vocab) {
        return Err(Error::Inference(format!(
            "logits 长度 {} 不是词表大小 {vocab} 的整数倍",
            logits.len()
        )));
    }
    let rows = logits.len() / vocab;
    let frames = valid_len.min(rows);
    let mut out = Vec::new();
    let mut prev = blank_id;
    for frame in 0..frames {
        let row = &logits[frame * vocab..(frame + 1) * vocab];
        let mut best = 0usize;
        for (cls, value) in row.iter().enumerate().skip(1) {
            if *value > row[best] {
                best = cls;
            }
        }
        if best != prev && best != blank_id {
            out.push((best, frame));
        }
        prev = best;
    }
    Ok(out)
}

/// `_sensevoice_timed` 里的时间映射：`rel = (frame - 4) / max(1, T - 4)` 先裁到 [0,1]，
/// 再映射到 `[span_start_ms, span_end_ms]`，取整方式与 Python `int()` 一致（向零截断）。
pub fn token_time_ms(frame: usize, enc_len: usize, span_start_ms: i64, span_end_ms: i64) -> i64 {
    let speech_frames = enc_len.saturating_sub(SV_PREFIX_FRAMES).max(1);
    let span = (span_end_ms - span_start_ms).max(0);
    let rel = (frame as f64 - SV_PREFIX_FRAMES as f64) / speech_frames as f64;
    // Python: `0.0 if rel < 0 else (1.0 if rel > 1 else rel)`
    let rel = rel.clamp(0.0, 1.0);
    (span_start_ms as f64 + rel * span as f64) as i64
}

// ---------------------------------------------------------------- CMVN

/// CMVN 统计量，**float64**（Python `frontend.py::load_cmvn` 返回的就是 float64，
/// `apply_cmvn` 也在 float64 下算完才在 `pad_feats` 里落 float32）。
///
/// 为什么不用 `frontend::Cmvn`：那里受结构体字段类型限制把统计量存成 f32，CMVN 因此
/// 与参考实现差 1 ULP（实测 feats max|diff|=4.77e-7，仅约 1/3 的值逐位相同）。
/// 这点差在 `sample_en_runon` 上正好把一个近 tie 的 CTC 帧翻成别的 token，导致 token
/// 序列与 golden 不一致；改用 float64 后 feats 与 golden 逐位相同（0 差异）。
/// 若 `frontend::Cmvn` 今后改为 f64，本段（含 `load_cmvn_f64`/`apply_cmvn_f64`）可以删掉，
/// 直接改回 `extract_features(chunk, sr, cfg, Some(&cmvn))`。
struct Cmvn64 {
    mean: Vec<f64>,
    istd: Vec<f64>,
}

/// 读 am.mvn：格式与 `frontend.py::load_cmvn` 一致——`<AddShift>` / `<Rescale>` 行的
/// 下一行是 `<LearnRateCoef> 0 [ v0 v1 ... vN ]`，取 `[3 .. len-1]` 之间的数值。
fn load_cmvn_f64(path: &Path) -> Result<Cmvn64> {
    let text = fs::read_to_string(path)?;
    let lines: Vec<&str> = text.lines().collect();
    let mut mean: Option<Vec<f64>> = None;
    let mut istd: Option<Vec<f64>> = None;
    for (i, line) in lines.iter().enumerate() {
        let tag = match line.split_whitespace().next() {
            Some(t) => t,
            None => continue,
        };
        let slot = match tag {
            "<AddShift>" => &mut mean,
            "<Rescale>" => &mut istd,
            _ => continue,
        };
        let next: Vec<&str> = match lines.get(i + 1) {
            Some(l) => l.split_whitespace().collect(),
            None => continue,
        };
        if next.first().copied() != Some("<LearnRateCoef>") || next.len() < 5 {
            continue;
        }
        let mut vals = Vec::with_capacity(next.len() - 4);
        for tok in &next[3..next.len() - 1] {
            let v = tok
                .parse::<f64>()
                .map_err(|e| Error::Model(format!("am.mvn 里的数值 {tok:?} 无法解析：{e}")))?;
            vals.push(v);
        }
        *slot = Some(vals);
    }
    match (mean, istd) {
        (Some(m), Some(v)) if !m.is_empty() && m.len() == v.len() => {
            Ok(Cmvn64 { mean: m, istd: v })
        }
        _ => Err(Error::Model(format!(
            "am.mvn 缺少 <AddShift>/<Rescale> 或两者维度不一致：{}",
            path.display()
        ))),
    }
}

/// `WavFrontend.apply_cmvn`：`(x + mean) * istd`，float64 下算完再落 float32。
fn apply_cmvn_f64(feats: &Array2<f32>, cmvn: &Cmvn64) -> Result<Array2<f32>> {
    let dim = feats.ncols();
    if cmvn.mean.len() < dim || cmvn.istd.len() < dim {
        return Err(Error::Frontend(format!(
            "cmvn 维度 {} 小于特征维度 {dim}",
            cmvn.mean.len().min(cmvn.istd.len())
        )));
    }
    let mut out = feats.to_owned();
    for mut row in out.rows_mut() {
        for (d, v) in row.iter_mut().enumerate() {
            *v = ((f64::from(*v) + cmvn.mean[d]) * cmvn.istd[d]) as f32;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- sentencepiece

/// `.bpe.model`（protobuf `ModelProto`）的只读 piece 表：下标即 token id。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpModel {
    pieces: Vec<String>,
}

impl SpModel {
    /// 读 `.bpe.model`。
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)
            .map_err(|e| Error::Model(format!("读取 {} 失败: {e}", path.display())))?;
        Self::parse(&bytes).map_err(|e| Error::Model(format!("解析 {} 失败: {e}", path.display())))
    }

    /// 解析 protobuf 字节流：`ModelProto.pieces`（field 1，repeated SentencePiece），
    /// `SentencePiece.piece`（field 1，string）。其余字段（score/type/trainer_spec/…）按
    /// wire type 跳过。piece 的出现顺序就是 id（sentencepiece 的 `id_to_piece` 语义）。
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let mut reader = PbReader::new(bytes);
        let mut pieces = Vec::new();
        while !reader.done() {
            let (field, wire) = reader.tag()?;
            if field == 1 && wire == WIRE_LEN {
                pieces.push(parse_sentence_piece(reader.bytes()?)?);
            } else {
                reader.skip(wire)?;
            }
        }
        if pieces.len() < MIN_PIECES {
            return Err(Error::Model(format!(
                "bpe 模型只有 {} 个 piece（至少应有 {MIN_PIECES} 个），文件可能不是 ModelProto",
                pieces.len()
            )));
        }
        Ok(Self { pieces })
    }

    /// `sp.id_to_piece(id)`。
    pub fn id_to_piece(&self, id: usize) -> Result<&str> {
        self.pieces.get(id).map(String::as_str).ok_or_else(|| {
            Error::Inference(format!(
                "piece id {id} 越界（词表 {} 个）",
                self.pieces.len()
            ))
        })
    }

    /// 词表大小（`sp.GetPieceSize()`）。
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }
}

/// protobuf 线格式：长度前缀（length-delimited）。
const WIRE_VARINT: u32 = 0;
const WIRE_64BIT: u32 = 1;
const WIRE_LEN: u32 = 2;
const WIRE_32BIT: u32 = 5;

/// 解析一条 `SentencePiece` 子消息，只取 `piece`（field 1）。
fn parse_sentence_piece(buf: &[u8]) -> Result<String> {
    let mut reader = PbReader::new(buf);
    let mut piece: Option<String> = None;
    while !reader.done() {
        let (field, wire) = reader.tag()?;
        if field == 1 && wire == WIRE_LEN {
            let raw = reader.bytes()?;
            let text = std::str::from_utf8(raw)
                .map_err(|e| Error::Model(format!("bpe 模型的 piece 不是合法 UTF-8: {e}")))?;
            piece = Some(text.to_string());
        } else {
            reader.skip(wire)?;
        }
    }
    piece.ok_or_else(|| Error::Model("bpe 模型的 SentencePiece 缺少 piece 字段".to_string()))
}

/// 最小 protobuf 读取器（只覆盖 `.bpe.model` 用到的 wire type）。
struct PbReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> PbReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            let byte = *self
                .buf
                .get(self.pos)
                .ok_or_else(|| Error::Model("bpe 模型 protobuf 截断（varint）".to_string()))?;
            self.pos += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
            if shift >= 64 {
                return Err(Error::Model(
                    "bpe 模型 protobuf varint 超过 64 位".to_string(),
                ));
            }
        }
    }

    /// 读 tag，返回 (字段号, wire type)。
    fn tag(&mut self) -> Result<(u64, u32)> {
        let tag = self.varint()?;
        let field = tag >> 3;
        if field == 0 {
            return Err(Error::Model(
                "bpe 模型 protobuf 字段号为 0（文件不是 ModelProto）".to_string(),
            ));
        }
        Ok((field, (tag & 0x7) as u32))
    }

    fn bytes(&mut self) -> Result<&'a [u8]> {
        let len = self.varint()? as usize;
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| Error::Model("bpe 模型 protobuf 长度溢出".to_string()))?;
        let slice = self.buf.get(self.pos..end).ok_or_else(|| {
            Error::Model("bpe 模型 protobuf 截断（length-delimited）".to_string())
        })?;
        self.pos = end;
        Ok(slice)
    }

    fn advance(&mut self, n: usize) -> Result<()> {
        self.pos = self
            .pos
            .checked_add(n)
            .filter(|end| *end <= self.buf.len())
            .ok_or_else(|| Error::Model("bpe 模型 protobuf 截断（定长字段）".to_string()))?;
        Ok(())
    }

    fn skip(&mut self, wire: u32) -> Result<()> {
        match wire {
            WIRE_VARINT => {
                self.varint()?;
            }
            WIRE_64BIT => self.advance(8)?,
            WIRE_LEN => {
                self.bytes()?;
            }
            WIRE_32BIT => self.advance(4)?,
            other => {
                return Err(Error::Model(format!(
                    "bpe 模型 protobuf 的 wire type {other} 不支持"
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------- 引擎本体

/// 进程级 ONNX Runtime 环境（ort 每进程只允许一个 Environment）。
///
/// 注意：其他模块若各自建 Environment，后建的那一方会失败（ort 会返回
/// "only one environment is allowed per process"）。这里把错误显式上抛，
/// 不做静默降级；统一到共享 helper 时可直接改调用点。
fn ort_environment() -> Result<Environment> {
    static ENV: OnceLock<Environment> = OnceLock::new();
    if let Some(env) = ENV.get() {
        return Ok(env.clone());
    }
    let env = ort::init()
        .with_name("funasr-subtitle-rs")
        .build()
        .map_err(|e| {
            Error::Model(format!(
                "初始化 ONNX Runtime 失败（每进程只允许一个 Environment）: {e}"
            ))
        })?;
    let _ = ENV.set(env);
    ENV.get()
        .cloned()
        .ok_or_else(|| Error::Model("初始化 ONNX Runtime 失败".to_string()))
}

/// 一次推理的解码结果：`(token_id, frame)` 序列、有效编码帧数 T、词表大小。
type CtcPairs = (Vec<(usize, usize)>, usize, usize);

/// SenseVoiceSmall 引擎（量化 ONNX，纯 CPU）。
pub struct SenseVoice {
    session: Session,
    /// sentencepiece 词表（`asr.tokenizer.sp.id_to_piece`）。
    pieces: SpModel,
    /// 前端参数（`WavFrontend(**config["frontend_conf"])`）。
    cfg: FrontendConfig,
    /// CMVN 统计量（`am.mvn`，float64；见 `Cmvn64` 的说明）。
    cmvn: Cmvn64,
    /// 输入名按模型声明顺序保存：Python 侧 `zip(get_input_names(), [feats, len, lid, tn])`
    /// 就是按声明顺序喂的，这里保持同样的语义。
    input_names: [String; INPUT_COUNT],
}

impl SenseVoice {
    pub fn new(model_dir: &Path, threads: usize) -> Result<Self> {
        let model_file = model_dir.join(MODEL_FILE);
        if !model_file.is_file() {
            return Err(Error::Model(format!(
                "SenseVoice 模型文件不存在: {}",
                model_file.display()
            )));
        }
        let pieces = SpModel::load(&model_dir.join(BPE_FILE))?;
        let cmvn = load_cmvn_f64(&model_dir.join(CMVN_FILE))?;
        let cfg = sensevoice_frontend();

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
        let session = builder.commit_from_file(&model_file).map_err(|e| {
            Error::Model(format!(
                "加载 SenseVoice 模型 {} 失败: {e}",
                model_file.display()
            ))
        })?;

        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|outlet| outlet.name().to_string())
            .collect();
        if input_names.len() != INPUT_COUNT {
            return Err(Error::Model(format!(
                "SenseVoice 模型输入个数 {}，期望 {INPUT_COUNT}（speech/speech_lengths/language/textnorm）",
                input_names.len()
            )));
        }
        if session.outputs().len() < OUTPUT_COUNT {
            return Err(Error::Model(format!(
                "SenseVoice 模型输出个数 {}，期望 {OUTPUT_COUNT}（ctc_logits/ctc_lengths）",
                session.outputs().len()
            )));
        }
        let input_names: [String; INPUT_COUNT] = <[String; INPUT_COUNT]>::try_from(input_names)
            .map_err(|_| Error::Model("SenseVoice 模型输入个数与期望不符".to_string()))?;

        Ok(Self {
            session,
            pieces,
            cfg,
            cmvn,
            input_names,
        })
    }

    /// `SenseVoiceSmall.infer` + `_ctc_greedy_with_frames`：跑一次 ONNX，直接返回
    /// (token_id, frame) 序列、有效编码帧数 T、词表大小。
    fn infer_pairs(&mut self, feats: &Array2<f32>, lid: i32, tn: i32) -> Result<CtcPairs> {
        let (rows, cols) = feats.dim();
        if rows == 0 || cols == 0 {
            return Err(Error::Inference(format!(
                "SenseVoice 输入特征为空: {rows}x{cols}"
            )));
        }
        let mut flat = Vec::with_capacity(rows * cols);
        flat.extend(feats.iter().copied());

        let speech = Value::from_array((vec![1_i64, rows as i64, cols as i64], flat))
            .map_err(|e| Error::Inference(format!("构造 speech 张量失败: {e}")))?
            .into_dyn();
        let speech_lengths = Value::from_array((vec![1_i64], vec![rows as i32]))
            .map_err(|e| Error::Inference(format!("构造 speech_lengths 张量失败: {e}")))?
            .into_dyn();
        let language = Value::from_array((vec![1_i64], vec![lid]))
            .map_err(|e| Error::Inference(format!("构造 language 张量失败: {e}")))?
            .into_dyn();
        let textnorm = Value::from_array((vec![1_i64], vec![tn]))
            .map_err(|e| Error::Inference(format!("构造 textnorm 张量失败: {e}")))?
            .into_dyn();

        let outputs = self
            .session
            .run(vec![
                (self.input_names[0].clone(), speech),
                (self.input_names[1].clone(), speech_lengths),
                (self.input_names[2].clone(), language),
                (self.input_names[3].clone(), textnorm),
            ])
            .map_err(|e| Error::Inference(format!("SenseVoice 推理失败: {e}")))?;
        if outputs.len() < OUTPUT_COUNT {
            return Err(Error::Inference(format!(
                "SenseVoice 输出个数 {}，期望 {OUTPUT_COUNT}",
                outputs.len()
            )));
        }

        // 输出 1 = ctc_lengths（也可能是 int64）；取第一个元素作有效帧数
        let enc_len = match outputs[1].try_extract_scalar::<i32>() {
            Ok(v) => v,
            Err(_) => outputs[1]
                .try_extract_scalar::<i64>()
                .map_err(|e| Error::Inference(format!("读取 ctc_lengths 失败: {e}")))?
                as i32,
        };
        if enc_len < 0 {
            return Err(Error::Inference(format!("ctc_lengths 为负: {enc_len}")));
        }
        let enc_len = enc_len as usize;

        // 输出 0 = ctc_logits，形状 [1, enc_len, vocab]
        let (shape, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Inference(format!("读取 ctc_logits 失败: {e}")))?;
        let dims: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
        if dims.len() != 3 || dims[0] == 0 || dims[2] == 0 {
            return Err(Error::Inference(format!(
                "SenseVoice 输出维度异常: {dims:?}"
            )));
        }
        let vocab = dims[2];
        if logits.len() < dims[1] * vocab {
            return Err(Error::Inference(format!(
                "SenseVoice 输出大小异常: {} < {}",
                logits.len(),
                dims[1] * vocab
            )));
        }
        // Python：ctc_logits[0]（batch=0 的 [enc_len, vocab] 平面）
        let pairs = ctc_greedy_with_frames(logits, vocab, enc_len, BLANK_ID)?;
        Ok((pairs, enc_len, vocab))
    }
}

impl AsrEngine for SenseVoice {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::SenseVoice
    }

    fn transcribe_chunk(
        &mut self,
        chunk: &[i16],
        sample_rate: u32,
        job: &JobOptions,
    ) -> Result<AsrChunkOutput> {
        // extract_feat：fbank → LFR → CMVN（顺序与 vendored 源码一致；CMVN 见 Cmvn64 说明）
        let lfr = extract_features(chunk, sample_rate, &self.cfg, None)?;
        let feats = apply_cmvn_f64(&lfr.feats, &self.cmvn)?;

        // read_tags：language → lid，textnorm 固定 withitn
        let lid = lid_for(job.language)?;
        let tn = textnorm_for_name("withitn")?;
        let (pairs, enc_len, _vocab) = self.infer_pairs(&feats, lid, tn)?;

        // 段内相对时间：段首 0、段尾 = 本段时长（VAD span 为 10ms 整数倍时与 Python 的
        // 绝对 span 等价，见模块文档）
        let span_end_ms = chunk_duration_ms(chunk.len(), sample_rate);
        let mut out = Vec::with_capacity(pairs.len());
        for (token_id, frame) in pairs {
            let raw = self.pieces.id_to_piece(token_id)?;
            if is_special_token(raw) {
                continue; // <|lang|><|emo|><|event|><|itn|> 等特殊标记
            }
            let text = raw.replace('▁', " "); // ▁ -> 空格（CJK 无 ▁，自然紧贴）
            if text.is_empty() {
                continue;
            }
            out.push(TimedPiece {
                text,
                time_ms: token_time_ms(frame, enc_len, 0, span_end_ms),
            });
        }
        Ok(AsrChunkOutput::Timed(out))
    }
}

/// 采样数 → 段时长（毫秒，向下取整）。16k 下即 `samples / 16`。
fn chunk_duration_ms(samples: usize, sample_rate: u32) -> i64 {
    if sample_rate == 0 {
        return 0;
    }
    (samples as i64) * 1000 / i64::from(sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_token_matches_reference_regex() {
        for raw in ["<|zh|>", "<|NEUTRAL|>", "<|Speech|>", "<|withitn|>", "<||>"] {
            assert!(is_special_token(raw), "{raw} 应判为特殊标记");
        }
        for raw in ["今", "▁the", "<|zh|>x", "x<|zh|>", "<|zh", "zh|>", ""] {
            assert!(!is_special_token(raw), "{raw} 不应判为特殊标记");
        }
    }

    #[test]
    fn lid_and_textnorm_tables_follow_source() {
        assert_eq!(lid_for_name("auto").ok(), Some(0));
        assert_eq!(lid_for_name("zh").ok(), Some(3));
        assert_eq!(lid_for_name("en").ok(), Some(4));
        assert_eq!(lid_for_name("yue").ok(), Some(7));
        assert_eq!(lid_for_name("ja").ok(), Some(11));
        assert_eq!(lid_for_name("ko").ok(), Some(12));
        assert_eq!(lid_for_name("nospeech").ok(), Some(13));
        assert!(lid_for_name("fr").is_err());
        assert_eq!(textnorm_for_name("withitn").ok(), Some(14));
        assert_eq!(textnorm_for_name("woitn").ok(), Some(15));
        assert!(textnorm_for_name("itn").is_err());
        assert_eq!(lid_for(Language::Auto).ok(), Some(0));
        assert_eq!(lid_for(Language::Zh).ok(), Some(3));
        assert_eq!(lid_for(Language::Yue).ok(), Some(7));
        assert_eq!(lid_for(Language::En).ok(), Some(4));
        assert_eq!(lid_for(Language::Ja).ok(), Some(11));
        assert_eq!(lid_for(Language::Ko).ok(), Some(12));
    }

    /// 手工拼一个最小 ModelProto（4 个 piece），校验 protobuf 解析与 id 顺序。
    #[test]
    fn sp_model_parses_pieces_in_order() {
        fn put_varint(out: &mut Vec<u8>, mut v: u64) {
            loop {
                let byte = (v & 0x7f) as u8;
                v >>= 7;
                if v == 0 {
                    out.push(byte);
                    return;
                }
                out.push(byte | 0x80);
            }
        }
        fn push_piece(out: &mut Vec<u8>, piece: &str, score: f32, kind: u64) {
            let mut sub = Vec::new();
            sub.push(0x0a);
            put_varint(&mut sub, piece.len() as u64);
            sub.extend_from_slice(piece.as_bytes());
            // field 2 (score), wire 5 (fixed32)
            sub.push(0x15);
            sub.extend_from_slice(&score.to_le_bytes());
            // field 3 (type), wire 0 (varint)
            sub.push(0x18);
            put_varint(&mut sub, kind);
            out.push(0x0a);
            put_varint(out, sub.len() as u64);
            out.extend_from_slice(&sub);
        }

        let mut proto = Vec::new();
        push_piece(&mut proto, "<unk>", 0.0, 2);
        push_piece(&mut proto, "<s>", 0.0, 3);
        push_piece(&mut proto, "</s>", 0.0, 3);
        push_piece(&mut proto, "▁the", -1.5, 1);
        // 末尾塞一个无关字段（trainer_spec, field 2 length-delimited）验证跳过
        proto.push(0x12);
        put_varint(&mut proto, 3);
        proto.extend_from_slice(&[0x08, 0x01, 0x10]);

        let model = match SpModel::parse(&proto) {
            Ok(m) => m,
            Err(e) => panic!("解析最小 ModelProto 失败: {e}"),
        };
        assert_eq!(model.len(), 4);
        assert!(!model.is_empty());
        assert_eq!(model.id_to_piece(0).ok(), Some("<unk>"));
        assert_eq!(model.id_to_piece(3).ok(), Some("▁the"));
        assert!(model.id_to_piece(4).is_err());
    }

    #[test]
    fn sp_model_rejects_garbage() {
        assert!(SpModel::parse(&[]).is_err(), "空文件应报错");
        assert!(SpModel::parse(&[0x0a, 0x05, 0x0a]).is_err(), "截断应报错");
        // 只有一个 piece：不足 MIN_PIECES
        let mut proto = Vec::new();
        proto.extend_from_slice(&[0x0a, 0x07, 0x0a, 0x05]);
        proto.extend_from_slice(b"<unk>");
        assert!(SpModel::parse(&proto).is_err());
    }

    #[test]
    fn ctc_greedy_keeps_first_frame_and_skips_blank() {
        // 4 帧、3 类：词表 A B <blank=0>
        // 帧0: blank, 帧1: A, 帧2: A, 帧3: B
        let logits = vec![
            1.0, 0.0, 0.0, // blank
            0.0, 1.0, 0.0, // A
            0.0, 2.0, 0.0, // A
            0.0, 0.0, 1.0, // B
        ];
        let pairs = match ctc_greedy_with_frames(&logits, 3, 4, 0) {
            Ok(p) => p,
            Err(e) => panic!("CTC 解码失败: {e}"),
        };
        assert_eq!(pairs, vec![(1, 1), (2, 3)]);

        // valid_len 截断：只看前 2 帧 → 只有 A@1
        let head = match ctc_greedy_with_frames(&logits, 3, 2, 0) {
            Ok(p) => p,
            Err(e) => panic!("CTC 解码失败: {e}"),
        };
        assert_eq!(head, vec![(1, 1)]);

        // valid_len 超过实际行数时按行数截断（numpy 语义）
        let over = match ctc_greedy_with_frames(&logits, 3, 99, 0) {
            Ok(p) => p,
            Err(e) => panic!("CTC 解码失败: {e}"),
        };
        assert_eq!(over, vec![(1, 1), (2, 3)]);
    }

    #[test]
    fn ctc_greedy_blank_between_same_token_splits() {
        // A, blank, A → 两个 A（_ctc_greedy_with_frames 用 prev=blank 起始）
        let logits = vec![
            0.0, 1.0, 0.0, // A
            1.0, 0.0, 0.0, // blank
            0.0, 1.0, 0.0, // A
        ];
        let pairs = match ctc_greedy_with_frames(&logits, 3, 3, 0) {
            Ok(p) => p,
            Err(e) => panic!("CTC 解码失败: {e}"),
        };
        assert_eq!(pairs, vec![(1, 0), (1, 2)]);
    }

    #[test]
    fn ctc_greedy_rejects_bad_inputs() {
        assert!(ctc_greedy_with_frames(&[], 0, 0, 0).is_err());
        assert!(ctc_greedy_with_frames(&[0.0, 1.0], 2, 1, 5).is_err());
        assert!(ctc_greedy_with_frames(&[0.0, 1.0, 2.0], 2, 1, 0).is_err());
    }

    #[test]
    fn token_time_maps_frames_linearly() {
        // T=102 → speech_frames=98，span 0..5880：frame 7 → int(3/98*5880)=180
        assert_eq!(token_time_ms(7, 102, 0, 5880), 180);
        assert_eq!(token_time_ms(102, 102, 0, 5880), 5880);
        // 前缀帧（frame < 4）夹到段首
        assert_eq!(token_time_ms(0, 102, 6170, 9730), 6170);
        // T <= 前缀帧时 speech_frames 兜底为 1，帧号减前缀可能为负 → 夹到段首
        assert_eq!(token_time_ms(4, 4, 100, 200), 100);
        assert_eq!(token_time_ms(5, 4, 100, 200), 200);
        assert_eq!(token_time_ms(0, 2, 100, 200), 100);
    }

    #[test]
    fn chunk_duration_floors_to_ms() {
        assert_eq!(chunk_duration_ms(94_080, 16_000), 5880);
        assert_eq!(chunk_duration_ms(1, 16_000), 0);
        assert_eq!(chunk_duration_ms(16, 0), 0);
    }

    fn write_cmvn(text: &str) -> Result<(tempfile::TempDir, std::path::PathBuf)> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("am.mvn");
        std::fs::write(&path, text)?;
        Ok((dir, path))
    }

    /// am.mvn 要按 float64 读：这些值存成 f32 会各差不到 1 ULP，实测足以翻转 CTC 帧。
    #[test]
    fn load_cmvn_f64_keeps_double_precision() {
        let text = "<Nnet> \n<Splice> 3 3\n[ 0 ]\n<AddShift> 3 3 \n\
                    <LearnRateCoef> 0 [ -1.5000000000000002 0.25 2 ] \n<Rescale> 3 3\n\
                    <LearnRateCoef> 0 [ 0.5000000000000001 2 0.125 ] \n";
        let (_dir, path) = match write_cmvn(text) {
            Ok(v) => v,
            Err(e) => panic!("准备 am.mvn 失败: {e}"),
        };
        let cmvn = match load_cmvn_f64(&path) {
            Ok(c) => c,
            Err(e) => panic!("load_cmvn_f64 失败: {e}"),
        };
        assert_eq!(cmvn.mean, vec![-1.500_000_000_000_000_2, 0.25, 2.0]);
        assert_eq!(cmvn.istd, vec![0.500_000_000_000_000_1, 2.0, 0.125]);
        // f32 存不下这些尾数
        assert!(cmvn.mean[0] as f32 as f64 != cmvn.mean[0]);
    }

    #[test]
    fn apply_cmvn_f64_matches_reference_formula() {
        let feats = Array2::from_shape_vec((2, 2), vec![1.0f32, 2.0, 3.0, 4.0])
            .unwrap_or_else(|e| panic!("构造特征失败: {e}"));
        let cmvn = Cmvn64 {
            mean: vec![-1.5, 0.25],
            istd: vec![0.5, 2.0],
        };
        let out = match apply_cmvn_f64(&feats, &cmvn) {
            Ok(o) => o,
            Err(e) => panic!("apply_cmvn_f64 失败: {e}"),
        };
        // (1-1.5)*0.5 = -0.25, (2+0.25)*2 = 4.5, (3-1.5)*0.5 = 0.75, (4+0.25)*2 = 8.5
        assert_eq!(out.row(0).to_vec(), vec![-0.25, 4.5]);
        assert_eq!(out.row(1).to_vec(), vec![0.75, 8.5]);

        // 维度不足要报错
        let narrow = Cmvn64 {
            mean: vec![0.0],
            istd: vec![1.0],
        };
        assert!(apply_cmvn_f64(&feats, &narrow).is_err());
    }

    #[test]
    fn load_cmvn_f64_rejects_broken_files() {
        assert!(load_cmvn_f64(Path::new("definitely/not/here.mvn")).is_err());
        let (_d, p) = match write_cmvn("<AddShift> 2 2\n<LearnRateCoef> 0 [ 1 ]\n") {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        assert!(load_cmvn_f64(&p).is_err(), "缺 <Rescale> 应报错");
    }
}
