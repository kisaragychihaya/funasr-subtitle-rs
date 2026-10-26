//! T10：转写管线编排。复刻 funasr-subtitle `engine/funasr_engine.py` 的
//! FunasrEngine.transcribe 全编排（VAD → 逐段 ASR → 标点三模式 → 切句 → 单调化）。
//!
//! 逐行对照：
//! * [`Engine::new`] ← `FunasrEngine.__init__`：只存配置 + 模型描述，不建任何 session。
//! * `Engine::ensure_loaded` ← `FunasrEngine.load`：VAD + ASR 在首次转写时才加载；
//!   ASR **不自带标点**（Paraformer / Contextual）时顺带把 ct-punc 也加载好。
//! * [`Engine::transcribe`] ← `FunasrEngine.transcribe`：VAD 切段 → 逐段 ASR →
//!   标点三分支 → `segment_timed` / `segment_span` → `enforce_monotonic`。
//! * `Engine::punctuate` ← `_punctuate`：**懒加载** ct-punc（SenseVoice 只有真需要补标点
//!   时才下载/加载 PUNC_MODEL），失败不致命（退回原文）。
//! * `Engine::latin_punct` ← `_latin_punct`：英文**原生标点优先**，只有原生没有句末标点才用
//!   ct-punc 兜底（zh ct-punc 会把英文句号插在错误位置）。
//! * [`bare_timed`] ← `transcribe` 里英文分支的 bare 文本构造（逐字去掉 `_PUNCT_STRIP`）。
//! * `latin_punct` / `align_punct_to_timed` / `strip_terminal_timed` 保住真实 CTC 时间戳。
//! * 说话人分离（`job.diarization`）← 上游 `inference_with_vad` 的 `spk_model="cam++"` 分支：
//!   VAD 段 → `speaker::diarize::sv_chunk` 切窗 → cam++ 嵌入 → `speaker::cluster::cluster` →
//!   `speaker::diarize::postprocess` → `speaker::diarize::distribute_spk` 得到每段说话人，
//!   再给该 VAD 段产出的每条 cue 打上 `spk{n}`。

use std::path::PathBuf;

use crate::asr::{AsrChunkOutput, AsrEngine, ContextualParaformer, Paraformer, SenseVoice};
use crate::error::{Error, Result};
use crate::media::Audio;
use crate::models::manager::ModelManager;
use crate::models::{ModelSpec, ModelTask, PUNC_MODEL, SPK_MODEL, VAD_MODEL, get_model};
use crate::punc::{Punctuator, align_punct_to_timed};
use crate::speaker::cluster::cluster;
use crate::speaker::diarize::{SpkChunk, distribute_spk, postprocess, sv_chunk};
use crate::speaker::embed::SpeakerEmbedder;
use crate::text::normalize::{
    has_cjk, has_terminal, normalize_latin_punct, py_trim, strip_terminal_timed,
};
use crate::text::{enforce_monotonic, segment_span, segment_timed, strip_terminal_punct};
use crate::types::{EngineKind, JobOptions, Language, PunctMode, Segment, TimedPiece};
use crate::vad::Vad;

/// 引擎固定 16kHz（`FunasrEngine.SAMPLE_RATE`）。
const SAMPLE_RATE: u32 = 16_000;
/// 短于这个采样数的 VAD 段直接跳过：Python 侧 `chunk.size < SAMPLE_RATE * 0.05`（= 50ms）。
const MIN_CHUNK_SAMPLES: usize = (SAMPLE_RATE as usize) / 20;
/// `opts.num_threads or 4`：0 表示用默认 4 线程。
const DEFAULT_THREADS: usize = 4;
/// `_PUNCT_STRIP = set("。，！？!?…；;：:、,.")`：英文重标点前先摘掉的零星标点。
const PUNCT_STRIP: [char; 14] = [
    '。', '，', '！', '？', '!', '?', '…', '；', ';', '：', ':', '、', ',', '.',
];

/// 引擎构造选项。
pub struct EngineOptions {
    /// 模型 id（registry 中的稳定 id，默认 DEFAULT_MODEL_ID）。
    pub model_id: String,
    pub models_root: PathBuf,
    pub num_threads: usize,
}

/// 转写引擎（懒加载：首次 transcribe 时才下载/加载模型）。
pub struct Engine {
    opts: EngineOptions,
    /// registry 里的模型描述（`get_model(opts.model_id)`，构造时就确定）。
    spec: &'static ModelSpec,
    manager: ModelManager,
    vad: Option<Vad>,
    asr: Option<Box<dyn AsrEngine>>,
    /// ct-punc；`None` 表示还没加载（懒加载，见 `Engine::punctuate`）。
    punc: Option<Punctuator>,
    /// cam++ 说话人嵌入；`None` 表示还没加载（懒加载，见 `Engine::ensure_spk`）。
    spk: Option<SpeakerEmbedder>,
    loaded: bool,
}

/// 进度回调：stage ∈ vad/asr/punc/assembling，percent ∈ `[0,100]`。
pub type ProgressFn<'a> = &'a dyn Fn(&str, f32);

// ---------------------------------------------------------------------------
// 纯逻辑辅助（可单独单测）
// ---------------------------------------------------------------------------

/// 语言代码 -> [`Language`]；未知代码返回 `None`（对应 `_valid_langs` 的合法性判定）。
pub fn language_from_code(code: &str) -> Option<Language> {
    match code {
        "auto" => Some(Language::Auto),
        "zh" => Some(Language::Zh),
        "yue" => Some(Language::Yue),
        "en" => Some(Language::En),
        "ja" => Some(Language::Ja),
        "ko" => Some(Language::Ko),
        _ => None,
    }
}

/// 语言代码 -> [`Language`]，非法代码回退 `auto`（脚本 `lang = options.language if ... else "auto"`）。
pub fn resolve_language(code: &str) -> Language {
    language_from_code(code).unwrap_or(Language::Auto)
}

/// 标点模式代码 -> [`PunctMode`]；未知代码返回 `None`。
pub fn punct_mode_from_code(code: &str) -> Option<PunctMode> {
    match code {
        "auto" => Some(PunctMode::Auto),
        "on" => Some(PunctMode::On),
        "off" => Some(PunctMode::Off),
        _ => None,
    }
}

/// 标点模式代码 -> [`PunctMode`]，非法代码回退 `auto`。
pub fn resolve_punct_mode(code: &str) -> PunctMode {
    punct_mode_from_code(code).unwrap_or(PunctMode::Auto)
}

/// SenseVoice 路径的标点三分支（`transcribe` 里的 if / elif / elif 三条）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PunctAction {
    /// `add_punct && !is_cjk`：原生拉丁标点优先，缺句末标点才用 ct-punc 兜底。
    Latin,
    /// `add_punct && is_cjk && !has_terminal`：用 ct-punc 补齐句末标点。
    FillCjk,
    /// `!add_punct`：删句末终止标点，只按停顿 + 行宽切。
    Strip,
}

/// 选择标点分支；`None` = 原生文本已带句末标点（中文），什么都不用做。
///
/// 与脚本一致：只要 `add_punct` 为真且非 CJK 就走 [`PunctAction::Latin`]（内部再判断原生
/// 标点），CJK 且已有终止标点时保持原样。空 `timed` 由调用方先排除。
pub fn punct_action(
    add_punct: bool,
    is_cjk: bool,
    native_has_terminal: bool,
) -> Option<PunctAction> {
    if !add_punct {
        return Some(PunctAction::Strip);
    }
    if !is_cjk {
        return Some(PunctAction::Latin);
    }
    if !native_has_terminal {
        return Some(PunctAction::FillCjk);
    }
    None
}

/// 英文分支的 bare 文本：逐字去掉 `PUNCT_STRIP` 里的标点，保留每个字符的真实时间。
pub fn bare_timed(timed: &[TimedPiece]) -> Vec<TimedPiece> {
    let mut out = Vec::with_capacity(timed.len());
    for piece in timed {
        let text: String = piece
            .text
            .chars()
            .filter(|c| !PUNCT_STRIP.contains(c))
            .collect();
        if !text.is_empty() {
            out.push(TimedPiece {
                text,
                time_ms: piece.time_ms,
            });
        }
    }
    out
}

/// 是否给本段加句末标点：on=都加；off=都不加；auto=中文加、英文按停顿不加。
fn should_add_punct(mode: PunctMode, is_cjk: bool) -> bool {
    match mode {
        PunctMode::On => true,
        PunctMode::Auto => is_cjk,
        PunctMode::Off => false,
    }
}

/// `int(ms / 1000 * SAMPLE_RATE)`（毫秒 -> 采样下标，向下取整）。
fn samples_at_ms(ms: i64) -> usize {
    if ms <= 0 {
        return 0;
    }
    (ms as f64 / 1000.0 * f64::from(SAMPLE_RATE)) as usize
}

/// `int(len(wav) / SAMPLE_RATE * 1000)`：音频总时长（毫秒）。
fn duration_ms(samples: usize) -> i64 {
    (samples as f64 / f64::from(SAMPLE_RATE) * 1000.0) as i64
}

// ---------------------------------------------------------------------------
// 引擎
// ---------------------------------------------------------------------------

impl Engine {
    pub fn new(opts: EngineOptions) -> Result<Self> {
        let spec = get_model(&opts.model_id)?;
        if spec.task != ModelTask::Asr {
            return Err(Error::InvalidInput(format!(
                "模型 {} 不是 ASR 模型（task = {:?}）",
                spec.id, spec.task
            )));
        }
        let manager = ModelManager::new(opts.models_root.clone());
        Ok(Self {
            opts,
            spec,
            manager,
            vad: None,
            asr: None,
            punc: None,
            spk: None,
            loaded: false,
        })
    }

    /// 线程数：`opts.num_threads or 4`。
    fn threads(&self) -> usize {
        if self.opts.num_threads == 0 {
            DEFAULT_THREADS
        } else {
            self.opts.num_threads
        }
    }

    /// `FunasrEngine.load`：确保模型就绪并建 session（幂等）。
    fn ensure_loaded(&mut self) -> Result<()> {
        if self.loaded {
            return Ok(());
        }
        let threads = self.threads();
        let vad_dir = self.manager.ensure_model(&VAD_MODEL)?;
        self.vad = Some(Vad::new(&vad_dir, threads)?);

        let asr_dir = self.manager.ensure_model(self.spec)?;
        let asr: Box<dyn AsrEngine> = match self.spec.engine {
            EngineKind::SenseVoice => Box::new(SenseVoice::new(&asr_dir, threads)?),
            EngineKind::Paraformer => Box::new(Paraformer::new(&asr_dir, threads)?),
            EngineKind::Contextual => Box::new(ContextualParaformer::new(&asr_dir, threads)?),
        };
        self.asr = Some(asr);

        // 仅当 ASR 不自带标点时才加载 ct-punc（省一次 ~290MB 下载）
        if !self.spec.builtin_punc {
            self.load_punc()?;
        }
        self.loaded = true;
        Ok(())
    }

    /// 建 ct-punc（`ensure_model` 缺则下载）。
    fn load_punc(&mut self) -> Result<()> {
        if self.punc.is_some() {
            return Ok(());
        }
        let threads = self.threads();
        let dir = self.manager.ensure_model(&PUNC_MODEL)?;
        self.punc = Some(Punctuator::new(&dir, threads)?);
        Ok(())
    }

    /// 建 cam++ 嵌入器（`ensure_model` 缺则从 GitHub 直链下载，约 27MB）。
    ///
    /// 只在 `job.diarization` 为真时懒加载：不开说话人分离就不下载。
    fn ensure_spk(&mut self) -> Result<()> {
        if self.spk.is_some() {
            return Ok(());
        }
        let threads = self.threads();
        let dir = self.manager.ensure_model(&SPK_MODEL)?;
        self.spk = Some(SpeakerEmbedder::new(&dir, threads)?);
        Ok(())
    }

    /// 逐 sv_chunk 求 cam++ 嵌入。
    ///
    /// 单独成方法是为了把 `self.spk` 的可变借用限制在这次调用内：
    /// ASR 循环里会同时用到 `self.asr`，两个字段不能同时可变借用。
    fn embed_chunks(&mut self, chunks: &[SpkChunk]) -> Result<Vec<Vec<f32>>> {
        let spk = self
            .spk
            .as_mut()
            .ok_or_else(|| Error::Inference("CAM++ 未加载".to_string()))?;
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            embeddings.push(spk.embed(&chunk.samples)?);
        }
        Ok(embeddings)
    }

    /// `_punctuate`：懒加载 ct-punc 后给文本补标点；**失败不致命，退回原文**。
    ///
    /// 对齐 Python 语义：`_punctuate` 用 `except Exception: return text` 把模型下载/加载/
    /// 推理的任何异常都兜成「原文」，这里同样把 [`Err`] 吞掉（含懒加载失败）。
    fn punctuate(&mut self, text: &str) -> String {
        match self.punctuate_inner(text) {
            Ok(out) => out,
            Err(_) => text.to_string(),
        }
    }

    fn punctuate_inner(&mut self, text: &str) -> Result<String> {
        self.load_punc()?;
        match self.punc.as_mut() {
            Some(punc) => punc.punctuate(text),
            None => Ok(text.to_string()),
        }
    }

    /// `_latin_punct`：原生（SenseVoice）标点优先，仅当原生无句末标点才用 ct-punc 兜底。
    fn latin_punct(&mut self, text: &str, bare_text: &str) -> String {
        let native = normalize_latin_punct(text);
        if has_terminal(&native) {
            return native;
        }
        normalize_latin_punct(&self.punctuate(bare_text))
    }

    /// 识别一个 VAD 段（16k 单声道 s16）。
    fn asr_chunk(&mut self, chunk: &[i16], job: &JobOptions) -> Result<AsrChunkOutput> {
        let asr = self
            .asr
            .as_mut()
            .ok_or_else(|| Error::Inference("ASR 引擎未加载".to_string()))?;
        asr.transcribe_chunk(chunk, SAMPLE_RATE, job)
    }

    /// 整段转写。progress 回调 (stage, percent)。
    pub fn transcribe(
        &mut self,
        audio: &Audio,
        job: &JobOptions,
        progress: Option<ProgressFn<'_>>,
    ) -> Result<Vec<Segment>> {
        self.ensure_loaded()?;
        if audio.sample_rate != SAMPLE_RATE {
            return Err(Error::InvalidInput(format!(
                "期望 {SAMPLE_RATE} Hz 音频，实际 {} Hz",
                audio.sample_rate
            )));
        }
        let report = |stage: &str, pct: f32| {
            if let Some(cb) = progress {
                cb(stage, pct);
            }
        };

        // 1) VAD -> [(start_ms, end_ms), ...]
        report("vad", 0.0);
        let vad = self
            .vad
            .as_mut()
            .ok_or_else(|| Error::Inference("VAD 未加载".to_string()))?;
        let mut spans = vad.infer(&audio.samples, audio.sample_rate)?;
        if spans.is_empty() {
            // 没切出语音：把整段作为一个片段兜底
            spans.push((0, duration_ms(audio.samples.len())));
        }
        report("vad", 100.0);

        // 1b) 说话人分离 -> 每个 VAD 段的说话人编号（`spk_model="cam++"` 分支）
        //
        // 上游 full 引擎只在 Paraformer 系开说话人分离（`app/engine/funasr_full_engine.py`：
        // `self._diar = bool(self.opts.diarization) and not self._builtin_punc`）；
        // Rust 侧是超集，对全部引擎生效（SenseVoice 也能给 `[spkN]` 前缀）。
        let span_speakers: Vec<usize> = if job.diarization {
            report("spk", 0.0);
            self.ensure_spk()?;
            let chunks = sv_chunk(&spans, &audio.samples);
            let embeddings = self.embed_chunks(&chunks)?;
            let labels = cluster(&embeddings, job.preset_spk_num);
            let timeline = postprocess(&chunks, &labels);
            let speakers = distribute_spk(&spans, &timeline);
            report("spk", 100.0);
            speakers
        } else {
            Vec::new()
        };

        // 2) 逐段 ASR -> 标点 -> 按自然语句切分（每段产出多条字幕）
        let engine_kind = self.spec.engine;
        let punct_mode = job.punct;
        let max_chars = job.max_chars; // 0=不限制；0 是合法值，不能回退默认
        let max_cue_ms = job.max_cue_ms;
        let n = spans.len();
        let mut segments: Vec<Segment> = Vec::new();

        for (i, (start_ms, end_ms)) in spans.iter().copied().enumerate() {
            let a = samples_at_ms(start_ms);
            let b = samples_at_ms(end_ms).min(audio.samples.len());
            if b <= a {
                continue;
            }
            let chunk = &audio.samples[a..b];
            if chunk.len() < MIN_CHUNK_SAMPLES {
                continue; // <50ms，跳过
            }
            let done_pct = (i + 1) as f32 / n as f32 * 100.0;
            // 本段产出的字幕区间起点：段末再按 span 序号打说话人标签
            let before = segments.len();

            if engine_kind == EngineKind::SenseVoice {
                // SenseVoice：CTC 帧位置推出**真实** token 时间，按自然语句切句
                let rel = match self.asr_chunk(chunk, job)? {
                    AsrChunkOutput::Timed(pieces) => pieces,
                    AsrChunkOutput::Plain(_) => {
                        return Err(Error::Inference(
                            "SenseVoice 引擎应返回带时间戳的文本片".to_string(),
                        ));
                    }
                };
                // 引擎给的是**段内相对**毫秒，加上段起点才是绝对时间（等价性见 T8）
                let mut timed: Vec<TimedPiece> = rel
                    .into_iter()
                    .map(|p| TimedPiece {
                        text: p.text,
                        time_ms: p.time_ms + start_ms,
                    })
                    .collect();
                report("asr", done_pct);

                let text: String = timed.iter().map(|p| p.text.as_str()).collect();
                let is_cjk = has_cjk(&text);
                let add_punct = should_add_punct(punct_mode, is_cjk);
                if !timed.is_empty() {
                    match punct_action(add_punct, is_cjk, has_terminal(&text)) {
                        Some(PunctAction::Latin) => {
                            let bare = bare_timed(&timed);
                            let bare_text: String = bare.iter().map(|p| p.text.as_str()).collect();
                            let punct = self.latin_punct(&text, &bare_text);
                            report("punc", done_pct);
                            // 兜底也拿不到句末标点时保留 bare（已去零星标点的版本）
                            timed = if has_terminal(&punct) {
                                align_punct_to_timed(&bare, &punct)
                            } else {
                                bare
                            };
                        }
                        Some(PunctAction::FillCjk) => {
                            let punct = self.punctuate(&text);
                            report("punc", done_pct);
                            if has_terminal(&punct) {
                                timed = align_punct_to_timed(&timed, &punct);
                            }
                        }
                        Some(PunctAction::Strip) => {
                            timed = strip_terminal_timed(&timed);
                        }
                        None => {}
                    }
                }
                segments.extend(segment_timed(&timed, end_ms, max_chars, max_cue_ms));
            } else {
                // Paraformer / Contextual：无逐 token 时间，先标点，再按字数比例切句
                let text = match self.asr_chunk(chunk, job)? {
                    AsrChunkOutput::Plain(text) => py_trim(&text).to_string(),
                    AsrChunkOutput::Timed(_) => {
                        return Err(Error::Inference(
                            "Paraformer/Contextual 引擎应返回整句文本".to_string(),
                        ));
                    }
                };
                report("asr", done_pct);
                if text.is_empty() {
                    continue;
                }
                let is_cjk = has_cjk(&text);
                let add_punct = should_add_punct(punct_mode, is_cjk);
                let text = if add_punct {
                    self.punctuate(&text)
                } else {
                    strip_terminal_punct(&text)
                };
                report("punc", done_pct);
                segments.extend(segment_span(&text, start_ms, end_ms, max_chars));
            }
            // 本段产出的每条字幕挂说话人（`span_speakers` 按 span 序号对齐；
            // 被 `continue` 跳过的段没有字幕，索引仍按 span 序号而不是字幕序号）
            if job.diarization {
                let speaker = format!("spk{}", span_speakers[i]);
                for seg in &mut segments[before..] {
                    seg.speaker = Some(speaker.clone());
                }
            }
        }
        report("punc", 100.0);

        report("assembling", 100.0);
        enforce_monotonic(&mut segments);
        Ok(segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_at_ms_matches_python_int_cast() {
        assert_eq!(samples_at_ms(0), 0);
        assert_eq!(samples_at_ms(-5), 0);
        assert_eq!(samples_at_ms(180), 2880);
        assert_eq!(samples_at_ms(9730), 155_680);
    }

    #[test]
    fn duration_ms_matches_python_int_cast() {
        assert_eq!(duration_ms(16_000), 1000);
        assert_eq!(duration_ms(0), 0);
        assert_eq!(duration_ms(155_680), 9730);
    }

    #[test]
    fn unknown_codes_fall_back_to_defaults() {
        assert_eq!(resolve_language("tlh"), Language::Auto);
        assert_eq!(resolve_language("zh"), Language::Zh);
        assert_eq!(resolve_punct_mode("maybe"), PunctMode::Auto);
        assert_eq!(resolve_punct_mode("off"), PunctMode::Off);
    }
}
