//! T7：FSMN-VAD。移植来源：funasr-onnx vendored `vad_bin.py` + `utils/e2e_vad.py`。
//!
//! 对应关系（逐行对照，不做「修正」）：
//! * [`Vad::new`] ← `Fsmn_vad.__init__`：读模型 `config.yaml` 的三段配置、`am.mvn`（有就
//!   `load_cmvn`）、`model_quant.onnx`（funasr 侧固定 `quantize=True`）建 ONNX session。
//! * [`Vad::infer`] ← `Fsmn_vad.__call__` 的 batch=1 离线路径：`fbank → lfr → cmvn` →
//!   `min(feats_len, 6000)` 帧分块送 ONNX（带 4 个 FSMN 状态 cache）→ 每块交
//!   [`E2EVad::call`] 后处理 → 累加输出段。
//! * [`E2EVad`] ← `E2EVadModel`：逐帧判决、滑窗状态机、段起止点、最大段长切割。
//! * [`WindowDetector`] ← `WindowDetector`：语音/静音滑窗（20 帧 = 200ms）累计。
//!
//! 参数全部照抄模型 `config.yaml`（ModelScope `iic/speech_fsmn_vad_zh-cn-16k-common-onnx`，
//! 本文件用常量钉住，理由同 `frontend` 的前端预设：参考实现只有这一套取值）：
//! * `frontend_conf`: `n_mels 80` / `frame_length 25` / `frame_shift 10` / `dither 0.0` /
//!   `lfr_m 5` / `lfr_n 1`（lfr_n=1 → 一个特征帧正好 10ms = 160 采样，分块切波形靠这个）。
//! * `model_conf`: 见 [`VadOpts::CONFIG`] 的逐字段注释（`max_single_segment_time` 是 20000，
//!   不是 `VADXOptions` 的默认 60000）。
//! * `encoder_conf`: `fsmn_layers 4` / `proj_dim 128` / `lorder 20` → cache 形状
//!   `[1, 128, 19, 1]`；ONNX 输入 `speech` + `in_cache0..3`，输出 `logits` + `out_cache0..3`。
//!
//! 故意不移植的部分（对 `Vad::infer` 的输出没有影响，逐条注明原因）：
//! * `Fsmn_vad_online`（流式类）与 `E2EVadModel.__call__` 的 `online=True` 分支、`next_seg`
//!   状态：离线整段推理恒走 `online=False`。
//! * `E2EVadSpeechBufWithDoa.buffer`/`doa` 与 `PopDataToOutputBuf` 里的采样拷贝算术
//!   （`expected_sample_number`/`data_to_pop`）：段输出只用 `start_ms`/`end_ms`，采样拷进
//!   buffer 后从不读回。保留了对状态有影响的 `data_buf_start_frame`/`data_buf_size`。
//! * `output_frame_probs`/`frame_probs`（config 里 `False`）、`voice_last_frame_count` 等三个
//!   只被 `Reset` 置零的计数器、`is_final`/`max_time_out`/`sil_frame`/
//!   `pre_end_silence_detected`（只写不读）以及 `print("error ...")` 诊断。
//! * `do_start_point_detection`/`do_end_point_detection`（源码里是 `pass`）。
//!
//! 上游怪癖一律保留：`GetFrameState` 在 `decibel < decibel_thres` 时**自己**调一次
//! `DetectOneFrame` 再返回，调用方（`DetectCommonFrames`/`DetectLastFrames`）又调一次，
//! 这种帧因此被处理两遍。
//!
//! 唯一的语义偏差（源码那里是死循环，不可能照抄）：`PopDataBufTillFrame` 里 `data_buf_size`
//! 不足一帧时 `break` 而不是原地打转，见该函数注释。

use std::path::Path;
use std::sync::OnceLock;

use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;

use crate::error::{Error, Result};
use crate::frontend::{Cmvn, FrontendConfig, Features, extract_features, load_cmvn};

/// 前端固定 16k（`frontend_conf.fs`）。
const SAMPLE_RATE: u32 = 16_000;
/// 量化权重（引擎侧 `quantize=True`；该模型仓库只提供这一个权重）。
const MODEL_FILE: &str = "model_quant.onnx";
/// CMVN 统计量（模型目录里存在就必须用，`WavFrontend(cmvn_file=...)` 无条件加载）。
const CMVN_FILE: &str = "am.mvn";
/// 一个特征帧对应的采样数（`frame_shift 10ms @16k`）：`vad_bin.py` 用 `t_offset * 160` 切波形。
const SAMPLES_PER_FRAME: usize = 160;
/// 一帧波形的采样数（`frame_length 25ms @16k` = 400）：分块切波形的尾部额外补这么多。
const FRAME_SAMPLES: usize = 400;
/// 单次送 ONNX 的最大特征帧数（`min(feats_len.max(), 6000)`）。
const MAX_CHUNK_FRAMES: usize = 6000;
/// `encoder_conf.fsmn_layers`：FSMN 层数 = 状态 cache 个数。
const CACHE_LAYERS: usize = 4;
/// `encoder_conf.proj_dim`。
const CACHE_PROJ_DIM: usize = 128;
/// `encoder_conf.lorder - 1`：cache 的时序长度。
const CACHE_LEN: usize = 19;
/// `model_conf.sil_pdf_ids` 只有一个元素 `0`，`silence_pdf_num` 为 1。
const SIL_PDF_ID: usize = 0;

// ---------------------------------------------------------------- 前端预设

/// VAD 前端参数（`config.yaml` 的 `frontend_conf`，逐字段照抄）。
///
/// 注意 `dither` 是 0.0：该模型本身就不抖，所以 VAD 输出是确定性的（ASR 侧不是）。
fn vad_frontend() -> FrontendConfig {
    FrontendConfig {
        n_mels: 80,
        frame_length_ms: 25.0,
        frame_shift_ms: 10.0,
        preemphasis: 0.97,
        dither: 0.0,
        lfr_m: 5,
        lfr_n: 1,
    }
}

// ---------------------------------------------------------------- 检测参数

/// `config.yaml` 的 `model_conf`（`VADXOptions` 的实参）。
///
/// 未列出的字段（`sample_rate`/`snr_mode`/`speech_noise_thresh_low`/`speech_noise_thresh_high`/
/// `output_frame_probs`/`do_start_point_detection`/`do_end_point_detection`）在移植路径上
/// 要么不参与计算、要么是 `pass`，见本文件头部说明。
#[derive(Debug, Clone, Copy)]
struct VadOpts {
    /// `detect_mode: 1` → 多段模式；单段模式（0）的 `max_start_silence_time` 超时分支不生效。
    multiple_utterance: bool,
    /// `max_end_silence_time`。
    max_end_silence_time: i64,
    /// `max_start_silence_time`（只在单段模式下生效）。
    max_start_silence_time: i64,
    /// `window_size_ms`。
    window_size_ms: i64,
    /// `sil_to_speech_time_thres`。
    sil_to_speech_time_thres: i64,
    /// `speech_to_sil_time_thres`。
    speech_to_sil_time_thres: i64,
    /// `speech_2_noise_ratio`。
    speech_2_noise_ratio: f64,
    /// `do_extend: 1`。
    do_extend: bool,
    /// `lookback_time_start_point`。
    lookback_time_start_point: i64,
    /// `lookahead_time_end_point`。
    lookahead_time_end_point: i64,
    /// `max_single_segment_time`（该模型是 20000，`VADXOptions` 默认值是 60000）。
    max_single_segment_time: i64,
    /// `snr_thres`。
    snr_thres: f64,
    /// `noise_frame_num_used_for_snr`。
    noise_frame_num_used_for_snr: f64,
    /// `decibel_thres`。
    decibel_thres: f64,
    /// `speech_noise_thres`。
    speech_noise_thres: f64,
    /// `fe_prior_thres`。
    fe_prior_thres: f64,
    /// `frame_in_ms`。
    frame_in_ms: i64,
    /// `frame_length_ms`。
    frame_length_ms: i64,
    /// `sample_rate`。
    sample_rate: i64,
}

impl VadOpts {
    const CONFIG: Self = Self {
        multiple_utterance: true,
        max_end_silence_time: 800,
        max_start_silence_time: 3000,
        window_size_ms: 200,
        sil_to_speech_time_thres: 150,
        speech_to_sil_time_thres: 150,
        speech_2_noise_ratio: 1.0,
        do_extend: true,
        lookback_time_start_point: 200,
        lookahead_time_end_point: 100,
        max_single_segment_time: 20_000,
        snr_thres: -100.0,
        noise_frame_num_used_for_snr: 100.0,
        decibel_thres: -100.0,
        speech_noise_thres: 0.6,
        fe_prior_thres: 1e-4,
        frame_in_ms: 10,
        frame_length_ms: 25,
        sample_rate: SAMPLE_RATE as i64,
    };

    /// `max_single_segment_time / frm_shift_in_ms`：段最长帧数（Python 侧是 int/int 的真除法，
    /// 这里保留浮点比较形式，避免整除取整带来的偏差）。
    fn max_single_segment_frames(&self) -> f64 {
        self.max_single_segment_time as f64 / self.frame_in_ms as f64
    }

    /// `int(lookahead_time_end_point / frm_shift_in_ms)`。
    fn lookahead_frames(&self) -> i64 {
        (self.lookahead_time_end_point as f64 / self.frame_in_ms as f64) as i64
    }
}

// ---------------------------------------------------------------- 枚举

/// 单帧语音/静音判定（`FrameState`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameState {
    /// `kFrameStateInvalid`
    Invalid,
    /// `kFrameStateSpeech`
    Speech,
    /// `kFrameStateSil`
    Sil,
}

/// 滑窗判定出的状态变化（`AudioChangeState`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioChangeState {
    /// `kChangeStateSpeech2Speech`
    SpeechToSpeech,
    /// `kChangeStateSpeech2Sil`
    SpeechToSil,
    /// `kChangeStateSil2Sil`
    SilToSil,
    /// `kChangeStateSil2Speech`
    SilToSpeech,
    /// `kChangeStateInvalid`
    Invalid,
}

/// 后处理状态机（`VadStateMachine`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VadStateMachine {
    /// `kVadInStateStartPointNotDetected`
    StartPointNotDetected,
    /// `kVadInStateInSpeechSegment`
    InSpeechSegment,
    /// `kVadInStateEndPointDetected`
    EndPointDetected,
}

// ---------------------------------------------------------------- 滑窗检测

/// 语音/静音滑窗（`WindowDetector`）：窗内语音帧个数越过阈值就翻转状态。
///
/// 只保留有读写意义的字段（Python 侧的 `cur_frame_state`、`voice_last_frame_count`、
/// `noise_last_frame_count`、`hydre_frame_count` 只被 `Reset` 置零，从不参与判定）。
pub struct WindowDetector {
    win_size_frame: usize,
    win_sum: i64,
    win_state: Vec<i64>,
    cur_win_pos: usize,
    pre_frame_state: FrameState,
    sil_to_speech_frmcnt_thres: i64,
    speech_to_sil_frmcnt_thres: i64,
}

impl WindowDetector {
    /// `WindowDetector(window_size_ms, sil_to_speech_time, speech_to_sil_time, frame_size_ms)`。
    pub fn new(
        window_size_ms: i64,
        sil_to_speech_time: i64,
        speech_to_sil_time: i64,
        frame_size_ms: i64,
    ) -> Self {
        let win_size_frame = (window_size_ms as f64 / frame_size_ms as f64) as usize;
        Self {
            win_size_frame,
            win_sum: 0,
            win_state: vec![0; win_size_frame],
            cur_win_pos: 0,
            pre_frame_state: FrameState::Sil,
            sil_to_speech_frmcnt_thres: (sil_to_speech_time as f64 / frame_size_ms as f64) as i64,
            speech_to_sil_frmcnt_thres: (speech_to_sil_time as f64 / frame_size_ms as f64) as i64,
        }
    }

    /// `Reset`。
    pub fn reset(&mut self) {
        self.cur_win_pos = 0;
        self.win_sum = 0;
        self.win_state = vec![0; self.win_size_frame];
        self.pre_frame_state = FrameState::Sil;
    }

    /// `GetWinSize`。
    pub fn win_size(&self) -> i64 {
        self.win_size_frame as i64
    }

    /// `DetectOneFrame`：环形窗累加 + 双向阈值翻转。
    ///
    /// 注意阈值比较是不对称的（语音靠 `>=`、静音靠 `<=`），`win_sum == 阈值` 时两个分支
    /// 都可能成立，谁生效取决于当前的 `pre_frame_state`。
    pub fn detect_one_frame(&mut self, frame_state: FrameState, _frame_count: i64) -> AudioChangeState {
        let cur_frame_state = match frame_state {
            FrameState::Speech => 1,
            FrameState::Sil => 0,
            FrameState::Invalid => return AudioChangeState::Invalid,
        };
        if self.win_size_frame == 0 {
            return AudioChangeState::Invalid;
        }
        self.win_sum -= self.win_state[self.cur_win_pos];
        self.win_sum += cur_frame_state;
        self.win_state[self.cur_win_pos] = cur_frame_state;
        self.cur_win_pos = (self.cur_win_pos + 1) % self.win_size_frame;

        if self.pre_frame_state == FrameState::Sil
            && self.win_sum >= self.sil_to_speech_frmcnt_thres
        {
            self.pre_frame_state = FrameState::Speech;
            return AudioChangeState::SilToSpeech;
        }
        if self.pre_frame_state == FrameState::Speech
            && self.win_sum <= self.speech_to_sil_frmcnt_thres
        {
            self.pre_frame_state = FrameState::Sil;
            return AudioChangeState::SpeechToSil;
        }
        if self.pre_frame_state == FrameState::Sil {
            return AudioChangeState::SilToSil;
        }
        if self.pre_frame_state == FrameState::Speech {
            return AudioChangeState::SpeechToSpeech;
        }
        AudioChangeState::Invalid
    }
}

// ---------------------------------------------------------------- 段缓冲

/// 输出段（`E2EVadSpeechBufWithDoa` 去掉采样 buffer 与 doa）。
#[derive(Debug, Clone, Copy)]
struct SpeechBuf {
    start_ms: i64,
    end_ms: i64,
    contain_seg_start_point: bool,
    contain_seg_end_point: bool,
}

impl SpeechBuf {
    fn new(start_ms: i64) -> Self {
        Self {
            start_ms,
            end_ms: start_ms,
            contain_seg_start_point: false,
            contain_seg_end_point: false,
        }
    }
}

// ---------------------------------------------------------------- 电平

/// 一帧的 10·log10(能量)（`ComputeDecibel` 的内层表达式）。
///
/// 波形是 `pcm/32768.0`（float32）而不是原始 i16：`ComputeDecibel` 拿的是 `load_data()`
/// 出来的 float 波形，和 fbank 内部再乘 32768 不是一回事，两边都别混用。
/// 求和是 float32（numpy 的累加 dtype），加地板那一步 numpy 会提升到 float64
/// （`np.float32 + 0.000001` → float64），所以地板与 log 都在 double 里做。
pub fn frame_decibel(frame: &[f32]) -> f64 {
    let mut sum = 0.0f32;
    for v in frame {
        sum += *v * *v;
    }
    10.0 * (f64::from(sum) + 1e-6).log10()
}

/// 静音段回看的帧数（`DetectOneFrame` 里 `kChangeStateSil2Sil` 的 `lookback_frame`）：
/// `int(max_end_sil_frame_cnt_thresh / frm_shift_in_ms)`，`do_extend` 时再减去
/// `int(lookahead_time_end_point / frm_shift_in_ms)` 和 1，最后夹到 0。
pub fn silence_lookback_frames(
    max_end_sil_frame_cnt_thresh: i64,
    frame_in_ms: i64,
    do_extend: bool,
    lookahead_time_end_point: i64,
) -> i64 {
    let mut lookback_frame = (max_end_sil_frame_cnt_thresh as f64 / frame_in_ms as f64) as i64;
    if do_extend {
        lookback_frame -= (lookahead_time_end_point as f64 / frame_in_ms as f64) as i64;
        lookback_frame -= 1;
        lookback_frame = lookback_frame.max(0);
    }
    lookback_frame
}

// ---------------------------------------------------------------- 后处理状态机

/// `E2EVadModel`：ONNX 给出的逐帧静音后验 + 波形电平 → 语音段。
///
/// 公开是为了让测试能脱开 ONNX 直接驱动状态机（`call` 只需要“每帧的静音后验”和波形）。
pub struct E2EVad {
    opts: VadOpts,
    windows_detector: WindowDetector,
    data_buf_start_frame: i64,
    frm_cnt: i64,
    latest_confirmed_speech_frame: i64,
    lastest_confirmed_silence_frame: i64,
    continous_silence_frame_count: i64,
    vad_state_machine: VadStateMachine,
    confirmed_start_frame: i64,
    confirmed_end_frame: i64,
    number_end_time_detected: i64,
    noise_average_decibel: f64,
    output_data_buf: Vec<SpeechBuf>,
    output_data_buf_offset: usize,
    max_end_sil_frame_cnt_thresh: i64,
    speech_noise_thres: f64,
    /// 当前 chunk 的 `scores[0][:][sil_pdf_id]`（`sil_pdf_ids` 只有 `[0]`）。
    sil_scores: Vec<f32>,
    idx_pre_chunk: i64,
    decibel: Vec<f64>,
    data_buf_size: i64,
    data_buf_all_size: i64,
}

impl E2EVad {
    /// 用模型 `config.yaml` 的 `model_conf` 构造（`E2EVadModel(vad_post_args)`）。
    pub fn offline() -> Self {
        let opts = VadOpts::CONFIG;
        Self {
            opts,
            windows_detector: WindowDetector::new(
                opts.window_size_ms,
                opts.sil_to_speech_time_thres,
                opts.speech_to_sil_time_thres,
                opts.frame_in_ms,
            ),
            data_buf_start_frame: 0,
            frm_cnt: 0,
            latest_confirmed_speech_frame: 0,
            lastest_confirmed_silence_frame: -1,
            continous_silence_frame_count: 0,
            vad_state_machine: VadStateMachine::StartPointNotDetected,
            confirmed_start_frame: -1,
            confirmed_end_frame: -1,
            number_end_time_detected: 0,
            noise_average_decibel: -100.0,
            output_data_buf: Vec::new(),
            output_data_buf_offset: 0,
            max_end_sil_frame_cnt_thresh: opts.max_end_silence_time
                - opts.speech_to_sil_time_thres,
            speech_noise_thres: opts.speech_noise_thres,
            sil_scores: Vec::new(),
            idx_pre_chunk: 0,
            decibel: Vec::new(),
            data_buf_size: 0,
            data_buf_all_size: 0,
        }
    }

    /// `AllResetDetection`：整段推理结束后清空所有状态（`__call__` 里的 `is_final` 分支）。
    fn all_reset_detection(&mut self) {
        self.data_buf_start_frame = 0;
        self.frm_cnt = 0;
        self.latest_confirmed_speech_frame = 0;
        self.lastest_confirmed_silence_frame = -1;
        self.continous_silence_frame_count = 0;
        self.vad_state_machine = VadStateMachine::StartPointNotDetected;
        self.confirmed_start_frame = -1;
        self.confirmed_end_frame = -1;
        self.number_end_time_detected = 0;
        self.noise_average_decibel = -100.0;
        self.output_data_buf.clear();
        self.output_data_buf_offset = 0;
        self.max_end_sil_frame_cnt_thresh =
            self.opts.max_end_silence_time - self.opts.speech_to_sil_time_thres;
        self.speech_noise_thres = self.opts.speech_noise_thres;
        self.sil_scores.clear();
        self.idx_pre_chunk = 0;
        self.decibel.clear();
        self.data_buf_size = 0;
        self.data_buf_all_size = 0;
        self.reset_detection();
    }

    /// `ResetDetection`：一段结束后的局部复位（多段模式每段末尾调）。
    fn reset_detection(&mut self) {
        self.continous_silence_frame_count = 0;
        self.latest_confirmed_speech_frame = 0;
        self.lastest_confirmed_silence_frame = -1;
        self.confirmed_start_frame = -1;
        self.confirmed_end_frame = -1;
        self.vad_state_machine = VadStateMachine::StartPointNotDetected;
        self.windows_detector.reset();
    }

    /// `ComputeDecibel`：给本块波形的每一帧算电平，追加到全局 `decibel` 列表（跨块累加）。
    fn compute_decibel(&mut self, waveform: &[f32]) {
        let frame_sample_length = self.opts.frame_length_ms * self.opts.sample_rate / 1000;
        let frame_shift_length = self.opts.frame_in_ms * self.opts.sample_rate / 1000;
        if frame_sample_length <= 0 || frame_shift_length <= 0 {
            return;
        }
        let frame_sample_length = frame_sample_length as usize;
        let frame_shift_length = frame_shift_length as usize;
        if self.data_buf_all_size == 0 {
            self.data_buf_all_size = waveform.len() as i64;
            self.data_buf_size = self.data_buf_all_size;
        } else {
            self.data_buf_all_size += waveform.len() as i64;
        }
        let mut offset = 0usize;
        while offset + frame_sample_length <= waveform.len() {
            self.decibel
                .push(frame_decibel(&waveform[offset..offset + frame_sample_length]));
            offset += frame_shift_length;
        }
    }

    /// `ComputeScores`：记下本块的静音后验并累计总帧数。
    fn compute_scores(&mut self, sil_scores: &[f32]) {
        self.frm_cnt += sil_scores.len() as i64;
        self.sil_scores.clear();
        self.sil_scores.extend_from_slice(sil_scores);
    }

    /// `PopDataBufTillFrame`：把还没“消费”的帧标记成已消费（`data_buf_start_frame`）。
    ///
    /// 唯一的语义偏差：`data_buf_size` 掉到不足一帧时源码的 `while` 会**死循环**（循环体里
    /// 什么都没做），这里 `break` 掉；真实输入下不会走到（`frame_idx` 恒不超过已收到的帧数）。
    fn pop_data_buf_till_frame(&mut self, frame_idx: i64) {
        let samples_per_frame = self.opts.frame_in_ms * self.opts.sample_rate / 1000;
        while self.data_buf_start_frame < frame_idx {
            if self.data_buf_size >= samples_per_frame {
                self.data_buf_start_frame += 1;
                self.data_buf_size =
                    self.data_buf_all_size - self.data_buf_start_frame * samples_per_frame;
            } else {
                break;
            }
        }
    }

    /// `PopDataToOutputBuf`：把 `[start_frm, start_frm + frm_cnt)` 这段挂到当前输出段上。
    ///
    /// 采样拷贝（`expected_sample_number`/`data_to_pop`）与 `print` 诊断不参与 ms 计算，
    /// 见文件头说明；保留 `data_buf_start_frame` 的推进。
    fn pop_data_to_output_buf(
        &mut self,
        start_frm: i64,
        frm_cnt: i64,
        first_frm_is_start_point: bool,
        last_frm_is_end_point: bool,
        _end_point_is_sent_end: bool,
    ) -> Result<()> {
        self.pop_data_buf_till_frame(start_frm);
        if self.output_data_buf.is_empty() || first_frm_is_start_point {
            self.output_data_buf
                .push(SpeechBuf::new(start_frm * self.opts.frame_in_ms));
        }
        self.data_buf_start_frame += frm_cnt;
        let frame_in_ms = self.opts.frame_in_ms;
        let cur_seg = self
            .output_data_buf
            .last_mut()
            .ok_or_else(|| Error::Inference("VAD 输出段缓冲为空".to_string()))?;
        cur_seg.end_ms = (start_frm + frm_cnt) * frame_in_ms;
        if first_frm_is_start_point {
            cur_seg.contain_seg_start_point = true;
        }
        if last_frm_is_end_point {
            cur_seg.contain_seg_end_point = true;
        }
        Ok(())
    }

    /// `OnSilenceDetected`。
    fn on_silence_detected(&mut self, valid_frame: i64) {
        self.lastest_confirmed_silence_frame = valid_frame;
        if self.vad_state_machine == VadStateMachine::StartPointNotDetected {
            self.pop_data_buf_till_frame(valid_frame);
        }
    }

    /// `OnVoiceDetected`。
    fn on_voice_detected(&mut self, valid_frame: i64) -> Result<()> {
        self.latest_confirmed_speech_frame = valid_frame;
        self.pop_data_to_output_buf(valid_frame, 1, false, false, false)
    }

    /// `OnVoiceStart`。
    fn on_voice_start(&mut self, start_frame: i64, fake_result: bool) -> Result<()> {
        if self.confirmed_start_frame != -1 {
            // 源码在这里 print("not reset vad properly")，不改状态
        } else {
            self.confirmed_start_frame = start_frame;
        }
        if !fake_result && self.vad_state_machine == VadStateMachine::StartPointNotDetected {
            self.pop_data_to_output_buf(self.confirmed_start_frame, 1, true, false, false)?;
        }
        Ok(())
    }

    /// `OnVoiceEnd`。
    fn on_voice_end(&mut self, end_frame: i64, fake_result: bool, is_last_frame: bool) -> Result<()> {
        let mut t = self.latest_confirmed_speech_frame + 1;
        while t < end_frame {
            self.on_voice_detected(t)?;
            t += 1;
        }
        if self.confirmed_end_frame != -1 {
            // 源码在这里 print("not reset vad properly")，不改状态
        } else {
            self.confirmed_end_frame = end_frame;
        }
        if !fake_result {
            self.pop_data_to_output_buf(self.confirmed_end_frame, 1, false, true, is_last_frame)?;
        }
        self.number_end_time_detected += 1;
        Ok(())
    }

    /// `MaybeOnVoiceEndIfLastFrame`。
    fn maybe_on_voice_end_if_last_frame(&mut self, is_final_frame: bool, cur_frm_idx: i64) -> Result<()> {
        if is_final_frame {
            self.on_voice_end(cur_frm_idx, false, true)?;
            self.vad_state_machine = VadStateMachine::EndPointDetected;
        }
        Ok(())
    }

    /// `LatencyFrmNumAtStartPoint`：起点回看帧数（窗长 20 + lookback 200/10 = 40）。
    fn latency_frm_num_at_start_point(&self) -> i64 {
        let mut vad_latency = self.windows_detector.win_size();
        if self.opts.do_extend {
            vad_latency += (self.opts.lookback_time_start_point as f64 / self.opts.frame_in_ms as f64) as i64;
        }
        vad_latency
    }

    /// `GetFrameState`：由电平 + 静音后验判定一帧。
    fn get_frame_state(&mut self, t: i64) -> Result<FrameState> {
        let cur_decibel = *self
            .decibel
            .get(t as usize)
            .ok_or_else(|| Error::Inference(format!("VAD 电平帧下标 {t} 越界")))?;
        let cur_snr = cur_decibel - self.noise_average_decibel;
        if cur_decibel < self.opts.decibel_thres {
            // 上游怪癖：这里自己调一次 DetectOneFrame，调用方随后还会再调一次。
            self.detect_one_frame(FrameState::Sil, t, false)?;
            return Ok(FrameState::Sil);
        }
        let local = t - self.idx_pre_chunk;
        if local < 0 {
            return Err(Error::Inference(format!("VAD 分数帧下标 {t} 落在本块之外")));
        }
        let sil_score = *self
            .sil_scores
            .get(local as usize)
            .ok_or_else(|| Error::Inference(format!("VAD 分数帧下标 {local} 越界")))? as f64;
        let noise_prob = sil_score.ln() * self.opts.speech_2_noise_ratio;
        let sum_score = 1.0 - sil_score;
        let speech_prob = sum_score.ln();
        let frame_state = if speech_prob.exp() >= noise_prob.exp() + self.speech_noise_thres {
            if cur_snr >= self.opts.snr_thres && cur_decibel >= self.opts.decibel_thres {
                FrameState::Speech
            } else {
                FrameState::Sil
            }
        } else {
            if self.noise_average_decibel < -99.9 {
                self.noise_average_decibel = cur_decibel;
            } else {
                self.noise_average_decibel = (cur_decibel
                    + self.noise_average_decibel * (self.opts.noise_frame_num_used_for_snr - 1.0))
                    / self.opts.noise_frame_num_used_for_snr;
            }
            FrameState::Sil
        };
        Ok(frame_state)
    }

    /// `__call__`（`online=False`）：吃一块分数+波形，返回这一段能确定的输出段。
    ///
    /// `sil_scores` 是 `logits[0, :, sil_pdf_id]`（本块逐帧的静音后验），`is_final` 时收尾。
    pub fn call(
        &mut self,
        sil_scores: &[f32],
        waveform: &[f32],
        is_final: bool,
        max_end_sil: i64,
    ) -> Result<Vec<(i64, i64)>> {
        self.max_end_sil_frame_cnt_thresh = max_end_sil - self.opts.speech_to_sil_time_thres;
        self.compute_decibel(waveform);
        self.compute_scores(sil_scores);
        if is_final {
            self.detect_last_frames()?;
        } else {
            self.detect_common_frames()?;
        }
        let mut segments = Vec::new();
        // Python 的 `for i in range(offset, len(buf))` 里 `i` 一次性取定，`offset` 只在真正
        // 吐出时 +1（跳过未闭合的段时两者会错位），这里照抄。
        let end_idx = self.output_data_buf.len();
        for i in self.output_data_buf_offset..end_idx {
            let buf = self.output_data_buf[i];
            if !is_final
                && (!buf.contain_seg_start_point || !buf.contain_seg_end_point)
            {
                continue;
            }
            segments.push((buf.start_ms, buf.end_ms));
            self.output_data_buf_offset += 1;
        }
        if is_final {
            self.all_reset_detection();
        }
        Ok(segments)
    }

    /// `DetectCommonFrames`：非最后一块，逐帧（倒序）推进状态机。
    fn detect_common_frames(&mut self) -> Result<()> {
        if self.vad_state_machine == VadStateMachine::EndPointDetected {
            return Ok(());
        }
        let block = self.sil_scores.len() as i64;
        for i in (0..block).rev() {
            let idx = self.frm_cnt - 1 - i;
            let frame_state = self.get_frame_state(idx)?;
            self.detect_one_frame(frame_state, idx, false)?;
        }
        self.idx_pre_chunk += block;
        Ok(())
    }

    /// `DetectLastFrames`：最后一块，最后一帧按 `is_final_frame=true` 处理（收尾成段）。
    fn detect_last_frames(&mut self) -> Result<()> {
        if self.vad_state_machine == VadStateMachine::EndPointDetected {
            return Ok(());
        }
        let block = self.sil_scores.len() as i64;
        for i in (0..block).rev() {
            let idx = self.frm_cnt - 1 - i;
            let frame_state = self.get_frame_state(idx)?;
            if i != 0 {
                self.detect_one_frame(frame_state, idx, false)?;
            } else {
                self.detect_one_frame(frame_state, self.frm_cnt - 1, true)?;
            }
        }
        Ok(())
    }

    /// `DetectOneFrame`：状态机的核心（起点/段内/终点的全部转移）。
    fn detect_one_frame(
        &mut self,
        cur_frm_state: FrameState,
        cur_frm_idx: i64,
        is_final_frame: bool,
    ) -> Result<()> {
        let tmp_cur_frm_state = match cur_frm_state {
            FrameState::Speech => {
                // `math.fabs(1.0) > fe_prior_thres`（1e-4）恒真，保留源码的比较形式
                if 1.0f64.abs() > self.opts.fe_prior_thres {
                    FrameState::Speech
                } else {
                    FrameState::Sil
                }
            }
            FrameState::Sil => FrameState::Sil,
            FrameState::Invalid => FrameState::Invalid,
        };
        let state_change = self
            .windows_detector
            .detect_one_frame(tmp_cur_frm_state, cur_frm_idx);
        let frm_shift_in_ms = self.opts.frame_in_ms;
        match state_change {
            AudioChangeState::SilToSpeech => {
                self.continous_silence_frame_count = 0;
                match self.vad_state_machine {
                    VadStateMachine::StartPointNotDetected => {
                        let start_frame = self
                            .data_buf_start_frame
                            .max(cur_frm_idx - self.latency_frm_num_at_start_point());
                        self.on_voice_start(start_frame, false)?;
                        self.vad_state_machine = VadStateMachine::InSpeechSegment;
                        let mut t = start_frame + 1;
                        while t <= cur_frm_idx {
                            self.on_voice_detected(t)?;
                            t += 1;
                        }
                    }
                    VadStateMachine::InSpeechSegment => {
                        let mut t = self.latest_confirmed_speech_frame + 1;
                        while t < cur_frm_idx {
                            self.on_voice_detected(t)?;
                            t += 1;
                        }
                        if (cur_frm_idx - self.confirmed_start_frame + 1) as f64
                            > self.opts.max_single_segment_frames()
                        {
                            self.on_voice_end(cur_frm_idx, false, false)?;
                            self.vad_state_machine = VadStateMachine::EndPointDetected;
                        } else if !is_final_frame {
                            self.on_voice_detected(cur_frm_idx)?;
                        } else {
                            self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx)?;
                        }
                    }
                    VadStateMachine::EndPointDetected => {}
                }
            }
            AudioChangeState::SpeechToSil => {
                self.continous_silence_frame_count = 0;
                if self.vad_state_machine == VadStateMachine::InSpeechSegment {
                    if (cur_frm_idx - self.confirmed_start_frame + 1) as f64
                        > self.opts.max_single_segment_frames()
                    {
                        self.on_voice_end(cur_frm_idx, false, false)?;
                        self.vad_state_machine = VadStateMachine::EndPointDetected;
                    } else if !is_final_frame {
                        self.on_voice_detected(cur_frm_idx)?;
                    } else {
                        self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx)?;
                    }
                }
            }
            AudioChangeState::SpeechToSpeech => {
                self.continous_silence_frame_count = 0;
                if self.vad_state_machine == VadStateMachine::InSpeechSegment {
                    if (cur_frm_idx - self.confirmed_start_frame + 1) as f64
                        > self.opts.max_single_segment_frames()
                    {
                        self.on_voice_end(cur_frm_idx, false, false)?;
                        self.vad_state_machine = VadStateMachine::EndPointDetected;
                    } else if !is_final_frame {
                        self.on_voice_detected(cur_frm_idx)?;
                    } else {
                        self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx)?;
                    }
                }
            }
            AudioChangeState::SilToSil => {
                self.continous_silence_frame_count += 1;
                match self.vad_state_machine {
                    VadStateMachine::StartPointNotDetected => {
                        let silence_timeout = !self.opts.multiple_utterance
                            && self.continous_silence_frame_count * frm_shift_in_ms
                                > self.opts.max_start_silence_time;
                        if silence_timeout
                            || (is_final_frame && self.number_end_time_detected == 0)
                        {
                            let mut t = self.lastest_confirmed_silence_frame + 1;
                            while t < cur_frm_idx {
                                self.on_silence_detected(t);
                                t += 1;
                            }
                            self.on_voice_start(0, true)?;
                            self.on_voice_end(0, true, false)?;
                            self.vad_state_machine = VadStateMachine::EndPointDetected;
                        } else if cur_frm_idx >= self.latency_frm_num_at_start_point() {
                            let valid = cur_frm_idx - self.latency_frm_num_at_start_point();
                            self.on_silence_detected(valid);
                        }
                    }
                    VadStateMachine::InSpeechSegment => {
                        if self.continous_silence_frame_count * frm_shift_in_ms
                            >= self.max_end_sil_frame_cnt_thresh
                        {
                            let lookback_frame = silence_lookback_frames(
                                self.max_end_sil_frame_cnt_thresh,
                                frm_shift_in_ms,
                                self.opts.do_extend,
                                self.opts.lookahead_time_end_point,
                            );
                            self.on_voice_end(cur_frm_idx - lookback_frame, false, false)?;
                            self.vad_state_machine = VadStateMachine::EndPointDetected;
                        } else if (cur_frm_idx - self.confirmed_start_frame + 1) as f64
                            > self.opts.max_single_segment_frames()
                        {
                            self.on_voice_end(cur_frm_idx, false, false)?;
                            self.vad_state_machine = VadStateMachine::EndPointDetected;
                        } else if self.opts.do_extend && !is_final_frame {
                            if self.continous_silence_frame_count <= self.opts.lookahead_frames() {
                                self.on_voice_detected(cur_frm_idx)?;
                            }
                        } else {
                            self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx)?;
                        }
                    }
                    VadStateMachine::EndPointDetected => {}
                }
            }
            AudioChangeState::Invalid => {}
        }

        if self.vad_state_machine == VadStateMachine::EndPointDetected
            && self.opts.multiple_utterance
        {
            self.reset_detection();
        }
        Ok(())
    }
}

// ---------------------------------------------------------------- ONNX

/// 进程级 ONNX Runtime 环境（ort 每进程只允许一个 `Environment`）。
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

// ---------------------------------------------------------------- 对外接口

/// FSMN-VAD 离线语音活动检测。
pub struct Vad {
    session: Session,
    frontend: FrontendConfig,
    cmvn: Option<Cmvn>,
    opts: VadOpts,
}

impl Vad {
    /// 从模型目录加载（model_quant.onnx + am.mvn）。
    pub fn new(model_dir: &Path, threads: usize) -> Result<Self> {
        let model_file = model_dir.join(MODEL_FILE);
        if !model_file.is_file() {
            return Err(Error::Model(format!(
                "VAD 模型文件不存在: {}",
                model_file.display()
            )));
        }
        // `WavFrontend(cmvn_file=os.path.join(model_dir, "am.mvn"))`：文件在就必须用
        let cmvn_file = model_dir.join(CMVN_FILE);
        let cmvn = if cmvn_file.is_file() {
            Some(load_cmvn(&cmvn_file)?)
        } else {
            None
        };

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
            .map_err(|e| Error::Model(format!("加载 VAD 模型 {} 失败: {e}", model_file.display())))?;

        Ok(Self {
            session,
            frontend: vad_frontend(),
            cmvn,
            opts: VadOpts::CONFIG,
        })
    }

    /// 对整段音频做 VAD，返回语音段时间 [(start_ms, end_ms), ...]。
    pub fn infer(&mut self, wave: &[i16], sample_rate: u32) -> Result<Vec<(i64, i64)>> {
        if sample_rate != SAMPLE_RATE {
            return Err(Error::InvalidInput(format!(
                "VAD 只支持 {SAMPLE_RATE} Hz，实际 {sample_rate} Hz"
            )));
        }
        // 参考实现的推理输入是 `pcm.astype(float32) / 32768.0`（fbank 内部再乘回 32768，
        // 所以 `extract_features` 仍然吃 i16；但 ComputeDecibel 吃的是这个 float 波形）
        let waveform: Vec<f32> = wave.iter().map(|v| *v as f32 / 32768.0).collect();
        let feats = extract_features(wave, sample_rate, &self.frontend, self.cmvn.as_ref())?;
        if feats.feats_len == 0 {
            return Ok(Vec::new());
        }
        self.run_vad(&feats, &waveform)
    }

    /// `Fsmn_vad.__call__` 的分块循环（batch=1）。
    fn run_vad(&mut self, feats: &Features, waveform: &[f32]) -> Result<Vec<(i64, i64)>> {
        let feats_len = feats.feats_len;
        let dim = feats.feats.ncols();
        if dim == 0 {
            return Err(Error::Inference("VAD 特征维度为 0".to_string()));
        }
        let mut scorer = E2EVad::offline();
        // prepare_cache：每层一个 [1, proj_dim, lorder-1, 1] 的零 cache，块间传递
        let mut caches: Vec<Vec<f32>> = (0..CACHE_LAYERS)
            .map(|_| vec![0.0f32; CACHE_PROJ_DIM * CACHE_LEN])
            .collect();
        let mut cache_shape: Vec<i64> = vec![1, CACHE_PROJ_DIM as i64, CACHE_LEN as i64, 1];

        let max_end_sil = self.opts.max_end_silence_time;
        let mut segments: Vec<(i64, i64)> = Vec::new();
        let mut step = feats_len.min(MAX_CHUNK_FRAMES);
        // `range(0, feats_len, min(step, feats_len - t_offset))` 的第三参数只在 range 创建时
        // 求值一次（t_offset=0），之后 `step` 的改写不影响迭代步长。
        let range_step = step.min(feats_len);
        let mut t_offset = 0usize;
        while t_offset < feats_len {
            let is_final = if t_offset + step >= feats_len - 1 {
                step = feats_len - t_offset;
                true
            } else {
                false
            };

            let mut feats_package = Vec::with_capacity(step * dim);
            for row in feats.feats.rows().into_iter().skip(t_offset).take(step) {
                feats_package.extend(row.iter().copied());
            }
            if feats_package.len() != step * dim {
                return Err(Error::Inference(format!(
                    "VAD 特征块大小异常：{} != {}",
                    feats_package.len(),
                    step * dim
                )));
            }

            // `waveform[:, t_offset*160 : min(len, (t_offset+step-1)*160 + 400)]`
            let w_start = (t_offset * SAMPLES_PER_FRAME).min(waveform.len());
            let w_end = ((t_offset + step - 1) * SAMPLES_PER_FRAME + FRAME_SAMPLES)
                .min(waveform.len())
                .max(w_start);
            let waveform_package = &waveform[w_start..w_end];

            let chunk = self.infer_chunk(&feats_package, step, dim, &caches, &cache_shape)?;
            caches = chunk.caches;
            cache_shape = chunk.cache_shape;
            segments.extend(scorer.call(
                &chunk.sil_scores,
                waveform_package,
                is_final,
                max_end_sil,
            )?);
            t_offset += range_step;
        }
        Ok(segments)
    }

    /// 一次 ONNX 推理：`speech` + 4 个 `in_cacheN` → `logits` + 4 个 `out_cacheN`。
    fn infer_chunk(
        &mut self,
        feats_package: &[f32],
        frames: usize,
        dim: usize,
        caches: &[Vec<f32>],
        cache_shape: &[i64],
    ) -> Result<ChunkOutput> {
        let speech = Value::from_array((
            vec![1_i64, frames as i64, dim as i64],
            feats_package.to_vec(),
        ))
        .map_err(|e| Error::Inference(format!("构造 VAD 特征张量失败: {e}")))?;
        let mut inputs = Vec::with_capacity(1 + caches.len());
        inputs.push(("speech".to_string(), speech));
        for (i, cache) in caches.iter().enumerate() {
            let value = Value::from_array((cache_shape.to_vec(), cache.clone()))
                .map_err(|e| Error::Inference(format!("构造 in_cache{i} 张量失败: {e}")))?;
            inputs.push((format!("in_cache{i}"), value));
        }
        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| Error::Inference(format!("VAD 推理失败: {e}")))?;

        let logits = outputs
            .get("logits")
            .ok_or_else(|| Error::Inference("VAD 输出里没有 logits".to_string()))?;
        let (shape, data) = logits
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Inference(format!("读取 VAD logits 失败: {e}")))?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        if dims.len() != 3 || dims[1] < 0 || dims[2] <= SIL_PDF_ID as i64 {
            return Err(Error::Inference(format!("VAD logits 维度异常: {dims:?}")));
        }
        let out_frames = dims[1] as usize;
        let out_dim = dims[2] as usize;
        if out_frames != frames {
            return Err(Error::Inference(format!(
                "VAD logits 帧数 {out_frames} 与输入帧数 {frames} 不一致"
            )));
        }
        if data.len() < out_frames * out_dim {
            return Err(Error::Inference(format!(
                "VAD logits 大小异常：{} < {}",
                data.len(),
                out_frames * out_dim
            )));
        }
        // `scores[0][t][sil_pdf_id]`（`sil_pdf_ids = [0]`，`silence_pdf_num = 1`）
        let sil_scores: Vec<f32> = (0..out_frames)
            .map(|t| data[t * out_dim + SIL_PDF_ID])
            .collect();

        let mut out_caches = Vec::with_capacity(caches.len());
        let mut out_shape = cache_shape.to_vec();
        for i in 0..caches.len() {
            let name = format!("out_cache{i}");
            let value = outputs
                .get(name.as_str())
                .ok_or_else(|| Error::Inference(format!("VAD 输出里没有 {name}")))?;
            let (shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Inference(format!("读取 {name} 失败: {e}")))?;
            out_shape = shape.iter().copied().collect();
            out_caches.push(data.to_vec());
        }
        Ok(ChunkOutput {
            sil_scores,
            caches: out_caches,
            cache_shape: out_shape,
        })
    }
}

/// 一次分块推理的结果。
struct ChunkOutput {
    /// 本块逐帧的静音后验（`logits[0, :, sil_pdf_id]`）。
    sil_scores: Vec<f32>,
    /// 传给下一块的 FSMN 状态 cache。
    caches: Vec<Vec<f32>>,
    /// cache 的实际形状（`lorder-1` 那一维在 ONNX 图里是动态的）。
    cache_shape: Vec<i64>,
}
