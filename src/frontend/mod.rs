//! T6：声学特征前端（fbank + LFR + CMVN）。
//!
//! 移植来源（参数一律取自这些源码，不猜）：
//! * `reference/funasr_onnx/sensevoice_bin.py` / `paraformer_bin.py`：`extract_feat` 的顺序是
//!   `WavFrontend.fbank()` → `lfr_cmvn()`，也就是 fbank → LFR → CMVN。
//! * `reference/funasr_onnx/utils/frontend.py`：fbank 选项、`apply_lfr`、`apply_cmvn`、`load_cmvn`。
//! * fbank 本体由 `kaldi_native_fbank`（Kaldi `feature-window`/`feature-fbank`/`mel-computations`
//!   的移植，实数 FFT 走 kiss_fftr）实现；Python 侧只覆盖了 `samp_freq/dither/window_type/
//!   frame_shift_ms/frame_length_ms/num_bins/energy_floor/snip_edges/debug_mel`，其余取该库默认值
//!   （preemph_coeff 0.97、remove_dc_offset、round_to_power_of_two、low_freq 20、high_freq 0→Nyquist、
//!   use_log_fbank + eps 做 log 下限）。
//! * 实参来自模型目录的 `config.yaml`：fs 16000 / window hamming / n_mels 80 / frame_length 25 /
//!   frame_shift 10 / lfr_m 7 / lfr_n 6（SenseVoiceSmall-onnx 与
//!   speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx 完全一致，两者的 am.mvn 逐字节相同）。
//!
//! 为什么要逐操作复刻 float32 算术：静音/高频处的 mel bin 里各 fft bin 的能量相差 9 个数量级，
//! 求 log 前的能量本身是 float32 近抵消的产物；用 float64 算反而会与参考实现差到 2e-4。
//! 因此帧预处理、kiss_fftr、功率谱、mel 点积、log 全部按参考实现的顺序做 float32 运算。

use std::path::Path;

use ndarray::Array2;

use crate::error::{Error, Result};

/// 前端配置（各字段取值由 T6 依据 vendored 源码确定；调用方使用下方预设函数）。
#[derive(Debug, Clone)]
pub struct FrontendConfig {
    pub n_mels: usize,
    pub frame_length_ms: f32,
    pub frame_shift_ms: f32,
    /// 预加重系数；0.0 = 关闭。
    pub preemphasis: f32,
    pub dither: f32,
    /// LFR 叠帧参数（m=叠帧数, n=跳帧数）。
    pub lfr_m: usize,
    pub lfr_n: usize,
}

/// CMVN 统计量（读自模型目录 am.mvn）。
#[derive(Debug, Clone)]
pub struct Cmvn {
    pub mean: Vec<f32>,
    pub istd: Vec<f32>,
}

/// 提取完成的特征：feats 形状 (T, n_mels * lfr_m)，feats_len = T。
#[derive(Debug, Clone)]
pub struct Features {
    pub feats: Array2<f32>,
    pub feats_len: usize,
}

/// 前端固定按 16k 建模（`samp_freq` 只有这一个取值被参考实现使用）。
const SUPPORTED_SAMPLE_RATE: u32 = 16_000;
/// `MelBanksOptions::low_freq` 默认值（Python 未覆盖）。
const MEL_LOW_FREQ: f32 = 20.0;
/// `MelBanksOptions::high_freq`：<= 0 表示 Nyquist + 该值（即 0 → Nyquist）。
const MEL_HIGH_FREQ: f32 = 0.0;
/// `FbankOptions::eps`（= FLT_EPSILON），log 前的地板。
const LOG_FLOOR: f32 = f32::EPSILON;

/// 读 am.mvn（Kaldi 文本格式 mean/istd）。
///
/// 格式与 `frontend.py::load_cmvn` 一致：`<AddShift>` / `<Rescale>` 行的下一行是
/// `<LearnRateCoef> 0 [ v0 v1 ... vN ]`，取 `[3 .. len-1]` 之间的数值。
pub fn load_cmvn(path: &Path) -> Result<Cmvn> {
    let text = std::fs::read_to_string(path)?;
    let lines: Vec<&str> = text.lines().collect();
    let mut mean: Option<Vec<f32>> = None;
    let mut istd: Option<Vec<f32>> = None;

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
        let next = match lines.get(i + 1) {
            Some(l) => l.split_whitespace().collect::<Vec<_>>(),
            None => continue,
        };
        if next.first().copied() != Some("<LearnRateCoef>") {
            continue;
        }
        // <LearnRateCoef> <coef> [ v0 .. vN ] —— 与 Python 一样丢掉首 3 个与末尾的 "]"
        if next.len() < 5 {
            return Err(Error::Frontend(format!(
                "{tag} 的数值行过短（{} 个字段）：{}",
                next.len(),
                path.display()
            )));
        }
        let mut vals = Vec::with_capacity(next.len() - 4);
        for tok in &next[3..next.len() - 1] {
            let v = tok
                .parse::<f32>()
                .map_err(|e| Error::Frontend(format!("am.mvn 里的数值 {tok:?} 无法解析：{e}")))?;
            vals.push(v);
        }
        *slot = Some(vals);
    }

    match (mean, istd) {
        (Some(m), Some(v)) => {
            if m.is_empty() {
                return Err(Error::Frontend(format!(
                    "am.mvn 的 <AddShift> 为空：{}",
                    path.display()
                )));
            }
            if m.len() != v.len() {
                return Err(Error::Frontend(format!(
                    "am.mvn 的 mean({}) 与 istd({}) 维度不一致：{}",
                    m.len(),
                    v.len(),
                    path.display()
                )));
            }
            Ok(Cmvn { mean: m, istd: v })
        }
        _ => Err(Error::Frontend(format!(
            "am.mvn 缺少 <AddShift> 或 <Rescale>：{}",
            path.display()
        ))),
    }
}

/// 提取特征：fbank(80维) → LFR 叠帧 → CMVN（顺序以 vendored 源码为准）。
/// wave 为 s16 原始采样；数值须与 golden feats 对齐（atol 1e-4）。
pub fn extract_features(
    wave: &[i16],
    sample_rate: u32,
    cfg: &FrontendConfig,
    cmvn: Option<&Cmvn>,
) -> Result<Features> {
    if sample_rate != SUPPORTED_SAMPLE_RATE {
        return Err(Error::Frontend(format!(
            "只支持 {SUPPORTED_SAMPLE_RATE} Hz，实际 {sample_rate} Hz"
        )));
    }
    if cfg.dither != 0.0 {
        return Err(Error::Frontend(format!(
            "不支持 dither={}：参考实现用 C 的 rand() 加高斯噪声，逐样本不可复现",
            cfg.dither
        )));
    }
    if cfg.lfr_m == 0 || cfg.lfr_n == 0 || cfg.n_mels == 0 {
        return Err(Error::Frontend(format!(
            "前端参数非法：n_mels={} lfr_m={} lfr_n={}",
            cfg.n_mels, cfg.lfr_m, cfg.lfr_n
        )));
    }

    let fbank = fbank(wave, cfg)?;
    let stacked = if cfg.lfr_m != 1 || cfg.lfr_n != 1 {
        apply_lfr(&fbank, cfg.lfr_m, cfg.lfr_n)?
    } else {
        fbank
    };
    let feats = match cmvn {
        Some(c) => apply_cmvn(&stacked, c)?,
        None => stacked,
    };
    let feats_len = feats.nrows();
    Ok(Features { feats, feats_len })
}

/// SenseVoiceSmall 前端参数预设（以 vendored sensevoice_bin.py 为准）。
pub fn sensevoice_frontend() -> FrontendConfig {
    // config.yaml frontend_conf: fs 16000, window hamming, n_mels 80, frame_length 25,
    // frame_shift 10, lfr_m 7, lfr_n 6；dither 未写，参考实现落到 1.0，golden 生成时钉成 0.0。
    FrontendConfig {
        n_mels: 80,
        frame_length_ms: 25.0,
        frame_shift_ms: 10.0,
        preemphasis: 0.97,
        dither: 0.0,
        lfr_m: 7,
        lfr_n: 6,
    }
}

/// Paraformer-large 前端参数预设（以 vendored paraformer_bin.py 为准）。
pub fn paraformer_frontend() -> FrontendConfig {
    // speech_paraformer-large...-onnx 的 config.yaml frontend_conf 与 SenseVoice 逐字段一致，
    // am.mvn 也逐字节相同，因此取值相同（同样 dither=0.0）。
    FrontendConfig {
        n_mels: 80,
        frame_length_ms: 25.0,
        frame_shift_ms: 10.0,
        preemphasis: 0.97,
        dither: 0.0,
        lfr_m: 7,
        lfr_n: 6,
    }
}

// ---------------------------------------------------------------- fbank

/// Kaldi 式 fbank：帧切分 → 去直流 → 预加重 → 窗 → 512 点实数 FFT → 功率谱 → mel → log。
fn fbank(wave: &[i16], cfg: &FrontendConfig) -> Result<Array2<f32>> {
    let sr = SUPPORTED_SAMPLE_RATE as f32;
    let frame_length = (sr * 0.001 * cfg.frame_length_ms) as usize;
    let frame_shift = (sr * 0.001 * cfg.frame_shift_ms) as usize;
    if frame_length == 0 || frame_shift == 0 {
        return Err(Error::Frontend(format!(
            "帧长/帧移非法：frame_length={frame_length} frame_shift={frame_shift}"
        )));
    }
    let padded = frame_length.next_power_of_two();
    if padded < 2 || !padded.is_multiple_of(2) {
        return Err(Error::Frontend(format!("padding 后窗长 {padded} 不支持")));
    }
    // snip_edges = true：完整帧才算一帧
    let num_frames = if wave.len() < frame_length {
        0
    } else {
        1 + (wave.len() - frame_length) / frame_shift
    };
    if num_frames == 0 {
        return Err(Error::Frontend(format!(
            "音频过短：{} 个采样不足一帧（{frame_length}）",
            wave.len()
        )));
    }

    let window = hamming_window(frame_length);
    let mel = MelBanks::new(cfg.n_mels, sr, padded)?;
    let rfft = RealFft::new(padded);

    let mut frame = vec![0.0f32; padded];
    let mut power = vec![0.0f32; padded / 2 + 1];
    let mut out = Array2::<f32>::zeros((num_frames, cfg.n_mels));

    for f in 0..num_frames {
        let start = f * frame_shift;
        for (i, slot) in frame.iter_mut().take(frame_length).enumerate() {
            *slot = wave[start + i] as f32;
        }
        remove_dc_offset(&mut frame[..frame_length]);
        preemphasize(&mut frame[..frame_length], cfg.preemphasis);
        for (slot, w) in frame.iter_mut().take(frame_length).zip(&window) {
            *slot *= *w;
        }
        for slot in frame.iter_mut().skip(frame_length) {
            *slot = 0.0;
        }

        rfft.power_spectrum(&frame, &mut power);
        for (b, energy) in out.row_mut(f).iter_mut().enumerate() {
            *energy = mel.energy(b, &power).max(LOG_FLOOR).ln();
        }
    }
    Ok(out)
}

/// `GetWindow("hamming", window_size)`：`0.54 - 0.46*cos(2πi/(L-1))`，double 计算后落 float。
fn hamming_window(len: usize) -> Vec<f32> {
    if len < 2 {
        return vec![1.0; len];
    }
    let a = std::f64::consts::TAU / (len - 1) as f64;
    (0..len)
        .map(|i| (0.54 - 0.46 * (a * i as f64).cos()) as f32)
        .collect()
}

/// `RemoveDcOffset`：顺序 float32 累加求均值（与 C 的写法一致，别换成 f64 或分块求和）。
fn remove_dc_offset(d: &mut [f32]) {
    if d.is_empty() {
        return;
    }
    let mut sum = 0.0f32;
    for v in d.iter() {
        sum += *v;
    }
    let mean = sum / d.len() as f32;
    for v in d.iter_mut() {
        *v -= mean;
    }
}

/// `Preemphasize`：**注意**参考实现（kaldi_native_fbank）与经典 Kaldi 不同——
/// 首样本同样要减 `coeff*x[0]`（经典 Kaldi 是 `y[0]=x[0]`），且从尾向头迭代。
fn preemphasize(d: &mut [f32], coeff: f32) {
    if coeff == 0.0 || d.is_empty() {
        return;
    }
    for i in (1..d.len()).rev() {
        d[i] -= coeff * d[i - 1];
    }
    d[0] -= coeff * d[0];
}

/// mel 滤波器组（Kaldi `InitKaldiMelBanks`，float32 逐式复刻）。
struct MelBanks {
    /// 每个 bin 的 (第一个非零 fft bin, 权重)
    bins: Vec<(usize, Vec<f32>)>,
}

impl MelBanks {
    fn new(num_bins: usize, sample_freq: f32, padded: usize) -> Result<Self> {
        if num_bins < 3 {
            return Err(Error::Frontend(format!("mel bin 太少：{num_bins}")));
        }
        let num_fft_bins = padded / 2;
        let nyquist = 0.5 * sample_freq;
        let low_freq = MEL_LOW_FREQ;
        let high_freq = if MEL_HIGH_FREQ > 0.0 {
            MEL_HIGH_FREQ
        } else {
            nyquist + MEL_HIGH_FREQ
        };
        if low_freq < 0.0 || low_freq >= nyquist || high_freq <= 0.0 || high_freq > nyquist {
            return Err(Error::Frontend(format!(
                "mel 频率范围非法：{low_freq}..{high_freq}（Nyquist {nyquist}）"
            )));
        }
        let fft_bin_width = sample_freq / padded as f32;
        let mel_low_freq = mel_scale(low_freq);
        let mel_high_freq = mel_scale(high_freq);
        let mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins as f32 + 1.0);

        let mut bins = Vec::with_capacity(num_bins);
        let mut this_bin = vec![0.0f32; num_fft_bins];
        for bin in 0..num_bins {
            let b = bin as f32;
            let left_mel = mel_low_freq + b * mel_freq_delta;
            let center_mel = mel_low_freq + (b + 1.0) * mel_freq_delta;
            let right_mel = mel_low_freq + (b + 2.0) * mel_freq_delta;
            this_bin.iter_mut().for_each(|v| *v = 0.0);
            let mut first = usize::MAX;
            let mut last = 0usize;
            for (i, slot) in this_bin.iter_mut().enumerate() {
                let freq = fft_bin_width * i as f32;
                let mel = mel_scale(freq);
                if mel > left_mel && mel < right_mel {
                    let weight = if mel <= center_mel {
                        (mel - left_mel) / (center_mel - left_mel)
                    } else {
                        (right_mel - mel) / (right_mel - center_mel)
                    };
                    *slot = weight;
                    if first == usize::MAX {
                        first = i;
                    }
                    last = i;
                }
            }
            if first == usize::MAX || last < first {
                return Err(Error::Frontend(format!(
                    "mel bin {bin} 没有覆盖任何 fft bin（num_bins 太大？）"
                )));
            }
            bins.push((first, this_bin[first..=last].to_vec()));
        }
        Ok(Self { bins })
    }

    /// `MelBanks::Compute`：顺序 float32 点积。
    fn energy(&self, bin: usize, power: &[f32]) -> f32 {
        let (offset, weights) = &self.bins[bin];
        let mut energy = 0.0f32;
        for (k, w) in weights.iter().enumerate() {
            energy += *w * power[offset + k];
        }
        energy
    }
}

/// `MelScale`：`1127 * log(1 + f/700)`（float32）。
fn mel_scale(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

// ---------------------------------------------------------------- kiss_fftr

#[derive(Clone, Copy, Default)]
struct Cpx {
    re: f32,
    im: f32,
}

impl Cpx {
    fn mul(self, o: Self) -> Self {
        Self {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
    fn add(self, o: Self) -> Self {
        Self {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
    fn sub(self, o: Self) -> Self {
        Self {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }
    fn add_assign(&mut self, o: Self) {
        self.re += o.re;
        self.im += o.im;
    }
}

/// kiss_fft 的分解序列 `[(radix, m), ..]`（`kf_factor`）。
fn kf_factors(mut n: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut p = 4usize;
    let floor_sqrt = (n as f64).sqrt().floor();
    loop {
        while !n.is_multiple_of(p) {
            p = match p {
                4 => 2,
                2 => 3,
                _ => p + 2,
            };
            if p as f64 > floor_sqrt {
                p = n;
            }
        }
        n /= p;
        out.push((p, n));
        if n <= 1 {
            break;
        }
    }
    out
}

/// kiss_fft 混合基复数 FFT（forward），逐式复刻 `kiss_fft.c`。
struct KissFft {
    nfft: usize,
    factors: Vec<(usize, usize)>,
    twiddles: Vec<Cpx>,
}

impl KissFft {
    fn new(nfft: usize) -> Self {
        let pi = std::f64::consts::PI;
        let twiddles = (0..nfft)
            .map(|i| {
                let phase = (-2.0 * pi) * (i as f64) / (nfft as f64);
                Cpx {
                    re: phase.cos() as f32,
                    im: phase.sin() as f32,
                }
            })
            .collect();
        Self {
            nfft,
            factors: kf_factors(nfft),
            twiddles,
        }
    }

    fn compute(&self, input: &[Cpx], out: &mut [Cpx]) {
        self.kf_work(out, input, 0, 1, 1, 0);
    }

    fn kf_work(
        &self,
        fout: &mut [Cpx],
        fin: &[Cpx],
        fin_start: usize,
        fstride: usize,
        in_stride: usize,
        fi: usize,
    ) {
        let (p, m) = self.factors[fi];
        let step = fstride * in_stride;
        if m == 1 {
            let mut k = fin_start;
            for slot in fout.iter_mut().take(p) {
                *slot = fin[k];
                k += step;
            }
        } else {
            for o in 0..p {
                self.kf_work(
                    &mut fout[o * m..(o + 1) * m],
                    fin,
                    fin_start + o * step,
                    fstride * p,
                    in_stride,
                    fi + 1,
                );
            }
        }
        let span = p * m;
        match p {
            2 => self.bfly2(&mut fout[..span], fstride, m),
            4 => self.bfly4(&mut fout[..span], fstride, m),
            _ => self.bfly_generic(&mut fout[..span], fstride, m, p),
        }
    }

    fn bfly2(&self, fout: &mut [Cpx], fstride: usize, m: usize) {
        let (lo, hi) = fout.split_at_mut(m);
        for (k, a) in lo.iter_mut().enumerate() {
            let t = hi[k].mul(self.twiddles[k * fstride]);
            hi[k] = a.sub(t);
            *a = a.add(t);
        }
    }

    fn bfly4(&self, fout: &mut [Cpx], fstride: usize, m: usize) {
        let m2 = 2 * m;
        let m3 = 3 * m;
        for k in 0..m {
            let s0 = fout[k + m].mul(self.twiddles[k * fstride]);
            let s1 = fout[k + m2].mul(self.twiddles[2 * k * fstride]);
            let s2 = fout[k + m3].mul(self.twiddles[3 * k * fstride]);

            let s5 = fout[k].sub(s1);
            fout[k] = fout[k].add(s1);
            let s3 = s0.add(s2);
            let s4 = s0.sub(s2);
            fout[k + m2] = fout[k].sub(s3);
            fout[k] = fout[k].add(s3);
            fout[k + m] = Cpx {
                re: s5.re + s4.im,
                im: s5.im - s4.re,
            };
            fout[k + m3] = Cpx {
                re: s5.re - s4.im,
                im: s5.im + s4.re,
            };
        }
    }

    fn bfly_generic(&self, fout: &mut [Cpx], fstride: usize, m: usize, p: usize) {
        let mut scratch = vec![Cpx::default(); p];
        for u in 0..m {
            let mut k = u;
            for slot in scratch.iter_mut() {
                *slot = fout[k];
                k += m;
            }
            let mut k = u;
            for _ in 0..p {
                let mut twidx = 0usize;
                fout[k] = scratch[0];
                for s in scratch.iter().take(p).skip(1) {
                    twidx += fstride * k;
                    if twidx >= self.nfft {
                        twidx -= self.nfft;
                    }
                    let t = s.mul(self.twiddles[twidx]);
                    fout[k].add_assign(t);
                }
                k += m;
            }
        }
    }
}

/// kiss_fftr：n 点实数 FFT，直接产出功率谱（n/2+1 个），逐式复刻 `kiss_fftr.c` 与
/// kaldi_native_fbank 的 `Rfft::Forward` + `ComputePowerSpectrum`。
struct RealFft {
    n: usize,
    sub: KissFft,
    super_twiddles: Vec<Cpx>,
}

impl RealFft {
    fn new(n: usize) -> Self {
        let ncfft = n / 2;
        let pi = std::f64::consts::PI;
        let super_twiddles = (0..ncfft / 2)
            .map(|i| {
                let phase = -pi * ((i + 1) as f64 / ncfft as f64 + 0.5);
                Cpx {
                    re: phase.cos() as f32,
                    im: phase.sin() as f32,
                }
            })
            .collect();
        Self {
            n,
            sub: KissFft::new(ncfft),
            super_twiddles,
        }
    }

    /// `input` 长度必须是 n；`out` 长度 n/2+1，写入 |X_k|²。
    fn power_spectrum(&self, input: &[f32], out: &mut [f32]) {
        let ncfft = self.n / 2;
        let mut packed = vec![Cpx::default(); ncfft];
        for (i, slot) in packed.iter_mut().enumerate() {
            *slot = Cpx {
                re: input[2 * i],
                im: input[2 * i + 1],
            };
        }
        let mut tmp = vec![Cpx::default(); ncfft];
        self.sub.compute(&packed, &mut tmp);

        let tdc = tmp[0];
        let dc = tdc.re + tdc.im;
        let nyq = tdc.re - tdc.im;
        out[0] = dc * dc;
        out[ncfft] = nyq * nyq;

        // for k=256 (k == ncfft-k) 时后者覆盖前者，顺序与 C 一致
        for k in 1..=ncfft / 2 {
            let fpk = tmp[k];
            let fpnk = Cpx {
                re: tmp[ncfft - k].re,
                im: -tmp[ncfft - k].im,
            };
            let f1k = fpk.add(fpnk);
            let f2k = fpk.sub(fpnk);
            let tw = f2k.mul(self.super_twiddles[k - 1]);
            let a_re = (f1k.re + tw.re) * 0.5;
            let a_im = (f1k.im + tw.im) * 0.5;
            let b_re = (f1k.re - tw.re) * 0.5;
            let b_im = (tw.im - f1k.im) * 0.5;
            out[k] = a_re * a_re + a_im * a_im;
            out[ncfft - k] = b_re * b_re + b_im * b_im;
        }
    }
}

// ---------------------------------------------------------------- LFR / CMVN

/// `WavFrontend::apply_lfr`：首帧重复 (m-1)/2 次做左补齐，每 n 帧取 m 帧；
/// 尾部不足 m 帧时用最后一帧补满。
fn apply_lfr(inputs: &Array2<f32>, lfr_m: usize, lfr_n: usize) -> Result<Array2<f32>> {
    let t = inputs.nrows();
    let dim = inputs.ncols();
    if t == 0 {
        return Err(Error::Frontend("LFR 输入为空".to_string()));
    }
    let left = (lfr_m - 1) / 2;
    let total = t + left;
    let t_lfr = t.div_ceil(lfr_n);
    let mut out = Array2::<f32>::zeros((t_lfr, lfr_m * dim));
    for i in 0..t_lfr {
        let start = i * lfr_n;
        let mut dst = out.row_mut(i);
        for j in 0..lfr_m {
            let padded = start + j;
            let padded = padded.min(total - 1);
            let src = padded.saturating_sub(left);
            let src_row = inputs.row(src);
            for d in 0..dim {
                dst[j * dim + d] = src_row[d];
            }
        }
    }
    Ok(out)
}

/// `WavFrontend::apply_cmvn`：`(x + mean) * istd`。
///
/// numpy 里 `inputs` 是 float32、`cmvn` 是 float64，结果在 float64 下算完再落 float32，
/// 这里保持同样的提升顺序（`Cmvn` 受冻结契约限制只能存 f32）。
fn apply_cmvn(inputs: &Array2<f32>, cmvn: &Cmvn) -> Result<Array2<f32>> {
    let dim = inputs.ncols();
    if cmvn.mean.len() != cmvn.istd.len() {
        return Err(Error::Frontend(format!(
            "cmvn 维度不一致：mean {} / istd {}",
            cmvn.mean.len(),
            cmvn.istd.len()
        )));
    }
    if cmvn.mean.len() < dim {
        return Err(Error::Frontend(format!(
            "cmvn 维度 {} 小于特征维度 {dim}",
            cmvn.mean.len()
        )));
    }
    let mut out = inputs.to_owned();
    for mut row in out.rows_mut() {
        for (d, v) in row.iter_mut().enumerate() {
            *v = ((*v as f64 + cmvn.mean[d] as f64) * cmvn.istd[d] as f64) as f32;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_power(frame: &[f32]) -> Vec<f64> {
        let n = frame.len();
        let mut out = vec![0.0f64; n / 2 + 1];
        for (k, slot) in out.iter_mut().enumerate() {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (i, x) in frame.iter().enumerate() {
                let ang = -std::f64::consts::TAU * (k * i) as f64 / n as f64;
                re += *x as f64 * ang.cos();
                im += *x as f64 * ang.sin();
            }
            *slot = re * re + im * im;
        }
        out
    }

    /// 确定性伪随机（无依赖）：xorshift64*。
    fn pseudo_random(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s ^= s >> 12;
                s ^= s << 25;
                s ^= s >> 27;
                let v = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
                ((v >> 40) as f32 / 8_388_608.0) - 1.0
            })
            .collect()
    }

    #[test]
    fn fft_power_matches_naive_dft() {
        let frame = pseudo_random(512, 0x1234_5678);
        let rfft = RealFft::new(512);
        let mut got = vec![0.0f32; 257];
        rfft.power_spectrum(&frame, &mut got);
        let want = naive_power(&frame);
        let scale = want.iter().cloned().fold(0.0f64, f64::max);
        let mut worst = 0.0f64;
        for (g, w) in got.iter().zip(&want) {
            worst = worst.max((*g as f64 - *w).abs());
        }
        println!("fft vs dft: 相对谱峰值误差 {:.3e}", worst / scale);
        assert!(worst <= 1e-4 * scale, "误差 {worst} 超过 {scale} 的 1e-4");
    }

    #[test]
    fn fft_bin_energy_matches_sine_theory() {
        // cos(2π·64·n/512) 幅度 A 的单音：X[64] = A·N/2，其余 bin 为 0
        let n = 512;
        let k = 64;
        let amp = 0.5f32;
        let frame: Vec<f32> = (0..n)
            .map(|i| amp * (std::f64::consts::TAU * (k * i) as f64 / n as f64).cos() as f32)
            .collect();
        let rfft = RealFft::new(n);
        let mut ps = vec![0.0f32; n / 2 + 1];
        rfft.power_spectrum(&frame, &mut ps);
        let want = ((amp as f64) * (n as f64) / 2.0).powi(2);
        println!("sine theory: |X[{k}]|^2 = {} 实测 {}", want, ps[k]);
        assert!(
            (ps[k] as f64 - want).abs() <= 1e-3 * want,
            "bin {k} 能量 {} 与理论 {} 不符",
            ps[k],
            want
        );
        // 其余 bin（除对称的 n-k）应当接近 0
        let floor = want * 1e-6;
        for (i, v) in ps.iter().enumerate() {
            if i == k || i == n - k {
                continue;
            }
            assert!(*v <= floor as f32, "bin {i} 泄漏 {}", v);
        }
    }

    #[test]
    fn kiss_factorization_of_256_is_radix4() {
        assert_eq!(
            kf_factors(256),
            vec![(4, 64), (4, 16), (4, 4), (4, 1)],
            "kiss_fft 对 256 的分解应与 kf_factor 一致"
        );
    }

    #[test]
    fn mel_banks_match_kaldi_reference() {
        // 期望值由参考实现 kaldi_native_fbank(1.22.4) 的 MelBanks.get_matrix() dump 而来
        let mel = match MelBanks::new(80, 16_000.0, 512) {
            Ok(m) => m,
            Err(e) => panic!("构建 mel 失败：{e}"),
        };
        let row0 = &mel.bins[0];
        assert_eq!(row0.0, 1, "第 0 个 mel bin 的起点");
        assert!((row0.1[0] - 0.503_979_7).abs() < 1e-5, "{:?}", row0.1);
        assert!((row0.1[1] - 0.135_724_84).abs() < 1e-5, "{:?}", row0.1);
        let row40 = &mel.bins[40];
        assert_eq!(row40.0, 57);
        let want40 = [
            0.218_915_7,
            0.625_752_57,
            0.972_431_4,
            0.575_524_87,
            0.183_406_74,
        ];
        for (got, want) in row40.1.iter().zip(want40) {
            assert!((*got - want).abs() < 1e-5, "bin40 {got} vs {want}");
        }
        let row79 = &mel.bins[79];
        assert_eq!(row79.0 + row79.1.len() - 1, 255, "最高 bin 的终点");
    }

    #[test]
    fn lfr_pads_head_and_tail_like_reference() {
        // T=10, m=7, n=6 → T_lfr=2；首帧重复 3 次，第 2 帧刚好取满 7 帧
        let mut inputs = Array2::<f32>::zeros((10, 1));
        for i in 0..10 {
            inputs[[i, 0]] = i as f32;
        }
        let out = match apply_lfr(&inputs, 7, 6) {
            Ok(o) => o,
            Err(e) => panic!("apply_lfr 失败：{e}"),
        };
        assert_eq!(out.nrows(), 2);
        assert_eq!(out.ncols(), 7);
        assert_eq!(
            out.row(0).to_vec(),
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            "首帧左补齐 3 帧"
        );
        assert_eq!(
            out.row(1).to_vec(),
            vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "第 2 帧从第 6 个补齐帧起正好取满"
        );

        // T=8：尾部不足一帧时用最后一帧补满（3,4,5,6,7 + 7,7）
        let mut short = Array2::<f32>::zeros((8, 1));
        for i in 0..8 {
            short[[i, 0]] = i as f32;
        }
        let out = match apply_lfr(&short, 7, 6) {
            Ok(o) => o,
            Err(e) => panic!("apply_lfr 失败：{e}"),
        };
        assert_eq!(out.nrows(), 2);
        assert_eq!(
            out.row(1).to_vec(),
            vec![3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0],
            "尾部用最后一帧补满"
        );
    }

    fn write_cmvn(text: &str) -> Result<(tempfile::TempDir, std::path::PathBuf)> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("am.mvn");
        std::fs::write(&path, text)?;
        Ok((dir, path))
    }

    #[test]
    fn load_cmvn_parses_kaldi_text_format() {
        let text = "<Nnet> \n<Splice> 3 3\n[ 0 ]\n<AddShift> 3 3 \n\
                    <LearnRateCoef> 0 [ -1.5 0.25 2 ] \n<Rescale> 3 3\n\
                    <LearnRateCoef> 0 [ 0.5 2 0.125 ] \n";
        let (_dir, path) = match write_cmvn(text) {
            Ok(v) => v,
            Err(e) => panic!("准备 am.mvn 失败：{e}"),
        };
        let cmvn = match load_cmvn(&path) {
            Ok(c) => c,
            Err(e) => panic!("load_cmvn 失败：{e}"),
        };
        assert_eq!(cmvn.mean, vec![-1.5, 0.25, 2.0]);
        assert_eq!(cmvn.istd, vec![0.5, 2.0, 0.125]);
    }

    #[test]
    fn load_cmvn_rejects_broken_files() {
        let (_d1, p1) = match write_cmvn("<AddShift> 2 2\n<LearnRateCoef> 0 [ 1 ]\n") {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        assert!(load_cmvn(&p1).is_err(), "缺 <Rescale> 应报错");

        let (_d2, p2) = match write_cmvn(
            "<AddShift> 2 2\n<LearnRateCoef> 0 [ 1 2 ]\n<Rescale> 2 2\n<LearnRateCoef> 0 [ 1 ]\n",
        ) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        assert!(load_cmvn(&p2).is_err(), "维度不一致应报错");

        assert!(load_cmvn(Path::new("definitely/not/here.mvn")).is_err());
    }

    #[test]
    fn extract_features_rejects_bad_inputs() {
        let cfg = sensevoice_frontend();
        assert!(extract_features(&[0i16; 100], 8_000, &cfg, None).is_err());
        // 不足一帧
        assert!(extract_features(&[0i16; 399], 16_000, &cfg, None).is_err());
        let mut dithered = sensevoice_frontend();
        dithered.dither = 1.0;
        assert!(extract_features(&[0i16; 1600], 16_000, &dithered, None).is_err());
    }

    #[test]
    fn presets_match_reference_configs() {
        for cfg in [sensevoice_frontend(), paraformer_frontend()] {
            assert_eq!(cfg.n_mels, 80);
            assert_eq!(cfg.frame_length_ms, 25.0);
            assert_eq!(cfg.frame_shift_ms, 10.0);
            assert_eq!(cfg.preemphasis, 0.97);
            assert_eq!(cfg.dither, 0.0);
            assert_eq!((cfg.lfr_m, cfg.lfr_n), (7, 6));
        }
    }
}
