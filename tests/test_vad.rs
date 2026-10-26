//! T7 验收测试：FSMN-VAD（`src/vad/mod.rs`）。
//!
//! 1) 真实模型：`Vad::new` + `infer` 跑 3 个 fixture 的 `input.s16bin`，与
//!    `tests/golden/vad/{fixture}/spans.json` 逐段毫秒级比对（段数、每段 start/end 全等）。
//!    模型只有 ~500KB，所以不 `#[ignore]`；`FUNASR_SUBTITLE_MODELS_ROOT` 指向已有的模型
//!    根目录即可跳过下载。
//! 2) 纯逻辑：`WindowDetector` 的滑窗翻转、`frame_decibel` 的电平、`silence_lookback_frames`
//!    的回看帧数、`E2EVad::call` 的段合并/收尾。期望值都是用参考实现
//!    `reference/funasr_onnx/utils/e2e_vad.py` 直接驱动出来的（不跑 ONNX），见各用例注释。

use std::fs;
use std::path::{Path, PathBuf};

use funasr_subtitle_rs::models::VAD_MODEL;
use funasr_subtitle_rs::models::manager::{ModelManager, default_models_root, has_onnx, model_dir};
use funasr_subtitle_rs::vad::{
    AudioChangeState, E2EVad, FrameState, Vad, WindowDetector, frame_decibel,
    silence_lookback_frames,
};

// ---------------------------------------------------------------------------
// 通用小工具
// ---------------------------------------------------------------------------

fn manifest_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn read_json<T: serde::de::DeserializeOwned>(rel: &str) -> T {
    let path = manifest_path(rel);
    let raw =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("读取 {} 失败: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("解析 {} 失败: {e}", path.display()))
}

/// 读 16k 单声道 s16le 原始采样（golden 的 `input.s16bin`）。
fn read_s16(rel: &str) -> Vec<i16> {
    let path = manifest_path(rel);
    let raw = fs::read(&path).unwrap_or_else(|e| panic!("读取 {} 失败: {e}", path.display()));
    raw.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// 模型根目录：默认 `default_models_root()`；`FUNASR_SUBTITLE_MODELS_ROOT` 可覆盖。
fn models_root() -> PathBuf {
    std::env::var_os("FUNASR_SUBTITLE_MODELS_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(default_models_root)
}

/// 定位 FSMN-VAD 模型目录：优先用已下载好的缓存，缺失时下载到临时目录（约 500KB）。
fn vad_model_dir(tmp: &mut Option<tempfile::TempDir>) -> PathBuf {
    let cached = model_dir(&models_root(), &VAD_MODEL);
    if has_onnx(&cached) {
        return cached;
    }
    let dir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(e) => panic!("创建临时模型目录失败: {e}"),
    };
    let path = ModelManager::new(dir.path().to_path_buf())
        .ensure_model(&VAD_MODEL)
        .unwrap_or_else(|e| panic!("下载 FSMN-VAD 模型失败（需要联网）: {e}"));
    *tmp = Some(dir);
    path
}

// ---------------------------------------------------------------------------
// 真实模型：3 个 fixture 与 golden 逐段对拍
// ---------------------------------------------------------------------------

#[test]
fn test_vad_real_model_matches_golden_spans() {
    let mut tmp = None;
    let dir = vad_model_dir(&mut tmp);
    println!("FSMN-VAD 模型目录: {}", dir.display());
    let mut vad = Vad::new(&dir, 4).unwrap_or_else(|e| panic!("加载 FSMN-VAD 失败: {e}"));

    let fixtures = ["sample_zh", "sample_en_runon", "sample_long"];
    let mut failures = Vec::new();
    for fixture in fixtures {
        let wave = read_s16(&format!("tests/golden/input/{fixture}/input.s16bin"));
        let golden: Vec<(i64, i64)> = read_json(&format!("tests/golden/vad/{fixture}/spans.json"));
        let got = vad
            .infer(&wave, 16_000)
            .unwrap_or_else(|e| panic!("{fixture} 推理失败: {e}"));
        println!(
            "{fixture}: {} 采样, golden={golden:?}, got={got:?}",
            wave.len()
        );
        if got != golden {
            failures.push(format!("{fixture}: golden={golden:?} got={got:?}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} / {} 个 fixture 的 VAD 段与 golden 不一致:\n{}",
        failures.len(),
        fixtures.len(),
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// 纯逻辑：滑窗检测（WindowDetector）
// ---------------------------------------------------------------------------

#[test]
fn test_window_detector_boundary_flip() {
    // 参考实现（WindowDetector(200, 150, 150, 10)，即窗 20 帧、双向阈值都 15 帧）跑
    // 3 静音 + 20 语音 + 6 静音 + 3 语音 + 30 静音，翻转发生在 i=17 与 i=27：
    //   i=17 时窗内正好 15 个语音帧（win_sum == 15 >= 15）-> Sil2Speech
    //   i=27 时窗内正好 15 个语音帧（win_sum == 15 <= 15）-> Speech2Sil
    // 也就是阈值相等时两个方向都可能成立，谁生效取决于当前的 pre_frame_state。
    let mut wd = WindowDetector::new(200, 150, 150, 10);
    assert_eq!(wd.win_size(), 20, "窗长 = 200ms / 10ms");

    let mut seq = Vec::new();
    seq.extend(std::iter::repeat_n(FrameState::Sil, 3));
    seq.extend(std::iter::repeat_n(FrameState::Speech, 20));
    seq.extend(std::iter::repeat_n(FrameState::Sil, 6));
    seq.extend(std::iter::repeat_n(FrameState::Speech, 3));
    seq.extend(std::iter::repeat_n(FrameState::Sil, 30));

    let mut flips: Vec<(usize, AudioChangeState)> = Vec::new();
    for (i, state) in seq.iter().enumerate() {
        let change = wd.detect_one_frame(*state, i as i64);
        if change != AudioChangeState::SilToSil && change != AudioChangeState::SpeechToSpeech {
            flips.push((i, change));
        }
    }
    assert_eq!(
        flips,
        vec![
            (17, AudioChangeState::SilToSpeech),
            (27, AudioChangeState::SpeechToSil)
        ],
        "滑窗翻转位置与参考实现不一致"
    );

    // Invalid 帧不改窗状态
    assert_eq!(
        wd.detect_one_frame(FrameState::Invalid, seq.len() as i64),
        AudioChangeState::Invalid
    );

    // reset 后窗与状态全清：同样的输入从头再来一遍，翻转点仍是 i=17
    wd.reset();
    let mut first_flip = None;
    for (i, state) in seq.iter().enumerate() {
        let change = wd.detect_one_frame(*state, i as i64);
        if first_flip.is_none()
            && change != AudioChangeState::SilToSil
            && change != AudioChangeState::SpeechToSpeech
        {
            first_flip = Some((i, change));
        }
    }
    assert_eq!(
        first_flip,
        Some((17, AudioChangeState::SilToSpeech)),
        "reset 后应从未翻转状态重新累计"
    );
}

// ---------------------------------------------------------------------------
// 纯逻辑：电平与静音回看
// ---------------------------------------------------------------------------

#[test]
fn test_frame_decibel_matches_reference() {
    // 期望值 = 参考实现的 10*log10(np.square(frame).sum() + 1e-6)（float32 求和后加地板）
    let zeros = vec![0.0f32; 400];
    assert!(
        (frame_decibel(&zeros) - (-60.0)).abs() < 1e-12,
        "全零帧应恰好是 10*log10(1e-6) = -60，实际 {}",
        frame_decibel(&zeros)
    );
    let half = vec![0.5f32; 400];
    let want_half = 20.000_000_043_429_445_f64;
    assert!(
        (frame_decibel(&half) - want_half).abs() < 1e-9,
        "0.5 恒定帧期望 {want_half}，实际 {}",
        frame_decibel(&half)
    );
    let tiny = vec![1e-5f32; 400];
    let want_tiny = -59.829_666_608_027_35_f64;
    // 这一条的容差放宽到 1e-6：numpy 的 float32 求和是成对累加、这里是顺序累加，400 项之后
    // 和值差 ~1e-8 相对量；电平只用于 -100dB 门槛判定，这点差对判决没有影响。
    assert!(
        (frame_decibel(&tiny) - want_tiny).abs() < 1e-6,
        "1e-5 恒定帧期望 {want_tiny}，实际 {}",
        frame_decibel(&tiny)
    );
}

#[test]
fn test_silence_lookback_frames() {
    // 模型实参：max_end_sil 800 - speech_to_sil 150 = 650 帧阈值 -> 65 帧回看
    // do_extend=1 再减 lookahead(100/10=10) 与 1 -> 54
    assert_eq!(silence_lookback_frames(650, 10, true, 100), 54);
    assert_eq!(silence_lookback_frames(650, 10, false, 100), 65);
    assert_eq!(silence_lookback_frames(155, 10, true, 100), 4);
    // 减到负数时夹到 0（`lookback_frame = max(0, lookback_frame)`）
    assert_eq!(silence_lookback_frames(100, 10, true, 200), 0);
}

// ---------------------------------------------------------------------------
// 纯逻辑：状态机的段输出（合成分数，不跑 ONNX）
// ---------------------------------------------------------------------------

/// 合成波形：220Hz 正弦、幅度 0.1（电平远高于 -100dB 门槛，逐帧判决只看分数）。
fn synthetic_waveform(frames: usize) -> Vec<f32> {
    let n = (frames - 1) * 160 + 400;
    (0..n)
        .map(|i| (0.1 * (std::f64::consts::TAU * 220.0 * i as f64 / 16_000.0).sin()) as f32)
        .collect()
}

/// 合成静音后验：默认 0.9（静音），`speech` 区间里是 0.05（语音）。
/// `(1-0.05) >= 0.05 + 0.6` 成立 -> 语音；`(1-0.9) >= 0.9 + 0.6` 不成立 -> 静音。
fn synthetic_sil_scores(frames: usize, speech: &[(usize, usize)]) -> Vec<f32> {
    let mut scores = vec![0.9f32; frames];
    for (start, end) in speech {
        for v in scores.iter_mut().take(*end).skip(*start) {
            *v = 0.05;
        }
    }
    scores
}

fn call_vad(m: &mut E2EVad, scores: &[f32], wave: &[f32], is_final: bool) -> Vec<(i64, i64)> {
    m.call(scores, wave, is_final, 800)
        .unwrap_or_else(|e| panic!("状态机出段失败: {e}"))
}

#[test]
fn test_state_machine_segments_match_reference() {
    // 用例期望值全部来自参考实现 E2EVadModel(MODEL_CONF) 直接吃合成 scores/waveform。

    // (1) 两段语音但间隔 400ms < max_end_silence 800ms -> 合并成一段；
    //     起点 = 第 15 个语音帧(114) - 起点回看 40 帧 = 74 -> 740ms；
    //     is_final 收尾在最后一帧 319 -> (319+1)*10 = 3200ms。
    let mut m = E2EVad::offline();
    let wave = synthetic_waveform(320);
    let scores = synthetic_sil_scores(320, &[(100, 160), (200, 260)]);
    assert_eq!(
        call_vad(&mut m, &scores, &wave, true),
        vec![(740, 3200)],
        "间隔不足 max_end_silence 的两段应合并"
    );

    // (2) 全静音 + is_final：只走 fake 的起止点，不产出任何段。
    let mut m = E2EVad::offline();
    let wave = synthetic_waveform(200);
    let scores = synthetic_sil_scores(200, &[]);
    assert_eq!(call_vad(&mut m, &scores, &wave, true), Vec::new());

    // (3) 语音长于 max_single_segment_time(20000ms=2000 帧) -> 在起点后第 2000 帧强行切段，
    //     后半段从 20250ms 起；末段按静音回看 54 帧收尾于 25160ms。
    let mut m = E2EVad::offline();
    let wave = synthetic_waveform(2600);
    let scores = synthetic_sil_scores(2600, &[(50, 2500)]);
    assert_eq!(
        call_vad(&mut m, &scores, &wave, true),
        vec![(240, 20250), (20250, 25160)],
        "超长段应按 max_single_segment_time 切开"
    );
}

#[test]
fn test_state_machine_chunked_call_matches_reference() {
    // 分两块喂（第一块非 final，第二块 final）：验证 idx_pre_chunk 的块内下标换算与
    // decibel 的跨块累加。波形切片照抄 vad_bin.py：
    //   块 k：wave[t_offset*160 : min(len, (t_offset+step-1)*160 + 400)]
    // 参考实现结果：第一块 []，第二块 [(1240, 2660), (2940, 4000)]。
    let step = 200usize;
    let wave = synthetic_waveform(2 * step);
    let mut scores = synthetic_sil_scores(step, &[(150, 200)]);
    scores.extend(synthetic_sil_scores(step, &[(0, 50), (120, 200)]));

    let w1_start = 0;
    let w1_end = (step - 1) * 160 + 400;
    let w2_start = step * 160;
    let w2_end = (2 * step - 1) * 160 + 400;
    assert_eq!(
        (w1_start, w1_end, w2_start, w2_end),
        (0, 32240, 32000, 64240)
    );

    let mut m = E2EVad::offline();
    assert_eq!(
        call_vad(&mut m, &scores[..step], &wave[w1_start..w1_end], false),
        Vec::new(),
        "第一块还没闭合的段不应吐出"
    );
    assert_eq!(
        call_vad(&mut m, &scores[step..], &wave[w2_start..w2_end], true),
        vec![(1240, 2660), (2940, 4000)],
        "跨块的段起点/终点与参考实现不一致"
    );
}
