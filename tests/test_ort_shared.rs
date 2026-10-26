//! 多引擎共存冒烟：同一进程内同时加载 VAD + Paraformer 两个模型。
//! ort 每进程只允许一个 `Environment`（`crate::ort_util` 共享）；
//! 本测试就是为锁住这个集成约束——回归表现是 "only one environment is allowed per process"。
//!
//! 运行：`FUNASR_SUBTITLE_MODELS_ROOT=<模型根> cargo test --test test_ort_shared -- --ignored`
//! （模型根可用 ModelManager 预下载；不设时会往默认缓存目录下载约 240MB）

use funasr_subtitle_rs::asr::{AsrChunkOutput, AsrEngine, Paraformer};
use funasr_subtitle_rs::models::manager::{default_models_root, ModelManager};
use funasr_subtitle_rs::models::{PARAFORMER_ZH, VAD_MODEL};
use funasr_subtitle_rs::vad::Vad;
use std::path::PathBuf;

fn models_root() -> PathBuf {
    std::env::var("FUNASR_SUBTITLE_MODELS_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_models_root())
}

#[test]
#[ignore = "需要 VAD+Paraformer 模型文件；设 FUNASR_SUBTITLE_MODELS_ROOT 后运行"]
fn vad_and_paraformer_coexist_in_one_process() {
    let root = models_root();
    let manager = ModelManager::new(root);

    let vad_dir = manager.ensure_model(&VAD_MODEL).expect("VAD 模型就绪");
    let asr_dir = manager
        .ensure_model(&PARAFORMER_ZH)
        .expect("Paraformer 模型就绪");

    // 同一进程先后建两个引擎的 session：第二个若自建 Environment 必失败。
    let mut vad = Vad::new(&vad_dir, 2).expect("VAD 加载");
    let mut asr = Paraformer::new(&asr_dir, 2).expect("Paraformer 加载");

    // 各跑一个最小推理，确认两个会话都真实可用。
    // 注意：不对 Paraformer 静音输出内容做强断言——非自回归模型对纯零输入可能产生
    // 少量幻觉 token，这是模型行为而非集成缺陷；本测试只锁「同进程共存 + 可推理」。
    let silence = vec![0i16; 16_000]; // 1 秒静音
    let spans = vad.infer(&silence, 16_000).expect("VAD 推理");
    assert!(spans.is_empty(), "纯静音不应切出语音段");
    let out = asr
        .transcribe_chunk(&silence, 16_000, &Default::default())
        .expect("Paraformer 推理");
    match out {
        AsrChunkOutput::Plain(_) => {}
        _ => panic!("Paraformer 应返回 Plain"),
    }
}
