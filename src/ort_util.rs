//! 进程级共享的 ONNX Runtime 基础设施。
//! ort 每进程只允许一个 `Environment`；所有模型模块（vad / asr / punc）必须共用本模块，
//! 禁止各自再持 OnceLock（跨模块同进程加载会建不出第二个 Environment）。

use std::path::Path;
use std::sync::OnceLock;

use ort::environment::Environment;
use ort::session::{Session, builder::GraphOptimizationLevel};

use crate::error::{Error, Result};

/// 进程级 ONNX Runtime 环境（ort 每进程只允许一个 `Environment`）。
pub fn ort_environment() -> Result<Environment> {
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

/// 建 session：线程数、图优化、日志级别对齐 funasr-onnx 的 `OrtInferSession`
///（intra_op 线程可配、GraphOptimizationLevel::Level3、log_severity_level=4 即只留 FATAL）。
pub fn build_session(model_file: &Path, threads: usize) -> Result<Session> {
    let env = ort_environment()?;
    let builder =
        Session::builder(&env).map_err(|e| Error::Model(format!("创建 ONNX Session 失败: {e}")))?;
    let mut builder = builder
        .with_intra_threads(threads.max(1))
        .map_err(|e| Error::Model(format!("设置 intra_op 线程数失败: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| Error::Model(format!("设置图优化级别失败: {e}")))?
        .with_log_level(ort::logging::LogLevel::Fatal)
        .map_err(|e| Error::Model(format!("设置日志级别失败: {e}")))?;
    builder
        .commit_from_file(model_file)
        .map_err(|e| Error::Model(format!("加载 ONNX 模型 {} 失败: {e}", model_file.display())))
}
