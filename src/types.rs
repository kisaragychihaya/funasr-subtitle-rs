//! 冻结契约：所有模块共用的公共类型。实现任务不得修改本文件。

use serde::{Deserialize, Serialize};

/// 一条字幕 cue。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Segment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    /// 说话人标签（diarization 开启时存在），格式 `spk{整数}`，对应原项目 Segment.speaker。
    pub speaker: Option<String>,
}

/// 带绝对时间（毫秒）的文本片；SenseVoice 路径下时间为真实 CTC 帧位置推出。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimedPiece {
    pub text: String,
    pub time_ms: i64,
}

/// 语言选择，对应原项目 JobOptions.language（auto|zh|yue|en|ja|ko）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Language {
    #[default]
    Auto,
    Zh,
    Yue,
    En,
    Ja,
    Ko,
}

/// 标点模式：Auto=中文加标点、英文按停顿；On=都加；Off=都不加。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PunctMode {
    #[default]
    Auto,
    On,
    Off,
}

/// ASR 引擎种类，由模型 registry 的 engine 字段决定。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    SenseVoice,
    Paraformer,
    Contextual,
}

/// 转写任务选项（对应原项目 JobOptions）。
#[derive(Debug, Clone)]
pub struct JobOptions {
    pub language: Language,
    /// 每行最大显示宽度；0 = 不限制。CJK 记 1、拉丁/数字记 0.5。
    pub max_chars: usize,
    /// 单条字幕最长时长（毫秒）。
    pub max_cue_ms: i64,
    pub punct: PunctMode,
    /// 热词，空格分隔（仅 contextual 模型使用）。
    pub hotwords: String,
    /// 说话人分离（cam++）。上游 full 引擎仅 Paraformer 系支持；Rust 实现对全部引擎开放。
    pub diarization: bool,
    /// 预设说话人数（对应 FunASR `preset_spk_num`；`None` = 自动 eigengap 估计）。
    pub preset_spk_num: Option<usize>,
}

impl Default for JobOptions {
    fn default() -> Self {
        Self {
            language: Language::Auto,
            max_chars: 30,
            max_cue_ms: 7000,
            punct: PunctMode::Auto,
            hotwords: String::new(),
            diarization: false,
            preset_spk_num: None,
        }
    }
}
