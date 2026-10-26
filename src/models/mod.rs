//! T4：模型注册表（数据已按原项目 registry.py 填好）与下载管理。
//! 移植自 funasr-subtitle `backend/app/models/registry.py` 与 `manager.py`。

pub mod manager;

use crate::error::{Error, Result};
use crate::types::EngineKind;

/// 模型任务类别。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTask {
    Asr,
    Vad,
    Punc,
    Spk,
}

/// 模型描述。字段语义与原项目 ModelSpec 一一对应。
#[derive(Debug, Clone, Copy)]
pub struct ModelSpec {
    /// 对外稳定 id（= ModelScope repo id）。
    pub id: &'static str,
    pub name: &'static str,
    pub task: ModelTask,
    pub size_mb: f64,
    pub repo: &'static str,
    /// 主仓库缺失、需从别处补的文件：(文件名, 来源 repo)。
    pub aux_files: &'static [(&'static str, &'static str)],
    /// ASR 输出是否自带标点（自带则跳过 ct-punc）。
    pub builtin_punc: bool,
    /// 引擎种类；VAD/PUNC 等辅助模型该字段无意义。
    pub engine: EngineKind,
    pub supports_hotwords: bool,
    /// 下载后模型目录内的复制/改名：(源文件名, 目标文件名)。
    pub local_copies: &'static [(&'static str, &'static str)],
    /// 直链下载（非 ModelScope 托管的单个文件）：`(完整 URL, 文件大小字节数)`。
    /// 设置后 `repo`/`aux_files`/`local_copies` 不参与下载（如 cam++ speaker 模型）。
    pub direct: Option<(&'static str, u64)>,
}

/// SenseVoiceSmall（多语言·默认·快），自带标点。
pub const SENSEVOICE: ModelSpec = ModelSpec {
    id: "iic/SenseVoiceSmall-onnx",
    name: "SenseVoiceSmall（多语言·默认·快）",
    task: ModelTask::Asr,
    size_mb: 235.0,
    repo: "iic/SenseVoiceSmall-onnx",
    aux_files: &[(
        "chn_jpn_yue_eng_ko_spectok.bpe.model",
        "iic/SenseVoiceSmall",
    )],
    builtin_punc: true,
    engine: EngineKind::SenseVoice,
    supports_hotwords: false,
    local_copies: &[],
    direct: None,
};

/// Paraformer-zh（中文高精度）。
pub const PARAFORMER_ZH: ModelSpec = ModelSpec {
    id: "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx",
    name: "Paraformer-zh（中文高精度）",
    task: ModelTask::Asr,
    size_mb: 240.0,
    repo: "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx",
    aux_files: &[],
    builtin_punc: false,
    engine: EngineKind::Paraformer,
    supports_hotwords: false,
    local_copies: &[],
    direct: None,
};

/// Paraformer-zh 热词版（需把 model_eb.onnx 复制为 model_eb_quant.onnx）。
pub const PARAFORMER_ZH_HOTWORD: ModelSpec = ModelSpec {
    id: "iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx",
    name: "Paraformer-zh 热词版（中文高精度·支持热词）",
    task: ModelTask::Asr,
    size_mb: 900.0,
    repo: "iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx",
    aux_files: &[],
    builtin_punc: false,
    engine: EngineKind::Contextual,
    supports_hotwords: true,
    local_copies: &[("model_eb.onnx", "model_eb_quant.onnx")],
    direct: None,
};

/// FSMN-VAD（所有 ASR 共享）。
pub const VAD_MODEL: ModelSpec = ModelSpec {
    id: "iic/speech_fsmn_vad_zh-cn-16k-common-onnx",
    name: "FSMN-VAD",
    task: ModelTask::Vad,
    size_mb: 2.0,
    repo: "iic/speech_fsmn_vad_zh-cn-16k-common-onnx",
    aux_files: &[],
    builtin_punc: false,
    engine: EngineKind::SenseVoice, // 辅助模型，engine 字段无意义
    supports_hotwords: false,
    local_copies: &[],
    direct: None,
};

/// CT-Punc 标点（Paraformer 必用；SenseVoice 懒加载兜底）。
pub const PUNC_MODEL: ModelSpec = ModelSpec {
    id: "iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx",
    name: "CT-Punc",
    task: ModelTask::Punc,
    size_mb: 290.0,
    repo: "iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx",
    aux_files: &[],
    builtin_punc: false,
    engine: EngineKind::SenseVoice, // 辅助模型，engine 字段无意义
    supports_hotwords: false,
    local_copies: &[],
    direct: None,
};

/// CAM++ 说话人嵌入（说话人分离用）。
///
/// 不在 ModelScope 托管 ONNX：用的是 lovemefan/campplus 从 damo
/// `speech_campplus_sv_zh-cn_16k-common` 导出的 ONNX（输入 `fbank`(1,T,80)，
/// 输出 192 维嵌入；等价性验证见 reference/PINNED_DIAR_VERSIONS.txt）。
pub const SPK_MODEL: ModelSpec = ModelSpec {
    id: "campplus/speaker-diarization-onnx",
    name: "CAM++ 说话人分离",
    task: ModelTask::Spk,
    size_mb: 26.6,
    repo: "campplus/speaker-diarization-onnx",
    aux_files: &[],
    builtin_punc: false,
    engine: EngineKind::SenseVoice, // 辅助模型，engine 字段无意义
    supports_hotwords: false,
    local_copies: &[],
    direct: Some((
        "https://raw.githubusercontent.com/lovemefan/campplus/main/campplus/onnx/campplus.onnx",
        27_844_629,
    )),
};

/// 面向用户的 ASR 模型目录。
pub const ASR_MODELS: &[ModelSpec] = &[SENSEVOICE, PARAFORMER_ZH, PARAFORMER_ZH_HOTWORD];

pub const DEFAULT_MODEL_ID: &str = SENSEVOICE.id;

/// 全部模型（含辅助）。
pub const ALL_MODELS: &[ModelSpec] = &[
    SENSEVOICE,
    PARAFORMER_ZH,
    PARAFORMER_ZH_HOTWORD,
    VAD_MODEL,
    PUNC_MODEL,
    SPK_MODEL,
];

/// 按 id 查模型（含辅助模型）。
pub fn get_model(model_id: &str) -> Result<&'static ModelSpec> {
    ALL_MODELS
        .iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| Error::NotFound(format!("model '{model_id}' is not registered")))
}
