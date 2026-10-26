//! 模型下载与缓存管理。对应 funasr-subtitle `backend/app/models/manager.py`。

use std::path::{Path, PathBuf};

use crate::error::Result;
use crate::models::ModelSpec;

/// 模型目录管理：确保模型文件在本地（缺则经 HTTP 从 ModelScope 下载）。
pub struct ModelManager {
    root: PathBuf,
}

impl ModelManager {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// 确保模型已下载（含 aux_files 补拉与 local_copies 复制/改名），返回模型目录。
    pub fn ensure_model(&self, spec: &ModelSpec) -> Result<PathBuf> {
        let _ = spec;
        todo!("T4: 移植 manager.py 下载/缓存逻辑")
    }
}

/// 默认模型缓存目录：Windows 为 %LOCALAPPDATA%\\funasr-subtitle-rs\\models，其余 ~/.cache/funasr-subtitle-rs/models。
pub fn default_models_root() -> PathBuf {
    todo!("T4")
}
