//! 模型下载与缓存管理。对应 funasr-subtitle `backend/app/models/manager.py`。
//!
//! 移植说明：原实现依赖 `modelscope.snapshot_download`（内部带 torch-stub 规避），
//! 这里改为直接打 ModelScope 的文件列表 API + `resolve/master` 单文件下载地址，
//! 语义保持一致：目录布局 `<root>/<model_id>`、忽略 PyTorch 权重、已存在即跳过、
//! 补拉 aux_files、目录内 local_copies 复制。

use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use serde::Deserialize;

use crate::error::{Error, Result};
use crate::models::ModelSpec;

/// 应用名，参与默认缓存目录与 UA。
const APP_DIR: &str = "funasr-subtitle-rs";
/// ModelScope 仓库文件列表 API（`{repo}` 形如 `iic/xxx`）。
const FILES_API_URL: &str = "https://modelscope.cn/api/v1/models/{repo}/repo/files?Recursive=true";
/// ModelScope 单文件下载地址（服务端 302 到 CDN，reqwest 默认跟随）。
const RESOLVE_URL: &str = "https://modelscope.cn/models/{repo}/resolve/master/{path}";
/// 与 manager.py 的 `_IGNORE_PATTERNS` 一致：跳过 PyTorch 权重等大文件。
const IGNORED_SUFFIXES: &[&str] = &[".pt", ".pth", ".bin", ".safetensors", ".onnx_data.bak"];
/// 下载分块大小。
const CHUNK_BYTES: usize = 64 * 1024;

/// 仓库文件列表里的一项（只保留我们需要的字段）。
#[derive(Debug, Clone, Deserialize)]
pub struct RepoFile {
    #[serde(rename = "Path")]
    pub path: String,
    #[serde(rename = "Size", default)]
    pub size: u64,
    #[serde(rename = "Type", default)]
    pub kind: String,
}

impl RepoFile {
    /// 普通文件（`Type` 为 `blob`；老接口缺该字段时按文件处理）。
    pub fn is_blob(&self) -> bool {
        self.kind.is_empty() || self.kind == "blob"
    }
}

/// 一个待下载文件：从 `repo` 的 `path` 下载到模型目录内的 `dest_name`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadItem {
    pub repo: String,
    pub path: String,
    pub size: u64,
    /// 模型目录内的相对目标路径（aux 文件只有文件名，主仓库文件保留子目录）。
    pub dest_name: String,
}

impl DownloadItem {
    /// 落到磁盘上的绝对目标路径。
    pub fn dest(&self, target: &Path) -> PathBuf {
        dest_for(target, &self.dest_name)
    }
}

#[derive(Debug, Deserialize)]
struct FilesResponse {
    #[serde(rename = "Data")]
    data: Option<FilesData>,
    #[serde(rename = "Code", default)]
    code: Option<i64>,
    #[serde(rename = "Message", default)]
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FilesData {
    #[serde(rename = "Files")]
    files: Vec<RepoFile>,
}

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
        let target = model_dir(&self.root, spec);
        fs::create_dir_all(&target)
            .map_err(|e| Error::Model(format!("创建模型目录 {} 失败: {e}", target.display())))?;

        let client = http_client()?;

        let files = self.fetch_file_list(&client, spec.repo)?;
        let mut items = pending_repo_downloads(spec.repo, &files, &target);

        // aux_files：主仓库缺失的文件从来源 repo 补拉。
        for (fname, src_repo) in spec.aux_files {
            if dest_for(&target, fname).is_file() {
                continue;
            }
            let src_files = self.fetch_file_list(&client, src_repo)?;
            let found = find_file(&src_files, fname).ok_or_else(|| {
                Error::Model(format!(
                    "模型 {} 缺辅助文件 {fname}：来源仓库 {src_repo} 中未找到",
                    spec.id
                ))
            })?;
            items.push(DownloadItem {
                repo: (*src_repo).to_string(),
                path: found.path.clone(),
                size: found.size,
                dest_name: (*fname).to_string(),
            });
        }

        if !items.is_empty() {
            let total: u64 = items.iter().map(|i| i.size).sum();
            let bar = progress_bar(total, spec.name);
            for (idx, item) in items.iter().enumerate() {
                bar.set_message(format!("{}/{} {}", idx + 1, items.len(), item.dest_name));
                download_item(&client, item, &target, &bar)?;
            }
            bar.finish_with_message(format!("{} 就绪", spec.id));
        }

        // local_copies：目录内复制/改名（如 contextual 的 model_eb_quant.onnx）。
        apply_local_copies(spec, &target)?;

        // 兜底校验：与原项目 is_downloaded 一致，目录里必须至少有一个 .onnx 权重。
        if !has_onnx(&target) {
            return Err(Error::Model(format!(
                "模型 {} 下载完成但目录 {} 内没有任何 .onnx 权重",
                spec.id,
                target.display()
            )));
        }

        Ok(target)
    }

    fn fetch_file_list(&self, client: &Client, repo: &str) -> Result<Vec<RepoFile>> {
        let url = files_api_url(repo);
        let resp = client
            .get(&url)
            .send()
            .map_err(|e| Error::Model(format!("获取仓库 {repo} 文件列表失败: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(Error::Model(format!(
                "获取仓库 {repo} 文件列表返回 HTTP {status}"
            )));
        }
        let body = resp
            .text()
            .map_err(|e| Error::Model(format!("读取仓库 {repo} 文件列表失败: {e}")))?;
        parse_repo_files(repo, &body)
    }
}

/// 默认模型缓存目录：Windows 为 `%LOCALAPPDATA%\funasr-subtitle-rs\models`，其余 `~/.cache/funasr-subtitle-rs/models`。
pub fn default_models_root() -> PathBuf {
    let local_app_data = std::env::var_os("LOCALAPPDATA");
    let xdg_cache_home = std::env::var_os("XDG_CACHE_HOME");
    let home = std::env::var_os("HOME");
    resolve_models_root(
        cfg!(windows),
        local_app_data.as_deref(),
        xdg_cache_home.as_deref(),
        home.as_deref(),
    )
}

/// `default_models_root` 的纯逻辑（便于在两个平台分支上都做单测）。
pub fn resolve_models_root(
    is_windows: bool,
    local_app_data: Option<&OsStr>,
    xdg_cache_home: Option<&OsStr>,
    home: Option<&OsStr>,
) -> PathBuf {
    let local_app_data = if is_windows || local_app_data.is_some() {
        local_app_data.filter(|v| !v.is_empty()).map(PathBuf::from)
    } else {
        None
    };
    if let Some(base) = local_app_data {
        return base.join(APP_DIR).join("models");
    }
    let cache = xdg_cache_home
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            home.filter(|v| !v.is_empty())
                .map(|h| PathBuf::from(h).join(".cache"))
        })
        .unwrap_or_else(|| PathBuf::from(".cache"));
    cache.join(APP_DIR).join("models")
}

/// 仓库文件列表 API 地址。
pub fn files_api_url(repo: &str) -> String {
    FILES_API_URL.replace("{repo}", repo)
}

/// 单文件下载地址：`https://modelscope.cn/models/<repo>/resolve/master/<path>`。
pub fn file_url(repo: &str, path: &str) -> String {
    RESOLVE_URL.replace("{repo}", repo).replace("{path}", path)
}

/// 模型目录 = `<root>/<spec.id>`，id 里的 `/` 保留为子目录。
pub fn model_dir(root: &Path, spec: &ModelSpec) -> PathBuf {
    let mut dir = root.to_path_buf();
    for seg in spec.id.split('/') {
        if is_safe_segment(seg) {
            dir.push(seg);
        }
    }
    dir
}

/// 模型目录内的相对路径 → 绝对路径（按 `/` 分段，跨平台安全）。
pub fn dest_for(target: &Path, rel: &str) -> PathBuf {
    let mut dest = target.to_path_buf();
    for seg in rel.split('/') {
        if is_safe_segment(seg) {
            dest.push(seg);
        }
    }
    dest
}

/// 下载中的临时文件名：`<name>.part`。
pub fn part_path(dest: &Path) -> PathBuf {
    let mut name = dest
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(".part");
    dest.with_file_name(name)
}

/// 目标文件已就绪（存在且非空）。
pub fn is_complete(dest: &Path) -> bool {
    fs::metadata(dest)
        .map(|m| m.is_file() && m.len() > 0)
        .unwrap_or(false)
}

/// 是否需要下载（幂等判断的唯一入口）。
pub fn needs_download(dest: &Path) -> bool {
    !is_complete(dest)
}

/// 模型目录内是否已有 ONNX 权重（对齐 manager.py `is_downloaded` 的 `*.onnx` 门槛）。
pub fn has_onnx(dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    entries.filter_map(|e| e.ok()).any(|e| {
        e.path()
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"))
    })
}

/// 命中 `_IGNORE_PATTERNS` 的文件（PyTorch 权重等）。
pub fn is_ignored_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    IGNORED_SUFFIXES
        .iter()
        .any(|suffix| lower.ends_with(suffix))
}

/// 解析 ModelScope 文件列表响应；`Success: false` / 缺 `Data` 视为错误。
pub fn parse_repo_files(repo: &str, body: &str) -> Result<Vec<RepoFile>> {
    let resp: FilesResponse = serde_json::from_str(body)
        .map_err(|e| Error::Model(format!("解析仓库 {repo} 文件列表 JSON 失败: {e}")))?;
    match resp.data {
        Some(data) => Ok(data.files),
        None => Err(Error::Model(format!(
            "仓库 {repo} 文件列表响应异常（Code={:?}, Message={:?}）",
            resp.code, resp.message
        ))),
    }
}

/// 在文件列表里按文件名查找（aux_files 用；也允许带子目录的路径）。
pub fn find_file<'a>(files: &'a [RepoFile], name: &str) -> Option<&'a RepoFile> {
    files
        .iter()
        .find(|f| f.is_blob() && (f.path == name || f.path.ends_with(&format!("/{name}"))))
}

/// 主仓库仍缺的文件（跳过非 blob、空文件、忽略列表、以及已存在且非空的目标）。
pub fn pending_repo_downloads(repo: &str, files: &[RepoFile], target: &Path) -> Vec<DownloadItem> {
    files
        .iter()
        .filter(|f| f.is_blob() && f.size > 0 && !is_ignored_path(&f.path))
        .filter(|f| needs_download(&dest_for(target, &f.path)))
        .map(|f| DownloadItem {
            repo: repo.to_string(),
            path: f.path.clone(),
            size: f.size,
            dest_name: f.path.clone(),
        })
        .collect()
}

/// 目录内复制/改名（local_copies）：目标已存在则跳过，源缺失则报错。
pub fn apply_local_copies(spec: &ModelSpec, target: &Path) -> Result<()> {
    for (src_name, dst_name) in spec.local_copies {
        let src = dest_for(target, src_name);
        let dst = dest_for(target, dst_name);
        if dst.exists() {
            continue;
        }
        if !src.is_file() {
            return Err(Error::Model(format!(
                "模型 {} 的本地复制源 {src_name} 缺失",
                spec.id
            )));
        }
        fs::copy(&src, &dst).map_err(|e| {
            Error::Model(format!(
                "复制 {} → {} 失败: {e}",
                src.display(),
                dst.display()
            ))
        })?;
    }
    Ok(())
}

fn is_safe_segment(seg: &str) -> bool {
    !seg.is_empty() && seg != "." && seg != ".."
}

fn http_client() -> Result<Client> {
    Client::builder()
        .user_agent(concat!("funasr-subtitle-rs/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| Error::Model(format!("初始化 HTTP 客户端失败: {e}")))
}

fn progress_bar(total: u64, label: &str) -> ProgressBar {
    let style = ProgressStyle::with_template(
        "{prefix} [{bar:30}] {bytes}/{total_bytes} ({percent}%) {msg}",
    )
    .map(|s| s.progress_chars("=>-"))
    .unwrap_or_else(|_| ProgressStyle::default_bar());
    let bar = ProgressBar::new(total);
    bar.set_style(style);
    bar.set_prefix(label.to_string());
    bar.enable_steady_tick(Duration::from_millis(200));
    bar
}

fn download_item(
    client: &Client,
    item: &DownloadItem,
    target: &Path,
    bar: &ProgressBar,
) -> Result<()> {
    let dest = item.dest(target);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| Error::Model(format!("创建目录 {} 失败: {e}", parent.display())))?;
    }
    let part = part_path(&dest);
    // 写 .part 再 rename：中断不会留下半截正式文件。
    if let Err(e) = stream_to_part(client, item, &part, bar) {
        let _ = fs::remove_file(&part);
        return Err(e);
    }
    fs::rename(&part, &dest).map_err(|e| {
        Error::Model(format!(
            "重命名 {} → {} 失败: {e}",
            part.display(),
            dest.display()
        ))
    })
}

fn stream_to_part(
    client: &Client,
    item: &DownloadItem,
    part: &Path,
    bar: &ProgressBar,
) -> Result<()> {
    let url = file_url(&item.repo, &item.path);
    let mut resp = client
        .get(&url)
        .send()
        .map_err(|e| Error::Model(format!("下载 {url} 失败: {e}")))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(Error::Model(format!(
            "下载 {}/{} 返回 HTTP {status}",
            item.repo, item.path
        )));
    }
    let mut file = File::create(part)
        .map_err(|e| Error::Model(format!("创建 {} 失败: {e}", part.display())))?;
    let mut buf = vec![0u8; CHUNK_BYTES];
    let mut written: u64 = 0;
    loop {
        let n = resp
            .read(&mut buf)
            .map_err(|e| Error::Model(format!("下载 {}/{} 读取中断: {e}", item.repo, item.path)))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| Error::Model(format!("写入 {} 失败: {e}", part.display())))?;
        written += n as u64;
        bar.inc(n as u64);
    }
    file.flush()
        .map_err(|e| Error::Model(format!("刷写 {} 失败: {e}", part.display())))?;
    if written == 0 {
        return Err(Error::Model(format!(
            "下载 {}/{} 得到空文件",
            item.repo, item.path
        )));
    }
    Ok(())
}
