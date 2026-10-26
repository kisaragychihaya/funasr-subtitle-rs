//! T4 验收测试：模型下载/缓存管理（`src/models/manager.rs`）。
//!
//! 纯逻辑单测（URL/路径拼接、幂等判断、JSON 解析、local_copies、默认目录）
//! + 一个需要联网的 `#[ignore]` 真实下载冒烟。

use std::fs;
use std::path::{Path, PathBuf};

use funasr_subtitle_rs::EngineKind;
use funasr_subtitle_rs::models::manager::{
    ModelManager, apply_local_copies, default_models_root, dest_for, file_url, files_api_url,
    find_file, has_onnx, is_complete, is_ignored_path, model_dir, needs_download, parse_repo_files,
    part_path, pending_repo_downloads, resolve_models_root,
};
use funasr_subtitle_rs::models::{
    ModelSpec, ModelTask, PARAFORMER_ZH_HOTWORD, SENSEVOICE, VAD_MODEL,
};

/// 与 curl 到的真实响应同构的最小样例（含 LFS 文件、忽略文件、目录条目）。
const VAD_FILES_JSON: &str = r#"{"Code":200,"Data":{"Files":[
{"CommitMessage":"upload model resource","CommittedDate":1687865493,"CommitterName":"speech_asr","InCheck":false,"IsLFS":false,"Mode":"33188","Name":".gitattributes","Path":".gitattributes","Revision":"f436483dfd1bf3eb1b410133ff86ecc9e10abc4a","Sha256":"7a50","Size":1467,"Type":"blob"},
{"CommitMessage":"upload model resource","CommittedDate":1687865493,"CommitterName":"speech_asr","InCheck":false,"IsLFS":true,"Mode":"33261","Name":"model_quant.onnx","Path":"model_quant.onnx","Revision":"f436483dfd1bf3eb1b410133ff86ecc9e10abc4a","Sha256":"5289","Size":506744,"Type":"blob"},
{"CommitMessage":"add weights","CommittedDate":1687865493,"CommitterName":"speech_asr","InCheck":false,"IsLFS":true,"Mode":"33261","Name":"model.pt","Path":"model.pt","Revision":"f436483dfd1bf3eb1b410133ff86ecc9e10abc4a","Sha256":"dead","Size":893000000,"Type":"blob"},
{"CommitMessage":"upload model resource","CommittedDate":1687865493,"CommitterName":"speech_asr","InCheck":false,"IsLFS":false,"Mode":"33188","Name":"sub","Path":"sub","Revision":"f436483dfd1bf3eb1b410133ff86ecc9e10abc4a","Sha256":"beef","Size":0,"Type":"tree"}
],"IsVisual":1},"Message":"success","RequestId":"0029981a","Success":true}"#;

/// local_copies 测试用 spec（contextual 的复制/改名场景，不联网）。
const COPY_SPEC: ModelSpec = ModelSpec {
    id: "test/local-copies",
    name: "test local copies",
    task: ModelTask::Asr,
    size_mb: 0.0,
    repo: "test/local-copies",
    aux_files: &[],
    builtin_punc: false,
    engine: EngineKind::SenseVoice,
    supports_hotwords: true,
    local_copies: &[("model_eb.onnx", "model_eb_quant.onnx")],
};

fn write_file(path: &Path, bytes: &[u8]) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(path, bytes).expect("write fixture file");
}

fn path_ends_with(path: &Path, suffix: &[&str]) -> bool {
    let parts: Vec<String> = path
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect();
    parts.len() >= suffix.len()
        && parts[parts.len() - suffix.len()..]
            .iter()
            .map(String::as_str)
            .eq(suffix.iter().copied())
}

#[test]
fn file_url_points_at_modelscope_resolve_master() {
    assert_eq!(
        file_url(
            "iic/speech_fsmn_vad_zh-cn-16k-common-onnx",
            "model_quant.onnx"
        ),
        "https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-onnx/resolve/master/model_quant.onnx"
    );
    // 路径里的子目录原样保留
    assert_eq!(
        file_url("iic/SenseVoiceSmall-onnx", "sub/dir/tokens.json"),
        "https://modelscope.cn/models/iic/SenseVoiceSmall-onnx/resolve/master/sub/dir/tokens.json"
    );
}

#[test]
fn files_api_url_requests_recursive_listing() {
    assert_eq!(
        files_api_url(VAD_MODEL.repo),
        "https://modelscope.cn/api/v1/models/iic/speech_fsmn_vad_zh-cn-16k-common-onnx/repo/files?Recursive=true"
    );
}

#[test]
fn model_dir_keeps_id_as_subdirectory() {
    let root = PathBuf::from("/tmp/models-root");
    assert_eq!(
        model_dir(&root, &VAD_MODEL),
        root.join("iic")
            .join("speech_fsmn_vad_zh-cn-16k-common-onnx")
    );
    assert_eq!(
        model_dir(&root, &PARAFORMER_ZH_HOTWORD),
        root.join("iic")
            .join("speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx")
    );
    // 只有一段的 id
    let flat = ModelSpec {
        id: "flat-model",
        ..SENSEVOICE
    };
    assert_eq!(model_dir(&root, &flat), root.join("flat-model"));
}

#[test]
fn dest_for_and_part_path_are_traversal_safe() {
    let target = PathBuf::from("/tmp/models/iic/x");
    assert_eq!(dest_for(&target, "tokens.json"), target.join("tokens.json"));
    assert_eq!(
        dest_for(&target, "sub/dir/am.mvn"),
        target.join("sub").join("dir").join("am.mvn")
    );
    // '.' / '..' / 空段被丢弃，不能跳出模型目录
    assert_eq!(dest_for(&target, "../evil.bin"), target.join("evil.bin"));
    assert_eq!(
        dest_for(&target, "a/./b//c"),
        target.join("a").join("b").join("c")
    );

    let dest = target.join("model_quant.onnx");
    assert_eq!(part_path(&dest), target.join("model_quant.onnx.part"));
}

#[test]
fn parse_repo_files_reads_modelscope_envelope() {
    let files = parse_repo_files("iic/speech_fsmn_vad_zh-cn-16k-common-onnx", VAD_FILES_JSON)
        .expect("real-shape JSON must parse");
    assert_eq!(files.len(), 4);
    let quant = find_file(&files, "model_quant.onnx").expect("model_quant.onnx listed");
    assert_eq!(quant.path, "model_quant.onnx");
    assert_eq!(quant.size, 506_744);
    assert!(quant.is_blob());
    // tree 条目（目录）不是可下载文件
    assert!(!files[3].is_blob());
    assert!(find_file(&files, "sub").is_none());
    assert!(find_file(&files, "missing.onnx").is_none());
}

#[test]
fn parse_repo_files_reports_api_failure() {
    let err = parse_repo_files(
        "iic/whatever",
        r#"{"Code":404,"Success":false,"Message":"not found"}"#,
    )
    .expect_err("missing Data must be an error");
    let msg = err.to_string();
    assert!(msg.contains("iic/whatever"), "got: {msg}");
    assert!(msg.contains("404"), "got: {msg}");

    let err =
        parse_repo_files("iic/whatever", "<html>502</html>").expect_err("bad JSON is an error");
    assert!(err.to_string().contains("JSON"), "got: {err}");
}

#[test]
fn ignored_patterns_match_python_manager() {
    for ignored in [
        "model.pt",
        "weights.PTH",
        "am.bin",
        "model.safetensors",
        "model_quant.onnx_data.bak",
        "sub/model.pt",
    ] {
        assert!(is_ignored_path(ignored), "{ignored} should be ignored");
    }
    for kept in [
        "model_quant.onnx",
        "tokens.json",
        "am.mvn",
        "config.yaml",
        "seg_dict",
    ] {
        assert!(!is_ignored_path(kept), "{kept} should be downloaded");
    }
}

#[test]
fn pending_downloads_skip_existing_non_empty_files() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let target = model_dir(tmp.path(), &VAD_MODEL);
    let files = parse_repo_files(VAD_MODEL.repo, VAD_FILES_JSON).expect("parse");

    // 空目录：除了忽略的 model.pt 与 tree 条目，其余都要下
    let all = pending_repo_downloads(VAD_MODEL.repo, &files, &target);
    let names: Vec<&str> = all.iter().map(|i| i.dest_name.as_str()).collect();
    assert_eq!(names, vec![".gitattributes", "model_quant.onnx"]);
    assert_eq!(all[1].repo, VAD_MODEL.repo);
    assert_eq!(all[1].size, 506_744);
    assert_eq!(all[1].dest(&target), target.join("model_quant.onnx"));

    // 预置一个非空文件 → 幂等跳过它，另一个仍待下
    write_file(&target.join("model_quant.onnx"), b"already here");
    let pending = pending_repo_downloads(VAD_MODEL.repo, &files, &target);
    let names: Vec<&str> = pending.iter().map(|i| i.dest_name.as_str()).collect();
    assert_eq!(names, vec![".gitattributes"]);

    // 0 字节文件视为未完成，仍要重下
    write_file(&target.join(".gitattributes"), b"");
    let pending = pending_repo_downloads(VAD_MODEL.repo, &files, &target);
    let names: Vec<&str> = pending.iter().map(|i| i.dest_name.as_str()).collect();
    assert_eq!(names, vec![".gitattributes"]);

    // 全部就位 → 什么都不用下
    write_file(&target.join(".gitattributes"), b"* text=auto\n");
    assert!(pending_repo_downloads(VAD_MODEL.repo, &files, &target).is_empty());
}

#[test]
fn is_complete_requires_existing_non_empty_file() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let file = tmp.path().join("model_quant.onnx");
    assert!(!is_complete(&file));
    assert!(needs_download(&file));

    write_file(&file, b"");
    assert!(!is_complete(&file));

    write_file(&file, b"\x00\x01");
    assert!(is_complete(&file));
    assert!(!needs_download(&file));

    // 目录不算就绪
    assert!(!is_complete(tmp.path()));
}

#[test]
fn has_onnx_matches_python_is_downloaded_gate() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("iic").join("x");
    fs::create_dir_all(&dir).expect("mkdir");
    assert!(!has_onnx(&dir), "empty dir has no weights");

    write_file(&dir.join("config.yaml"), b"cfg: 1");
    assert!(!has_onnx(&dir), "config only is not enough");

    write_file(&dir.join("MODEL_QUANT.ONNX"), b"\x00\x01");
    assert!(has_onnx(&dir), "uppercase .ONNX counts");

    assert!(!has_onnx(&tmp.path().join("does-not-exist")));
}

#[test]
fn local_copies_rename_inside_model_dir() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let target = tmp.path().join("test").join("local-copies");
    write_file(&target.join("model_eb.onnx"), b"eb-weights");

    apply_local_copies(&COPY_SPEC, &target).expect("copy should succeed");
    let copied = target.join("model_eb_quant.onnx");
    assert_eq!(fs::read(&copied).expect("read copy"), b"eb-weights");

    // 目标已存在 → 幂等，不覆盖
    write_file(&copied, b"keep-me");
    apply_local_copies(&COPY_SPEC, &target).expect("second copy should be a no-op");
    assert_eq!(fs::read(&copied).expect("read copy"), b"keep-me");

    // 源缺失 → Model 错误，带模型 id 与源文件名
    let empty = tmp.path().join("empty");
    fs::create_dir_all(&empty).expect("mkdir");
    let err = apply_local_copies(&COPY_SPEC, &empty).expect_err("missing source must fail");
    let msg = err.to_string();
    assert!(msg.contains("test/local-copies"), "got: {msg}");
    assert!(msg.contains("model_eb.onnx"), "got: {msg}");
}

#[test]
fn default_models_root_uses_local_app_data_on_windows() {
    // 本机是 Windows：LOCALAPPDATA 存在，断言后缀即可（不写死盘符）
    let root = default_models_root();
    assert!(
        path_ends_with(&root, &["funasr-subtitle-rs", "models"]),
        "unexpected default root: {}",
        root.display()
    );
    if cfg!(windows) {
        let local = std::env::var_os("LOCALAPPDATA").expect("LOCALAPPDATA present on Windows");
        assert_eq!(
            root,
            PathBuf::from(local)
                .join("funasr-subtitle-rs")
                .join("models")
        );
        assert!(
            root.is_absolute(),
            "default root must be absolute: {}",
            root.display()
        );
    }
}

#[test]
fn resolve_models_root_covers_both_platform_branches() {
    // Windows 分支：%LOCALAPPDATA%\funasr-subtitle-rs\models
    let win = resolve_models_root(
        true,
        Some(std::ffi::OsStr::new("/fake/LocalAppData")),
        Some(std::ffi::OsStr::new("/fake/.cache")),
        Some(std::ffi::OsStr::new("/fake/home")),
    );
    assert_eq!(
        win,
        PathBuf::from("/fake/LocalAppData")
            .join("funasr-subtitle-rs")
            .join("models")
    );

    // 非 Windows：XDG_CACHE_HOME 优先
    let unix = resolve_models_root(
        false,
        None,
        Some(std::ffi::OsStr::new("/fake/.cache")),
        Some(std::ffi::OsStr::new("/fake/home")),
    );
    assert_eq!(
        unix,
        PathBuf::from("/fake/.cache")
            .join("funasr-subtitle-rs")
            .join("models")
    );

    // 非 Windows 且无 XDG → ~/.cache
    let home_only =
        resolve_models_root(false, None, None, Some(std::ffi::OsStr::new("/fake/home")));
    assert_eq!(
        home_only,
        PathBuf::from("/fake/home")
            .join(".cache")
            .join("funasr-subtitle-rs")
            .join("models")
    );

    // 非 Windows 且环境变量全空 → 相对 .cache，不 panic
    let bare = resolve_models_root(
        false,
        Some(std::ffi::OsStr::new("")),
        Some(std::ffi::OsStr::new("")),
        None,
    );
    assert_eq!(
        bare,
        PathBuf::from(".cache")
            .join("funasr-subtitle-rs")
            .join("models")
    );
}

#[test]
fn manager_exposes_root_and_dir_layout() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manager = ModelManager::new(tmp.path().to_path_buf());
    assert_eq!(manager.root(), tmp.path());
    // 目录布局（ensure 之前）由 spec.id 决定
    assert_eq!(
        model_dir(manager.root(), &SENSEVOICE),
        tmp.path().join("iic").join("SenseVoiceSmall-onnx")
    );
}

/// 真实下载冒烟：`cargo test download_vad_smoke -- --ignored`（需联网）。
#[test]
#[ignore = "requires network: downloads the real FSMN-VAD repo from ModelScope"]
fn download_vad_smoke() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manager = ModelManager::new(tmp.path().to_path_buf());

    let dir = manager
        .ensure_model(&VAD_MODEL)
        .expect("ensure_model(VAD_MODEL)");
    assert_eq!(dir, model_dir(tmp.path(), &VAD_MODEL));

    let onnx = dir.join("model_quant.onnx");
    let meta = fs::metadata(&onnx).expect("model_quant.onnx must exist");
    assert!(meta.is_file());
    assert!(
        meta.len() > 100_000,
        "model_quant.onnx looks truncated: {} bytes",
        meta.len()
    );
    // 无残留 .part 临时文件
    let leftovers: Vec<String> = fs::read_dir(&dir)
        .expect("read model dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".part"))
        .collect();
    assert!(leftovers.is_empty(), "leftover .part files: {leftovers:?}");

    // 幂等：再跑一次不重写文件
    let before = meta.modified().expect("mtime");
    let dir2 = manager
        .ensure_model(&VAD_MODEL)
        .expect("second ensure_model");
    assert_eq!(dir2, dir);
    let after = fs::metadata(&onnx)
        .expect("metadata")
        .modified()
        .expect("mtime");
    assert_eq!(before, after, "existing model file was rewritten");

    // tmp 清理前确认 VAD 模型目录下确实有主权重
    assert_eq!(
        onnx.file_name().and_then(|n| n.to_str()),
        Some("model_quant.onnx")
    );
}
