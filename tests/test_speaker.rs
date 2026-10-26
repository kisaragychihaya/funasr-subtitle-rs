//! W3 验收测试：CAM++ 嵌入（`src/speaker/embed.rs`）与定种子谱聚类（`src/speaker/cluster.rs`）。
//!
//! 1) **MT19937 已知答案**：硬编码 `reference/diar_common.MT19937(0)` 在 golden venv 里
//!    的前 5 个 `next_u32()` 与前 4 个 `uniform()`，逐位比对 Rust 实现（u32 全等、
//!    double 比 `to_bits`）。定种子 k-means 的可复现性完全依赖这个 PRNG。
//! 2) **纯逻辑**：N < `MIN_CHUNKS_FOR_CLUSTER` → 全 0；两组明显可分的合成嵌入 → 两类且分组
//!    正确；`oracle_num` 生效（`Some(1)` 全 0、`Some(2)` 与自动模式同分区）。
//! 3) **真实模型（`#[ignore]`）**：对 golden 的 70×192 嵌入跑 `cluster(None)` /
//!    `cluster(Some(2))`，标签与 `cluster.json` 的 labels_auto / labels_oracle2
//!    **逐元素相等**；再用 cam++ ONNX 重算 70 个 chunk 的嵌入（sv_chunk 在本文件内联
//!    复刻），与 `embeddings.f32` 逐行比余弦 ≥ 1 - 1e-4。
//!
//! 运行（b/a 都跑）：
//! `FUNASR_SUBTITLE_MODELS_ROOT=<模型根> cargo test --locked --test test_speaker -- --ignored --test-threads=1`

use std::fs;
use std::path::{Path, PathBuf};

use funasr_subtitle_rs::models::SPK_MODEL;
use funasr_subtitle_rs::models::manager::{ModelManager, default_models_root};
use funasr_subtitle_rs::speaker::cluster::{Mt19937, cluster};
use funasr_subtitle_rs::speaker::embed::SpeakerEmbedder;
use funasr_subtitle_rs::speaker::{CHUNK_LEN, CHUNK_SHIFT, MIN_CHUNKS_FOR_CLUSTER};

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
    raw.as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_le_bytes(*c))
        .collect()
}

/// 模型根目录：默认 `default_models_root()`；`FUNASR_SUBTITLE_MODELS_ROOT` 可覆盖。
fn models_root() -> PathBuf {
    std::env::var_os("FUNASR_SUBTITLE_MODELS_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(default_models_root)
}

/// 余弦相似度（嵌入未归一，逐维 double 累加）。
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// 确定性伪随机（无依赖）：xorshift64*，范围 [-1, 1)。
fn pseudo_random(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed | 1;
    move || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        let v = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 40) as f64 / 8_388_608.0) - 1.0
    }
}

// ---------------------------------------------------------------------------
// 1) MT19937 已知答案（与 golden venv 的 numpy 参考逐位对拍）
// ---------------------------------------------------------------------------

#[test]
fn test_mt19937_matches_reference_bits() {
    // 期望值来源：
    //   ../.venv-golden/Scripts/python.exe -c "import sys; sys.path.insert(0,'reference');
    //   from diar_common import MT19937; m=MT19937(0); print([m.next_u32() for _ in range(5)]);
    //   m=MT19937(0); print([m.uniform() for _ in range(4)])"
    let want_u32: [u32; 5] = [
        2_357_136_044,
        2_546_248_239,
        3_071_714_933,
        3_626_093_760,
        2_588_848_963,
    ];
    let mut rng = Mt19937::new(0);
    for (i, want) in want_u32.iter().enumerate() {
        assert_eq!(
            rng.next_u32(),
            *want,
            "第 {i} 个 next_u32 与 numpy 参考不一致"
        );
    }

    let want_uniform: [f64; 4] = [
        0.548_813_503_927_324_8,
        0.715_189_366_372_419_5,
        0.602_763_376_071_643_9,
        0.544_883_182_996_896_9,
    ];
    let mut rng = Mt19937::new(0);
    for (i, want) in want_uniform.iter().enumerate() {
        let got = rng.uniform();
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "第 {i} 个 uniform 与 numpy 参考不一致：{got} vs {want}"
        );
    }

    // 换个种子也要与 numpy 一致（种子 1 的前 3 个 u32，来源同上，只换 MT19937(1)）。
    let mut rng = Mt19937::new(1);
    assert_eq!(rng.next_u32(), 1_791_095_845);
    assert_eq!(rng.next_u32(), 4_282_876_139);
    assert_eq!(rng.next_u32(), 3_093_770_124);
}

// ---------------------------------------------------------------------------
// 2) 纯逻辑
// ---------------------------------------------------------------------------

/// 两组明显可分的合成嵌入：0..n/2 围绕 +e0、其余围绕 +e1，噪声幅度 0.05。
fn synthetic_two_clusters(per_cluster: usize) -> Vec<Vec<f32>> {
    const DIM: usize = 192;
    let mut rng = pseudo_random(0x5eed_1234);
    let mut out = Vec::with_capacity(2 * per_cluster);
    for group in 0..2 {
        for _ in 0..per_cluster {
            let mut row = vec![0.0f32; DIM];
            for v in row.iter_mut() {
                *v = (0.05 * rng()) as f32;
            }
            row[group] += 1.0;
            out.push(row);
        }
    }
    out
}

#[test]
fn test_cluster_small_n_is_single_speaker() {
    // 上游原行为：chunk 数不足 20 -> 全 0（不聚类）
    let emb = synthetic_two_clusters(MIN_CHUNKS_FOR_CLUSTER / 2 - 1); // 18 个
    assert!(emb.len() < MIN_CHUNKS_FOR_CLUSTER);
    assert_eq!(cluster(&emb, None), vec![0usize; emb.len()]);
    assert_eq!(
        cluster(&emb, Some(3)),
        vec![0usize; emb.len()],
        "oracle 也走同一门槛"
    );
    assert_eq!(cluster(&[], None), Vec::<usize>::new());
    // 正好 20 个才开始聚类
    let emb = synthetic_two_clusters(MIN_CHUNKS_FOR_CLUSTER / 2);
    assert_eq!(emb.len(), MIN_CHUNKS_FOR_CLUSTER);
    assert_eq!(*cluster(&emb, Some(1)).iter().max().unwrap_or(&0), 0);
}

#[test]
fn test_cluster_splits_two_obvious_groups() {
    let per = 20usize;
    let emb = synthetic_two_clusters(per);
    let labels = cluster(&emb, None);
    assert_eq!(labels.len(), emb.len());
    // 分区正确：前 per 个同类、后 per 个同类、两组不同类
    let first = labels[0];
    let last = labels[per];
    assert!(
        labels[..per].iter().all(|l| *l == first),
        "第一组被打散: {:?}",
        labels
    );
    assert!(
        labels[per..].iter().all(|l| *l == last),
        "第二组被打散: {:?}",
        labels
    );
    assert_ne!(first, last, "两组应判为不同说话人");
    assert_eq!(
        labels.iter().copied().max().map(|m| m + 1),
        Some(2),
        "应恰好 2 类"
    );

    // oracle_num 生效：Some(2) 得到同一分区（编号可能不同）
    let oracle = cluster(&emb, Some(2));
    for i in 0..emb.len() {
        for j in 0..emb.len() {
            assert_eq!(
                labels[i] == labels[j],
                oracle[i] == oracle[j],
                "第 {i}/{j} 个 chunk 的分区在自动/oracle 模式间不一致"
            );
        }
    }

    // Some(1)：单一说话人，且不做 merge_by_cos
    assert_eq!(cluster(&emb, Some(1)), vec![0usize; emb.len()]);
}

// ---------------------------------------------------------------------------
// 3) 真实模型（#[ignore]）
// ---------------------------------------------------------------------------

/// golden 目录内与聚类相关的两份 json。
#[derive(serde::Deserialize)]
struct ClusterGolden {
    n_chunks: usize,
    labels_auto: Vec<usize>,
    labels_oracle2: Vec<usize>,
}

#[derive(serde::Deserialize)]
struct ChunksGolden {
    vad_spans: Vec<[i64; 2]>,
}

const DIAR_DIR: &str = "tests/golden/diar/diar_two_speakers";
const N_CHUNKS: usize = 70;
const EMB_DIM: usize = 192;

/// 读 `embeddings.f32`（70×192 f32 小端行主序）。
fn read_embeddings() -> Vec<Vec<f32>> {
    let path = manifest_path(&format!("{DIAR_DIR}/embeddings.f32"));
    let raw = fs::read(&path).unwrap_or_else(|e| panic!("读取 {} 失败: {e}", path.display()));
    assert_eq!(
        raw.len(),
        N_CHUNKS * EMB_DIM * 4,
        "{} 字节数应与 {N_CHUNKS}×{EMB_DIM} f32 相符",
        path.display()
    );
    let flat: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    flat.chunks(EMB_DIM).map(|r| r.to_vec()).collect()
}

/// `reference/diar_common.sv_chunk` 的内联复刻（窗 `CHUNK_LEN`/移 `CHUNK_SHIFT`，
/// 尾窗前移贴段尾、不足补零，采样下标 `floor(ms/1000*16000)`）。
fn sv_chunk(spans_ms: &[[i64; 2]], wav: &[i16]) -> Vec<Vec<i16>> {
    const SAMPLE_RATE: f64 = 16_000.0;
    let mut chunks: Vec<Vec<i16>> = Vec::new();
    for [st_ms, ed_ms] in spans_ms {
        let a = ((*st_ms as f64 / 1000.0 * SAMPLE_RATE) as usize).min(wav.len());
        let b = ((*ed_ms as f64 / 1000.0 * SAMPLE_RATE) as usize).min(wav.len());
        let data = if b > a { &wav[a..b] } else { &[] };
        let mut last_chunk_ed = 0usize;
        let mut pos = 0usize;
        while pos < data.len() {
            let chunk_ed = (pos + CHUNK_LEN).min(data.len());
            if chunk_ed <= last_chunk_ed {
                break;
            }
            last_chunk_ed = chunk_ed;
            let chunk_st = chunk_ed.saturating_sub(CHUNK_LEN);
            let window = &data[chunk_st..chunk_ed];
            let mut padded = vec![0i16; CHUNK_LEN];
            padded[..window.len()].copy_from_slice(window);
            chunks.push(padded);
            pos += CHUNK_SHIFT;
        }
    }
    chunks
}

#[test]
#[ignore = "需要 tests/golden/diar 的 golden 嵌入（无需 ONNX）"]
fn test_cluster_matches_golden_labels() {
    let golden: ClusterGolden = read_json(&format!("{DIAR_DIR}/cluster.json"));
    let embeddings = read_embeddings();
    assert_eq!(embeddings.len(), golden.n_chunks);
    assert_eq!(golden.labels_auto.len(), N_CHUNKS);
    assert_eq!(golden.labels_oracle2.len(), N_CHUNKS);

    let auto = cluster(&embeddings, None);
    if auto != golden.labels_auto {
        let diff: Vec<usize> = (0..N_CHUNKS)
            .filter(|i| auto[*i] != golden.labels_auto[*i])
            .collect();
        panic!(
            "自动模式标签与 golden 不一致（{} 个位置）: {diff:?}\n golden={:?}\n got   ={auto:?}",
            diff.len(),
            golden.labels_auto
        );
    }

    let oracle = cluster(&embeddings, Some(2));
    if oracle != golden.labels_oracle2 {
        let diff: Vec<usize> = (0..N_CHUNKS)
            .filter(|i| oracle[*i] != golden.labels_oracle2[*i])
            .collect();
        panic!(
            "oracle=2 标签与 golden 不一致（{} 个位置）: {diff:?}\n golden={:?}\n got   ={oracle:?}",
            diff.len(),
            golden.labels_oracle2
        );
    }
    println!("cluster 标签与 golden 逐元素一致（auto 与 oracle=2）");
}

#[test]
#[ignore = "需要 cam++ ONNX（~28MB，首次会下载）"]
fn test_embed_matches_golden_embeddings() {
    let dir = ModelManager::new(models_root())
        .ensure_model(&SPK_MODEL)
        .unwrap_or_else(|e| panic!("准备 CAM++ 模型失败（首次需要联网下载 28MB）: {e}"));
    println!("CAM++ 模型目录: {}", dir.display());
    let mut embedder =
        SpeakerEmbedder::new(&dir, 4).unwrap_or_else(|e| panic!("加载 CAM++ 失败: {e}"));

    let wav = read_s16(&format!("{DIAR_DIR}/input.s16bin"));
    let chunk_golden: ChunksGolden = read_json(&format!("{DIAR_DIR}/chunks.json"));
    let chunks = sv_chunk(&chunk_golden.vad_spans, &wav);
    assert_eq!(
        chunks.len(),
        N_CHUNKS,
        "sv_chunk 切出的 chunk 数应等于 golden 的 {N_CHUNKS}"
    );
    for chunk in &chunks {
        assert_eq!(chunk.len(), CHUNK_LEN, "chunk 必须补零到满窗");
    }

    let golden = read_embeddings();
    let mut worst = 1.0f64;
    let mut worst_index = 0usize;
    let mut failures = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let got = embedder
            .embed(chunk)
            .unwrap_or_else(|e| panic!("第 {i} 个 chunk 嵌入失败: {e}"));
        assert_eq!(got.len(), EMB_DIM, "第 {i} 个 chunk 的嵌入维数");
        let cos = cosine(&got, &golden[i]);
        if cos < worst {
            worst = cos;
            worst_index = i;
        }
        if cos < 1.0 - 1e-4 {
            failures.push(format!("#{i}: cos={cos:.9}"));
        }
    }
    println!("{N_CHUNKS} 个 chunk：最小余弦 {worst:.9}（chunk #{worst_index}）");
    assert!(
        failures.is_empty(),
        "{} / {N_CHUNKS} 个 chunk 的嵌入与 golden 不符（余弦 < 1-1e-4）: {}",
        failures.len(),
        failures.join(", ")
    );
}
