"""说话人分离 golden 共享件：切窗 / 特征 / 嵌入 / 定种子谱聚类 / 后处理 / 分配。

两个消费方：
* `dump_golden_diar.py`（.venv-golden）：生成 tests/golden/diar/ 基准；
* `check_torch_equivalence.py`（.venv-diar）：ONNX vs 官方 torch CAMPPlus 权重级验证。

算法契约（冻结，Rust 侧 src/speaker/ 逐行对应）：
* sv_chunk / extract_feature(fbank+均值减) / postprocess / distribute_spk 逐行移植
  FunASR `funasr/models/campplus/utils.py`；
* 谱聚类移植 `cluster_backend.py` 的 SpectralCluster；k-means 为 golden 契约的
  定种子实现（MT19937(0) k-means++，n_init=10，Lloyd），因上游 sklearn 无种子。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
CHUNK_LEN = int(1.5 * SAMPLE_RATE)     # 24000
CHUNK_SHIFT = int(0.75 * SAMPLE_RATE)  # 12000

MIN_NUM_SPKS = 1
MAX_NUM_SPKS = 15
PVAL = 0.022
MERGE_THR = 0.78

_WORKSPACE = Path(__file__).resolve().parent.parent.parent
CAMPPLUS_ONNX = _WORKSPACE / ".dev-cache/models/campplus/campplus.onnx"

_sess = None


def campplus_session():
    global _sess
    if _sess is None:
        if not CAMPPLUS_ONNX.exists():
            raise RuntimeError(f"缺少 {CAMPPLUS_ONNX}，请先下载 campplus.onnx")
        import onnxruntime as ort
        _sess = ort.InferenceSession(str(CAMPPLUS_ONNX), providers=["CPUExecutionProvider"])
    return _sess


# ---------------------------------------------------------------- sv_chunk
def sv_chunk(spans_ms: list[list[int]], wav: np.ndarray) -> list[tuple[float, float, np.ndarray]]:
    """FunASR campplus/utils.sv_chunk 逐行移植（seg_dur=1.5, seg_shift=0.75）。"""
    chunks: list[tuple[float, float, np.ndarray]] = []
    for st_ms, ed_ms in spans_ms:
        seg_st = st_ms / 1000.0
        data = wav[int(st_ms / 1000.0 * SAMPLE_RATE): int(ed_ms / 1000.0 * SAMPLE_RATE)]
        last_chunk_ed = 0
        for chunk_st in range(0, data.shape[0], CHUNK_SHIFT):
            chunk_ed = min(chunk_st + CHUNK_LEN, data.shape[0])
            if chunk_ed <= last_chunk_ed:
                break
            last_chunk_ed = chunk_ed
            chunk_st = max(0, chunk_ed - CHUNK_LEN)
            chunk_data = data[chunk_st:chunk_ed]
            if chunk_data.shape[0] < CHUNK_LEN:
                chunk_data = np.pad(chunk_data, (0, CHUNK_LEN - chunk_data.shape[0]), "constant")
            chunks.append((chunk_st / SAMPLE_RATE + seg_st,
                           chunk_ed / SAMPLE_RATE + seg_st,
                           chunk_data.astype(np.float32)))
    return chunks


# ---------------------------------------------------------------- 特征 + 嵌入
def chunk_fbank(pcm: np.ndarray) -> np.ndarray:
    """knf fbank(80) + 句内均值减；与 Rust frontend 同一 fbank 语义。"""
    import kaldi_native_fbank as knf
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = float(SAMPLE_RATE)
    opts.frame_opts.dither = 0.0
    opts.energy_floor = 1.0
    opts.mel_opts.num_bins = 80
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(SAMPLE_RATE, pcm.tolist())
    mat = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
    mat = mat.astype(np.float32)
    return mat - mat.mean(axis=0, keepdims=True)


def embed(feat: np.ndarray) -> np.ndarray:
    out = campplus_session().run(None, {"fbank": feat[None, ...]})[0]
    return out[0].astype(np.float32)  # (192,)


# ---------------------------------------------------------------- MT19937（契约 PRNG）
class MT19937:
    """标准 32 位 MT19937；双精度取 genrand_res53 式 53 位（Rust 侧移植必须逐位一致）。"""

    def __init__(self, seed: int) -> None:
        self.mt = [0] * 624
        self.mt[0] = seed & 0xFFFFFFFF
        for i in range(1, 624):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
        self.idx = 624

    def _twist(self) -> None:
        for i in range(624):
            y = (self.mt[i] & 0x80000000) | (self.mt[(i + 1) % 624] & 0x7FFFFFFF)
            self.mt[i] = self.mt[(i + 397) % 624] ^ (y >> 1)
            if y & 1:
                self.mt[i] ^= 2567483615
        self.idx = 0

    def next_u32(self) -> int:
        if self.idx >= 624:
            self._twist()
        y = self.mt[self.idx]
        self.idx += 1
        y ^= y >> 11
        y ^= (y << 7) & 2636928640
        y ^= (y << 15) & 4022730752
        y ^= y >> 18
        return y & 0xFFFFFFFF

    def uniform(self) -> float:
        a = self.next_u32() >> 5
        b = self.next_u32() >> 6
        return (a * 67108864.0 + b) / 9007199254740992.0

    def randint(self, n: int) -> int:
        return int(self.uniform() * n)

    def choice_d2(self, d2: np.ndarray) -> int:
        total = float(d2.sum())
        thr = self.uniform() * total
        cum = 0.0
        for i, v in enumerate(d2):
            cum += float(v)
            if cum >= thr:
                return i
        return len(d2) - 1


# ---------------------------------------------------------------- 聚类
def kmeans_seeded(emb: np.ndarray, k: int) -> np.ndarray:
    """定种子 k-means++（MT19937(0)，n_init=10，Lloyd 标签不变即停，inertia 平局取先达）。"""
    n = emb.shape[0]
    if k <= 0 or n == 0:
        return np.zeros(n, dtype=np.int64)
    x = emb.astype(np.float64)
    best_labels: np.ndarray | None = None
    best_inertia = np.inf
    for _init in range(10):
        rng = MT19937(0)  # 每个 init 重新播种：实现简单且完全确定
        centers = np.empty((k, x.shape[1]), dtype=np.float64)
        centers[0] = x[rng.randint(n)]
        d2 = ((x - centers[0]) ** 2).sum(axis=1)
        for c in range(1, k):
            idx = rng.choice_d2(d2)
            centers[c] = x[idx]
            d2 = np.minimum(d2, ((x - centers[c]) ** 2).sum(axis=1))
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(300):
            dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = dist.argmin(axis=1)
            if _ > 0 and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                pts = x[labels == c]
                if len(pts):
                    centers[c] = pts.mean(axis=0)
        inertia = float(((x - centers[labels]) ** 2).sum())
        if inertia < best_inertia - 1e-9:
            best_inertia = inertia
            best_labels = labels.copy()
    assert best_labels is not None
    return best_labels


def spectral_cluster(x: np.ndarray, oracle_num: int | None = None) -> np.ndarray:
    """SpectralCluster 逐行移植（除 k-means 定种子化，见上）。"""
    n = x.shape[0]
    if n < 20:
        return np.zeros(n, dtype=np.int64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    xn = x / np.maximum(norm, 1e-12)
    sim = xn @ xn.T
    pval = PVAL
    if n * pval < 6:
        pval = 6.0 / n
    n_elems = int((1 - pval) * n)
    pruned = sim.copy()
    for i in range(n):
        low = np.argsort(pruned[i, :], kind="stable")[:n_elems]
        pruned[i, low] = 0.0
    pruned = 0.5 * (pruned + pruned.T)
    pruned[np.diag_indices(n)] = 0.0
    deg = np.abs(pruned).sum(axis=1)
    lap = np.diag(deg) - pruned
    n_eig = min(MAX_NUM_SPKS + 1, n)
    lambdas, vecs = np.linalg.eigh(lap)
    lambdas, vecs = lambdas[:n_eig], vecs[:, :n_eig]
    # 符号规范化（golden 契约）：每列取 |分量| 最大者，若为负则整列翻转。
    # numpy/LAPACK 与 Rust 手搓 Jacobi 的特征向量符号本来不定，不规范化两边
    # k-means 输入就不可复现。
    for c in range(vecs.shape[1]):
        k = int(np.argmax(np.abs(vecs[:, c])))
        if vecs[k, c] < 0:
            vecs[:, c] = -vecs[:, c]
    if oracle_num is not None:
        num_spk = int(oracle_num)
    else:
        gaps = np.diff(lambdas[MIN_NUM_SPKS - 1: MAX_NUM_SPKS + 1])
        num_spk = int(np.argmax(gaps)) + MIN_NUM_SPKS
    emb = vecs[:, :num_spk]
    labels = kmeans_seeded(emb, num_spk)
    if oracle_num is None:
        labels = merge_by_cos(labels, x, MERGE_THR)
    return labels


def merge_by_cos(labels: np.ndarray, embs: np.ndarray, cos_thr: float) -> np.ndarray:
    labels = labels.copy()
    while True:
        spk_num = int(labels.max()) + 1
        if spk_num <= 1:
            break
        centers = np.stack([embs[labels == i].mean(0) for i in range(spk_num)])
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        aff = np.triu(centers @ centers.T, 1)
        spks = np.unravel_index(int(np.argmax(aff)), aff.shape)
        if aff[spks] < cos_thr:
            break
        for i in range(len(labels)):
            if labels[i] == spks[1]:
                labels[i] = spks[0]
            elif labels[i] > spks[1]:
                labels[i] -= 1
    return labels


# ---------------------------------------------------------------- 后处理 / 分配
def correct_labels(labels: np.ndarray) -> np.ndarray:
    remap: dict[int, int] = {}
    nxt = 0
    out = np.empty_like(labels)
    for i, lab in enumerate(labels):
        if lab not in remap:
            remap[lab] = nxt
            nxt += 1
        out[i] = remap[lab]
    return out


def merge_seque(res: list[list]) -> list[list]:
    out = [list(res[0])]
    for st, ed, spk in res[1:]:
        if spk != out[-1][2] or st > out[-1][1]:
            out.append([st, ed, spk])
        else:
            out[-1][1] = ed
    return out


def smooth(res: list[list], mindur: float = 0.7) -> list[list]:
    if len(res) < 2:
        return res
    for i in range(len(res)):
        res[i][0] = round(res[i][0], 2)   # Python round = half-even
        res[i][1] = round(res[i][1], 2)
        if res[i][1] - res[i][0] < mindur:
            if i == 0:
                res[i][2] = res[i + 1][2]
            elif i == len(res) - 1:
                res[i][2] = res[i - 1][2]
            elif res[i][0] - res[i - 1][1] <= res[i + 1][0] - res[i][1]:
                res[i][2] = res[i - 1][2]
            else:
                res[i][2] = res[i + 1][2]
    return merge_seque(res)


def postprocess(segments: list[tuple[float, float, np.ndarray]],
                labels: np.ndarray) -> list[list]:
    """FunASR campplus/utils.postprocess（不含 spk_center 分支）。"""
    labels = correct_labels(np.asarray(labels))
    res: list[list] = [[segments[i][0], segments[i][1], int(labels[i])]
                       for i in range(len(segments))]
    res = merge_seque(res)
    for i in range(1, len(res)):
        if res[i - 1][1] > res[i][0] + 1e-4:  # is_overlapped
            p = (res[i][0] + res[i - 1][1]) / 2
            res[i][0] = p
            res[i - 1][1] = p
    return smooth(res)


def distribute_spk(spans_ms: list[list[int]], sd: list[list]) -> list[int]:
    """最大总重叠分配；无重叠默认 0；并列取先出现的说话人。"""
    sd_ms = [(st * 1000, ed * 1000, spk) for st, ed, spk in sd]
    out: list[int] = []
    for s_st, s_ed in spans_ms:
        overlaps: dict[int, float] = {}
        for st, ed, spk in sd_ms:
            ov = max(min(s_ed, ed) - max(s_st, st), 0.0)
            if ov > 0:
                overlaps[spk] = overlaps.get(spk, 0.0) + ov
        best = 0
        best_ov = -1.0
        for spk, ov in overlaps.items():
            if ov > best_ov:
                best_ov = ov
                best = spk
        out.append(int(best))
    return out
