//! 谱聚类。契约见 `super` 模块文档（FunASR SpectralCluster 移植 +
//! MT19937(0) 定种子 k-means，必须与 golden 的 numpy 参考逐次一致）。
//!
//! 逐行对应 `reference/diar_common.py`：
//! * `spectral_cluster`：余弦相似度 → p 剪枝 → `0.5*(A+Aᵀ)` → 对角清零 →
//!   未归一化 Laplacian `L = D - A` → 升序前 `min(16, N)` 个特征对 → 特征向量符号规范化 →
//!   eigengap（或 oracle）定 k → `kmeans_seeded` →（自动模式）`merge_by_cos`。
//! * `kmeans_seeded`：k-means++（**每次 init 重新** `MT19937(0)`）+ Lloyd ≤300 轮
//!   （先算新标签、与旧标签相同即停，再赋标签/更新中心）+ n_init=10 取 inertia 最小者
//!   （严格小于 `best - 1e-9` 才替换，平局取先达）。
//! * `Mt19937`：标准 32 位 MT19937；双精度取 `(a*2^26+b)/2^53`（`genrand_res53`）。
//! * `jacobi_eigh` 替代 `np.linalg.eigh`（f64 循环 Jacobi，不引依赖）。特征向量的符号
//!   本来就由实现决定，numpy/LAPACK 与 Jacobi 不一致，靠与 golden 相同的
//!   `normalize_signs`（每列 |分量| 最大者置正）对齐，否则 k-means 输入不可复现。
//!
//! 内部全部用 f64（numpy 侧同样是 f64：`xn`/`sim`/`lap` 与 `x.astype(float64)`）。
//! N ≥ `SPECTRAL_MAX_N` 时上游改走 umap+hdbscan（不移植），本实现回退谱聚类：
//! 结果仍正确，只是 N 很大时手写 Jacobi 的 O(N³) 会更慢。

use crate::speaker::{MAX_NUM_SPKS, MERGE_THR, MIN_CHUNKS_FOR_CLUSTER, MIN_NUM_SPKS, PVAL};

/// k-means 初始化次数（golden `n_init=10`）。
const KMEANS_N_INIT: usize = 10;
/// Lloyd 迭代上限（golden ≤300 轮）。
const LLOYD_MAX_ITERS: usize = 300;
/// 每次 init 的 PRNG 种子（golden 固定 `MT19937(0)`：实现简单且完全确定）。
const KMEANS_SEED: u32 = 0;
/// 余弦归一化的分母下限（`np.maximum(norm, 1e-12)`）。
const NORM_FLOOR: f64 = 1e-12;
/// inertia 比较容差（golden `inertia < best_inertia - 1e-9`）。
const INERTIA_TIE_EPS: f64 = 1e-9;
/// Jacobi 收敛阈值：上三角最大非对角元相对 |对角线| 最大值的比例。
const JACOBI_TOL: f64 = 1e-14;
/// Jacobi 扫描上限（循环 Jacobi 二次收敛，正常十几次就到阈值）。
const JACOBI_MAX_SWEEPS: usize = 100;

/// 说话人聚类。
///
/// * `embeddings[i]`：第 i 个 chunk 的 192 维嵌入（原始值，不需要 L2 归一）。
/// * `oracle_num`：`preset_spk_num`（跳过 eigengap 与 merge_by_cos）。
/// * 返回每 chunk 的说话人编号（未后处理的原始标签）。
///
/// 行为契约：N < `MIN_CHUNKS_FOR_CLUSTER` → 全 0；否则谱聚类
/// （N ≥ `SPECTRAL_MAX_N` 时上游用 umap+hdbscan，本实现回退谱聚类，见模块文档）。
pub fn cluster(embeddings: &[Vec<f32>], oracle_num: Option<usize>) -> Vec<usize> {
    let n = embeddings.len();
    if n < MIN_CHUNKS_FOR_CLUSTER {
        // 上游 FunASR 原行为：chunk 太少直接判单说话人
        return vec![0; n];
    }
    // 各 chunk 的公共维度（正常情况都是 192）
    let dim = embeddings.iter().map(Vec::len).min().unwrap_or(0);
    if dim == 0 {
        return vec![0; n];
    }
    let x: Vec<Vec<f64>> = embeddings
        .iter()
        .map(|row| row[..dim].iter().map(|v| f64::from(*v)).collect())
        .collect();

    // xn = x / np.maximum(||x||, 1e-12)；sim = xn @ xn.T
    let xn: Vec<Vec<f64>> = x.iter().map(|row| normalize(row)).collect();
    let mut sim = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let value = dot(&xn[i], &xn[j]);
            sim[i][j] = value;
            sim[j][i] = value;
        }
    }

    // p 剪枝：每行 argsort(stable) 的前 n_elems 个（最小的一批）清零 = 保留最大的 n-n_elems 个
    let n_elems = prune_len(n);
    let mut pruned = sim;
    let mut order: Vec<usize> = (0..n).collect();
    for row in pruned.iter_mut() {
        order.clear();
        order.extend(0..n);
        // 稳定排序：并列时保持下标序（np.argsort(kind="stable")）
        order.sort_by(|a, b| row[*a].total_cmp(&row[*b]));
        for &idx in order.iter().take(n_elems) {
            row[idx] = 0.0;
        }
    }

    // pruned = 0.5*(pruned + pruned.T)；对角清零
    let mut sym = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let value = 0.5 * (pruned[i][j] + pruned[j][i]);
            sym[i][j] = value;
            sym[j][i] = value;
        }
        sym[i][i] = 0.0;
    }

    // lap = diag(deg) - pruned，deg = 行 |值| 和（行主序 n×n）
    let mut lap = vec![0.0f64; n * n];
    for (i, row) in sym.iter().enumerate() {
        let deg: f64 = row.iter().map(|v| v.abs()).sum();
        let base = i * n;
        lap[base + i] = deg;
        for (j, v) in row.iter().enumerate() {
            lap[base + j] -= *v;
        }
    }

    // 升序特征值/向量 → 取前 n_eig 列 → 符号规范化（golden 契约）
    let n_eig = n.min(MAX_NUM_SPKS + 1);
    let (lambdas, vecs) = jacobi_eigh(lap, n);
    let mut feats = vec![vec![0.0f64; n_eig]; n];
    for (r, row) in feats.iter_mut().enumerate() {
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = vecs[r * n + c];
        }
    }
    normalize_signs(&mut feats);

    // 说话人数：oracle 直接给定；否则 num_spk = argmax(diff(λ[0..16])) + 1
    let num_spk = match oracle_num {
        Some(k) => k,
        None => eigengap_num_spk(&lambdas[..n_eig]),
    };
    // vecs[:, :num_spk]：特征向量不够 num_spk 列时 numpy 只给 n_eig 列
    let used = num_spk.min(n_eig);
    let points: Vec<Vec<f64>> = feats.iter().map(|row| row[..used].to_vec()).collect();

    let mut labels = kmeans_seeded(&points, num_spk);
    if oracle_num.is_none() {
        // 自动模式才做余弦合并（oracle 模式跳过，见 golden）
        labels = merge_by_cos(&labels, &x, MERGE_THR);
    }
    labels
}

/// `n_elems = int((1-pval)*n)`：`pval=PVAL`，`n*pval < 6` 时抬到 `6/n`。
fn prune_len(n: usize) -> usize {
    let mut pval = PVAL;
    if (n as f64) * pval < 6.0 {
        pval = 6.0 / n as f64;
    }
    ((1.0 - pval) * n as f64) as usize
}

/// `num_spk = argmax(np.diff(lambdas[MIN_NUM_SPKS-1 : MAX_NUM_SPKS+1])) + MIN_NUM_SPKS`
/// （并列取第一个：严格大于才更新）。
fn eigengap_num_spk(lambdas: &[f64]) -> usize {
    let start = MIN_NUM_SPKS - 1;
    let end = lambdas.len().min(MAX_NUM_SPKS + 1);
    if end <= start + 1 {
        return MIN_NUM_SPKS;
    }
    let mut best = start;
    let mut best_gap = f64::NEG_INFINITY;
    for i in start..end - 1 {
        let gap = lambdas[i + 1] - lambdas[i];
        if gap > best_gap {
            best_gap = gap;
            best = i;
        }
    }
    best + MIN_NUM_SPKS
}

/// 特征向量列的符号规范化：每列取 |分量| 最大者（并列取第一个），若为负则整列取反。
fn normalize_signs(feats: &mut [Vec<f64>]) {
    let cols = feats.first().map_or(0, Vec::len);
    for c in 0..cols {
        let mut best = 0usize;
        let mut best_abs = 0.0f64;
        for (r, row) in feats.iter().enumerate() {
            let v = row[c].abs();
            if v > best_abs {
                best_abs = v;
                best = r;
            }
        }
        if feats[best][c] < 0.0 {
            for row in feats.iter_mut() {
                row[c] = -row[c];
            }
        }
    }
}

// ---------------------------------------------------------------- 线性代数小件

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn normalize(row: &[f64]) -> Vec<f64> {
    let norm = dot(row, row).sqrt().max(NORM_FLOOR);
    row.iter().map(|v| v / norm).collect()
}

/// 对称矩阵 Jacobi 特征分解（f64，行主序 n×n；入参按值传入以免再复制一份 N² 矩阵）。
///
/// 返回 `(升序特征值, 特征向量)`：特征向量同样是行主序 n×n，第 j 列是 `evals[j]`
/// 的单位特征向量。符号不保证与 LAPACK 一致——调用方必须做
/// [`normalize_signs`]（golden 契约，见模块文档）。
fn jacobi_eigh(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..JACOBI_MAX_SWEEPS {
        // 收敛判据：上三角最大非对角元相对 |对角| 最大值
        let mut off = 0.0f64;
        let mut scale = 0.0f64;
        for p in 0..n {
            scale = scale.max(a[p * n + p].abs());
            for q in (p + 1)..n {
                off = off.max(a[p * n + q].abs());
            }
        }
        if off <= JACOBI_TOL * scale.max(1.0) {
            break;
        }

        for p in 0..n.saturating_sub(1) {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                // cot(2θ) = τ = (aqq - app) / (2·apq)；t = tanθ 取绝对值较小的根
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // A ← Jᵀ A J（先列后行，J = [[c, s], [-s, c]]）
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                // V ← V J
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    // 按特征值升序重排（np.linalg.eigh 返回升序）
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|i, j| a[i * n + i].total_cmp(&a[j * n + j]));
    let evals: Vec<f64> = idx.iter().map(|i| a[i * n + i]).collect();
    let mut vecs = vec![0.0f64; n * n];
    for (dst, &src) in idx.iter().enumerate() {
        for r in 0..n {
            vecs[r * n + dst] = v[r * n + src];
        }
    }
    (evals, vecs)
}

// ---------------------------------------------------------------- MT19937

/// 契约 PRNG：标准 32 位 MT19937（`reference/diar_common.MT19937` 的逐位移植）。
///
/// 之所以 `pub`：golden 契约要求与 numpy 参考逐位一致，`tests/test_speaker.rs`
/// 用已知答案（u32 ×5、double ×4）做跨语言对拍；生产路径只经 k-means++ 使用它。
///
/// * `next_u32`：标准 tempering（`y ^= y>>11; y ^= (y<<7)&0x9D2C5680; ...`）。
/// * `uniform`：`genrand_res53` 式 53 位双精度 `(a*2^26 + b) / 2^53`。
#[derive(Debug, Clone)]
pub struct Mt19937 {
    mt: [u32; 624],
    idx: usize,
}

impl Mt19937 {
    /// `MT19937.__init__`：`mt[0]=seed`，其后 Knuth 初始化。
    pub fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1_812_433_253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { mt, idx: 624 }
    }

    /// `MT19937._twist`。
    fn twist(&mut self) {
        for i in 0..624 {
            let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
            let mut next = self.mt[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= 2_567_483_615;
            }
            self.mt[i] = next;
        }
        self.idx = 0;
    }

    /// 一个 32 位输出（含 tempering）。
    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            self.twist();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 2_636_928_640;
        y ^= (y << 15) & 4_022_730_752;
        y ^= y >> 18;
        y
    }

    /// `[0, 1)` 的 53 位双精度。
    pub fn uniform(&mut self) -> f64 {
        let a = f64::from(self.next_u32() >> 5);
        let b = f64::from(self.next_u32() >> 6);
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    }

    /// `int(uniform() * n)`（`randint`）。
    fn randint(&mut self, n: usize) -> usize {
        (self.uniform() * n as f64) as usize
    }

    /// `choice_d2`：`thr = u*sum(d2)`，顺序累加取第一个 `cum >= thr` 的下标。
    fn choice_d2(&mut self, d2: &[f64]) -> usize {
        let total: f64 = d2.iter().sum();
        let thr = self.uniform() * total;
        let mut cum = 0.0f64;
        for (i, v) in d2.iter().enumerate() {
            cum += *v;
            if cum >= thr {
                return i;
            }
        }
        d2.len().saturating_sub(1)
    }
}

// ---------------------------------------------------------------- k-means

/// `kmeans_seeded`：定种子 k-means++（每次 init 重新 `MT19937(0)`，n_init=10，
/// Lloyd 标签不变即停，inertia 平局取先达）。
fn kmeans_seeded(x: &[Vec<f64>], k: usize) -> Vec<usize> {
    let n = x.len();
    if k == 0 || n == 0 {
        return vec![0; n];
    }
    let dim = x.first().map_or(0, Vec::len);
    // 10 轮 init 里第一轮的 inertia 必然 < inf，golden 的 `assert best_labels is not None` 恒成立
    let mut best_labels = vec![0usize; n];
    let mut best_inertia = f64::INFINITY;

    for _init in 0..KMEANS_N_INIT {
        // 每个 init 重新播种：实现简单且完全确定（golden 同）
        let mut rng = Mt19937::new(KMEANS_SEED);
        let mut centers = vec![vec![0.0f64; dim]; k];
        centers[0] = x[rng.randint(n)].clone();
        let mut d2: Vec<f64> = x.iter().map(|p| squared_distance(p, &centers[0])).collect();
        for center in centers.iter_mut().skip(1) {
            let idx = rng.choice_d2(&d2);
            *center = x[idx].clone();
            for (slot, p) in d2.iter_mut().zip(x.iter()) {
                // np.minimum(d2, ||x - centers[c]||²)
                *slot = slot.min(squared_distance(p, center));
            }
        }

        let mut labels = vec![0usize; n];
        for it in 0..LLOYD_MAX_ITERS {
            let new_labels = assign(x, &centers);
            // golden：先算新标签，`_ > 0` 且与旧标签相同则 break（未赋值的轮次不算）
            if it > 0 && new_labels == labels {
                break;
            }
            labels = new_labels;
            update_centers(x, &mut centers, &labels);
        }

        let inertia = inertia_of(x, &centers, &labels);
        if inertia < best_inertia - INERTIA_TIE_EPS {
            best_inertia = inertia;
            best_labels = labels;
        }
    }
    best_labels
}

/// 每点到最近中心的标签（`dist.argmin(axis=1)`，并列取第一个）。
fn assign(x: &[Vec<f64>], centers: &[Vec<f64>]) -> Vec<usize> {
    x.iter()
        .map(|p| {
            let mut best = 0usize;
            let mut best_dist = f64::INFINITY;
            for (c, center) in centers.iter().enumerate() {
                let dist = squared_distance(p, center);
                if dist < best_dist {
                    best_dist = dist;
                    best = c;
                }
            }
            best
        })
        .collect()
}

/// 空簇保留旧中心（numpy 侧 `if len(pts): centers[c] = pts.mean(axis=0)`）。
fn update_centers(x: &[Vec<f64>], centers: &mut [Vec<f64>], labels: &[usize]) {
    let dim = centers.first().map_or(0, Vec::len);
    let mut sums = vec![vec![0.0f64; dim]; centers.len()];
    let mut counts = vec![0usize; centers.len()];
    for (p, &label) in x.iter().zip(labels.iter()) {
        if label >= centers.len() {
            continue;
        }
        for (slot, v) in sums[label].iter_mut().zip(p.iter()) {
            *slot += *v;
        }
        counts[label] += 1;
    }
    for (c, center) in centers.iter_mut().enumerate() {
        if counts[c] == 0 {
            continue;
        }
        let denom = counts[c] as f64;
        for (slot, sum) in center.iter_mut().zip(sums[c].iter()) {
            *slot = *sum / denom;
        }
    }
}

/// `((x - centers[labels])**2).sum()`。
fn inertia_of(x: &[Vec<f64>], centers: &[Vec<f64>], labels: &[usize]) -> f64 {
    x.iter()
        .zip(labels.iter())
        .filter_map(|(p, &label)| centers.get(label).map(|c| squared_distance(p, c)))
        .sum()
}

// ---------------------------------------------------------------- merge_by_cos

/// `merge_by_cos`：标签均值中心 L2 归一后取上三角余弦亲和的最大对，
/// 低于阈值即停；否则把大标签并入小标签并重编号（> 被并标签的减一）。
fn merge_by_cos(labels_in: &[usize], embs: &[Vec<f64>], cos_thr: f64) -> Vec<usize> {
    let mut labels = labels_in.to_vec();
    let dim = embs.first().map_or(0, Vec::len);
    loop {
        let spk_num = labels.iter().copied().max().unwrap_or(0) + 1;
        if spk_num <= 1 {
            break;
        }
        // centers = embs[labels == i].mean(0)，再逐行 L2 归一
        let mut centers = vec![vec![0.0f64; dim]; spk_num];
        let mut counts = vec![0usize; spk_num];
        for (p, &label) in embs.iter().zip(labels.iter()) {
            if label >= spk_num {
                continue;
            }
            for (slot, v) in centers[label].iter_mut().zip(p.iter()) {
                *slot += *v;
            }
            counts[label] += 1;
        }
        for (c, center) in centers.iter_mut().enumerate() {
            if counts[c] > 0 {
                let denom = counts[c] as f64;
                for slot in center.iter_mut() {
                    *slot /= denom;
                }
            }
            let norm = dot(center, center).sqrt();
            if norm > 0.0 {
                for slot in center.iter_mut() {
                    *slot /= norm;
                }
            }
        }
        // aff = np.triu(centers @ centers.T, 1)；np.argmax 等价于按行主序取第一个最大值
        // （下三角与对角恒为 0，所以先填 0 再只在 c > r 时算余弦）。
        let mut best = (0usize, 0usize);
        let mut best_aff = 0.0f64;
        for (r, row) in centers.iter().enumerate() {
            for (c, other) in centers.iter().enumerate() {
                if c <= r {
                    continue;
                }
                let aff = dot(row, other);
                if aff > best_aff {
                    best_aff = aff;
                    best = (r, c);
                }
            }
        }
        if best_aff < cos_thr {
            break;
        }
        let (keep, drop) = best;
        for label in labels.iter_mut() {
            if *label == drop {
                *label = keep;
            } else if *label > drop {
                *label -= 1;
            }
        }
    }
    labels
}
