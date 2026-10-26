"""一次性等价验证：lovemefan cam++.onnx vs damo 官方 torch CAMPPlus（权重级）。

 fairness 约定：两边喂**同一份** knf fbank + 均值减特征（golden 的 `chunk_fbank`），
 torch 侧直接调 CAMPPlus.forward，绕过其自带前端——这样比较的是纯模型权重。

判定：逐 chunk 余弦相似度 min >= 0.999 视为等价（ONNX 导出无损）。

运行：.venv-diar/Scripts/python.exe reference/check_torch_equivalence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WORKSPACE = REPO.parent
GOLDEN = REPO / "tests" / "golden" / "diar" / "diar_two_speakers"

sys.path.insert(0, str(HERE))
from diar_common import sv_chunk, chunk_fbank, embed  # noqa: E402

MODEL_ID = "damo/speech_campplus_sv_zh-cn_16k-common"


def main() -> None:
    import torch
    from funasr import AutoModel

    pcm = np.frombuffer((GOLDEN / "input.s16bin").read_bytes(), dtype="<i2")
    wav = pcm.astype(np.float32) / 32768.0
    meta = json.loads((GOLDEN / "chunks.json").read_text(encoding="utf-8"))
    spans = meta["vad_spans"]
    chunks = sv_chunk(spans, wav)
    feats = [chunk_fbank(c[2]) for c in chunks]
    onnx_embs = np.stack([embed(f) for f in feats])

    model = AutoModel(model=MODEL_ID, device="cpu", disable_update=True)
    net = model.model.eval()
    torch_embs = []
    with torch.no_grad():
        for f in feats:
            t = torch.from_numpy(f[None, ...]).to(torch.float32)
            torch_embs.append(net.forward(t)[0].numpy())
    torch_embs = np.stack(torch_embs)

    a = onnx_embs / np.linalg.norm(onnx_embs, axis=1, keepdims=True)
    b = torch_embs / np.linalg.norm(torch_embs, axis=1, keepdims=True)
    cos = (a * b).sum(axis=1)
    rel = np.linalg.norm(onnx_embs - torch_embs, axis=1) / np.linalg.norm(torch_embs, axis=1)
    print(f"chunks={len(chunks)}")
    print(f"cos min={cos.min():.6f} mean={cos.mean():.6f}")
    print(f"rel_l2 max={rel.max():.6f} mean={rel.mean():.6f}")
    ok = cos.min() >= 0.999
    print("EQUIVALENT" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
