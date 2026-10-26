"""说话人分离 golden：cam++ 嵌入 + 谱聚类 + 后处理 + 分配，全链路基准落盘。

算法实现全部在 `diar_common.py`（契约冻结，Rust 侧 `src/speaker/` 逐行对应；
本脚本只做编排与落盘）。产物说明与运行方式见仓库 reference/README 或
`tests/golden/diar/diar_two_speakers/` 下各 JSON 的字段自述。

运行
----
    .venv-golden/Scripts/python.exe reference/dump_golden_diar.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BACKEND = HERE / "funasr-subtitle" / "backend"
FIXTURE_DIR = REPO / "tests" / "fixtures"

sys.path.insert(0, str(HERE))       # 复用 dump_golden 的工具函数 + diar_common
sys.path.insert(0, str(BACKEND))    # app.*

from diar_common import (  # noqa: E402
    CHUNK_LEN,
    CHUNK_SHIFT,
    MERGE_THR,
    PVAL,
    SAMPLE_RATE,
    distribute_spk,
    embed,
    chunk_fbank,
    postprocess,
    spectral_cluster,
    sv_chunk,
)
from dump_golden import decode_s16le, pcm_to_float, write_json  # noqa: E402

from app.engine import build_engine  # noqa: E402
from app.models.registry import PARAFORMER_ZH, SENSEVOICE  # noqa: E402
from app.schemas import JobOptions  # noqa: E402
from app.subtitle import to_srt  # noqa: E402

FIXTURE = ("diar_two_speakers", "diar_two_speakers.wav")


class _Seg:  # to_srt 只需要这几个字段
    def __init__(self, d, speaker=None):
        self.start_ms = d["start_ms"]
        self.end_ms = d["end_ms"]
        self.text = d["text"]
        self.speaker = speaker


def transcribe_with_spk(model_spec, wav: np.ndarray, spans_ms: list[list[int]],
                        span_spk: list[int]) -> dict:
    """整段转写（funasr_onnx 引擎），再按 cue 所在 VAD 段挂说话人。"""
    engine = build_engine(model_id=model_spec.id)
    engine.load()
    job = JobOptions()  # 默认 max_chars=30
    segs = engine.transcribe(wav, SAMPLE_RATE, job)
    plain = [{"start_ms": int(s.start_ms), "end_ms": int(s.end_ms), "text": s.text}
             for s in segs]
    srt_plain = to_srt([_Seg(d) for d in plain])

    # cue -> VAD 段归属（cue 完全落在段内；双指针一遍扫）
    diarized: list[dict] = []
    j = 0
    for cue in plain:
        while j < len(spans_ms) and cue["start_ms"] >= spans_ms[j][1]:
            j += 1
        spk = f"spk{span_spk[j]}" if j < len(spans_ms) else None
        d = dict(cue)
        if spk is not None:
            d["speaker"] = spk
        diarized.append(d)

    srt = to_srt([_Seg(d, d.get("speaker")) for d in diarized])
    return {"segments": diarized, "srt": srt, "srt_plain": srt_plain}


def main() -> None:
    out_dir = REPO / "tests" / "golden" / "diar" / FIXTURE[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    pcm = decode_s16le(FIXTURE_DIR / FIXTURE[1])
    wav = pcm_to_float(pcm)
    (out_dir / "input.s16bin").write_bytes(np.ascontiguousarray(pcm, dtype="<i2").tobytes())
    write_json(out_dir / "input.json", {
        "fixture": FIXTURE[0], "source": f"tests/fixtures/{FIXTURE[1]}",
        "sample_rate": SAMPLE_RATE, "channels": 1, "samples": int(pcm.size),
        "duration_ms": int(round(pcm.size / SAMPLE_RATE * 1000)),
        "dtype": "int16", "endianness": "little", "file": "input.s16bin",
        "float_scale": 32768.0, "engine_array": "pcm.astype(float32) / 32768.0",
        "note": "SAPI TTS 长回合交替：Huihui(zh) 15s | 0.8s | Zira(en) 15s | 0.8s | "
                "Huihui 15s | 0.8s | Zira 15s（reference/assets/diar_*.wav 前 15s）",
    })

    # 1) VAD（funasr_onnx 引擎，与 Rust 侧已 byte-equal）
    engine = build_engine(model_id=SENSEVOICE.id)
    engine.load()
    vad_out = engine._vad(wav)
    spans = [[int(s[0]), int(s[1])] for s in (vad_out[0] if vad_out else [])]
    if not spans:
        raise RuntimeError("VAD 未切出语音段")
    print(f"[vad] {len(spans)} spans: {spans}")

    # 2) sv_chunk + 嵌入
    chunks = sv_chunk(spans, wav)
    feats = [chunk_fbank(c[2]) for c in chunks]
    embs = np.stack([embed(f) for f in feats])
    print(f"[spk] {len(chunks)} chunks, emb {embs.shape}")
    write_json(out_dir / "chunks.json", {
        "vad_spans": spans,
        "chunks": [[round(c[0], 6), round(c[1], 6)] for c in chunks],
        "chunk_len_samples": CHUNK_LEN, "chunk_shift_samples": CHUNK_SHIFT,
        "fbank": "knf dither=0 energy_floor=1.0 num_bins=80 + utterance mean-sub",
    })
    (out_dir / "embeddings.f32").write_bytes(
        np.ascontiguousarray(embs, dtype="<f4").tobytes())

    # 3) 聚类（自动 + oracle=2）
    labels_auto = spectral_cluster(embs, oracle_num=None)
    labels_oracle2 = spectral_cluster(embs, oracle_num=2)
    # 可复现性自检：再来一遍必须逐位相同
    assert np.array_equal(labels_auto, spectral_cluster(embs, None)), "聚类不可复现!"
    write_json(out_dir / "cluster.json", {
        "n_chunks": len(chunks),
        "labels_auto": [int(x) for x in labels_auto],
        "labels_oracle2": [int(x) for x in labels_oracle2],
        "seed": 0, "n_init": 10, "merge_thr": MERGE_THR, "pval": PVAL,
    })

    # 4) 后处理 -> 说话人时间线；分配到 VAD 段
    sd = postprocess(chunks, labels_auto)
    write_json(out_dir / "sv_output.auto.json", sd)
    sd_oracle = postprocess(chunks, labels_oracle2)
    write_json(out_dir / "sv_output.oracle2.json", sd_oracle)
    span_spk = distribute_spk(spans, sd)
    write_json(out_dir / "span_labels.json", {
        "vad_spans": spans, "span_speakers": span_spk, "sv_output": sd,
    })
    print(f"[spk] span speakers: {span_spk}")

    # 5) 端到端：两个 ASR 引擎 + 说话人标签
    for spec in (PARAFORMER_ZH, SENSEVOICE):
        res = transcribe_with_spk(spec, wav, spans, span_spk)
        key = "paraformer" if spec is PARAFORMER_ZH else "sensevoice"
        write_json(out_dir / f"pipeline.{key}.json", res)
        print(f"[asr:{key}] {len(res['segments'])} cues")

    print(f"[done] -> {out_dir.relative_to(REPO)}")


if __name__ == "__main__":
    main()
