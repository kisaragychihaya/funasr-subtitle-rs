"""生成 punct=on / punct=off 的期望 SRT（参考实现直跑）。

`tests/golden/` 只存了 `punct=auto` 的 `output.srt`；on/off 两种模式的期望值改用同一个
参考引擎（`FunasrEngine.transcribe`）现场跑出来，只改 `JobOptions.punctuation`，
Rust 侧 `tests/test_pipeline.rs` 与这里落盘的 SRT 逐字节比对。

**必须把 fbank 的 dither 钉成 0**（与 `reference/dump_golden.py` 的 `pin_dither` 同一原因）：
SenseVoice / Paraformer 的 config.yaml 没写 dither，funasr-onnx 的 `WavFrontend` 默认
`dither=1.0`，会在波形上叠 ~1LSB 高斯噪声，同一段音频两次 `extract_feat` 结果都不同，
期望值就不可能和 Rust 侧逐字节一致。Rust 的 `frontend` 预设本来就是 dither=0。

用法（仓库根目录）：

    ../.venv-golden/Scripts/python.exe reference/testdata/gen_punct_modes.py

输出：

    reference/testdata/e2e_punct_on/<fixture>.srt
    reference/testdata/e2e_punct_off/<fixture>.srt
    reference/testdata/e2e_punct_modes.json   # 生成参数与 dither 原始取值留档
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # reference/testdata
REPO = HERE.parent.parent                       # funasr-subtitle-rs
BACKEND = REPO / "reference" / "funasr-subtitle" / "backend"
FIXTURE_DIR = REPO / "tests" / "fixtures"

sys.path.insert(0, str(BACKEND))

from app.engine import build_engine  # noqa: E402
from app.models.registry import SENSEVOICE  # noqa: E402
from app.schemas import JobOptions  # noqa: E402
from app.subtitle import to_srt  # noqa: E402

SAMPLE_RATE = 16_000
MAX_CHARS = 30          # 与原项目 DEFAULT_MAX_CHARS 一致
MODES = ("on", "off")
FIXTURES = ("sample_zh", "sample_en_runon", "sample_long")


def decode_s16le(path: Path) -> np.ndarray:
    """ffmpeg -> 16kHz 单声道 s16le（与 dump_golden.py / Rust decode 同一条命令）。"""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {path}: {proc.stderr.decode('utf-8', 'replace')[:400]}"
        )
    raw = proc.stdout
    return np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").copy()


def pcm_to_float(pcm: np.ndarray) -> np.ndarray:
    return (pcm.astype(np.float32) / 32768.0).astype(np.float32)


def pin_dither(engine) -> dict:  # noqa: ANN001
    """把 ASR/VAD 前端的 dither 钉成 0（同 dump_golden.pin_dither），返回原始取值。"""
    pinned: dict[str, float] = {}
    asr = getattr(engine, "_asr", None)
    if asr is not None and getattr(asr, "frontend", None) is not None:
        pinned["asr_dither_from"] = float(asr.frontend.opts.frame_opts.dither)
        asr.frontend.opts.frame_opts.dither = 0.0
    vad = getattr(engine, "_vad", None)
    if vad is not None and getattr(vad, "frontend", None) is not None:
        pinned["vad_dither_from"] = float(vad.frontend.opts.frame_opts.dither)
        vad.frontend.opts.frame_opts.dither = 0.0
    return pinned


def main() -> int:
    t0 = time.perf_counter()
    print(f"[load] {SENSEVOICE.id}", flush=True)
    engine = build_engine(SENSEVOICE.id)
    engine.load()
    pinned = pin_dither(engine)
    print(f"[load] done in {time.perf_counter() - t0:.1f}s  dither pinned: {pinned}", flush=True)

    wavs = {name: pcm_to_float(decode_s16le(FIXTURE_DIR / f"{name}.wav")) for name in FIXTURES}
    files: list[dict] = []
    for mode in MODES:
        out_dir = HERE / f"e2e_punct_{mode}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in FIXTURES:
            opts = JobOptions(model_id=SENSEVOICE.id, punctuation=mode, max_chars=MAX_CHARS)
            segs = engine.transcribe(wavs[name], SAMPLE_RATE, opts)
            srt = to_srt(segs)
            rel = f"e2e_punct_{mode}/{name}.srt"
            (HERE / rel).write_bytes(srt.encode("utf-8"))
            files.append({"path": rel, "bytes": len(srt.encode("utf-8")), "segments": len(segs)})
            print(f"  {rel}: {len(segs)} segments", flush=True)
            for s in segs:
                print(f"    [{s.start_ms:>6},{s.end_ms:>6}] {s.text}", flush=True)

    manifest = {
        "generated_by": "funasr-subtitle-rs/reference/testdata/gen_punct_modes.py",
        "reference_impl": "github.com/rockbenben/funasr-subtitle (backend/app) FunasrEngine.transcribe",
        "why": "tests/golden 只有 punct=auto 的 output.srt；on/off 两种模式的期望值用同一参考引擎"
               "现场生成，只改 JobOptions.punctuation。",
        "params": {
            "sample_rate": SAMPLE_RATE,
            "model_id": SENSEVOICE.id,
            "max_chars": MAX_CHARS,
            "punct_modes": list(MODES),
            "language": "auto",
            "quantize": True,
            "device": "cpu",
            "fbank_dither": 0.0,
        },
        "dither_pinned": pinned,
        "files": files,
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (HERE / "e2e_punct_modes.json").write_bytes(text.encode("utf-8"))
    print(f"[done] {time.perf_counter() - t0:.1f}s -> {HERE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
