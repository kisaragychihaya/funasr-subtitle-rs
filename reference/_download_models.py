"""预下载 dump_golden.py 需要的全部 modelscope 模型（走原项目 ModelManager，行为一致）。

只用于一次性准备缓存；dump_golden.py 自己也会按需 ensure_dir。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE / "funasr-subtitle" / "backend"
sys.path.insert(0, str(BACKEND))

from app.models import (  # noqa: E402
    PUNC_MODEL,
    VAD_MODEL,
    ModelManager,
)
from app.models.registry import PARAFORMER_ZH, PARAFORMER_ZH_HOTWORD, SENSEVOICE  # noqa: E402


def main() -> None:
    mgr = ModelManager()
    print(f"models root: {mgr.root}", flush=True)
    for spec in (VAD_MODEL, SENSEVOICE, PUNC_MODEL, PARAFORMER_ZH, PARAFORMER_ZH_HOTWORD):
        t0 = time.time()
        d = mgr.ensure_dir(spec.id)
        dt = time.time() - t0
        size = sum(p.stat().st_size for p in Path(d).rglob("*") if p.is_file())
        print(
            f"OK {spec.id} -> {d}  {size / 1024 / 1024:.1f} MB  ({dt:.1f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
