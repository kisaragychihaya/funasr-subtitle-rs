"""模型目录里 ModelScope id 的护栏（修复 issue #2：热词版 full 后端 id 写错）。

背景：full（torch）后端把 `ModelSpec.full_model` 直接交给 funasr 的 `AutoModel(model=...)`。
funasr 在**拿不到模型目录时不会抛错**——`download_model()` 把下载失败 catch 掉只 print 一行，
原始 id 留在 kwargs 里，随后 `tables.model_classes.get(<原始 id>)` 查不到类，于是报出

    model 'iic/...' is not registered. Registered model keys (51): ...

而真实原因只是「这个 ModelScope 仓库不存在」：官方 id 是
`iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
（seaco 与 paraformer 之间是**下划线**），0.2.0 里误写成了连字符，ModelScope 报
`record not found` / `Repo ... not exists`，于是转写直接失败。

这个报错信息完全指不到真凶，所以在这里把 id 钉住：
1. 黄金值断言——任何手抖改动都会在 CI 里被拦下（CI 只装 [dev]，没有 funasr 也能跑）；
2. 装了 funasr 的环境（.venv-full）再和上游 `name_maps_ms` 对一次，
   确认 funasr 自己也认为这是合法仓库名。
"""
from __future__ import annotations

from app.models import REGISTRY

HOTWORD_ID = "iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx"
SEACO_PT = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"


def test_hotword_full_model_id():
    spec = REGISTRY[HOTWORD_ID]
    assert spec.supports_hotwords is True
    # 黄金值：seaco_paraformer_ 是下划线，写成 seaco-paraformer- 就是 issue #2。
    assert spec.full_model == SEACO_PT, (
        f"full 后端的 ModelScope id 被改动了：{spec.full_model!r}；"
        f"上游官方 id 是 {SEACO_PT!r}（seaco_paraformer_large，中间是下划线）"
    )
    try:
        from funasr.download.name_maps_from_hub import name_maps_ms
    except ImportError:  # CI 只装 [dev]，没装 funasr -> 黄金值断言已经够了
        return
    assert spec.full_model in set(name_maps_ms.values()), (
        f"{spec.full_model!r} 不在 funasr 的 name_maps_ms 里，"
        "funasr 不认这个仓库名，AutoModel 会以「not registered」的形式失败"
    )
