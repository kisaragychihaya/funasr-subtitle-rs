"""生成 golden 比对基准：用参考实现（reference/funasr-subtitle）自己的引擎跑 fixture，
把每个阶段的中间结果写成 tests/golden/ 下的二进制/JSON 文件，供 Rust 侧逐阶段对齐。

设计原则
--------
* **不重写引擎**：VAD / SenseVoice / Paraformer / 热词版 / ct-punc 全部通过
  `app.engine.build_engine()` 加载（=> `FunasrEngine.load()`，与原项目完全一致），
  segment 切分也直接调 `FunasrEngine.transcribe()`（完整端到端路径）。
* **中间结果**只有 sensevoice 需要额外插桩：按原项目 `_sensevoice_timed()` 的同一顺序
  （`extract_feat` → `infer` → `_ctc_greedy_with_frames` → `sp.id_to_piece` → 线性映射）
  把 feats / logits / tokens 顺手 dump 出来，并在脚本里 assert 重建的 token 序列与
  `engine._sensevoice_timed()` 的输出**逐元素相同**（防止插桩漂移）。
* **推理输入 = 落盘 PCM**：fixture 用 ffmpeg 解成 16k 单声道 s16le，i16 字节直接写
  `input.s16bin`，同时 `pcm / 32768.0` 作为送引擎的 float32 数组。
  另跑一遍原项目的 `decode_to_16k_mono()`（f32le 路径）做交叉校验，差值记进 input.json。
* 只用 CPU + quantize=True 的 int8 ONNX，推理应当是确定性的，两遍运行应字节一致。

用法
----
    python dump_golden.py                       # 写到 ../tests/golden
    python dump_golden.py --out <临时目录>        # 可复现性验证用
    python dump_golden.py --only sensevoice     # 只跑某几段（调试用）
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # funasr-subtitle-rs/reference
REPO = HERE.parent                              # funasr-subtitle-rs
WORKSPACE = REPO.parent                         # 工作区根
BACKEND = HERE / "funasr-subtitle" / "backend"  # 参考实现 Python 包根（app.*）
FIXTURE_DIR = REPO / "tests" / "fixtures"
ASSET_DIR = HERE / "assets"

sys.path.insert(0, str(BACKEND))

from app.config import MAX_CUE_MS, PUNC_CHUNK  # noqa: E402
from app.engine import build_engine  # noqa: E402
from app.engine.funasr_engine import (  # noqa: E402
    _SPECIAL_TOKEN_RE,
    _ctc_greedy_with_frames,
)
from app.media import decode_to_16k_mono  # noqa: E402
from app.models.registry import (  # noqa: E402
    PARAFORMER_ZH,
    PARAFORMER_ZH_HOTWORD,
    SENSEVOICE,
)
from app.schemas import JobOptions  # noqa: E402
from app.subtitle import to_srt  # noqa: E402

SAMPLE_RATE = 16_000
SV_PREFIX_FRAMES = 4  # FunasrEngine._SV_PREFIX_FRAMES
MAX_CHARS = 30        # 与原项目默认 DEFAULT_MAX_CHARS 一致
HOTWORDS = "机器学习 深度学习"

FIXTURES = (
    ("sample_zh", "sample_zh.wav"),
    ("sample_en_runon", "sample_en_runon.wav"),
    ("sample_long", "sample_long.wav"),
)
# 只对这两个跑 Paraformer（中文模型）
PARAFORMER_FIXTURES = ("sample_zh", "sample_long")


# ---------------------------------------------------------------- 落盘工具
def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    path.write_bytes(text.encode("utf-8"))


def write_f32(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def seg_to_dict(seg) -> dict:  # noqa: ANN001  # app.schemas.Segment
    return {"start_ms": int(seg.start_ms), "end_ms": int(seg.end_ms), "text": seg.text}


def segs_to_list(segs) -> list[dict]:  # noqa: ANN001
    return [seg_to_dict(s) for s in segs]


# ---------------------------------------------------------------- 解码
def decode_s16le(path: Path) -> np.ndarray:
    """ffmpeg -> 16kHz 单声道 s16le 原始字节 -> int16 采样（原项目同一条 ffmpeg 命令，只是 s16le）。"""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {proc.stderr.decode('utf-8', 'replace')[:400]}")
    raw = proc.stdout
    raw = raw[: len(raw) // 2 * 2]  # 尾部残留不足 2 字节时截断
    return np.frombuffer(raw, dtype="<i2").copy()


def pcm_to_float(pcm: np.ndarray) -> np.ndarray:
    return (pcm.astype(np.float32) / 32768.0).astype(np.float32)


# ---------------------------------------------------------------- TTS 素材（热词用例）
HOTWORD_SENTENCE = "我们正在学习机器学习的基础知识"


def ensure_hotword_asset() -> Path:
    """合成（或复用缓存的）含「机器学习」的短句 wav。

    SAPI 合成只做一次并缓存到 reference/assets/，保证两遍 golden 运行拿到同一个 wav；
    缓存存在时不再重新合成。
    """
    out = ASSET_DIR / "hotword_case_src.wav"
    if out.exists() and out.stat().st_size > 0:
        return out
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$s.SelectVoice('Microsoft Huihui Desktop');"
        f"$s.SetOutputToWaveFile('{out}');"
        f"$s.Speak('{HOTWORD_SENTENCE}');"
        "$s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True)
    if not out.exists():
        raise RuntimeError("SAPI 合成失败：找不到 Microsoft Huihui Desktop 语音")
    return out


# ---------------------------------------------------------------- stage: input
def stage_inputs(out_root: Path) -> dict[str, np.ndarray]:
    wavs: dict[str, np.ndarray] = {}
    print("[input] decode 3 fixtures -> 16k mono s16le", flush=True)
    for name, fname in FIXTURES:
        src = FIXTURE_DIR / fname
        pcm = decode_s16le(src)
        wav = pcm_to_float(pcm)
        # 交叉校验：原项目的解码路径（f32le @16k）量化到 s16 后应与我们落盘的一致
        ref_wav, ref_sr = decode_to_16k_mono(src)
        if ref_sr != SAMPLE_RATE:
            raise RuntimeError(f"unexpected sample rate {ref_sr}")
        if ref_wav.size != pcm.size:
            raise RuntimeError(
                f"{name}: f32le path produced {ref_wav.size} samples, s16le path {pcm.size}"
            )
        ref_pcm = np.round(ref_wav * 32768.0).clip(-32768, 32767).astype(np.int16)
        max_lsb = int(np.max(np.abs(ref_pcm.astype(np.int32) - pcm.astype(np.int32))))
        max_f = float(np.max(np.abs(ref_wav - wav)))

        d = out_root / "input" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "input.s16bin").write_bytes(np.ascontiguousarray(pcm, dtype="<i2").tobytes())
        write_json(d / "input.json", {
            "fixture": name,
            "source": f"tests/fixtures/{fname}",
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "samples": int(pcm.size),
            "duration_ms": int(round(pcm.size / SAMPLE_RATE * 1000)),
            "dtype": "int16",
            "endianness": "little",
            "file": "input.s16bin",
            "ffmpeg_args": f"-vn -ac 1 -ar {SAMPLE_RATE} -f s16le pipe:1",
            "float_scale": 32768.0,
            "engine_array": "pcm.astype(float32) / 32768.0",
            "vs_project_f32le_path": {
                "max_abs_float_diff": max_f,
                "max_abs_s16_lsb_diff": max_lsb,
            },
        })
        wavs[name] = wav
        print(f"  {name}: {pcm.size} samples ({pcm.size / SAMPLE_RATE:.2f}s), "
              f"vs f32le path max {max_lsb} LSB", flush=True)
    return wavs


# ---------------------------------------------------------------- stage: vad
def stage_vad(engine, wavs: dict[str, np.ndarray], out_root: Path) -> dict[str, list[list[int]]]:
    print("[vad] Fsmn_vad(quantize=True) on 3 fixtures", flush=True)
    spans_all: dict[str, list[list[int]]] = {}
    for name, _ in FIXTURES:
        wav = wavs[name]
        vad_out = engine._vad(wav)
        spans = vad_out[0] if vad_out else []
        spans = [[int(s[0]), int(s[1])] for s in spans]
        spans_all[name] = spans
        write_json(out_root / "vad" / name / "spans.json", spans)
        print(f"  {name}: {spans}", flush=True)
    return spans_all


# ---------------------------------------------------------------- stage: sensevoice
def sensevoice_span_dump(asr, chunk: np.ndarray, language: str,
                         span_start_ms: int, span_end_ms: int):
    """复刻 FunasrEngine._sensevoice_timed，同时把 feats / logits / tokens 抓出来。"""
    feats, feats_len = asr.extract_feat([chunk])
    lid = np.array([asr._get_lid(language)], dtype=np.int32)
    tn = np.array([asr._get_tnid("withitn")], dtype=np.int32)
    ctc_logits, enc_lens = asr.infer(feats, feats_len, lid, tn)
    ctc_logits = np.asarray(ctc_logits)
    T = int(enc_lens[0])
    pairs = _ctc_greedy_with_frames(ctc_logits[0], T, blank_id=0)

    sp = asr.tokenizer.sp
    prefix = SV_PREFIX_FRAMES
    speech_frames = max(1, T - prefix)
    span = max(0, span_end_ms - span_start_ms)

    tokens: list[dict] = []
    timed: list[tuple[str, int]] = []
    for tok_id, frame in pairs:
        raw = sp.id_to_piece(int(tok_id))
        is_special = bool(_SPECIAL_TOKEN_RE.fullmatch(raw))
        piece = raw.replace("▁", " ")
        rel = (frame - prefix) / speech_frames
        rel = 0.0 if rel < 0 else (1.0 if rel > 1 else rel)
        ms = int(span_start_ms + rel * span)
        tokens.append({
            "token_id": int(tok_id),
            "frame": int(frame),
            "piece": raw,
            "text": "" if is_special else piece,
            "special": is_special,
            "ms": ms,
        })
        if is_special or not piece:
            continue
        timed.append((piece, ms))

    return feats, feats_len, ctc_logits, T, tokens, timed


def pin_dither(engine) -> dict:  # noqa: ANN001
    """把 fbank 的 dither 钉成 0，让推理可复现（重要！）。

    SenseVoice / Paraformer 的 config.yaml **没写 dither**，于是落到 funasr-onnx
    `WavFrontend.__init__` 的默认 `dither=1.0`，即 kaldi fbank 会在波形上叠一层
    1 LSB 级高斯噪声（波形先 ×32768）。同一段音频两次 `extract_feat` 结果不同，
    量化 ONNX 的 logits 能差到十几，CTC 帧位置也会漂 1 帧 → 上游这条路径本身就
    不是确定性的，golden 若照抄就没法比对。VAD 的 config.yaml 里本来就写了
    `dither: 0.0`（因此 VAD spans 是确定的）。

    这里把 ASR/VAD 前端都钉成 0，返回原始取值以便写进 index.json 留档。
    """
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


def stage_sensevoice(engine, wavs: dict[str, np.ndarray], spans_all, out_root: Path) -> None:
    print("[sensevoice] per-span feats/logits/tokens + full-path segments", flush=True)
    asr = engine._asr
    for name, _ in FIXTURES:
        wav = wavs[name]
        spans = spans_all[name] or [[0, int(len(wav) / SAMPLE_RATE * 1000)]]
        base = out_root / "sensevoice" / name
        used_spans: list[dict] = []
        first: dict | None = None
        for idx, (start_ms, end_ms) in enumerate(spans):
            a = max(0, int(start_ms / 1000 * SAMPLE_RATE))
            b = min(len(wav), int(end_ms / 1000 * SAMPLE_RATE))
            chunk = wav[a:b]
            if chunk.size < SAMPLE_RATE * 0.05:  # 与原引擎一致：<50ms 跳过
                print(f"  {name} span {idx}: skipped (<50ms)", flush=True)
                continue
            feats, feats_len, logits, T, tokens, timed = sensevoice_span_dump(
                asr, chunk, "auto", start_ms, end_ms
            )
            # 防漂移：插桩结果必须与原引擎逐元素一致
            ref_timed = engine._sensevoice_timed(chunk, "auto", start_ms, end_ms)
            if ref_timed != timed:
                raise AssertionError(f"{name} span {idx}: instrumented tokens != engine tokens")

            d = base / "spans" / f"span_{idx:03d}"
            write_f32(d / "feats.f32bin", feats[0])
            write_json(d / "feats.json", {"shape": [int(v) for v in feats[0].shape]})
            write_f32(d / "logits.f32bin", logits[0][:T])
            write_json(d / "logits.json", {"shape": [int(v) for v in logits[0][:T].shape]})
            write_json(d / "enc_len.json", [T])
            write_json(d / "tokens.json", tokens)
            meta = {
                "span_index": idx,
                "start_ms": int(start_ms),
                "end_ms": int(end_ms),
                "sample_range": [a, b],
                "samples": int(chunk.size),
                "feats_len": int(feats_len[0]),
                "enc_len": T,
                "lid": int(asr._get_lid("auto")),
                "textnorm": int(asr._get_tnid("withitn")),
                "prefix_frames": SV_PREFIX_FRAMES,
                "speech_frames": max(1, T - SV_PREFIX_FRAMES),
            }
            write_json(d / "span.json", meta)
            used_spans.append(meta)
            if first is None:
                first = {"dir": d, "tokens": tokens}

        # 顶层文件 = span_000 的同一份数据（便于 Rust 侧简单路径直取）
        if first is not None:
            for f in ("feats.f32bin", "feats.json", "logits.f32bin",
                      "logits.json", "enc_len.json", "tokens.json", "span.json"):
                (base / f).write_bytes((first["dir"] / f).read_bytes())
            write_json(base / "index.json", {
                "note": "顶层 feats/logits/tokens/enc_len/span = spans/span_000 的副本；"
                        "全部分段在 spans/<span_xxx>/ 下",
                "spans": used_spans,
            })

        # 完整引擎路径（punct=auto, max_chars=30）
        opts = JobOptions(model_id=SENSEVOICE.id, punctuation="auto", max_chars=MAX_CHARS)
        segs = engine.transcribe(wav, SAMPLE_RATE, opts)
        seg_list = segs_to_list(segs)
        write_json(base / "segments.json", seg_list)
        srt = to_srt(segs)
        (base / "output.srt").write_bytes(srt.encode("utf-8"))
        print(f"  {name}: {len(used_spans)} spans, {len(seg_list)} segments", flush=True)
        for s in seg_list:
            print(f"    [{s['start_ms']:>6},{s['end_ms']:>6}] {s['text']}", flush=True)


# ---------------------------------------------------------------- stage: punc
PUNC_CASES = [
    # 1) 中文无标点长句
    "今天我们来讲一下语音识别技术的基本原理它把声音信号一步一步地转成文字",
    # 2) 英文无句末标点
    "this is a simple english sentence without any final punctuation",
    # 3) 大小写混合的英文句
    "Apple Released a New iPhone and the Stock Market Reacted Strongly",
    # 4) 110 字以上长文本（> PUNC_CHUNK+30 = 140 字符，触发 _punctuate 的分块路径）
    "今天我们来讲一下语音识别技术的基本原理它把声音信号转成文字的过程大致分为三步第一步是把声音"
    "切成一小段一小段的帧第二步是用神经网络把每一帧变成一个高维向量第三步是把这些向量解码成文字"
    "在解码的时候我们会用到一种叫做CTC的方法它可以在不知道每个字具体位置的情况下完成对齐这就是"
    "为什么端到端模型可以省掉传统语音识别里复杂的声学模型和语言模型分步训练的流程标点在语音识别"
    "里同样重要因为标点能帮助读者断句和理清语义关系如果没有标点读者往往需要反复回读才能理解一句话"
    "的意思",
    # 5) 已含句末标点的句子
    "今天天气很好。",
]


def stage_punc(engine, out_root: Path) -> None:
    print(f"[punc] CT_Transformer(quantize=True), PUNC_CHUNK={PUNC_CHUNK}", flush=True)
    # 触发懒加载（SenseVoice 内置标点，引擎默认不预载 ct-punc）
    engine._punctuate("预热")
    punc = engine._punc

    cases: list[dict] = []
    raw_cases: list[dict] = []
    chunk_info: list[dict] = []
    for text in PUNC_CASES:
        out = engine._punctuate(text)
        cases.append({"input": text, "output": out})
        raw = punc(text)
        raw_cases.append({"input": text, "output": raw[0], "punc_ids": [int(x) for x in raw[1]]})
        # 复刻 _punctuate 的分块计划（只有超长文本才会真的分块）
        if len(text) <= PUNC_CHUNK + 30:
            chunk_info.append({"input_len": len(text), "chunked": False, "chunks": [text]})
        else:
            parts, i, n = [], 0, len(text)
            while i < n:
                end = min(n, i + PUNC_CHUNK)
                if end < n:
                    sp = text.rfind(" ", i, end)
                    if sp > i:
                        end = sp
                chunk = text[i:end].strip()
                if chunk:
                    parts.append(chunk)
                i = end
            chunk_info.append({"input_len": len(text), "chunked": True, "chunks": parts})
        print(f"  len={len(text):>3} -> {out[:40]!r}", flush=True)

    write_json(out_root / "punc" / "cases.json", cases)
    write_json(out_root / "punc" / "raw_ct.json", raw_cases)
    write_json(out_root / "punc" / "chunk_plan.json", {
        "punc_chunk": PUNC_CHUNK,
        "single_call_limit": PUNC_CHUNK + 30,
        "note": "FunasrEngine._punctuate：len<=punc_chunk+30 单次调用；否则按 punc_chunk 分块，"
                "块内回退到最后一个空格，逐块补标点后用单个空格 join",
        "cases": chunk_info,
    })


# ---------------------------------------------------------------- stage: paraformer / contextual
def paraformer_dump(engine, spec, wavs, spans_all, out_root: Path, stage_name: str,
                    hotwords: str, fixtures) -> None:
    for name in fixtures:
        wav = wavs[name]
        spans = spans_all[name] or [[0, int(len(wav) / SAMPLE_RATE * 1000)]]
        per_span: list[dict] = []
        texts: list[str] = []
        for start_ms, end_ms in spans:
            a = max(0, int(start_ms / 1000 * SAMPLE_RATE))
            b = min(len(wav), int(end_ms / 1000 * SAMPLE_RATE))
            chunk = wav[a:b]
            if chunk.size < SAMPLE_RATE * 0.05:
                continue
            text = engine._asr_paraformer(chunk, hotwords)
            per_span.append({
                "start_ms": int(start_ms), "end_ms": int(end_ms),
                "sample_range": [a, b], "text": text,
            })
            texts.append(text)

        opts = JobOptions(model_id=spec.id, punctuation="auto", max_chars=MAX_CHARS,
                          hotwords=hotwords)
        segs = engine.transcribe(wav, SAMPLE_RATE, opts)
        seg_list = segs_to_list(segs)
        d = out_root / stage_name / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "text.txt").write_bytes(("\n".join(texts) + "\n").encode("utf-8"))
        write_json(d / "preds.json", per_span)
        write_json(d / "segments.json", seg_list)
        print(f"  {name}: {len(per_span)} spans / {len(seg_list)} segments", flush=True)
        for s in seg_list:
            print(f"    [{s['start_ms']:>6},{s['end_ms']:>6}] {s['text']}", flush=True)


def stage_hotword_case(engine, out_root: Path) -> None:
    """「机器学习」短句：带热词 vs 不带热词的识别对比。"""
    print("[contextual] hotword_case (TTS 短句, 带/不带热词)", flush=True)
    src = ensure_hotword_asset()
    pcm = decode_s16le(src)
    wav = pcm_to_float(pcm)
    d = out_root / "contextual"
    d.mkdir(parents=True, exist_ok=True)
    (d / "hotword_case.s16bin").write_bytes(np.ascontiguousarray(pcm, dtype="<i2").tobytes())
    write_json(d / "hotword_case_audio.json", {
        "sentence": HOTWORD_SENTENCE,
        "samples": int(pcm.size),
        "sample_rate": SAMPLE_RATE,
        "duration_ms": int(round(pcm.size / SAMPLE_RATE * 1000)),
        "dtype": "int16",
        "endianness": "little",
        "file": "hotword_case.s16bin",
        "tts": "Windows SAPI / Microsoft Huihui Desktop (zh-CN)，源 wav 缓存在 reference/assets/",
    })

    def run(hotwords: str) -> dict:
        opts = JobOptions(model_id=PARAFORMER_ZH_HOTWORD.id, punctuation="auto",
                          max_chars=MAX_CHARS, hotwords=hotwords)
        segs = engine.transcribe(wav, SAMPLE_RATE, opts)
        seg_list = segs_to_list(segs)
        return {
            "hotwords": hotwords,
            "text": "".join(s["text"] for s in seg_list),
            "segments": seg_list,
        }

    case = {
        "audio": "hotword_case.s16bin",
        "note": "ContextualParaformer(quantize=True)，hotwords 空格分隔；"
                "不带热词=空字符串（与引擎 options.hotwords or '' 一致）。"
                "hotword_token_ids/lengths 是 ContextualParaformer.proc_hotword() 的输出"
                "（N×10 的 token id 矩阵 + 每行真实长度），用于校验 Rust 侧热词分词。",
        "with_hotwords": run(HOTWORDS),
        "without_hotwords": run(""),
    }
    ids, lens = engine._asr.proc_hotword(HOTWORDS)
    case["hotword_token_ids"] = ids.tolist()
    case["hotword_lengths"] = lens.tolist()
    case["outputs_identical"] = (
        case["with_hotwords"]["text"] == case["without_hotwords"]["text"]
    )
    write_json(d / "hotword_case.json", case)
    print(f"  hot: {case['with_hotwords']['text']!r}", flush=True)
    print(f"  non: {case['without_hotwords']['text']!r}", flush=True)
    print(f"  identical={case['outputs_identical']}, hotword ids={ids.tolist()}", flush=True)


# ---------------------------------------------------------------- manifest
def build_manifest(out_root: Path, pinned: dict[str, dict]) -> None:
    files = []
    for p in sorted(out_root.rglob("*")):
        if p.is_file() and p.name != "index.json":
            files.append({"path": p.relative_to(out_root).as_posix(), "size": p.stat().st_size})
    write_json(out_root / "index.json", {
        "generated_by": "funasr-subtitle-rs/reference/dump_golden.py",
        "reference_impl": "github.com/rockbenben/funasr-subtitle (backend/app)",
        "note": "所有 .f32bin 是小端 float32、行主序；s16bin 是小端 int16。"
                "推理输入 = s16bin/32768.0。",
        "dither_pinned": {
            "why": "SenseVoice/Paraformer 的 config.yaml 未写 dither，funasr-onnx 的 "
                   "WavFrontend 默认 dither=1.0（kaldi fbank 在波形上叠 ~1LSB 高斯噪声），"
                   "同一音频两次 extract_feat 结果不同、量化 logits 能差十几、CTC 帧会漂，"
                   "上游这条路径本身不是确定性的。golden 全流程强制 dither=0（VAD 的 "
                   "config.yaml 本来就写 0.0），这样两遍运行才能字节一致、Rust 侧才有可比对象。",
            "original_values": pinned,
        },
        "params": {
            "sample_rate": SAMPLE_RATE,
            "punct": "auto",
            "max_chars": MAX_CHARS,
            "max_cue_ms": MAX_CUE_MS,
            "punc_chunk": PUNC_CHUNK,
            "sensevoice_prefix_frames": SV_PREFIX_FRAMES,
            "language": "auto",
            "textnorm": "withitn",
            "quantize": True,
            "device": "cpu",
            "fbank_dither": 0.0,
            "intra_op_num_threads": 4,
        },
        "files": files,
    })


def print_manifest(out_root: Path) -> None:
    rows = []
    for p in sorted(out_root.rglob("*")):
        if p.is_file():
            rows.append((p.relative_to(out_root).as_posix(), p.stat().st_size))
    print("\n==== golden files ====", flush=True)
    total = 0
    for rel, size in rows:
        total += size
        print(f"{size:>10}  {out_root.as_posix()}/{rel}", flush=True)
    print(f"{total:>10}  TOTAL ({len(rows)} files)", flush=True)


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "tests" / "golden"),
                    help="输出根目录（默认 funasr-subtitle-rs/tests/golden）")
    ap.add_argument("--only", default="", help="逗号分隔：input,vad,sensevoice,punc,paraformer,contextual")
    args = ap.parse_args(argv)

    out_root = Path(args.out).resolve()
    only = {s.strip() for s in args.only.split(",") if s.strip()} or None

    def want(stage: str) -> bool:
        return only is None or stage in only

    t0 = time.perf_counter()
    out_root.mkdir(parents=True, exist_ok=True)

    wavs = stage_inputs(out_root)

    # ---- SenseVoice 引擎（同时提供 VAD 与 ct-punc）----
    need_sv = want("vad") or want("sensevoice") or want("punc") or want("paraformer") or want("contextual")
    spans_all: dict[str, list[list[int]]] = {}
    pinned: dict[str, dict] = {}
    engine_sv = None
    if need_sv:
        print(f"[load] {SENSEVOICE.id}", flush=True)
        engine_sv = build_engine(SENSEVOICE.id)
        engine_sv.load()
        pinned[SENSEVOICE.id] = pin_dither(engine_sv)
        print(f"[load] done in {time.perf_counter() - t0:.1f}s  "
              f"dither pinned: {pinned[SENSEVOICE.id]}", flush=True)
        # VAD 是所有 ASR 阶段的输入，永远先跑（--only 调试时也一样）
        spans_all = stage_vad(engine_sv, wavs, out_root)
        if want("sensevoice"):
            stage_sensevoice(engine_sv, wavs, spans_all, out_root)
        if want("punc"):
            stage_punc(engine_sv, out_root)

    # ---- Paraformer ----
    if want("paraformer"):
        print(f"[load] {PARAFORMER_ZH.id}", flush=True)
        eng_pf = build_engine(PARAFORMER_ZH.id)
        eng_pf.load()
        pinned[PARAFORMER_ZH.id] = pin_dither(eng_pf)
        print(f"  dither pinned: {pinned[PARAFORMER_ZH.id]}", flush=True)
        paraformer_dump(eng_pf, PARAFORMER_ZH, wavs, spans_all, out_root,
                        "paraformer", "", PARAFORMER_FIXTURES)
        del eng_pf
        gc.collect()

    # ---- Contextual（热词版）----
    if want("contextual"):
        print(f"[load] {PARAFORMER_ZH_HOTWORD.id}", flush=True)
        eng_ctx = build_engine(PARAFORMER_ZH_HOTWORD.id)
        eng_ctx.load()
        pinned[PARAFORMER_ZH_HOTWORD.id] = pin_dither(eng_ctx)
        print(f"  dither pinned: {pinned[PARAFORMER_ZH_HOTWORD.id]}", flush=True)
        paraformer_dump(eng_ctx, PARAFORMER_ZH_HOTWORD, wavs, spans_all, out_root,
                        "contextual", HOTWORDS, ("sample_zh",))
        stage_hotword_case(eng_ctx, out_root)
        del eng_ctx
        gc.collect()

    build_manifest(out_root, pinned)
    print_manifest(out_root)
    print(f"\n[done] {time.perf_counter() - t0:.1f}s -> {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
