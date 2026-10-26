# funasr-subtitle-rs

纯本地离线的音视频字幕生成 CLI：输入音频/视频，输出带时间戳字幕（SRT / VTT / TXT / JSON）。
[rockbenben/funasr-subtitle](https://github.com/rockbenben/funasr-subtitle) 的 **Rust 移植版**：
同一批量化 ONNX 模型、同一条推理管线（VAD → ASR → 标点 → 切句），端到端输出与原项目**逐字节一致**（golden 对拍验证），纯 CPU，无需 Python / torch / CUDA。

中文 / 粤语识别通常优于通用 Whisper（FunASR 模型针对中文场景训练）；SenseVoiceSmall 为非自回归模型，实测 RTF ≈ 0.04（约 25× 实时）。

## 特性

- 🔒 纯本地离线：首次运行自动从 ModelScope 下载模型（默认 SenseVoiceSmall ≈ 235MB），之后完全离线
- ⚡ 纯 CPU、单文件可执行：推理用 ONNX Runtime（`ort`），无需显卡
- 🎯 三模型可选：SenseVoiceSmall（多语言·默认·快）/ Paraformer-zh（中文高精度）/ Paraformer-zh 热词版
- 🌐 多语言：中 / 粤 / 英 / 日 / 韩，自动识别或手动指定
- ✂️ 专业字幕切分：按标点/停顿切句、显示宽度（CJK=1 / 拉丁=0.5）行宽限制、单条 ≤7s、缩写感知断句、cue 严格单调不重叠
- 🔤 标点三模式：`auto`（中文加标点、英文按停顿）/ `on` / `off`
- 📝 导出 SRT / VTT / TXT / JSON

## 使用

前置：ffmpeg 在 PATH（任意近期版本）。

```bash
# 构建
cargo build --release
# 可执行文件：target/release/funasr-subtitle-rs(.exe)

# 基本用法：拖个音视频进来，得到同名 .srt
funasr-subtitle-rs input.mp4

# 常用选项
funasr-subtitle-rs input.mp4 -o out.vtt --format vtt
funasr-subtitle-rs input.mp4 --language zh --model iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx
funasr-subtitle-rs input.mp4 --punct off --max-chars 20
funasr-subtitle-rs input.mp4 --hotwords "机器学习 深度学习"   # 仅热词版模型
funasr-subtitle-rs input.mp4 -o - --format txt               # 输出到 stdout
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `-o, --output` | 输入改后缀 | 输出路径；`-` 表示 stdout |
| `--model` | SenseVoiceSmall-onnx | ASR 模型 id（见下表） |
| `--language` | `auto` | `auto/zh/yue/en/ja/ko` |
| `--punct` | `auto` | `auto/on/off` |
| `--max-chars` | 30 | 每行显示宽度上限（0=不限；CJK=1、拉丁=0.5） |
| `--max-cue-ms` | 7000 | 单条字幕最长毫秒 |
| `--hotwords` | 空 | 热词（空格分隔，仅热词版生效） |
| `--threads` | 4 | ONNX Runtime 线程数 |
| `--format` | `srt` | `srt/vtt/txt/json` |
| `--models-dir` | 见下 | 模型缓存目录 |

模型缓存目录缺省读环境变量 `FUNASR_SUBTITLE_MODELS_ROOT`，否则用系统缓存目录
（Windows：`%LOCALAPPDATA%\funasr-subtitle-rs\models`；Linux/macOS：`~/.cache/funasr-subtitle-rs/models`）。

### 可选模型

| id | 说明 | 体积 |
|---|---|---|
| `iic/SenseVoiceSmall-onnx` | 多语言·默认·快（自带标点） | ~235MB |
| `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx` | 中文高精度 | ~240MB |
| `iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx` | 中文高精度·支持热词 | ~900MB |

首次使用某模型会自动下载（含共享的 VAD / ct-punc 辅助模型）。

## 工作原理

```
ffmpeg 解码 → 16kHz 单声道 PCM
  → FSMN-VAD 切出语音段
  → ASR（SenseVoice: CTC 贪心 + 逐 token 时间 / Paraformer: 整句）
  → ct-punc 补标点（按语言与 punct 模式）
  → 按标点/停顿/行宽切句，时间戳对齐
  → enforce_monotonic → SRT/VTT/TXT/JSON
```

与原项目的移植对齐说明：

- 特征前端逐 op 复刻 kaldi-native-fbank（含 kissfft 混合基 f32 FFT、预加重尾迭代、LFR m=7/n=6、CMVN f64），与参考实现 golden 特征逐位一致（max diff ≈ 4 ulp）
- 推理输入缩放 s16/32768.0；fbank `dither=0`（上游默认 dither=1.0 本身不确定，已按 golden 口径钉零）
- SenseVoice 时间戳 = CTC 帧位置线性映射（前 4 前缀帧扣除），句级粒度
- 英文标点「原生 SenseVoice 优先、ct-punc 兜底」、全角转半角、按停顿模式去句末标点等策略逐行移植（含上游怪癖，如给已含句号的中文句补逗号）

## 测试

```bash
cargo test                # 纯逻辑与 golden 对齐（不需模型下载，约 150 个用例）
cargo clippy --all-targets   # 零警告
```

golden 资产（`tests/golden/`）由原项目 Python 引擎 dump 生成、两遍字节一致后冻结；
真实模型端到端对拍为 `#[ignore]` 测试，备好模型缓存后运行：

```bash
FUNASR_SUBTITLE_MODELS_ROOT=<模型根> cargo test -- --include-ignored --test-threads=1
```

## 与原项目的差异

- 去掉 Web UI / 本地服务：纯 CLI；说话人分离（cam++，原 full/torch 版功能）未移植
- 无 GPU 路径（纯 CPU，正是原项目 ONNX 便携包的后端）
- 模型下载用 ModelScope HTTP API 直拉（不经 modelscope SDK）

## License

MIT（与原项目一致）。注意预训练模型（SenseVoice / Paraformer / FSMN-VAD / CT-Punc）各有其
Model License，见对应 ModelScope 模型页；分发模型文件前请逐项确认。
