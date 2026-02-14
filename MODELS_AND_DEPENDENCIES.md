# 🤖 AI 模型和依赖项说明

## 项目使用的 AI 模型

本项目使用多个预训练的 AI 模型来实现精确的词级对齐和视频生成。

### 1. 主要模型

#### 1.1 Wav2Vec2 (torchaudio) - 强制对齐 ⭐ 核心模型
- **用途**: 精确的词级时间戳对齐（Forced Alignment）
- **模型**: `WAV2VEC2_ASR_BASE_960H`
- **来源**: Facebook AI Research (FAIR)
- **大小**: ~360 MB
- **下载位置**: 
  - Windows: `C:\Users\[用户名]\.cache\torch\hub\checkpoints\`
  - Linux/Mac: `~/.cache/torch/hub/checkpoints/`
- **自动下载**: 首次运行时自动下载
- **代码位置**: `karaoke_alignment_generator.py` 第 95 行
  ```python
  bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
  model = bundle.get_model()
  ```

**为什么使用这个模型？**
- 提供毫秒级精度的词级对齐
- 解决 Whisper 数字识别不一致问题（如 "two hundred" vs "200"）
- 使用原文文本进行强制对齐，而不依赖语音识别结果

#### 1.2 OpenAI Whisper - 语音识别（备用方法）
- **用途**: 词级时间戳提取（旧方法，保留兼容）
- **模型**: `base` (也可选择 tiny, small, medium, large)
- **来源**: OpenAI
- **大小**: 
  - tiny: ~39 MB
  - base: ~74 MB
  - small: ~244 MB
  - medium: ~769 MB
  - large: ~1550 MB
- **下载位置**:
  - Windows: `C:\Users\[用户名]\.cache\whisper\`
  - Linux/Mac: `~/.cache/whisper/`
- **自动下载**: 首次运行时自动下载
- **代码位置**: `karaoke_alignment_generator.py` 第 308 行
  ```python
  import whisper
  model = whisper.load_model("base")
  ```

**注意**: 
- 当前版本主要使用 Wav2Vec2 进行强制对齐
- Whisper 作为备用方法，当强制对齐失败时使用

### 2. 支持库和框架

#### 2.1 PyTorch (torch)
- **版本**: >= 2.0.0
- **用途**: 深度学习框架，运行所有 AI 模型
- **大小**: ~2 GB（包含 CUDA 支持）
- **安装**: `pip install torch>=2.0.0`

#### 2.2 Torchaudio
- **版本**: >= 2.0.0
- **用途**: 音频处理和 Wav2Vec2 模型
- **安装**: `pip install torchaudio>=2.0.0`

#### 2.3 Transformers (Hugging Face)
- **版本**: >= 4.21.0
- **用途**: 模型加载和推理
- **安装**: `pip install transformers>=4.21.0`

#### 2.4 其他依赖
- **librosa**: 音频分析和处理
- **soundfile**: 音频文件读写
- **numpy**: 数值计算
- **moviepy**: 视频生成和编辑
- **Pillow**: 图像处理
- **opencv-python**: 视频处理

## 模型下载和缓存

### 首次运行时会发生什么？

1. **自动下载模型**
   - Wav2Vec2 模型 (~360 MB)
   - Whisper base 模型 (~74 MB)
   - 总计约 ~450 MB

2. **下载时间**
   - 取决于网络速度
   - 通常需要 5-15 分钟

3. **缓存位置**
   ```
   Windows:
   C:\Users\[用户名]\.cache\torch\hub\checkpoints\
   C:\Users\[用户名]\.cache\whisper\
   
   Linux/Mac:
   ~/.cache/torch/hub/checkpoints/
   ~/.cache/whisper/
   ```

4. **后续运行**
   - 模型已缓存，无需重新下载
   - 启动速度快

### 手动下载模型（可选）

如果网络不稳定，可以手动下载模型：

#### Wav2Vec2 模型
```python
import torchaudio
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()  # 这会触发下载
```

#### Whisper 模型
```python
import whisper
model = whisper.load_model("base")  # 这会触发下载
```

或者使用命令行：
```bash
python -c "import whisper; whisper.load_model('base')"
```

## 系统要求

### 最低配置
- **CPU**: 4 核心以上
- **内存**: 8 GB RAM
- **存储**: 5 GB 可用空间（包含模型）
- **Python**: 3.8+

### 推荐配置
- **CPU**: 8 核心以上
- **内存**: 16 GB RAM
- **GPU**: NVIDIA GPU with CUDA support（可选，但强烈推荐）
  - 显存: 4 GB+
  - CUDA: 11.7+
- **存储**: 10 GB 可用空间

### GPU 加速（推荐）

如果有 NVIDIA GPU，安装 CUDA 版本的 PyTorch：

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

GPU 加速可以将处理速度提升 5-10 倍！

## 模型性能对比

### Wav2Vec2 vs Whisper

| 特性 | Wav2Vec2 (强制对齐) | Whisper (语音识别) |
|------|---------------------|-------------------|
| 精度 | ⭐⭐⭐⭐⭐ 毫秒级 | ⭐⭐⭐⭐ 较高 |
| 速度 | ⭐⭐⭐⭐ 快 | ⭐⭐⭐ 中等 |
| 数字处理 | ✅ 完美（使用原文） | ⚠️ 可能不一致 |
| 依赖原文 | ✅ 需要 | ❌ 不需要 |
| 适用场景 | 已知文本的对齐 | 未知文本的转录 |

**本项目选择**: 主要使用 Wav2Vec2，因为我们有原文文本，需要精确对齐。

## 常见问题

### Q1: 首次运行很慢？
A: 正常现象，正在下载模型（~450 MB）。后续运行会快很多。

### Q2: 如何查看模型是否已下载？
A: 检查缓存目录：
```bash
# Windows
dir C:\Users\[用户名]\.cache\torch\hub\checkpoints\
dir C:\Users\[用户名]\.cache\whisper\

# Linux/Mac
ls ~/.cache/torch/hub/checkpoints/
ls ~/.cache/whisper/
```

### Q3: 可以删除模型缓存吗？
A: 可以，但下次运行会重新下载。不建议删除。

### Q4: 如何使用 GPU 加速？
A: 
1. 安装 NVIDIA 驱动和 CUDA
2. 安装 CUDA 版本的 PyTorch
3. 代码会自动检测并使用 GPU

检查 GPU 是否可用：
```python
import torch
print(torch.cuda.is_available())  # 应该返回 True
```

### Q5: 内存不足怎么办？
A: 
- 关闭其他程序
- 使用更小的 Whisper 模型（tiny 而不是 base）
- 处理较短的音频文件

### Q6: 模型下载失败？
A: 
- 检查网络连接
- 使用代理或 VPN
- 手动下载模型文件并放到缓存目录

## 模型文件清单

运行项目需要的所有模型文件：

```
~/.cache/torch/hub/checkpoints/
├── wav2vec2_fairseq_base_ls960_asr_ls960.pth  (~360 MB)

~/.cache/whisper/
├── base.pt  (~74 MB)

总计: ~450 MB
```

## 许可证

- **Wav2Vec2**: MIT License (Facebook AI Research)
- **Whisper**: MIT License (OpenAI)
- **PyTorch**: BSD License

所有模型都是开源的，可以免费用于商业和非商业用途。

## 参考资料

- [Wav2Vec2 论文](https://arxiv.org/abs/2006.11477)
- [Whisper 论文](https://arxiv.org/abs/2212.04356)
- [Torchaudio 文档](https://pytorch.org/audio/stable/index.html)
- [OpenAI Whisper GitHub](https://github.com/openai/whisper)
