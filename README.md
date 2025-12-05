# 🎙️ FunASR MCP 服务器

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FunASR](https://img.shields.io/badge/FunASR-1.2.0%2B-green.svg)](https://github.com/modelscope/FunASR)
[![FastMCP](https://img.shields.io/badge/FastMCP-2.5.1%2B-orange.svg)](https://github.com/jlowin/fastmcp)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> 专业的中文语音识别MCP服务器，支持实时流式识别、语音增强和LLM后处理

**快速导航**: [快速开始](#-快速开始) • [功能特性](#-功能特性) • [使用指南](#-使用指南) • [API文档](#-api-端点)

---

## 📖 简介

基于阿里达摩院 [FunASR](https://github.com/modelscope/FunASR) 和 [FastMCP](https://github.com/jlowin/fastmcp) 框架构建的语音识别服务器，提供企业级中文语音识别能力。

### 🎯 核心优势

- 🏆 **高精度识别** - 采用Paraformer系列模型，业界领先准确率
- ⚡ **低延迟流式** - 600ms实时响应，支持对话式交互
- 🔇 **专业降噪** - ClearerVoice-Studio深度语音增强
- 🤖 **AI后处理** - Qwen2.5-7B蒸馏模型智能优化识别文本
- 🔒 **本地部署** - 完全离线运行，数据隐私安全
- 🌐 **标准协议** - 完整实现MCP规范，易于集成

## ✨ 功能特性

### 🎯 识别能力

| 功能 | 说明 | 技术 |
|------|------|------|
| **批量识别** | 高精度离线识别 | Paraformer-large |
| **流式识别** | 600ms低延迟实时响应 | Paraformer-Streaming |
| **标点恢复** | 自动添加标点符号 | CT-Transformer |
| **说话人分离** | 识别不同说话人 | CAM++ |
| **热词定制** | 提升特定词汇准确率 | 自定义词表 |

### 🔊 音频处理

| 功能 | 说明 | 技术 |
|------|------|------|
| **语音增强** | 专业级降噪处理 | ClearerVoice-Studio |
| **噪声抑制** | 多场景降噪（空调/键盘/环境噪音） | DNS-Challenge |
| **VAD检测** | 智能语音活动检测 | 模型内置 |
| **LLM优化** | 智能文本后处理 | Qwen2.5-7B |

### 🛠️ 系统能力

- ✅ **浏览器支持** - 直接上传录音文件识别
- ✅ **高并发** - 线程安全，多客户端并发
- ✅ **MCP协议** - 完整实现标准规范
- ✅ **实时监控** - 连接状态和性能监控
- ✅ **本地部署** - 无需API，数据安全

## 📋 系统要求

- **Python**: 3.10 或更高版本
- **操作系统**: Linux / macOS / Windows
- **内存**: 推荐 8GB 以上
- **磁盘空间**: 约 3GB (用于模型缓存)
- **GPU** (可选): CUDA 11.x+ 用于加速推理

## 🚀 快速开始

### 步骤1: 环境准备

```bash
# 克隆项目
git clone https://github.com/WAASSTT/mcp-server-funasr.git
cd mcp-server-funasr

# 安装依赖
chmod +x setup.sh
./setup.sh
```

### 步骤2: 启动服务

```bash
# 基础启动（默认启用增强和LLM）
python main.py

# 生产环境（多进程）
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

服务启动后会自动下载所需模型到 `./Model` 目录。

### 步骤3: 验证服务

```bash
# 健康检查
curl http://localhost:8000/health

# 查看活跃连接
curl http://localhost:8000/connections
```

**提示**: 首次启动需要下载约3GB模型文件，请耐心等待。

## 📚 使用指南

### 方式1: Python客户端 - 批量识别

```bash
# 基础识别
python client_batch.py transcribe audio/test.wav

# VAD分段识别
python client_batch.py transcribe audio/test.wav --vad
```

### 方式2: Python客户端 - 实时识别

```bash
# 显示模式（终端显示）
python client_realtime.py

# 输入法模式（自动输入文本）
python client_realtime.py --input-mode --show-status
```

### 方式3: WebSocket API

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/realtime');

ws.onopen = () => {
    ws.send(JSON.stringify({ type: 'start' }));
    ws.send(audioBuffer);  // 16kHz, 16-bit PCM
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('识别结果:', data.text);
};
```

## 🎯 技术架构

### 处理流程

```mermaid
graph LR
    A[音频输入<br/>16kHz PCM] --> B[语音增强<br/>ClearerVoice]
    B --> C[语音识别<br/>Paraformer]
    C --> D[LLM后处理<br/>Qwen3]
    D --> E[识别结果<br/>优化文本]
```

### 核心技术栈

#### 1. 语音增强 - ClearerVoice-Studio

**技术**: DNS-Challenge (Deep Noise Suppression)
**模型**: iic/ClearerVoice-Studio
**功能**: 深度降噪、去混响、语音清晰度提升

- 🔇 专业级多场景降噪（空调、键盘、环境噪音）
- ⚡ 实时处理，延迟可控
- 🎯 显著提升识别准确率

#### 2. 语音识别 - Paraformer系列

**模型**: Paraformer-large / Paraformer-Streaming
**特性**: 内置VAD、高精度、低延迟

- 🎯 模型级智能VAD，零额外延迟
- 📝 自动标点恢复（CT-Transformer）
- 👥 说话人分离（CAM++）
- ⚡ 600ms实时响应

#### 3. LLM后处理 - Qwen2.5-7B

**模型**: Qwen2.5-7B-Instruct (蒸馏版)
**功能**: 智能文本优化
**特点**: 轻量级、快速推理、低资源占用

- ✨ 口语转书面语
- 📝 优化标点和分段
- 🎯 修正语法错误
- 💡 保持原意不失真

### 技术优势

| 特性 | 传统方案 | 本方案 |
|------|---------|--------|
| 降噪方案 | 简单滤波器 | ✅ DNS-Challenge深度学习 |
| VAD检测 | 独立模块 | ✅ 模型内置，零延迟 |
| 文本优化 | 规则后处理 | ✅ LLM智能优化 |
| 部署方式 | 依赖API | ✅ 完全本地化 |
| 数据安全 | 云端传输 | ✅ 本地处理 |

## ⚙️ 配置说明

所有配置在 `main.py` 的 `Config` 类中集中管理：

### 服务器配置

```python
class Config:
    SERVER_HOST = "0.0.0.0"          # 监听地址
    SERVER_PORT = 8000               # 监听端口
    TIMEOUT_KEEP_ALIVE = 75          # 连接保持超时
```

### 识别配置

```python
    # 实时识别
    REALTIME_MODEL = "paraformer-zh-streaming"
    REALTIME_CHUNK_SIZE = [0, 10, 5]  # 600ms延迟
    REALTIME_DEVICE = "cpu"           # 或 "cuda"

    # 批量识别
    BATCH_MODEL = "paraformer-zh"
    BATCH_VAD_MODEL = "fsmn-vad"
    BATCH_PUNC_MODEL = "ct-punc-c"
    BATCH_SPK_MODEL = "cam++"
    BATCH_DEVICE = "cpu"
    BATCH_HOTWORD = "魔搭"            # 热词定制
```

### 延迟优化

通过 `REALTIME_CHUNK_SIZE` 调整延迟：

| chunk_size | 延迟 | 适用场景 |
|-----------|------|----------|
| [0, 5, 5] | 300ms | 对话式交互 |
| [0, 8, 4] | 480ms | 一般场景 |
| [0, 10, 5] | 600ms | 默认推荐 |

### 功能开关

```python
    ENABLE_AUDIO_ENHANCEMENT = True   # 语音增强
    ENABLE_LLM_POSTPROCESS = True     # LLM后处理
    LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 蒸馏模型
    LLM_DEVICE = "cuda"               # LLM设备（可选cpu）
```

## 📊 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/mcp` | POST | MCP 协议端点 |
| `/upload-audio` | POST | 浏览器音频上传 |
| `/ws/realtime` | WebSocket | 实时流式识别 |
| `/health` | GET | 健康检查 |
| `/connections` | GET | 活跃连接状态 |

## 🎯 使用的模型

### 批量识别

- **ASR**: `paraformer-zh` - 高精度非流式识别
- **VAD**: `fsmn-vad` - 语音活动检测
- **标点**: `ct-punc` - 标点符号恢复
- **说话人**: `cam++` - 说话人分离

### 实时识别

- **流式 ASR**: `paraformer-zh-streaming` - 低延迟流式识别，内置 VAD

## 📁 项目结构

```text
mcp-server-funasr/
├── main.py                       # 服务器主程序
├── pyproject.toml                # 项目配置
├── client_batch.py               # 批量识别客户端
├── client_realtime.py            # 实时识别客户端
├── core/                         # 核心模块
│   ├── batch_transcriber.py      # 批量识别器
│   └── realtime_transcriber.py   # 实时识别器
├── audio/                        # 测试音频
└── Model/                        # 模型缓存(自动下载)
```

## 🔍 故障排除

### 模型下载失败

```bash
export HF_ENDPOINT=https://hf-mirror.com
python download_models.py
```

### GPU 问题

```bash
# 检查 CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 使用 CPU
# 在 main.py 中设置 device="cpu"
```

### 内存不足

- 降低 `batch_size_s` 参数
- 减少 `ncpu` 线程数
- 限制并发连接数

### 连接问题

```bash
# 检查端口
netstat -tulpn | grep 8000

# 查看连接
curl http://localhost:8000/connections
```

## 📝 性能优化

### CPU 优化

- 增加 `ncpu` 参数(如 8-16)
- 使用多进程: `--workers 4`

### GPU 优化

- 设置 `device="cuda:0"`
- 调整 `batch_size_s`

### 并发优化

- Nginx 反向代理
- Redis 会话管理
- 多实例部署

## 📝 更新日志

### v0.5.0 (2025-12-05) - AI增强版

#### 核心功能

- ✨ 深度语音增强 (ClearerVoice-Studio DNS-Challenge)
- ✨ LLM后处理优化 (本地Qwen3-235B)
- ✨ 代码重构 (Config类集中管理)
- 🎯 专业级降噪处理
- 🎯 模型内置VAD（零额外延迟）

#### 系统改进

- 🔧 完全本地化部署，移除OpenAI依赖
- 🔧 简化依赖，清理不必要的包
- 🔧 并发安全保护
- 📊 实时监控和统计

### v0.3.0 (2025-12-04)

- ✨ 统一实时客户端（显示/输入法模式）
- 🐛 修复内存泄漏和噪音误触发

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！请遵循以下规范：

- 提交前运行测试
- 遵循代码风格
- 更新相关文档

## 📄 许可证

本项目采用 MIT License - 详见 [LICENSE](LICENSE)

## 🙏 致谢

- [FunASR](https://github.com/modelscope/FunASR) - 阿里达摩院语音实验室提供的强大ASR框架
- [FastMCP](https://github.com/jlowin/fastmcp) - 优秀的MCP协议框架
- [ModelScope](https://www.modelscope.cn/) - 提供模型托管和下载服务
- [ClearerVoice-Studio](https://www.modelscope.cn/models/iic/ClearerVoice-Studio) - 专业语音增强模型
- [Qwen2.5](https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct) - 智能文本后处理支持

## 🔗 相关链接

- **项目文档**: [GitHub Repository](https://github.com/WAASSTT/mcp-server-funasr)
- **FunASR文档**: [官方教程](https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README_zh.md)
- **MCP规范**: [Model Context Protocol](https://modelcontextprotocol.io/)
- **ModelScope**: [模型库](https://www.modelscope.cn/models)

---

**当前版本**: v0.5.0
**最后更新**: 2025-12-05
**作者**: WAASSTT

[⬆ 返回顶部](#️-funasr-mcp-服务器)
