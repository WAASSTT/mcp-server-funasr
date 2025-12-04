# FunASR MCP 服务器

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![FunASR](https://img.shields.io/badge/FunASR-1.2.0%2B-green.svg)](https://github.com/modelscope/FunASR)
[![FastMCP](https://img.shields.io/badge/FastMCP-2.5.1%2B-orange.svg)](https://github.com/jlowin/fastmcp)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

基于 [FunASR](https://github.com/modelscope/FunASR) 的模型上下文协议(MCP)服务器，提供专业的中文语音识别服务。支持批量识别、实时流式识别和语音活动检测(VAD)等功能。

## ✨ 特性

- 🎯 **批量语音识别** - 使用 Paraformer 大模型进行高精度离线识别
- 🚀 **实时流式识别** - 支持 WebSocket 实时语音输入，延迟低至 600ms
- 🎤 **语音活动检测(VAD)** - 自动分段处理，智能过滤静音
- 📝 **标点符号恢复** - 自动添加标点，提升文本可读性
- 🌐 **浏览器支持** - 直接支持浏览器录音上传识别
- 🔄 **多客户端并发** - 线程安全设计，支持多用户同时使用
- 🛠️ **MCP 协议兼容** - 完整实现 Model Context Protocol 规范

## 📋 系统要求

- **Python**: 3.8 或更高版本
- **操作系统**: Linux / macOS / Windows
- **内存**: 推荐 8GB 以上
- **磁盘空间**: 约 2GB (用于模型缓存)
- **GPU** (可选): CUDA 11.x+ 用于加速推理

## 🚀 快速开始

### 1. 安装依赖

使用安装脚本一键安装:

```bash
chmod +x setup.sh
./setup.sh
```

或手动安装:

```bash
# 安装服务器依赖
pip install -e .

# 安装客户端依赖(可选)
pip install -e ".[client]"

# 安装所有依赖
pip install -e ".[all]"
```

### 2. 下载模型(可选)

首次运行时会自动下载模型，也可以预先下载:

```bash
python download_models.py
```

模型将保存在 `./Model/` 目录下。

### 3. 启动服务器

```bash
# 开发环境
python main.py

# 或使用 uvicorn(推荐生产环境)
uvicorn main:app --host 0.0.0.0 --port 8000

# 多进程模式(提升并发性能)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

服务器启动后将监听在 `http://0.0.0.0:8000`

### 4. 验证服务

```bash
# 健康检查
curl http://localhost:8000/health

# 查看活跃连接
curl http://localhost:8000/connections
```

## 📚 使用方式

### 方式一: 批量语音识别 (HTTP/MCP)

使用 Python 客户端:

```bash
# 检查服务器状态
python client_requests.py health

# 列出可用工具
python client_requests.py list-tools

# 验证音频文件
python client_requests.py validate audio/test.wav

# 识别音频文件
python client_requests.py transcribe audio/test.wav

# 识别并返回 VAD 分段信息
python client_requests.py transcribe audio/test.wav --vad
```

使用 curl 调用:

```bash
# MCP 工具调用示例
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "transcribe_audio",
      "arguments": {
        "audio_path": "audio/test.wav",
        "return_vad_segments": false
      }
    }
  }'
```

### 方式二: 实时流式识别 (WebSocket)

使用 Python 客户端:

```bash
# 需要先安装客户端依赖
pip install -e ".[client]"

# 使用麦克风进行实时识别
python client_microphone.py
```

WebSocket 协议:

```javascript
// JavaScript 示例
const ws = new WebSocket('ws://localhost:8000/ws/realtime');

ws.onopen = () => {
    // 发送开始命令
    ws.send(JSON.stringify({ type: 'start' }));
    
    // 发送音频数据 (16kHz, 16-bit PCM)
    ws.send(audioBuffer);
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'result') {
        console.log('识别结果:', data.text);
    }
};

// 停止识别
ws.send(JSON.stringify({ type: 'stop' }));
```

### 方式三: 浏览器录音上传

```bash
# 上传音频文件进行识别
curl -X POST http://localhost:8000/upload-audio \
  -H "Content-Type: audio/webm" \
  --data-binary "@recording.webm"
```

## 🔧 配置说明

### 模型配置

编辑 `main.py` 中的模型配置:

```python
# 批量识别配置
batch_transcriber = BatchTranscriber(
    asr_model_path="paraformer-zh",      # ASR 模型
    vad_model_path="fsmn-vad",           # VAD 模型
    device="cpu",                         # 使用 "cuda:0" 启用 GPU
    ncpu=4,                              # CPU 线程数
    vad_kwargs={
        "max_single_segment_time": 30000  # VAD 最大分段时长(ms)
    },
    asr_kwargs={
        "batch_size_s": 60,              # 批处理时长(秒)
        "use_itn": True,                 # 逆文本归一化
        "merge_vad": True,               # 合并短 VAD 片段
        "merge_length_s": 15,            # VAD 合并长度(秒)
    }
)

# 实时识别配置
realtime_transcriber = RealtimeTranscriber(
    asr_model_path="paraformer-zh-streaming",
    device="cpu",
    ncpu=4,
    chunk_size=[0, 10, 5],               # 延迟配置: 600ms
    encoder_chunk_look_back=4,           # 编码器回溯块数
    decoder_chunk_look_back=1,           # 解码器回溯块数
)
```

### 延迟配置

调整 `chunk_size` 参数以平衡延迟和准确性:

| chunk_size | 延迟 | 适用场景 |
|-----------|------|---------|
| [0, 5, 5] | 300ms | 对话式交互 |
| [0, 8, 4] | 480ms | 一般实时场景 |
| [0, 10, 5] | 600ms | 默认配置(推荐) |

## 🎯 使用的模型

### 批量识别模型

- **ASR 模型**: `paraformer-zh` ([Paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch))
  - 高精度非流式语音识别
  - 支持长语音处理
  - 自动标点恢复
  
- **VAD 模型**: `fsmn-vad` ([FSMN-VAD](https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch))
  - 高精度语音活动检测
  - 智能分段处理

### 实时识别模型

- **流式 ASR**: `paraformer-zh-streaming` ([Paraformer-online](https://www.modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online))
  - 真正的流式识别
  - 低延迟输出
  - 内置 VAD 功能

## 📊 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/mcp` | POST | MCP 协议端点 |
| `/upload-audio` | POST | 浏览器音频上传 |
| `/ws/realtime` | WebSocket | 实时流式识别 |
| `/health` | GET | 健康检查 |
| `/connections` | GET | 查看活跃连接 |

## 🛠️ MCP 工具列表

| 工具名称 | 说明 |
|---------|------|
| `transcribe_audio` | 批量语音识别，支持 VAD 分段 |
| `validate_audio_file` | 验证音频文件格式和属性 |

## 📁 项目结构

```
mcp-server-funasr/
├── main.py                  # 服务器主程序
├── pyproject.toml           # 项目配置
├── setup.sh                 # 安装脚本
├── restart_server.sh        # 重启脚本
├── download_models.py       # 模型下载工具
├── client_requests.py       # HTTP 客户端示例
├── client_microphone.py     # WebSocket 实时客户端
├── core/                    # 核心模块
│   ├── batch_transcriber.py      # 批量识别器
│   └── realtime_transcriber.py   # 实时识别器
├── audio/                   # 测试音频文件
└── Model/                   # 模型缓存目录
    └── models/
        └── iic/            # ModelScope 模型
```

## 🔍 故障排除

### 模型下载失败

```bash
# 手动设置镜像源
export HF_ENDPOINT=https://hf-mirror.com
export MODELSCOPE_CACHE=./Model

# 重新下载
python download_models.py
```

### GPU 相关问题

```bash
# 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 如果不可用，使用 CPU 模式
# 在 main.py 中设置 device="cpu"
```

### 内存不足

- 降低 `batch_size_s` 参数
- 减少 `ncpu` 线程数
- 使用更小的模型
- 限制并发连接数

### WebSocket 连接问题

```bash
# 检查防火墙设置
sudo ufw allow 8000/tcp

# 检查端口占用
netstat -tulpn | grep 8000
```

## 🔄 开发模式

```bash
# 启用调试日志
export LOG_LEVEL=DEBUG

# 热重载开发
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📝 性能优化建议

### CPU 优化
- 增加 `ncpu` 参数值(如 8-16)
- 使用多进程模式: `--workers 4`

### GPU 优化
- 设置 `device="cuda:0"`
- 调整批处理大小: `batch_size_s`
- 使用混合精度推理

### 并发优化
- 使用 Nginx 反向代理进行负载均衡
- 部署多个服务实例
- 使用 Redis 做会话管理

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [FunASR](https://github.com/modelscope/FunASR) - 阿里达摩院语音实验室
- [FastMCP](https://github.com/jlowin/fastmcp) - MCP 框架
- [ModelScope](https://www.modelscope.cn/) - 模型托管平台

## 📮 联系方式

如有问题或建议，请通过以下方式联系:

- 提交 [GitHub Issue](https://github.com/WAASSTT/mcp-server-funasr/issues)
- 发送邮件至项目维护者

## 🔗 相关链接

- [FunASR 官方文档](https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README_zh.md)
- [Model Context Protocol 规范](https://modelcontextprotocol.io/)
- [ModelScope 模型库](https://www.modelscope.cn/models)

---

**版本**: 0.2.0  
**更新日期**: 2025-12-04
