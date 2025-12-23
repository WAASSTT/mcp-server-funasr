"""FunASR MCP 服务器核心模块 v3.0.0

提供语音识别的核心功能模块:
- RealtimeTranscriber: 实时流式语音识别器
- BatchTranscriber: 批量语音识别器
- StreamingLLMPostProcessor: 流式LLM后处理器 (GGUF)
- AudioEnhancer: 语音增强处理器
- device_utils: 统一GPU/CPU设备检测工具

版本: 3.0.0
更新日期: 2025-12-22
"""

from .realtime_transcriber import RealtimeTranscriber
from .batch_transcriber import BatchTranscriber

__all__ = ["RealtimeTranscriber", "BatchTranscriber"]
__version__ = "3.0.0"
