"""FunASR MCP 服务器核心模块 v0.3.0

提供语音识别的核心功能模块:
- RealtimeTranscriber: 实时流式语音识别器
- BatchTranscriber: 批量语音识别器

版本: 0.3.0
更新日期: 2025-12-04
"""

from .realtime_transcriber import RealtimeTranscriber
from .batch_transcriber import BatchTranscriber

__all__ = ["RealtimeTranscriber", "BatchTranscriber"]
__version__ = "0.3.0"
