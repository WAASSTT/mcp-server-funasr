# Core module for FunASR server

from .realtime_transcriber import RealtimeTranscriber
from .batch_transcriber import BatchTranscriber

__all__ = ['RealtimeTranscriber', 'BatchTranscriber']