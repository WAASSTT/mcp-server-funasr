"""实时语音识别模块

参考FunASR流式识别最佳实践,实现边输入边识别的实时转录功能。

参考文档:
- https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README_zh.md#实时语音识别
- https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/paraformer_streaming/
"""

import funasr
import numpy as np
import logging
from typing import Generator, Optional, Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeTranscriber:
    """实时语音识别器类

    参考FunASR官方流式识别实现:
    1. 使用cache维护流式状态
    2. 支持chunk_size配置延迟
    3. 每个chunk独立推理,通过is_final控制输出

    模型说明:
    - 使用 paraformer-zh-streaming (官方流式ASR模型)
    - ModelScope ID: iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online
    - chunk_size [0,10,5]: 600ms实时粒度, 300ms未来信息
    - chunk_stride = chunk_size[1]*960 = 9600 samples (600ms@16kHz)

    参考文档:
    - https://www.modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online
    - https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/paraformer_streaming/
    """

    def __init__(
        self,
        asr_model_path: str = "paraformer-zh-streaming",
        vad_model_path: str = None,
        device: str = "cpu",
        ncpu: int = 4,
        chunk_size: list = None,
        encoder_chunk_look_back: int = 4,
        decoder_chunk_look_back: int = 1,
        vad_kwargs: Optional[Dict[str, Any]] = None,
        asr_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """初始化实时语音识别器

        参数:
            asr_model_path: ASR流式模型ModelScope ID或本地路径
                           默认: paraformer-zh-streaming (官方流式识别模型)
                           ModelScope: iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online
                           首次运行自动下载到./Model/
            vad_model_path: VAD模型路径(可选,流式ASR通常内置VAD)
            device: 运行设备 ("cpu", "cuda:0")
            ncpu: CPU线程数(仅device="cpu"时有效)
            chunk_size: 流式延迟配置 [0, 10, 5]:
                       - 0: 保留位
                       - 10: 当前chunk大小(10*60ms=600ms)
                       - 5: 未来信息(5*60ms=300ms)
                       可选: [0,8,4] (480ms), [0,5,5] (300ms)
            encoder_chunk_look_back: encoder自注意力回溯块数
            decoder_chunk_look_back: decoder交叉注意力回溯块数
            vad_kwargs: VAD额外参数
            asr_kwargs: ASR推理额外参数
        """
        self.device = device
        self.ncpu = ncpu
        self.asr_model_path = asr_model_path
        self.vad_model_path = vad_model_path

        # 流式识别参数(参考FunASR文档)
        self.chunk_size = chunk_size or [0, 10, 5]  # 600ms延迟
        self.encoder_chunk_look_back = encoder_chunk_look_back
        self.decoder_chunk_look_back = decoder_chunk_look_back

        # VAD参数配置(流式识别通常不需要VAD)
        self.vad_kwargs = vad_kwargs or {}

        # ASR推理默认参数
        self.asr_kwargs = asr_kwargs or {}

        # 加载模型
        logger.info("正在加载实时识别模型...")
        logger.info(f"  ASR模型: {asr_model_path}")
        if vad_model_path:
            logger.info(f"  VAD模型: {vad_model_path}")
        logger.info(f"  设备: {device}")
        logger.info(f"  Chunk配置: {self.chunk_size} (延迟: {self.chunk_size[1]*60}ms)")

        try:
            # 参考FunASR流式识别模型加载方式
            # FunASR会自动下载到./Model/目录(由MODELSCOPE_CACHE指定)
            model_kwargs = {
                "model": asr_model_path,
                "device": device,
                "disable_update": True,
                "model_hub": "ms",  # 使用ModelScope
            }

            # CPU模式添加线程数
            if device == "cpu":
                model_kwargs["ncpu"] = ncpu

            # 如果指定了VAD模型则添加
            if vad_model_path:
                model_kwargs["vad_model"] = vad_model_path
                model_kwargs["vad_kwargs"] = self.vad_kwargs

            self.model = funasr.AutoModel(**model_kwargs)
            logger.info("✓ 实时识别模型加载成功")

        except Exception as e:
            logger.error(f"✗ 模型加载失败: {e}", exc_info=True)
            raise

    def transcribe_chunk(
        self,
        audio_chunk: np.ndarray,
        cache: dict,
        is_final: bool = False,
        sample_rate: int = 16000,
        **kwargs,
    ) -> Dict[str, Any]:
        """处理单个音频块的流式识别

        参考FunASR流式识别标准流程:
        ```python
        cache = {}  # 会话级cache
        for chunk in audio_chunks:
            result = model.generate(
                input=chunk,
                cache=cache,
                is_final=(i==last),
                chunk_size=[0,10,5],
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1
            )
        ```

        参数:
            audio_chunk: 音频数据块(numpy.ndarray, float32)
                        建议大小: chunk_size[1]*960 samples (如600ms@16kHz=9600)
            cache: 会话级缓存字典(外部维护,用于保持流式状态)
            is_final: 是否为最后一个chunk(True时强制输出)
            sample_rate: 采样率(默认16000Hz)
            **kwargs: 额外参数,会覆盖默认配置

        返回:
            识别结果字典:
            - status: "success" 或 "error"
            - text: 当前chunk识别的文本
            - is_final: 是否为最终结果
            - timestamp: 时间戳信息(如果模型支持)
        """
        try:
            # 确保音频数据格式正确
            if not isinstance(audio_chunk, np.ndarray):
                audio_chunk = np.array(audio_chunk, dtype=np.float32)
            elif audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)

            # 记录处理信息
            chunk_duration = len(audio_chunk) / sample_rate * 1000
            logger.debug(
                f"处理chunk: {len(audio_chunk)}样本 ({chunk_duration:.0f}ms), is_final={is_final}"
            )

            # 构建generate参数(参考FunASR文档)
            generate_kwargs = self.asr_kwargs.copy()
            generate_kwargs.update(
                {
                    "cache": cache,
                    "is_final": is_final,
                    "chunk_size": self.chunk_size,
                    "encoder_chunk_look_back": self.encoder_chunk_look_back,
                    "decoder_chunk_look_back": self.decoder_chunk_look_back,
                }
            )
            generate_kwargs.update(kwargs)  # 允许外部覆盖

            # 执行流式识别
            result = self.model.generate(input=audio_chunk, **generate_kwargs)

            # 格式化结果
            if result:
                formatted = self._format_result(result, is_final=is_final)
                if formatted.get("text"):
                    logger.info(f"识别结果: {formatted['text']} (final={is_final})")
                return formatted
            else:
                # 空结果(正常情况,可能是静音段)
                return {
                    "status": "success",
                    "text": "",
                    "is_final": is_final,
                    "timestamp": None,
                }

        except Exception as e:
            logger.error(f"流式识别错误: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e),
                "text": "",
                "is_final": is_final,
            }

    def _format_result(self, result: Any, is_final: bool = False) -> Dict[str, Any]:
        """格式化识别结果

        参数:
            result: 原始识别结果
            is_final: 是否为最终结果

        返回:
            格式化的结果字典
        """
        formatted = {
            "status": "success",
            "is_final": is_final,
            "text": "",
            "timestamp": None,
        }

        # 解析结果
        if isinstance(result, list) and len(result) > 0:
            first_item = result[0]
            if isinstance(first_item, dict):
                formatted["text"] = first_item.get("text", "")
                if "timestamp" in first_item:
                    formatted["timestamp"] = first_item["timestamp"]
            elif isinstance(first_item, str):
                formatted["text"] = first_item
        elif isinstance(result, dict):
            formatted["text"] = result.get("text", "")
            if "timestamp" in result:
                formatted["timestamp"] = result["timestamp"]
        elif isinstance(result, str):
            formatted["text"] = result

        return formatted
