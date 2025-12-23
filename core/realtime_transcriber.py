"""实时流式语音识别模块 v4.0.0

参考FunASR和ModelScope官方流式识别最佳实践，实现边输入边识别的实时转录功能。

功能特性:
- 使用 Paraformer-Streaming 模型进行实时流式识别
- 支持低延迟输出 (600ms 实时粒度)
- 内置VAD语音活动检测
- 通过 cache 维护流式状态
- 支持 chunk_size 配置延迟
- 线程安全设计，支持多客户端并发
- 自动GPU/CPU设备检测
- 集成统一流式后处理器 (v4.0.0+)

协同设计原则 (v4.0.0+):
- ASR 负责"听清"：准确的语音识别
- LLM 负责"说人话"：将口语转换为通顺的书面语

注意: 流式模型仅支持实时ASR,不支持标点恢复和说话人识别

模型配置:
- ASR: paraformer-zh-streaming (Paraformer-Streaming, 内置VAD)

延迟配置:
- chunk_size [0, 10, 5]: 600ms 实时粒度 (默认推荐)
- chunk_size [0, 8, 4]: 480ms 实时粒度 (更低延迟)
- chunk_size [0, 5, 5]: 300ms 实时粒度 (对话式交互)

参考文档:
- https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README_zh.md#实时语音识别
- https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online
- https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/paraformer_streaming/

版本: 4.0.0
更新日期: 2025-12-23
"""

import funasr
import numpy as np
import logging
from typing import Generator, Optional, Dict, Any, List, Type

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入统一流式后处理器
StreamingPostProcessorType: Optional[Type] = None
try:
    from .streaming_postprocessor import StreamingPostProcessor

    POSTPROCESSOR_AVAILABLE = True
    StreamingPostProcessorType = StreamingPostProcessor
except ImportError:
    logger.warning("流式后处理器不可用，将跳过文本优化功能")
    POSTPROCESSOR_AVAILABLE = False


class RealtimeTranscriber:
    """实时语音识别器类

    参考FunASR官方流式识别实现:
    1. 使用cache维护流式状态
    2. 支持chunk_size配置延迟
    3. 每个chunk独立推理,通过is_final控制输出

    模型说明:
    - 使用 paraformer-zh-streaming (官方流式ASR模型)
    - chunk_size [0,10,5]: 600ms实时粒度, 300ms未来信息
    - chunk_stride = chunk_size[1]*960 = 9600 samples (600ms@16kHz)

    参考文档:
    - https://www.modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online
    - https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/paraformer_streaming/
    """

    def __init__(
        self,
        model: str = "paraformer-zh-streaming",
        device: str = "cpu",
        ncpu: int = 4,
        chunk_size: Optional[List[int]] = None,
        encoder_chunk_look_back: int = 4,
        decoder_chunk_look_back: int = 1,
        model_hub: str = "ms",
        enable_llm_postprocess: bool = False,
        llm_model_path: Optional[str] = None,
        llm_temperature: float = 0.3,
        llm_n_threads: Optional[int] = None,
        llm_n_gpu_layers: Optional[int] = None,
        **kwargs,
    ):
        """初始化实时语音识别器

        参数:
            model: ASR流式模型 (默认: "paraformer-zh-streaming")
            device: 运行设备 ("cpu" 或 "cuda:0")
            ncpu: CPU线程数
            chunk_size: 延迟配置 [0,10,5]=600ms, [0,8,4]=480ms, [0,5,5]=300ms
            encoder_chunk_look_back: encoder回溯块数
            decoder_chunk_look_back: decoder回溯块数
            model_hub: 模型仓库 ("ms"=ModelScope, "hf"=HuggingFace")
            enable_llm_postprocess: 是否启用LLM流式后处理优化
            llm_model_path: LLM模型路径 (GGUF格式，如qwen2.5-7b-instruct-q4_k_m.gguf)
            llm_temperature: LLM温度参数 (0.1-0.5)
            llm_n_threads: LLM CPU线程数 (None=自动)
            llm_n_gpu_layers: LLM GPU加速层数 (None=自动检测，0=纯CPU)
            **kwargs: 其他参数

        注意: 流式模型内置VAD,不支持外部VAD/标点/说话人模型
        """
        self.model = model
        self.device = device
        self.ncpu = ncpu
        self.model_hub = model_hub
        self.enable_llm_postprocess = enable_llm_postprocess

        # 流式参数配置
        self.chunk_size = chunk_size or [0, 10, 5]
        self.encoder_chunk_look_back = encoder_chunk_look_back
        self.decoder_chunk_look_back = decoder_chunk_look_back

        # 其他配置参数
        self.kwargs = kwargs

        # 初始化流式后处理器（协同层：ASR听清 + LLM说人话）
        self.postprocessor = None
        if (
            enable_llm_postprocess
            and POSTPROCESSOR_AVAILABLE
            and StreamingPostProcessorType is not None
            and llm_model_path
        ):
            try:
                logger.info("正在初始化流式后处理器...")
                logger.info("  架构: ASR听清 + LLM说人话 (协同设计)")
                self.postprocessor = StreamingPostProcessorType(
                    model_path=llm_model_path,
                    temperature=llm_temperature,
                    max_tokens=512,
                    enable_fallback=True,
                    n_threads=llm_n_threads,
                    n_gpu_layers=llm_n_gpu_layers,
                    context_window_size=3,
                    min_buffer_size=2,
                    max_buffer_size=5,
                    enable_timestamp_alignment=True,
                    enable_quality_check=True,
                    min_retention_rate=0.4
                )
                if self.postprocessor.is_available():
                    logger.info("✓ 流式后处理器已启用")
                    logger.info("  特性: LLM优化 + 智能缓冲 + 上下文感知 + 质量保证")
                    stats = self.postprocessor.get_stats()
                    llm_info = stats.get('llm_info', {})
                    device_info = "GPU" if llm_info.get('n_gpu_layers', 0) > 0 else "CPU"
                    logger.info(f"  运行设备: {device_info}")
                    if llm_info.get('n_gpu_layers', 0) > 0:
                        logger.info(f"  GPU加速: {llm_info['n_gpu_layers']} 层")
                else:
                    logger.warning("✗ 流式后处理器不可用")
                    self.postprocessor = None
            except Exception as e:
                logger.warning(f"流式后处理器初始化失败: {e}")
                self.postprocessor = None
        elif enable_llm_postprocess and not llm_model_path:
            logger.warning("✗ 启用LLM后处理但未指定模型路径 (llm_model_path)")

        # 加载模型
        logger.info("正在加载流式识别模型...")
        logger.info(f"  模型: {model}")
        logger.info(f"  设备: {device}")
        logger.info(f"  延迟: {self.chunk_size[1]*60}ms")
        logger.info("  注意: 流式模型内置VAD,不支持标点和说话人识别")

        try:
            # 构建模型参数 (参考魔搭官网最佳实践)
            model_kwargs = {
                "model": model,
                "device": device,
                "model_hub": model_hub,
                "disable_update": True,
            }

            if device == "cpu":
                model_kwargs["ncpu"] = ncpu

            # 加载模型 (流式模型内置VAD,不需要额外配置)
            self.asr_model = funasr.AutoModel(**model_kwargs)

            logger.info("✓ 模型加载成功")

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
            - timestamp: 时间戳信息
            - is_speech: 是否为语音段
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

            # 构建generate参数
            generate_kwargs = {
                "cache": cache,
                "is_final": is_final,
                "chunk_size": self.chunk_size,
                "encoder_chunk_look_back": self.encoder_chunk_look_back,
                "decoder_chunk_look_back": self.decoder_chunk_look_back,
            }
            generate_kwargs.update(kwargs)

            # 执行流式识别
            result = self.asr_model.generate(input=audio_chunk, **generate_kwargs)

            # 格式化结果
            if result:
                formatted = self._format_result(result, is_final=is_final)

                # 协同后处理：ASR听清 + LLM说人话
                if self.postprocessor:
                    original_text = formatted.get("text", "")
                    if original_text.strip():
                        try:
                            # 提取时间戳信息
                            timestamp_info = None
                            if formatted.get("timestamp"):
                                ts = formatted["timestamp"]
                                if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                                    timestamp_info = {
                                        'start': float(ts[0]) if ts[0] is not None else 0.0,
                                        'end': float(ts[1]) if ts[1] is not None else 0.0
                                    }

                            # 协同处理：ASR提供原始识别，LLM优化为人话
                            post_result = self.postprocessor.process_chunk(
                                text=original_text,
                                is_final=is_final,
                                timestamp_info=timestamp_info
                            )

                            # 如果后处理器返回了结果（缓冲区已刷新）
                            if post_result['should_output']:
                                formatted["text_asr_raw"] = post_result['original']  # ASR听清的结果
                                formatted["text"] = post_result['optimized']  # LLM说人话的结果
                                formatted["llm_optimized"] = True
                                formatted["optimization_method"] = post_result['method']
                                formatted["optimization_success"] = post_result['success']
                                formatted["quality_score"] = post_result['quality_score']

                                # 添加词级时间戳（可选）
                                if post_result.get('word_timestamps'):
                                    formatted["word_timestamps"] = post_result['word_timestamps']

                                logger.info(
                                    f"协同优化: ASR听清[{post_result['original'][:20]}...] -> "
                                    f"LLM说人话[{post_result['optimized'][:20]}...] "
                                    f"(方法={post_result['method']}, 质量={post_result['quality_score']:.2f})"
                                )
                            else:
                                # 还在缓冲中，暂不输出优化结果
                                formatted["llm_optimized"] = False
                                formatted["buffering"] = True
                                logger.debug(f"chunk已缓冲(等待句子边界): {original_text[:30]}...")

                        except Exception as e:
                            logger.warning(f"协同后处理失败: {e}")
                            formatted["llm_optimized"] = False
                    else:
                        formatted["llm_optimized"] = False
                else:
                    formatted["llm_optimized"] = False

                if formatted.get("text"):
                    logger.info(
                        f"识别结果: {formatted['text']} (final={is_final})"
                    )
                return formatted
            else:
                # 空结果(正常情况,可能是静音段)
                return {
                    "status": "success",
                    "text": "",
                    "is_final": is_final,
                    "timestamp": None,
                    "llm_optimized": False,
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
            item = result[0]
            if isinstance(item, dict):
                formatted["text"] = item.get("text", "")
                formatted["timestamp"] = item.get("timestamp")
                formatted["is_speech"] = bool(formatted["text"])
            elif isinstance(item, str):
                formatted["text"] = item
                formatted["is_speech"] = bool(item.strip())
        elif isinstance(result, dict):
            formatted["text"] = result.get("text", "")
            formatted["timestamp"] = result.get("timestamp")
            formatted["is_speech"] = bool(formatted["text"])
        elif isinstance(result, str):
            formatted["text"] = result
            formatted["is_speech"] = bool(result.strip())

        return formatted

    def reset_postprocessor(self):
        """重置后处理器状态

        在开始新的识别会话时调用，清空上下文缓存。
        适用于多用户场景或会话切换。
        """
        if self.postprocessor:
            self.postprocessor.reset()
            logger.info("流式后处理器已重置")

    def get_postprocessor_stats(self) -> Dict[str, Any]:
        """获取后处理器统计信息

        返回:
            统计信息字典
        """
        stats = {}

        if self.postprocessor:
            stats = self.postprocessor.get_stats()
            stats['enabled'] = True
        else:
            stats['enabled'] = False

        return stats

    def close(self):
        """关闭转录器并释放资源"""
        try:
            if self.postprocessor and hasattr(self.postprocessor, 'close'):
                self.postprocessor.close()
                logger.info("后处理器资源已释放")
        except Exception:
            pass

    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            self.close()
        except Exception:
            pass
