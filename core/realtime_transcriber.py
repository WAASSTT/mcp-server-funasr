"""实时流式语音识别模块 v0.3.0

参考FunASR和ModelScope官方流式识别最佳实践，实现边输入边识别的实时转录功能。

功能特性:
- 使用 Paraformer-Streaming 模型进行实时流式识别
- 支持低延迟输出 (600ms 实时粒度)
- 内置VAD语音活动检测
- 通过 cache 维护流式状态
- 支持 chunk_size 配置延迟
- 线程安全设计，支持多客户端并发

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

版本: 0.3.0
更新日期: 2025-12-05
"""

import funasr
import numpy as np
import logging
from typing import Generator, Optional, Dict, Any, List, TYPE_CHECKING, Type

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 类型检查时导入，运行时可选
if TYPE_CHECKING:
    from .audio_enhancer import AudioEnhancer
    from .llm_postprocessor import LLMPostProcessor

# 尝试导入语音增强模块
AudioEnhancerType: Optional[Type["AudioEnhancer"]] = None
try:
    from .audio_enhancer import AudioEnhancer as AudioEnhancerImpl

    ENHANCER_AVAILABLE = True
    AudioEnhancerType = AudioEnhancerImpl
except ImportError:
    logger.warning("语音增强模块不可用，将跳过增强功能")
    ENHANCER_AVAILABLE = False

# 尝试导入LLM后处理模块
LLMPostProcessorType: Optional[Type["LLMPostProcessor"]] = None
try:
    from .llm_postprocessor import LLMPostProcessor as LLMPostProcessorImpl

    LLM_AVAILABLE = True
    LLMPostProcessorType = LLMPostProcessorImpl
except ImportError:
    logger.warning("LLM后处理模块不可用，将跳过文本优化功能")
    LLM_AVAILABLE = False


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
        enable_enhancement: bool = False,
        enable_llm_postprocess: bool = False,
        llm_model: str = "Qwen3-235B-A22B-Instruct-2507",
        llm_device: str = "cuda",
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
            enable_enhancement: 是否启用实时语音增强 (DNS-Challenge技术)
            enable_llm_postprocess: 是否启用LLM后处理优化
            llm_model: LLM模型名称
            llm_device: LLM计算设备 (默认: "cuda")
            **kwargs: 其他参数

        注意: 流式模型内置VAD,不支持外部VAD/标点/说话人模型
        """
        self.model = model
        self.device = device
        self.ncpu = ncpu
        self.model_hub = model_hub
        self.enable_enhancement = enable_enhancement
        self.enable_llm_postprocess = enable_llm_postprocess

        # 流式参数配置
        self.chunk_size = chunk_size or [0, 10, 5]
        self.encoder_chunk_look_back = encoder_chunk_look_back
        self.decoder_chunk_look_back = decoder_chunk_look_back

        # 其他配置参数
        self.kwargs = kwargs

        # 初始化语音增强器
        self.enhancer: Optional["AudioEnhancer"] = None
        if enable_enhancement and ENHANCER_AVAILABLE and AudioEnhancerType is not None:
            try:
                logger.info("正在初始化实时语音增强器 (ClearerVoice-Studio)...")
                self.enhancer = AudioEnhancerType(
                    device=device,
                    sample_rate=16000,
                    model_hub="ms",
                )
                if self.enhancer.is_available():
                    logger.info("✓ 实时语音增强器已启用")
                else:
                    logger.warning("✗ 实时语音增强器不可用")
                    self.enhancer = None
            except Exception as e:
                logger.warning(f"实时语音增强器初始化失败: {e}")
                self.enhancer = None

        # 初始化LLM后处理器
        self.llm_processor: Optional["LLMPostProcessor"] = None
        if (
            enable_llm_postprocess
            and LLM_AVAILABLE
            and LLMPostProcessorType is not None
        ):
            try:
                logger.info(f"正在初始化LLM后处理器 (本地模型: {llm_model})...")
                self.llm_processor = LLMPostProcessorType(
                    model=llm_model,
                    temperature=0.3,
                    device=llm_device,
                )
                if self.llm_processor.is_available():
                    logger.info("✓ LLM后处理器已启用")
                else:
                    logger.warning("✗ LLM后处理器不可用 (需要GPU和transformers库)")
                    self.llm_processor = None
            except Exception as e:
                logger.warning(f"LLM后处理器初始化失败: {e}")
                self.llm_processor = None

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
            - enhanced: 是否使用了语音增强
        """
        try:
            # 确保音频数据格式正确
            if not isinstance(audio_chunk, np.ndarray):
                audio_chunk = np.array(audio_chunk, dtype=np.float32)
            elif audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)

            # 实时语音增强预处理
            enhanced = False
            if self.enhancer and self.enhancer.is_available():
                try:
                    audio_chunk = self.enhancer.enhance_audio_stream(
                        audio_chunk, sample_rate
                    )
                    enhanced = True
                except Exception as e:
                    logger.warning(f"实时语音增强失败: {e}")
                    enhanced = False

            # 记录处理信息
            chunk_duration = len(audio_chunk) / sample_rate * 1000
            logger.debug(
                f"处理chunk: {len(audio_chunk)}样本 ({chunk_duration:.0f}ms), "
                f"enhanced={enhanced}, is_final={is_final}"
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
                formatted["enhanced"] = enhanced

                # LLM后处理优化 (仅在is_final=True时进行)
                if (
                    is_final
                    and self.llm_processor
                    and self.llm_processor.is_available()
                ):
                    original_text = formatted.get("text", "")
                    if original_text.strip():
                        try:
                            logger.debug(f"正在进行LLM后处理: {original_text[:50]}...")
                            llm_result = self.llm_processor.optimize_text(
                                original_text,
                                style="natural",
                            )
                            if llm_result["success"]:
                                formatted["text_original"] = original_text
                                formatted["text"] = llm_result["optimized"]
                                formatted["llm_optimized"] = True
                                logger.info(
                                    f"LLM优化完成: {original_text[:30]}... -> {llm_result['optimized'][:30]}..."
                                )
                            else:
                                formatted["llm_optimized"] = False
                        except Exception as e:
                            logger.warning(f"LLM后处理失败: {e}")
                            formatted["llm_optimized"] = False
                    else:
                        formatted["llm_optimized"] = False
                else:
                    formatted["llm_optimized"] = False

                if formatted.get("text"):
                    logger.info(
                        f"识别结果: {formatted['text']} (enhanced={enhanced}, final={is_final})"
                    )
                return formatted
            else:
                # 空结果(正常情况,可能是静音段)
                return {
                    "status": "success",
                    "text": "",
                    "is_final": is_final,
                    "timestamp": None,
                    "enhanced": enhanced,
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
