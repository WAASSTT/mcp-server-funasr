"""批量语音识别模块 v0.3.0

参考FunASR和ModelScope官方最佳实践，实现非流式批量语音识别，支持VAD分段和批量处理。

功能特性:
- 使用 Paraformer-large 模型进行高精度离线识别
- 集成 FSMN-VAD 进行智能语音分段
- 支持 CT-Transformer 标点符号恢复 (可选)
- 支持 CAM++ 说话人分离 (可选)
- 支持热词定制 (hotword参数)
- 动态批处理优化性能 (batch_size_s=300推荐)
- 线程安全设计

模型配置:
- ASR: paraformer-zh (Paraformer-large)
- VAD: fsmn-vad (FSMN-VAD)
- PUNC: ct-punc-c (CT-Transformer, 可选)
- SPK: cam++ (CAM++, 可选)

参考文档:
- https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README_zh.md
- https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
- https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/paraformer/

版本: 0.3.0
更新日期: 2025-12-05
"""

import funasr
import soundfile as sf
import os
import logging
import tempfile
import numpy as np
from typing import Optional, Dict, Any, List, TYPE_CHECKING, Type

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 类型检查时导入
if TYPE_CHECKING:
    from .audio_enhancer import AudioEnhancer

# 尝试导入语音增强模块
AudioEnhancerType: Optional[Type["AudioEnhancer"]] = None
try:
    from .audio_enhancer import AudioEnhancer as AudioEnhancerImpl

    ENHANCER_AVAILABLE = True
    AudioEnhancerType = AudioEnhancerImpl
except ImportError:
    logger.warning("语音增强模块不可用，将跳过增强功能")
    ENHANCER_AVAILABLE = False


class BatchTranscriber:
    """批量语音识别器类

    参考FunASR官方最佳实践实现:
    1. 使用AutoModel统一接口
    2. 支持动态批处理(batch_size_s)
    3. 集成VAD进行智能分段
    4. 支持热词、ITN等高级功能
    """

    def __init__(
        self,
        model: str = "paraformer-zh",
        vad_model: str = "fsmn-vad",
        punc_model: Optional[str] = None,
        spk_model: Optional[str] = None,
        device: str = "cpu",
        ncpu: int = 4,
        vad_kwargs: Optional[Dict[str, Any]] = None,
        batch_size_s: int = 300,
        model_hub: str = "ms",
        enable_enhancement: bool = False,
        **kwargs,
    ):
        """初始化批量语音识别器

        参数:
            model: ASR批量模型 (默认: "paraformer-zh")
            vad_model: VAD模型 (默认: "fsmn-vad")
            punc_model: 标点恢复模型 (可选: "ct-punc-c")
            spk_model: 说话人识别模型 (可选: "cam++")
            device: 运行设备 ("cpu" 或 "cuda:0")
            ncpu: CPU线程数
            vad_kwargs: VAD参数，如 {"max_single_segment_time": 30000}
            batch_size_s: 动态批处理每批总时长(秒, 推荐300)
            model_hub: 模型仓库 ("ms"=ModelScope, "hf"=HuggingFace)
            enable_enhancement: 是否启用语音增强 (DNS-Challenge技术)
            **kwargs: 其他参数 (如hotword热词)
        """
        self.model = model
        self.vad_model = vad_model
        self.punc_model = punc_model
        self.spk_model = spk_model
        self.device = device
        self.ncpu = ncpu
        self.model_hub = model_hub
        self.batch_size_s = batch_size_s
        self.enable_enhancement = enable_enhancement

        # VAD参数
        self.vad_kwargs = {"max_single_segment_time": 30000}
        if vad_kwargs:
            self.vad_kwargs.update(vad_kwargs)

        # 热词支持 (参考魔搭官网示例)
        self.hotword = kwargs.pop("hotword", None)

        # 其他配置参数
        self.kwargs = kwargs

        # 初始化语音增强器
        self.enhancer: Optional["AudioEnhancer"] = None
        if enable_enhancement and ENHANCER_AVAILABLE and AudioEnhancerType is not None:
            try:
                logger.info("正在初始化语音增强器 (ClearerVoice-Studio)...")
                self.enhancer = AudioEnhancerType(
                    device=device,
                    sample_rate=16000,
                    model_hub="ms",
                )
                if self.enhancer.is_available():
                    logger.info("✓ 语音增强器已启用")
                else:
                    logger.warning("✗ 语音增强器不可用")
                    self.enhancer = None
            except Exception as e:
                logger.warning(f"语音增强器初始化失败: {e}")
                self.enhancer = None

        # 加载模型
        logger.info("正在加载批量识别模型...")
        logger.info(f"  模型: {model}")
        logger.info(f"  VAD: {vad_model}")
        if punc_model:
            logger.info(f"  标点: {punc_model}")
        if spk_model:
            logger.info(f"  说话人: {spk_model}")
        logger.info(f"  设备: {device}")
        logger.info(f"  Batch: {batch_size_s}s")

        try:
            # 构建模型参数 (参考魔搭官网最佳实践)
            model_kwargs = {
                "model": model,
                "vad_model": vad_model,
                "vad_kwargs": self.vad_kwargs,
                "device": device,
                "model_hub": model_hub,
                "disable_update": True,
            }

            if device == "cpu":
                model_kwargs["ncpu"] = ncpu
            if punc_model:
                model_kwargs["punc_model"] = punc_model
            if spk_model:
                model_kwargs["spk_model"] = spk_model

            # 加载模型
            self.asr_model = funasr.AutoModel(**model_kwargs)
            logger.info("✓ 模型加载成功")
        except Exception as e:
            logger.error(f"✗ 模型加载失败: {e}")
            raise

    def transcribe(
        self, audio_path: str, language: str = "auto", **kwargs
    ) -> Dict[str, Any]:
        """对音频文件进行批量识别

        使用VAD进行语音分段,然后对所有语音段进行批量识别。
        参考FunASR generate方法的标准用法。

        参数:
            audio_path: 音频文件路径、URL或wav.scp文件
            language: 语言代码 ("auto", "zh", "en", "yue", "ja", "ko")
            **kwargs: 额外的generate参数,会覆盖默认配置

        返回:
            识别结果字典:
            - status: "success" 或 "error"
            - text: 完整识别文本
            - results: FunASR原始结果列表,每个元素包含:
                - text: 识别文本
                - punc_text: 带标点的文本
                - timestamp: 时间戳 [[start, end], ...]
                - spk: 说话人ID (如启用说话人模型)
                - emotion: 情感标签 (如启用情感模型)
            - audio_path: 音频文件路径
            - audio_info: 音频文件信息
            - enhanced: 是否使用了语音增强
        """
        # 验证音频文件
        if not os.path.exists(audio_path):
            return {
                "status": "error",
                "message": f"音频文件不存在: {audio_path}",
                "text": "",
                "results": [],
            }

        # 初始化变量
        processing_path = audio_path
        enhanced = False

        try:
            # 获取音频信息
            audio_info = sf.info(audio_path)
            logger.info(f"处理音频: {audio_path} ({audio_info.duration:.2f}秒)")

            # 语音增强预处理
            if self.enhancer and self.enhancer.is_available():
                try:
                    logger.info("正在进行语音增强 (DNS-Challenge技术)...")
                    # 读取音频
                    audio_data, sr = sf.read(audio_path, dtype="float32")

                    # 增强处理
                    enhanced_audio = self.enhancer.enhance_audio(audio_data, sr)

                    # 保存到临时文件
                    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    sf.write(temp_file.name, enhanced_audio, self.enhancer.sample_rate)
                    processing_path = temp_file.name
                    enhanced = True
                    logger.info("✓ 语音增强完成")
                except Exception as e:
                    logger.warning(f"语音增强失败，使用原始音频: {e}")
                    processing_path = audio_path
                    enhanced = False

            # 构建参数
            generate_kwargs = {
                "language": language,
                "batch_size_s": self.batch_size_s,
                "use_itn": True,
                "merge_vad": True,
                "merge_length_s": 15,
            }

            # 添加热词支持 (参考魔搭官网: hotword='魔搭')
            if self.hotword:
                generate_kwargs["hotword"] = self.hotword
            if "hotword" in kwargs:
                generate_kwargs["hotword"] = kwargs.pop("hotword")

            generate_kwargs.update(kwargs)

            # 执行识别
            if "hotword" in generate_kwargs:
                logger.info(
                    f"开始识别: {processing_path} (热词: {generate_kwargs['hotword']})"
                )
            else:
                logger.info(f"开始识别: {processing_path}")
            result = self.asr_model.generate(input=processing_path, **generate_kwargs)

            # 清理临时文件
            if enhanced and processing_path != audio_path:
                try:
                    os.unlink(processing_path)
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {e}")

            # 格式化结果
            formatted_result = self._format_result(result)
            formatted_result.update(
                {
                    "status": "success",
                    "audio_path": audio_path,
                    "enhanced": enhanced,
                    "audio_info": {
                        "duration": audio_info.duration,
                        "sample_rate": audio_info.samplerate,
                        "channels": audio_info.channels,
                        "format": audio_info.format,
                    },
                }
            )

            logger.info(f"识别完成,文本长度: {len(formatted_result.get('text', ''))}")
            return formatted_result

        except Exception as e:
            logger.error(f"批量识别错误: {e}", exc_info=True)
            # 清理临时文件
            if "processing_path" in locals() and processing_path != audio_path:
                try:
                    os.unlink(processing_path)
                except:
                    pass
            return {
                "status": "error",
                "message": str(e),
                "text": "",
                "results": [],
                "audio_path": audio_path,
            }

    def transcribe_with_vad_segments(
        self, audio_path: str, return_vad_segments: bool = True, **kwargs
    ) -> Dict[str, Any]:
        """对音频文件进行识别并返回VAD分段信息

        参数:
            audio_path: 音频文件路径
            return_vad_segments: 是否返回VAD分段的时间戳
            **kwargs: 传递给模型的额外参数

        返回:
            识别结果字典，包含VAD分段信息
        """
        result = self.transcribe(audio_path, **kwargs)

        if result["status"] == "success" and return_vad_segments:
            # VAD分段信息通常在results中的timestamp字段
            vad_segments = []
            for item in result.get("results", []):
                if isinstance(item, dict) and "timestamp" in item:
                    vad_segments.append(item["timestamp"])

            result["vad_segments"] = vad_segments

        return result

    def _format_result(self, result: Any) -> Dict[str, Any]:
        """格式化识别结果

        按照FunASR官方推荐格式返回原始结果，保持完整信息不做额外转换

        FunASR官方返回格式说明:
        - 列表格式: 每个元素是一个句子的识别结果字典
        - 字典包含: text(识别文本), timestamp(时间戳), 以及其他模型特有字段
        - 支持的额外字段: punc_text(标点文本), spk(说话人), emotion(情感)等

        参数:
            result: FunASR generate方法返回的原始结果

        返回:
            包含text和结果列表的字典
        """
        # 按照官方推荐，直接使用FunASR原始返回格式
        if isinstance(result, list):
            # 提取完整文本用于快速访问
            text_parts = []
            for item in result:
                if isinstance(item, dict):
                    # 优先使用punc_text（带标点），否则使用text
                    text_parts.append(item.get("punc_text") or item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)

            return {"text": "".join(text_parts), "results": result}  # 官方推荐字段名

        elif isinstance(result, dict):
            # 单个结果
            text = result.get("punc_text") or result.get("text", "")
            return {"text": text, "results": [result]}

        elif isinstance(result, str):
            # 纯文本结果
            return {"text": result, "results": [{"text": result}]}

        return {"text": "", "results": []}

    def validate_audio(self, audio_path: str) -> Dict[str, Any]:
        """验证音频文件

        参数:
            audio_path: 音频文件路径

        返回:
            验证结果字典
        """
        if not os.path.exists(audio_path):
            return {"status": "invalid", "message": f"文件不存在: {audio_path}"}

        if not os.access(audio_path, os.R_OK):
            return {"status": "invalid", "message": f"文件不可读: {audio_path}"}

        try:
            audio_info = sf.info(audio_path)
            duration = audio_info.duration

            return {
                "status": "valid",
                "message": "音频文件有效",
                "details": {
                    "duration": duration,
                    "sample_rate": audio_info.samplerate,
                    "channels": audio_info.channels,
                    "format": audio_info.format,
                    "subtype": audio_info.subtype,
                },
            }
        except Exception as e:
            return {"status": "invalid", "message": f"音频文件无效: {e}"}
