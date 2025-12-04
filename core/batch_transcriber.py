"""批量语音识别模块

参考FunASR最佳实践,实现非流式批量语音识别,支持VAD分段和批量处理。

参考文档:
- https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README_zh.md
- https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/sense_voice/README.md
"""

import funasr
import soundfile as sf
import os
import logging
from typing import Optional, Dict, Any, List

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        asr_model_path: str = "paraformer-zh",
        vad_model_path: str = "fsmn-vad",
        device: str = "cpu",
        ncpu: int = 4,
        vad_kwargs: Optional[Dict[str, Any]] = None,
        asr_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """初始化批量语音识别器

        参数:
            asr_model_path: ASR模型简称或ModelScope ID
                           默认: paraformer-zh (官方批量识别模型)
                           ModelScope: damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
                           支持VAD分段、时间戳输出、热词等功能
            vad_model_path: VAD模型简称或ModelScope ID
                           默认: fsmn-vad
                           ModelScope: damo/speech_fsmn_vad_zh-cn-16k-common-pytorch
            device: 运行设备 ("cpu", "cuda:0", "cuda:1", etc.)
            ncpu: CPU线程数 (仅在device="cpu"时有效)
            vad_kwargs: VAD模型参数,如 {"max_single_segment_time": 30000}
            asr_kwargs: ASR推理参数,如 {"batch_size_s": 60, "use_itn": True}
        """
        self.device = device
        self.ncpu = ncpu
        self.asr_model_path = asr_model_path
        self.vad_model_path = vad_model_path

        # VAD参数配置 (参考FunASR文档)
        self.vad_kwargs = {"max_single_segment_time": 30000}  # VAD最大分段时长(ms)
        if vad_kwargs:
            self.vad_kwargs.update(vad_kwargs)

        # ASR推理默认参数 (参考FunASR最佳实践)
        self.asr_kwargs = {
            "batch_size_s": 60,  # 动态批处理:每批总时长(秒)
            "use_itn": True,  # 使用逆文本归一化
            "merge_vad": True,  # 合并短VAD片段
            "merge_length_s": 15,  # VAD片段合并长度(秒)
        }
        if asr_kwargs:
            self.asr_kwargs.update(asr_kwargs)

        # 加载模型
        logger.info("正在加载批量识别模型...")
        logger.info(f"  ASR模型: {asr_model_path}")
        logger.info(f"  VAD模型: {vad_model_path}")
        logger.info(f"  设备: {device}")

        try:
            # 参考FunASR文档的AutoModel初始化方式
            self.model = funasr.AutoModel(
                model=asr_model_path,
                vad_model=vad_model_path,
                vad_kwargs=self.vad_kwargs,
                device=device,
                ncpu=ncpu if device == "cpu" else None,
                disable_update=True,  # 禁用版本检查
            )
            logger.info("✓ 批量识别模型加载成功")
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
            - segments: 分段识别结果列表(含时间戳)
            - audio_path: 音频文件路径
            - audio_info: 音频文件信息
        """
        # 验证音频文件
        if not os.path.exists(audio_path):
            return {
                "status": "error",
                "message": f"音频文件不存在: {audio_path}",
                "text": "",
                "segments": [],
            }

        try:
            # 获取音频信息
            audio_info = sf.info(audio_path)
            logger.info(f"处理音频: {audio_path} ({audio_info.duration:.2f}秒)")

            # 合并参数 (kwargs优先级更高)
            generate_kwargs = self.asr_kwargs.copy()
            generate_kwargs["language"] = language
            generate_kwargs["cache"] = {}  # 批量识别使用空cache
            generate_kwargs.update(kwargs)

            # 执行识别 (参考FunASR文档的标准调用方式)
            logger.info(f"开始识别,参数: {generate_kwargs}")
            result = self.model.generate(input=audio_path, **generate_kwargs)

            # 格式化结果
            formatted_result = self._format_result(result)
            formatted_result.update(
                {
                    "status": "success",
                    "audio_path": audio_path,
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
            return {
                "status": "error",
                "message": str(e),
                "text": "",
                "segments": [],
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
            # VAD分段信息通常在segments中的timestamp字段
            vad_segments = []
            for segment in result.get("segments", []):
                if "timestamp" in segment:
                    vad_segments.append(segment["timestamp"])

            result["vad_segments"] = vad_segments

        return result

    def _format_result(self, result: Any) -> Dict[str, Any]:
        """格式化识别结果

        参考FunASR返回格式,支持多种结果类型:
        - SenseVoice: 返回带language, emotion, event等信息
        - Paraformer: 返回带timestamp的分段结果

        参数:
            result: FunASR generate方法返回的原始结果

        返回:
            统一格式的结果字典
        """
        formatted = {"text": "", "segments": []}

        # 处理列表类型结果 (最常见)
        if isinstance(result, list):
            full_text_parts = []
            for item in result:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    full_text_parts.append(text)

                    # 构建分段信息
                    segment_info = {"text": text}

                    # 添加时间戳 (Paraformer格式: [[start, end], ...])
                    if "timestamp" in item:
                        segment_info["timestamp"] = item["timestamp"]

                    # 添加SenseVoice特有信息
                    for key in ["language", "emotion", "event"]:
                        if key in item:
                            segment_info[key] = item[key]

                    formatted["segments"].append(segment_info)
                elif isinstance(item, str):
                    # 简单文本结果
                    full_text_parts.append(item)
                    formatted["segments"].append({"text": item})

            formatted["text"] = "".join(full_text_parts)

        # 处理字典类型结果
        elif isinstance(result, dict):
            formatted["text"] = result.get("text", "")
            formatted["segments"] = [result]

        # 处理字符串类型结果
        elif isinstance(result, str):
            formatted["text"] = result
            formatted["segments"] = [{"text": result}]

        return formatted

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
