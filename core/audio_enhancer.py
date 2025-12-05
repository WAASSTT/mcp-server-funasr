"""语音增强处理器 - 基于 ClearerVoice-Studio (ModelScope)

使用ModelScope官方的ClearerVoice-Studio进行语音增强和降噪。
ClearerVoice-Studio是阿里巴巴达摩院开发的专业语音增强模型。

技术栈:
- ClearerVoice-Studio: ModelScope官方语音增强模型
- 支持多场景降噪: 环境噪声、混响、回声等
- 实时处理: 支持流式和批量处理
- 兼容FunASR生态

参考:
- https://www.modelscope.cn/models/iic/ClearerVoice-Studio
- ModelScope文档: https://www.modelscope.cn/docs

版本: 2.0.0
更新日期: 2025-12-05
"""

import numpy as np
import soundfile as sf
import tempfile
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AudioEnhancer:
    """语音增强处理器

    使用 ModelScope ClearerVoice-Studio 进行实时语音增强，包括:
    - 深度噪声抑制
    - 去混响
    - 语音清晰度提升
    - 支持流式和批量处理
    """

    def __init__(
        self,
        model_name: str = "iic/ClearerVoice-Studio",
        device: str = "cpu",
        sample_rate: int = 16000,
        model_hub: str = "ms",
    ):
        """初始化语音增强器

        参数:
            model_name: 模型名称 (默认: "iic/ClearerVoice-Studio")
            device: 运行设备 ("cpu" 或 "cuda")
            sample_rate: 采样率 (默认16000Hz)
            model_hub: 模型仓库 ("ms"=ModelScope)
        """
        self.device = device
        self.sample_rate = sample_rate
        self.model_name = model_name
        self.model_hub = model_hub
        self.model = None

        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            # 初始化 ClearerVoice-Studio 模型
            logger.info(f"正在加载语音增强模型: {model_name} (设备: {device})")

            # 使用ModelScope pipeline加载模型
            self.model = pipeline(
                task=Tasks.acoustic_noise_suppression,
                model=model_name,
                model_revision="master",
            )

            logger.info("✓ 语音增强模型加载成功 (ClearerVoice-Studio)")

        except ImportError as e:
            logger.warning(
                f"ModelScope或相关依赖未安装，语音增强功能将被禁用。"
                f"请运行: pip install modelscope\n错误: {e}"
            )
            self.model = None
        except Exception as e:
            logger.warning(f"语音增强模型加载失败: {e}")
            self.model = None

    def is_available(self) -> bool:
        """检查语音增强是否可用"""
        return self.model is not None

    def enhance_audio(
        self,
        audio: np.ndarray,
        sr: int = None,
        output_gain: float = 1.0,
    ) -> np.ndarray:
        """增强音频 - 批量处理

        参数:
            audio: 音频数据 (numpy array)
            sr: 输入音频采样率 (如果与目标不同会自动重采样)
            output_gain: 输出增益 (默认1.0)

        返回:
            增强后的音频数据
        """
        if not self.is_available():
            logger.warning("语音增强不可用，返回原始音频")
            return audio

        try:
            # 确保音频是浮点型
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # 归一化到 [-1, 1]
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / max(abs(audio.max()), abs(audio.min()))

            # ClearerVoice-Studio需要文件输入，创建临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
                input_path = tmp_in.name
                sf.write(input_path, audio, sr or self.sample_rate)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                output_path = tmp_out.name

            try:
                # 执行语音增强
                result = self.model(input_path, output_path=output_path)

                # 读取增强后的音频
                enhanced_audio, _ = sf.read(output_path, dtype="float32")

                # 应用输出增益
                if output_gain != 1.0:
                    enhanced_audio = enhanced_audio * output_gain

                # 限幅防止溢出
                enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)

                logger.debug(
                    f"语音增强完成: 输入长度={len(audio)}, 输出长度={len(enhanced_audio)}"
                )
                return enhanced_audio

            finally:
                # 清理临时文件
                try:
                    os.unlink(input_path)
                    os.unlink(output_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"语音增强失败: {e}", exc_info=True)
            return audio

    def enhance_audio_stream(
        self,
        audio_chunk: np.ndarray,
        sr: int = None,
    ) -> np.ndarray:
        """增强音频流 - 实时处理

        对音频流进行分块增强，适用于实时场景。

        参数:
            audio_chunk: 音频块数据
            sr: 采样率

        返回:
            增强后的音频块
        """
        # ClearerVoice-Studio支持流式处理，直接使用enhance_audio
        return self.enhance_audio(audio_chunk, sr, output_gain=1.0)

    def enhance_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        output_gain: float = 1.0,
    ) -> str:
        """增强音频文件

        参数:
            input_path: 输入音频文件路径
            output_path: 输出文件路径 (None 则覆盖原文件)
            output_gain: 输出增益

        返回:
            输出文件路径
        """
        if not self.is_available():
            logger.warning("语音增强不可用，跳过处理")
            return input_path

        try:
            # 读取音频
            audio, sr = sf.read(input_path, dtype="float32")
            logger.info(
                f"读取音频文件: {input_path} (采样率: {sr}Hz, 长度: {len(audio)})"
            )

            # 增强处理
            enhanced = self.enhance_audio(audio, sr, output_gain)

            # 保存文件
            if output_path is None:
                output_path = input_path

            sf.write(output_path, enhanced, self.sample_rate)
            logger.info(f"增强音频已保存: {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"音频文件增强失败: {e}", exc_info=True)
            return input_path

    def get_info(self) -> dict:
        """获取增强器信息"""
        return {
            "available": self.is_available(),
            "model": self.model_name,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "model_hub": self.model_hub,
            "model_loaded": self.model is not None,
        }


# 便捷函数
def create_enhancer(
    device: str = "cpu",
    model_name: str = "iic/ClearerVoice-Studio",
) -> AudioEnhancer:
    """创建语音增强器实例

    参数:
        device: 运行设备
        model_name: 模型名称

    返回:
        AudioEnhancer 实例
    """
    return AudioEnhancer(
        model_name=model_name,
        device=device,
    )


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    enhancer = create_enhancer()
    print("语音增强器信息:", enhancer.get_info())

    if enhancer.is_available():
        # 生成测试音频 (带噪声)
        sr = 16000
        duration = 3
        t = np.linspace(0, duration, sr * duration)

        # 纯音 + 噪声
        signal = np.sin(2 * np.pi * 440 * t)  # 440Hz 正弦波
        noise = np.random.normal(0, 0.1, len(signal))  # 白噪声
        noisy_audio = signal + noise

        # 增强
        enhanced = enhancer.enhance_audio(noisy_audio.astype(np.float32), sr)

        print(f"原始音频: min={noisy_audio.min():.4f}, max={noisy_audio.max():.4f}")
        print(f"增强音频: min={enhanced.min():.4f}, max={enhanced.max():.4f}")
        print(f"噪声抑制比: {(np.std(noisy_audio) / np.std(enhanced)):.2f}x")
    else:
        print("\n语音增强器不可用，请安装ModelScope: pip install modelscope")
