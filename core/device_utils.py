"""设备检测工具模块

统一的GPU/CPU自动检测工具，供所有模块使用。

版本: 3.0.0
更新日期: 2025-12-22
"""

import logging
import subprocess

logger = logging.getLogger(__name__)


def detect_device() -> str:
    """自动检测可用的计算设备

    返回:
        "cuda" 如果有GPU可用
        "cpu" 如果没有GPU
    """
    try:
        # 尝试导入torch检测CUDA
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"检测到 CUDA GPU: {gpu_name} ({gpu_mem_gb:.1f}GB)")
            return "cuda"
        else:
            logger.info("未检测到 CUDA GPU (torch可用但无GPU)")
            return "cpu"
    except ImportError:
        # torch未安装，尝试nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_info = result.stdout.strip().split(',')
                logger.info(f"检测到 NVIDIA GPU: {gpu_info[0].strip()}")
                logger.warning("推荐安装torch以获得更好的GPU支持: pip install torch")
                return "cuda"
        except:
            pass

        logger.info("未检测到 GPU，使用 CPU 模式")
        return "cpu"


def detect_gpu_layers(model_size_gb: float = 7.0) -> int:
    """为LLM模型检测推荐的GPU层数

    参数:
        model_size_gb: 模型大小（GB），用于估算

    返回:
        推荐的GPU层数 (0=纯CPU)
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0

        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # 根据显存和模型大小推荐层数
        if model_size_gb <= 2:  # 1.8B模型
            if gpu_mem_gb >= 6:
                return 35  # 全GPU
            elif gpu_mem_gb >= 4:
                return 24
            else:
                return 0
        elif model_size_gb <= 5:  # 7B模型
            if gpu_mem_gb >= 12:
                return 35  # 全GPU
            elif gpu_mem_gb >= 8:
                return 24
            elif gpu_mem_gb >= 6:
                return 16
            else:
                return 8
        else:  # 更大模型
            if gpu_mem_gb >= 24:
                return 35
            elif gpu_mem_gb >= 12:
                return 24
            else:
                return 0
    except ImportError:
        pass

    return 0


def get_device_info() -> dict:
    """获取详细的设备信息

    返回:
        设备信息字典
    """
    device = detect_device()
    info = {
        "device": device,
        "gpu_available": device == "cuda",
    }

    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                info["torch_version"] = torch.__version__
                info["cuda_version"] = torch.version.cuda
        except:
            pass

    return info
