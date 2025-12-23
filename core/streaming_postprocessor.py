"""流式后处理模块 v4.0.0

整合 LLM 后处理和高级流式处理功能，参考 WhisperX 和 NVIDIA NeMo 的最佳实践。

核心特性：
- LLM 流式优化（支持 GGUF 量化模型，CPU 友好）
- 智能句子边界缓冲（避免不自然的分割）
- 增强的滑动窗口上下文管理
- 时间戳对齐和平滑（词级精度）
- 多层 Fallback 机制（LLM → 规则 → 原文）
- 质量检查和验证

协同设计原则：
- ASR 负责"听清"：准确的语音识别
- LLM 负责"说人话"：将口语转换为通顺的书面语

推荐配置：
- CPU: Qwen2.5-7B-Instruct-GGUF (Q4_K_M)
- 轻量: Qwen2.5-1.8B-Instruct-GGUF

参考文献：
- WhisperX: https://github.com/m-bain/whisperX
- NVIDIA NeMo: https://docs.nvidia.com/nemo-framework/

版本: 4.0.0
更新日期: 2025-12-23
"""

import os
import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import re
from .device_utils import detect_gpu_layers

logger = logging.getLogger(__name__)


# ============================================================================
# 基础数据结构
# ============================================================================

@dataclass
class ChunkTimestamp:
    """Chunk 时间戳信息"""
    text: str
    start: float
    end: float
    is_speech: bool = True


# ============================================================================
# LLM 后处理器（基础类）
# ============================================================================

class StreamingLLMPostProcessor:
    """流式 LLM 后处理器

    专为实时语音识别设计，使用 GGUF 量化模型进行流式文本优化。

    核心功能：
    - 加载和管理 GGUF 模型
    - 文本优化（将口语转换为书面语）
    - 智能 Fallback（LLM 失败时使用规则处理）
    - 上下文管理
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
        enable_fallback: bool = True,
        n_threads: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
    ):
        """初始化流式 LLM 后处理器

        参数:
            model_path: GGUF 模型文件路径
            temperature: 温度参数 (0.1-0.5, 越低越稳定)
            max_tokens: 最大生成 token 数
            enable_fallback: 启用 fallback 机制
            n_threads: CPU 线程数（None=自动）
            n_gpu_layers: GPU 加速层数（None=自动检测，0=纯CPU）
        """
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_fallback = enable_fallback
        self.n_threads = n_threads or os.cpu_count() or 4

        # 自动检测GPU
        if n_gpu_layers is None:
            self.n_gpu_layers = self._detect_gpu_layers()
        else:
            self.n_gpu_layers = n_gpu_layers

        self.llama_model: Any = None
        self._init_model()

    def _detect_gpu_layers(self) -> int:
        """自动检测 GPU 可用性并返回推荐的层数"""
        # 根据模型文件名估算大小
        model_size_gb = 7.0  # 默认 7B 模型
        if "1.8b" in self.model_path.lower() or "1_8b" in self.model_path.lower():
            model_size_gb = 1.8
        elif "7b" in self.model_path.lower() or "7_b" in self.model_path.lower():
            model_size_gb = 7.0

        # 使用统一的 GPU 层数检测
        n_layers = detect_gpu_layers(model_size_gb)

        if n_layers > 0:
            logger.info(f"LLM 后处理: 使用 GPU 加速 ({n_layers}层)")
        else:
            logger.info("LLM 后处理: 使用纯 CPU 模式")

        return n_layers

    def _init_model(self):
        """初始化 GGUF 模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        try:
            from llama_cpp import Llama

            device_info = "GPU" if self.n_gpu_layers > 0 else "CPU"
            logger.info(f"正在加载 GGUF 模型: {self.model_path}")
            logger.info(f"  运行设备: {device_info}")
            logger.info(f"  CPU 线程: {self.n_threads}")
            if self.n_gpu_layers > 0:
                logger.info(f"  GPU 加速: {self.n_gpu_layers} 层")

            self.llama_model = Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )

            logger.info("✓ GGUF 模型加载成功")

        except ImportError:
            logger.error("llama-cpp-python 未安装！请运行: pip install llama-cpp-python")
            self.llama_model = None
        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            self.llama_model = None

    def is_available(self) -> bool:
        """检查 LLM 是否可用"""
        return self.llama_model is not None

    def _fallback_processing(self, text: str) -> str:
        """Fallback 文本处理 - 使用简单规则

        当 LLM 不可用或失败时使用。
        """
        if not text or not text.strip():
            return text

        result = text.strip()
        result = ' '.join(result.split())

        # 移除口语填充词
        filler_words = ['嗯', '呃', '啊', '那个', '这个', '就是说', '然后呢', '的话']
        for _ in range(2):
            for filler in filler_words:
                result = result.replace(f' {filler} ', ' ')
                result = result.replace(f'{filler} ', '')
                result = result.replace(f' {filler}', '')
                if result.startswith(filler):
                    result = result[len(filler):].strip()

        result = ' '.join(result.split())

        if not result:
            return text

        # 添加标点
        punctuation = ['。', '！', '？', '.', '!', '?', '…']
        if not any(result.endswith(p) for p in punctuation):
            question_words = ['吗', '呢', '什么', '怎么', '为什么', '哪里', '谁', '几']
            if any(word in result for word in question_words):
                result += '？'
            else:
                result += '。'

        return result

    def _build_prompt(self, text: str, context: Optional[str] = None) -> str:
        """构建优化提示词

        设计原则：ASR 负责"听清"，LLM 负责"说人话"
        - ASR 已经提供了准确的语音识别结果
        - LLM 的任务是将口语转换为通顺的书面语
        """
        context_part = f"\n\n【上下文】{context}" if context else ""

        prompt = f"""你是专业的语音转文本后处理助手。

【任务】将 ASR 识别的口语化文本转换为通顺的书面语，让机器"说人话"。

【处理规则】
1. 添加合适的标点符号（句号、逗号、问号、感叹号等）
2. 移除口语填充词（嗯、呃、啊、那个、这个等）
3. 保持原意，不添加、删减或修改实质内容
4. 保留专有名词和关键信息
5. 如果是疑问语气，用问号结尾
6. 输出简洁、通顺的书面语{context_part}

【ASR 原始识别】{text}

【优化后文本】"""

        return prompt

    def _parse_llm_output(self, raw_output: str) -> str:
        """解析 LLM 输出，提取优化后的文本

        处理情况：
        1. 如果输出中包含【优化后文本】标记，提取标记之后的内容
        2. 如果输出中包含【任务】等prompt内容，移除这些部分
        3. 如果以上都不包含，返回原始输出
        """
        # 移除可能的 prompt 泄露内容
        markers_to_remove = [
            "【任务】", "【处理规则】", "【上下文】", "【ASR 原始识别】"
        ]

        # 先检查是否有【优化后文本】标记（这是期望的情况）
        if "【优化后文本】" in raw_output:
            parts = raw_output.split("【优化后文本】", 1)
            if len(parts) > 1:
                return parts[1].strip()

        # 移除所有可能泄露的 prompt 标记
        cleaned = raw_output
        for marker in markers_to_remove:
            if marker in cleaned:
                # 如果发现 prompt 标记，截取到该标记之前的内容
                cleaned = cleaned.split(marker)[0]

        return cleaned.strip()

    def optimize_text(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """优化文本（非流式）

        参数:
            text: 待优化的文本
            context: 上下文信息

        返回:
            优化结果字典:
            - original: 原始文本
            - optimized: 优化后的文本
            - success: 是否成功
            - fallback_used: 是否使用了 fallback
            - error: 错误信息（如果失败）
        """
        if not text or not text.strip():
            return {
                "original": text,
                "optimized": text,
                "success": True,
            }

        if not self.is_available():
            if self.enable_fallback:
                optimized = self._fallback_processing(text)
                return {
                    "original": text,
                    "optimized": optimized,
                    "success": True,
                    "fallback_used": True,
                }
            else:
                return {
                    "original": text,
                    "optimized": text,
                    "success": False,
                    "reason": "LLM not available",
                }

        try:
            prompt = self._build_prompt(text, context)

            # 非流式生成
            output = self.llama_model(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["【ASR 原始识别】", "【任务】", "\n\n【"],
            )

            raw_output = output["choices"][0]["text"].strip()

            # 解析 LLM 输出，提取【优化后文本】之后的内容
            optimized_text = self._parse_llm_output(raw_output)

            return {
                "original": text,
                "optimized": optimized_text,
                "success": True,
            }

        except Exception as e:
            logger.error(f"优化失败: {e}")
            if self.enable_fallback:
                optimized = self._fallback_processing(text)
                return {
                    "original": text,
                    "optimized": optimized,
                    "success": True,
                    "fallback_used": True,
                    "error": str(e),
                }
            else:
                return {
                    "original": text,
                    "optimized": text,
                    "success": False,
                    "reason": str(e),
                }

    def get_info(self) -> dict:
        """获取处理器信息"""
        return {
            "available": self.is_available(),
            "model_path": self.model_path,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "enable_fallback": self.enable_fallback,
            "n_threads": self.n_threads,
            "n_gpu_layers": self.n_gpu_layers,
        }

    def close(self):
        """显式关闭模型并释放资源"""
        if self.llama_model is not None:
            try:
                # 尝试显式关闭模型
                if hasattr(self.llama_model, 'close'):
                    self.llama_model.close()
                logger.info("✓ LLM 模型资源已释放")
            except Exception as e:
                logger.warning(f"关闭 LLM 模型时出错: {e}")
            finally:
                self.llama_model = None

    def __del__(self):
        """析构函数，确保资源被释放"""
        try:
            self.close()
        except:
            pass  # 静默失败，避免在析构时报错


# ============================================================================
# 高级流式处理组件
# ============================================================================

class EnhancedContextWindow:
    """增强的上下文窗口管理器

    参考 WhisperX 的滑动窗口策略，结合句子边界检测。

    特性：
    - 维护固定大小的滑动窗口
    - 智能上下文截断（基于句子边界）
    - 分离原始和优化后的缓冲区
    """

    def __init__(
        self,
        window_size: int = 3,
        overlap: int = 1,
        max_context_chars: int = 200
    ):
        """初始化上下文窗口管理器

        参数:
            window_size: 窗口大小（chunk 数量）
            overlap: 重叠大小（用于上下文的 chunk 数量）
            max_context_chars: 最大上下文字符数
        """
        self.window_size = window_size
        self.overlap = overlap
        self.max_context_chars = max_context_chars

        # 存储原始 chunks 和优化后的 chunks
        self.raw_buffer: List[str] = []
        self.optimized_buffer: List[str] = []

        # 句子边界标识符
        self.sentence_endings = ['。', '！', '？', '.', '!', '?', '…']

    def add_chunk(self, raw_text: str, optimized_text: Optional[str] = None):
        """添加新的 chunk 到窗口

        参数:
            raw_text: 原始文本
            optimized_text: 优化后的文本（可选）
        """
        self.raw_buffer.append(raw_text)
        if optimized_text:
            self.optimized_buffer.append(optimized_text)

        # 保持窗口大小
        if len(self.raw_buffer) > self.window_size:
            self.raw_buffer.pop(0)
        if len(self.optimized_buffer) > self.window_size:
            self.optimized_buffer.pop(0)

    def get_context(self, include_current: bool = False) -> str:
        """获取上下文信息

        参考 WhisperX 的 sentence-level 处理策略。

        参数:
            include_current: 是否包含当前 chunk

        返回:
            上下文字符串
        """
        if not self.optimized_buffer:
            return ""

        # 获取前 overlap 个句子作为上下文
        if include_current:
            context_chunks = self.optimized_buffer[-self.overlap:]
        else:
            # 不包含当前（最后一个）
            context_chunks = self.optimized_buffer[:-1][-self.overlap:] if len(self.optimized_buffer) > 1 else []

        if not context_chunks:
            return ""

        context = ' '.join(context_chunks)

        # 限制上下文长度（从句子边界截断）
        if len(context) > self.max_context_chars:
            # 从句子边界截断
            for i in range(len(context) - self.max_context_chars, len(context)):
                if i < len(context) and context[i] in self.sentence_endings:
                    context = context[i+1:].lstrip()
                    break
            else:
                # 如果没找到句子边界，直接截断
                context = context[-self.max_context_chars:]

        return context

    def is_sentence_boundary(self, text: str) -> bool:
        """检测是否为句子边界

        参考 WhisperX 使用 NLTK 的句子分割策略。

        参数:
            text: 待检测的文本

        返回:
            是否为句子边界
        """
        if not text:
            return False
        text_stripped = text.rstrip()
        return any(text_stripped.endswith(p) for p in self.sentence_endings)

    def clear(self):
        """清空上下文窗口"""
        self.raw_buffer.clear()
        self.optimized_buffer.clear()


class SentenceBoundaryBuffer:
    """句子边界缓冲器

    参考 WhisperX 的 sentence-level 分段和 NVIDIA NeMo 的 buffering 策略。

    特性：
    - 基于停顿的智能缓冲（VAD检测到静音时输出）
    - 防止过早或过晚刷新
    - 超时强制刷新机制

    工作原理：
    - 持续累积文本，不基于标点符号判断
    - 当 is_final=True（VAD检测到停顿）时刷新缓冲区
    - 输出完整的说话段落
    """

    def __init__(
        self,
        min_buffer_size: int = 2,
        max_buffer_size: int = 5,
        force_flush_timeout: float = 5.0
    ):
        """初始化句子边界缓冲器

        参数:
            min_buffer_size: 最小缓冲区大小（chunk 数量）
            max_buffer_size: 最大缓冲区大小（chunk 数量）
            force_flush_timeout: 强制刷新超时时间（秒）
        """
        self.min_buffer_size = min_buffer_size
        self.max_buffer_size = max_buffer_size
        self.force_flush_timeout = force_flush_timeout

        self.buffer: List[str] = []
        self.last_flush_time: Optional[float] = None
        self.sentence_endings = ['。', '！', '？', '.', '!', '?', '…']

    def add_chunk(self, text: str, is_final: bool = False) -> Optional[str]:
        """添加 chunk 到缓冲区

        参数:
            text: 文本内容
            is_final: 是否为最后一个 chunk（VAD检测到停顿）

        返回:
            如果需要刷新，返回合并后的文本；否则返回 None
        """
        if not text.strip():
            return None

        self.buffer.append(text)

        # 初始化时间戳
        if self.last_flush_time is None:
            self.last_flush_time = time.time()

        current_time = time.time()
        time_since_flush = current_time - self.last_flush_time

        # 检查是否需要刷新
        should_flush = False

        # 1. 基于停顿的刷新（VAD检测到静音）- 主要判断条件
        if is_final:
            should_flush = True

        # 2. 达到最大缓冲区大小（防止无限累积）
        elif len(self.buffer) >= self.max_buffer_size:
            should_flush = True

        # 3. 超时强制刷新（防止长时间不输出）
        elif time_since_flush >= self.force_flush_timeout:
            should_flush = True

        # 如果需要刷新
        if should_flush:
            result = ' '.join(self.buffer).strip()
            self.buffer.clear()
            self.last_flush_time = current_time
            return result

        return None

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.last_flush_time = None


class TimestampSmoother:
    """时间戳平滑器

    参考 WhisperX 的词级时间戳对齐策略。

    特性：
    - 线性插值平滑
    - 词级时间戳估算
    - 累积 chunk 时间戳信息
    """

    def __init__(self):
        """初始化时间戳平滑器"""
        self.chunk_timestamps: List[ChunkTimestamp] = []

    def add_chunk_with_timestamp(
        self,
        text: str,
        start: float,
        end: float,
        is_speech: bool = True
    ):
        """添加带时间戳的 chunk

        参数:
            text: 文本内容
            start: 开始时间（秒）
            end: 结束时间（秒）
            is_speech: 是否为语音段
        """
        chunk = ChunkTimestamp(
            text=text,
            start=start,
            end=end,
            is_speech=is_speech
        )
        self.chunk_timestamps.append(chunk)

    def estimate_word_timestamps(self, optimized_text: str) -> List[Dict[str, Any]]:
        """估算词级时间戳

        参考 WhisperX 的 DTW 对齐策略（简化版：线性插值）。

        参数:
            optimized_text: 优化后的文本

        返回:
            词级时间戳列表: [{'word': str, 'start': float, 'end': float}, ...]
        """
        if not self.chunk_timestamps:
            return []

        # 分词（简单空格分割，可替换为更复杂的分词器）
        words = optimized_text.split()
        if not words:
            return []

        # 计算总时间跨度
        total_start = self.chunk_timestamps[0].start
        total_end = self.chunk_timestamps[-1].end
        total_duration = total_end - total_start

        if total_duration <= 0:
            return []

        # 简单线性插值（假设词均匀分布）
        word_timestamps = []
        for i, word in enumerate(words):
            ratio_start = i / len(words)
            ratio_end = (i + 1) / len(words)

            word_start = total_start + ratio_start * total_duration
            word_end = total_start + ratio_end * total_duration

            word_timestamps.append({
                'word': word,
                'start': round(word_start, 3),
                'end': round(word_end, 3)
            })

        return word_timestamps

    def clear(self):
        """清空时间戳缓存"""
        self.chunk_timestamps.clear()


class RobustPostProcessor:
    """鲁棒的后处理器

    参考 WhisperX 的多层 fallback 策略。

    特性：
    - LLM 优化（主要方法）
    - 规则优化（fallback）
    - 原文保留（最终 fallback）
    - 质量检查和验证
    """

    def __init__(
        self,
        llm_processor: StreamingLLMPostProcessor,
        enable_rule_fallback: bool = True,
        enable_quality_check: bool = True,
        min_retention_rate: float = 0.4
    ):
        """初始化鲁棒后处理器

        参数:
            llm_processor: LLM 处理器实例
            enable_rule_fallback: 是否启用规则 fallback
            enable_quality_check: 是否启用质量检查
            min_retention_rate: 最小词汇保留率（用于质量检查）
        """
        self.llm_processor = llm_processor
        self.enable_rule_fallback = enable_rule_fallback
        self.enable_quality_check = enable_quality_check
        self.min_retention_rate = min_retention_rate

    def process_with_fallback(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """使用多层 fallback 处理文本

        处理层级:
        1. LLM 优化（如果可用）
        2. 规则优化（如果 LLM 失败或质量不佳）
        3. 原文（如果都失败）

        参数:
            text: 待处理的文本
            context: 上下文信息（可选）

        返回:
            处理结果字典:
            - original: 原始文本
            - optimized: 优化后的文本
            - method: 使用的方法（'llm', 'rule', 'original'）
            - success: 是否成功
            - quality_score: 质量分数（0.0-1.0）
        """
        result = {
            'original': text,
            'optimized': text,
            'method': 'none',
            'success': False,
            'quality_score': 0.0
        }

        # 第一层: LLM 处理
        if self.llm_processor and self.llm_processor.is_available():
            try:
                llm_result = self.llm_processor.optimize_text(text, context)

                if llm_result.get('success'):
                    optimized = llm_result['optimized']

                    # 质量检查
                    if self.enable_quality_check:
                        quality_score = self._quality_check(text, optimized)

                        if quality_score >= 0.5:  # 质量阈值
                            result['optimized'] = optimized
                            result['method'] = 'llm'
                            result['success'] = True
                            result['quality_score'] = quality_score
                            logger.debug(f"LLM 优化成功 (质量分数: {quality_score:.2f})")
                            return result
                        else:
                            logger.warning(f"LLM 优化质量不佳 (分数: {quality_score:.2f})，使用 fallback")
                    else:
                        result['optimized'] = optimized
                        result['method'] = 'llm'
                        result['success'] = True
                        result['quality_score'] = 1.0
                        logger.debug("LLM 优化成功（未启用质量检查）")
                        return result

            except Exception as e:
                logger.warning(f"LLM 处理失败: {e}")

        # 第二层: 规则 fallback
        if self.enable_rule_fallback and self.llm_processor:
            try:
                rule_result = self.llm_processor._fallback_processing(text)
                result['optimized'] = rule_result
                result['method'] = 'rule'
                result['success'] = True
                result['quality_score'] = 0.7
                logger.debug("使用规则优化")
                return result
            except Exception as e:
                logger.warning(f"规则处理失败: {e}")

        # 第三层: 返回原文
        result['method'] = 'original'
        result['quality_score'] = 0.0
        logger.debug("保留原文（所有优化方法失败）")
        return result

    def _quality_check(self, original: str, optimized: str) -> float:
        """质量检查

        参考 WhisperX 的对齐验证策略。

        检查项:
        1. 长度合理性
        2. 内容非空
        3. 关键词保留率
        4. 语言一致性

        参数:
            original: 原始文本
            optimized: 优化后的文本

        返回:
            质量分数 (0.0 - 1.0)
        """
        score = 0.0
        max_score = 4.0

        # 1. 长度检查 - 优化后不应显著变长或过短
        if optimized.strip():
            length_ratio = len(optimized) / len(original) if len(original) > 0 else 1.0
            if 0.5 <= length_ratio <= 1.5:
                score += 1.0
            elif 0.3 <= length_ratio <= 2.0:
                score += 0.5

        # 2. 内容检查 - 优化后不应为空
        if optimized.strip():
            score += 1.0

        # 3. 关键词保留检查
        original_words = set(original.split())
        optimized_words = set(optimized.split())

        if len(original_words) > 0:
            overlap = len(original_words & optimized_words)
            retention_rate = overlap / len(original_words)

            if retention_rate >= self.min_retention_rate:
                score += 1.0
            elif retention_rate >= self.min_retention_rate * 0.7:
                score += 0.5
        else:
            score += 1.0  # 原文为空时，给满分

        # 4. 标点检查 - 优化后应该有合理的标点
        if any(optimized.endswith(p) for p in ['。', '！', '？', '.', '!', '?']):
            score += 1.0
        elif any(p in optimized for p in ['，', ',', '、']):
            score += 0.5

        return score / max_score


# ============================================================================
# 统一流式后处理器（主接口）
# ============================================================================

class StreamingPostProcessor:
    """统一流式后处理器

    整合 WhisperX 和 NVIDIA NeMo 的最佳实践。

    核心组件：
    - StreamingLLMPostProcessor: LLM 处理
    - EnhancedContextWindow: 上下文窗口管理
    - SentenceBoundaryBuffer: 句子边界缓冲
    - TimestampSmoother: 时间戳平滑
    - RobustPostProcessor: 鲁棒处理

    工作流程：
    1. 接收 chunk → 2. 缓冲判断 → 3. 提取上下文 → 4. LLM 优化 → 5. 质量检查 → 6. 输出
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
        enable_fallback: bool = True,
        n_threads: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        context_window_size: int = 3,
        min_buffer_size: int = 2,
        max_buffer_size: int = 5,
        enable_timestamp_alignment: bool = True,
        enable_quality_check: bool = True,
        min_retention_rate: float = 0.4
    ):
        """初始化统一流式后处理器

        参数:
            model_path: GGUF 模型文件路径
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            enable_fallback: 启用 fallback 机制
            n_threads: CPU 线程数
            n_gpu_layers: GPU 加速层数
            context_window_size: 上下文窗口大小
            min_buffer_size: 最小缓冲区大小
            max_buffer_size: 最大缓冲区大小
            enable_timestamp_alignment: 是否启用时间戳对齐
            enable_quality_check: 是否启用质量检查
            min_retention_rate: 最小词汇保留率
        """
        # 初始化 LLM 处理器
        self.llm_processor = StreamingLLMPostProcessor(
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_fallback=enable_fallback,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
        )

        # 上下文窗口管理器
        self.context_window = EnhancedContextWindow(
            window_size=context_window_size,
            overlap=1,
            max_context_chars=200
        )

        # 句子边界缓冲器
        self.sentence_buffer = SentenceBoundaryBuffer(
            min_buffer_size=min_buffer_size,
            max_buffer_size=max_buffer_size,
            force_flush_timeout=5.0
        )

        # 时间戳平滑器
        self.timestamp_smoother = TimestampSmoother() if enable_timestamp_alignment else None

        # 鲁棒处理器
        self.robust_processor = RobustPostProcessor(
            llm_processor=self.llm_processor,
            enable_rule_fallback=True,
            enable_quality_check=enable_quality_check,
            min_retention_rate=min_retention_rate
        )

        logger.info("流式后处理器已初始化")
        logger.info(f"  模型路径: {model_path}")
        logger.info(f"  上下文窗口: {context_window_size}")
        logger.info(f"  缓冲区大小: {min_buffer_size}-{max_buffer_size}")
        logger.info(f"  时间戳对齐: {'启用' if enable_timestamp_alignment else '禁用'}")
        logger.info(f"  质量检查: {'启用' if enable_quality_check else '禁用'}")

    def process_chunk(
        self,
        text: str,
        is_final: bool = False,
        timestamp_info: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """处理单个 chunk

        参数:
            text: 输入文本
            is_final: 是否为最后一个 chunk
            timestamp_info: 时间戳信息 {'start': float, 'end': float}

        返回:
            处理结果字典:
            - status: 'buffering' 或 'processed'
            - original: 原始文本
            - optimized: 优化后的文本（仅 status='processed' 时）
            - should_output: 是否应该输出
            - method: 优化方法（'llm', 'rule', 'original'）
            - success: 是否成功
            - quality_score: 质量分数
            - word_timestamps: 词级时间戳（可选）
        """
        result = {
            'status': 'buffering',
            'original': text,
            'optimized': None,
            'should_output': False,
            'method': 'none',
            'success': False,
            'quality_score': 0.0
        }

        if not text or not text.strip():
            return result

        # 添加时间戳信息
        if self.timestamp_smoother and timestamp_info:
            self.timestamp_smoother.add_chunk_with_timestamp(
                text,
                timestamp_info.get('start', 0.0),
                timestamp_info.get('end', 0.0),
                is_speech=True
            )

        # 添加到缓冲区
        buffered_text = self.sentence_buffer.add_chunk(text, is_final)

        # 如果缓冲区返回了文本，说明需要处理
        if buffered_text:
            logger.debug(f"缓冲区刷新: {buffered_text[:50]}...")

            # 获取上下文
            context = self.context_window.get_context(include_current=False)
            if context:
                logger.debug(f"使用上下文: {context[:50]}...")

            # 使用鲁棒处理器优化
            opt_result = self.robust_processor.process_with_fallback(
                buffered_text,
                context
            )

            # 更新上下文窗口
            self.context_window.add_chunk(buffered_text, opt_result['optimized'])

            # 生成词级时间戳
            word_timestamps = None
            if self.timestamp_smoother:
                word_timestamps = self.timestamp_smoother.estimate_word_timestamps(
                    opt_result['optimized']
                )

            # 更新结果
            result['status'] = 'processed'
            result['original'] = buffered_text
            result['optimized'] = opt_result['optimized']
            result['method'] = opt_result['method']
            result['success'] = opt_result['success']
            result['quality_score'] = opt_result.get('quality_score', 0.0)
            result['should_output'] = True
            result['word_timestamps'] = word_timestamps

            logger.info(
                f"处理完成: {buffered_text[:30]}... -> {opt_result['optimized'][:30]}... "
                f"(方法={opt_result['method']}, 质量={opt_result.get('quality_score', 0):.2f})"
            )

        return result

    def reset(self):
        """重置所有状态

        在开始新的会话时调用。
        """
        self.context_window.clear()
        self.sentence_buffer.clear()
        if self.timestamp_smoother:
            self.timestamp_smoother.clear()
        logger.info("流式后处理器状态已重置")

    def close(self):
        """关闭后处理器并释放资源"""
        if self.llm_processor:
            self.llm_processor.close()
        logger.info("流式后处理器已关闭")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息

        返回:
            统计信息字典
        """
        return {
            'llm_available': self.llm_processor.is_available(),
            'context_buffer_size': len(self.context_window.optimized_buffer),
            'sentence_buffer_size': len(self.sentence_buffer.buffer),
            'timestamp_cache_size': len(self.timestamp_smoother.chunk_timestamps) if self.timestamp_smoother else 0,
            'llm_info': self.llm_processor.get_info(),
        }

    def is_available(self) -> bool:
        """检查后处理器是否可用"""
        return self.llm_processor.is_available()

    def __del__(self):
        """析构函数，确保资源被释放"""
        try:
            self.close()
        except:
            pass  # 静默失败，避免在析构时报错
