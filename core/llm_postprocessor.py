"""LLM后处理模块 - 优化语音识别结果

使用大语言模型对ASR识别结果进行后处理优化，使其更符合人类语言习惯。

功能特性:
- 修正口语化表达为书面语
- 优化标点符号和分段
- 修正语法错误和不通顺的表达
- 智能断句和语义分析
- 本地模型推理 (无需API调用)
- 流式和批量处理模式

支持模型:
- Qwen2.5-7B-Instruct (蒸馏模型，推荐)
- Qwen2.5-14B-Instruct (更高精度)
- Qwen3-235B-A22B-Instruct-2507 (最大模型)

应用场景:
- 实时会议记录
- 语音转文字输入
- 字幕生成
- 语音备忘录整理

参考:
- https://www.modelscope.cn/models/Qwen/Qwen3-235B-A22B-Instruct-2507

版本: 2.0.0
更新日期: 2025-12-05
"""

import os
import logging
from typing import Optional, Dict, Any, List, Tuple
import json

logger = logging.getLogger(__name__)


class LLMPostProcessor:
    """LLM后处理器

    使用大语言模型优化语音识别结果，使其更加自然流畅。
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        model_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """初始化LLM后处理器

        参数:
            model: 模型名称
            temperature: 温度参数 (0-1, 越低越保守)
            max_tokens: 最大token数
            model_path: 本地模型路径 (可选)
            device: 计算设备 (cuda/cpu/mps)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_path = model_path or os.getenv("LOCAL_MODEL_PATH")
        self.device = device
        self.local_model: Any = None
        self.local_tokenizer: Any = None

        # 初始化本地模型
        self._init_local_model()

    def _init_local_model(self):
        """初始化本地模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            # 确定模型路径
            if self.model_path:
                model_path = self.model_path
            else:
                # 尝试从ModelScope缓存加载
                # 提取模型名称（去除前缀）
                model_name = (
                    self.model.split("/")[-1] if "/" in self.model else self.model
                )
                model_path = os.path.join("./Model/models/Qwen", model_name)

            if not os.path.exists(model_path):
                logger.warning(
                    f"本地模型路径不存在: {model_path}，将尝试从ModelScope下载"
                )
                from modelscope import snapshot_download

                # 使用完整的模型ID下载
                model_id = self.model if "/" in self.model else f"Qwen/{self.model}"
                model_path = snapshot_download(model_id, cache_dir="./Model/models")

            logger.info(f"正在加载本地模型: {model_path}")

            # 加载tokenizer
            self.local_tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

            # 加载模型
            self.local_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )

            if self.device != "cuda":
                self.local_model = self.local_model.to(self.device)

            self.local_model.eval()

            logger.info(f"本地模型加载成功 (设备: {self.device})")

        except ImportError:
            logger.warning(
                "transformers库未安装，无法使用本地模型。请运行: pip install transformers torch"
            )
            self.local_model = None
        except Exception as e:
            logger.error(f"本地模型加载失败: {e}", exc_info=True)
            self.local_model = None

    def is_available(self) -> bool:
        """检查LLM后处理是否可用"""
        return self.local_model is not None and self.local_tokenizer is not None

    def optimize_text(
        self,
        text: str,
        context: Optional[str] = None,
        style: str = "natural",
    ) -> Dict[str, Any]:
        """优化文本

        参数:
            text: 待优化的文本
            context: 上下文信息 (用于更好的理解)
            style: 输出风格
                - "natural": 自然口语 (默认)
                - "formal": 正式书面语
                - "concise": 简洁精炼

        返回:
            优化结果字典:
            - original: 原始文本
            - optimized: 优化后的文本
            - changes: 修改说明
            - success: 是否成功
        """
        if not self.is_available():
            logger.warning("LLM后处理不可用，返回原始文本")
            return {
                "original": text,
                "optimized": text,
                "changes": [],
                "success": False,
                "reason": "LLM not available",
            }

        if not text or not text.strip():
            return {
                "original": text,
                "optimized": text,
                "changes": [],
                "success": True,
            }

        try:
            # 构建提示词
            system_prompt = self._get_system_prompt(style)
            user_prompt = self._build_user_prompt(text, context)

            logger.debug(f"正在优化文本: {text[:50]}...")

            # 使用本地模型推理
            optimized_text, tokens_used = self._generate_local(
                system_prompt, user_prompt
            )

            # 尝试解析JSON格式的响应
            try:
                result = json.loads(optimized_text)
                if isinstance(result, dict) and "text" in result:
                    optimized_text = result["text"]
                    changes = result.get("changes", [])
                else:
                    changes = []
            except json.JSONDecodeError:
                # 如果不是JSON，直接使用文本
                changes = []

            logger.info(f"文本优化完成: {text[:30]}... -> {optimized_text[:30]}...")

            return {
                "original": text,
                "optimized": optimized_text,
                "changes": changes,
                "success": True,
                "tokens_used": tokens_used,
            }

        except Exception as e:
            logger.error(f"LLM优化失败: {e}", exc_info=True)
            return {
                "original": text,
                "optimized": text,
                "changes": [],
                "success": False,
                "reason": str(e),
            }

    def optimize_streaming(
        self,
        text_chunks: List[str],
        context: Optional[str] = None,
        style: str = "natural",
    ) -> List[Dict[str, Any]]:
        """优化流式文本块

        对多个连续的文本块进行批量优化，保持上下文连贯性。

        参数:
            text_chunks: 文本块列表
            context: 上下文信息
            style: 输出风格

        返回:
            优化结果列表
        """
        results = []
        accumulated_context = context or ""

        for chunk in text_chunks:
            if not chunk.strip():
                results.append(
                    {
                        "original": chunk,
                        "optimized": chunk,
                        "success": True,
                    }
                )
                continue

            # 优化当前块
            result = self.optimize_text(
                chunk,
                context=accumulated_context,
                style=style,
            )
            results.append(result)

            # 更新上下文（使用优化后的文本）
            if result["success"]:
                accumulated_context = (
                    accumulated_context + " " + result["optimized"]
                ).strip()
                # 限制上下文长度
                if len(accumulated_context) > 500:
                    accumulated_context = accumulated_context[-500:]

        return results

    def _get_system_prompt(self, style: str) -> str:
        """获取系统提示词"""
        base_prompt = (
            "你是一个专业的文本优化助手。你的任务是优化语音识别的结果，"
            "使其更加自然流畅、符合人类语言习惯。\n\n"
            "优化要求：\n"
            "1. 修正口语化表达，但保持原意\n"
            "2. 添加或优化标点符号\n"
            "3. 修正明显的语法错误\n"
            "4. 合并重复或冗余的内容\n"
            "5. 保持原始语义和关键信息\n"
            "6. 不要添加原文中没有的信息\n"
        )

        style_prompts = {
            "natural": "输出风格：自然口语，保持对话感。",
            "formal": "输出风格：正式书面语，适合文档和报告。",
            "concise": "输出风格：简洁精炼，去除冗余表达。",
        }

        style_instruction = style_prompts.get(style, style_prompts["natural"])

        return (
            f"{base_prompt}\n{style_instruction}\n\n"
            "直接输出优化后的文本，不要解释或添加其他内容。"
        )

    def _build_user_prompt(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> str:
        """构建用户提示词"""
        if context:
            return f"上下文：{context}\n\n待优化文本：{text}\n\n请优化上述文本。"
        else:
            return f"待优化文本：{text}\n\n请优化上述文本。"

    def _generate_local(self, system_prompt: str, user_prompt: str) -> Tuple[str, int]:
        """使用本地模型生成文本

        参数:
            system_prompt: 系统提示词
            user_prompt: 用户提示词

        返回:
            (生成的文本, token数)
        """
        import torch

        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 使用tokenizer应用聊天模板
        text = self.local_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 编码输入
        model_inputs = self.local_tokenizer([text], return_tensors="pt").to(
            self.local_model.device
        )

        # 生成
        with torch.no_grad():
            generated_ids = self.local_model.generate(
                **model_inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                top_p=0.9,
                pad_token_id=self.local_tokenizer.eos_token_id,
            )

        # 解码输出
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.local_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # 统计token数（近似）
        tokens_used = len(model_inputs.input_ids[0]) + len(generated_ids[0])

        return response.strip(), tokens_used

    def get_info(self) -> dict:
        """获取后处理器信息"""
        return {
            "available": self.is_available(),
            "backend": "Local",
            "model": self.model_path or self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "device": self.device,
        }


# 便捷函数
def create_postprocessor(
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
) -> LLMPostProcessor:
    """创建LLM后处理器实例

    参数:
        model: 模型名称
        temperature: 温度参数

    返回:
        LLMPostProcessor 实例
    """
    return LLMPostProcessor(
        model=model,
        temperature=temperature,
    )


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    processor = create_postprocessor()
    print("LLM后处理器信息:", processor.get_info())

    if processor.is_available():
        # 测试优化
        test_text = "那个嗯我想说就是这个那个项目的话呢就是进展还是挺顺利的嗯然后那个我们下周应该就可以完成了吧"
        result = processor.optimize_text(test_text)

        print("\n原始文本:", result["original"])
        print("优化文本:", result["optimized"])
        print("成功:", result["success"])
    else:
        print("\nLLM后处理器不可用，请设置OPENAI_API_KEY环境变量")
