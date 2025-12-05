#!/usr/bin/env python
"""预下载 FunASR 模型 v0.3.0

此脚本用于预下载所需的 FunASR 模型到本地缓存目录。
模型会保存到 ./Model/ 目录下。

支持的模型:
1. paraformer-zh - 批量语音识别模型 (Paraformer-large)
   ModelScope: iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
   功能: 高精度非流式语音识别，支持长语音处理

2. paraformer-zh-streaming - 流式识别模型 (Paraformer-online)
   ModelScope: iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online
   功能: 实时流式识别，延迟低至600ms

3. fsmn-vad - 语音活动检测模型
   ModelScope: iic/speech_fsmn_vad_zh-cn-16k-common-pytorch
   功能: 高精度语音活动检测，智能分段处理

4. ct-punc - 标点符号恢复模型 (CT-Transformer)
   ModelScope: iic/punc_ct-transformer_cn-en-common-vocab471067-large
   功能: 自动添加标点符号，支持中英文混合

5. cam++ - 说话人分离模型 (CAM++)
   ModelScope: iic/speech_campplus_sv_zh-cn_16k-common
   功能: 说话人验证和分离，支持多说话人场景

6. emotion2vec+large - 情感识别模型 (可选)
   ModelScope: iic/emotion2vec_plus_large
   功能: 语音情感分析

版本: 0.3.0
更新日期: 2025-12-04
"""

import os
import sys

# 设置模型缓存目录
os.environ["MODELSCOPE_CACHE"] = "./Model"

try:
    from funasr import AutoModel
    from modelscope.hub.api import HubApi
except ImportError:
    print("❌ 错误: 缺少必要的依赖")
    print("请先安装依赖: uv pip install funasr modelscope")
    sys.exit(1)


def download_model(model_id: str, model_name: str, device: str = "cpu"):
    """下载单个模型

    参数:
        model_id: ModelScope模型ID
        model_name: 模型显示名称
        device: 运行设备
    """
    print(f"\n{'='*60}")
    print(f"📦 正在下载: {model_name}")
    print(f"   模型ID: {model_id}")
    print(f"{'='*60}")

    try:
        # 使用 AutoModel 自动下载
        print("正在初始化模型...")
        model_kwargs = {
            "model": model_id,
            "device": device,
            "disable_update": True,
            "model_hub": "ms",
        }

        model = AutoModel(**model_kwargs)
        print(f"✅ {model_name} 下载成功!")
        return True
    except Exception as e:
        print(f"❌ {model_name} 下载失败: {e}")
        return False


def check_existing_models():
    """检查已下载的模型"""
    print("\n" + "=" * 60)
    print("🔍 检查现有模型...")
    print("=" * 60)

    model_dir = "./Model"
    if not os.path.exists(model_dir):
        print("📁 模型目录不存在，将创建...")
        os.makedirs(model_dir, exist_ok=True)
        return []

    # 检查目录内容
    existing = []
    for root, dirs, files in os.walk(model_dir):
        if "model.pt" in files or "config.yaml" in files:
            model_path = os.path.relpath(root, model_dir)
            existing.append(model_path)
            print(f"  ✓ 已存在: {model_path}")

    if not existing:
        print("  ℹ️  未找到已下载的模型")

    return existing


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("FunASR 模型下载工具")
    print("=" * 60)
    print(f"缓存目录: {os.path.abspath('./Model')}")

    # 检查现有模型
    existing_models = check_existing_models()

    # 定义核心模型（必需）
    core_models = [
        {
            "id": "paraformer-zh",
            "name": "Paraformer-zh (批量识别)",
            "required": True,
        },
        {
            "id": "paraformer-zh-streaming",
            "name": "Paraformer-Streaming (流式识别)",
            "required": True,
        },
        {
            "id": "fsmn-vad",
            "name": "FSMN-VAD (语音活动检测)",
            "required": True,
        },
    ]

    # 定义可选模型（增强功能）
    optional_models = [
        {
            "id": "ct-punc-c",
            "name": "CT-Punc-C (标点恢复)",
            "required": False,
        },
        {
            "id": "cam++",
            "name": "CAM++ (说话人分离)",
            "required": False,
        },
        {
            "id": "iic/emotion2vec_plus_large",
            "name": "Emotion2Vec+Large (情感识别)",
            "required": False,
        },
    ]

    all_models = core_models + optional_models

    print(f"\n共 {len(all_models)} 个可用模型:")
    print(f"  - 核心模型: {len(core_models)} 个 (必需)")
    print(f"  - 可选模型: {len(optional_models)} 个 (增强功能)")

    print("\n请选择操作:")
    print("  1. 仅下载核心模型 (最小安装，仅ASR+VAD)")
    print("  2. 下载核心模型 + 标点恢复")
    print("  3. 下载核心模型 + 标点恢复 + 说话人分离")
    print("  4. 下载所有模型 (包含情感识别)")
    print("  5. 自定义选择")
    print("  6. 退出")

    choice = input("\n请输入选项 [1-6]: ").strip()

    success_count = 0
    models_to_download = []

    if choice == "1":
        # 仅核心模型
        models_to_download = core_models
        print("\n将下载核心模型...")

    elif choice == "2":
        # 核心 + 标点
        models_to_download = core_models + [optional_models[0]]
        print("\n将下载核心模型 + 标点恢复...")

    elif choice == "3":
        # 核心 + 标点 + 说话人
        models_to_download = core_models + optional_models[:2]
        print("\n将下载核心模型 + 标点恢复 + 说话人分离...")

    elif choice == "4":
        # 所有模型
        models_to_download = all_models
        print("\n将下载所有模型...")

    elif choice == "5":
        # 自定义选择
        print("\n可选模型列表:")
        models_to_download = core_models.copy()
        print("  核心模型 (自动包含):")
        for i, model in enumerate(core_models):
            print(f"    {i+1}. {model['name']}")

        print("\n  可选模型 (输入序号选择，多个用空格分隔):")
        for i, model in enumerate(optional_models):
            print(f"    {i+1}. {model['name']}")

        selected = input("\n请输入要下载的可选模型序号 (直接回车跳过): ").strip()
        if selected:
            try:
                indices = [int(x.strip()) - 1 for x in selected.split()]
                for idx in indices:
                    if 0 <= idx < len(optional_models):
                        models_to_download.append(optional_models[idx])
                    else:
                        print(f"  警告: 序号 {idx+1} 无效，已跳过")
            except ValueError:
                print("  警告: 输入格式错误，仅下载核心模型")

    elif choice == "6":
        print("\n已取消下载")
        sys.exit(0)

    else:
        print("\n❌ 无效的选项")
        sys.exit(1)

    # 开始下载
    print(f"\n开始下载 {len(models_to_download)} 个模型...")
    for model in models_to_download:
        if download_model(model["id"], model["name"]):
            success_count += 1

    # 总结
    print("\n" + "=" * 60)
    print("下载完成!")
    print("=" * 60)
    print(f"✅ 成功: {success_count} 个模型")

    if success_count > 0:
        print("\n💡 提示:")
        print("  - 模型已保存到 ./Model/ 目录")
        print("  - 现在可以运行: python main.py")
        print('  - 批量识别默认启用热词 (hotword="魔搭"),可在 main.py 中修改')
        print("  - 或使用: ./restart_server.sh")

    print("")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断下载")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
