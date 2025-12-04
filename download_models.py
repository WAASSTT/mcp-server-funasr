#!/usr/bin/env python
"""预下载 FunASR 模型

此脚本用于预下载所需的 FunASR 模型到本地缓存目录。
模型会保存到 ./Model/ 目录下。

支持的模型:
1. paraformer-zh - 批量语音识别模型
   ModelScope: damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
2. paraformer-zh-streaming - 流式识别模型
   ModelScope: iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online
3. fsmn-vad - VAD模型
   ModelScope: damo/speech_fsmn_vad_zh-cn-16k-common-pytorch
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
        model = AutoModel(
            model=model_id,
            device=device,
            disable_update=True,
            model_hub="ms",
        )
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

    # 定义要下载的模型
    models = [
        {
            "id": "paraformer-zh",
            "name": "Paraformer-zh (批量识别)",
        },
        {
            "id": "paraformer-zh-streaming",
            "name": "Paraformer-Streaming (流式识别)",
        },
        {
            "id": "fsmn-vad",
            "name": "FSMN-VAD (语音活动检测)",
        },
    ]

    print(f"\n将下载 {len(models)} 个模型")
    print("\n请选择操作:")
    print("  1. 下载所有模型")
    print("  2. 仅下载批量识别模型 (Paraformer-zh + VAD)")
    print("  3. 仅下载流式识别模型 (Paraformer-Streaming)")
    print("  4. 退出")

    choice = input("\n请输入选项 [1-4]: ").strip()

    success_count = 0

    if choice == "1":
        # 下载所有模型
        print("\n开始下载所有模型...")
        for model in models:
            if download_model(model["id"], model["name"]):
                success_count += 1

    elif choice == "2":
        # 仅下载批量识别模型
        print("\n开始下载批量识别模型...")
        if download_model(models[0]["id"], models[0]["name"]):
            success_count += 1
        if download_model(models[2]["id"], models[2]["name"]):
            success_count += 1

    elif choice == "3":
        # 仅下载流式识别模型
        print("\n开始下载流式识别模型...")
        if download_model(models[1]["id"], models[1]["name"]):
            success_count += 1

    elif choice == "4":
        print("\n已取消下载")
        sys.exit(0)

    else:
        print("\n❌ 无效的选项")
        sys.exit(1)

    # 总结
    print("\n" + "=" * 60)
    print("下载完成!")
    print("=" * 60)
    print(f"✅ 成功: {success_count} 个模型")

    if success_count > 0:
        print("\n💡 提示:")
        print("  - 模型已保存到 ./Model/ 目录")
        print("  - 现在可以运行: python main.py")
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
