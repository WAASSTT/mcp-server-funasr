#!/usr/bin/env python3
"""预下载 FunASR 和 LLM 模型 v4.0.0

支持的模型:
【语音识别模型 - 必需】
1. paraformer-zh-streaming - 实时流式识别模型 (Paraformer-online)
2. paraformer-zh - 批量语音识别模型 (Paraformer-large)
3. fsmn-vad - 语音活动检测模型

【增强模型 - 批量识别必需】
4. ct-punc - 标点符号恢复模型 (CT-Transformer, 批量识别用)
5. cam++ - 说话人分离模型 (CAM++, 批量识别用)

【LLM 后处理模型 - GGUF 格式 - 实时识别用】
6. Qwen2.5-7B-Instruct-GGUF - 流式 LLM 后处理（推荐，~4.5GB）
7. Qwen2.5-1.8B-Instruct-GGUF - 轻量级 LLM（低配推荐，~1.5GB）

版本: 4.0.0
更新日期: 2025-12-23
"""

import os
import sys

os.environ["MODELSCOPE_CACHE"] = "./Model"

try:
    from funasr import AutoModel
    from modelscope.hub.api import HubApi
except ImportError:
    print("❌ 错误: 缺少必要的依赖")
    print("请先安装依赖: uv pip install funasr modelscope")
    sys.exit(1)


def download_model(model_id: str, model_name: str, device: str = "cpu"):
    """下载ASR模型"""
    print(f"\n{'='*60}\n📦 正在下载: {model_name}\n   模型ID: {model_id}\n{'='*60}")
    try:
        model = AutoModel(model=model_id, device=device, disable_update=True, model_hub="ms")
        print(f"✅ {model_name} 下载成功!")
        return True
    except Exception as e:
        print(f"❌ {model_name} 下载失败: {e}")
        return False


def download_pipeline_model(model_id: str, model_name: str, task: str):
    """下载Pipeline模型（如语音增强）"""
    print(f"\n{'='*60}\n📦 正在下载: {model_name}\n   模型ID: {model_id}\n{'='*60}")
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        pipe = pipeline(task=getattr(Tasks, task, task), model=model_id)
        print(f"✅ {model_name} 下载成功!")
        return True
    except Exception as e:
        print(f"❌ {model_name} 下载失败: {e}")
        return False


def download_gguf_model(repo_id: str, filename: str, model_name: str):
    """下载GGUF格式的LLM模型（从HuggingFace）"""
    print(f"\n{'='*60}\n📦 正在下载: {model_name}\n   仓库: {repo_id}\n   文件: {filename}\n{'='*60}")
    try:
        from huggingface_hub import hf_hub_download
        save_dir = os.path.join("./Model/models/Qwen")
        os.makedirs(save_dir, exist_ok=True)
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=save_dir)
        print(f"✅ {model_name} 下载成功!\n   保存路径: {file_path}")
        return True
    except Exception as e:
        print(f"❌ {model_name} 下载失败: {e}\n💡 手动下载: https://huggingface.co/{repo_id}/tree/main")
        if "huggingface_hub" in str(e):
            print("   请安装: pip install huggingface-hub")
        return False


def check_existing_models():
    """检查已下载的模型"""
    print("\n" + "=" * 60 + "\n🔍 检查现有模型...\n" + "=" * 60)
    model_dir = "./Model"
    os.makedirs(model_dir, exist_ok=True)

    existing = []
    for root, dirs, files in os.walk(model_dir):
        if "model.pt" in files or "config.yaml" in files or any(".gguf" in f for f in files):
            existing.append(os.path.relpath(root, model_dir))

    if existing:
        for e in existing:
            print(f"  ✓ {e}")
    else:
        print("  未找到已下载的模型")
    return existing


def main():
    """主函数"""
    print("\n" + "=" * 60 + "\nFunASR 模型下载工具 v3.0.0\n" + "=" * 60)
    print(f"缓存目录: {os.path.abspath('./Model')}")
    check_existing_models()

    # 定义所有模型
    core_models = [
        {"id": "paraformer-zh-streaming", "name": "Paraformer-Streaming (实时)", "type": "asr"},
        {"id": "paraformer-zh", "name": "Paraformer-zh (批量)", "type": "asr"},
        {"id": "fsmn-vad", "name": "FSMN-VAD (VAD)", "type": "asr"},
    ]
    batch_models = [
        {"id": "ct-punc-c", "name": "CT-Punc (标点-批量必需)", "type": "asr"},
        {"id": "cam++", "name": "CAM++ (说话人-批量必需)", "type": "asr"},
    ]
    gguf_models = [
        {"repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF", "filename": "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", "name": "Qwen2.5-7B-Q4_K_M (part 1/2)", "type": "gguf"},
        {"repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF", "filename": "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf", "name": "Qwen2.5-7B-Q4_K_M (part 2/2)", "type": "gguf"},
        {"repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF", "filename": "qwen2.5-7b-instruct-q3_k_m.gguf", "name": "Qwen2.5-7B-Q3_K_M (轻量)", "type": "gguf"},
    ]

    print(f"\n可用模型: {len(core_models)}核心 + {len(batch_models)}批量 + {len(gguf_models)}LLM")
    print("\n请选择:")
    print("  1. 核心模型 (最小-仅实时识别)")
    print("  2. 核心+批量模型 (全功能)")
    print("  3. 核心+Qwen2.5-7B-GGUF (实时+LLM优化)")
    print("  4. 完整安装 (全部-推荐)")
    print("  5. 自定义")
    print("  6. 退出")

    choice = input("\n请输入 [1-6]: ").strip()
    models_to_download = []

    if choice == "1": models_to_download = core_models
    elif choice == "2": models_to_download = core_models + batch_models
    elif choice == "3": models_to_download = core_models + gguf_models[:2]  # part1 + part2
    elif choice == "4": models_to_download = core_models + batch_models + gguf_models[:2]
    elif choice == "5":
        models_to_download = core_models.copy()
        print("\n批量识别模型 (必选一起):")
        add_batch = input("  添加批量模型? (y/n): ").strip().lower() == 'y'
        if add_batch: models_to_download.extend(batch_models)
        print("\nLLM模型:")
        for i, m in enumerate(gguf_models): print(f"  {i+1}. {m['name']}")
        llm = input("选择LLM (序号,留空跳过): ").strip()
        if llm.isdigit() and 0 <= int(llm)-1 < len(gguf_models):
            idx = int(llm)-1
            if idx <= 1:  # 如果选part1或part2，都下载
                models_to_download.extend(gguf_models[:2])
            else:
                models_to_download.append(gguf_models[idx])
    elif choice == "6": return
    else:
        print("无效选项")
        return

    if not models_to_download:
        print("未选择模型")
        return

    # 下载
    success = 0
    print(f"\n{'='*60}\n开始下载 {len(models_to_download)} 个模型\n{'='*60}")
    for m in models_to_download:
        t = m.get("type", "asr")
        if t == "gguf":
            r = download_gguf_model(m["repo_id"], m["filename"], m["name"])
        else:
            r = download_model(m["id"], m["name"])
        if r: success += 1

    print(f"\n{'='*60}\n下载完成! ✅ {success}/{len(models_to_download)}\n{'='*60}")
    if success > 0:
        print("\n💡 提示:")
        print("  - 模型保存在: ./Model/")
        print("  - 运行服务器: python main.py")
        if any(m.get("type")=="gguf" for m in models_to_download):
            print("\n🚀 LLM配置:")
            print("  pip install llama-cpp-python")
            print("  enable_llm_postprocess=True")
            print("  llm_model_path='./Model/models/Qwen/*.gguf'")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
