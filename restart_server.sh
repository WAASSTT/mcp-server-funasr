#!/bin/bash

# 重启 FunASR MCP 服务器

echo "正在停止旧的服务器进程..."
pkill -f "python main.py"
sleep 2

# 启动 FunASR MCP 服务器
# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo "❌ 错误: 虚拟环境不存在"
    echo "请先运行: uv venv && uv pip install -e ."
    exit 1
fi

# 激活虚拟环境
source .venv/bin/activate

# 创建模型目录
if [ ! -d "Model" ]; then
    echo "📁 创建模型缓存目录..."
    mkdir -p Model
fi

echo "✅ 环境检查完成"
echo "💡 提示: 首次运行会自动从ModelScope下载模型(约1.1GB)"
echo ""
echo "🌐 启动服务器..."
echo "   MCP 服务器: http://localhost:8000"
echo "   MCP 端点: http://localhost:8000/mcp"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""

# 启动服务器
python main.py
