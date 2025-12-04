#!/bin/bash
# 快速启动脚本

set -e

echo "=================================="
echo "MCP-Server-FunASR 快速启动"
echo "=================================="

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查Python版本
echo -e "\n${YELLOW}[1/5] 检查Python版本...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo -e "${GREEN}✓ Python版本: $python_version${NC}"
else
    echo -e "${RED}✗ Python版本过低: $python_version (需要 >= $required_version)${NC}"
    exit 1
fi

# 检查并安装uv
echo -e "\n${YELLOW}[2/5] 检查uv包管理器...${NC}"
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}uv未安装,正在安装...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo -e "${GREEN}✓ uv安装完成${NC}"
else
    echo -e "${GREEN}✓ uv已安装: $(uv --version)${NC}"
fi

# 创建虚拟环境
echo -e "\n${YELLOW}[3/5] 创建虚拟环境...${NC}"
if [ ! -d ".venv" ]; then
    uv venv
    echo -e "${GREEN}✓ 虚拟环境创建成功${NC}"
else
    echo -e "${GREEN}✓ 虚拟环境已存在${NC}"
fi

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
echo -e "\n${YELLOW}[4/5] 安装依赖...${NC}"
uv pip install -r requirements.txt
echo -e "${GREEN}✓ 依赖安装完成${NC}"

# 创建模型目录
echo -e "\n${YELLOW}[5/5] 检查模型目录...${NC}"
if [ ! -d "Model" ]; then
    echo -e "${YELLOW}创建模型缓存目录...${NC}"
    mkdir -p Model
    echo -e "${GREEN}✓ 模型目录创建完成${NC}"
else
    echo -e "${GREEN}✓ 模型目录已存在${NC}"
fi

# 检查是否有已下载的模型
model_count=$(find Model -name "model.pt" 2>/dev/null | wc -l)
if [ "$model_count" -eq 0 ]; then
    echo -e "${YELLOW}💡 提示: 未检测到已下载的模型${NC}"
    echo -e "${YELLOW}   首次运行时会自动从ModelScope下载(约1.1GB)${NC}"
    echo ""
    read -p "是否现在预下载模型? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python download_models.py
    fi
else
    echo -e "${GREEN}✓ 检测到 $model_count 个已下载的模型${NC}"
fi

# 启动服务器
echo -e "\n${GREEN}=================================="
echo "环境准备完成!"
echo "==================================${NC}"
echo ""
echo "启动服务器命令:"
echo -e "  ${YELLOW}uvicorn main:app --host 0.0.0.0 --port 8000${NC}"
echo ""
echo "或直接运行:"
echo -e "  ${YELLOW}python main.py${NC}"
echo ""
read -p "是否现在启动服务器? (Y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo -e "\n${GREEN}正在启动服务器...${NC}\n"
    python main.py
fi
