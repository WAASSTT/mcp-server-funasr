@echo off
chcp 65001 >nul
REM 快速启动脚本

echo ==================================
echo MCP-Server-FunASR 快速启动
echo ==================================

REM 检查Python版本
echo.
echo [1/5] 检查Python版本...
python --version >nul 2>&1
if errorlevel 1 (
    echo ✗ 未找到Python,请先安装Python 3.10或更高版本
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✓ Python版本: %PYTHON_VERSION%

REM 检查并安装uv
echo.
echo [2/5] 检查uv包管理器...
uv --version >nul 2>&1
if errorlevel 1 (
    echo uv未安装,正在安装...
    powershell -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo ✗ uv安装失败,请手动安装: https://github.com/astral-sh/uv
        pause
        exit /b 1
    )
    echo ✓ uv安装完成
) else (
    for /f "tokens=*" %%i in ('uv --version') do set UV_VERSION=%%i
    echo ✓ uv已安装: %UV_VERSION%
)

REM 创建虚拟环境
echo.
echo [3/5] 创建虚拟环境...
if not exist ".venv" (
    uv venv
    if errorlevel 1 (
        echo ✗ 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo ✓ 虚拟环境创建成功
) else (
    echo ✓ 虚拟环境已存在
)

REM 激活虚拟环境
call .venv\Scripts\activate.bat

REM 安装依赖
echo.
echo [4/5] 安装依赖...
uv pip install -e .
if errorlevel 1 (
    echo ✗ 依赖安装失败
    pause
    exit /b 1
)
echo ✓ 依赖安装完成

REM 创建模型目录
echo.
echo [5/5] 检查模型目录...
if not exist "Model" (
    echo 创建模型缓存目录...
    mkdir Model
    echo ✓ 模型目录创建完成
) else (
    echo ✓ 模型目录已存在
)

REM 检查是否有已下载的模型
set MODEL_COUNT=0
for /r Model %%f in (model.pt) do set /a MODEL_COUNT+=1

if %MODEL_COUNT% EQU 0 (
    echo 💡 提示: 未检测到已下载的模型
    echo    首次运行时会自动从 ModelScope 下载：
    echo    【实时识别】
    echo    - Paraformer-Streaming: ~850MB ^(实时ASR^)
    echo    - FSMN-VAD: ~4MB ^(VAD检测^)
    echo    - Qwen2.5-7B GGUF: ~4.5GB ^(LLM后处理^)
    echo    【批量识别】
    echo    - Paraformer-large: ~950MB ^(批量ASR^)
    echo    - CT-Punc: ~283MB ^(标点恢复^)
    echo    - CAM++: ~28MB ^(说话人分离^)
    echo.
    set /p DOWNLOAD="是否现在预下载模型? (y/N) "
    if /i "%DOWNLOAD%"=="y" (
        python download_models.py
    )
) else (
    echo ✓ 检测到 %MODEL_COUNT% 个已下载的模型
)

REM 启动服务器
echo.
echo ==================================
echo 环境准备完成!
echo ==================================
echo.
echo 启动服务器命令:
echo   uvicorn main:app --host 0.0.0.0 --port 8000
echo.
echo 或直接运行:
echo   python main.py
echo.
set /p START="是否现在启动服务器? (Y/n) "
if /i not "%START%"=="n" (
    echo.
    echo 正在启动服务器...
    echo.
    python main.py
)

pause
