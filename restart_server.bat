@echo off
chcp 65001 >nul
REM 重启 FunASR MCP 服务器

echo 正在停止旧的服务器进程...
taskkill /F /FI "WINDOWTITLE eq main.py*" /T >nul 2>&1
taskkill /F /FI "IMAGENAME eq python.exe" /FI "COMMANDLINE eq *main.py*" >nul 2>&1
timeout /t 2 /nobreak >nul

REM 检查虚拟环境
if not exist ".venv" (
    echo ❌ 错误: 虚拟环境不存在
    echo 请先运行: setup.bat
    pause
    exit /b 1
)

REM 激活虚拟环境
call .venv\Scripts\activate.bat

REM 创建模型目录
if not exist "Model" (
    echo 📁 创建模型缓存目录...
    mkdir Model
)

echo ✅ 环境检查完成
echo 💡 提示: 首次运行会自动从 ModelScope 下载模型
echo    【实时识别】
echo    - Paraformer-Streaming: ~850MB
echo    - FSMN-VAD: ~4MB
echo    - Qwen2.5-7B GGUF: ~4.5GB
echo    【批量识别】
echo    - Paraformer-large: ~950MB
echo    - CT-Punc: ~283MB
echo    - CAM++: ~28MB
echo.
echo 🌐 启动服务器...
echo    MCP 服务器: http://localhost:8000
echo    MCP 端点: http://localhost:8000/mcp
echo.
echo 按 Ctrl+C 停止服务器
echo.

REM 启动服务器
python main.py

pause
