#!/usr/bin/env python3
"""FunASR 实时流式语音识别客户端 v0.3.0

集成麦克风录音、实时显示和输入法模式的流式识别客户端

功能:
- 实时麦克风录音: 16kHz 采样率，支持自定义音频设备
- 流式识别: 使用 Paraformer-Streaming 模型，低延迟输出（600ms）
- 两种工作模式:
  * 显示模式: 将识别结果显示在终端（默认）
  * 输入法模式: 将识别结果作为键盘输入发送到焦点窗口
- 输入法跨平台支持:
  * Linux: 优先使用 xdotool（推荐，更好的中文支持）
  * 通用: pynput（跨平台兼容）

使用场景:
- 语音转文字记录（显示模式）
- 实时会议记录（显示模式）
- 语音输入到任何应用程序（输入法模式）
- 语音撰写文档（输入法模式）

版本: 0.3.0
更新日期: 2025-12-05
"""

import asyncio
import json
import sys
import signal
import numpy as np
import websockets
from typing import Optional
import shutil
import subprocess

try:
    import pyaudio
except ImportError:
    print("错误: 需要安装 pyaudio")
    print("安装方法:")
    print(
        "  Ubuntu/Debian: sudo apt-get install portaudio19-dev && pip install pyaudio"
    )
    print("  macOS: brew install portaudio && pip install pyaudio")
    print("  Windows: pip install pyaudio")
    sys.exit(1)

# 检查输入法模式依赖
try:
    from pynput.keyboard import Controller, Key

    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

# 检查是否有 xdotool (Linux 环境推荐)
XDOTOOL_AVAILABLE = shutil.which("xdotool") is not None


class UnifiedRealtimeClient:
    """统一实时语音识别客户端"""

    def __init__(
        self,
        ws_url: str = "ws://localhost:8000/ws/realtime",
        sample_rate: int = 16000,
        chunk_size_ms: int = 600,
        input_mode: bool = False,
        show_status: bool = True,
        use_xdotool: bool = None,
        output_all: bool = False,
    ):
        """初始化客户端

        参数:
            ws_url: WebSocket 服务器地址
            sample_rate: 采样率
            chunk_size_ms: 音频块大小（毫秒）
            input_mode: 是否启用输入法模式（将识别结果作为键盘输入）
            show_status: 是否显示状态信息
            use_xdotool: 是否使用 xdotool（None=自动选择，仅输入法模式）
            output_all: 是否输出所有结果（包括中间结果）
        """
        self.ws_url = ws_url
        self.sample_rate = sample_rate
        self.chunk_size_ms = chunk_size_ms
        self.chunk_size = int(sample_rate * chunk_size_ms / 1000)
        self.input_mode = input_mode
        self.show_status = show_status
        self.output_all = output_all

        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.audio = None
        self.stream = None
        self.running = False
        self.results = []  # 仅显示模式使用

        # 输入法模式配置
        if self.input_mode:
            # 确定使用哪种输入方法
            if use_xdotool is None:
                # 自动选择: Linux 优先使用 xdotool
                self.use_xdotool = XDOTOOL_AVAILABLE
            else:
                self.use_xdotool = use_xdotool and XDOTOOL_AVAILABLE

            if self.use_xdotool:
                self.log("使用 xdotool 进行键盘输入")
                self.keyboard = None
            elif PYNPUT_AVAILABLE:
                self.log("使用 pynput 进行键盘输入")
                self.keyboard = Controller()
            else:
                print("错误: 输入法模式需要以下任意一个工具:")
                print("  - Linux: sudo apt-get install xdotool")
                print("  - 通用: pip install pynput")
                sys.exit(1)
        else:
            self.use_xdotool = False
            self.keyboard = None

        # 用于优雅退出
        self.loop = None

    def log(self, message: str):
        """条件性日志输出"""
        if self.show_status:
            print(message)

    def type_text(self, text: str):
        """模拟键盘输入文字（仅输入法模式）"""
        if not self.input_mode or not text or not text.strip():
            return

        try:
            self.log(f"[准备输入] {text[:50]}...")

            if self.use_xdotool:
                success = self._type_with_xdotool(text)
                if success:
                    self.log(f"[输入成功] {text}")
                else:
                    self.log(f"[输入失败] {text}")
            else:
                # pynput 方式
                import time

                time.sleep(0.05)
                self.keyboard.type(text)
                self.log(f"[输入完成] {text}")

        except Exception as e:
            self.log(f"[输入异常] {e}")
            if self.show_status:
                import traceback

                traceback.print_exc()

    def _type_with_xdotool(self, text: str) -> bool:
        """使用 xdotool 输入文本"""
        try:
            result = subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "10", text],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return True
        except Exception as e:
            self.log(f"[xdotool 失败] {e}")
            return False

    async def connect(self):
        """连接到 WebSocket 服务器"""
        self.log(f"连接到 WebSocket 服务器: {self.ws_url}")
        self.websocket = await websockets.connect(self.ws_url)

        # 接收连接确认消息
        message = await self.websocket.recv()
        data = json.loads(message)
        if data.get("type") == "connected":
            self.log(f"✓ {data['message']}")
            self.log(f"  Session ID: {data.get('session_id')}")

    async def start_recognition(self):
        """发送开始识别命令"""
        await self.websocket.send(json.dumps({"type": "start"}))
        self.log("✓ 已发送 start 命令")

    async def stop_recognition(self):
        """发送停止识别命令"""
        if self.websocket and self.websocket.state.name == "OPEN":
            await self.websocket.send(json.dumps({"type": "stop"}))
            self.log("\n✓ 已发送 stop 命令")

    async def send_audio_chunk(self, audio_data: bytes):
        """发送音频数据块"""
        if self.websocket and self.websocket.state.name == "OPEN":
            await self.websocket.send(audio_data)

    async def receive_results(self):
        """接收识别结果"""
        try:
            while self.running:
                if not self.websocket or self.websocket.state.name != "OPEN":
                    break

                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                    data = json.loads(message)

                    if data.get("type") == "result":
                        text = data.get("text", "")
                        is_final = data.get("is_final", False)
                        chunk_num = data.get("chunk_number", 0)

                        # 只处理有文本内容的结果
                        if text and text.strip():
                            if self.input_mode:
                                # 输入法模式：作为键盘输入
                                if self.output_all or is_final:
                                    if is_final:
                                        self.log(f"[{chunk_num}] ✓ {text}")
                                    else:
                                        self.log(f"[{chunk_num}] ... {text}")
                                    self.type_text(text)
                                else:
                                    self.log(f"[{chunk_num}] 跳过中间结果")
                            else:
                                # 显示模式：在终端显示
                                if is_final:
                                    print(f"\n[{chunk_num}] ✓ {text}")
                                    self.results.append(text)
                                else:
                                    print(f"\n[{chunk_num}] ... {text}")

                    elif data.get("type") == "started":
                        self.log(f"✓ {data['message']}")

                    elif data.get("type") == "stopped":
                        self.log(
                            f"\n✓ 识别结束，共处理 {data.get('total_chunks', 0)} 个音频块"
                        )
                        break

                    elif data.get("type") == "error":
                        self.log(f"\n✗ 错误: {data['message']}")
                        break

                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    self.log("\n✗ WebSocket 连接已关闭")
                    break

        except Exception as e:
            if self.running:
                self.log(f"\n✗ 接收结果时出错: {e}")

    def list_audio_devices(self):
        """列出所有可用的音频设备"""
        p = pyaudio.PyAudio()
        print("\n可用的音频输入设备:")
        print("-" * 60)

        default_device = p.get_default_input_device_info()
        default_index = default_device["index"]

        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                is_default = " (默认)" if i == default_index else ""
                print(f"[{i}] {info['name']}{is_default}")
                print(f"    采样率: {int(info['defaultSampleRate'])} Hz")
                print(f"    输入声道: {info['maxInputChannels']}")

        p.terminate()
        print("-" * 60)

    def init_audio(self, device_index: Optional[int] = None):
        """初始化音频设备"""
        self.audio = pyaudio.PyAudio()

        # 获取设备信息
        if device_index is not None:
            device_info = self.audio.get_device_info_by_index(device_index)
        else:
            device_info = self.audio.get_default_input_device_info()
            device_index = device_info["index"]

        self.log(f"\n使用音频设备: {device_info['name']}")
        self.log(f"采样率: {self.sample_rate} Hz")
        self.log(f"块大小: {self.chunk_size} 采样点 ({self.chunk_size_ms}ms)")

        # 打开音频流
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=None,
        )

    async def stream_audio(self):
        """从麦克风读取并发送音频数据"""
        if self.show_status:
            print("\n🎤 开始录音... (按 Ctrl+C 停止)")
            print("=" * 60)
        else:
            self.log("🎤 开始录音...")

        try:
            chunk_count = 0
            sent_count = 0

            while self.running:
                # 读取音频数据
                audio_data = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.stream.read,
                    self.chunk_size,
                    False,  # exception_on_overflow
                )

                # 转换为 numpy 数组
                audio_array = np.frombuffer(audio_data, dtype=np.int16)

                chunk_count += 1

                # 直接发送音频，由服务器端模型的内置VAD和增强模块处理
                await self.send_audio_chunk(audio_array.tobytes())
                sent_count += 1

                # 定期显示状态
                if self.show_status and sent_count % 10 == 0:
                    print(f"\n🎙️ 录音中... 已发送: {sent_count} 块")

        except Exception as e:
            if self.running:
                self.log(f"\n✗ 读取音频时出错: {e}")

    async def run(self, device_index: Optional[int] = None):
        """运行实时识别"""
        self.running = True

        try:
            # 初始化音频
            self.init_audio(device_index)

            # 连接 WebSocket
            await self.connect()

            # 开始识别
            await self.start_recognition()

            # 创建音频流和结果接收任务
            audio_task = asyncio.create_task(self.stream_audio())
            result_task = asyncio.create_task(self.receive_results())

            # 等待任务完成
            await asyncio.gather(audio_task, result_task, return_exceptions=True)

        except KeyboardInterrupt:
            self.log("\n\n⚠️  检测到 Ctrl+C，正在停止...")

        except Exception as e:
            self.log(f"\n✗ 运行时错误: {e}")
            import traceback

            traceback.print_exc()

        finally:
            await self.cleanup()

    async def cleanup(self):
        """清理资源"""
        self.log("\n清理资源...")
        self.running = False

        # 停止识别
        try:
            if self.websocket and self.websocket.state.name == "OPEN":
                await self.stop_recognition()
                await asyncio.sleep(0.5)
        except Exception as e:
            self.log(f"停止识别时出错: {e}")

        # 关闭音频流
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                self.log(f"关闭音频流时出错: {e}")

        # 终止 PyAudio
        if self.audio:
            try:
                self.audio.terminate()
            except Exception as e:
                self.log(f"终止 PyAudio 时出错: {e}")

        # 关闭 WebSocket
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                self.log(f"关闭 WebSocket 时出错: {e}")

        # 显示模式：打印完整结果
        if not self.input_mode and self.results and self.show_status:
            print("\n" + "=" * 60)
            print("完整识别结果:")
            print("=" * 60)
            print("".join(self.results))
            print("=" * 60)

    def get_full_text(self) -> str:
        """获取完整识别文本（仅显示模式）"""
        return "".join(self.results)


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="FunASR 实时流式语音识别客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 显示模式（默认）- 在终端显示识别结果
  python client_realtime.py

  # 输入法模式 - 将识别结果作为键盘输入
  python client_realtime.py --input-mode

  # 输入法模式 + 显示状态 + 输出所有结果（推荐）
  python client_realtime.py --input-mode --show-status --output-all

  # 列出所有音频设备
  python client_realtime.py --list-devices

  # 指定音频设备
  python client_realtime.py --device 1

  # 自定义服务器地址
  python client_realtime.py --server ws://192.168.1.100:8000/ws/realtime

  # 调整音频块大小（降低延迟）
  python client_realtime.py --chunk-size 480

  # 输入法模式使用 pynput（Linux 默认使用 xdotool）
  python client_realtime.py --input-mode --use-pynput

工作模式:
  显示模式（默认）:
    - 将识别结果显示在终端
    - 适合语音转文字记录、会议记录等

  输入法模式（--input-mode）:
    - 将识别结果作为键盘输入发送到焦点窗口
    - 适合语音输入到文本编辑器、聊天软件等
    - Linux 推荐安装 xdotool: sudo apt-get install xdotool

提示:
  - 按 Ctrl+C 停止录音
  - 建议在安静环境中使用
  - 输入法模式需要确保目标应用的输入框已获得焦点
  - 使用 --output-all 可以输出所有识别结果（包括中间结果）
        """,
    )

    parser.add_argument(
        "--server",
        default="ws://localhost:8000/ws/realtime",
        help="WebSocket 服务器地址 (默认: ws://localhost:8000/ws/realtime)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="音频输入设备索引 (默认使用系统默认设备)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="列出所有可用的音频输入设备",
    )
    parser.add_argument(
        "--input-mode",
        action="store_true",
        help="启用输入法模式（将识别结果作为键盘输入）",
    )
    parser.add_argument(
        "--no-status",
        action="store_true",
        help="禁用状态信息显示（静默模式）",
    )
    parser.add_argument(
        "--output-all",
        action="store_true",
        help="输出所有识别结果（包括中间结果）",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="采样率 (默认: 16000 Hz)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=600,
        help="音频块大小(毫秒) (默认: 600ms)",
    )
    parser.add_argument(
        "--use-pynput",
        action="store_true",
        help="强制使用 pynput 而不是 xdotool (仅输入法模式)",
    )

    args = parser.parse_args()

    # 创建客户端
    client = UnifiedRealtimeClient(
        ws_url=args.server,
        sample_rate=args.sample_rate,
        chunk_size_ms=args.chunk_size,
        input_mode=args.input_mode,
        show_status=not args.no_status,
        use_xdotool=not args.use_pynput if XDOTOOL_AVAILABLE else False,
        output_all=args.output_all,
    )

    # 列出设备
    if args.list_devices:
        client.list_audio_devices()
        return 0

    # 设置信号处理
    def signal_handler(sig, frame):
        if client.show_status:
            print("\n\n⚠️  收到停止信号...")
        client.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 显示启动信息
    if client.show_status:
        print("\n" + "=" * 60)
        print("FunASR 实时流式语音识别客户端")
        print("=" * 60)
        print(f"工作模式: {'输入法模式' if args.input_mode else '显示模式'}")
        if args.input_mode:
            print(f"输入方式: {'xdotool' if client.use_xdotool else 'pynput'}")
            print("提示: 切换到目标应用并点击输入框获得焦点")
        else:
            print("提示: 识别结果将显示在终端")
        print("按 Ctrl+C 停止")
        print("=" * 60)

    # 运行识别
    try:
        await client.run(device_index=args.device)
        return 0

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n\n程序已退出")
        sys.exit(0)
