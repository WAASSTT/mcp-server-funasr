#!/usr/bin/env python3
"""FunASR 流式语音识别客户端

使用 PyAudio 从麦克风获取音频流，通过 WebSocket 进行实时流式识别
"""

import asyncio
import json
import sys
import signal
import numpy as np
import websockets
from typing import Optional

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


class RealtimeMicrophoneClient:
    """实时麦克风识别客户端"""

    def __init__(
        self,
        ws_url: str = "ws://localhost:8000/ws/realtime",
        sample_rate: int = 16000,
        chunk_size_ms: int = 600,
    ):
        self.ws_url = ws_url
        self.sample_rate = sample_rate
        self.chunk_size_ms = chunk_size_ms
        self.chunk_size = int(sample_rate * chunk_size_ms / 1000)

        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.audio = None
        self.stream = None
        self.running = False
        self.results = []

        # 用于优雅退出
        self.loop = None

    async def connect(self):
        """连接到 WebSocket 服务器"""
        print(f"连接到 WebSocket 服务器: {self.ws_url}")
        self.websocket = await websockets.connect(self.ws_url)

        # 接收连接确认消息
        message = await self.websocket.recv()
        data = json.loads(message)
        if data.get("type") == "connected":
            print(f"✓ {data['message']}")
            print(f"  Session ID: {data.get('session_id')}")

    async def start_recognition(self):
        """发送开始识别命令"""
        await self.websocket.send(json.dumps({"type": "start"}))
        print("✓ 已发送 start 命令")

    async def stop_recognition(self):
        """发送停止识别命令"""
        if self.websocket and self.websocket.state.name == "OPEN":
            await self.websocket.send(json.dumps({"type": "stop"}))
            print("\n✓ 已发送 stop 命令")

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

                        # 只显示有文本内容的结果
                        if text and text.strip():
                            if is_final:
                                print(f"\n[{chunk_num}] ✓ {text}")
                                self.results.append(text)
                            else:
                                # 中间结果也换行显示
                                print(f"\n[{chunk_num}] ... {text}")

                    elif data.get("type") == "started":
                        print(f"✓ {data['message']}")

                    elif data.get("type") == "stopped":
                        print(
                            f"\n✓ 识别结束，共处理 {data.get('total_chunks', 0)} 个音频块"
                        )
                        break

                    elif data.get("type") == "error":
                        print(f"\n✗ 错误: {data['message']}")
                        break

                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("\n✗ WebSocket 连接已关闭")
                    break

        except Exception as e:
            if self.running:
                print(f"\n✗ 接收结果时出错: {e}")

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

        print(f"\n使用音频设备: {device_info['name']}")
        print(f"采样率: {self.sample_rate} Hz")
        print(f"块大小: {self.chunk_size} 采样点 ({self.chunk_size_ms}ms)")

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
        print("\n🎤 开始录音... (按 Ctrl+C 停止)")
        print("=" * 60)

        try:
            chunk_count = 0
            sent_count = 0
            silence_threshold = 100  # 静音阈值，低于此值认为是静音

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

                # 检测是否为静音（空录音）
                max_amplitude = np.max(np.abs(audio_array))
                rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))

                chunk_count += 1

                # 只发送非静音的音频块
                if max_amplitude > silence_threshold or rms > 50:
                    # 发送到服务器
                    await self.send_audio_chunk(audio_array.tobytes())
                    sent_count += 1

                    # 每5个有效块换行显示状态
                    if sent_count % 5 == 0:
                        print(
                            f"\n🎙️ 录音中... 已发送: {sent_count} 块 (音量: {int(rms)})"
                        )
                else:
                    # 静音块，不发送，覆盖显示
                    if chunk_count % 20 == 0:
                        print(
                            f"\r🔇 静音中... (总块数: {chunk_count}, 已发送: {sent_count})",
                            end="",
                            flush=True,
                        )

        except Exception as e:
            if self.running:
                print(f"\n✗ 读取音频时出错: {e}")

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
            print("\n\n⚠️  检测到 Ctrl+C，正在停止...")

        except Exception as e:
            print(f"\n✗ 运行时错误: {e}")
            import traceback

            traceback.print_exc()

        finally:
            await self.cleanup()

    async def cleanup(self):
        """清理资源"""
        print("\n清理资源...")
        self.running = False

        # 停止识别
        try:
            if self.websocket and self.websocket.state.name == "OPEN":
                await self.stop_recognition()
                await asyncio.sleep(0.5)  # 等待服务器处理
        except Exception as e:
            print(f"停止识别时出错: {e}")

        # 关闭音频流
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"关闭音频流时出错: {e}")

        # 终止 PyAudio
        if self.audio:
            try:
                self.audio.terminate()
            except Exception as e:
                print(f"终止 PyAudio 时出错: {e}")

        # 关闭 WebSocket
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                print(f"关闭 WebSocket 时出错: {e}")

        # 打印完整结果
        if self.results:
            print("\n" + "=" * 60)
            print("完整识别结果:")
            print("=" * 60)
            print("".join(self.results))
            print("=" * 60)

    def get_full_text(self) -> str:
        """获取完整识别文本"""
        return "".join(self.results)


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="FunASR 实时麦克风语音识别客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认麦克风
  python client_microphone.py

  # 列出所有音频设备
  python client_microphone.py --list-devices

  # 指定音频设备
  python client_microphone.py --device 1

  # 自定义服务器地址
  python client_microphone.py --server ws://192.168.1.100:8000/ws/realtime

  # 调整音频块大小（降低延迟）
  python client_microphone.py --chunk-size 480

提示:
  - 按 Ctrl+C 停止录音
  - 建议在安静环境中使用
  - 清晰地对着麦克风说话
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
        "--list-devices", action="store_true", help="列出所有可用的音频输入设备"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="采样率 (默认: 16000 Hz)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=600, help="音频块大小(毫秒) (默认: 600ms)"
    )

    args = parser.parse_args()

    # 创建客户端
    client = RealtimeMicrophoneClient(
        ws_url=args.server, sample_rate=args.sample_rate, chunk_size_ms=args.chunk_size
    )

    # 列出设备
    if args.list_devices:
        client.list_audio_devices()
        return 0

    # 设置信号处理
    def signal_handler(sig, frame):
        print("\n\n⚠️  收到停止信号...")
        client.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

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
