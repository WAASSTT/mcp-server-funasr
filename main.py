"""FunASR MCP服务器主程序

基于FastMCP框架提供语音识别服务,包括:
- 实时流式语音识别 (结合VAD+ASR)
- 批量语音识别 (结合VAD分段+批量ASR)
"""

import os
import tempfile
import json
import asyncio
import threading
import uuid
from datetime import datetime

# 设置模型缓存到项目目录 (必须在导入funasr之前)
os.environ["MODELSCOPE_CACHE"] = "./Model"

import soundfile as sf
import numpy as np
from fastmcp import FastMCP
from core.realtime_transcriber import RealtimeTranscriber
from core.batch_transcriber import BatchTranscriber
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.requests import Request
from starlette.websockets import WebSocket, WebSocketDisconnect

# 初始化FastMCP服务器
mcp = FastMCP(name="FunASR语音服务")

# ========== 并发控制 ==========
# 线程锁保护模型推理 (FunASR模型非线程安全)
realtime_model_lock = threading.Lock()
batch_model_lock = threading.Lock()

# 连接管理
active_connections = {}
connection_counter = 0
connection_lock = threading.Lock()

# ========== 实例化处理器 ==========

# 实时语音识别器配置 (参考FunASR流式识别最佳实践)
# 使用 paraformer-zh-streaming: FunASR官方流式识别模型
# ModelScope: iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online
# 支持真正的流式识别,延迟可控,适合实时场景
# 首次运行会自动从ModelScope下载到 ./Model/ 目录
realtime_transcriber = RealtimeTranscriber(
    asr_model_path="paraformer-zh-streaming",  # 官方流式模型简称,自动下载
    vad_model_path=None,  # 流式模型内置VAD,无需单独指定
    device="cpu",  # 生产环境改为 "cuda:0" 启用GPU
    ncpu=4,
    chunk_size=[0, 10, 5],  # 600ms延迟配置 (可选: [0,8,4]为480ms)
    encoder_chunk_look_back=4,  # 编码器回溯块数
    decoder_chunk_look_back=1,  # 解码器回溯块数
    vad_kwargs={},
    asr_kwargs={},
)

# 批量语音识别器配置 (参考FunASR批处理最佳实践)
# 使用 paraformer-zh: FunASR官方批量识别模型
# ModelScope: damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
# 支持VAD分段、标点恢复、时间戳输出等功能
# 首次运行会自动从ModelScope下载到 ./Model/ 目录
batch_transcriber = BatchTranscriber(
    asr_model_path="paraformer-zh",  # 官方批量识别模型简称,自动下载
    vad_model_path="fsmn-vad",  # 官方VAD模型简称,自动下载
    device="cpu",  # 生产环境改为 "cuda:0" 启用GPU
    ncpu=4,  # CPU线程数,生产环境可增加至8
    vad_kwargs={"max_single_segment_time": 30000},  # VAD最大分段时长(ms)
    asr_kwargs={
        "batch_size_s": 60,  # 动态批处理:每批总时长(秒)
        "use_itn": True,  # 启用逆文本归一化
        "merge_vad": True,  # 合并短VAD片段
        "merge_length_s": 15,  # VAD片段合并长度(秒)
    },
)

# ========== 注册MCP工具 ==========


# ---------- 批量语音识别工具 ----------
@mcp.tool(
    name="transcribe_audio",
    description="对音频文件进行批量语音识别，使用VAD分段后进行批量识别",
)
def transcribe_audio(audio_path: str, return_vad_segments: bool = False) -> dict:
    """批量语音识别

    使用VAD进行语音分段，然后对所有语音段进行批量识别。
    适用于完整音频文件的离线处理。

    参数:
        audio_path: 音频文件路径
        return_vad_segments: 是否返回VAD分段的时间戳信息

    返回:
        包含识别结果的字典:
        - status: "success" 或 "error"
        - text: 完整识别文本
        - segments: 分段识别结果列表
        - audio_path: 音频文件路径
        - audio_info: 音频文件信息
        - vad_segments: VAD分段信息(如果return_vad_segments=True)
    """
    # 使用锁保护模型推理 (支持并发调用)
    with batch_model_lock:
        if return_vad_segments:
            return batch_transcriber.transcribe_with_vad_segments(
                audio_path=audio_path, return_vad_segments=True
            )
        else:
            return batch_transcriber.transcribe(audio_path=audio_path)


@mcp.tool(
    name="validate_audio_file", description="验证音频文件是否适合处理并提供其属性信息"
)
def validate_audio_file(file_path: str) -> dict:
    """验证音频文件

    检查文件是否存在、可读且为有效的音频格式。

    参数:
        file_path: 音频文件路径

    返回:
        包含验证状态、消息和音频属性的字典
    """
    return batch_transcriber.validate_audio(file_path)


# ========== 配置 CORS 中间件 ==========
cors_middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有源，生产环境应限制为特定域名
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=[
            "mcp-protocol-version",
            "mcp-session-id",
            "Authorization",
            "Content-Type",
            "Accept",
            "Cache-Control",
            "X-Requested-With",
        ],
        expose_headers=[
            "mcp-session-id",
            "Content-Type",
        ],
        allow_credentials=True,
        max_age=3600,  # 预检请求缓存1小时
    )
]


# ========== 音频上传端点 ==========
async def upload_audio_endpoint(request: Request):
    """接收浏览器录制的音频并进行批量识别"""
    try:
        # 读取上传的音频数据
        audio_data = await request.body()

        if not audio_data:
            return JSONResponse(
                {"status": "error", "message": "没有接收到音频数据"}, status_code=400
            )

        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name

        try:
            # 使用批量识别器进行转录 (使用锁保护模型推理)
            with batch_model_lock:
                result = batch_transcriber.transcribe(temp_path)

            return JSONResponse({"status": "success", "result": result})

        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": f"转录失败: {str(e)}"}, status_code=500
        )


# ========== WebSocket实时语音识别端点 ==========
async def websocket_realtime_endpoint(websocket: WebSocket):
    """WebSocket端点：实时接收音频流并进行识别

    使用paraformer-zh-streaming进行流式识别:
    - chunk_size配置为[0,10,5]，即600ms实时粒度
    - 每次输入600ms音频(9600 samples @ 16kHz)
    - 通过cache维护流式状态
    - 支持多客户端并发（使用线程锁保护模型推理）
    - 添加超时保护和资源管理
    """
    await websocket.accept()

    # 为每个连接分配唯一ID
    global connection_counter
    with connection_lock:
        connection_counter += 1
        session_id = f"session_{connection_counter}_{uuid.uuid4().hex[:8]}"
        active_connections[session_id] = {
            "start_time": datetime.now(),
            "chunk_count": 0,
            "websocket": websocket,
            "last_activity": datetime.now(),
        }

    print(
        f"[{session_id}] WebSocket客户端已连接 (当前活跃连接: {len(active_connections)})"
    )

    try:
        # 发送连接成功消息
        await websocket.send_json(
            {
                "type": "connected",
                "message": "WebSocket连接成功，使用Paraformer流式模型",
                "session_id": session_id,
                "active_connections": len(active_connections),
            }
        )

        # 会话级cache (FunASR流式识别必需，每个连接独立)
        cache_asr = {}
        chunk_count = 0

        # Buffer管理 - 使用固定大小数组避免np.append性能问题
        CHUNK_SIZE_MS = 600
        chunk_size = int(CHUNK_SIZE_MS * 16000 / 1000)  # 9600 samples
        max_buffer_size = chunk_size * 3  # 最多缓冲3个chunk
        audio_buffer = np.zeros(max_buffer_size, dtype=np.float32)
        buffer_write_index = 0
        buffer_bytes = b""

        while True:
            # 接收消息
            message = await websocket.receive()

            # 处理文本消息（控制命令）
            if "text" in message:
                data = json.loads(message["text"])

                if data.get("type") == "start":
                    print(f"[{session_id}] 收到start命令，清空缓存")
                    cache_asr.clear()
                    chunk_count = 0
                    audio_buffer.fill(0)
                    buffer_write_index = 0
                    buffer_bytes = b""
                    active_connections[session_id]["chunk_count"] = 0
                    await websocket.send_json(
                        {
                            "type": "started",
                            "message": "开始识别",
                            "session_id": session_id,
                        }
                    )

                elif data.get("type") == "stop":
                    print(
                        f"[{session_id}] 收到stop命令，总共处理了 {chunk_count} 个音频块"
                    )

                    # 处理剩余的音频缓冲区
                    if buffer_write_index > 1600:  # 至少100ms的数据
                        try:
                            print(
                                f"[{session_id}] 处理剩余音频: {buffer_write_index} 样本"
                            )

                            # 使用线程锁保护模型推理
                            with realtime_model_lock:
                                result = realtime_transcriber.transcribe_chunk(
                                    audio_chunk=audio_buffer[:buffer_write_index],
                                    cache=cache_asr,
                                    is_final=True,
                                    sample_rate=16000,
                                )

                            if result and result.get("text"):
                                chunk_count += 1
                                print(f"[{session_id}] 最终识别结果: {result['text']}")
                                await websocket.send_json(
                                    {
                                        "type": "result",
                                        "text": result["text"],
                                        "is_final": True,
                                        "chunk_number": chunk_count,
                                        "session_id": session_id,
                                    }
                                )
                        except Exception as e:
                            print(f"[{session_id}] 处理剩余音频失败: {e}")
                            import traceback

                            traceback.print_exc()

                    # 清空缓存
                    cache_asr.clear()
                    audio_buffer.fill(0)
                    buffer_write_index = 0
                    buffer_bytes = b""

                    # 结束识别
                    await websocket.send_json(
                        {
                            "type": "stopped",
                            "message": "识别结束",
                            "total_chunks": chunk_count,
                        }
                    )
                    break

            # 处理二进制消息（音频数据）
            elif "bytes" in message:
                audio_bytes = message["bytes"]
                # print(f"[{session_id}] 收到音频数据: {len(audio_bytes)} 字节")  # 太频繁，注释掉

                try:
                    # 检查数据有效性
                    if len(audio_bytes) < 2:
                        continue

                    # 累积字节数据
                    buffer_bytes += audio_bytes

                    # 防止缓冲区无限增长
                    MAX_BUFFER_BYTES = 64000  # 约2秒的音频数据 @ 16kHz
                    if len(buffer_bytes) > MAX_BUFFER_BYTES:
                        print(
                            f"[{session_id}] ⚠️ 字节缓冲区过大({len(buffer_bytes)}字节)，清理旧数据"
                        )
                        buffer_bytes = buffer_bytes[-MAX_BUFFER_BYTES:]

                    if len(buffer_bytes) < 2:
                        continue

                    # 转换为float32数组（从int16）
                    num_samples = len(buffer_bytes) - (len(buffer_bytes) % 2)
                    if num_samples > 0:
                        new_samples = (
                            np.frombuffer(
                                buffer_bytes[:num_samples],
                                dtype=np.int16,
                            ).astype(np.float32)
                            / 32768.0
                        )

                        # 检查缓冲区空间 - 添加安全边界
                        if buffer_write_index + len(new_samples) > max_buffer_size:
                            # 缓冲区满，移动数据到开头
                            remaining = min(
                                buffer_write_index % chunk_size, max_buffer_size // 2
                            )
                            if remaining > 0 and buffer_write_index >= remaining:
                                audio_buffer[:remaining] = audio_buffer[
                                    buffer_write_index - remaining : buffer_write_index
                                ]
                            buffer_write_index = remaining
                            print(
                                f"[{session_id}] 缓冲区已满，重置（保留 {remaining} 样本）"
                            )

                        # 写入新数据 - 确保不越界
                        samples_to_write = min(
                            len(new_samples), max_buffer_size - buffer_write_index
                        )
                        if (
                            samples_to_write > 0
                            and buffer_write_index + samples_to_write <= max_buffer_size
                        ):
                            audio_buffer[
                                buffer_write_index : buffer_write_index
                                + samples_to_write
                            ] = new_samples[:samples_to_write]
                            buffer_write_index += samples_to_write
                        buffer_bytes = buffer_bytes[num_samples:]

                        # print(f"[{session_id}] 音频缓冲区: {buffer_write_index} 样本")  # 太频繁

                    # 当缓冲区达到600ms时进行流式识别
                    while buffer_write_index >= chunk_size:
                        chunk = audio_buffer[:chunk_size].copy()
                        # 移动剩余数据
                        remaining = buffer_write_index - chunk_size
                        if remaining > 0:
                            audio_buffer[:remaining] = audio_buffer[
                                chunk_size:buffer_write_index
                            ]
                        buffer_write_index = remaining

                        print(
                            f"[{session_id}] 处理音频块 {chunk_count + 1}: {len(chunk)} 样本 ({CHUNK_SIZE_MS}ms)"
                        )

                        try:
                            # Paraformer流式识别 (标准FunASR流式用法)
                            # 使用线程锁保护模型推理（支持多客户端并发）
                            # 添加超时保护避免长时间阻塞
                            lock_acquired = realtime_model_lock.acquire(timeout=5.0)
                            if not lock_acquired:
                                print(f"[{session_id}] ⚠️ 获取模型锁超时，跳过本次识别")
                                continue

                            try:
                                result = realtime_transcriber.transcribe_chunk(
                                    audio_chunk=chunk,
                                    cache=cache_asr,
                                    is_final=False,
                                    sample_rate=16000,
                                )
                            finally:
                                realtime_model_lock.release()

                            # print(f"[{session_id}] 识别结果: {result}")  # 调试用

                            # 更新最后活动时间
                            with connection_lock:
                                if session_id in active_connections:
                                    active_connections[session_id][
                                        "last_activity"
                                    ] = datetime.now()

                            if result and result.get("text"):
                                chunk_count += 1
                                active_connections[session_id][
                                    "chunk_count"
                                ] = chunk_count
                                print(
                                    f"[{session_id}] ✓ 识别文本[{chunk_count}]: {result['text']}"
                                )
                                await websocket.send_json(
                                    {
                                        "type": "result",
                                        "text": result["text"],
                                        "is_final": False,
                                        "chunk_number": chunk_count,
                                        "timestamp": result.get("timestamp"),
                                        "session_id": session_id,
                                    }
                                )
                            else:
                                pass  # print(f"[{session_id}] 识别结果为空(可能是静音段)")

                        except Exception as e:
                            print(f"[{session_id}] ASR识别错误: {e}")
                            import traceback

                            traceback.print_exc()
                            await websocket.send_json(
                                {"type": "error", "message": f"识别错误: {str(e)}"}
                            )

                except Exception as e:
                    print(f"[{session_id}] 处理音频数据失败: {e}")
                    import traceback

                    traceback.print_exc()
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"处理音频数据失败: {str(e)}",
                            "session_id": session_id,
                        }
                    )

    except WebSocketDisconnect:
        print(f"[{session_id}] WebSocket客户端断开连接")
    except Exception as e:
        print(f"[{session_id}] WebSocket错误: {e}")
        import traceback

        traceback.print_exc()
        try:
            await websocket.send_json(
                {"type": "error", "message": f"连接错误: {str(e)}"}
            )
        except:
            pass
    finally:
        # 清理连接记录
        with connection_lock:
            if session_id in active_connections:
                session_info = active_connections[session_id]
                duration = (datetime.now() - session_info["start_time"]).total_seconds()
                print(
                    f"[{session_id}] 会话结束: 时长={duration:.1f}秒, 处理块数={session_info['chunk_count']}"
                )
                del active_connections[session_id]
                print(f"当前活跃连接数: {len(active_connections)}")

        try:
            await websocket.close()
        except:
            pass


# ========== 配置 Starlette 应用（用于 uvicorn） ==========
# 使用 Streamable HTTP 传输（推荐，性能更好），并添加 CORS 支持
app = mcp.http_app(transport="streamable-http", middleware=cors_middleware)

# 添加自定义路由到 MCP 应用
app.add_route("/upload-audio", upload_audio_endpoint, methods=["POST"])
app.add_websocket_route("/ws/realtime", websocket_realtime_endpoint)


# 添加健康检查端点
@app.route("/health")
async def health_check(request: Request):
    """健康检查端点，用于确认服务器正常运行"""
    return JSONResponse(
        {
            "status": "healthy",
            "service": "FunASR MCP Server",
            "timestamp": str(asyncio.get_event_loop().time()),
            "active_connections": len(active_connections),
        }
    )


# 添加连接状态查询端点
@app.route("/connections")
async def connections_status(request: Request):
    """查询当前活跃连接状态"""
    with connection_lock:
        connections_info = []
        for session_id, info in active_connections.items():
            duration = (datetime.now() - info["start_time"]).total_seconds()
            connections_info.append(
                {
                    "session_id": session_id,
                    "duration_seconds": round(duration, 1),
                    "chunk_count": info["chunk_count"],
                    "start_time": info["start_time"].isoformat(),
                }
            )

        return JSONResponse(
            {
                "total_connections": len(active_connections),
                "connections": connections_info,
            }
        )


# ========== 启动信息 ==========
if __name__ == "__main__":
    import uvicorn

    print("正在启动FunASR MCP服务器 v0.3.0 (并发优化版)...")
    print(f"服务器地址: http://0.0.0.0:8000")
    print(f"MCP端点: http://0.0.0.0:8000/mcp")
    print("\n使用的模型:")
    print(f"  - 批量识别: {batch_transcriber.asr_model_path}")
    print(f"  - 流式识别: {realtime_transcriber.asr_model_path} (600ms延迟)")
    print(f"  - VAD模型: {batch_transcriber.vad_model_path}")
    print(f"  - 运行设备: {batch_transcriber.device}")
    print("\n可用功能:")
    print("  ✓ 批量语音识别 (VAD分段+批量ASR，适合音频文件)")
    print("  ✓ 实时语音识别 (WebSocket流式识别，Paraformer-Streaming)")
    print("  ✓ 多客户端并发支持 (线程锁保护模型推理)")
    print("  ✓ 音频文件验证")
    print("  ✓ 浏览器录音上传识别")
    print("\nWebSocket端点:")
    print("  ws://0.0.0.0:8000/ws/realtime (Paraformer流式识别)")
    print("\n监控端点:")
    print("  http://0.0.0.0:8000/health - 健康检查")
    print("  http://0.0.0.0:8000/connections - 活跃连接状态")
    print("\n使用 uvicorn 启动服务器...")
    print("提示: 生产环境可使用多进程:")
    print("  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4")
    print("")

    # 使用 uvicorn 启动服务器（增加超时配置以支持长连接）
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=75,  # Keep-alive 超时时间（秒）
        timeout_graceful_shutdown=30,  # 优雅关闭超时
    )
