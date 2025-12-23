"""FunASR MCP服务器主程序 v4.0.0

基于FastMCP框架提供专业的中文语音识别服务:
- 实时流式语音识别 (Paraformer-Streaming)
- 批量语音识别 (Paraformer-large + VAD分段 + 标点恢复 + 说话人分离)
- LLM流式后处理 (GGUF量化模型，CPU友好)
- 热词定制支持
- 多客户端并发支持

版本: 4.0.0
更新日期: 2025-12-23
"""

import os
import tempfile
import json
import asyncio
import threading
import uuid
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, Any

# 设置模型缓存到项目目录 (必须在导入funasr之前)
os.environ["MODELSCOPE_CACHE"] = "./Model"

import numpy as np
from fastmcp import FastMCP
from core.realtime_transcriber import RealtimeTranscriber
from core.batch_transcriber import BatchTranscriber
from core.device_utils import detect_device
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.requests import Request
from starlette.websockets import WebSocket, WebSocketDisconnect

# ========== 配置 ==========


class Config:
    """服务器配置"""

    # 服务器配置
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 8000
    TIMEOUT_KEEP_ALIVE = 75
    TIMEOUT_GRACEFUL_SHUTDOWN = 30

    # 模型配置
    MODEL_CACHE_DIR = "./Model"

    # 实时识别配置
    REALTIME_MODEL = "paraformer-zh-streaming"
    REALTIME_CHUNK_SIZE = [0, 10, 5]  # 600ms延迟
    REALTIME_NCPU = 4

    # 批量识别配置
    BATCH_MODEL = "paraformer-zh"
    BATCH_VAD_MODEL = "fsmn-vad"
    BATCH_PUNC_MODEL = "ct-punc-c"
    BATCH_SPK_MODEL = "cam++"
    BATCH_NCPU = 4
    BATCH_SIZE_S = 300
    BATCH_HOTWORD = "魔搭"

    # 流式后处理配置（GGUF格式）- 协同设计：ASR听清 + LLM说人话
    ENABLE_POSTPROCESSOR = True
    POSTPROCESSOR_MODEL_PATH = "./Model/models/Qwen/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    POSTPROCESSOR_TEMPERATURE = 0.3
    POSTPROCESSOR_N_GPU_LAYERS = None  # None=自动检测
    POSTPROCESSOR_CONTEXT_WINDOW = 3   # 上下文窗口大小
    POSTPROCESSOR_MIN_BUFFER = 2       # 最小缓冲区大小
    POSTPROCESSOR_MAX_BUFFER = 5       # 最大缓冲区大小
    POSTPROCESSOR_QUALITY_CHECK = True # 启用质量检查

    # WebSocket配置
    WS_CHUNK_SIZE_MS = 600
    WS_MAX_BUFFER_SIZE_CHUNKS = 3
    WS_MAX_BUFFER_BYTES = 64000
    WS_LOCK_TIMEOUT = 5.0


# ========== 日志配置 ==========

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ========== 全局变量 ==========

# FastMCP服务器
mcp = FastMCP(name="FunASR语音服务")

# 并发控制
realtime_model_lock = threading.Lock()
batch_model_lock = threading.Lock()
connection_lock = threading.Lock()

# 连接管理
active_connections: Dict[str, Dict[str, Any]] = {}
connection_counter = 0


# ========== 模型初始化 ==========


def init_models() -> tuple[RealtimeTranscriber, BatchTranscriber]:
    """初始化语音识别模型"""
    # 自动检测设备
    device = detect_device()
    logger.info(f"🖥️  计算设备: {device.upper()}")
    logger.info(f"✨ LLM后处理: 已启用 (GGUF模型)")

    # 实时识别器
    realtime = RealtimeTranscriber(
        model=Config.REALTIME_MODEL,
        device=device,
        ncpu=Config.REALTIME_NCPU,
        chunk_size=Config.REALTIME_CHUNK_SIZE,
        encoder_chunk_look_back=4,
        decoder_chunk_look_back=1,
        model_hub="ms",
        enable_llm_postprocess=Config.ENABLE_POSTPROCESSOR,
        llm_model_path=Config.POSTPROCESSOR_MODEL_PATH,
        llm_temperature=Config.POSTPROCESSOR_TEMPERATURE,
        llm_n_gpu_layers=Config.POSTPROCESSOR_N_GPU_LAYERS,
    )

    # 批量识别器
    batch = BatchTranscriber(
        model=Config.BATCH_MODEL,
        vad_model=Config.BATCH_VAD_MODEL,
        punc_model=Config.BATCH_PUNC_MODEL,
        spk_model=Config.BATCH_SPK_MODEL,
        device=device,
        ncpu=Config.BATCH_NCPU,
        vad_kwargs={"max_single_segment_time": 30000},
        batch_size_s=Config.BATCH_SIZE_S,
        model_hub="ms",
        hotword=Config.BATCH_HOTWORD,
    )

    return realtime, batch


realtime_transcriber, batch_transcriber = init_models()


# ========== 优雅关闭处理 ==========

def cleanup_resources():
    """清理所有资源"""
    try:
        logger.info("开始清理资源...")

        # 清理实时转录器
        if hasattr(realtime_transcriber, 'close'):
            realtime_transcriber.close()
            logger.info("实时转录器已关闭")

        # 清理批量转录器
        if hasattr(batch_transcriber, 'close'):
            batch_transcriber.close()
            logger.info("批量转录器已关闭")

        logger.info("✓ 资源清理完成")
    except Exception as e:
        logger.error(f"资源清理错误: {e}")


def signal_handler(signum, frame):
    """信号处理器"""
    logger.info(f"收到信号 {signum}，正在优雅关闭...")
    cleanup_resources()
    sys.exit(0)


# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ========== 注册MCP工具 ==========


# ---------- 批量语音识别工具 ----------
@mcp.tool(
    name="transcribe_audio",
    description="对音频文件进行批量语音识别，使用VAD分段后进行批量识别，支持热词定制",
)
def transcribe_audio(
    audio_path: str,
    return_vad_segments: bool = False,
    hotword: str | None = None,
) -> dict:
    """批量语音识别

    使用VAD进行语音分段，然后对所有语音段进行批量识别。
    适用于完整音频文件的离线处理。

    参数:
        audio_path: 音频文件路径
        return_vad_segments: 是否返回VAD分段的时间戳信息
        hotword: 热词，用于提高特定词汇的识别准确率 (例: "魔搭")

    返回:
        包含识别结果的字典:
        - status: "success" 或 "error"
        - text: 完整识别文本
        - results: FunASR原始结果列表
        - audio_path: 音频文件路径
        - audio_info: 音频文件信息
        - vad_segments: VAD分段信息(如果return_vad_segments=True)
    """
    # 使用锁保护模型推理 (支持并发调用)
    with batch_model_lock:
        kwargs = {}
        if hotword:
            kwargs["hotword"] = hotword

        if return_vad_segments:
            return batch_transcriber.transcribe_with_vad_segments(
                audio_path=audio_path, return_vad_segments=True, **kwargs
            )
        else:
            return batch_transcriber.transcribe(audio_path=audio_path, **kwargs)


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
            os.unlink(temp_path)

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": f"转录失败: {str(e)}"}, status_code=500
        )


# ========== WebSocket实时语音识别端点 ==========
async def websocket_realtime_endpoint(websocket: WebSocket):
    """WebSocket端点：实时接收音频流并进行识别

    使用paraformer-zh-streaming进行流式识别:
    - 模型内置VAD，自动检测语音活动
    - chunk_size配置为[0,10,5]，即600ms实时粒度
    - 每次输入600ms音频(9600 samples @ 16kHz)
    - 通过cache维护流式状态
    - 支持多客户端并发（使用线程锁保护模型推理）
    - 添加超时保护和资源管理
    - LLM流式后处理（协同设计）
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
        chunk_size = int(Config.WS_CHUNK_SIZE_MS * 16000 / 1000)  # 9600 samples
        max_buffer_size = (
            chunk_size * Config.WS_MAX_BUFFER_SIZE_CHUNKS
        )  # 最多缓冲3个chunk
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

                    # 重置后处理器状态（协同设计：新会话开始）
                    try:
                        if realtime_transcriber and hasattr(realtime_transcriber, 'reset_postprocessor'):
                            realtime_transcriber.reset_postprocessor()
                            logger.info(f"[{session_id}] 后处理器已重置（新会话开始）")
                    except Exception as e:
                        logger.warning(f"[{session_id}] 后处理器重置失败: {e}")

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
                    if len(buffer_bytes) > Config.WS_MAX_BUFFER_BYTES:
                        print(
                            f"[{session_id}] ⚠️ 字节缓冲区过大({len(buffer_bytes)}字节)，清理旧数据"
                        )
                        buffer_bytes = buffer_bytes[-Config.WS_MAX_BUFFER_BYTES :]

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
                            f"[{session_id}] 处理音频块 {chunk_count + 1}: {len(chunk)} 样本 ({Config.WS_CHUNK_SIZE_MS}ms)"
                        )

                        try:
                            # Paraformer流式识别 (标准FunASR流式用法)
                            # 使用线程锁保护模型推理（支持多客户端并发）
                            # 添加超时保护避免长时间阻塞
                            lock_acquired = realtime_model_lock.acquire(
                                timeout=Config.WS_LOCK_TIMEOUT
                            )
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
                                text = result["text"]

                                # 检查是否处于缓冲状态（LLM后处理器正在累积文本）
                                buffering = result.get("buffering", False)
                                llm_optimized = result.get("llm_optimized", False)

                                # 只有非缓冲状态的结果才是完整的、可以输出的
                                should_output = not buffering

                                # 构建日志信息
                                log_parts = [
                                    f"[{session_id}] {'[缓冲中]' if buffering else '✓'} 识别文本[{chunk_count}]: {text}"
                                ]
                                if llm_optimized:
                                    log_parts.append(f"[LLM优化]")
                                if result.get("speaker_id"):
                                    log_parts.append(f"[说话人:{result['speaker_id']}]")
                                if result.get("emotion"):
                                    log_parts.append(f"[情感:{result['emotion']}]")
                                print(" ".join(log_parts))

                                # 发送识别结果（包含缓冲状态和输出标志）
                                response = {
                                    "type": "result",
                                    "text": text,
                                    "is_final": result.get("is_final", False),
                                    "chunk_number": chunk_count,
                                    "timestamp": result.get("timestamp"),
                                    "session_id": session_id,
                                    "buffering": buffering,  # 是否正在缓冲
                                    "should_output": should_output,  # 是否应该输出给用户
                                    "llm_optimized": llm_optimized,  # 是否经过LLM优化
                                }

                                # 添加说话人信息（如果有）
                                if result.get("speaker_id") is not None:
                                    response["speaker_id"] = result["speaker_id"]

                                # 添加情感信息（如果有）
                                if result.get("emotion") is not None:
                                    response["emotion"] = result["emotion"]

                                # 添加VAD信息（如果有）
                                if result.get("is_speech") is not None:
                                    response["is_speech"] = result["is_speech"]

                                await websocket.send_json(response)
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

    print("正在启动FunASR MCP服务器 v3.0.0 (AI增强版)...")
    print(f"服务器地址: http://0.0.0.0:{Config.SERVER_PORT}")
    print(f"MCP端点: http://0.0.0.0:{Config.SERVER_PORT}/mcp")
    print("\n已加载模型:")
    print(f"  ASR批量: {Config.BATCH_MODEL}")
    print(
        f"  ASR流式: {Config.REALTIME_MODEL} ({Config.REALTIME_CHUNK_SIZE[1]*60}ms延迟)"
    )
    print(f"  VAD: {Config.BATCH_VAD_MODEL}")
    print(f"  标点: {Config.BATCH_PUNC_MODEL or '未启用'}")
    print(f"  说话人: {Config.BATCH_SPK_MODEL or '未启用'}")
    print(f"  LLM后处理: GGUF模型 (自动检测GPU/CPU)")
    print(f"  计算设备: {detect_device().upper()} (自动检测)")
    print("\n可用功能:")
    print("　✓ 批量语音识别 (VAD分段+批量ASR)")
    print("　✓ 实时语音识别 (WebSocket流式，Paraformer-Streaming)")
    print("  ✓ LLM流式后处理 (GGUF量化模型，自动GPU/CPU)")
    print("  ✓ 标点符号恢复")
    print("  ✓ 说话人分离")
    print("  ✓ 多客户端并发支持")
    print("  ✓ 音频文件验证")
    print("  ✓ 浏览器录音上传识别")
    print("\nWebSocket端点:")
    print(f"  ws://0.0.0.0:{Config.SERVER_PORT}/ws/realtime (Paraformer流式识别)")
    print("\n监控端点:")
    print(f"  http://0.0.0.0:{Config.SERVER_PORT}/health - 健康检查")
    print(f"  http://0.0.0.0:{Config.SERVER_PORT}/connections - 活跃连接状态")
    print("\n使用 uvicorn 启动服务器...")
    print("提示: 生产环境可使用多进程:")
    print(f"  uvicorn main:app --host 0.0.0.0 --port {Config.SERVER_PORT} --workers 4")
    print("")

    # 使用 uvicorn 启动服务器（增加超时配置以支持长连接）
    uvicorn.run(
        app,
        host=Config.SERVER_HOST,
        port=Config.SERVER_PORT,
        timeout_keep_alive=75,  # Keep-alive 超时时间（秒）
        timeout_graceful_shutdown=30,  # 优雅关闭超时
    )
