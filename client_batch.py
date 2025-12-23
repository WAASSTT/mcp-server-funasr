#!/usr/bin/env python3
"""FunASR 批量语音识别客户端 v4.0.0

使用 HTTP/MCP 协议调用进行批量语音识别（非流式）
适合处理音频文件，支持 VAD 分段、标点恢复、说话人分离和热词定制

功能:
- 健康检查: 验证服务器状态
- 工具列表: 查看可用的 MCP 工具
- 音频验证: 检查音频文件格式和属性
- 批量识别: 使用 Paraformer 进行高精度识别
- 热词支持: 提高特定词汇识别准确率

版本: 4.0.0
更新日期: 2025-12-23
"""

import sys
import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional


class FunASRClient:
    """FunASR MCP 客户端 (HTTP)"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.mcp_url = f"{base_url}/mcp"
        self.session = requests.Session()
        self.session_id = None
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }
        )
        self._initialize_session()

    def _initialize_session(self):
        """初始化 MCP 会话"""
        payload = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "funasr-client", "version": "4.0.0"},
            },
        }

        response = self.session.post(self.mcp_url, json=payload)
        response.raise_for_status()

        # 获取 session ID
        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self.session_id = session_id
            self.session.headers.update({"mcp-session-id": session_id})

    def check_health(self) -> Dict[str, Any]:
        """检查服务器健康状态"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"无法连接到服务器: {e}")

    def list_tools(self) -> Dict[str, Any]:
        """列出可用的 MCP 工具"""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
        }

        response = self.session.post(self.mcp_url, json=payload, stream=True)
        response.raise_for_status()

        # 解析 SSE 流
        result = self._parse_sse_response(response)

        if "error" in result:
            raise RuntimeError(f"MCP 错误: {result['error']}")

        return result.get("result", {})

    def _parse_sse_response(self, response) -> Dict[str, Any]:
        """解析 Server-Sent Events 响应"""
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]  # 移除 'data: ' 前缀
                    try:
                        return json.loads(data)
                    except json.JSONDecodeError:
                        continue
        return {}

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用 MCP 工具"""
        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        response = self.session.post(
            self.mcp_url, json=payload, timeout=300, stream=True
        )
        response.raise_for_status()

        # 解析 SSE 流
        result = self._parse_sse_response(response)

        if "error" in result:
            raise RuntimeError(f"工具调用失败: {result['error']}")

        # 解析返回的内容
        tool_result = result.get("result", {})
        if "content" in tool_result and isinstance(tool_result["content"], list):
            for content in tool_result["content"]:
                if content.get("type") == "text":
                    return json.loads(content["text"])

        return tool_result

    def validate_audio(self, file_path: str) -> Dict[str, Any]:
        """验证音频文件"""
        return self.call_tool("validate_audio_file", {"file_path": file_path})

    def transcribe_audio(
        self, audio_path: str, return_vad_segments: bool = False
    ) -> Dict[str, Any]:
        """批量语音识别"""
        return self.call_tool(
            "transcribe_audio",
            {"audio_path": audio_path, "return_vad_segments": return_vad_segments},
        )

    def upload_and_transcribe(self, audio_file_path: str) -> Dict[str, Any]:
        """上传音频文件并识别 (使用直接上传端点)"""
        file_path = Path(audio_file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {audio_file_path}")

        with open(file_path, "rb") as f:
            files = {"audio": (file_path.name, f, "audio/webm")}
            response = self.session.post(
                f"{self.base_url}/upload-audio", files=files, timeout=300
            )
            response.raise_for_status()
            return response.json()


def print_result(result: Dict[str, Any], title: str = "结果"):
    """格式化打印结果"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 60)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="FunASR MCP Python 客户端")
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="服务器地址 (默认: http://localhost:8000)",
    )
    parser.add_argument(
        "command",
        choices=["health", "list-tools", "validate", "transcribe"],
        help="要执行的命令",
    )
    parser.add_argument(
        "audio_file", nargs="?", help="音频文件路径 (validate/transcribe 命令需要)"
    )
    parser.add_argument("--vad", action="store_true", help="返回 VAD 分段信息")

    args = parser.parse_args()

    # 创建客户端
    client = FunASRClient(args.server)

    try:
        if args.command == "health":
            # 健康检查
            result = client.check_health()
            print_result(result, "服务器健康状态")

        elif args.command == "list-tools":
            # 列出工具
            result = client.list_tools()
            print_result(result, "可用工具列表")

            # 打印工具摘要
            if "tools" in result:
                print("\n工具摘要:")
                for tool in result["tools"]:
                    print(f"  • {tool['name']}")
                    print(f"    {tool.get('description', '无描述')}")

        elif args.command == "validate":
            # 验证音频
            if not args.audio_file:
                print("错误: validate 命令需要提供音频文件路径")
                return 1

            result = client.validate_audio(args.audio_file)
            print_result(result, f"音频验证: {args.audio_file}")

        elif args.command == "transcribe":
            # 语音识别
            if not args.audio_file:
                print("错误: transcribe 命令需要提供音频文件路径")
                return 1

            print(f"正在识别音频: {args.audio_file}")
            print("(这可能需要一些时间...)")

            result = client.transcribe_audio(args.audio_file, args.vad)
            print_result(result, f"识别结果: {args.audio_file}")

            # 打印文本摘要
            if result.get("status") == "success" and "text" in result:
                print(f"\n完整文本:\n{result['text']}\n")

        return 0

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
