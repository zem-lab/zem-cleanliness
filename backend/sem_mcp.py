"""Standard MCP stdio bridge to the existing, authenticated local backend."""
from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from backend.cleanliness_core.device_control import DeviceBusy
from backend.sem_control import SemError
from backend.sem_tools import TOOLS


class BackendTransport:
    def __init__(self, url=None, token=None, caller=None):
        url = url or os.environ.get("ZEM_MCP_BACKEND_URL", "")
        parsed = urlsplit(url)
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port
                or parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.username or parsed.password):
            raise ValueError("MCP 只允许连接明确配置的本机后端 http://127.0.0.1:端口")
        self.url = url.rstrip("/")
        self.token = token or os.environ.get("ZEM_MCP_TOKEN", "")
        if not self.token:
            raise ValueError("未配置本机 MCP 访问凭据，请从桌面软件启动助手")
        self.caller = caller or os.environ.get("ZEM_MCP_CALLER_ID", "external-mcp")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(self, name, arguments):
        request = urllib.request.Request(self.url + "/mcp/call", method="POST",
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"},
            data=json.dumps({"name": name, "arguments": arguments, "caller": self.caller}, allow_nan=False).encode("utf-8"))
        try:
            with self.opener.open(request, timeout=30) as response:
                data = response.read(32 * 1024 * 1024 + 1)
                if len(data) > 32 * 1024 * 1024:
                    raise SemError("INVALID_RESPONSE", "后端图像响应超过大小上限")
                payload = json.loads(data)
        except urllib.error.HTTPError as error:
            raise SemError("BACKEND_REJECTED", f"本机后端拒绝访问（HTTP {error.code}），请重新从软件启动助手") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise SemError("SERVICE_UNAVAILABLE", "本机设备服务未响应；先核对原请求作业，不要换标识重复发送动作") from error
        if "error" in payload:
            raise SemError(payload["error"]["code"], payload["error"]["message"])
        return payload["result"]


def create_server(transport):
    from mcp.server import Server
    from mcp.types import CallToolResult, ImageContent, TextContent, Tool
    server = Server("zem-sem", version="0.1.0", instructions=(
        "电镜控制：先查询能力；台坐标使用 µm；动作返回作业后查询结果。"
        "结果未知时禁止重复下发。取消会话不代表设备停止。模拟图不是真机证据。"))

    @server.list_tools()
    async def list_tools():
        return [Tool(**tool) for tool in TOOLS]

    @server.call_tool()
    async def call_tool(name, arguments):
        try:
            result = await asyncio.to_thread(transport.call, name, arguments or {})
            result = dict(result)
            picture = result.pop("image", None)
            # A job result has an image metadata dictionary, not encoded pixels.
            content = [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, allow_nan=False))]
            if picture is not None:
                content.append(ImageContent(type="image", **picture))
            return CallToolResult(content=content, structuredContent=result)
        except (SemError, DeviceBusy) as error:
            record = error.record() if isinstance(error, SemError) else {"code": "DEVICE_BUSY", "message": str(error)}
        except Exception as error:
            record = {"code": "INTERNAL_ERROR", "message": str(error)}
        return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": record}, ensure_ascii=False))],
                              structuredContent={"error": record}, isError=True)

    return server


async def run_stdio(transport=None):
    import os
    import sys
    import anyio
    from mcp.server.stdio import stdio_server
    server = create_server(transport or BackendTransport())
    # The SDK's temporary TextIOWrapper closes the original buffers when it is
    # destroyed. PyInstaller flushes those buffers on exit and reports a closed
    # file error. Own duplicate descriptors so process stdio remains available.
    with os.fdopen(os.dup(sys.stdin.fileno()), "r", encoding="utf-8", errors="replace") as incoming, \
         os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as outgoing:
        async with stdio_server(stdin=anyio.wrap_file(incoming), stdout=anyio.wrap_file(outgoing)) as (read, write):
            await server.run(read, write, server.create_initialization_options())


def main():
    asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
