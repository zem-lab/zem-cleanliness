"""Use the same real MCP protocol from the built-in assistant engines."""
from __future__ import annotations

import asyncio
import json
from datetime import timedelta
import os
from pathlib import Path
import sys


def bridge_command():
    if getattr(sys, "frozen", False):
        return sys.executable, ["--mcp-stdio"]
    return sys.executable, [str(Path(__file__).with_name("server.py")), "--mcp-stdio"]


def available():
    return bool(os.environ.get("ZEM_MCP_BACKEND_URL") and os.environ.get("ZEM_MCP_TOKEN"))


def claude_config():
    command, args = bridge_command()
    return {"mcpServers": {"zem-sem": {"type": "stdio", "command": command, "args": args}}}


def codex_config_args():
    """Per-invocation TOML overrides; credentials travel only in environment.

    Codex filters inherited MCP environment variables, so explicitly name the
    backend credentials and session identity that the bridge needs.

    ``codex exec`` defaults to ``approval_policy=never``. Non-readonly MCP tools
    then fail with “requires approval, but approval policy is never” before the
    bridge runs. Approve zem-sem tools here; device gates (allowCapture 等) still
    apply in the backend.
    """
    command, args = bridge_command()
    settings = {
        "command": command,
        "args": args,
        "env_vars": ["ZEM_MCP_BACKEND_URL", "ZEM_MCP_TOKEN", "ZEM_MCP_CALLER_ID",
                     "PYTHONUTF8", "PYTHONIOENCODING"],
        "enabled": True,
        "required": True,
        "startup_timeout_sec": 20,
        "tool_timeout_sec": 45,
        "default_tools_approval_mode": "approve",
    }
    result = []
    for key, value in settings.items():
        result.extend(["-c", f"mcp_servers.zem-sem.{key}={json.dumps(value, ensure_ascii=False)}"])
    return result


async def _call(name, arguments, caller):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    command, args = bridge_command()
    env = {**os.environ, "ZEM_MCP_CALLER_ID": caller, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    params = StdioServerParameters(command=command, args=args, env=env, cwd=Path(__file__).resolve().parents[1])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=40)) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            return result.model_dump(mode="json", exclude_none=True)


def call_tool(name, arguments, caller):
    if not available():
        raise RuntimeError("设备 MCP 未就绪，请从新版桌面后端启动助手")
    return asyncio.run(_call(name, arguments, caller))
