"""Chat 只接入两套 MCP：软件界面 zem-cleanliness，电镜 zem-sem。"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from backend import sem_mcp_client

APP_ROOT = Path(__file__).resolve().parents[1]
UI_SERVER = "zem-cleanliness"
SEM_SERVER = "zem-sem"
UI_SCRIPT = APP_ROOT / "tools" / "mcp" / "server.cjs"


def ui_command() -> tuple[str, list[str]]:
    node = shutil.which("node") or "node"
    return node, [str(UI_SCRIPT)]


def ui_available() -> bool:
    return UI_SCRIPT.is_file()


def allowed_servers() -> list[str]:
    names: list[str] = []
    if sem_mcp_client.available():
        names.append(SEM_SERVER)
    if ui_available():
        names.append(UI_SERVER)
    return names


def claude_config() -> dict:
    servers: dict[str, dict] = {}
    if sem_mcp_client.available():
        servers.update(sem_mcp_client.claude_config()["mcpServers"])
    if ui_available():
        command, args = ui_command()
        servers[UI_SERVER] = {"type": "stdio", "command": command, "args": args}
    return {"mcpServers": servers}


def _codex_server_args(name: str, settings: dict) -> list[str]:
    result: list[str] = []
    for key, value in settings.items():
        result.extend(["-c", f"mcp_servers.{name}.{key}={json.dumps(value, ensure_ascii=False)}"])
    return result


def codex_config_args() -> list[str]:
    """只注入允许的两套 MCP；凭据走环境变量，不写进命令行。"""
    args: list[str] = []
    if sem_mcp_client.available():
        args.extend(sem_mcp_client.codex_config_args())
    if ui_available():
        command, command_args = ui_command()
        args.extend(_codex_server_args(UI_SERVER, {
            "command": command,
            "args": command_args,
            "env_vars": [
                "ZEM_MCP_CDP_URL", "ZEM_MCP_ATTACH", "ZEM_MCP_ALLOW_HARDWARE",
                "PYTHONUTF8", "PYTHONIOENCODING",
            ],
            "enabled": True,
            "required": True,
            "startup_timeout_sec": 30,
            "tool_timeout_sec": 180,
            "default_tools_approval_mode": "approve",
        }))
    return args
