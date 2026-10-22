"""Vibe 模式聊天后端：LLM 引擎管理（claude-code / DeepSeek 直连 / codex cli）。

默认引擎为 claude-code（Claude Code CLI 做代理壳，DeepSeek API 做大脑）：
- spawn ``claude -p <prompt> --output-format stream-json --dangerously-skip-permissions``
- env 注入 ANTHROPIC_BASE_URL 指向 DeepSeek 的 Anthropic 兼容端点
- 代理壳可读写文件、运行命令、浏览仓库，能力不局限于预定义动作
- 会话续接用 ``--resume <session-id>``（session id 取自 stream-json 的 init 事件）

DeepSeek 直连引擎（``ZEM_VIBE_ENGINE=deepseek``）：后端自建工具调用循环，支持
run_vibe_action 与电镜 MCP 工具。

Codex 引擎（``ZEM_VIBE_ENGINE=codex``）：通过本机 CPA 启动器接入，支持电镜 MCP。

测试可用 ``ZEM_VIBE_CLAUDE_FAKE`` / ``ZEM_VIBE_CODEX_FAKE``（假 CLI 脚本）或
monkeypatch ``_llm_chat``（假 LLM）避免外部依赖。
"""

from __future__ import annotations

import json
import base64
import codecs
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from backend.cleanliness_core.atomic_json import write_json_atomic
from backend.sem_tools import TOOL_MAP, tools_for_model
from backend import chat_mcp, sem_mcp_client

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("ZEM_VIBE_CHAT_DIR", str(APP_ROOT / ".tmp" / "vibe-chat")))
SESSIONS_FILE = DATA_ROOT / "sessions.json"


def _load_env_file(path: Path) -> None:
    """加载 KEY=VALUE 本地配置（不覆盖已存在的环境变量）。"""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(APP_ROOT / ".env.local")

ENGINE = os.environ.get("ZEM_VIBE_ENGINE", "claude").strip().lower()

# claude-code 代理壳配置（默认大脑 = DeepSeek 的 Anthropic 兼容端点）
CLAUDE_CMD = os.environ.get("ZEM_VIBE_CLAUDE_CMD", "claude")
CLAUDE_BASE_URL = os.environ.get("ZEM_VIBE_CLAUDE_BASE_URL", "https://api.deepseek.com/anthropic")
CLAUDE_MODEL = os.environ.get("ZEM_VIBE_CLAUDE_MODEL", "deepseek-chat")
CLAUDE_AUTH_TOKEN = os.environ.get("ZEM_VIBE_CLAUDE_AUTH_TOKEN", "").strip()
CLAUDE_MAX_TURNS = int(os.environ.get("ZEM_VIBE_CLAUDE_MAX_TURNS", "40"))
CLAUDE_CONTEXT_TOKENS = os.environ.get("ZEM_VIBE_CLAUDE_CONTEXT_TOKENS", "1048576")
CLAUDE_COMPACT_WINDOW = os.environ.get("ZEM_VIBE_CLAUDE_COMPACT_WINDOW", "900000")
CLAUDE_THINKING_TOKENS = os.environ.get("ZEM_VIBE_CLAUDE_THINKING_TOKENS", "65536")
CLAUDE_FAKE = os.environ.get("ZEM_VIBE_CLAUDE_FAKE", "").strip()

# DeepSeek / OpenAI 兼容直连配置
LLM_BASE_URL = os.environ.get("ZEM_VIBE_LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
LLM_MODEL = os.environ.get("ZEM_VIBE_LLM_MODEL", "deepseek-chat")
LLM_API_KEY = os.environ.get("ZEM_VIBE_LLM_API_KEY", "").strip()

# codex cli 引擎配置（保留，ZEM_VIBE_ENGINE=codex 时使用）
CODEX_CMD = os.environ.get("ZEM_VIBE_CODEX_CMD", "").strip()
CODEX_PROVIDER = os.environ.get("ZEM_VIBE_CODEX_PROVIDER", "cpa")
CODEX_EFFORT = os.environ.get("ZEM_VIBE_CODEX_EFFORT", "medium")
FAKE_CODEX = os.environ.get("ZEM_VIBE_CODEX_FAKE", "").strip()

MAX_EVENTS_PER_SESSION = 600
MAX_LLM_STEPS = 12
MAX_HISTORY_MESSAGES = 60
ACTION_OUTPUT_TAIL = 4000
EVENT_SEQ_LOCK = threading.Lock()

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
# 约定：[图片] 绝对路径（单独一行）。兼容模型误写的 Markdown：[图片](path) / [图片](<path>)
IMAGE_MARKER_RE = re.compile(
    r"\[图片\][ \t]*(?:\()?[ \t]*(?:<)?[ \t]*(\"?)([A-Za-z]:[\\/][^\r\n`|]+?\.(?:png|jpe?g|bmp|gif))\1(?:\>)?(?:\))?",
    re.IGNORECASE,
)
IMAGE_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/][^\s`'\"<>|]+?\.(?:png|jpe?g|bmp|gif))", re.IGNORECASE
)


def _normalize_image_candidate(candidate: str) -> str:
    path = candidate.strip().strip('"').strip("'")
    path = path.removeprefix("<").removesuffix(">")
    path = path.removesuffix(")").rstrip()
    return path


def _extract_image_paths(text: str) -> list[str]:
    """从事件文本中提取要展示的图片路径（标记优先，其次已存在的图片路径）。"""
    candidates: list[str] = []
    for match in IMAGE_MARKER_RE.finditer(text or ""):
        candidates.append(match.group(2))
    if not candidates:
        candidates = IMAGE_PATH_RE.findall(text or "")
    seen: set[str] = set()
    paths: list[str] = []
    for candidate in candidates:
        path = _normalize_image_candidate(candidate)
        try:
            resolved = str(Path(path).resolve())
        except Exception:
            continue
        if resolved in seen or Path(resolved).suffix.lower() not in IMAGE_EXTS:
            continue
        if not Path(resolved).is_file():
            continue
        seen.add(resolved)
        paths.append(resolved)
    return paths


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


MCP_POLICY = """\
工作时只能使用这两个 MCP，禁止第三个 MCP，也禁止用 shell/HTTP/脚本顶替：
- zem-cleanliness：操作当前已打开的清洁度软件界面。创建「直接 SEM 分析」任务必须 navigate(target=sem-direct)。软件已在运行时不要 app_launch 再开窗口。
- zem-sem：电镜设备（状态、台坐标、导航图、GetImg、采图、移动、RPC）。
禁止：GET/POST 后端（含 /sem/navigation、/tasks、/mcp/call）、Invoke-RestMethod、curl、e2e/vibe-actions.cjs、直接调用 zemrpc。
外部仿真上位机（GetModel=ZEM_SIM 或 SIMULATED）不能开枪。不要要求用户开枪，不要调用 SetHV_ON。关枪不阻止创建任务、预览网格和开始分析。只有用户明确说「开枪」且真机允许时才可以开枪。
"""

# 三档取图 + 作业等待：Codex / Claude / DeepSeek 共用，避免漂移。
IMAGE_ROUTING = """\
取图分流（三者互不替代）：
- 导航图 / 导航概览 / GetNav：只用 sem_get_navigation（优先）或 sem_rpc_read 的 GetNavImg；禁止用 GetImg 或 sem_capture 冒充。
- 看图 / 拿当前画面：只读 GetImg（sem_rpc_read，values=[0]，带 request_id）→ sem_wait_job → sem_get_image；不要为看图调用 sem_capture，不要查枪/要求开枪。
- 新拍 / 重新采集：仅用户明确说时用 sem_capture 或 sem_move_and_capture（JSON-RPC snap）；禁止用 UDP Snap 或 GetImg 冒充新拍。
展示图片必须单独一行：`[图片] D:\\绝对路径\\preview.png`（空格分隔，禁止 Markdown 链接或尖括号）。须先 sem_get_image 拿到真实 display_path；作业未完成或失败不得编造路径。
GetImg 若得到雪花噪声/空图：如实说明当前缓冲无效，不得当成导航图或成功画面。
作业等待：sem_wait_job 的 timeout_seconds 每次最多 20（默认 15）；未结束继续等原 job_id，禁止传 30+，禁止换 request_id 重发。超时/unknown 先核验原作业。取消对话不代表设备已停止。
"""

CODEX_PREAMBLE_TEMPLATE = """\
你是 ZEPTOOLS 清洁度检测软件的 vibe 操作助手。软件窗口已经打开，你只通过 MCP 做事。

""" + MCP_POLICY + """
当前软件状态快照：
{snapshot}

规则：
1. 每一步操作前用一句话告诉用户你要做什么。
2. 用户提到的 JSON 文件如果不确定路径，先向用户确认，不要猜。
3. 命令失败时如实报告错误原文与截图路径，不要编造成功。
4. 软件界面动作（新建任务、翻页、填表、点按钮）只用 zem-cleanliness；电镜读写只用 zem-sem。
5. 每个会话首次查询 sem_get_capabilities，地址或权限改变时刷新。状态和坐标优先一次 sem_get_snapshot；台坐标单位为 µm。
6. """ + IMAGE_ROUTING + """
7. 电镜参数用 sem_rpc_read；设置、对焦、开关枪和换样先 sem_list_rpc 再 sem_rpc_execute。未开放不绕过；受理不等于已验证完成。
8. sem_rpc_call 可调全部原始命令；accepted 仅受理（completion_confirmed=false）。ZemRpc 只查服务地址；SetIe 未实现。动作审批被拒绝时如实说明。
"""


def build_prompt(user_text: str, snapshot: str) -> str:
    return CODEX_PREAMBLE_TEMPLATE.format(snapshot=snapshot) + f"\n用户消息：\n{user_text}\n"


CLAUDE_PREAMBLE_TEMPLATE = """\
你是 ZEPTOOLS 清洁度检测软件的 vibe 操作助手。软件窗口已经打开。回答问题可以读仓库，\
执行软件或电镜动作只能走下面两个 MCP。

""" + MCP_POLICY + """
当前状态快照：
{snapshot}

规则：
1. 动手前用一句话说明你要做什么。
2. 真实驱动设备、耗时很长的操作（如开始自动分析）必须先获得用户明确同意。
3. 失败时如实报告错误原文与相关文件路径，不要编造成功。
4. 软件界面只用 zem-cleanliness；电镜只用 zem-sem。
5. 每个会话首次查 sem_get_capabilities；状态坐标优先 sem_get_snapshot。台坐标单位 µm。
6. """ + IMAGE_ROUTING + """
7. 设置类先 sem_list_rpc 再 sem_rpc_execute；sem_rpc_call 仅按需。accepted 仅受理（completion_confirmed=false）。
8. ZemRpc 只查服务地址；SetIe 未实现。动作审批被拒绝时如实说明。
"""


def build_claude_prompt(user_text: str, snapshot: str) -> str:
    return CLAUDE_PREAMBLE_TEMPLATE.format(snapshot=snapshot) + f"\n用户消息：\n{user_text}\n"


def _resolve_claude_command() -> str:
    """Popen 不解析 PATHEXT，npm 的 claude shim 是 .cmd，需要显式找到。"""
    import shutil

    for candidate in ("claude.cmd", "claude.exe", "claude", "claude.ps1"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return CLAUDE_CMD


def _claude_command(prompt: str, claude_session_id: str | None) -> list[str]:
    """真实 claude 走 stdin 传 prompt（多行文本经 .cmd shim 的 argv 会被换行截断）；
    假脚本仍用 argv 便于测试断言。"""
    if CLAUDE_FAKE:
        command = [sys.executable, CLAUDE_FAKE]
        if claude_session_id:
            command += ["--resume", claude_session_id]
        command += [prompt]
        return command
    command = [
        _resolve_claude_command(), "-p",
        "--output-format", "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--max-turns", str(CLAUDE_MAX_TURNS),
    ]
    if chat_mcp.allowed_servers():
        config_path = DATA_ROOT / "zem-mcp.json"
        write_json_atomic(config_path, chat_mcp.claude_config())
        command += ["--mcp-config", str(config_path)]
    if claude_session_id:
        command += ["--resume", claude_session_id]
    return command


def _claude_env() -> dict[str, str]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    if CLAUDE_FAKE:
        return env
    if CLAUDE_BASE_URL:
        env["ANTHROPIC_BASE_URL"] = CLAUDE_BASE_URL
    if CLAUDE_AUTH_TOKEN:
        env["ANTHROPIC_AUTH_TOKEN"] = CLAUDE_AUTH_TOKEN
    if CLAUDE_MODEL:
        env["ANTHROPIC_MODEL"] = CLAUDE_MODEL
        env["ANTHROPIC_SMALL_FAST_MODEL"] = CLAUDE_MODEL
    # 上下文与思考预算：deepseek-flash 实际 1M 窗口，claude 壳对未知模型默认按 200k 规划，
    # 显式声明才能吃满；思考预算给到最大档。
    env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = CLAUDE_CONTEXT_TOKENS
    env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = CLAUDE_COMPACT_WINDOW
    env["MAX_THINKING_TOKENS"] = CLAUDE_THINKING_TOKENS
    return env


def _extract_claude_events(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """把一条 claude stream-json 事件转成 (kind, text) 列表。"""
    kind_type = str(payload.get("type") or "")
    events: list[tuple[str, str]] = []
    if kind_type == "system" and payload.get("subtype") == "init":
        return events  # session id 单独提取
    if kind_type == "assistant":
        message = payload.get("message") or {}
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and str(block.get("text") or "").strip():
                events.append(("assistant", str(block["text"]).strip()))
            elif block.get("type") == "tool_use":
                detail = json.dumps(block.get("input") or {}, ensure_ascii=False)[:400]
                events.append(("action", f"⚙ {block.get('tool_name') or block.get('name')} {detail}"))
    elif kind_type == "user":
        message = payload.get("message") or {}
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                content = block.get("content")
                if isinstance(content, list):
                    content = " ".join(str(item.get("text") or "") for item in content if isinstance(item, dict))
                text = str(content or "").strip()
                if text:
                    events.append(("action", text[-1500:]))
    elif kind_type == "result":
        if payload.get("subtype") not in {None, "success"}:
            events.append(("error", str(payload.get("subtype")) + " " + str(payload.get("result") or "")[:500]))
    return events


def _extract_claude_session_id(payload: dict[str, Any]) -> str | None:
    if str(payload.get("type") or "") == "system" and payload.get("subtype") == "init":
        value = payload.get("session_id")
        if isinstance(value, str) and value:
            return value
    return None


SYSTEM_PROMPT_TEMPLATE = """\
你是 ZEPTOOLS 清洁度检测软件的 vibe 操作助手。软件窗口已经打开。

""" + MCP_POLICY + """
当前软件状态快照：
{snapshot}

规则：
1. 每次调用工具前用一句话告诉用户你要做什么。
2. 用户提到的 JSON 文件路径不确定时先问用户，不要猜。
3. 工具失败时如实报告错误原文与截图路径；可以先修正参数重试一次，再失败就告诉用户。
4. 不要编造检测结果。run_vibe_action 已停用。
5. 电镜操作用 sem_ 工具（zem-sem）；状态坐标用 sem_get_snapshot。台坐标 µm；不要隐式开枪或发起 EDS。
6. """ + IMAGE_ROUTING + """
7. sem_get_image 展示已保存图；未传入模型的图片不能声称已看图。
8. 设置先 sem_list_rpc 再 execute；sem_rpc_call 的 accepted 仅受理（completion_confirmed=false）。
9. ZemRpc 只查服务地址；SetIe 未实现。动作审批被拒绝时如实说明。
"""


def build_system_prompt(snapshot: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(snapshot=snapshot)


def model_configuration() -> dict[str, Any]:
    """Configuration presence only; never claim an untested key authenticated."""
    if ENGINE == "deepseek":
        return {"engine": ENGINE, "model": LLM_MODEL, "credentialConfigured": bool(LLM_API_KEY)}
    if ENGINE == "claude":
        configured = bool(CLAUDE_AUTH_TOKEN or os.environ.get("ANTHROPIC_AUTH_TOKEN")) if "deepseek.com" in CLAUDE_BASE_URL else None
        return {"engine": ENGINE, "model": CLAUDE_MODEL, "credentialConfigured": configured}
    if ENGINE == "codex":
        return {"engine": ENGINE, "model": "", "credentialConfigured": None,
                "provider": CODEX_PROVIDER, "launcherConfigured": Path(CODEX_CMD).is_file()}
    return {"engine": ENGINE, "model": "", "credentialConfigured": None}


def _vibe_action_command(arguments: dict[str, Any]) -> list[str]:
    command = [
        "node", str(APP_ROOT / "e2e" / "vibe-actions.cjs"),
        "detect", "--json", str(arguments.get("json") or ""),
    ]
    if arguments.get("min"):
        command += ["--min", str(arguments["min"])]
    if arguments.get("max"):
        command += ["--max", str(arguments["max"])]
    if arguments.get("task_root"):
        command += ["--task-root", str(arguments["task_root"])]
    return command


def _shellish(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


VIBE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_vibe_action",
            "description": (
                "运行 ZEPTOOLS 清洁度软件的 e2e 动作（像人工一样驱动软件界面）。"
                "detect：导入光镜颗粒 JSON → 筛选 → AB 标定 → SEM 精定位 → 检测 → PDF 报告。"
                "前提：电镜与能谱在线；会真实驱动设备；耗时可能长达 1 小时。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "json": {"type": "string", "description": "光镜颗粒 JSON 的绝对路径"},
                    "min": {"type": "string", "description": "最小长度 um（可选）"},
                    "max": {"type": "string", "description": "最大长度 um（可选）"},
                    "task_root": {"type": "string", "description": "任务输出目录（可选，默认自动管理）"},
                },
                "required": ["json"],
            },
        },
    }
]


def _llm_chat(system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    """调用 DeepSeek（OpenAI 兼容）chat/completions，非流式。"""
    if not LLM_API_KEY:
        raise RuntimeError("未配置 DeepSeek API key：请在仓库根目录 .env.local 或环境变量 ZEM_VIBE_LLM_API_KEY 中配置")
    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system}, *messages],
        "tools": tools,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_API_KEY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"LLM HTTP {error.code}：{detail}") from error


def _codex_command(prompt: str, codex_session_id: str | None) -> list[str]:
    if FAKE_CODEX:
        command = [sys.executable, FAKE_CODEX]
        if codex_session_id:
            command += ["--resume", codex_session_id]
        command += ["--cd", str(APP_ROOT), prompt]
        return command
    # Use CPA's Python launcher directly when configured: the .cmd wrapper only
    # invokes that file, but adds a Windows shell quoting layer to TOML values.
    # A frozen backend executable is not a Python interpreter. Use the same
    # external Python launcher as CPA's .cmd entry in that environment.
    python = "py" if getattr(sys, "frozen", False) else sys.executable
    launcher = [python, CODEX_CMD] if Path(CODEX_CMD).suffix.lower() == ".py" else [CODEX_CMD]
    # Packaged backends run under resources/, which is not a trusted git checkout.
    # Codex refuses untrusted --cd unless skip-git-repo-check is set on `exec`;
    # the desktop app already chooses the working directory explicitly via --cd.
    codex_cd = Path(os.environ.get("ZEM_VIBE_CODEX_CD", str(APP_ROOT))).resolve()
    command = [*launcher, "-p", CODEX_PROVIDER, "--", "--cd", str(codex_cd),
               "-c", f'model_reasoning_effort="{CODEX_EFFORT}"']
    if chat_mcp.allowed_servers():
        command.extend(chat_mcp.codex_config_args())
    command += ["exec", "--skip-git-repo-check"]
    if codex_session_id:
        command += ["resume", codex_session_id]
    command += ["--json", "-"]
    return command


def _terminate_process_tree(process) -> None:
    """Stop CPA/CLI/bridge children too; backend device jobs remain independent."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        completed = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        if completed.returncode == 0:
            return
    if process.poll() is None:
        process.kill()


def _codex_output_lines(process):
    """Drain Codex output without waiting for handles inherited by background children.

    CPA may exit successfully while a Codex background Git process still owns the
    pipe. EOF alone therefore cannot determine whether the conversation finished.
    Nonblocking reads preserve all bytes queued before the launcher's exit.
    """
    assert process.stdout is not None
    descriptor = process.stdout.fileno()
    os.set_blocking(descriptor, False)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    try:
        while True:
            try:
                chunk = os.read(descriptor, 65536)
            except BlockingIOError:
                chunk = None
            if chunk:
                pending += decoder.decode(chunk)
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    yield line
                continue
            if chunk == b"" or process.poll() is not None:
                pending += decoder.decode(b"", final=True)
                if pending:
                    yield pending
                return
            time.sleep(.05)
    finally:
        process.stdout.close()


def _extract_events(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """把一条 codex JSONL 事件转成 (kind, text) 列表，兼容新旧两种协议。"""
    events: list[tuple[str, str]] = []
    msg = payload.get("msg") if isinstance(payload.get("msg"), dict) else payload
    kind = str(msg.get("type") or "")
    item = msg.get("item") if isinstance(msg.get("item"), dict) else {}

    if kind in {"thread.started", "session_created", "session_configured"}:
        return events  # session id 单独提取，不进事件流
    if kind == "agent_message" or item.get("type") == "agent_message":
        text = str(msg.get("message") or item.get("text") or "").strip()
        if text:
            events.append(("assistant", text))
    elif kind == "agent_message_delta":
        text = str(msg.get("delta") or "").strip()
        if text:
            events.append(("assistant", text))
    elif kind in {"exec_command_begin", "command_execution"} or item.get("type") == "command_execution":
        command = msg.get("command") or item.get("command") or ""
        if isinstance(command, list):
            command = " ".join(str(part) for part in command)
        command = str(command).strip()
        if command:
            events.append(("action", f"$ {command}"))
    elif kind in {"exec_command_end"}:
        output = str(msg.get("aggregated_output") or "").strip()
        if output:
            events.append(("action", output[-2000:]))
    elif item.get("type") == "mcp_tool_call":
        events.append(("action", f"MCP {item.get('server', '')}/{item.get('tool', '')}"))
        result = item.get("result") or {}
        for block in result.get("content", []) if isinstance(result, dict) else []:
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                events.append(("action", text))
                try:
                    picture = json.loads(text).get("display_path")
                except (ValueError, AttributeError):
                    picture = None
                if isinstance(picture, str):
                    events.append(("action", f"[图片] {picture}"))
        if item.get("error"):
            events.append(("error", str(item["error"])))
    elif kind in {"task_started", "turn.started"}:
        events.append(("system", "codex 开始执行任务"))
    elif kind in {"task_complete", "turn.completed"}:
        events.append(("system", "codex 任务完成"))
    elif kind in {"error", "stream_error", "turn.failed"}:
        error = msg.get("error") or msg.get("message") or payload
        if isinstance(error, dict):
            error = error.get("message") or json.dumps(error, ensure_ascii=False)
        events.append(("error", str(error)))
    else:
        text = json.dumps(msg, ensure_ascii=False)[:500]
        events.append(("debug", text))
    return events


def _extract_session_id(payload: dict[str, Any]) -> str | None:
    for container in (payload, payload.get("msg") or {}, payload.get("item") or {}):
        if not isinstance(container, dict):
            continue
        for key in ("thread_id", "session_id", "sessionId"):
            value = container.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _trim_history(history: list[dict[str, Any]], limit: int = MAX_HISTORY_MESSAGES) -> list[dict[str, Any]]:
    """截断历史，但保证首条不是孤立的 tool 消息（工具消息必须跟在 assistant 后）。"""
    if len(history) <= limit:
        return history
    trimmed = history[-limit:]
    while trimmed and trimmed[0].get("role") == "tool":
        trimmed = trimmed[1:]
    return trimmed


class ChatError(Exception):
    """带 HTTP 状态码语义的聊天错误。"""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class ChatManager:
    """内存会话表 + 落盘持久化 + 每会话一个引擎执行线程。"""

    def __init__(
        self,
        snapshot_provider: Callable[[], str] | None = None,
        data_root: Path | None = None,
        engine: str | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._processes: dict[str, subprocess.Popen[Any]] = {}
        self._snapshot_provider = snapshot_provider or (lambda: "（快照不可用）")
        self._sessions_file = (data_root or DATA_ROOT) / "sessions.json"
        self._engine = (engine or ENGINE).strip().lower()
        self._next_seq = 0
        self._cancel_requested: set[str] = set()
        self._mcp_images: dict[str, list[dict[str, Any]]] = {}
        self._load()

    # ---------- 持久化 ----------

    def _load(self) -> None:
        if not self._sessions_file.is_file():
            return
        try:
            payload = json.loads(self._sessions_file.read_text(encoding="utf-8"))
        except Exception:
            return
        for item in payload.get("sessions", []):
            if not isinstance(item, dict) or not item.get("id"):
                continue
            item["status"] = "failed" if item.get("status") == "running" else item.get("status", "idle")
            self._sessions[str(item["id"])] = item
            self._next_seq = max(self._next_seq, max(
                (event["seq"] for event in item.get("events", []) if isinstance(event.get("seq"), int)), default=0))

    def _persist(self) -> None:
        sessions = sorted(self._sessions.values(), key=lambda item: item.get("createdAt", ""), reverse=True)
        write_json_atomic(self._sessions_file, {"sessions": sessions})

    # ---------- 视图 ----------

    @staticmethod
    def _view(session: dict[str, Any], include_events: bool = False, after: int = 0) -> dict[str, Any]:
        view = {
            "id": session["id"],
            "title": session["title"],
            "createdAt": session["createdAt"],
            "status": session["status"],
            "engine": session.get("engine"),
            "codexSessionId": session.get("codexSessionId"),
            "claudeSessionId": session.get("claudeSessionId"),
            "lastEventAt": session["events"][-1]["ts"] if session["events"] else session["createdAt"],
        }
        if include_events:
            view["events"] = [event for event in session["events"] if event["seq"] > after]
            view["cursor"] = session["events"][-1]["seq"] if session["events"] else after
        return view

    # ---------- 对外 API ----------

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._view(session) for session in self._sessions.values()]

    def create_session(self, title: str | None = None) -> dict[str, Any]:
        session = {
            "id": uuid.uuid4().hex,
            "title": (title or "新会话").strip()[:60] or "新会话",
            "createdAt": now(),
            "status": "idle",
            "engine": self._engine,
            "codexSessionId": None,
            "claudeSessionId": None,
            "events": [],
            "messages": [],
        }
        with self._lock:
            self._sessions[session["id"]] = session
            self._persist()
        return self._view(session, include_events=True)

    def get_session(self, session_id: str, after: int = 0) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ChatError("会话不存在", status=404)
            return self._view(session, include_events=True, after=after)

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                raise ChatError("会话不存在", status=404)
            self._persist()
        self._cancel_process(session_id)

    def cancel(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ChatError("会话不存在", status=404)
            if session["status"] != "running":
                raise ChatError("会话没有正在运行的任务")
            # 进程可能还没被 _run_turn 注册（启动竞态），先记下取消意图，
            # 引擎线程会在同一把锁下复查并补杀。
            self._cancel_requested.add(session_id)
            process = self._processes.get(session_id)
        if process is not None and process.poll() is None:
            _terminate_process_tree(process)
        self._append_event(session_id, "system", "用户已取消当前对话；已提交的设备作业可能仍在运行，请查询作业状态，不要据此判断设备已停止")

    def send_message(self, session_id: str, text: str) -> dict[str, Any]:
        text = str(text or "").strip()
        if not text:
            raise ChatError("消息不能为空")
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ChatError("会话不存在", status=404)
            if session["status"] == "running":
                raise ChatError("当前会话正在执行任务，请等待完成或先取消", status=409)
            session["status"] = "running"
            self._cancel_requested.discard(session_id)
            self._append_event(session_id, "user", text)
            thread = threading.Thread(
                target=self._run_turn,
                args=(session_id, text),
                name=f"vibe-chat-{session_id[:8]}",
                daemon=True,
            )
            thread.start()
            return self._view(session)

    # ---------- 内部 ----------

    def _append_event(self, session_id: str, kind: str, text: str) -> None:
        # self._lock 是 RLock，持锁调用时直接重入。
        with self._lock:
            with EVENT_SEQ_LOCK:
                self._next_seq += 1
                seq = self._next_seq
            session = self._sessions.get(session_id)
            if session is None:
                return
            session["events"].append({"seq": seq, "ts": now(), "kind": kind, "text": text})
            del session["events"][:-MAX_EVENTS_PER_SESSION]
            self._persist()
        # 文本里出现可展示的图片（[图片] 标记或存在的图片路径）时，追加独立图片事件。
        if kind in {"assistant", "action", "user"}:
            for image_path in _extract_image_paths(text):
                self._append_event(session_id, "image", image_path)

    def _is_cancelled(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._cancel_requested

    def _cancel_process(self, session_id: str) -> None:
        with self._lock:
            process = self._processes.pop(session_id, None)
        if process is not None and process.poll() is None:
            _terminate_process_tree(process)

    def _run_turn(self, session_id: str, user_text: str) -> None:
        if self._engine == "codex":
            self._run_turn_codex(session_id, user_text)
        elif self._engine == "deepseek":
            self._run_turn_deepseek(session_id, user_text)
        else:
            self._run_turn_claude(session_id, user_text)

    # ---------- claude-code 代理壳引擎（默认） ----------

    def _run_turn_claude(self, session_id: str, user_text: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            claude_session_id = session.get("claudeSessionId") if session else None
            snapshot = self._snapshot_provider()
        prompt = build_claude_prompt(user_text, snapshot)
        command = _claude_command(prompt, claude_session_id)
        try:
            process = subprocess.Popen(
                command,
                cwd=str(APP_ROOT),
                stdin=subprocess.PIPE if not CLAUDE_FAKE else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={**_claude_env(), "ZEM_MCP_CALLER_ID": session_id},
            )
        except Exception as error:
            self._append_event(session_id, "error", f"无法启动 claude cli：{error}")
            self._finish_turn(session_id, success=False)
            return
        if process.stdin is not None:
            try:
                process.stdin.write(prompt)
                process.stdin.close()
            except Exception as error:
                self._append_event(session_id, "error", f"写入 claude 标准输入失败：{error}")
        with self._lock:
            self._processes[session_id] = process
            # cancel 可能在注册前就到达（启动竞态），注册后在同一把锁下复查补杀。
            if session_id in self._cancel_requested and process.poll() is None:
                _terminate_process_tree(process)
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line or line.startswith("[claude-code:"):
                continue
            try:
                payload = json.loads(line)
            except Exception as parse_error:
                self._append_event(session_id, "action", f"{line[-2000:]} <parse-fail: {parse_error}>")
                continue
            if not isinstance(payload, dict):
                continue
            new_session_id = _extract_claude_session_id(payload)
            if new_session_id:
                with self._lock:
                    session = self._sessions.get(session_id)
                    if session is not None and not session.get("claudeSessionId"):
                        session["claudeSessionId"] = new_session_id
                        self._persist()
            for kind, text in _extract_claude_events(payload):
                self._append_event(session_id, kind, text)
        code = process.wait()
        with self._lock:
            self._processes.pop(session_id, None)
        self._finish_turn(session_id, success=code == 0, exit_code=code)

    # ---------- DeepSeek 直连引擎 ----------

    def _run_turn_deepseek(self, session_id: str, user_text: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            history: list[dict[str, Any]] = list(session.get("messages") or [])
        history.append({"role": "user", "content": user_text})
        system = build_system_prompt(self._snapshot_provider())

        outcome = "max_steps"
        for _step in range(MAX_LLM_STEPS):
            if self._is_cancelled(session_id):
                outcome = "cancelled"
                break
            try:
                tools = tools_for_model() if sem_mcp_client.available() else []
                response = _llm_chat(system, history, tools)
            except Exception as error:
                self._append_event(session_id, "error", f"LLM 请求失败：{error}")
                outcome = "failed"
                break
            message = dict(response["choices"][0]["message"])
            message.pop("reasoning_content", None)  # deepseek-reasoner 的思考内容不回传
            history.append(message)
            content = str(message.get("content") or "").strip()
            if content:
                self._append_event(session_id, "assistant", content)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                outcome = "done"
                break
            for tool_call in tool_calls:
                if self._is_cancelled(session_id):
                    outcome = "cancelled"
                    break
                function = tool_call.get("function") or {}
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except Exception:
                    arguments = {}
                result_text = self._execute_tool(session_id, str(function.get("name") or ""), arguments)
                history.append({"role": "tool", "tool_call_id": tool_call.get("id"), "content": result_text})
            if outcome == "cancelled":
                break
            with self._lock:
                images = self._mcp_images.pop(session_id, [])
            if images:
                history.append({"role": "user", "content": [
                    {"type": "text", "text": "本轮 MCP 采集图像；模拟图不代表真实设备结果。"}, *images]})
            history = _trim_history(history)
            with self._lock:
                session = self._sessions.get(session_id)
                if session is not None:
                    session["messages"] = history
                    self._persist()

        if outcome == "max_steps":
            self._append_event(session_id, "error", f"已达到单轮工具调用上限（{MAX_LLM_STEPS} 次），请拆分任务或开新会话")
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session["messages"] = history
                self._persist()
        self._finish_turn(session_id, success=outcome == "done")

    def _execute_tool(self, session_id: str, name: str, arguments: dict[str, Any]) -> str:
        if name in TOOL_MAP:
            if self._is_cancelled(session_id):
                return "会话已取消，未调用设备工具；已提交的硬件作业仍需单独查询状态"
            self._append_event(session_id, "action", f"MCP {name} {json.dumps(arguments, ensure_ascii=False)}")
            try:
                result = sem_mcp_client.call_tool(name, arguments, session_id)
                texts = []
                for content in result.get("content", []):
                    if content["type"] == "text":
                        texts.append(content["text"])
                    elif content["type"] == "image":
                        directory = self._sessions_file.parent / "mcp-images" / session_id
                        directory.mkdir(parents=True, exist_ok=True)
                        path = directory / f"{uuid.uuid4().hex}.png"
                        path.write_bytes(base64.b64decode(content["data"], validate=True))
                        self._append_event(session_id, "action", f"[图片] {path.resolve()}")
                        if os.environ.get("ZEM_VIBE_VISION_ENABLED") == "1":
                            with self._lock:
                                self._mcp_images.setdefault(session_id, []).append({"type": "image_url", "image_url": {
                                    "url": f"data:{content['mimeType']};base64,{content['data']}"}})
                        else:
                            texts.append("图片已展示给用户；当前模型未启用视觉输入，不能声称已看图。")
                text = "\n".join(texts)
                self._append_event(session_id, "action", text)
                return text
            except Exception as error:
                return f"MCP 调用失败：{error}。先核验原作业，不要重复发送动作。"
        if name != "run_vibe_action":
            return f"未知工具：{name}。聊天只允许 zem-cleanliness 与 zem-sem"
        message = "run_vibe_action 已停用。软件界面请用 zem-cleanliness（navigate / ui_*），电镜请用 zem-sem。"
        self._append_event(session_id, "action", message)
        return message

    # ---------- codex 经 CPA 启动与 MCP 工具调用 ----------

    def _run_turn_codex(self, session_id: str, user_text: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            codex_session_id = session.get("codexSessionId") if session else None
            snapshot = self._snapshot_provider()
        prompt = build_prompt(user_text, snapshot)
        command = _codex_command(prompt, codex_session_id)
        try:
            # 子进程输出统一 UTF-8：Windows 下 Python 子进程默认跟随本地编码（GBK），
            # 不注入会导致中文事件乱码。
            child_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
                         "ZEM_MCP_CALLER_ID": session_id}
            process = subprocess.Popen(
                command,
                cwd=str(APP_ROOT),
                stdin=subprocess.PIPE if not FAKE_CODEX else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=child_env,
            )
        except Exception as error:
            self._append_event(session_id, "error", f"无法启动 codex：{error}")
            self._finish_turn(session_id, success=False)
            return
        if process.stdin is not None:
            try:
                process.stdin.write(prompt)
                process.stdin.close()
            except Exception as error:
                self._append_event(session_id, "error", f"写入 Codex 标准输入失败：{error}")
        with self._lock:
            self._processes[session_id] = process
            # cancel 可能在注册前就到达（启动竞态），注册后在同一把锁下复查补杀。
            if session_id in self._cancel_requested and process.poll() is None:
                _terminate_process_tree(process)
        assert process.stdout is not None
        for line in _codex_output_lines(process):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception as parse_error:
                if line.startswith(("使用 provider:", "注入环境变量:", "启动 codex")):
                    continue
                self._append_event(session_id, "action", f"{line[-2000:]} <parse-fail: {parse_error}>")
                continue
            if not isinstance(payload, dict):
                continue
            new_session_id = _extract_session_id(payload)
            if new_session_id:
                with self._lock:
                    session = self._sessions.get(session_id)
                    if session is not None and not session.get("codexSessionId"):
                        session["codexSessionId"] = new_session_id
                        self._persist()
            for kind, text in _extract_events(payload):
                if kind != "debug" or os.environ.get("ZEM_VIBE_CHAT_DEBUG"):
                    self._append_event(session_id, kind, text)
        code = process.wait()
        with self._lock:
            self._processes.pop(session_id, None)
        self._finish_turn(session_id, success=code == 0, exit_code=code)

    # ---------- 收尾 ----------

    def _finish_turn(self, session_id: str, *, success: bool, exit_code: int | None = None) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session["status"] = "idle"
            cancelled = session_id in self._cancel_requested
            self._cancel_requested.discard(session_id)
            self._persist()
        if cancelled:
            self._append_event(session_id, "system", "任务已取消")
        elif not success:
            detail = f"（退出码 {exit_code}）" if exit_code is not None else ""
            self._append_event(session_id, "error", f"引擎异常退出{detail}")


MANAGER: ChatManager | None = None
MANAGER_LOCK = threading.Lock()


def get_manager(snapshot_provider: Callable[[], str] | None = None) -> ChatManager:
    global MANAGER
    with MANAGER_LOCK:
        if MANAGER is None:
            MANAGER = ChatManager(snapshot_provider)
        return MANAGER
