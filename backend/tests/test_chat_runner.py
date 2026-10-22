"""Vibe chat 后端防退化测试：假 codex / 假 DeepSeek 全链路，不依赖网络与真实模型。

覆盖：
- codex JSONL 事件解析（新旧协议）与会话 ID 提取
- ChatManager codex 引擎完整轮次：新会话事件流、落盘、resume 传参、并发拒绝、失败与取消
- ChatManager deepseek 引擎工具调用循环：助手文本、工具执行、历史落盘、失败与上限
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import chat_runner
from backend.chat_runner import (
    ChatError,
    ChatManager,
    _extract_claude_events,
    _extract_claude_session_id,
    _extract_events,
    _extract_image_paths,
    _extract_session_id,
    _trim_history,
    build_prompt,
    build_system_prompt,
)

FAKE_CODEX_SOURCE = '''
import json, os, sys, time

marker = os.environ.get("ZEM_FAKE_CODEX_MARKER")
if marker:
    with open(marker, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(sys.argv[1:], ensure_ascii=False) + "\\n")

mode = os.environ.get("ZEM_FAKE_CODEX_MODE", "ok")
print(json.dumps({"type": "thread.started", "thread_id": "fake-thread-1"}), flush=True)
if mode == "old_protocol":
    print(json.dumps({"id": 1, "msg": {"type": "session_configured", "session_id": "old-session-9"}}), flush=True)
    print(json.dumps({"id": 2, "msg": {"type": "agent_message", "message": "旧协议回复"}}), flush=True)
else:
    print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "收到，开始处理"}}), flush=True)
    print(json.dumps({"type": "item.completed", "item": {"type": "command_execution", "command": ["node", "e2e/vibe-actions.cjs", "detect"]}}), flush=True)
if mode == "fail":
    print(json.dumps({"type": "turn.failed", "error": {"message": "动作脚本失败"}}), flush=True)
    sys.exit(3)
if mode == "slow":
    time.sleep(30)
print(json.dumps({"type": "turn.completed"}), flush=True)
'''

USER_TEXT = "把这个 json 跑一遍光电颗粒检测"


def write_fake_codex(tmp_path: Path) -> Path:
    path = tmp_path / "fake_codex.py"
    path.write_text(FAKE_CODEX_SOURCE, encoding="utf-8")
    return path


def wait_for(condition, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = condition()
        if result:
            return result
        time.sleep(0.05)
    raise AssertionError("等待超时")


@pytest.fixture()
def codex_manager(tmp_path, monkeypatch):
    fake = write_fake_codex(tmp_path)
    monkeypatch.setattr(chat_runner, "FAKE_CODEX", str(fake))
    monkeypatch.setenv("ZEM_FAKE_CODEX_MARKER", str(tmp_path / "marker.jsonl"))
    return ChatManager(data_root=tmp_path, engine="codex")


def test_extract_events_new_protocol():
    events = _extract_events({"type": "item.completed", "item": {"type": "agent_message", "text": "你好"}})
    assert events == [("assistant", "你好")]
    events = _extract_events({"type": "item.completed", "item": {"type": "command_execution", "command": ["node", "x.cjs"]}})
    assert events == [("action", "$ node x.cjs")]
    assert _extract_events({"type": "turn.completed"}) == [("system", "codex 任务完成")]
    assert _extract_events({"type": "turn.failed", "error": {"message": "炸了"}}) == [("error", "炸了")]
    assert _extract_events({"type": "thread.started", "thread_id": "t"}) == []


def test_extract_events_old_protocol():
    events = _extract_events({"id": 1, "msg": {"type": "agent_message", "message": "旧协议回复"}})
    assert events == [("assistant", "旧协议回复")]
    assert _extract_session_id({"id": 1, "msg": {"type": "session_configured", "session_id": "old-session-9"}}) == "old-session-9"


def test_extract_session_id_variants():
    assert _extract_session_id({"type": "thread.started", "thread_id": "abc"}) == "abc"
    assert _extract_session_id({"type": "item.started", "item": {"id": "xyz"}}) is None
    assert _extract_session_id({"type": "turn.completed"}) is None


def test_build_prompt_contains_snapshot():
    prompt = build_prompt(USER_TEXT, "- 电镜已连接")
    assert "zem-cleanliness" in prompt and "zem-sem" in prompt
    assert "e2e/vibe-actions.cjs" in prompt and "禁止" in prompt
    assert "/sem/navigation" in prompt
    assert "- 电镜已连接" in prompt
    assert USER_TEXT in prompt


def test_prompts_and_tools_image_routing_three_ways():
    """取图三档：导航图 / 当前画面 GetImg / 明确新拍 sem_capture；等待上限 20 秒。"""
    from backend.chat_runner import (
        CLAUDE_PREAMBLE_TEMPLATE,
        CODEX_PREAMBLE_TEMPLATE,
        IMAGE_ROUTING,
        SYSTEM_PROMPT_TEMPLATE,
    )
    from backend.sem_tools import TOOL_MAP

    assert "sem_get_navigation" in IMAGE_ROUTING
    assert "禁止用 GetImg" in IMAGE_ROUTING or "禁止用 GetImg 或 sem_capture" in IMAGE_ROUTING
    assert "最多 20" in IMAGE_ROUTING
    assert "雪花" in IMAGE_ROUTING
    assert "display_path" in IMAGE_ROUTING
    for text in (CODEX_PREAMBLE_TEMPLATE, CLAUDE_PREAMBLE_TEMPLATE, SYSTEM_PROMPT_TEMPLATE):
        assert IMAGE_ROUTING in text
        assert "GetImg" in text and "sem_capture" in text and "sem_get_navigation" in text
    nav = TOOL_MAP["sem_get_navigation"]["description"]
    assert "导航图" in nav and "禁止用 GetImg" in nav
    assert "GetImg" in TOOL_MAP["sem_rpc_read"]["description"]
    assert "导航图优先 sem_get_navigation" in TOOL_MAP["sem_rpc_read"]["description"]
    assert "要导航图改用 sem_get_navigation" in TOOL_MAP["sem_capture"]["description"]
    assert "冒充导航图" in TOOL_MAP["sem_rpc_call"]["description"]
    wait = TOOL_MAP["sem_wait_job"]
    assert wait["inputSchema"]["properties"]["timeout_seconds"]["maximum"] == 20
    assert "最大值 20" in wait["description"]


@pytest.mark.parametrize("quoted", [False, True])
def test_image_marker_preserves_spaces_in_navigation_path(tmp_path, quoted):
    image = tmp_path / "ZEM Cleanliness" / "导航图 preview.png"
    image.parent.mkdir()
    image.write_bytes(b"existing image")
    rendered = f'"{image}"' if quoted else str(image)
    text = f"导航图如下：\n[图片] {rendered}\n图像读取完成。"
    assert _extract_image_paths(text) == [str(image.resolve())]
    manager = ChatManager(data_root=tmp_path / "chat", engine="codex")
    session = manager.create_session()
    manager._append_event(session["id"], "action", text)
    events = manager.get_session(session["id"])["events"]
    assert any(event["kind"] == "image" and event["text"] == str(image.resolve()) for event in events)


@pytest.mark.parametrize("text_template", [
    "[图片] {path}",
    "[图片]<{path}>",
    "[图片](<{path}>)",
    "[图片]({path})",
    "已采集。\n[图片](<{path}>)\n分辨率 512×512。",
])
def test_image_marker_accepts_markdown_wrappers(tmp_path, text_template):
    image = tmp_path / "preview.png"
    image.write_bytes(b"png-bytes")
    text = text_template.format(path=str(image).replace("\\", "/"))
    assert _extract_image_paths(text) == [str(image.resolve())]
    manager = ChatManager(data_root=tmp_path / "chat", engine="codex")
    session = manager.create_session()
    manager._append_event(session["id"], "assistant", text)
    events = manager.get_session(session["id"])["events"]
    assert any(event["kind"] == "image" and event["text"] == str(image.resolve()) for event in events)


def test_restarted_chat_keeps_event_cursor_ahead_of_all_saved_sessions(tmp_path):
    manager = ChatManager(data_root=tmp_path, engine="codex")
    first = manager.create_session()
    second = manager.create_session()
    manager._append_event(first["id"], "assistant", "上一轮")
    old_cursor = manager.get_session(first["id"])["cursor"]
    for _ in range(4):
        manager._append_event(second["id"], "assistant", "其他会话")
    maximum = manager.get_session(second["id"])["cursor"]
    restored = ChatManager(data_root=tmp_path, engine="codex")
    restored._append_event(first["id"], "assistant", "重启后的回复")
    changes = restored.get_session(first["id"], after=old_cursor)
    assert [event["text"] for event in changes["events"]] == ["重启后的回复"]
    assert changes["cursor"] > maximum


def test_frozen_codex_uses_external_python_for_cpa_launcher(monkeypatch):
    monkeypatch.setattr(chat_runner, "FAKE_CODEX", "")
    monkeypatch.setattr(chat_runner, "CODEX_CMD", r"C:\path\to\cx_launcher.py")
    monkeypatch.setattr(chat_runner.sys, "frozen", True, raising=False)
    command = chat_runner._codex_command("test", None)
    assert command[:2] == ["py", chat_runner.CODEX_CMD]
    assert command[-4:] == ["exec", "--skip-git-repo-check", "--json", "-"]


def test_full_turn_and_resume(codex_manager: ChatManager, tmp_path):
    session = codex_manager.create_session("测试会话")
    codex_manager.send_message(session["id"], USER_TEXT)

    view = wait_for(lambda: (
        codex_manager.get_session(session["id"]) if codex_manager.get_session(session["id"])["status"] == "idle" else None
    ))
    kinds = [event["kind"] for event in view["events"]]
    assert kinds[0] == "user"
    assert "assistant" in kinds and "action" in kinds
    assert view["codexSessionId"] == "fake-thread-1"
    assert view["events"][0]["text"] == USER_TEXT
    assert any("detect" in event["text"] for event in view["events"] if event["kind"] == "action")

    # 落盘
    persisted = json.loads((tmp_path / "sessions.json").read_text(encoding="utf-8"))
    assert persisted["sessions"][0]["codexSessionId"] == "fake-thread-1"

    # 第二条消息走 resume
    marker = tmp_path / "marker.jsonl"
    assert marker.is_file(), "假 codex 没有留下调用记录"
    first_call = json.loads(marker.read_text(encoding="utf-8").splitlines()[0])
    assert "--resume" not in first_call
    codex_manager.send_message(session["id"], "继续")

    def _second_call():
        if codex_manager.get_session(session["id"])["status"] != "idle":
            return None
        lines = marker.read_text(encoding="utf-8").splitlines()
        return json.loads(lines[1]) if len(lines) >= 2 else None

    second_call = wait_for(_second_call)
    assert "--resume" in second_call and "fake-thread-1" in second_call


def test_send_while_running_rejected(codex_manager: ChatManager, monkeypatch, tmp_path):
    monkeypatch.setenv("ZEM_FAKE_CODEX_MODE", "slow")
    session = codex_manager.create_session()
    codex_manager.send_message(session["id"], USER_TEXT)
    wait_for(lambda: codex_manager.get_session(session["id"])["status"] == "running")
    with pytest.raises(ChatError) as excinfo:
        codex_manager.send_message(session["id"], "再来一次")
    assert excinfo.value.status == 409
    codex_manager.cancel(session["id"])
    wait_for(lambda: codex_manager.get_session(session["id"])["status"] == "idle")


def test_failing_codex_records_error(codex_manager: ChatManager, monkeypatch):
    monkeypatch.setenv("ZEM_FAKE_CODEX_MODE", "fail")
    session = codex_manager.create_session()
    codex_manager.send_message(session["id"], USER_TEXT)
    view = wait_for(lambda: (
        codex_manager.get_session(session["id"]) if codex_manager.get_session(session["id"])["status"] == "idle" else None
    ))
    errors = [event for event in view["events"] if event["kind"] == "error"]
    assert any("动作脚本失败" in event["text"] for event in errors)
    assert any("退出码 3" in event["text"] for event in errors)
    assert view["status"] == "idle"


@pytest.mark.parametrize("exit_code", [0, 7])
def test_codex_finishes_when_exited_launcher_child_still_holds_stdout(tmp_path, monkeypatch, exit_code):
    launcher = tmp_path / "launcher-with-background-child.py"
    child_pid = tmp_path / "child.pid"
    launcher.write_text('''
import json, os, subprocess, sys
from pathlib import Path
sys.stdin.read()
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
Path(os.environ['ZEM_TEST_CHILD_PID']).write_text(str(child.pid))
print(json.dumps({'type':'thread.started','thread_id':'background-thread'}), flush=True)
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'中文回复已完成'}}), flush=True)
print(json.dumps({'type':'turn.completed'}), flush=True)
sys.exit(int(os.environ['ZEM_TEST_LAUNCHER_EXIT']))
''', encoding="utf-8")
    monkeypatch.setattr(chat_runner, "FAKE_CODEX", "")
    monkeypatch.setattr(chat_runner, "CODEX_CMD", str(launcher))
    monkeypatch.setenv("ZEM_TEST_CHILD_PID", str(child_pid))
    monkeypatch.setenv("ZEM_TEST_LAUNCHER_EXIT", str(exit_code))
    manager = ChatManager(data_root=tmp_path / "chat", engine="codex")
    session = manager.create_session("后台进程继承输出管道")
    try:
        manager.send_message(session["id"], "回复")
        view = wait_for(lambda: manager.get_session(session["id"])
                        if manager.get_session(session["id"])["status"] == "idle" else None, timeout=5)
        assert any(event["text"] == "中文回复已完成" for event in view["events"])
        assert view["codexSessionId"] == "background-thread"
        errors = [event["text"] for event in view["events"] if event["kind"] == "error"]
        assert bool(errors) == bool(exit_code)
        if exit_code:
            assert any("退出码 7" in text for text in errors)
    finally:
        if manager.get_session(session["id"])["status"] == "running":
            manager.cancel(session["id"])
        if child_pid.exists():
            pid = int(child_pid.read_text())
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            else:
                import os, signal
                os.kill(pid, signal.SIGKILL)


def test_unknown_session_404(codex_manager: ChatManager):
    with pytest.raises(ChatError) as excinfo:
        codex_manager.get_session("no-such-session")
    assert excinfo.value.status == 404


# ---------- deepseek 直连引擎（假 LLM + 假动作脚本） ----------

FAKE_ACTION_SOURCE = '''
import json, os, sys

mode = os.environ.get("ZEM_FAKE_ACTION_MODE", "ok")
args = sys.argv[1:]
if "--json" not in args:
    print("missing --json")
    sys.exit(2)
if mode == "fail":
    print(json.dumps({"ok": False, "error": "电镜未连接"}, ensure_ascii=False))
    sys.exit(1)
print(json.dumps({"ok": True, "action": "detect", "taskId": "fake-task-1", "status": "COMPLETED",
                  "reports": {"pdf": "D:/tmp/report.pdf"}}, ensure_ascii=False))
'''


@pytest.fixture()
def deepseek_manager(tmp_path, monkeypatch):
    action = tmp_path / "fake_action.py"
    action.write_text(FAKE_ACTION_SOURCE, encoding="utf-8")
    monkeypatch.setattr(chat_runner, "_vibe_action_command", lambda arguments: [sys.executable, str(action), "--json", str(arguments.get("json") or "")])
    monkeypatch.setenv("ZEM_FAKE_ACTION_MARKER", str(tmp_path / "action-marker.jsonl"))
    return ChatManager(data_root=tmp_path, engine="deepseek")


def make_llm(plans, calls):
    """按次数返回预置响应；plans[i] 是第 i 次调用的响应结构。"""

    def fake_llm(system, messages, tools):
        calls.append([dict(message) for message in messages])
        index = min(len(calls) - 1, len(plans) - 1)
        return plans[index]

    return fake_llm


def test_deepseek_tool_loop_full_turn(deepseek_manager: ChatManager, tmp_path, monkeypatch):
    sample_json = tmp_path / "sample.json"
    sample_json.write_text("{}", encoding="utf-8")
    calls: list = []
    plans = [
        {"choices": [{"message": {"role": "assistant", "content": "好的，我来运行检测", "tool_calls": [
            {"id": "call-1", "type": "function",
             "function": {"name": "run_vibe_action", "arguments": json.dumps({"json": str(sample_json)})}}]}}]},
        {"choices": [{"message": {"role": "assistant", "content": "检测完成，报告在 D:/tmp/report.pdf"}}]},
    ]
    monkeypatch.setattr(chat_runner, "_llm_chat", make_llm(plans, calls))
    session = deepseek_manager.create_session("直连会话")
    deepseek_manager.send_message(session["id"], USER_TEXT)

    view = wait_for(lambda: (
        deepseek_manager.get_session(session["id"]) if deepseek_manager.get_session(session["id"])["status"] == "idle" else None
    ))
    kinds = [event["kind"] for event in view["events"]]
    assert kinds[0] == "user"
    assert "好的，我来运行检测" in [event["text"] for event in view["events"] if event["kind"] == "assistant"]
    assert any("已停用" in event["text"] for event in view["events"] if event["kind"] == "action")
    assert "检测完成，报告在 D:/tmp/report.pdf" in [event["text"] for event in view["events"] if event["kind"] == "assistant"]
    assert kinds[-1] == "assistant"

    # 历史落盘：user → assistant(tool_calls) → tool → assistant
    persisted = json.loads((tmp_path / "sessions.json").read_text(encoding="utf-8"))
    messages = persisted["sessions"][0]["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[2]["tool_call_id"] == "call-1"
    assert "已停用" in messages[2]["content"]


def test_deepseek_missing_json_path_reports_back(deepseek_manager: ChatManager, monkeypatch):
    """模型没传 json 或文件不存在时，工具返回参数错误，模型能把错误转述给用户。"""
    calls: list = []
    plans = [
        {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call-1", "type": "function",
             "function": {"name": "run_vibe_action", "arguments": json.dumps({"json": "D:/not/exist.json"})}}]}}]},
        {"choices": [{"message": {"role": "assistant", "content": "文件不存在，请确认路径"}}]},
    ]
    monkeypatch.setattr(chat_runner, "_llm_chat", make_llm(plans, calls))
    session = deepseek_manager.create_session()
    deepseek_manager.send_message(session["id"], USER_TEXT)
    view = wait_for(lambda: (
        deepseek_manager.get_session(session["id"]) if deepseek_manager.get_session(session["id"])["status"] == "idle" else None
    ))
    assert any("文件不存在，请确认路径" in event["text"] for event in view["events"] if event["kind"] == "assistant")
    assert view["status"] == "idle"


def test_deepseek_llm_error_records_error(deepseek_manager: ChatManager, monkeypatch):
    def broken_llm(system, messages, tools):
        raise RuntimeError("LLM HTTP 401：bad key")

    monkeypatch.setattr(chat_runner, "_llm_chat", broken_llm)
    session = deepseek_manager.create_session()
    deepseek_manager.send_message(session["id"], USER_TEXT)
    view = wait_for(lambda: (
        deepseek_manager.get_session(session["id"]) if deepseek_manager.get_session(session["id"])["status"] == "idle" else None
    ))
    assert any("LLM HTTP 401" in event["text"] for event in view["events"] if event["kind"] == "error")
    assert view["status"] == "idle"


def test_deepseek_system_prompt_contains_snapshot_and_tools():
    prompt = build_system_prompt("- 电镜已连接")
    assert "zem-cleanliness" in prompt and "zem-sem" in prompt
    assert "run_vibe_action 已停用" in prompt
    assert "- 电镜已连接" in prompt


def test_trim_history_keeps_tool_pairing():
    history = [{"role": "user", "content": "1"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
               {"role": "tool", "tool_call_id": "a", "content": "r"}, {"role": "assistant", "content": "done"}]
    trimmed = _trim_history(history, limit=2)
    assert trimmed[0]["role"] != "tool"
    assert trimmed[-1]["content"] == "done"
    assert _trim_history(history[:2], limit=10) == history[:2]


def test_extract_image_paths(tmp_path):
    image = tmp_path / "nav.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    missing = tmp_path / "missing.png"
    text = f"导航图如下\n[图片] {image}\n另附 {missing} 和 https://x/y.png"
    paths = _extract_image_paths(text)
    assert paths == [str(image.resolve())]
    assert _extract_image_paths("没有图片") == []


# ---------- claude-code 代理壳引擎（假 CLI 流式输出） ----------

FAKE_CLAUDE_SOURCE = '''
import json, os, sys, time

marker = os.environ.get("ZEM_FAKE_CLAUDE_MARKER")
if marker:
    with open(marker, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(sys.argv[1:], ensure_ascii=False) + "\\n")

mode = os.environ.get("ZEM_FAKE_CLAUDE_MODE", "ok")
print(json.dumps({"type": "system", "subtype": "init", "session_id": "claude-session-1"}), flush=True)
if mode == "old_session":
    print(json.dumps({"type": "system", "subtype": "init", "session_id": "claude-session-OLD"}), flush=True)
print(json.dumps({"type": "assistant", "message": {"content": [
    {"type": "text", "text": "我先看看设置文件"},
    {"type": "tool_use", "name": "Read", "input": {"file_path": "settings.json"}},
]}}, ensure_ascii=False), flush=True)
print(json.dumps({"type": "user", "message": {"content": [
    {"type": "tool_result", "content": [{"type": "text", "text": "microscopeHost=192.168.1.10"}]}
]}}, ensure_ascii=False), flush=True)
if mode == "fail":
    print(json.dumps({"type": "result", "subtype": "error_during_execution", "result": "boom"}), flush=True)
    sys.exit(1)
print(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "电镜 IP 是 192.168.1.10"}]}}, ensure_ascii=False), flush=True)
print(json.dumps({"type": "result", "subtype": "success", "result": "电镜 IP 是 192.168.1.10"}), flush=True)
'''


@pytest.fixture()
def claude_manager(tmp_path, monkeypatch):
    fake = tmp_path / "fake_claude.py"
    fake.write_text(FAKE_CLAUDE_SOURCE, encoding="utf-8")
    monkeypatch.setattr(chat_runner, "CLAUDE_FAKE", str(fake))
    monkeypatch.setenv("ZEM_FAKE_CLAUDE_MARKER", str(tmp_path / "claude-marker.jsonl"))
    return ChatManager(data_root=tmp_path, engine="claude")


def test_extract_claude_events_and_session_id():
    events = _extract_claude_events({"type": "assistant", "message": {"content": [
        {"type": "text", "text": "你好"}, {"type": "tool_use", "name": "Bash", "input": {"command": "dir"}}]}})
    assert events == [("assistant", "你好"), ("action", '⚙ Bash {"command": "dir"}')]
    assert _extract_claude_events({"type": "system", "subtype": "init", "session_id": "s"}) == []
    assert _extract_claude_session_id({"type": "system", "subtype": "init", "session_id": "abc"}) == "abc"
    assert _extract_claude_session_id({"type": "assistant"}) is None


def test_claude_full_turn_and_resume(claude_manager: ChatManager, tmp_path):
    session = claude_manager.create_session("壳引擎会话")
    claude_manager.send_message(session["id"], "电镜和能谱是一个机器吗？IP 是多少？")

    view = wait_for(lambda: (
        claude_manager.get_session(session["id"]) if claude_manager.get_session(session["id"])["status"] == "idle" else None
    ))
    texts = [event["text"] for event in view["events"]]
    assert "我先看看设置文件" in texts
    assert any("⚙ Read" in text for text in texts)
    assert "电镜 IP 是 192.168.1.10" in texts
    assert view["claudeSessionId"] == "claude-session-1"

    # 续会话：第二次调用带 --resume
    marker = tmp_path / "claude-marker.jsonl"
    claude_manager.send_message(session["id"], "继续")
    wait_for(lambda: (claude_manager.get_session(session["id"])["status"] == "idle") and len(marker.read_text(encoding="utf-8").splitlines()) >= 2 or None)
    second_line = json.loads(marker.read_text(encoding="utf-8").splitlines()[1])
    assert "--resume" in second_line and "claude-session-1" in second_line


def test_claude_failure_records_error(claude_manager: ChatManager, monkeypatch):
    monkeypatch.setenv("ZEM_FAKE_CLAUDE_MODE", "fail")
    session = claude_manager.create_session()
    claude_manager.send_message(session["id"], "去失败")
    view = wait_for(lambda: (
        claude_manager.get_session(session["id"]) if claude_manager.get_session(session["id"])["status"] == "idle" else None
    ))
    assert any("error_during_execution" in event["text"] for event in view["events"] if event["kind"] == "error")
    assert view["status"] == "idle"
