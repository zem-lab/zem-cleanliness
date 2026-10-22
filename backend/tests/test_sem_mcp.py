from __future__ import annotations

import asyncio
from datetime import timedelta
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from backend import chat_runner, sem_mcp_client, server
from backend.cleanliness_core.device_control import DeviceBusy, DeviceCoordinator
from backend.sem_control import RealMicroscope, SemControl, SemError, SimulatedMicroscope
from backend.sem_mcp import BackendTransport
from backend.sem_rpc_catalog import WRITES


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    control = SemControl(tmp_path / "mcp", SimulatedMicroscope(), DeviceCoordinator(), allow_capture=True,
                         bounds=[-100, -100, 100, 100], max_move_um=100, poll_interval=.001, allow_rpc=WRITES)
    monkeypatch.setattr(server, "MCP_CONTROL", control)
    monkeypatch.setattr(server, "MCP_TOKEN", "test-only-token")
    monkeypatch.setattr(server, "log_event", lambda *a, **k: None)
    manager = server.JobManager(control.coordinator)
    manager.allow_hardware_jobs = False
    monkeypatch.setattr(server, "JOBS", manager)
    http = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{http.server_address[1]}"
    monkeypatch.setenv("ZEM_MCP_BACKEND_URL", url)
    monkeypatch.setenv("ZEM_MCP_TOKEN", "test-only-token")
    yield control, url
    http.shutdown()
    http.server_close()
    worker.join(5)
    control.close()


def test_stdio_client_discovers_and_runs_complete_protocol(runtime):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    control, _ = runtime
    async def check():
        command, args = sem_mcp_client.bridge_command()
        env = {**os.environ, "ZEM_MCP_CALLER_ID": "protocol-test", "PYTHONUTF8": "1"}
        async with stdio_client(StdioServerParameters(command=command, args=args, env=env)) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=10)) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                assert {tool.name for tool in tools} == set(control.capabilities()["tools"])
                assert not control.device.commands
                capabilities = await session.call_tool("sem_get_capabilities", {})
                assert capabilities.structuredContent["mode"] == "simulated"
                arguments = {"request_id": "protocol-1", "x_um": 10, "y_um": 20}
                result = await session.call_tool("sem_move_and_capture", arguments)
                job = result.structuredContent
                for _ in range(100):
                    result = await session.call_tool("sem_get_job", {"job_id": job["job_id"]})
                    if result.structuredContent["status"] == "completed":
                        break
                    await asyncio.sleep(.01)
                assert result.structuredContent["status"] == "completed", result
                image_id = result.structuredContent["result"]["image"]["image_id"]
                picture = await session.call_tool("sem_get_image", {"image_id": image_id})
                assert not picture.isError
                assert any(content.type == "image" for content in picture.content)
                repeated = await session.call_tool("sem_move_and_capture", arguments)
                assert repeated.structuredContent["job_id"] == job["job_id"]
                bad = await session.call_tool("sem_move_xy", {"request_id": "bad", "x_um": 9999, "y_um": 0})
                assert bad.isError and bad.structuredContent["error"]["code"] == "OUT_OF_RANGE"
    asyncio.run(check())
    assert control.device.commands.count("move") == 1
    assert control.device.commands.count("capture") == 1


def test_expanded_tools_through_real_stdio_protocol(runtime):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    control, _ = runtime
    async def check():
        command, args = sem_mcp_client.bridge_command()
        env = {**os.environ, "ZEM_MCP_CALLER_ID": "expanded-protocol", "PYTHONUTF8": "1"}
        async with stdio_client(StdioServerParameters(command=command, args=args, env=env)) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=10)) as session:
                await session.initialize()
                snapshot = await session.call_tool("sem_get_snapshot", {})
                assert snapshot.structuredContent["state_text"]["gun_state"] == "开启"
                assert snapshot.structuredContent["position_um"] == [0, 0]
                entries = await session.call_tool("sem_list_rpc", {"kind": "write"})
                assert len(entries.structuredContent["commands"]) == 34
                requests = await session.call_tool('sem_list_rpc', {'kind':'request'})
                assert len(requests.structuredContent['commands']) == 17
                photo_average = next(e for e in requests.structuredContent['commands'] if e['command']=='SetAvLenth')
                assert photo_average['completion'] == 'acknowledgement_only'
                arguments = {"request_id": "one-write", "command": "SetMag", "values": [5000]}
                job = (await session.call_tool("sem_rpc_execute", arguments)).structuredContent
                result = await session.call_tool("sem_wait_job", {"job_id": job["job_id"], "timeout_seconds": 5})
                assert result.structuredContent["status"] == "completed", result
                assert result.structuredContent["result"]["rpc"]["readback"] == [5000]
                readback = await session.call_tool("sem_rpc_read", {"command": "GetMag"})
                assert readback.structuredContent["values"] == [5000]
                duplicate = await session.call_tool("sem_rpc_execute", arguments)
                assert duplicate.structuredContent["job_id"] == job["job_id"]
                invalid = await session.call_tool("sem_rpc_execute", {**arguments, "values": [0]})
                assert invalid.isError
                unsupported = await session.call_tool("sem_rpc_read", {"command": "ZemRpc"})
                assert unsupported.isError
    asyncio.run(check())
    assert control.device.commands.count("SetMag") == 1


def test_stdio_shutdown_preserves_process_streams(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    import mcp.server.stdio
    from backend import sem_mcp
    owned = []
    @asynccontextmanager
    async def sdk(stdin=None, stdout=None):
        assert stdin is not None and stdout is not None
        owned.extend([stdin, stdout])
        yield None, None
        # SDK stream cleanup must never close the process' own handles.
        await stdin.aclose()
        await stdout.aclose()
    class Server:
        def create_initialization_options(self): return None
        async def run(self, *_): pass
    monkeypatch.setattr(mcp.server.stdio, 'stdio_server', sdk)
    monkeypatch.setattr(sem_mcp, 'create_server', lambda _: Server())
    with (tmp_path/'input').open('w+') as incoming, (tmp_path/'output').open('w+') as outgoing:
        monkeypatch.setattr(sys, 'stdin', incoming)
        monkeypatch.setattr(sys, 'stdout', outgoing)
        asyncio.run(sem_mcp.run_stdio(object()))
        assert not incoming.closed and not outgoing.closed
        outgoing.write('still open')
        outgoing.flush()
    assert all(stream.closed for stream in owned)


@pytest.mark.parametrize("url", ["http://192.168.0.15:61557", "https://127.0.0.1:8765", "http://localhost:8765",
                                 "http://x:y@127.0.0.1:8765", "http://127.0.0.1:8765/other", "http://127.0.0.1:8765?q=x"])
def test_bridge_rejects_remote_or_ambiguous_backend(url):
    with pytest.raises(ValueError):
        BackendTransport(url, "test")


def test_backend_requires_local_credential_and_rejects_browser_origin(runtime):
    control, url = runtime
    body = json.dumps({"name": "sem_get_status", "arguments": {}, "caller": "test"}).encode()
    for headers in [{}, {"Authorization": "Bearer wrong"},
                    {"Authorization": "Bearer test-only-token", "Origin": "http://untrusted.example"}]:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(url + "/mcp/call", data=body, headers=headers), timeout=3)
        assert error.value.code == 403
    assert not control.device.commands


def test_assistant_runtime_exposes_configuration_presence_without_credentials(runtime, monkeypatch):
    _, url = runtime
    monkeypatch.setattr(chat_runner, "ENGINE", "claude")
    monkeypatch.setattr(chat_runner, "CLAUDE_BASE_URL", "https://api.deepseek.com/anthropic")
    monkeypatch.setattr(chat_runner, "CLAUDE_AUTH_TOKEN", "sensitive-test-value")
    payload = urllib.request.urlopen(url + "/chat/runtime", timeout=3).read().decode()
    assert "sensitive-test-value" not in payload and "test-only-token" not in payload
    result = json.loads(payload)
    assert result["credentialConfigured"] is True
    assert result["mcp"]["mode"] == "simulated"
    monkeypatch.setattr(chat_runner, "CLAUDE_AUTH_TOKEN", "")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert chat_runner.model_configuration()["credentialConfigured"] is False


def test_simulated_runtime_never_probes_real_devices_or_starts_desktop_hardware(runtime, monkeypatch):
    monkeypatch.setattr(server, "_microscope_status_bounded", lambda *a: pytest.fail("real status probe"))
    assert all("模拟" in row["detail"] for row in server.device_status())
    with pytest.raises(ValueError, match="模拟"):
        server.JOBS.create("test", "analysis")
    with pytest.raises(ValueError, match="模拟"):
        server.fetch_and_detect_ab("test")


def test_builtin_assistant_calls_actual_mcp_and_displays_image(runtime, tmp_path):
    manager = chat_runner.ChatManager(data_root=tmp_path / "chat", engine="deepseek")
    session = manager.create_session("MCP 设备测试")
    session_id = session["id"]
    result = json.loads(manager._execute_tool(session_id, "sem_capture", {"request_id": "chat-image"}))
    for _ in range(100):
        job = json.loads(manager._execute_tool(session_id, "sem_get_job", {"job_id": result["job_id"]}))
        if job["status"] == "completed":
            break
        time.sleep(.01)
    assert job["status"] == "completed"
    text = manager._execute_tool(session_id, "sem_get_image", {"image_id": job["result"]["image"]["image_id"]})
    view = manager.get_session(session_id)
    assert any(event["kind"] == "image" for event in view["events"])
    assert "未启用视觉" in text


def test_claude_engine_receives_mcp_config_without_embedding_secret(runtime, tmp_path, monkeypatch):
    monkeypatch.setattr(chat_runner, "CLAUDE_FAKE", "")
    monkeypatch.setattr(chat_runner, "DATA_ROOT", tmp_path / "chat")
    monkeypatch.setattr(chat_runner, "_resolve_claude_command", lambda: "claude")
    command = chat_runner._claude_command("test", None)
    config_path = command[command.index("--mcp-config") + 1]
    from pathlib import Path
    config_text = Path(config_path).read_text(encoding="utf-8")
    assert "zem-sem" in config_text and "--mcp-stdio" in config_text
    assert "zem-cleanliness" in config_text and "server.cjs" in config_text
    assert list(json.loads(config_text)["mcpServers"]) == ["zem-sem", "zem-cleanliness"]
    assert "test-only-token" not in config_text


def test_codex_cpa_preserves_stdin_session_identity_and_mcp_images(runtime, tmp_path, monkeypatch):
    control, _ = runtime
    launcher = tmp_path / "cpa-launcher.py"
    marker = tmp_path / "cpa-calls.jsonl"
    launcher.write_text(r'''
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.getcwd())
from backend.sem_mcp_client import call_tool
caller = os.environ['ZEM_MCP_CALLER_ID']
with open(os.environ['ZEM_TEST_CPA_MARKER'], 'a', encoding='utf-8') as handle:
    handle.write(json.dumps({'args':sys.argv[1:], 'prompt':sys.stdin.read(), 'caller':caller}) + '\n')
print(json.dumps({'type':'thread.started', 'thread_id':'codex-' + caller}), flush=True)
def call(name, args):
    result = call_tool(name, args, caller)
    texts = [c for c in result.get('content',[]) if c['type']=='text']
    print(json.dumps({'type':'item.completed', 'item':{'type':'mcp_tool_call', 'server':'zem-sem',
          'tool':name, 'result':{'content':texts}}}), flush=True)
    return json.loads(texts[0]['text'])
job = call('sem_capture', {'request_id':'same-request-in-each-session'})
for _ in range(10):
    job = call('sem_get_job', {'job_id':job['job_id']})
    if job['status']=='completed': break
    time.sleep(.02)
assert job['status']=='completed', job
call('sem_get_image', {'image_id':job['result']['image']['image_id']})
print(json.dumps({'type':'item.completed', 'item':{'type':'agent_message','text':'模拟采图已完成'}}), flush=True)
print(json.dumps({'type':'turn.completed'}), flush=True)
''', encoding="utf-8")
    monkeypatch.setattr(chat_runner, "FAKE_CODEX", "")
    monkeypatch.setattr(chat_runner, "CODEX_CMD", str(launcher))
    monkeypatch.setenv("ZEM_TEST_CPA_MARKER", str(marker))
    manager = chat_runner.ChatManager(data_root=tmp_path / "chat", engine="codex")
    first = manager.create_session("会话一")
    second = manager.create_session("会话二")
    prompts = ["第一行\n第二行，含引号\"与 $(literal)", "续聊", "不同会话"]
    for session, prompt in zip([first, first, second], prompts):
        manager._run_turn_codex(session["id"], prompt)
        events = manager.get_session(session["id"])["events"]
        assert not [event for event in events if event["kind"] == "error"], events
        assert any(event["kind"] == "image" and Path(event["text"]).is_file() for event in events)
        assert any("MCP zem-sem/sem_capture" in event["text"] for event in events)
    assert control.device.commands.count("capture") == 2
    calls = [json.loads(line) for line in marker.read_text(encoding="utf-8").splitlines()]
    assert [call["caller"] for call in calls] == [first["id"], first["id"], second["id"]]
    for call, prompt in zip(calls, prompts):
        assert prompt in call["prompt"] and prompt not in call["args"]
        assert call["args"][-1] == "-"
        assert "mcp_servers.zem-sem.required=true" in call["args"]
        assert 'mcp_servers.zem-sem.default_tools_approval_mode="approve"' in call["args"]
        assert "mcp_servers.zem-cleanliness.required=true" in call["args"]
        assert "server.cjs" in " ".join(call["args"])
        assert "ZEM_MCP_TOKEN" in " ".join(call["args"]) and "test-only-token" not in " ".join(call["args"])
    resumed = calls[1]["args"]
    assert resumed.index("--cd") < resumed.index("exec") < resumed.index("resume")
    assert "codex-" + first["id"] in resumed


def test_codex_runtime_reports_cpa_launcher_without_deepseek_credentials(runtime, tmp_path, monkeypatch):
    _, url = runtime
    launcher = tmp_path / "cpa.py"
    launcher.write_text("# test launcher", encoding="utf-8")
    monkeypatch.setattr(chat_runner, "ENGINE", "codex")
    monkeypatch.setattr(chat_runner, "CODEX_CMD", str(launcher))
    monkeypatch.setattr(chat_runner, "CODEX_PROVIDER", "cpa")
    payload = urllib.request.urlopen(url + "/chat/runtime", timeout=3).read().decode()
    result = json.loads(payload)
    assert result["engine"] == "codex" and result["provider"] == "cpa"
    assert result["launcherConfigured"] is True and result["credentialConfigured"] is None
    assert "test-only-token" not in payload


def test_codex_mcp_error_event_is_visible_without_image_bytes():
    events = chat_runner._extract_events({"type": "item.completed", "item": {
        "type": "mcp_tool_call", "server": "zem-sem", "tool": "sem_get_status",
        "error": {"message": "backend unavailable"},
        "result": {"content": [{"type": "image", "data": "not-for-logs"}]}}})
    assert any(kind == "error" and "backend unavailable" in text for kind, text in events)
    assert "not-for-logs" not in str(events)


@pytest.fixture
def endpoint_control(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(server, "MCP_CONTROL", None)
    server.save_app_settings({"microscopeHost": "192.168.0.15", "microscopePort": 61557})
    device = RealMicroscope("192.168.0.15", 61557)
    control = SemControl(tmp_path / "endpoint-mcp", device, DeviceCoordinator(), allow_capture=True,
                         bounds=[-100, -100, 100, 100], max_move_um=100,
                         allow_rpc=['SetMag'], rpc_limits={'SetMag': [[100, 5000]]})
    monkeypatch.setattr(server, "MCP_CONTROL", control)
    yield control
    control.close()


def test_saving_microscope_address_updates_mcp_navigation_target(endpoint_control, monkeypatch):
    control = endpoint_control
    queries = []
    class Reader:
        def read(self, command):
            queries.append((control.device.host, control.device.port, command))
            return {"StatusCheck": "Idle", "GetStagePos": (0., 0.), "GetMag": 500.,
                    "IsChangingSamp": False,
                    "GetScanInfo": {"fieldSizeUm": 254., "rotationDeg": 0.}}.get(command, 1)
    monkeypatch.setattr(control.device, "_reader", lambda: Reader())
    def navigation():
        queries.append((control.device.host, control.device.port, "navigation"))
        return SimulatedMicroscope().navigation()
    monkeypatch.setattr(control.device, "navigation", navigation)
    server.save_app_settings({"microscopeHost": "192.168.0.94", "microscopePort": 61558})
    caps = control.call("sem_get_capabilities", {})
    assert caps["device_endpoint"] == {"host": "192.168.0.94", "port": 61558}
    assert not caps["capture_enabled"] and not caps["move_enabled"]
    assert caps['rpc_write_enabled'] == [] and control.rpc_limits == {}
    job = control.call("sem_get_navigation", {"request_id": "after-target-change"}, "test")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = control.call("sem_get_job", {"job_id": job["job_id"]}, "test")
        if job["status"] in control.TERMINAL:
            break
        time.sleep(.01)
    assert job["status"] == "completed", job
    assert any(command == "navigation" for _, _, command in queries)
    assert {(host, port) for host, port, _ in queries} == {("192.168.0.94", 61558)}
    with pytest.raises(SemError, match="未开放采图"):
        control.call("sem_capture", {"request_id": "new-device-no-approval"}, "test")


@pytest.mark.parametrize("owner", ["desktop:analysis", "mcp:navigation-job"])
def test_address_change_is_rejected_while_device_operation_is_active(endpoint_control, owner):
    control = endpoint_control
    before = server.SETTINGS_PATH.read_bytes()
    with control.coordinator.operation(owner, writes=False):
        with pytest.raises(DeviceBusy):
            server.save_app_settings({"microscopeHost": "192.168.0.94"})
        assert (control.device.host, control.device.port) == ("192.168.0.15", 61557)
        assert server.SETTINGS_PATH.read_bytes() == before
    assert control.allow_capture and control.bounds is not None
    assert control.allow_rpc == {'SetMag'} and control.rpc_limits == {'SetMag': [[100, 5000]]}


def test_unrelated_settings_do_not_reset_mcp_device_permissions(endpoint_control):
    control = endpoint_control
    with control.coordinator.operation("desktop:analysis", writes=False):
        server.save_app_settings({"showReportCoordinates": False})
    assert (control.device.host, control.device.port) == ("192.168.0.15", 61557)
    assert control.allow_capture and control.bounds is not None


def test_same_microscope_endpoint_with_coerced_types_keeps_capture(endpoint_control, tmp_path, monkeypatch):
    """保存设置时 host/port 类型不同但值相同，不得清掉采图权限。"""
    control = endpoint_control
    monkeypatch.setattr(server, "TASK_ROOT", tmp_path / "tasks")
    (tmp_path / "tasks").mkdir()
    (tmp_path / "mcp-device.json").write_text('{"allowCapture": true}\n', encoding="utf-8")
    server.save_app_settings({"microscopeHost": "192.168.0.15", "microscopePort": "61557"})
    assert control.allow_capture and control.bounds is not None
    assert (control.device.host, control.device.port) == ("192.168.0.15", 61557)


def test_failed_settings_write_keeps_mcp_endpoint_and_permissions(endpoint_control, monkeypatch):
    control = endpoint_control
    def fail(*args):
        raise OSError("disk write failed")
    monkeypatch.setattr(server, "atomic_json", fail)
    with pytest.raises(OSError, match="disk write failed"):
        server.save_app_settings({"microscopeHost": "192.168.0.94"})
    assert control.device.host == "192.168.0.15"
    assert control.allow_capture and control.bounds is not None
    assert control.coordinator.owner is None


@pytest.mark.skipif(os.name != "nt", reason="Windows CPA launcher process tree")
def test_cancelling_codex_stops_launcher_children(tmp_path, monkeypatch):
    launcher = tmp_path / "cpa-with-child.py"
    ready, survivor = tmp_path / "ready", tmp_path / "survivor"
    launcher.write_text(r'''
import os, subprocess, sys, time
from pathlib import Path
sys.stdin.read()
subprocess.Popen([sys.executable, '-c', "import time; from pathlib import Path; time.sleep(1.5); Path(" + repr(os.environ['ZEM_TEST_SURVIVOR']) + ").write_text('child still alive')"])
Path(os.environ['ZEM_TEST_READY']).write_text('ready')
time.sleep(30)
''', encoding="utf-8")
    monkeypatch.setattr(chat_runner, "FAKE_CODEX", "")
    monkeypatch.setattr(chat_runner, "CODEX_CMD", str(launcher))
    monkeypatch.setenv("ZEM_TEST_READY", str(ready))
    monkeypatch.setenv("ZEM_TEST_SURVIVOR", str(survivor))
    manager = chat_runner.ChatManager(data_root=tmp_path / "chat", engine="codex")
    session = manager.create_session("取消进程树")
    manager.send_message(session["id"], "等待取消")
    try:
        deadline = time.monotonic() + 5
        while not ready.is_file() and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.is_file()
        manager.cancel(session["id"])
        deadline = time.monotonic() + 5
        while manager.get_session(session["id"])["status"] == "running" and time.monotonic() < deadline:
            time.sleep(.02)
        assert manager.get_session(session["id"])["status"] == "idle"
        time.sleep(1.7)
        assert not survivor.exists(), "Cancelled CPA launcher left the Codex child running"
    finally:
        if manager.get_session(session["id"])["status"] == "running":
            manager.cancel(session["id"])


def test_deepseek_turn_offers_mcp_tools_and_consumes_protocol_response(runtime, tmp_path, monkeypatch):
    manager = chat_runner.ChatManager(data_root=tmp_path / "chat", engine="deepseek")
    calls = []
    def model(system, messages, tools):
        calls.append(tools)
        assert "sem_get_status" in {t["function"]["name"] for t in tools}
        if len(calls) == 1:
            return {"choices": [{"message": {"role": "assistant", "content": "我先查询设备状态。", "tool_calls": [
                {"id": "status-1", "type": "function", "function": {"name": "sem_get_status", "arguments": "{}"}}]}}]}
        assert "Idle" in messages[-1]["content"]
        return {"choices": [{"message": {"role": "assistant", "content": "模拟设备空闲。"}}]}
    monkeypatch.setattr(chat_runner, "_llm_chat", model)
    session = manager.create_session("MCP 工具循环")
    manager._run_turn_deepseek(session["id"], "查询电镜状态")
    assert len(calls) == 2
    assert any(event["text"] == "模拟设备空闲。" for event in manager.get_session(session["id"])["events"])


def test_real_backend_entrypoint_starts_simulated_mcp_without_touching_hardware(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "ZEM_TASK_ROOT": str(tmp_path / "tasks"), "ZEM_LOG_DIR": str(tmp_path / "logs"),
           "ZEM_MCP_TOKEN": "simulated-startup-test", "PYTHONUTF8": "1"}
    process = subprocess.Popen([sys.executable, str(root / "backend/server.py"), "--port", "0",
        "--mcp-simulated", "--mcp-state-dir", str(tmp_path / "mcp")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True, encoding="utf-8")
    try:
        ready = queue.Queue()
        threading.Thread(target=lambda: ready.put(process.stdout.readline()), daemon=True).start()
        line = ready.get(timeout=15)
        assert line, process.stderr.read() if process.poll() is not None else "backend startup failed"
        info = json.loads(line)
        url = f"http://127.0.0.1:{info['port']}"
        monkeypatch.setenv("ZEM_MCP_BACKEND_URL", url)
        monkeypatch.setenv("ZEM_MCP_TOKEN", "simulated-startup-test")
        result = sem_mcp_client.call_tool("sem_get_capabilities", {}, "startup-test")
        assert result["structuredContent"]["mode"] == "simulated"
        devices = json.loads(urllib.request.urlopen(url + "/devices", timeout=3).read())
        assert all("模拟" in device["detail"] for device in devices["devices"])
    finally:
        process.terminate()
        process.communicate(timeout=5)
