from backend import chat_mcp, chat_runner, sem_mcp_client


def test_chat_only_allows_two_named_mcp_servers(monkeypatch):
    monkeypatch.setenv("ZEM_MCP_BACKEND_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("ZEM_MCP_TOKEN", "x")
    assert chat_mcp.allowed_servers() == ["zem-sem", "zem-cleanliness"]
    config = chat_mcp.claude_config()
    assert list(config["mcpServers"]) == ["zem-sem", "zem-cleanliness"]
    args = " ".join(chat_mcp.codex_config_args())
    assert args.count("mcp_servers.") >= 2
    assert "zem-sem" in args and "zem-cleanliness" in args
    assert "vibe-actions" not in args


def test_prompts_forbid_http_and_third_mcp():
    for prompt in (
        chat_runner.build_prompt("创建一个直接sem分析的任务", "- 快照"),
        chat_runner.build_claude_prompt("创建一个直接sem分析的任务", "- 快照"),
        chat_runner.build_system_prompt("- 快照"),
    ):
        assert "只能使用这两个 MCP" in prompt
        assert "zem-cleanliness" in prompt and "zem-sem" in prompt
        assert "navigate(target=sem-direct)" in prompt
        assert "禁止" in prompt and "/sem/navigation" in prompt
        assert "e2e/vibe-actions.cjs" in prompt
        assert "GetImg" in prompt
        assert "sem_get_navigation" in prompt
        assert "最多 20" in prompt
        assert "completion_confirmed=false" in prompt
        assert "仿真" in prompt and "不能开枪" in prompt
        assert "SetHV_ON" in prompt
        assert "关枪不阻止" in prompt and "开始分析" in prompt


def test_sem_mcp_client_stays_device_only(monkeypatch):
    monkeypatch.setenv("ZEM_MCP_BACKEND_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("ZEM_MCP_TOKEN", "x")
    assert list(sem_mcp_client.claude_config()["mcpServers"]) == ["zem-sem"]
