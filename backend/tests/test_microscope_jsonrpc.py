import json
from urllib.error import URLError

from backend.cleanliness_core.eds import microscope_jsonrpc as rpc


class _JsonResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def setup_function():
    rpc._resolved_jsonrpc_ports.clear()


def test_start_scan_points_uses_udp_for_discovery_port(monkeypatch):
    calls = []

    def send_udp(host, port, points, timeout):
        calls.append((host, port, points, timeout))
        return {"jsonrpc": "2.0", "result": {"transport": "udp", "status": "scanning"}, "id": 1}

    monkeypatch.setattr(rpc, "_send_udp_scan_points", send_udp)

    result = rpc.start_scan_points("192.168.0.249", rpc.ZEM_DISCOVERY_PORT, [(362, 48)])

    assert result["result"]["transport"] == "udp"
    assert calls == [("192.168.0.249", rpc.ZEM_DISCOVERY_PORT, [(362, 48)], 10.0)]


def test_lost_udp_ack_retries_same_point_before_unsupported_fallback(monkeypatch):
    calls = []

    def send_udp(host, port, points, timeout):
        calls.append((host, port, points, timeout))
        if len(calls) == 1:
            raise TimeoutError("timed out")
        return {"result": {"transport": "udp", "status": "scanning"}}

    def unexpected_fallback(*args, **kwargs):
        raise AssertionError("A recovered UDP timeout must not call JSON-RPC")

    monkeypatch.setattr(rpc, "_send_udp_scan_points", send_udp)
    monkeypatch.setattr(rpc, "_request_jsonrpc", unexpected_fallback)
    result = rpc.start_scan_points("192.168.0.15", rpc.ZEM_DISCOVERY_PORT, [(100, 200)])
    assert result["result"]["transport"] == "udp"
    assert len(calls) == 2
    assert calls[0] == calls[1]


def test_udp_transport_retry_is_bounded_and_does_not_retry_device_errors(monkeypatch):
    for error, expected_count in [(TimeoutError("timeout"), 2), (rpc.MicroscopeJsonRpcError("device rejected"), 1)]:
        calls = []

        def send_udp(*args):
            calls.append(args)
            raise error

        monkeypatch.setattr(rpc, "_send_udp_scan_points", send_udp)
        monkeypatch.setattr(rpc, "_request_jsonrpc", lambda *args: ({"result": {"status": "scanning"}}, 61600))
        assert rpc.start_scan_points("192.168.0.15", rpc.ZEM_DISCOVERY_PORT, [(100, 200)])["result"]["status"] == "scanning"
        assert len(calls) == expected_count


def test_probe_microscope_status_uses_one_udp_status_check(monkeypatch):
    sent = []

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def settimeout(self, value):
            assert value == 0.8

        def bind(self, address):
            assert address == ("192.168.0.206", 0)

        def getsockname(self):
            return ("192.168.0.206", 52006)

        def sendto(self, payload, address):
            sent.append((payload, address))

        def recvfrom(self, size):
            assert size == 4096
            return b"$StatusCheck:Idle,0D87%", ("192.168.0.15", 64633)

    monkeypatch.setattr(rpc, "_local_ip_for_target", lambda host, port: "192.168.0.206")
    monkeypatch.setattr(
        rpc,
        "_build_udp_command_payload",
        lambda command, args, local_ip, local_port: b"status-payload",
    )
    monkeypatch.setattr(rpc.socket, "socket", lambda *args: FakeSocket())

    status = rpc.probe_microscope_status(
        "192.168.0.15",
        rpc.ZEM_DISCOVERY_PORT,
        timeout=0.8,
    )

    assert status == "Idle"
    assert sent == [(b"status-payload", ("192.168.0.15", rpc.ZEM_DISCOVERY_PORT))]


def test_start_scan_points_falls_back_to_jsonrpc_when_udp_fails(monkeypatch):
    calls = []

    def urlopen(req, timeout):
        calls.append(req.full_url)
        return _JsonResponse({"jsonrpc": "2.0", "result": {"status": "scanning"}, "id": 1})

    def send_udp(host, port, points, timeout):
        raise TimeoutError("udp timeout")

    monkeypatch.setattr(rpc.urllib_request, "urlopen", urlopen)
    monkeypatch.setattr(rpc, "_send_udp_scan_points", send_udp)
    monkeypatch.setattr(rpc, "_discover_jsonrpc_port", lambda host, port: 60217)

    result = rpc.start_scan_points("192.168.0.249", rpc.ZEM_DISCOVERY_PORT, [(362, 48)])

    assert result["result"]["status"] == "scanning"
    assert calls == ["http://192.168.0.249:60217/api/v1/jsonrpc"]


def test_start_scan_points_falls_back_to_discovery_port(monkeypatch):
    calls = []

    def urlopen(req, timeout):
        calls.append(req.full_url)
        if ":61600/" in req.full_url:
            raise URLError("connection refused")
        return _JsonResponse({"jsonrpc": "2.0", "result": {"status": "scanning"}, "id": 1})

    def send_udp(host, port, points, timeout):
        raise TimeoutError("udp timeout")

    monkeypatch.setattr(rpc.urllib_request, "urlopen", urlopen)
    monkeypatch.setattr(rpc, "_send_udp_scan_points", send_udp)
    monkeypatch.setattr(rpc, "_discover_jsonrpc_port", lambda host, port: None)

    result = rpc.start_scan_points("192.168.0.207", rpc.ZEM_DISCOVERY_PORT, [(362, 48)])

    assert result["result"]["status"] == "scanning"
    assert calls == [
        "http://192.168.0.207:61600/api/v1/jsonrpc",
        "http://192.168.0.207:61557/api/v1/jsonrpc",
    ]


def test_start_scan_points_keeps_explicit_jsonrpc_port(monkeypatch):
    calls = []

    def urlopen(req, timeout):
        calls.append(req.full_url)
        return _JsonResponse({"jsonrpc": "2.0", "result": {"status": "scanning"}, "id": 1})

    monkeypatch.setattr(rpc.urllib_request, "urlopen", urlopen)

    rpc.start_scan_points("192.168.0.207", 61600, [(362, 48)])

    assert calls == ["http://192.168.0.207:61600/api/v1/jsonrpc"]


def test_start_scan_points_prefers_discovered_jsonrpc_port(monkeypatch):
    calls = []

    def urlopen(req, timeout):
        calls.append(req.full_url)
        return _JsonResponse({"jsonrpc": "2.0", "result": {"status": "scanning"}, "id": 1})

    def send_udp(host, port, points, timeout):
        raise TimeoutError("udp timeout")

    monkeypatch.setattr(rpc.urllib_request, "urlopen", urlopen)
    monkeypatch.setattr(rpc, "_send_udp_scan_points", send_udp)
    monkeypatch.setattr(rpc, "_discover_jsonrpc_port", lambda host, port: 60217)

    rpc.start_scan_points("192.168.0.249", rpc.ZEM_DISCOVERY_PORT, [(362, 48)])

    assert calls == ["http://192.168.0.249:60217/api/v1/jsonrpc"]
