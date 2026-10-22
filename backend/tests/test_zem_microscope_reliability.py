import threading
import time
from types import SimpleNamespace

import pytest

from backend.cleanliness_core.optical_particle_task import (
    ZEM_MIN_COMMAND_TIMEOUT_MS,
    ZemMicroscope,
    _cache_zemrpc_interface,
    _configure_zemrpc_device,
    _configure_zemrpc_timeouts,
)


class _Socket:
    def __init__(self, address=("192.168.0.206", 58722)):
        self._address = address

    def getsockname(self):
        return self._address


class _ReliableDevice:
    def __init__(self):
        self.events: list[str] = []
        self._iface_by_device = {}
        self.socket = SimpleNamespace(sock=_Socket())
        self.m = SimpleNamespace(addr=("192.168.0.15", 61557))

    def _prepare_comm(self):
        self.events.append("prepare")

    def pubsub_send(self, request):
        self.events.append(f"send:{request}:start")
        time.sleep(0.01)
        self.events.append(f"send:{request}:end")
        return request


class _StageApi:
    def __init__(self):
        self.device = SimpleNamespace()
        self.position = (0.0, 0.0)
        self.moves: list[tuple[float, float]] = []
        self._attempt = 0
        self._reads_after_move = 0

    def get_stage_pos(self):
        if self._attempt == 1:
            position = (80.0, 0.0)
        elif self._attempt >= 2:
            position = (100.0, 0.0)
        else:
            position = self.position
        self._reads_after_move += 1
        return SimpleNamespace(x=position[0], y=position[1])

    def stage_xy_goto_closed(self, target, stable_after_idle_ms=0):
        self._attempt += 1
        self._reads_after_move = 0
        self.moves.append((float(target[0]), float(target[1])))


def test_short_zem_commands_use_one_second_minimum_timeout():
    from zemrpc.zem.conn.protocol import EnumCmd

    _configure_zemrpc_timeouts()

    assert EnumCmd.GetStagePos.timeout == ZEM_MIN_COMMAND_TIMEOUT_MS
    assert EnumCmd.SetStagePos.timeout == ZEM_MIN_COMMAND_TIMEOUT_MS
    assert EnumCmd.GetScanInfo.timeout == ZEM_MIN_COMMAND_TIMEOUT_MS
    assert EnumCmd.Snap.timeout == 60_000


def test_device_prepares_before_send_serializes_and_caches_interface():
    device = _ReliableDevice()
    _configure_zemrpc_device(device)

    results: list[str] = []
    threads = [
        threading.Thread(target=lambda value=value: results.append(device.pubsub_send(value)))
        for value in ("one", "two")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == ["one", "two"]
    assert len(device.events) == 6
    for offset in (0, 3):
        assert device.events[offset] == "prepare"
        request_name = device.events[offset + 1].split(":")[1]
        assert device.events[offset + 1] == f"send:{request_name}:start"
        assert device.events[offset + 2] == f"send:{request_name}:end"
    assert _cache_zemrpc_interface(device, "192.168.0.15", 61557) == "192.168.0.206"
    assert device._iface_by_device[("192.168.0.15", 61557)] == "192.168.0.206"


def test_stage_move_retries_when_idle_readback_is_stable_but_off_target(monkeypatch):
    microscope = ZemMicroscope(
        "192.168.0.15", 61557, {"imageShape": [512, 512]}
    )
    api = _StageApi()
    microscope._api = api
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    move = microscope.move_xy(100.0, 0.0)

    assert api.moves == [(100.0, 0.0), (100.0, 0.0)]
    assert move["after"] == {"x": 100.0, "y": 0.0}
    assert move["positionErrorUm"] == pytest.approx(0.0)
    assert [item["withinTolerance"] for item in move["attempts"]] == [False, True]


def test_http_adapter_preserves_error_and_never_replays_snap(monkeypatch):
    import requests
    device=_ReliableDevice()
    device.zem_rpc=lambda payload:None
    device.m.protocol_ng=SimpleNamespace(url='http://instrument.invalid/api')
    calls=[]
    class Session:
        trust_env=True
        def post(self,url,json,timeout):
            calls.append((url,json,timeout,self.trust_env))
            raise requests.ConnectionError('service unavailable')
    monkeypatch.setattr(requests,'Session',Session)
    _configure_zemrpc_device(device)
    with pytest.raises(RuntimeError,match='service unavailable'):
        device.zem_rpc({'method':'Snap'})
    assert calls==[('http://instrument.invalid/api',{'method':'Snap'},(5,90),False)]
