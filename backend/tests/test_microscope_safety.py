from __future__ import annotations

from enum import Enum

import pytest

from backend.cleanliness_core import microscope_safety


class _HvStatus(Enum):
    Close = 0
    Open = 1
    Changing = 2


class _Request:
    def __init__(self, data: bytes):
        self.data = data


class _Builder:
    def __init__(self, modern: bool):
        self.modern = modern

    def build_gun_on(self):
        name = "SetHV_ON" if self.modern else "SetHvON"
        return _Request(f"${name}:,127.0.0.1:1,0000%".encode())

    def build_gun_off(self):
        name = "SetHV_OFF" if self.modern else "SetHvOFF"
        return _Request(f"${name}:,127.0.0.1:1,0000%".encode())


class _Device:
    def __init__(self, statuses, *, modern=False):
        self.statuses = list(statuses)
        self.builder = _Builder(modern)
        self.commands = []
        self.work_modes = []

    def get_hv_status(self):
        if len(self.statuses) > 1:
            return self.statuses.pop(0)
        return self.statuses[0]

    def gun_on(self):
        self.commands.append("on")

    def gun_off(self):
        self.commands.append("off")

    def set_work_mode(self, mode):
        self.work_modes.append(mode)


class _Api:
    def __init__(self, device):
        self.device = device


def test_already_closed_is_idempotent_and_requires_consecutive_reads():
    device = _Device([_HvStatus.Close, _HvStatus.Close, _HvStatus.Close])
    status = microscope_safety.set_gun_state(
        _Api(device),
        "example",
        61557,
        enabled=False,
        poll_interval=0.001,
        consecutive_reads=3,
    )
    assert status.endswith(".Close")
    assert device.commands == []


def test_old_sdk_uses_modern_raw_command(monkeypatch):
    device = _Device(
        [_HvStatus.Open, _HvStatus.Changing, _HvStatus.Close],
        modern=False,
    )
    calls = []
    monkeypatch.setattr(
        microscope_safety,
        "_send_modern_hv_command",
        lambda host, port, **kwargs: calls.append((host, port, kwargs)),
    )
    status = microscope_safety.set_gun_state(
        _Api(device),
        "192.168.0.15",
        61557,
        enabled=False,
        poll_interval=0.001,
    )
    assert status.endswith(".Close")
    assert calls == [
        (
            "192.168.0.15",
            61557,
            {"enabled": False, "timeout": 5.0},
        )
    ]
    assert device.commands == []


def test_modern_sdk_uses_advanced_mode_and_native_command(monkeypatch):
    device = _Device([_HvStatus.Close, _HvStatus.Changing, _HvStatus.Open], modern=True)
    monkeypatch.setattr(
        microscope_safety,
        "_get_allow_open_gun",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        microscope_safety,
        "_enable_advanced_work_mode",
        lambda value: value.work_modes.append("advanced"),
    )
    status = microscope_safety.set_gun_state(
        _Api(device),
        "192.168.0.15",
        61557,
        enabled=True,
        poll_interval=0.001,
    )
    assert status.endswith(".Open")
    assert device.work_modes == ["advanced"]
    assert device.commands == ["on"]


def test_open_is_blocked_when_device_safety_interlock_is_false(monkeypatch):
    device = _Device([_HvStatus.Close], modern=False)
    monkeypatch.setattr(
        microscope_safety,
        "_get_allow_open_gun",
        lambda *args, **kwargs: False,
    )
    with pytest.raises(RuntimeError, match="AllowOpenGun=False"):
        microscope_safety.set_gun_state(
            _Api(device),
            "192.168.0.15",
            61557,
            enabled=True,
            poll_interval=0.001,
        )


def test_timeout_reports_last_status(monkeypatch):
    device = _Device([_HvStatus.Open], modern=False)
    monkeypatch.setattr(
        microscope_safety,
        "_send_modern_hv_command",
        lambda *args, **kwargs: "ok",
    )
    with pytest.raises(TimeoutError, match="最后状态"):
        microscope_safety.set_gun_state(
            _Api(device),
            "192.168.0.15",
            61557,
            enabled=False,
            timeout=0.01,
            poll_interval=0.001,
        )
