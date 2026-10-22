from __future__ import annotations

import re
import socket
import time
from typing import Any


def hv_status_name(status: Any) -> str:
    """Return a stable lower-case high-voltage status name."""
    return str(status).rsplit(".", 1)[-1].strip().casefold()


def _builder_uses_modern_hv_command(device: Any, enabled: bool) -> bool:
    try:
        builder = device.builder
        request = builder.build_gun_on() if enabled else builder.build_gun_off()
        data = bytes(request.data)
    except Exception:
        return False
    command = b"$SetHV_ON:" if enabled else b"$SetHV_OFF:"
    return data.startswith(command)


def _enable_advanced_work_mode(device: Any) -> None:
    set_work_mode = getattr(device, "set_work_mode", None)
    if not callable(set_work_mode):
        return
    try:
        from zemrpc.zem.models import EnumCmdLevel
    except (ImportError, AttributeError):
        return
    set_work_mode(EnumCmdLevel.ADVANCED)


def _route_local_ip(host: str, port: int) -> str:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect((str(host), int(port)))
        return str(probe.getsockname()[0])
    finally:
        probe.close()


def _exchange_modern_command(
    host: str,
    port: int,
    *,
    command: str,
    timeout: float = 5.0,
) -> str:
    from zemrpc.zem.conn.protocol import append_crc

    local_ip = _route_local_ip(host, port)
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        client.bind((local_ip, 0))
        client.settimeout(max(0.1, float(timeout)))
        local_port = int(client.getsockname()[1])
        message = append_crc(f"${command}:,{local_ip}:{local_port},")
        client.sendto(message, (str(host), int(port)))
        response, _address = client.recvfrom(4096)
    finally:
        client.close()

    return response.decode("utf-8", errors="replace")


def _get_allow_open_gun(host: str, port: int, *, timeout: float = 5.0) -> bool | None:
    text = _exchange_modern_command(
        host,
        port,
        command="AllowOpenGun",
        timeout=timeout,
    )
    if not text.startswith("$AllowOpenGun:"):
        raise RuntimeError(f"开枪安全联锁回包不匹配：{text}")
    value = text.split(":", 1)[1].split(",", 1)[0].strip().casefold()
    if value in {"true", "1"}:
        return True
    if value in {"false", "0"}:
        return False
    return None


def _send_modern_hv_command(
    host: str,
    port: int,
    *,
    enabled: bool,
    timeout: float = 5.0,
) -> str:
    """Send the modern SetHV_ON/SetHV_OFF UDP command.

    pyzemrpc 1.0 emits the obsolete SetHvON/SetHvOFF spellings.  The Zem20F
    controller at 192.168.0.15 rejects those commands, while pyzemrpc 1.2
    uses the spellings below.  A separate UDP socket keeps this compatibility
    path isolated from pyzemrpc's background receiver.
    """
    command = "SetHV_ON" if enabled else "SetHV_OFF"
    text = _exchange_modern_command(
        host,
        port,
        command=command,
        timeout=timeout,
    )
    if not text.startswith(f"${command}:"):
        raise RuntimeError(f"电子枪命令回包不匹配：{text}")
    error_match = re.search(r"Error Code\s*:\s*(\d+)", text, re.IGNORECASE)
    if error_match and int(error_match.group(1)) != 0:
        raise RuntimeError(f"电子枪命令执行失败：{text}")
    return text


def _send_hv_command(
    api: Any,
    host: str,
    port: int,
    *,
    enabled: bool,
    command_timeout: float,
) -> None:
    device = api.device
    if enabled:
        allowed = _get_allow_open_gun(
            host,
            port,
            timeout=command_timeout,
        )
        if allowed is False:
            raise RuntimeError(
                "设备安全联锁当前不允许开枪（AllowOpenGun=False）；"
                "请在电镜控制端确认真空、换样状态和远程开枪许可"
            )
    if _builder_uses_modern_hv_command(device, enabled):
        _enable_advanced_work_mode(device)
        if enabled:
            device.gun_on()
        else:
            device.gun_off()
        return
    _send_modern_hv_command(
        host,
        port,
        enabled=enabled,
        timeout=command_timeout,
    )


def set_gun_state(
    api: Any,
    host: str,
    port: int,
    *,
    enabled: bool,
    timeout: float = 60.0,
    poll_interval: float = 1.0,
    consecutive_reads: int = 1,
) -> str:
    """Set and verify the electron-gun state across pyzemrpc versions.

    The operation is idempotent.  Closing can request multiple consecutive
    reads so a task only reports success after the controller remains closed.
    """
    target = "open" if enabled else "close"
    required_reads = max(1, int(consecutive_reads))
    deadline = time.monotonic() + max(0.1, float(timeout))
    status = api.device.get_hv_status()
    matching_reads = 1 if hv_status_name(status) == target else 0

    if not matching_reads:
        _send_hv_command(
            api,
            host,
            port,
            enabled=enabled,
            command_timeout=min(5.0, max(0.1, float(timeout))),
        )

    while matching_reads < required_reads:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"等待电子枪状态 {target} 超时，最后状态：{status}"
            )
        time.sleep(min(max(0.01, float(poll_interval)), remaining))
        status = api.device.get_hv_status()
        if hv_status_name(status) == target:
            matching_reads += 1
        else:
            matching_reads = 0

    return str(status)
