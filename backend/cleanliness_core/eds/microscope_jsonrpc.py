from __future__ import annotations

import json
import logging
import socket
from typing import Any
from urllib import request as urllib_request
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_req_id_counter = 0
ZEM_DISCOVERY_PORT = 61557
DEFAULT_JSONRPC_PORT = 61600
UDP_SCAN_POINTS_TIMEOUT = 2.0
_resolved_jsonrpc_ports: dict[tuple[str, int], int] = {}


class MicroscopeJsonRpcError(RuntimeError):
    """电镜 JSON-RPC 调用异常。"""


def _next_id() -> int:
    global _req_id_counter
    _req_id_counter += 1
    return _req_id_counter


def _append_unique_port(ports: list[int], port: int | None):
    if port is None:
        return
    port = int(port)
    if port not in ports:
        ports.append(port)


def _discover_jsonrpc_port(host: str, discovery_port: int) -> int | None:
    try:
        from zemrpc.zem.zem_api import zem_api

        from backend.cleanliness_core.microscope_discovery import validate_zem_status

        validate_zem_status(zem_api, (host, int(discovery_port)))
        zem_api.device.get_protocol_ng()
        protocol_ng = getattr(zem_api.device.m, "protocol_ng", None)
        url = getattr(protocol_ng, "url", "")
        parsed = urlparse(url)
        return parsed.port
    except Exception as exc:
        logger.info(f"ZEM JSON-RPC 动态端口发现失败: {host}:{discovery_port}, {exc}")
        return None


def _candidate_jsonrpc_ports(host: str, port: int) -> tuple[int, ...]:
    port = int(port)
    if port == ZEM_DISCOVERY_PORT:
        ports: list[int] = []
        _append_unique_port(ports, _discover_jsonrpc_port(host, port))
        _append_unique_port(ports, DEFAULT_JSONRPC_PORT)
        _append_unique_port(ports, ZEM_DISCOVERY_PORT)
        return tuple(ports)
    return (port,)


def _post_jsonrpc(host: str, port: int, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    url = f"http://{host}:{port}/api/v1/jsonrpc"
    body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout) as resp:
        resp_body = resp.read().decode("utf-8")
        return json.loads(resp_body)


def _local_ip_for_target(host: str, port: int) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route_sock:
        route_sock.connect((host, int(port)))
        return route_sock.getsockname()[0]


def _build_udp_command_payload(command: str, args: str, local_ip: str, local_port: int) -> bytes:
    from zemrpc.zem.conn.protocol import append_crc

    message = f"${command}:{args},{local_ip}:{int(local_port)},"
    return append_crc(message)


def _send_udp_command(host: str, port: int, command: str, args: str, timeout: float) -> str:
    local_ip = _local_ip_for_target(host, port)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(min(float(timeout), UDP_SCAN_POINTS_TIMEOUT))
        sock.bind((local_ip, 0))
        _, local_port = sock.getsockname()
        payload = _build_udp_command_payload(command, args, local_ip, local_port)
        logger.info(f"UDP {command} -> {host}:{port} args={args}")
        sock.sendto(payload, (host, int(port)))
        resp_body, resp_addr = sock.recvfrom(4096)

    resp_text = resp_body.decode("utf-8", errors="replace")
    logger.info(f"UDP {command} 响应 <- {resp_addr}: {resp_text}")
    if "Acknowledge" not in resp_text or "Error Code:0" not in resp_text:
        raise MicroscopeJsonRpcError(f"{command} UDP 返回错误 ({host}:{port}): {resp_text}")
    return resp_text


def probe_microscope_status(
    host: str,
    port: int = ZEM_DISCOVERY_PORT,
    *,
    timeout: float = 1.0,
) -> str:
    """通过单次 UDP ``StatusCheck`` 快速探测电镜，不触发完整设备同步。"""
    local_ip = _local_ip_for_target(host, port)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(float(timeout))
        sock.bind((local_ip, 0))
        _, local_port = sock.getsockname()
        payload = _build_udp_command_payload("StatusCheck", "", local_ip, local_port)
        sock.sendto(payload, (host, int(port)))
        resp_body, _ = sock.recvfrom(4096)

    resp_text = resp_body.decode("utf-8", errors="replace").strip()
    prefix = "$StatusCheck:"
    if not resp_text.startswith(prefix):
        raise MicroscopeJsonRpcError(
            f"StatusCheck UDP 返回异常 ({host}:{port}): {resp_text}"
        )
    status = resp_text[len(prefix) :].split(",", 1)[0].strip()
    if not status:
        raise MicroscopeJsonRpcError(
            f"StatusCheck UDP 未返回状态 ({host}:{port}): {resp_text}"
        )
    return status


def _send_udp_scan_points(
    host: str,
    port: int,
    points: list[tuple[int, int]],
    timeout: float,
) -> dict[str, Any]:
    sent_points = []

    for x, y in points:
        _send_udp_command(host, port, "StopScanPoints", "", timeout)
        _send_udp_command(host, port, "StartScanPoints", f"{int(x)},{int(y)}", timeout)
        sent_points.append([int(x), int(y)])

    return {
        "jsonrpc": "2.0",
        "result": {
            "status": "scanning",
            "scanning": True,
            "count": len(sent_points),
            "points": sent_points,
            "transport": "udp",
        },
        "id": _next_id(),
    }


def _request_jsonrpc(host: str, port: int, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], int]:
    requested_port = int(port)
    cache_key = (host, requested_port)
    cached_port = _resolved_jsonrpc_ports.get(cache_key)
    candidate_ports = _candidate_jsonrpc_ports(host, requested_port)
    if cached_port is not None:
        candidate_ports = (cached_port, *(p for p in candidate_ports if p != cached_port))

    last_exc: Exception | None = None
    for candidate_port in candidate_ports:
        try:
            result = _post_jsonrpc(host, candidate_port, payload, timeout)
        except Exception as exc:
            last_exc = exc
            if cached_port == candidate_port:
                _resolved_jsonrpc_ports.pop(cache_key, None)
            continue

        _resolved_jsonrpc_ports[cache_key] = candidate_port
        if candidate_port != requested_port:
            logger.info(f"ZEM JSON-RPC 端口解析: {host}:{requested_port} -> {host}:{candidate_port}")
        return result, candidate_port

    assert last_exc is not None
    raise MicroscopeJsonRpcError(f"startScanPoints 失败 ({host}:{requested_port}): {last_exc}") from last_exc


def start_scan_points(
    host: str,
    port: int,
    points: list[tuple[int, int]],
    timeout: float = 10.0,
) -> dict[str, Any]:
    """通过 JSON-RPC 调用电镜 startScanPoints 移束到指定坐标。

    points: [(x, y), ...]
    """
    if int(port) == ZEM_DISCOVERY_PORT:
        for attempt in range(2):
            try:
                return _send_udp_scan_points(host, port, points, timeout)
            except OSError as exc:
                # A lost UDP acknowledgement does not mean the microscope lacks
                # this command. Repeating Stop + Start at the same absolute
                # point is safe and avoids an unsupported JSON-RPC fallback.
                if attempt == 0:
                    logger.warning(f"UDP StartScanPoints 通信失败，重试同一打点一次: {exc}")
                    continue
                logger.warning(f"UDP StartScanPoints 重试失败，尝试 JSON-RPC fallback: {exc}")
            except Exception as exc:
                logger.warning(f"UDP StartScanPoints 失败，尝试 JSON-RPC fallback: {exc}")
            break

    for x, y in points:
        payload = {
            "jsonrpc": "2.0",
            "id": _next_id(),
            "method": "startScanPoints",
            "params": {"points": [x, y]},
        }
        logger.info(f"JSON-RPC startScanPoints -> {host}:{port}  point=({x}, {y})")
        try:
            result, resolved_port = _request_jsonrpc(host, port, payload, timeout)
        except MicroscopeJsonRpcError:
            raise

        if "error" in result:
            err = result["error"]
            raise MicroscopeJsonRpcError(
                f"startScanPoints 返回错误 ({host}:{resolved_port}): code={err.get('code')}, message={err.get('message')}"
            )
        logger.info(f"startScanPoints 响应: {result.get('result')}")
    return result
