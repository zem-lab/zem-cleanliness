"""Helpers for ZEM microscope discovery."""

import ipaddress
import os
import re
import socket
import subprocess

ZEM_DISCOVERY_PORT = 61557
_BENCHMARK_NET = ipaddress.ip_network("198.18.0.0/15")
_LAN_NETS = (
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _get_remote_addr(zem_api):
    device = getattr(zem_api, "device", None)
    model = getattr(device, "m", None)
    if model is None or not hasattr(model, "addr"):
        return None, None
    return model, getattr(model, "addr", None)


def _get_udp_socket(zem_api):
    device = getattr(zem_api, "device", None)
    return getattr(device, "socket", None)


def _append_unique_ip(addresses, ip):
    if ip and ip not in addresses:
        addresses.append(ip)


def _socket_reported_ipv4_addresses():
    addresses = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_DGRAM):
            _append_unique_ip(addresses, info[4][0])
        for ip in socket.gethostbyname_ex(hostname)[2]:
            _append_unique_ip(addresses, ip)
    except OSError:
        pass
    return addresses


def _default_route_ipv4_address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


def _netifaces_ipv4_addresses():
    try:
        import netifaces
    except ImportError:
        return []

    addresses = []
    try:
        for interface in netifaces.interfaces():
            for addr_info in netifaces.ifaddresses(interface).get(netifaces.AF_INET, []):
                _append_unique_ip(addresses, addr_info.get("addr"))
    except Exception:
        return addresses
    return addresses


def _run_discovery_command(command):
    try:
        completed = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=3, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout or ""


def _parse_ipv4_lines(text, require_ipv4_label=False):
    addresses = []
    for line in text.splitlines():
        if require_ipv4_label and "IPv4" not in line:
            continue
        for ip in _IPV4_RE.findall(line):
            _append_unique_ip(addresses, ip)
    return addresses


def _windows_powershell_ipv4_addresses():
    if os.name != "nt":
        return []
    output = _run_discovery_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-NetIPAddress -AddressFamily IPv4 | "
            "Where-Object {$_.AddressState -eq 'Preferred'} | "
            "Select-Object -ExpandProperty IPAddress",
        ]
    )
    return _parse_ipv4_lines(output)


def _windows_ipconfig_ipv4_addresses():
    if os.name != "nt":
        return []
    output = _run_discovery_command(["ipconfig"])
    return _parse_ipv4_lines(output, require_ipv4_label=True)


def _local_ipv4_addresses():
    addresses = []
    for ip in _socket_reported_ipv4_addresses():
        _append_unique_ip(addresses, ip)
    _append_unique_ip(addresses, _default_route_ipv4_address())
    for source in (_netifaces_ipv4_addresses, _windows_powershell_ipv4_addresses, _windows_ipconfig_ipv4_addresses):
        for ip in source():
            _append_unique_ip(addresses, ip)
    return addresses


def _is_usable_unicast_ip(ip):
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (addr.is_loopback or addr.is_unspecified or addr.is_link_local or addr.is_multicast or addr in _BENCHMARK_NET)


def _ip_rank(ip):
    addr = ipaddress.ip_address(ip)
    if addr in _BENCHMARK_NET:
        return (50, ip)
    for index, network in enumerate(_LAN_NETS):
        if addr in network:
            return (index, ip)
    if addr.is_private:
        return (10, ip)
    return (20, ip)


def _candidate_bind_ips(zem_api):
    candidates = []
    udp_socket = _get_udp_socket(zem_api)
    socket_model = getattr(udp_socket, "m", None)
    current_bind_ip = getattr(socket_model, "ip", None)
    if current_bind_ip:
        candidates.append(current_bind_ip)
    candidates.extend(_local_ipv4_addresses())

    unique = []
    seen = set()
    for ip in candidates:
        if ip in seen or not _is_usable_unicast_ip(ip):
            continue
        seen.add(ip)
        unique.append(ip)
    return sorted(unique, key=_ip_rank)


def _reset_udp_socket(udp_socket):
    active_socket = getattr(udp_socket, "_socket", None)
    if active_socket is None:
        return

    worker = getattr(udp_socket, "rcv_worker", None)
    if worker is not None and hasattr(worker, "stop"):
        worker.stop()

    try:
        active_socket.close()
    finally:
        udp_socket._socket = None


def _set_udp_bind_ip(zem_api, bind_ip):
    udp_socket = _get_udp_socket(zem_api)
    socket_model = getattr(udp_socket, "m", None)
    if udp_socket is None or socket_model is None:
        return

    active_socket = getattr(udp_socket, "_socket", None)
    bound_ip = None
    if active_socket is not None:
        try:
            bound_ip = active_socket.getsockname()[0]
        except OSError:
            bound_ip = None

    if active_socket is None:
        socket_model.port = 0
    elif bound_ip != bind_ip:
        _reset_udp_socket(udp_socket)
        socket_model.port = 0

    socket_model.ip = bind_ip


def _sync_udp_builder_addr(zem_api):
    device = getattr(zem_api, "device", None)
    udp_socket = _get_udp_socket(zem_api)
    builder = getattr(device, "builder", None)
    if udp_socket is None or builder is None:
        return
    builder.ip, builder.port = udp_socket.sock.getsockname()


def prepare_discovery_port(zem_api, discovery_port: int = ZEM_DISCOVERY_PORT):
    """Make the next pyzemrpc handshake broadcast use the discovery port.

    pyzemrpc broadcasts handshake packets to ``zem_api.device.m.addr[1]``.
    Connecting to a device mutates that address, so a later scan can be sent
    to the previously connected device port instead of the fixed discovery
    port. Keep this as a small adapter around the third-party API.

    Returns the previous remote address so callers can restore it after the
    handshake finishes.
    """
    model, addr = _get_remote_addr(zem_api)
    if model is None:
        return None

    host = "UNK"
    if isinstance(addr, (tuple, list)) and addr:
        host = addr[0]

    next_addr = (host, int(discovery_port))
    if addr != next_addr:
        model.addr = next_addr
    return addr


def restore_remote_addr(zem_api, previous_addr):
    if previous_addr is None:
        return

    model, _ = _get_remote_addr(zem_api)
    if model is not None:
        model.addr = previous_addr


def scan_zem_devices(zem_api):
    previous_addr = prepare_discovery_port(zem_api)
    try:
        bind_ips = _candidate_bind_ips(zem_api)
        if not bind_ips:
            return zem_api.handshake()

        last_error = None
        had_successful_scan = False
        for bind_ip in bind_ips:
            _set_udp_bind_ip(zem_api, bind_ip)
            try:
                devices = zem_api.handshake()
            except OSError as exc:
                last_error = exc
                continue
            had_successful_scan = True
            if devices:
                return list(dict.fromkeys(devices))

        if last_error is not None and not had_successful_scan:
            raise last_error
        return []
    finally:
        restore_remote_addr(zem_api, previous_addr)


def validate_zem_status(zem_api, device_addr, *, maximum_attempts=None):
    """Connect to a device address and verify it responds to UDP StatusCheck."""
    zem_api.update_remote_addr(device_addr)

    bind_ips = _candidate_bind_ips(zem_api)
    last_error = None
    if not bind_ips:
        resp = zem_api.device.check_status()
        if err := resp.is_error():
            raise RuntimeError(f"StatusCheck 返回错误: {err}")
        return resp.parse_status()

    attempts = bind_ips
    if maximum_attempts is not None:
        attempts = bind_ips[: max(1, int(maximum_attempts))]
    for bind_ip in attempts:
        _set_udp_bind_ip(zem_api, bind_ip)
        _sync_udp_builder_addr(zem_api)
        try:
            resp = zem_api.device.check_status()
            if err := resp.is_error():
                raise RuntimeError(f"StatusCheck 返回错误: {err}")
            return resp.parse_status()
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"StatusCheck 无响应: {last_error}") from last_error
