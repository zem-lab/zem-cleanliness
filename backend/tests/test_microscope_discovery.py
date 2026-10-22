from types import SimpleNamespace
import sys

import pytest

import backend.cleanliness_core.microscope_discovery as discovery
from backend.cleanliness_core.microscope_discovery import (
    ZEM_DISCOVERY_PORT,
    prepare_discovery_port,
    restore_remote_addr,
    scan_zem_devices,
)


def test_discovery_command_preserves_ipv4_with_non_utf8_output():
    output = discovery._run_discovery_command([
        sys.executable, "-c",
        "import sys; sys.stdout.buffer.write(b'\\xff IPv4: 192.168.0.15\\n')",
    ])
    assert discovery._parse_ipv4_lines(output, require_ipv4_label=True) == ["192.168.0.15"]


def _api_with_addr(addr):
    return SimpleNamespace(device=SimpleNamespace(m=SimpleNamespace(addr=addr)))


def _api_with_socket(addr, bind_ip, handshake):
    return SimpleNamespace(
        device=SimpleNamespace(
            m=SimpleNamespace(addr=addr),
            socket=SimpleNamespace(m=SimpleNamespace(ip=bind_ip, port=0), _socket=None),
        ),
        handshake=handshake,
    )


def test_prepare_discovery_port_resets_previous_device_port():
    api = _api_with_addr(("192.168.1.20", 61600))

    previous_addr = prepare_discovery_port(api)

    assert previous_addr == ("192.168.1.20", 61600)
    assert api.device.m.addr == ("192.168.1.20", ZEM_DISCOVERY_PORT)


def test_prepare_discovery_port_keeps_unknown_host_when_no_device_selected():
    api = _api_with_addr(("UNK", 61600))

    previous_addr = prepare_discovery_port(api)

    assert previous_addr == ("UNK", 61600)
    assert api.device.m.addr == ("UNK", ZEM_DISCOVERY_PORT)


def test_prepare_discovery_port_is_safe_for_missing_zem_api_shape():
    assert prepare_discovery_port(SimpleNamespace()) is None


def test_restore_remote_addr_restores_connected_device_after_scan():
    api = _api_with_addr(("192.168.1.20", 61600))
    previous_addr = prepare_discovery_port(api)

    restore_remote_addr(api, previous_addr)

    assert api.device.m.addr == ("192.168.1.20", 61600)


def test_scan_zem_devices_uses_discovery_port_and_restores_connection(monkeypatch):
    # This is a fake-handshake unit test, not a local DNS/interface probe.
    monkeypatch.setattr(discovery, "_local_ipv4_addresses", lambda: [])
    observed_addr = None

    def handshake():
        nonlocal observed_addr
        observed_addr = api.device.m.addr
        return [("192.168.1.30", 61600)]

    api = _api_with_addr(("192.168.1.20", 61600))
    api.handshake = handshake

    assert scan_zem_devices(api) == [("192.168.1.30", 61600)]
    assert observed_addr == ("192.168.1.20", ZEM_DISCOVERY_PORT)
    assert api.device.m.addr == ("192.168.1.20", 61600)


def test_scan_zem_devices_restores_connection_when_handshake_fails(monkeypatch):
    monkeypatch.setattr(discovery, "_local_ipv4_addresses", lambda: [])
    def handshake():
        raise RuntimeError("network down")

    api = _api_with_addr(("192.168.1.20", 61600))
    api.handshake = handshake

    with pytest.raises(RuntimeError, match="network down"):
        scan_zem_devices(api)

    assert api.device.m.addr == ("192.168.1.20", 61600)


def test_scan_zem_devices_prefers_lan_interface_over_virtual_default(monkeypatch):
    observed_bind_ips = []

    def handshake():
        observed_bind_ips.append(api.device.socket.m.ip)
        if api.device.socket.m.ip == "192.168.0.206":
            return [("192.168.0.242", ZEM_DISCOVERY_PORT)]
        return []

    api = _api_with_socket(("UNK", 61600), "198.18.0.1", handshake)
    monkeypatch.setattr(
        "backend.cleanliness_core.microscope_discovery._local_ipv4_addresses",
        lambda: ["198.18.0.1", "192.168.0.206"],
    )

    assert scan_zem_devices(api) == [("192.168.0.242", ZEM_DISCOVERY_PORT)]
    assert observed_bind_ips == ["192.168.0.206"]
    assert api.device.socket.m.ip == "192.168.0.206"
    assert api.device.m.addr == ("UNK", 61600)


def test_candidate_bind_ips_ignores_benchmark_fake_ip(monkeypatch):
    api = _api_with_socket(("UNK", 61600), "198.18.0.1", lambda: [])
    monkeypatch.setattr(
        "backend.cleanliness_core.microscope_discovery._local_ipv4_addresses",
        lambda: ["198.18.0.1", "192.168.0.206"],
    )

    assert discovery._candidate_bind_ips(api) == ["192.168.0.206"]


def test_local_ipv4_addresses_combines_os_adapter_sources(monkeypatch):
    monkeypatch.setattr(discovery, "_socket_reported_ipv4_addresses", lambda: ["192.168.1.4"])
    monkeypatch.setattr(discovery, "_default_route_ipv4_address", lambda: "192.168.1.4")
    monkeypatch.setattr(discovery, "_netifaces_ipv4_addresses", lambda: ["10.10.0.5"])
    monkeypatch.setattr(discovery, "_windows_powershell_ipv4_addresses", lambda: ["192.168.0.206", "127.0.0.1"])
    monkeypatch.setattr(discovery, "_windows_ipconfig_ipv4_addresses", lambda: ["192.168.0.206", "198.18.0.1"])

    assert discovery._local_ipv4_addresses() == ["192.168.1.4", "10.10.0.5", "192.168.0.206", "127.0.0.1", "198.18.0.1"]


def test_parse_ipconfig_ipv4_lines_ignores_non_ipv4_addresses():
    output = """
Windows IP Configuration
   IPv4 Address. . . . . . . . . . . : 192.168.0.206
   Subnet Mask . . . . . . . . . . . : 255.255.255.0
   Default Gateway . . . . . . . . . : 192.168.0.1
"""

    assert discovery._parse_ipv4_lines(output, require_ipv4_label=True) == ["192.168.0.206"]


def test_set_udp_bind_ip_resets_stale_port_when_socket_is_closed():
    api = _api_with_socket(("UNK", 61600), "192.168.0.206", lambda: [])
    api.device.socket.m.port = 60713
    api.device.socket._socket = None

    discovery._set_udp_bind_ip(api, "192.168.0.206")

    assert api.device.socket.m.port == 0


def test_scan_zem_devices_falls_back_to_next_lan_interface(monkeypatch):
    observed_bind_ips = []

    def handshake():
        observed_bind_ips.append(api.device.socket.m.ip)
        if api.device.socket.m.ip == "10.10.0.5":
            return [("10.10.0.20", ZEM_DISCOVERY_PORT)]
        return []

    api = _api_with_socket(("192.168.1.20", 61600), "198.18.0.1", handshake)
    monkeypatch.setattr(
        "backend.cleanliness_core.microscope_discovery._local_ipv4_addresses",
        lambda: ["192.168.0.206", "10.10.0.5"],
    )

    assert scan_zem_devices(api) == [("10.10.0.20", ZEM_DISCOVERY_PORT)]
    assert observed_bind_ips == ["192.168.0.206", "10.10.0.5"]
    assert api.device.m.addr == ("192.168.1.20", 61600)


def test_scan_zem_devices_deduplicates_results(monkeypatch):
    def handshake():
        return [
            ("192.168.0.242", ZEM_DISCOVERY_PORT),
            ("192.168.0.242", ZEM_DISCOVERY_PORT),
        ]

    api = _api_with_socket(("UNK", 61600), "192.168.0.206", handshake)
    monkeypatch.setattr(
        "backend.cleanliness_core.microscope_discovery._local_ipv4_addresses",
        lambda: ["192.168.0.206"],
    )

    assert scan_zem_devices(api) == [("192.168.0.242", ZEM_DISCOVERY_PORT)]


def test_scan_zem_devices_ignores_failed_interface_when_another_scan_completed(monkeypatch):
    def handshake():
        if api.device.socket.m.ip == "192.168.0.206":
            raise OSError("cannot bind")
        return []

    api = _api_with_socket(("UNK", 61600), "192.168.0.206", handshake)
    monkeypatch.setattr(
        "backend.cleanliness_core.microscope_discovery._local_ipv4_addresses",
        lambda: ["192.168.0.206", "10.10.0.5"],
    )

    assert scan_zem_devices(api) == []
