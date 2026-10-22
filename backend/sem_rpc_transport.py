"""Single-attempt typed RPC transport; never initializes the microscope SDK API."""
from __future__ import annotations

import math
import re
import socket


def parse_reply(command, data):
    from zemrpc.zem.conn.protocol import append_crc
    from backend.sem_control import SemError
    if not data.startswith(("$" + command + ":").encode()) or not data.endswith(b"%"):
        raise SemError("INVALID_RESPONSE", "电镜 RPC 回包命令不匹配或帧不完整")
    if len(data) < 6 or append_crc(data[:-5]) != data:
        raise SemError("INVALID_RESPONSE", "电镜 RPC 回包 CRC 校验失败")
    payload = data[len(command) + 2:-5].decode("utf-8", errors="strict").removesuffix(",")
    match = re.search(r"Error Code\s*:\s*(\d+)", payload, re.I)
    if match and int(match.group(1)) != 0:
        code = int(match.group(1))
        raise SemError({1: "INVALID_ARGUMENT", 3: "DEVICE_BUSY", 4: "INVALID_RESPONSE", 5: "UNSUPPORTED",
                        6: "UNSUPPORTED"}.get(code, "RPC_REJECTED"), f"{command} 上位机拒绝：{payload}")
    values = []
    for token in payload.split(",") if payload else []:
        text = token.strip()
        if text.lower() in {"true", "false"}:
            values.append(text.lower() == "true")
        elif re.fullmatch(r"[-+]?\d+", text):
            values.append(int(text))
        else:
            try:
                number = float(text)
            except ValueError:
                values.append(text)
            else:
                if not math.isfinite(number):
                    raise SemError("INVALID_RESPONSE", f"{command} 返回非有限数值")
                values.append(number)
    return values


def exchange(host, port, command, values=(), *, timeout=3.):
    from backend.sem_rpc_full import wire_values
    from backend.sem_control import SemError
    from backend.cleanliness_core.microscope_safety import _route_local_ip
    from zemrpc.zem.conn.protocol import append_crc
    values = wire_values(command, list(values))
    local_ip = _route_local_ip(host, port)
    target_ip = socket.gethostbyname(host)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as channel:
        channel.bind((local_ip, 0))
        channel.settimeout(timeout)
        args = ",".join(str(v) for v in values)
        frame = append_crc(f"${command}:{args},{local_ip}:{channel.getsockname()[1]},")
        channel.sendto(frame, (host, port))
        data, address = channel.recvfrom(16384)
        if address[0] != target_ip:
            raise SemError("INVALID_RESPONSE", "RPC 回包来源与目标设备不一致")
    return parse_reply(command, data)


def image_exchange(host, port, command, values=(), *, timeout=60.):
    """Receive the legacy image socket exactly once; no endpoint redirection."""
    import time
    import numpy as np
    from backend.sem_control import SemError
    from backend.sem_rpc_full import IMAGES
    if command not in IMAGES:
        raise SemError('INVALID_ARGUMENT', '不是图像 RPC')
    reply = exchange(host, port, command, values)
    if len(reply) != 4 or not isinstance(reply[0], str):
        raise SemError('INVALID_RESPONSE', '图像 RPC 缺少端点、位深或尺寸')
    match = re.fullmatch(r'ZEM20 Image service:([^:]+):(\d+)', reply[0])
    if not match or socket.gethostbyname(match[1]) != socket.gethostbyname(host) or not 0 < int(match[2]) <= 65535:
        raise SemError('INVALID_RESPONSE', '图像端点与目标电镜不一致')
    depth, width, height = reply[1:]
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (depth, width, height)) or not (0 < width <= 8192 and 0 < height <= 8192):
        raise SemError('INVALID_RESPONSE', '图像尺寸或位深无效')
    if depth not in {0, 7, 8, 24, 32}:
        raise SemError('INVALID_RESPONSE', '未知图像位深')
    channels = {24: 3, 32: 4}.get(depth, 1)
    dtype = np.dtype('<u2') if command == 'Snap' and depth == 7 else np.dtype('u1')
    size = width * height * channels * dtype.itemsize
    if size > 64 * 1024 * 1024:
        raise SemError('INVALID_RESPONSE', '图像超过 64 MiB 上限')
    data = bytearray()
    deadline = time.monotonic() + timeout
    with socket.create_connection((match[1], int(match[2])), timeout=min(3., timeout)) as stream:
        while len(data) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SemError('INVALID_IMAGE', '图像传输超时，未收到完整图像')
            stream.settimeout(remaining)
            chunk = stream.recv(min(65536, size - len(data)))
            if not chunk:
                raise SemError('INVALID_IMAGE', '图像传输提前结束')
            data.extend(chunk)
    shape = (height, width, channels) if channels > 1 else (height, width)
    array = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
    if channels > 1:
        import cv2
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2BGR if channels == 4 else cv2.COLOR_RGB2BGR)
    return array, {'rpc_command': command, 'wire_depth': depth, 'image_endpoint': reply[0], 'raw_transfer_bytes': size}
