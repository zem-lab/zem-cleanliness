"""Device operations behind the desktop backend; the MCP bridge owns no hardware."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend.cleanliness_core.device_control import DeviceBusy, DeviceCoordinator
from backend.sem_tools import TOOL_MAP
from backend.sem_rpc_catalog import READS, WRITES, catalog
from backend.sem_rpc_full import SPECS, WRITE_COMMANDS, STOPS, IMAGES, validate as validate_rpc, validate_paths


def stamp():
    return datetime.now(timezone.utc).isoformat()


class SemError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code

    def record(self):
        return {"code": self.code, "message": str(self)}


class SimulatedMicroscope:
    simulated = True

    def __init__(self):
        self.position = (0., 0.)
        self.commands = []
        self.rpc_values = {key: [0] for key in READS}
        self.rpc_values.update(GetMag=[100.], GetWD=[8.], GetOB=[14000], GetHvStatus=[1], GetStgStatus=[1],
                               AllowOpenGun=[True], IsChangingSamp=[0], GetModel=["SIMULATED"],
                               GetAst=[0, 0], GetWindowPos=[0, 0], GetBseCfg=[0] * 8, StatusCheck=["Idle"])

    def status(self):
        self.commands.append("status")
        return {"device_status": "Idle", "gun_state": self.rpc_values["GetHvStatus"][0], "stage_state": 1,
                "sample_changing": self.rpc_values["IsChangingSamp"][0],
                "magnification": self.rpc_values["GetMag"][0], "field_um": 512., "rotation_deg": 0.}

    def get_position(self):
        self.commands.append("position")
        return self.position

    def move(self, x, y):
        self.commands.append("move")
        self.position = (x, y)

    def capture(self, size, pixel_avg, directory):
        self.commands.append("capture")
        return self._image(size), {"pixel_size_um": 512. / size, "field_um": 512.,
                                  "magnification": 100., "rotation_deg": 0.}

    def navigation(self):
        self.commands.append("navigation")
        return self._image(512), {"bounds_um": [-256., -256., 256., 256.],
                                  "pixel_size_um": 1., "navigation_axis_convention": "simulated"}

    def rpc_read(self, command):
        self.commands.append(command)
        return list(self.rpc_values[command])

    def rpc_write(self, command, values):
        self.commands.append(command)
        spec = WRITES[command]
        if spec['verify'] is not None:
            self.rpc_values[spec["verify"]] = list(spec["expected"] if spec["expected"] is not None else values)
        else:
            return ['Acknowledge', 'Error Code:0']
        if command == "SetOB":
            self.rpc_values["GetOB"] = [round(values[0] / 100 * 65535)]
        if command in {"RequestOpenVac", "RequestCloseVac"}:
            self.rpc_values["IsChangingSamp"] = [int(command == "RequestOpenVac")]
        return []

    def rpc_call(self, command, values):
        if command in WRITES:
            return self.rpc_write(command, values)
        self.commands.append(command)
        if command == 'SetIe':
            raise SemError('UNSUPPORTED', '上位机实现返回 InvalidCmd')
        if command == 'ZemRpc':
            return {'jsonrpc': '2.0', 'id': values[0], 'result': {'name': 'SIMULATED', 'url': 'http://simulated.invalid'}}
        if command == 'GetNavImgBuf':
            return [8, 512, 512]
        return ['Acknowledge', 'Error Code:0']

    def rpc_image(self, command, values):
        self.commands.append(command)
        size = {0: 512, 1: 2048, 2: 1024, 3: 128}[values[3]] if command == 'Snap' else 512
        return self._image(size), {'rpc_command': command}

    @staticmethod
    def _image(size):
        import cv2
        import numpy as np
        image = np.zeros((size, size), dtype=np.uint16)
        cv2.circle(image, (size // 2, size // 2), size // 5, 45000, -1)
        cv2.putText(image, "SIMULATED", (12, size // 8), cv2.FONT_HERSHEY_SIMPLEX,
                    size / 700., 65535, max(1, size // 250))
        return image


class RealMicroscope:
    simulated = False

    def __init__(self, host, port):
        self.host, self.port = host, port

    def _reader(self):
        from backend.cleanliness_core.preflight import ZemPreflightReader
        return ZemPreflightReader(self.host, self.port)

    def rpc_read(self, command):
        from backend.sem_rpc_transport import exchange
        return exchange(self.host, self.port, command)

    def rpc_write(self, command, values):
        from backend.sem_rpc_transport import exchange
        return exchange(self.host, self.port, command, values)

    def rpc_call(self, command, values):
        from backend.sem_rpc_transport import exchange
        return exchange(self.host, self.port, command, values)

    def rpc_image(self, command, values):
        from backend.sem_rpc_transport import image_exchange
        return image_exchange(self.host, self.port, command, values)

    def status(self):
        reader = self._reader()
        status = str(reader.read("StatusCheck"))
        result = {"device_status": status}
        # A busy device must not be flooded with additional queries.
        if status.lower() != "idle":
            return result
        for field, command in {"gun_state": "GetHvStatus", "stage_state": "GetStgStatus",
                               "sample_changing": "IsChangingSamp", "magnification": "GetMag"}.items():
            try:
                result[field] = reader.read(command)
            except Exception as error:
                result[field] = None
                result.setdefault("unavailable", {})[field] = str(error)
        try:
            scan = reader.read("GetScanInfo") or {}
            result.update(field_um=scan.get("fieldSizeUm"), rotation_deg=scan.get("rotationDeg"))
        except Exception as error:
            result.setdefault("unavailable", {})["scan"] = str(error)
        return result

    def get_position(self):
        return self._reader().read("GetStagePos")

    def _api(self):
        from zemrpc.zem.zem_api import zem_api
        from backend.cleanliness_core.microscope_discovery import validate_zem_status
        from backend.cleanliness_core.optical_particle_task import (
            _configure_zemrpc_device, _configure_zemrpc_timeouts, _cache_zemrpc_interface,
        )
        _configure_zemrpc_timeouts()
        _configure_zemrpc_device(zem_api.device)
        if str(validate_zem_status(zem_api, (self.host, self.port))).lower() != "idle":
            raise SemError("DEVICE_BUSY", "电镜未空闲，未下发动作")
        _cache_zemrpc_interface(zem_api.device, self.host, self.port)
        # Deliberately never call ZemMicroscope.connect(), set_mag() or gun_on().
        return zem_api

    def move(self, x, y):
        from zemrpc.zem.models import StagePos
        from backend.cleanliness_core.optical_particle_task import zemrpc_single_attempt
        api = self._api()
        # Both stage_xy_goto_closed and pubsub_send have retry decorators.
        with zemrpc_single_attempt():
            api.device.set_stage_pos(StagePos(x=x, y=y))

    def capture(self, size, pixel_avg, directory):
        from zemrpc.zem.models import EnumImageSize, SnapArgs
        from backend.cleanliness_core.optical_particle_task import zemrpc_single_attempt, _json_safe
        api = self._api()
        api.device.get_model()
        api.device.get_hv()
        api.device.get_beam_size()
        api.device.get_protocol_ng()
        api.sync_stage_and_scan_field()
        image_size = {512: EnumImageSize.S512x512, 1024: EnumImageSize.S1024x1024,
                      2048: EnumImageSize.S2048x2048}[size]
        with zemrpc_single_attempt():
            image = api.snap(snap_args=SnapArgs(size=image_size, pixel_avg=pixel_avg, frame_avg=1, save=False),
                             save=False, raw=True)
        if image is None or getattr(image, "img", None) is None:
            raise RuntimeError("电镜 Snap 未返回图像")
        image.dump(directory / "driver-original.png", raw=True, u8=False, pkl=False)
        return image.img.copy(), {"pixel_size_um": float(image.pixel_size),
                       "bounds_um": list(image.bounds) if image.bounds is not None else None,
                       "driver_metadata": _json_safe(image.meta2dict()), "raw_file": "driver-original.png",
                       "raw_sha256": hashlib.sha256((directory / "driver-original.png").read_bytes()).hexdigest()}

    def navigation(self):
        api = self._api()
        api.sync_stage_and_scan_field()
        nav = api.get_nav_img()
        if nav is None or getattr(nav, "img", None) is None or getattr(nav, "bounds", None) is None:
            raise SemError("NO_NAVIGATION", "设备没有可用导航图，请在电镜软件中准备后重试")
        import cv2
        import numpy as np
        image = np.asarray(nav.img)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR if image.shape[2] == 4 else cv2.COLOR_RGB2BGR)
        return image, {"bounds_um": list(map(float, nav.bounds)),
                                             "navigation_axis_convention": "device_navigation"}


class SemControl:
    ACTIONS = {"sem_get_navigation", "sem_capture", "sem_move_xy", "sem_move_and_capture"}
    MOVES = {"sem_move_xy", "sem_move_and_capture"}
    CAPTURES = {"sem_capture", "sem_move_and_capture"}
    TERMINAL = {"completed", "failed", "cancelled", "unknown", "accepted"}

    def __init__(self, root: Path, device, coordinator: DeviceCoordinator, *, allow_capture=False,
                 bounds=None, max_move_um=None, poll_interval=.3, move_timeout=30.,
                 allow_rpc=(), rpc_limits=None, rpc_timeout=120., rpc_path_roots=()):
        # The vendor SDK may change cwd while initializing its application home.
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.device, self.coordinator = device, coordinator
        self.allow_capture = bool(allow_capture)
        self.bounds = tuple(bounds) if bounds is not None else None
        self.max_move_um = max_move_um
        self.allow_rpc = frozenset(allow_rpc)
        if not self.allow_rpc.issubset(WRITE_COMMANDS):
            raise ValueError("RPC 开放清单中有未知或上位机未实现的命令")
        self.rpc_limits = rpc_limits or {}
        self.rpc_path_roots = rpc_path_roots
        if not self._finite(rpc_timeout) or not 0 < rpc_timeout <= 600:
            raise ValueError("RPC 完成确认超时必须在 0..600 秒内")
        self.rpc_timeout = rpc_timeout
        if self.bounds is not None:
            if (len(self.bounds) != 4 or not all(self._finite(v) for v in self.bounds)
                    or self.bounds[0] >= self.bounds[2] or self.bounds[1] >= self.bounds[3]
                    or not self._finite(max_move_um) or max_move_um <= 0):
                raise ValueError("必须配置有效 XY 安全范围和单次最大移动距离（µm）")
        self.poll_interval, self.move_timeout = poll_interval, move_timeout
        self._lock = threading.RLock()
        self._job_changed = threading.Condition(self._lock)
        self._threads = set()
        self._closed = False
        self._db = sqlite3.connect(self.root / "operations.sqlite3", check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, caller TEXT NOT NULL, "
                         "request TEXT NOT NULL, payload TEXT NOT NULL, record TEXT NOT NULL, UNIQUE(caller,request))")
        self._db.execute("CREATE TABLE IF NOT EXISTS images (id TEXT PRIMARY KEY, caller TEXT NOT NULL, metadata TEXT NOT NULL)")
        for job_id, record_json in self._db.execute("SELECT id,record FROM jobs").fetchall():
            record = json.loads(record_json)
            if record["status"] in {"queued", "running", "cancelling"}:
                record.update(status="unknown", finished_at=stamp(), error={"code": "TIMEOUT_UNKNOWN",
                              "message": "后端曾中断，未重发指令；需人工核验实际执行结果"})
                self._save(record)
            if (record["status"] == "unknown" and record.get('writes_device', record["operation"] != "sem_get_navigation")
                    or record['status'] == 'accepted' and not record.get('operator_acknowledgement')):
                coordinator.quarantine(job_id)
        self._db.commit()

    @staticmethod
    def _finite(value):
        return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)

    @classmethod
    def _position(cls, position):
        if len(position) != 2 or not all(cls._finite(v) for v in position):
            raise SemError("INVALID_POSITION", "电镜台坐标回读无效")
        return tuple(map(float, position))

    def _save(self, record):
        with self._lock:
            self._db.execute("UPDATE jobs SET record=? WHERE id=?", (json.dumps(record, ensure_ascii=False, allow_nan=False), record["job_id"]))
            self._db.commit()
            self._job_changed.notify_all()

    def _job(self, job_id, caller):
        row = self._db.execute("SELECT record FROM jobs WHERE id=? AND caller=?", (job_id, caller)).fetchone()
        if row is None:
            raise SemError("NOT_FOUND", "作业不存在或不属于当前会话")
        return json.loads(row[0])

    def capabilities(self):
        return {"mode": "simulated" if self.device.simulated else "real", "coordinate_unit": "µm",
                "device_endpoint": None if self.device.simulated else {"host": self.device.host, "port": self.device.port},
                "coordinate_system": "SEM_STAGE_XY", "position_tolerance_um": 2.,
                "capture_enabled": self.allow_capture, "move_enabled": self.bounds is not None,
                "move_bounds_um": self.bounds, "max_move_um": self.max_move_um,
                "tools": list(TOOL_MAP), "unsupported": ["EDS", "SetIe", "arbitrary_rpc_tunnel"],
                "rpc_wrapped_count": len(SPECS), "rpc_call_tool": "sem_rpc_call",
                "rpc_write_enabled": sorted(self.allow_rpc), "rpc_catalog_count": len(catalog()),
                "verification_scope": "内部模拟设备" if self.device.simulated else "外部 RPC 连接；是否为物理设备需现场确认",
                "limits": ["基础工具不提供真空、腔门或各轴故障完整诊断", "取消请求不等于硬件已停止"],
                "unresolved_jobs": self.coordinator.unresolved}

    def call(self, name, arguments, caller="mcp"):
        from jsonschema import Draft202012Validator
        if name not in TOOL_MAP:
            raise SemError("UNSUPPORTED", "不支持的工具")
        errors = list(Draft202012Validator(TOOL_MAP[name]["inputSchema"]).iter_errors(arguments))
        if errors:
            raise SemError("INVALID_ARGUMENT", errors[0].message)
        if not isinstance(caller, str) or not caller or len(caller) > 128:
            raise SemError("INVALID_ARGUMENT", "调用方标识无效")
        if name == 'sem_rpc_call':
            return self._call_full_rpc(arguments, caller)
        if name == 'sem_rpc_read' and arguments['command'] not in READS:
            return self._call_full_rpc({'values': [], **arguments}, caller)
        if name == "sem_get_capabilities":
            return self.capabilities()
        if name == "sem_list_rpc":
            entries = catalog(self.allow_rpc)
            for entry in entries:
                if entry['command'] == 'SetStagePos':
                    entry['enabled'] = self.bounds is not None
                elif entry['command'] == 'Snap':
                    entry['enabled'] = entry['enabled'] and self.allow_capture
                entry['permission_note'] = 'enabled 表示基础权限；实际调用仍校验参数范围、路径及设备联锁'
            return {"source": "tribf_dev_hisi:cc785a3", "commands": [entry for entry in entries
                    if "kind" not in arguments or entry["kind"] == arguments["kind"]]}
        if name == "sem_wait_job":
            timeout = arguments.get("timeout_seconds", 15)
            if not self._finite(timeout):
                raise SemError("INVALID_ARGUMENT", "等待时间必须为有限数值")
            deadline = time.monotonic() + timeout
            with self._job_changed:
                while True:
                    record = self._job(arguments["job_id"], caller)
                    remaining = deadline - time.monotonic()
                    if record["status"] in self.TERMINAL or remaining <= 0:
                        return {**record, "wait_timed_out": record["status"] not in self.TERMINAL}
                    self._job_changed.wait(remaining)
        if name in {"sem_get_status", "sem_get_position", "sem_get_snapshot", "sem_rpc_read"}:
            started = time.monotonic()
            with self.coordinator.query(f"{caller}:{name}"):
                acquired = time.monotonic()
                try:
                    if name == "sem_rpc_read":
                        command = arguments["command"]
                        if command != "StatusCheck" and self.device.rpc_read("StatusCheck") != ["Idle"]:
                            raise SemError("DEVICE_BUSY", "设备未空闲，未继续读取参数")
                        result = {"command": command, "values": self.device.rpc_read(command),
                                  "description": READS[command][0], "unit": READS[command][1]}
                    elif name == "sem_get_position":
                        result = self._read_position()
                    else:
                        result = self._describe_status(self.device.status())
                        if name == "sem_get_snapshot" and str(result.get("device_status")).lower() == "idle":
                            try:
                                result.update(self._read_position())
                            except Exception as error:
                                result.setdefault("unavailable", {})["position_um"] = str(error)
                except SemError:
                    raise
                except Exception as error:
                    raise SemError("DEVICE_OFFLINE", f"设备查询失败：{error}") from error
            return {**result, "read_at": stamp(), "simulated": self.device.simulated,
                    "unresolved_jobs": self.coordinator.unresolved,
                    "timing_ms": {"queue": round((acquired - started) * 1000, 2),
                                  "device": round((time.monotonic() - acquired) * 1000, 2)}}
        with self._lock:
            if name == "sem_get_job":
                return self._job(arguments["job_id"], caller)
            if name == "sem_cancel_job":
                record = self._job(arguments["job_id"], caller)
                if record["status"] not in self.TERMINAL:
                    record.update(cancel_requested=True, status="cancelling",
                                  message="已请求取消，等待当前硬件指令结束；尚未确认设备停止")
                    self._save(record)
                return record
            if name == "sem_get_image":
                return self._image(arguments["image_id"], caller, arguments.get("variant", "preview"))
            return self._submit(name, arguments, caller)

    def _read_position(self):
        return {"position_um": self._position(self.device.get_position()), "coordinate_system": "SEM_STAGE_XY"}

    def _call_full_rpc(self, arguments, caller):
        command, values = arguments['command'], arguments['values']
        validate_rpc(command, values)
        spec = SPECS[command]
        if command == 'SetIe':
            raise SemError('UNSUPPORTED', 'SetIe 上位机实现明确返回 InvalidCmd，未伪造执行成功')
        if command in READS:
            return self.call('sem_rpc_read', {'command': command}, caller)
        if command in WRITES:
            return self.call('sem_rpc_execute', arguments, caller)
        if command == 'SetStagePos':
            return self.call('sem_move_xy', {'request_id': arguments['request_id'], 'x_um': values[0], 'y_um': values[1]}, caller)
        if spec['category'] == 'read':
            with self.coordinator.query(caller + ':' + command):
                if self.device.rpc_read('StatusCheck') != ['Idle']:
                    raise SemError('DEVICE_BUSY', '设备未空闲，未继续查询')
                try:
                    reply = self.device.rpc_call(command, values)
                    if command == 'ZemRpc' and not self.device.simulated:
                        try:
                            reply = json.loads(base64.b64decode(reply[0], validate=True).decode('utf-8'))
                            if not isinstance(reply, dict) or reply.get('id') != values[0] or not isinstance(reply.get('result'), dict):
                                raise ValueError('服务发现回包不匹配')
                        except (ValueError, IndexError, TypeError) as error:
                            raise SemError('INVALID_RESPONSE', 'ZemRpc 服务发现回包无效') from error
                    return {'command': command, 'values': reply, 'completion_confirmed': True,
                            'read_at': stamp(), 'simulated': self.device.simulated}
                except SemError:
                    raise
                except Exception as error:
                    raise SemError('DEVICE_OFFLINE', str(error)) from error
        with self._lock:
            return self._submit('sem_rpc_call', arguments, caller)

    def _full_rpc_permissions(self, command, values):
        spec = SPECS[command]
        if command in WRITE_COMMANDS and command not in self.allow_rpc:
            raise SemError('OPERATION_DISABLED', f'本机未开放 {command}，未下发命令')
        if command == 'Snap' and not self.allow_capture:
            raise SemError('OPERATION_DISABLED', '当前配置未开放采图')
        validate_paths(command, values, self.rpc_path_roots)
        if not self.device.simulated and command in WRITE_COMMANDS:
            numeric = [(index, v) for index, v in enumerate(values) if isinstance(v, (int, float))]
            limits = self.rpc_limits.get(command)
            if numeric and (not isinstance(limits, list) or len(limits) != len(values)):
                raise SemError('OPERATION_DISABLED', '未配置逐参数的上位机原始参数范围 rpcLimits')
            for index, value in numeric:
                bounds = limits[index]
                if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                        or not all(self._finite(v) for v in bounds) or bounds[0] > bounds[1]):
                    raise SemError('OPERATION_DISABLED', 'RPC 参数范围配置无效')
                if not bounds[0] <= value <= bounds[1]:
                    raise SemError('OUT_OF_RANGE', 'RPC 参数超出本机范围，未下发命令')

    @staticmethod
    def _describe_status(status):
        labels = {"gun_state": {0: "关闭", 1: "开启", 2: "切换中"},
                  "stage_state": {0: "运行中", 1: "静止"},
                  "sample_changing": {0: "未换样", 1: "换样中", -1: "未知"}}
        return {**status, "state_text": {key: values.get(status.get(key), "未知") for key, values in labels.items()},
                "diagnostic_limits": "样品台静止不代表真空、腔门及所有机械轴健康"}

    def _submit(self, name, arguments, caller):
        if self._closed:
            raise SemError("SERVICE_STOPPING", "服务正在停止，不接收新操作")
        try:
            payload = json.dumps({"operation": name, "arguments": arguments}, sort_keys=True, allow_nan=False)
        except (ValueError, TypeError) as error:
            raise SemError("INVALID_ARGUMENT", "参数必须是有效 JSON，坐标必须为有限数值") from error
        row = self._db.execute("SELECT id,payload FROM jobs WHERE caller=? AND request=?", (caller, arguments["request_id"])).fetchone()
        if row:
            if row[1] != payload:
                raise SemError("REQUEST_CONFLICT", "相同请求标识已用于不同参数，不会执行新动作")
            return self._job(row[0], caller)
        if name == 'sem_rpc_call':
            self._full_rpc_permissions(arguments['command'], arguments['values'])
        if name == "sem_rpc_execute":
            command, values = arguments["command"], arguments["values"]
            if command not in self.allow_rpc:
                raise SemError("OPERATION_DISABLED", f"本机未开放 {command}；不会下发命令")
            if not all(self._finite(value) for value in values):
                raise SemError("INVALID_ARGUMENT", "RPC 参数必须为有限数值")
            limits = self.rpc_limits.get(command)
            if values and not self.device.simulated:
                if not isinstance(limits, list) or len(limits) != len(values):
                    raise SemError("OPERATION_DISABLED", "该命令尚未配置经现场确认的逐参数范围")
                for value, bounds in zip(values, limits):
                    if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                            or not all(self._finite(v) for v in bounds) or bounds[0] > bounds[1]):
                        raise SemError("OPERATION_DISABLED", "本机 RPC 参数范围配置无效")
                    if not bounds[0] <= value <= bounds[1]:
                        raise SemError("OUT_OF_RANGE", "RPC 参数超出本机配置范围，未下发命令")
        if name in self.CAPTURES and not self.allow_capture:
            raise SemError("OPERATION_DISABLED", "当前配置未开放采图；不会自动开枪")
        if name in self.MOVES:
            if self.bounds is None:
                raise SemError("OPERATION_DISABLED", "尚未配置经现场确认的 XY 安全范围和单次移动上限")
            x, y = arguments["x_um"], arguments["y_um"]
            if not all(self._finite(v) for v in (x, y)):
                raise SemError("INVALID_ARGUMENT", "坐标必须是有限数值")
            if not (self.bounds[0] <= x <= self.bounds[2] and self.bounds[1] <= y <= self.bounds[3]):
                raise SemError("OUT_OF_RANGE", "目标坐标超出配置的 XY 安全范围，未下发动作")
        job_id = uuid.uuid4().hex
        owner = f"mcp:{job_id}"
        if name == 'sem_rpc_call' and arguments['command'] in STOPS:
            self.coordinator.acquire_interrupt(owner)
        else:
            self.coordinator.acquire(owner, writes=name != "sem_get_navigation" and
                                     not (name == 'sem_rpc_call' and SPECS[arguments['command']]['category'] == 'image_read'))
        try:
            record = {"job_id": job_id, "operation": name, "request_id": arguments["request_id"],
                      "status": "queued", "phase": "queued", "cancel_requested": False,
                      "writes_device": name != 'sem_get_navigation' and not
                          (name == 'sem_rpc_call' and SPECS[arguments['command']]['category'] == 'image_read'),
                      "created_at": stamp(), "simulated": self.device.simulated}
            self._db.execute("INSERT INTO jobs VALUES (?,?,?,?,?)", (job_id, caller, arguments["request_id"], payload,
                                                                      json.dumps(record, ensure_ascii=False)))
            self._db.commit()
            thread = threading.Thread(target=self._execute, args=(job_id, name, arguments, caller, owner),
                                      name=f"sem-mcp-{job_id[:8]}", daemon=True)
            self._threads.add(thread)
            thread.start()
            return record.copy()
        except BaseException:
            self.coordinator.release(owner)
            raise

    def _check_cancel(self, job_id, caller):
        with self._lock:
            if self._job(job_id, caller)["cancel_requested"]:
                raise SemError("CANCELLED", "已在安全边界结束，未执行后续动作")

    def _phase(self, job_id, caller, phase):
        with self._lock:
            record = self._job(job_id, caller)
            record.update(status="cancelling" if record["cancel_requested"] else "running", phase=phase)
            self._save(record)

    def _ready(self, status, *, capture=False):
        # capture 仅保留调用语义；采图/对焦不再要求电子枪已开启。
        if str(status.get("device_status")).lower() != "idle":
            raise SemError("DEVICE_BUSY", f"电镜未空闲：{status.get('device_status')}")
        if status.get("stage_state") != 1 or status.get("sample_changing") != 0:
            raise SemError("PRECHECK_FAILED", "未确认样品台静止且未换样，未下发动作")

    def _execute(self, job_id, name, arguments, caller, owner):
        started = time.monotonic()
        hardware_sent = False
        result = {}
        error = None
        state = "completed"
        try:
            self._check_cancel(job_id, caller)
            self._phase(job_id, caller, "precheck")
            if name == 'sem_rpc_call':
                command, values = arguments['command'], arguments['values']
                if command not in STOPS:
                    status = self.device.status()
                    if SPECS[command]['category'] == 'vacuum':
                        if (str(status.get('device_status')).lower() != 'idle' or status.get('gun_state') != 0
                                or status.get('stage_state') != 1 or status.get('sample_changing') not in (0, 1)):
                            raise SemError('PRECHECK_FAILED', '真空操作须确认关枪、台静止及已知换样状态')
                    else:
                        self._ready(status, capture=command in {'Snap', 'SetAutoSnapTask', 'StartScanPoints'})
                self._check_cancel(job_id, caller)
                self._phase(job_id, caller, 'rpc_dispatch')
                hardware_sent = command not in {'GetImg', 'GetNavImg'}
                if command in IMAGES:
                    before = self._position(self.device.get_position())
                    image, metadata = self.device.rpc_image(command, values)
                    after = self._position(self.device.get_position())
                    metadata.update(source='simulated' if self.device.simulated else 'external_rpc',
                                    simulated=self.device.simulated, rpc_command=command, captured_at=stamp(),
                                    job_id=job_id, request_id=arguments['request_id'], operation=name,
                                    coordinate_system='SEM_STAGE_XY',
                                    position_um=before if math.dist(before, after) <= .5 else None,
                                    position_before_um=before, position_after_um=after, position_verified=math.dist(before, after) <= .5)
                    directory = self.root / 'images' / job_id
                    directory.mkdir(parents=True, exist_ok=True)
                    result['image'] = self._store_image(image, metadata, directory, caller)
                    if not metadata['position_verified']:
                        raise SemError('POSITION_CHANGED', '图像传输期间坐标变化，保留图像供核验')
                    if command == 'Snap' and (image.shape[0], image.shape[1]) != ({0:512,1:2048,2:1024,3:128}[values[3]],) * 2:
                        raise SemError('INVALID_IMAGE', 'Snap 图像尺寸与参数不一致')
                    result['rpc'] = {'command': command, 'arguments': values, 'completion_confirmed': True, 'verified': True}
                else:
                    acknowledgement = self.device.rpc_call(command, values)
                    if acknowledgement != ['Acknowledge', 'Error Code:0']:
                        raise SemError('INVALID_RESPONSE', '未收到明确的命令受理回执，不能确认是否执行')
                    result['rpc'] = {'command': command, 'arguments': values, 'acknowledgement': acknowledgement,
                                     'accepted': True, 'completion_confirmed': False, 'verified': False,
                                     'message': '命令已受理；上位机无完成回读，不能确认生效或物理停止，请现场核验',
                                     'requires_operator_verification': True}
                    state = 'accepted'
            elif name == "sem_rpc_execute":
                self._rpc_precheck(arguments["command"])
                self._check_cancel(job_id, caller)
                self._phase(job_id, caller, "rpc_dispatch")
                hardware_sent = True
                acknowledgement = self.device.rpc_write(arguments["command"], arguments["values"])
                if WRITES[arguments['command']]['verify'] is None:
                    if acknowledgement != ['Acknowledge', 'Error Code:0']:
                        raise SemError('INVALID_RESPONSE', '未收到明确的命令受理回执，不能确认是否执行')
                    result['rpc'] = {'command': arguments['command'], 'arguments': arguments['values'],
                        'acknowledgement': acknowledgement, 'accepted': True, 'completion_confirmed': False,
                        'verified': False, 'requires_operator_verification': True,
                        'message': '拍照逐点平均档位已受理；GetAvLenth 读取视频档位，不能用于完成确认'}
                    state = 'accepted'
                else:
                    self._phase(job_id, caller, "rpc_verification")
                    result["rpc"] = self._verify_rpc(arguments["command"], arguments["values"], acknowledgement)
                    self._check_cancel(job_id, caller)
            else:
                self._execute_basic(job_id, name, arguments, caller, result)
            result["acknowledged_at" if state == 'accepted' else "completed_at"] = stamp()
        except SemError as exception:
            error = exception.record()
            state = "cancelled" if exception.code == "CANCELLED" else "failed"
            if exception.code in {"POSITION_CHANGED", "POSITION_UNCONFIRMED", "RPC_UNCONFIRMED"}:
                state = "unknown"
            if name in {"sem_rpc_execute", "sem_rpc_call"} and hardware_sent and exception.code != "CANCELLED":
                state = "unknown"
                if name == 'sem_rpc_call' and arguments['command'] != 'Snap' and exception.code in {'RPC_REJECTED', 'UNSUPPORTED', 'INVALID_ARGUMENT', 'DEVICE_BUSY'}:
                    state = 'failed'
        except Exception as exception:
            # Basic operations set their sent state on the local execution context.
            hardware_sent = hardware_sent or result.pop("_hardware_sent", False)
            state = "unknown" if hardware_sent else "failed"
            error = {"code": "TIMEOUT_UNKNOWN" if hardware_sent else "DEVICE_OFFLINE", "message": str(exception)}
        finally:
            result.pop("_hardware_sent", None)
            with self._lock:
                try:
                    record = self._job(job_id, caller)
                    record.update(status=state, result=result, error=error, finished_at=stamp(),
                                  execution_ms=round((time.monotonic() - started) * 1000, 2))
                    try:
                        self._save(record)
                    except BaseException:
                        self.coordinator.quarantine(job_id)
                        raise
                    if state in {"unknown", "accepted"} and record.get('writes_device', True):
                        self.coordinator.quarantine(job_id)
                finally:
                    self.coordinator.release(owner)
                    self._threads.discard(threading.current_thread())

    def _execute_basic(self, job_id, name, arguments, caller, result):
        status = self.device.status()
        self._ready(status, capture=name in self.CAPTURES)
        before = self._position(self.device.get_position())
        self._check_cancel(job_id, caller)
        if name in self.MOVES:
            target = (arguments["x_um"], arguments["y_um"])
            if not (self.bounds[0] <= before[0] <= self.bounds[2] and self.bounds[1] <= before[1] <= self.bounds[3]):
                raise SemError("OUT_OF_RANGE", "当前台坐标不在配置的安全区域内，未下发移动")
            if math.dist(before, target) > self.max_move_um:
                raise SemError("OUT_OF_RANGE", "移动距离超过配置的单次上限，未下发动作")
            self._phase(job_id, caller, "moving")
            result["_hardware_sent"] = True
            self.device.move(*target)
            after = self._confirm_position(target)
            result["movement"] = {"before_um": before, "target_um": target, "after_um": after,
                                  "position_error_um": math.dist(target, after)}
            self._check_cancel(job_id, caller)
        if name in self.CAPTURES or name == "sem_get_navigation":
            capture_status = self.device.status()
            self._ready(capture_status, capture=name in self.CAPTURES)
            position = self._position(self.device.get_position())
            if name in self.MOVES and math.dist(position, (arguments["x_um"], arguments["y_um"])) > 2.:
                raise SemError("POSITION_UNCONFIRMED", "采图前台坐标已偏离移动目标，未采图；请核验设备实际位置")
            directory = self.root / "images" / job_id
            directory.mkdir(parents=True, exist_ok=True)
            self._phase(job_id, caller, "capturing" if name in self.CAPTURES else "navigation")
            self._check_cancel(job_id, caller)
            result["_hardware_sent"] = name in self.CAPTURES or result.get("_hardware_sent", False)
            image, metadata = (self.device.navigation() if name == "sem_get_navigation" else
                               self.device.capture(arguments.get("size", 512), arguments.get("pixel_avg", 7), directory))
            after_capture = self._position(self.device.get_position())
            position_verified = math.dist(position, after_capture) <= .5
            metadata.update(position_um=position if position_verified else None,
                            position_before_um=position, position_after_um=after_capture, position_verified=position_verified,
                            coordinate_system="SEM_STAGE_XY", captured_at=stamp(),
                            simulated=self.device.simulated, source="simulated" if self.device.simulated else "microscope",
                            operation=name, job_id=job_id, request_id=arguments["request_id"])
            for field in ("magnification", "field_um", "rotation_deg"):
                metadata.setdefault(field, capture_status.get(field))
            result["image"] = self._store_image(image, metadata, directory, caller)
            if not position_verified:
                raise SemError("POSITION_CHANGED", "采图期间台坐标变化；原图已保留，但不能作为可靠定位结果")
            if name in self.CAPTURES and metadata["width"] != arguments.get("size", 512):
                raise SemError("INVALID_IMAGE", "设备返回尺寸与请求不一致；原图已保留，采图不判定成功")
            if name in self.CAPTURES and metadata["height"] != arguments.get("size", 512):
                raise SemError("INVALID_IMAGE", "设备返回尺寸与请求不一致；原图已保留，采图不判定成功")
            self._check_cancel(job_id, caller)

    def _rpc_precheck(self, command):
        status = self.device.status()
        category = WRITES[command]["category"]
        if category in {"sample_begin", "sample_end", "gun_off", "navigation_capture"}:
            if str(status.get("device_status")).lower() != "idle":
                raise SemError("DEVICE_BUSY", "设备未空闲，未下发命令")
            if category != "gun_off" and (status.get("gun_state") != 0 or status.get("stage_state") != 1):
                raise SemError("PRECHECK_FAILED", "换样必须先确认关枪及样品台静止")
            if category != "gun_off" and status.get("sample_changing") not in (0, 1):
                raise SemError("PRECHECK_FAILED", "未确认换样状态，未下发命令")
            # Physical optical navigation requires exchange; ZEM_SIM permits
            # idle panoramic capture. Keep the upstream device-specific guard.
        else:
            self._ready(status, capture=category == "autofocus")
        if category == "gun_on" and self.device.rpc_read("AllowOpenGun") != [True]:
            raise SemError("PRECHECK_FAILED", "设备联锁未明确允许开枪")

    def _verify_rpc(self, command, values, acknowledgement):
        spec = WRITES[command]
        target = list(spec["expected"] if spec["expected"] is not None else values)
        if command == "SetOB":
            target = [round(values[0] / 100 * 65535)]
        deadline = time.monotonic() + self.rpc_timeout
        last = None
        consecutive = 0
        while time.monotonic() < deadline:
            last = self.device.rpc_read(spec["verify"])
            if spec["category"] in {"sample_begin", "sample_end", "navigation_capture"} and last and last[0] == -1:
                raise SemError("RPC_UNCONFIRMED", f"{command} 设备报告失败：{last}")
            checked = last[:1] if spec["expected"] is not None else last
            matched = len(checked) == len(target) and all(
                isinstance(a, (int, float)) and isinstance(b, (int, float)) and
                math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-3 if command != "SetOB" else 1)
                for a, b in zip(checked, target))
            consecutive = consecutive + 1 if matched else 0
            if consecutive >= 2:
                if spec["category"] in {"sample_begin", "sample_end"}:
                    expected_sample = int(spec["category"] == "sample_begin")
                    if self.device.rpc_read("IsChangingSamp") != [expected_sample]:
                        consecutive = 0
                        continue
                return {"command": command, "arguments": values, "acknowledgement": acknowledgement,
                        "readback_command": spec["verify"], "readback": last, "verified": True}
            time.sleep(self.poll_interval)
        raise SemError("RPC_UNCONFIRMED", f"{command} 未确认完成；最后回读 {last}，不要重发，请核验设备")

    def _confirm_position(self, target):
        deadline = time.monotonic() + self.move_timeout
        previous = None
        while time.monotonic() <= deadline:
            position = self._position(self.device.get_position())
            status = self.device.status()
            if (status.get("stage_state") == 1 and previous is not None
                    and math.dist(position, previous) <= .5 and math.dist(position, target) <= 2.):
                return position
            previous = position
            time.sleep(self.poll_interval)
        raise SemError("POSITION_UNCONFIRMED", "移动后未确认稳定到位，未继续采图；请核验设备实际位置")

    def _store_image(self, image, metadata, directory, caller):
        import cv2
        import numpy as np
        array = np.asarray(image)
        if array.ndim not in (2, 3) or array.size == 0 or not np.isfinite(array).all():
            raise SemError("INVALID_IMAGE", "设备图像为空或包含无效数值")
        if array.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
            raise SemError("INVALID_IMAGE", "原图格式不是可无损保存的 uint8/uint16，不冒充原始 PNG")
        if not cv2.imwrite(str(directory / "original.png"), array):
            raise OSError("原图保存失败")
        preview = array if array.dtype == np.uint8 else cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if max(preview.shape[:2]) > 1024:
            scale = 1024 / max(preview.shape[:2])
            preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(directory / "preview.png"), preview):
            raise OSError("预览保存失败")
        image_id = directory.name
        metadata.update(image_id=image_id, width=int(array.shape[1]), height=int(array.shape[0]),
                        dtype=str(array.dtype), sha256=hashlib.sha256((directory / "original.png").read_bytes()).hexdigest(),
                        preview_sha256=hashlib.sha256((directory / "preview.png").read_bytes()).hexdigest())
        with self._lock:
            self._db.execute("INSERT INTO images VALUES (?,?,?)", (image_id, caller, json.dumps(metadata, allow_nan=False)))
            self._db.commit()
        return metadata

    def _image(self, image_id, caller, variant="preview"):
        row = self._db.execute("SELECT metadata FROM images WHERE id=? AND caller=?", (image_id, caller)).fetchone()
        if row is None:
            raise SemError("NOT_FOUND", "图像不存在或不属于当前会话")
        path = self.root / "images" / image_id / f"{variant}.png"
        data = path.read_bytes()
        expected_hash = json.loads(row[0])["sha256" if variant == "original" else "preview_sha256"]
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise SemError("IMAGE_CHANGED", "保存的图像指纹不匹配，不能当作原始采集结果")
        return {"metadata": json.loads(row[0]), "display_path": str((self.root / "images" / image_id / "preview.png").resolve()),
                "variant": variant,
                "image": {"mimeType": "image/png", "data": base64.b64encode(data).decode("ascii")}}

    def close(self, *, wait=True):
        with self._lock:
            self._closed = True
            threads = list(self._threads)
        if wait:
            for thread in threads:
                thread.join()
        if not any(thread.is_alive() for thread in threads):
            self._db.close()
