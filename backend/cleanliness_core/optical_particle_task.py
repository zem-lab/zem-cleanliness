from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
from contextlib import contextmanager
from contextvars import ContextVar
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

import cv2
import numpy as np

from backend.cleanliness_core.atomic_json import write_json_atomic as _write_json_atomic
from backend.cleanliness_core.optical_particle_analysis import (
    Calibration,
    MatchDecision,
    OpticalParticle,
    annotate_sem_match,
    decode_optical_image,
    decide_particle_match,
    detect_particle_candidates,
    deterministic_fake_elements,
    load_calibration,
    load_optical_export,
    normalize_uint8,
    select_particles,
)
from backend.cleanliness_core.optical_particle_report import build_reports
from backend.cleanliness_core.microscope_safety import set_gun_state
from backend.cleanliness_core.route_optimizer import optimize_route


DEFAULT_CALIBRATION_FILENAME = "optical_particle_calibration.json"
ProgressCallback = Callable[[int, int, str], None]
StopCallback = Callable[[], bool]

ZEM_MIN_COMMAND_TIMEOUT_MS = 1_000
ZEM_STAGE_MOVE_ATTEMPTS = 3
ZEM_STAGE_POSITION_TOLERANCE_UM = 2.0
ZEM_STAGE_STABILITY_TOLERANCE_UM = 0.5
ZEM_STAGE_CONFIRM_INTERVAL_SECONDS = 0.3
ZEM_STAGE_CONFIRM_TIMEOUT_SECONDS = 3.0

_ZEM_COMMAND_LOCK = threading.RLock()
_ZEM_SINGLE_ATTEMPT = ContextVar("zem_single_attempt", default=False)


@contextmanager
def zemrpc_single_attempt():
    """MCP operations must not replay an actuator command after a lost reply."""
    token = _ZEM_SINGLE_ATTEMPT.set(True)
    try:
        yield
    finally:
        _ZEM_SINGLE_ATTEMPT.reset(token)


def _configure_zemrpc_timeouts() -> None:
    """Raise pyzemrpc's short UDP command timeouts to a practical minimum.

    pyzemrpc 1.2.6 starts its response timer before the asynchronous writer has
    finished selecting/binding the local interface.  On Windows that preparation
    can itself exceed the library's 100 ms default, causing a valid response to
    arrive after its subscriber has already been removed.
    """
    from zemrpc.zem.conn.protocol import EnumCmd

    current_getter = EnumCmd.timeout.fget
    if current_getter is None or getattr(
        current_getter, "_zem_cleanliness_timeout_patch", False
    ):
        return

    def reliable_timeout(command) -> int:
        return max(int(current_getter(command)), ZEM_MIN_COMMAND_TIMEOUT_MS)

    reliable_timeout._zem_cleanliness_timeout_patch = True  # type: ignore[attr-defined]
    EnumCmd.timeout = property(reliable_timeout)


def _configure_zemrpc_device(device: Any) -> None:
    """Serialize requests and complete interface preparation before timing."""
    if getattr(device, "_zem_cleanliness_reliability_patch", False):
        return

    original_pubsub_send = device.pubsub_send

    def reliable_pubsub_send(request):
        with _ZEM_COMMAND_LOCK:
            if getattr(getattr(request, "cmd", None), "name", "") != "Handshake":
                device._prepare_comm()
            single = getattr(original_pubsub_send, "__wrapped__", None)
            if _ZEM_SINGLE_ATTEMPT.get() and single is not None:
                return single(device, request)
            return original_pubsub_send(request)

    device.pubsub_send = reliable_pubsub_send
    # The vendor HTTP helper swallows requests errors and passes None to
    # json.loads, losing the actual cause. Keep microscope LAN traffic direct,
    # bounded and explicit; never automatically replay a Snap request here.
    if callable(getattr(device,'zem_rpc',None)):
        import requests
        direct_http = requests.Session()
        direct_http.trust_env = False
        def reliable_zem_rpc(payload):
            url=device.m.protocol_ng.url
            try:
                response=direct_http.post(url,json=payload,timeout=(5,90))
                response.raise_for_status()
                data=response.json()
            except (requests.RequestException,ValueError) as error:
                raise RuntimeError(f"电镜 HTTP {payload.get('method','RPC')} 请求失败：{error}") from error
            if not isinstance(data,dict):
                raise RuntimeError('电镜 HTTP RPC 返回格式错误')
            return data
        device.zem_rpc=reliable_zem_rpc
    device._zem_cleanliness_reliability_patch = True


def _cache_zemrpc_interface(device: Any, host: str, port: int) -> str | None:
    """Remember the interface proven by the successful status probe."""
    try:
        local_ip = str(device.socket.sock.getsockname()[0])
    except Exception:
        return None
    if not local_ip or local_ip == "0.0.0.0":
        return None
    interface_cache = getattr(device, "_iface_by_device", None)
    if isinstance(interface_cache, dict):
        interface_cache[(str(host), int(port))] = local_ip
    return local_ip


def default_calibration_path() -> Path:
    source_root = Path(__file__).resolve().parents[2]
    resource_root = Path(
        os.environ.get("ZEM_RESOURCE_ROOT", str(source_root / "resources"))
    )
    candidates = (
        resource_root / "conf" / DEFAULT_CALIBRATION_FILENAME,
        Path(sys.prefix)
        / "data"
        / "zem_cleanliness"
        / "conf"
        / DEFAULT_CALIBRATION_FILENAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def default_task_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path.cwd()
        / "output"
        / "optical_particle_tasks"
        / f"optical_sem_{stamp}"
    )


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(value or ""))
    if not match:
        return None
    try:
        parsed = float(match.group(0))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


@dataclass(frozen=True)
class TaskConfig:
    input_path: Path
    output_dir: Path
    calibration_path: Path
    host: str = "192.168.0.15"
    port: int = 61557
    sample_count: int | None = 10
    sequences: tuple[int, ...] | None = None
    exclude_sequences: tuple[int, ...] = (1, 2)
    pixel_avg: int = 7
    max_capture_attempts: int = 2
    restore_start_position: bool = True


class Microscope(Protocol):
    host: str
    port: int

    def connect(self) -> dict[str, Any]: ...

    def get_position(self) -> tuple[float, float]: ...

    def move_xy(self, x: float, y: float) -> dict[str, Any]: ...

    def capture(self, raw_path: Path) -> tuple[np.ndarray, dict[str, Any]]: ...

    def restore_video(self) -> None: ...

    def gun_on(self) -> str: ...

    def gun_off(self) -> str: ...

    def get_hv_status(self) -> str: ...


class ZemMicroscope:
    def __init__(
        self,
        host: str,
        port: int,
        imaging: dict[str, Any],
        *,
        pixel_avg: int = 7,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.imaging = dict(imaging)
        self.pixel_avg = int(pixel_avg)
        self._api = None
        self._image_size_enum = None

    def connect(self) -> dict[str, Any]:
        from zemrpc.zem.models import EnumImageSize
        from zemrpc.zem.zem_api import zem_api

        from backend.cleanliness_core.microscope_discovery import validate_zem_status

        _configure_zemrpc_timeouts()
        _configure_zemrpc_device(zem_api.device)
        status = validate_zem_status(zem_api, (self.host, self.port))
        if str(status).lower() != "idle":
            raise RuntimeError(f"电镜当前状态不是 Idle：{status}")
        local_ip = _cache_zemrpc_interface(
            zem_api.device, self.host, self.port
        )
        self._api = zem_api
        sync_done = bool(
            getattr(
                zem_api,
                "_init_sync_done",
                getattr(zem_api, "_init_sync", False),
            )
        )
        if not sync_done:
            zem_api.sync()

        expected_shape = list(self.imaging.get("imageShape") or [512, 512])
        image_size_map = {
            (512, 512): EnumImageSize.S512x512,
            (1024, 1024): EnumImageSize.S1024x1024,
            (2048, 2048): EnumImageSize.S2048x2048,
        }
        shape_key = tuple(map(int, expected_shape))
        if shape_key not in image_size_map:
            raise ValueError(f"不支持的标定图像尺寸：{expected_shape}")
        self._image_size_enum = image_size_map[shape_key]

        zem_api.sync_stage_and_scan_field()
        expected_mag = _as_float(self.imaging.get("magnification"))
        current_mag = _as_float(getattr(zem_api.device.m, "mag", None))
        if expected_mag is not None and (
            current_mag is None
            or not math.isclose(current_mag, expected_mag, abs_tol=0.1)
        ):
            zem_api.set_mag(int(round(expected_mag)))
            zem_api.sync_stage_and_scan_field()
            current_mag = _as_float(getattr(zem_api.device.m, "mag", None))
        if expected_mag is not None and (
            current_mag is None
            or not math.isclose(current_mag, expected_mag, abs_tol=0.1)
        ):
            raise RuntimeError(
                f"无法切换到标定倍率：当前 {current_mag}，标定 {expected_mag}"
            )

        scan_info = getattr(zem_api.device.m, "scan_info", None)
        scan_um = _as_float(getattr(scan_info, "scan_um", None))
        expected_field = _as_float(self.imaging.get("fieldSizeUm"))
        field_mismatch = expected_field is not None and (
            scan_um is None
            or not math.isclose(scan_um, expected_field, rel_tol=0.002, abs_tol=0.5)
        )
        if field_mismatch and expected_mag is not None:
            # 个别设备会出现“倍率回读已更新、扫描视场仍是上一倍率”的不同步状态。
            # SetMag 应答早于扫描场更新，重发目标倍率后轮询，避免把过渡态误判为失败。
            zem_api.set_mag(int(round(expected_mag)))
            deadline = time.monotonic() + 5.0
            while True:
                time.sleep(0.2)
                zem_api.sync_stage_and_scan_field()
                current_mag = _as_float(getattr(zem_api.device.m, "mag", None))
                scan_info = getattr(zem_api.device.m, "scan_info", None)
                scan_um = _as_float(getattr(scan_info, "scan_um", None))
                if scan_um is not None and math.isclose(
                    scan_um, expected_field, rel_tol=0.002, abs_tol=0.5
                ):
                    break
                if time.monotonic() >= deadline:
                    break
        if expected_field is not None and (
            scan_um is None
            or not math.isclose(scan_um, expected_field, rel_tol=0.002, abs_tol=0.5)
        ):
            raise RuntimeError(
                f"当前扫描视场与标定不一致：{scan_um} μm != {expected_field} μm"
            )
        position = self.get_position()
        return {
            "status": str(status),
            "position": {"x": position[0], "y": position[1]},
            "magnification": current_mag,
            "fieldSizeUm": scan_um,
            "localIp": local_ip,
            "minimumCommandTimeoutMs": ZEM_MIN_COMMAND_TIMEOUT_MS,
        }

    @property
    def api(self):
        if self._api is None:
            raise RuntimeError("电镜尚未连接")
        return self._api

    def get_position(self) -> tuple[float, float]:
        getter = getattr(self.api, "get_stage_pos", None)
        position = getter() if callable(getter) else self.api.device.get_stage_pos()
        result = float(position.x), float(position.y)
        if not all(math.isfinite(v) for v in result):
            raise ValueError('样品台回读坐标无效')
        return result

    def _confirmed_stage_position(
        self, target: tuple[float, float]
    ) -> tuple[tuple[float, float], float, bool]:
        deadline = time.monotonic() + ZEM_STAGE_CONFIRM_TIMEOUT_SECONDS
        previous: tuple[float, float] | None = None
        stable_reads = 0
        latest = self.get_position()

        while True:
            if previous is not None and math.hypot(
                latest[0] - previous[0], latest[1] - previous[1]
            ) <= ZEM_STAGE_STABILITY_TOLERANCE_UM:
                stable_reads += 1
            else:
                stable_reads = 1

            error_um = math.hypot(latest[0] - target[0], latest[1] - target[1])
            if stable_reads >= 2:
                return (
                    latest,
                    error_um,
                    error_um <= ZEM_STAGE_POSITION_TOLERANCE_UM,
                )
            if time.monotonic() >= deadline:
                return latest, error_um, False

            previous = latest
            time.sleep(ZEM_STAGE_CONFIRM_INTERVAL_SECONDS)
            latest = self.get_position()

    def move_xy(self, x: float, y: float) -> dict[str, Any]:
        before = self.get_position()
        target = (float(x), float(y))
        if not all(math.isfinite(v) for v in (*before, *target)):
            raise ValueError('样品台坐标必须是有限数值')
        attempts: list[dict[str, Any]] = []
        after = before
        error_um = math.hypot(after[0] - target[0], after[1] - target[1])
        for attempt in range(1, ZEM_STAGE_MOVE_ATTEMPTS + 1):
            move_started = time.perf_counter()
            if self.imaging.get('responsiveStagePolling'):
                self._move_with_responsive_polling(target)
            else:
                self.api.stage_xy_goto_closed(target, stable_after_idle_ms=300)
            idle_seconds = time.perf_counter()-move_started
            confirm_started = time.perf_counter()
            after, error_um, within_tolerance = self._confirmed_stage_position(target)
            attempts.append(
                {
                    "attempt": attempt,
                    "after": {"x": after[0], "y": after[1]},
                    "positionErrorUm": error_um,
                    "withinTolerance": within_tolerance,
                    "moveAndIdleSeconds": idle_seconds,
                    "positionConfirmationSeconds": time.perf_counter()-confirm_started,
                }
            )
            if within_tolerance:
                break
        return {
            "before": {"x": before[0], "y": before[1]},
            "target": {"x": target[0], "y": target[1]},
            "after": {"x": after[0], "y": after[1]},
            "positionErrorUm": error_um,
            "attempts": attempts,
        }

    def _move_with_responsive_polling(self, target):
        """Keep vendor start/settle delays; only shorten the idle polling gap."""
        from zemrpc.zem.conn.protocol import EnumStageStatus
        self.api.stage_xy_goto(target)
        time.sleep(.5)  # Vendor guard: give the movement time to start.
        deadline = time.monotonic()+30
        while True:
            status = self.api.device.get_stage_status()
            if time.monotonic() >= deadline:
                raise TimeoutError('等待样品台停止超时')
            if status == EnumStageStatus.Idle:
                break
            time.sleep(min(.2, max(0, deadline-time.monotonic())))
        time.sleep(.3)  # Same settling guard as stage_xy_goto_closed(..., 300).

    def capture(self, raw_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        from zemrpc.zem.models import SnapArgs

        if self._image_size_enum is None:
            raise RuntimeError("电镜采图参数尚未初始化")
        snap_args = SnapArgs(
            pixel_avg=self.pixel_avg,
            frame_avg=1,
            size=self._image_size_enum,
            save=False,
        )
        image = self.api.snap(snap_args=snap_args, save=False)
        if image is None or getattr(image, "img", None) is None:
            raise RuntimeError("电镜 Snap 未返回图像")
        raw_path = Path(raw_path).resolve()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        image.dump(raw_path, raw=True, u8=True, pkl=False)
        display_path = raw_path.with_name(f"{raw_path.stem}.u8.png")
        if not raw_path.is_file():
            raise RuntimeError(f"原始 SEM 图保存失败：{raw_path}")
        if not display_path.is_file():
            display = normalize_uint8(np.asarray(image.img))
            if not cv2.imwrite(str(display_path), display):
                raise RuntimeError(f"显示 SEM 图保存失败：{display_path}")
        metadata = _json_safe(image.meta2dict())
        bounds = getattr(image, "bounds", None)
        record = {
            "rawPath": str(raw_path),
            "displayPath": str(display_path),
            "shape": list(np.asarray(image.img).shape),
            "pixelSizeUm": float(image.pixel_size),
            "bounds": list(bounds) if bounds is not None else None,
            "meta": metadata,
        }
        return np.asarray(image.img).copy(), record

    def restore_video(self) -> None:
        try:
            restore = getattr(self.api, "video_play_on", None)
            if callable(restore):
                restore()
            else:
                self.api.device.video_play_on()
        except Exception:
            pass

    def get_hv_status(self) -> str:
        return str(self.api.device.get_hv_status())

    def gun_on(self) -> str:
        return set_gun_state(
            self.api,
            self.host,
            self.port,
            enabled=True,
            consecutive_reads=1,
        )

    def gun_off(self) -> str:
        return set_gun_state(
            self.api,
            self.host,
            self.port,
            enabled=False,
            consecutive_reads=3,
        )


def _capture_signature(capture: dict[str, Any]) -> dict[str, Any]:
    metadata = capture.get("meta") or {}
    return {
        "shape": list(capture.get("shape") or []),
        "pixelSizeUm": float(capture.get("pixelSizeUm") or 0.0),
        "detector": metadata.get("Detector"),
        "magnification": metadata.get("Magnification"),
        "fieldSizeUm": metadata.get("FieldSize"),
        "fieldRotationDeg": metadata.get("FieldRotation"),
        "kv": metadata.get("KV"),
    }


def _validate_calibrated_capture(
    capture: dict[str, Any],
    calibration: Calibration,
) -> None:
    signature = _capture_signature(capture)
    expected_shape = list(calibration.imaging.get("imageShape") or [512, 512])
    actual_shape = signature["shape"][:2]
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"图像尺寸与标定不一致：{actual_shape} != {expected_shape}"
        )
    expected_pixel = _as_float(calibration.imaging.get("pixelSizeUm"))
    if expected_pixel is not None and not math.isclose(
        signature["pixelSizeUm"],
        expected_pixel,
        rel_tol=0.01,
        abs_tol=0.01,
    ):
        raise RuntimeError(
            "像素尺寸与标定不一致："
            f"{signature['pixelSizeUm']} != {expected_pixel} μm/px"
        )
    expected_detector = str(calibration.imaging.get("detector") or "").strip()
    actual_detector = str(signature.get("detector") or "").strip()
    if expected_detector and actual_detector and (
        expected_detector.casefold() not in actual_detector.casefold()
    ):
        raise RuntimeError(
            f"探测器与标定不一致：{actual_detector!r} != {expected_detector!r}"
        )
    expected_rotation = _as_float(calibration.imaging.get("fieldRotationDeg"))
    actual_rotation = _as_float(signature.get("fieldRotationDeg"))
    if (
        expected_rotation is not None
        and actual_rotation is not None
        and not math.isclose(actual_rotation, expected_rotation, abs_tol=0.1)
    ):
        raise RuntimeError(
            f"FieldRotation 与标定不一致：{actual_rotation} != {expected_rotation}"
        )


def _validate_same_signature(
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> None:
    if list(current["shape"]) != list(baseline["shape"]):
        raise RuntimeError("批次内 SEM 图像尺寸发生变化")
    if not math.isclose(
        float(current["pixelSizeUm"]),
        float(baseline["pixelSizeUm"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("批次内 SEM 像素尺寸发生变化")
    for key in (
        "detector",
        "magnification",
        "fieldSizeUm",
        "fieldRotationDeg",
        "kv",
    ):
        if current.get(key) != baseline.get(key):
            raise RuntimeError(f"批次内 SEM 成像参数 {key} 发生变化")


def _base_record(
    particle: OpticalParticle,
    target: tuple[float, float],
    optical_relative: str,
) -> dict[str, Any]:
    return {
        "sequence": particle.sequence,
        "particleId": particle.particle_id,
        "label": particle.label,
        "status": "PENDING",
        "lengthUm": particle.length_um,
        "widthUm": particle.width_um,
        "areaUm2": particle.area_um2,
        "equivalentDiameterUm": particle.equivalent_diameter_um,
        "opticalAbPositionUm": {
            "x": particle.ab_x_um,
            "y": particle.ab_y_um,
        },
        "targetStageUm": {"x": target[0], "y": target[1]},
        "actualStageUm": None,
        "stagePositionErrorUm": None,
        "captureAttempts": [],
        "match": None,
        "localization": None,
        "energySpectrum": None,
        "images": {"opticalImage": optical_relative},
        "error": None,
    }


def _decision_rank(decision: MatchDecision) -> tuple[int, float]:
    status_rank = {
        "MATCHED_HIGH": 3,
        "MATCHED_MEDIUM": 2,
        "FAILED_LOW_CONFIDENCE": 1,
        "FAILED_NOT_FOUND": 0,
    }.get(decision.status, 0)
    score = (
        -float(decision.best.score)
        if decision.best is not None
        else -float("inf")
    )
    return status_rank, score


def run_optical_particle_task(
    config: TaskConfig,
    *,
    microscope: Microscope | None = None,
    progress: ProgressCallback | None = None,
    should_stop: StopCallback | None = None,
) -> dict[str, Any]:
    progress = progress or (lambda _index, _total, _message: None)
    should_stop = should_stop or (lambda: False)
    task_dir = Path(config.output_dir).resolve()
    task_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = task_dir / "task.json"
    if manifest_path.exists():
        raise FileExistsError(f"任务目录已存在 task.json：{task_dir}")

    optical_export = load_optical_export(config.input_path)
    calibration = load_calibration(config.calibration_path)
    calibration.validate_export(optical_export)
    expected_host = str(calibration.device.get("host") or "")
    expected_port = int(calibration.device.get("port") or config.port)
    if expected_host and config.host != expected_host:
        raise ValueError(
            f"电镜地址与标定设备不一致：{config.host} != {expected_host}"
        )
    if int(config.port) != expected_port:
        raise ValueError(
            f"电镜端口与标定设备不一致：{config.port} != {expected_port}"
        )
    selected = select_particles(
        optical_export,
        sequences=config.sequences,
        sample_count=config.sample_count,
        exclude_sequences=config.exclude_sequences,
    )
    if not selected:
        raise ValueError("没有选中任何待定位颗粒")

    optical_dir = task_dir / "optical"
    for particle in selected:
        decode_optical_image(
            particle,
            optical_dir / f"{particle.sequence:04d}.png",
        )

    started_at = datetime.now().astimezone().isoformat()
    manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "taskId": str(uuid.uuid4()),
        "status": "INITIALIZING",
        "startedAt": started_at,
        "completedAt": None,
        "input": {
            "path": str(optical_export.path),
            "sha256": optical_export.sha256,
            "projectId": optical_export.project_id,
            "workpieceId": optical_export.workpiece_id,
            "workpieceName": optical_export.workpiece_name,
            "coordinateUnit": optical_export.coordinate_unit,
            "coordinateSpace": optical_export.coordinate_space,
            "validParticleCount": len(optical_export.particles),
        },
        "calibration": calibration.to_audit_dict(),
        "device": {
            "host": config.host,
            "port": config.port,
            "live": True,
        },
        "settings": {
            "requestedSampleCount": config.sample_count,
            "selectedParticleCount": len(selected),
            "sequences": [particle.sequence for particle in selected],
            "excludeSequences": list(config.exclude_sequences),
            "pixelAvg": config.pixel_avg,
            "maxCaptureAttempts": config.max_capture_attempts,
            "restoreStartPosition": config.restore_start_position,
            "energySpectrumMode": "SIMULATED",
        },
        "startStageUm": None,
        "finalStageUm": None,
        "baselineImaging": None,
        "particles": [],
        "reports": None,
        "error": None,
    }
    _write_json_atomic(manifest_path, manifest)

    microscope = microscope or ZemMicroscope(
        config.host,
        config.port,
        calibration.imaging,
        pixel_avg=config.pixel_avg,
    )
    total = len(selected)
    start_position: tuple[float, float] | None = None
    consecutive_device_failures = 0
    progress(0, total, f"连接电镜 {config.host}:{config.port}")
    try:
        connection = microscope.connect()
        start_position = microscope.get_position()
        manifest["device"]["connection"] = connection
        manifest["startStageUm"] = {
            "x": start_position[0],
            "y": start_position[1],
        }
        manifest["status"] = "RUNNING"
        route_plan = optimize_route(
            selected,
            calibration.target_stage,
            start_position,
            return_to_start=config.restore_start_position,
        )
        ordered = list(route_plan.items)
        manifest["executionOrder"] = [
            particle.sequence for particle in ordered
        ]
        manifest["executionRoute"] = route_plan.to_audit_dict()
        _write_json_atomic(manifest_path, manifest)

        for index, particle in enumerate(ordered, start=1):
            if should_stop():
                manifest["status"] = "STOPPED"
                break
            target = calibration.target_stage(particle)
            optical_path = optical_dir / f"{particle.sequence:04d}.png"
            record = _base_record(
                particle,
                target,
                _relative(optical_path, task_dir),
            )
            manifest["particles"].append(record)
            progress(
                index - 1,
                total,
                f"[{index}/{total}] 移动到光镜颗粒 #{particle.sequence}",
            )

            if not calibration.target_inside_roi(target):
                record["status"] = "FAILED_STAGE_MOVE"
                record["match"] = {
                    "status": "FAILED_STAGE_MOVE",
                    "confidence": "low",
                    "reason": "预测 Stage 坐标超出标定有效 ROI",
                    "candidateCount": 0,
                }
                record["error"] = "预测 Stage 坐标超出标定有效 ROI"
                _write_json_atomic(manifest_path, manifest)
                continue

            try:
                move = microscope.move_xy(*target)
                record["actualStageUm"] = move["after"]
                record["stagePositionErrorUm"] = float(
                    move["positionErrorUm"]
                )
                if record["stagePositionErrorUm"] > 2.0:
                    raise RuntimeError(
                        "样品台回读误差超过 2 μm："
                        f"{record['stagePositionErrorUm']:.3f} μm"
                    )
            except Exception as error:
                record["status"] = "FAILED_STAGE_MOVE"
                record["match"] = {
                    "status": "FAILED_STAGE_MOVE",
                    "confidence": "low",
                    "reason": str(error),
                    "candidateCount": 0,
                }
                record["error"] = str(error)
                consecutive_device_failures += 1
                _write_json_atomic(manifest_path, manifest)
                if consecutive_device_failures >= 3:
                    raise RuntimeError("连续 3 次样品台移动失败，任务已停止") from error
                continue

            attempts: list[
                tuple[
                    MatchDecision,
                    np.ndarray,
                    dict[str, Any],
                    int,
                ]
            ] = []
            capture_error: Exception | None = None
            for attempt in range(1, max(1, config.max_capture_attempts) + 1):
                if should_stop():
                    break
                progress(
                    index - 1,
                    total,
                    f"[{index}/{total}] 采集并匹配 #{particle.sequence}"
                    f"（第 {attempt} 次）",
                )
                raw_path = (
                    task_dir
                    / "sem"
                    / f"{particle.sequence:04d}_attempt{attempt}.png"
                )
                try:
                    image, capture = microscope.capture(raw_path)
                    _validate_calibrated_capture(capture, calibration)
                    signature = _capture_signature(capture)
                    if manifest["baselineImaging"] is None:
                        manifest["baselineImaging"] = signature
                    else:
                        _validate_same_signature(
                            manifest["baselineImaging"],
                            signature,
                        )
                    pixel_size = float(capture["pixelSizeUm"])
                    candidates = detect_particle_candidates(
                        image,
                        expected_length_px=particle.length_um / pixel_size,
                        expected_width_px=particle.width_um / pixel_size,
                        expected_area_px=particle.area_um2 / (pixel_size**2),
                    )
                    decision = decide_particle_match(candidates)
                    capture_record = {
                        **capture,
                        "rawPath": _relative(
                            Path(capture["rawPath"]),
                            task_dir,
                        ),
                        "displayPath": _relative(
                            Path(capture["displayPath"]),
                            task_dir,
                        ),
                        "attempt": attempt,
                        "match": decision.to_dict(),
                    }
                    record["captureAttempts"].append(capture_record)
                    attempts.append((decision, image, capture_record, attempt))
                    capture_error = None
                    if decision.matched:
                        break
                except Exception as error:
                    capture_error = error
                    record["captureAttempts"].append(
                        {
                            "attempt": attempt,
                            "error": str(error),
                        }
                    )

            if not attempts:
                record["status"] = "FAILED_CAPTURE"
                record["match"] = {
                    "status": "FAILED_CAPTURE",
                    "confidence": "low",
                    "reason": str(capture_error or "采图已停止"),
                    "candidateCount": 0,
                }
                record["error"] = str(capture_error or "采图已停止")
                consecutive_device_failures += 1
                _write_json_atomic(manifest_path, manifest)
                if consecutive_device_failures >= 3:
                    raise RuntimeError("连续 3 个颗粒采图失败，任务已停止")
                continue

            consecutive_device_failures = 0
            decision, best_image, capture_record, best_attempt = max(
                attempts,
                key=lambda item: _decision_rank(item[0]),
            )
            annotated_path = (
                task_dir / "annotated" / f"{particle.sequence:04d}.png"
            )
            particle_path = (
                task_dir / "particles" / f"{particle.sequence:04d}.png"
            )
            image_outputs = annotate_sem_match(
                best_image,
                particle,
                decision,
                annotated_path,
                particle_path,
            )
            pixel_size = float(capture_record["pixelSizeUm"])
            center_error_px = (
                float(decision.best.distance_px)
                if decision.best is not None
                else None
            )
            record["match"] = decision.to_dict()
            record["match"]["selectedAttempt"] = best_attempt
            record["localization"] = {
                "centerErrorPx": center_error_px,
                "centerErrorUm": (
                    center_error_px * pixel_size
                    if center_error_px is not None
                    else None
                ),
                "pixelSizeUm": pixel_size,
                "safePointPx": image_outputs["safePointPx"],
                "safeBoundaryDistancePx": image_outputs[
                    "safeBoundaryDistancePx"
                ],
                "safeBoundaryDistanceUm": (
                    image_outputs["safeBoundaryDistancePx"] * pixel_size
                    if image_outputs["safeBoundaryDistancePx"] is not None
                    else None
                ),
            }
            record["images"].update(
                {
                    "semRawImage": capture_record["rawPath"],
                    "semDisplayImage": capture_record["displayPath"],
                    "semAnnotatedImage": _relative(
                        annotated_path,
                        task_dir,
                    ),
                    "semParticleImage": _relative(
                        particle_path,
                        task_dir,
                    ),
                }
            )
            if decision.matched:
                record["status"] = "SUCCEEDED"
                record["energySpectrum"] = deterministic_fake_elements(
                    particle
                )
                record["error"] = None
                consecutive_device_failures = 0
            else:
                record["status"] = decision.status
                record["error"] = decision.reason
            _write_json_atomic(manifest_path, manifest)
            progress(
                index,
                total,
                f"[{index}/{total}] #{particle.sequence} "
                f"{decision.status}",
            )

        if manifest["status"] == "RUNNING":
            failed = sum(
                record.get("status") != "SUCCEEDED"
                for record in manifest["particles"]
            )
            manifest["status"] = (
                "COMPLETED" if failed == 0 else "COMPLETED_WITH_FAILURES"
            )
    except Exception as error:
        manifest["status"] = "FAILED"
        manifest["error"] = str(error)
    finally:
        if (
            config.restore_start_position
            and start_position is not None
        ):
            try:
                progress(total, total, "恢复任务开始时的样品台位置")
                restore = microscope.move_xy(*start_position)
                manifest["stageRestore"] = restore
            except Exception as error:
                manifest["stageRestore"] = {"error": str(error)}
        try:
            final_position = microscope.get_position()
            manifest["finalStageUm"] = {
                "x": final_position[0],
                "y": final_position[1],
            }
        except Exception:
            pass
        try:
            microscope.restore_video()
        except Exception:
            pass

    manifest["completedAt"] = datetime.now().astimezone().isoformat()
    _write_json_atomic(manifest_path, manifest)
    try:
        report_paths = build_reports(task_dir, manifest)
        manifest["reports"] = {
            key: _relative(Path(value), task_dir)
            for key, value in report_paths.items()
        }
    except Exception as error:
        manifest["reportError"] = str(error)
    _write_json_atomic(manifest_path, manifest)
    progress(total, total, f"任务结束：{manifest['status']}")
    return manifest


def _parse_sequences(text: str | None) -> tuple[int, ...] | None:
    if not text:
        return None
    values = [
        int(part)
        for part in re.split(r"[\s,，;；]+", text.strip())
        if part
    ]
    return tuple(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据光镜颗粒 JSON 自动移动 SEM、拍图、匹配同一颗粒并生成双图报告"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=default_calibration_path(),
    )
    parser.add_argument("--host", default="192.168.0.15")
    parser.add_argument("--port", type=int, default=61557)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument(
        "--sequences",
        help="逗号或空格分隔的光镜 sequence；指定后忽略 sample-count",
    )
    parser.add_argument(
        "--exclude-sequences",
        default="1,2",
        help="自动抽样时排除的标定颗粒",
    )
    parser.add_argument("--pixel-avg", type=int, default=7)
    parser.add_argument("--max-capture-attempts", type=int, default=2)
    parser.add_argument(
        "--keep-final-position",
        action="store_true",
        help="任务结束后不恢复开始时的 Stage 位置",
    )
    args = parser.parse_args()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_task_dir().resolve()
    )
    config = TaskConfig(
        input_path=args.input.resolve(),
        output_dir=output_dir,
        calibration_path=args.calibration.resolve(),
        host=args.host,
        port=args.port,
        sample_count=args.sample_count,
        sequences=_parse_sequences(args.sequences),
        exclude_sequences=_parse_sequences(args.exclude_sequences) or (),
        pixel_avg=args.pixel_avg,
        max_capture_attempts=args.max_capture_attempts,
        restore_start_position=not args.keep_final_position,
    )

    def print_progress(index: int, total: int, message: str) -> None:
        print(f"[{index}/{total}] {message}", flush=True)

    manifest = run_optical_particle_task(config, progress=print_progress)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "taskId": manifest["taskId"],
                "outputDir": str(output_dir),
                "summary": manifest.get("summary"),
                "reports": manifest.get("reports"),
                "error": manifest.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if manifest["status"] == "FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
