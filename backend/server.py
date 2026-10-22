from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import hmac
import json
import zlib
from functools import lru_cache
import math
import os
import queue
import secrets
import socket
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


APP_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = Path(os.environ.get("ZEM_RESOURCE_ROOT", str(APP_ROOT / "resources"))).resolve()
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from backend.cleanliness_core.atomic_json import write_json_atomic
from backend.chat_runner import ChatError
from backend.cleanliness_core.device_control import BackendOwnerLock, DeviceBusy, DeviceCoordinator

TASK_ROOT = Path(os.environ.get("ZEM_TASK_ROOT", str(APP_ROOT / "tasks"))).resolve()
TASK_ROOT.mkdir(parents=True, exist_ok=True)
LOG_ROOT = Path(os.environ.get("ZEM_LOG_DIR", str(TASK_ROOT.parent / "logs"))).resolve()
LOG_ROOT.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_ROOT / "backend.jsonl"
LOG_ARCHIVE = LOG_ROOT / "backend.previous.jsonl"
LOG_LOCK = threading.RLock()
LOG_MAX_BYTES = 10 * 1024 * 1024
SESSION_ID = os.environ.get("ZEM_SESSION_ID") or str(uuid.uuid4())
SETTINGS_PATH = TASK_ROOT.parent / "settings.json"
ENGINEER_OVERRIDES_PATH = TASK_ROOT.parent / ".engineer-settings.json"
ENGINEER_PASSWORD_SHA256 = "82dbccadeabbd198734e4f516888996187418e265226e34fed4262e54e6f4fe7"
ENGINEER_SESSION_SECONDS = 30 * 60
ENGINEER_SESSIONS: dict[str, float] = {}
ENGINEER_FAILURES: list[float] = []
FAKE_EDS_ENABLED = str(os.environ.get("ZEM_FAKE_EDS", "")).strip().lower() in {"1", "true", "yes", "on"}
MICROSCOPE_STATUS_LOCK = threading.Lock()
SETTINGS_LOCK = threading.RLock()


def _safe_log_value(value: Any, depth: int = 0) -> Any:
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value), "traceback": traceback.format_exc()}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 8000 else f"{value[:8000]}...[truncated]"
    if depth >= 4:
        return "[max-depth]"
    if isinstance(value, dict):
        return {
            str(key): "[redacted]" if any(part in str(key).lower() for part in ("password", "token", "secret", "authorization", "api_key")) else _safe_log_value(item, depth + 1)
            for key, item in list(value.items())[:80]
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe_log_value(item, depth + 1) for item in list(value)[:50]]
    return str(value)


def log_event(level: str, event: str, **fields: Any) -> None:
    record = {
        "timestamp": now() if "now" in globals() else datetime.now().astimezone().isoformat(),
        "level": level,
        "process": "backend",
        "pid": os.getpid(),
        "sessionId": SESSION_ID,
        "thread": threading.current_thread().name,
        "event": event,
        **_safe_log_value(fields),
    }
    line = f"{json.dumps(record, ensure_ascii=False)}\n"
    try:
        with LOG_LOCK:
            if LOG_FILE.is_file() and LOG_FILE.stat().st_size >= LOG_MAX_BYTES:
                if LOG_ARCHIVE.exists():
                    LOG_ARCHIVE.unlink()
                LOG_FILE.replace(LOG_ARCHIVE)
            with LOG_FILE.open("a", encoding="utf-8") as handle:
                handle.write(line)
    except Exception as error:
        print(json.dumps({"level": "error", "event": "logging_failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr, flush=True)

ENGINEER_FIELDS: tuple[dict[str, Any], ...] = (
    {
        "path": "abPoint.fixtureConfidence", "group": "navigation", "groupLabel": "导航图检测",
        "label": "夹具检测置信度", "unit": "", "kind": "number", "minimum": 0.01, "maximum": 0.99, "step": 0.01,
        "summary": "导航图中的夹具候选框达到该置信度后才会保留。",
        "lowerEffect": "更容易检出偏暗或模糊的夹具，同时可能增加背景误检。",
        "higherEffect": "误检会减少，但图像质量较差时可能漏掉夹具。",
        "example": "夹具明显存在但没有检测框时，可从 0.10 调到 0.08；错误框较多时可尝试 0.15。",
        "risk": "medium",
    },
    {
        "path": "abPoint.holeConfidence", "group": "navigation", "groupLabel": "导航图检测",
        "label": "AB 孔检测置信度", "unit": "", "kind": "number", "minimum": 0.01, "maximum": 0.99, "step": 0.01,
        "summary": "控制夹具区域内 AB 孔候选被接受的最低置信度。",
        "lowerEffect": "可以保留更弱的孔特征，但也更容易把污点或纹理当成孔。",
        "higherEffect": "候选更可靠，但较暗、边缘不完整的孔可能被漏检。",
        "example": "只识别到一个孔时可从 0.10 降到 0.07；出现多个伪孔时可提高到 0.15。",
        "risk": "medium",
    },
    {
        "path": "abPoint.nmsIouThreshold", "group": "navigation", "groupLabel": "导航图检测",
        "label": "重叠框抑制阈值", "unit": "", "kind": "number", "minimum": 0.10, "maximum": 0.90, "step": 0.05,
        "summary": "决定两个重叠检测框在多大程度上被视为同一个目标。",
        "lowerEffect": "更积极地合并重叠框，同一目标的重复框更少。",
        "higherEffect": "保留更多相邻框，靠得很近的真实目标不易被合并。",
        "example": "同一个孔反复出现两个框时可从 0.45 降到 0.35；相邻孔被合并时可提高到 0.55。",
        "risk": "medium",
    },
    {
        "path": "abPoint.cropMargin", "group": "navigation", "groupLabel": "导航图检测",
        "label": "夹具裁剪边距", "unit": "比例", "kind": "number", "minimum": 0.01, "maximum": 0.50, "step": 0.01,
        "summary": "在夹具检测框外额外保留的图像范围，按夹具框尺寸计算。",
        "lowerEffect": "裁剪区域更紧，背景干扰少，但靠近边缘的孔可能被截断。",
        "higherEffect": "保留更多夹具周边内容，同时会带入更多背景。",
        "example": "0.12 表示四周各扩展约 12%；孔贴近夹具框边缘时可尝试 0.16。",
        "risk": "medium",
    },
    {
        "path": "abPoint.dedupDistancePx", "group": "navigation", "groupLabel": "导航图检测",
        "label": "孔候选去重距离", "unit": "px", "kind": "number", "minimum": 1, "maximum": 300, "step": 1,
        "summary": "两个孔候选中心小于该像素距离时，只保留置信度更高的一个。",
        "lowerEffect": "相近候选更容易同时保留，可能出现重复孔。",
        "higherEffect": "重复候选减少，但距离很近的真实孔可能被合并。",
        "example": "同一孔出现两个中心点时可从 30 px 提高到 40 px；两个真实孔被合并时应降低。",
        "risk": "medium",
    },
    {
        "path": "abPoint.semRefine.magnification", "group": "sem", "groupLabel": "SEM 精定位",
        "label": "精定位拍摄倍率", "unit": "x", "kind": "integer", "minimum": 50, "maximum": 500, "step": 10,
        "summary": "AB 孔闭环精定位时使用的电镜放大倍率。",
        "lowerEffect": "视野更大、找孔更容易，但圆心测量精度会下降。",
        "higherEffect": "孔边缘细节更多，但视野变小，粗定位偏差大时可能看不全孔。",
        "example": "当前 100x 可完整看到约 0.9 mm 孔；粗定位稳定且需要更精确时可试 120x。",
        "risk": "high",
    },
    {
        "path": "abPoint.semRefine.pixelAvg", "group": "sem", "groupLabel": "SEM 精定位",
        "label": "图像平均次数", "unit": "次", "kind": "integer", "minimum": 1, "maximum": 64, "step": 1,
        "summary": "每张 SEM 图像采集时进行像素平均的次数。",
        "lowerEffect": "拍摄更快，但噪声增加，孔边缘可能不稳定。",
        "higherEffect": "图像更平滑，但每次采图耗时更长。",
        "example": "7 次是速度和质量的平衡值；图像颗粒噪声明显时可尝试 10 次。",
        "risk": "medium",
    },
    {
        "path": "abPoint.semRefine.holeDiameterUm", "group": "sem", "groupLabel": "SEM 精定位",
        "label": "孔径先验值", "unit": "um", "kind": "number", "minimum": 100, "maximum": 3000, "step": 10,
        "summary": "圆检测算法预期看到的实际孔内径，用于限制合理的圆半径。",
        "lowerEffect": "算法会偏向寻找更小的圆形结构。",
        "higherEffect": "算法会偏向寻找更大的圆形结构。",
        "example": "当前夹具实测孔内径约 910 um；更换为 0.8 mm 孔夹具时应改为约 800 um。",
        "risk": "high",
    },
    {
        "path": "abPoint.semRefine.maxSearchRadiusUm", "group": "sem", "groupLabel": "SEM 精定位",
        "label": "最大搜索半径", "unit": "um", "kind": "number", "minimum": 100, "maximum": 10000, "step": 50,
        "summary": "闭环找孔允许样品台离开粗定位点的最远距离。",
        "lowerEffect": "搜索更快、移动范围更小，但粗定位偏差较大时会找不到孔。",
        "higherEffect": "可覆盖更大的粗定位误差，但移台次数和搜索时间都会增加。",
        "example": "粗定位误差通常小于 1 mm 时可设 1500 um；只有确认夹具安全范围后才提高。",
        "risk": "high",
    },
    {
        "path": "abPoint.semRefine.maxCaptures", "group": "sem", "groupLabel": "SEM 精定位",
        "label": "单孔最大采图数", "unit": "张", "kind": "integer", "minimum": 1, "maximum": 100, "step": 1,
        "summary": "寻找一个 AB 孔时允许拍摄的最大图像数量。",
        "lowerEffect": "失败会更早结束，节省时间，但搜索容错降低。",
        "higherEffect": "搜索机会更多，但异常情况下任务等待时间会明显增加。",
        "example": "25 张通常足够覆盖当前搜索策略；经常在最后几张才找到孔时可提高到 30。",
        "risk": "high",
    },
    {
        "path": "calibration.groupCount", "group": "calibration", "groupLabel": "标定与验收",
        "label": "标定方案组数", "unit": "组", "kind": "integer", "minimum": 1, "maximum": 10, "step": 1,
        "summary": "自动生成多少组独立锚点方案，供验证后比较选择。",
        "lowerEffect": "标定速度更快，但可比较的方案更少。",
        "higherEffect": "更容易找到稳定方案，但会增加标定采图数量和耗时。",
        "example": "设备稳定时使用 3 组；标定结果波动较大时可尝试 4 组。",
        "risk": "medium",
    },
    {
        "path": "calibration.anchorCount", "group": "calibration", "groupLabel": "标定与验收",
        "label": "每组锚点数", "unit": "个", "kind": "integer", "minimum": 3, "maximum": 12, "step": 1,
        "summary": "每组用于拟合光镜坐标到电镜台坐标变换的颗粒数量。",
        "lowerEffect": "采图更少，但单个错误匹配对结果影响更大。",
        "higherEffect": "拟合更稳健，但标定耗时增加，并需要更多可靠颗粒。",
        "example": "相似变换至少需要 3 个锚点；常规使用 3 个，复杂样品可尝试 4 个。",
        "risk": "high",
    },
    {
        "path": "calibration.validationCount", "group": "calibration", "groupLabel": "标定与验收",
        "label": "每组验证点数", "unit": "个", "kind": "integer", "minimum": 1, "maximum": 10, "step": 1,
        "summary": "每组中不参与拟合、只用于检验定位误差的独立颗粒数量。",
        "lowerEffect": "验收更快，但误差评估代表性降低。",
        "higherEffect": "验收更可信，但需要额外拍摄更多 SEM 图像。",
        "example": "常规使用 2 个验证点；正式验收或样品范围很大时可使用 3 个。",
        "risk": "medium",
    },
    {
        "path": "calibration.acceptance.meanErrorUm", "group": "calibration", "groupLabel": "标定与验收",
        "label": "平均误差上限", "unit": "um", "kind": "number", "minimum": 1, "maximum": 2000, "step": 5,
        "summary": "所有独立验证点定位误差的平均值必须不超过该值。",
        "lowerEffect": "验收更严格，定位质量更高，但标定失败率会上升。",
        "higherEffect": "标定更容易通过，但允许更大的整体定位偏差。",
        "example": "设为 80 um 时，验证点平均误差为 65 um 可通过，95 um 则不通过。",
        "risk": "high",
    },
    {
        "path": "calibration.acceptance.maxErrorUm", "group": "calibration", "groupLabel": "标定与验收",
        "label": "最大误差上限", "unit": "um", "kind": "number", "minimum": 1, "maximum": 5000, "step": 5,
        "summary": "任意一个独立验证点的误差都不能超过该值。",
        "lowerEffect": "会拒绝存在单个明显偏差点的方案，验收更严格。",
        "higherEffect": "对偶发偏差更宽容，但可能放过局部定位不准的方案。",
        "example": "设为 160 um 时，即使平均误差合格，只要某一点为 180 um，整组仍不通过。",
        "risk": "high",
    },
)

MODEL_FILES = {
    "particle": RESOURCE_ROOT / "models" / "cleanliness" / "particle_yolo11n_seg_reann_20260811_best.onnx",
    "fixture": RESOURCE_ROOT / "models" / "cleanliness_ab" / "fixture_det_v3_best.onnx",
    "hole": RESOURCE_ROOT / "models" / "cleanliness_ab" / "hole_det_v3_best.onnx",
}


def now() -> str:
    return datetime.now().astimezone().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    write_json_atomic(path, payload)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_status() -> list[dict[str, Any]]:
    rows = []
    for key, path in MODEL_FILES.items():
        exists = path.is_file()
        rows.append({
            "id": key,
            "file": path.name,
            "path": str(path),
            "ready": exists,
            "size": path.stat().st_size if exists else 0,
            "sha256": hash_file(path) if exists else None,
        })
    return rows


def _load_optical(path: Path, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    from backend.cleanliness_core.cleanliness_detection import ParticleFilter, filter_optical_particles
    from backend.cleanliness_core.optical_particle_analysis import load_optical_export

    export = load_optical_export(path)
    raw = filters or {}
    particle_filter = ParticleFilter(
        min_length_um=float(raw["minLengthUm"]) if raw.get("minLengthUm") not in (None, "") else None,
        max_length_um=float(raw["maxLengthUm"]) if raw.get("maxLengthUm") not in (None, "") else None,
        min_ecd_um=float(raw["minEcdUm"]) if raw.get("minEcdUm") not in (None, "") else None,
        max_ecd_um=float(raw["maxEcdUm"]) if raw.get("maxEcdUm") not in (None, "") else None,
        min_area_um2=float(raw["minAreaUm2"]) if raw.get("minAreaUm2") not in (None, "") else None,
        max_area_um2=float(raw["maxAreaUm2"]) if raw.get("maxAreaUm2") not in (None, "") else None,
        labels=tuple(str(label) for label in raw.get("labels", []) if str(label)),
    )
    selected = filter_optical_particles(export.particles, particle_filter)
    return {
        "path": str(export.path),
        "sha256": export.sha256,
        "projectId": export.project_id,
        "workpieceId": export.workpiece_id,
        "workpieceName": export.workpiece_name,
        "coordinateUnit": export.coordinate_unit,
        "coordinateSpace": export.coordinate_space,
        "registrationMode": export.registration_mode,
        "referencePoints": {
            "a": export.reference_a_um,
            "b": export.reference_b_um,
        },
        "particleCount": len(export.particles),
        "selectedCount": len(selected),
        "particles": [particle.to_dict() for particle in selected],
    }


def _selected_preview_particles(preview: dict[str, Any], requested: Any) -> list[dict[str, Any]]:
    particles = list(preview["particles"])
    if requested is None:
        return particles
    if not isinstance(requested, list):
        raise ValueError("selectedSequences 必须是颗粒序号数组")
    sequences = [int(value) for value in requested]
    if len(sequences) != len(set(sequences)):
        raise ValueError("selectedSequences 不能包含重复序号")
    available = {int(item["sequence"]): item for item in particles}
    missing = [sequence for sequence in sequences if sequence not in available]
    if missing:
        raise ValueError(f"所选颗粒不在当前筛选结果中：{missing[:10]}")
    if not sequences:
        raise ValueError("至少选择一个颗粒进入任务")
    return [available[sequence] for sequence in sequences]


class JobManager:
    HARDWARE_OPERATIONS = {"calibration", "ab_refine", "analysis", "analysis_resume", "sem_analysis", "sem_navigation"}

    def __init__(self, control=None) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._finished_order: list[str] = []
        self.control = control or DeviceCoordinator()
        self._device_jobs: set[str] = set()
        self.allow_hardware_jobs = True

    def create(self, task_id: str, operation: str) -> dict[str, Any]:
        with self._lock:
            hardware_operations = self.HARDWARE_OPERATIONS
            if operation in hardware_operations and not self.allow_hardware_jobs:
                raise ValueError("模拟 MCP 模式不允许调用旧桌面硬件入口")
            for active in self._jobs.values():
                if active.get("status") not in {"queued", "running"}:
                    continue
                if active.get("taskId") == task_id or (operation in hardware_operations and active.get("operation") in hardware_operations):
                    raise ValueError("已有任务正在运行，请先停止并等待设备恢复，不能重复启动")
            job_id = str(uuid.uuid4())
            if operation in hardware_operations:
                self.control.acquire(f"desktop:{job_id}")
                self._device_jobs.add(job_id)
            job = {"id": job_id, "taskId": task_id, "operation": operation, "status": "queued", "progress": 0, "total": 0, "message": "等待执行", "createdAt": now()}
            self._jobs[job_id] = job
            self._cancel[job_id] = threading.Event()
            log_event("info", "job_created", jobId=job_id, taskId=task_id, operation=operation)
            return dict(job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._jobs.get(job_id)
            result = dict(item) if item else None
        if result is not None:
            packed = result.pop('_packed_result', None)
            if packed is not None:
                result['result'] = json.loads(zlib.decompress(packed))
        return result

    def update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(values)
                if 'result' in values:
                    self._jobs[job_id].pop('_packed_result',None)
                    result=values['result']
                    if isinstance(result,dict) and result.get('particles'):
                        self._jobs[job_id]['_packed_result']=zlib.compress(
                            json.dumps(result,ensure_ascii=False,separators=(',',':')).encode('utf-8'),level=1)
                        self._jobs[job_id].pop('result',None)
                if any(key in values for key in ("status", "error", "completedAt")):
                    current = self._jobs[job_id]
                    logged=dict(values)
                    if isinstance(logged.get('result'),dict):
                        source=logged['result']
                        logged['result']={'taskId':source.get('taskId'),'status':source.get('status'),
                                          'particleCount':len(source.get('particles') or [])}
                    log_event("error" if values.get("status") == "failed" else "info", "job_updated", jobId=job_id, taskId=current.get("taskId"), operation=current.get("operation"), values=logged)
                if values.get('status') in {'completed','failed'}:
                    if job_id in self._device_jobs:
                        self.control.release(f"desktop:{job_id}")
                        self._device_jobs.discard(job_id)
                    if job_id in self._finished_order:
                        self._finished_order.remove(job_id)
                    self._finished_order.append(job_id)
                    while len(self._finished_order)>64:
                        expired=self._finished_order.pop(0)
                        self._jobs.pop(expired,None)
                        self._cancel.pop(expired,None)

    def cancelled(self, job_id: str) -> bool:
        with self._lock:
            event=self._cancel.get(job_id)
            return event.is_set() if event is not None else False

    def cancel(self, job_id: str) -> None:
        with self._lock:
            event=self._cancel.get(job_id)
            job=self._jobs.get(job_id)
            if event is None or job is None:
                return
            event.set()
            task_id,operation=job.get('taskId'),job.get('operation')
        log_event("info", "job_cancel_requested", jobId=job_id, taskId=task_id, operation=operation)

    def active_for_task(self, task_id: str, operation: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            for item in self._jobs.values():
                if item.get("taskId") != task_id:
                    continue
                if operation is not None and item.get("operation") != operation:
                    continue
                if item.get("status") in {"queued", "running"}:
                    return dict(item)
        return None


JOBS = JobManager()
AB_SESSIONS: dict[str, dict[str, Any]] = {}
BACKEND_PORT = int(os.environ.get("ZEM_BACKEND_PORT", "0"))
MCP_CONTROL = None
MCP_TOKEN = ""


def _chat_snapshot() -> str:
    """给 vibe 助手的当前软件状态快照（拼进 prompt 前导）。"""
    lines = [
        f"- 仓库根目录：{APP_ROOT}",
        f"- 任务目录：{TASK_ROOT}",
        f"- 软件设置文件：{SETTINGS_PATH}",
        f"- 假 EDS 模式：{'开' if FAKE_EDS_ENABLED else '关'}",
        "- 聊天只允许两个 MCP：zem-cleanliness（软件界面）、zem-sem（电镜）",
        "- 关枪不拦截开始分析；仿真不能开枪，禁止主动开枪",
    ]
    if BACKEND_PORT:
        lines.append(f"- 后端已连接（端口 {BACKEND_PORT}）；聊天禁止直接 HTTP，软件动作走 zem-cleanliness，电镜走 zem-sem")
    try:
        options = app_settings()
        lines.append(f"- 电镜地址：{options['microscopeHost']}:{options['microscopePort']}")
        lines.append(f"- 能谱地址：{options['edsHost']}:{options['edsPort']}")
    except Exception as error:
        lines.append(f"- 软件设置读取失败：{error}")
    try:
        for device in device_status():
            lines.append(f"- {device['name']}: {device['status']}（{device['detail']}）")
    except Exception as error:
        lines.append(f"- 设备状态获取失败：{error}")
    try:
        task_files = sorted(TASK_ROOT.glob("*/task.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:3]
        for task_file in task_files:
            manifest = read_json(task_file)
            lines.append(
                f"- 最近任务 {task_file.parent.name}: status={manifest.get('status')} "
                f"workpiece={(manifest.get('input') or {}).get('workpieceName')!r}"
            )
    except Exception:
        pass
    return "\n".join(lines)


def _chat_manager():
    from backend.chat_runner import get_manager

    return get_manager(snapshot_provider=_chat_snapshot)


def create_sem_navigation() -> dict[str, Any]:
    """Read current navigation and freeze device/config provenance; no movement."""
    from backend.cleanliness_core.microscope_discovery import validate_zem_status
    from backend.cleanliness_core.ab_point_calibration import nav_pixel_to_stage
    from backend.cleanliness_core.optical_particle_task import (
        _configure_zemrpc_timeouts, _configure_zemrpc_device, _cache_zemrpc_interface,
    )
    from zemrpc.zem.zem_api import zem_api
    import cv2

    task_id = str(uuid.uuid4())
    job = JOBS.create(task_id, 'sem_navigation')
    try:
        options = app_settings()
        _configure_zemrpc_timeouts()
        _configure_zemrpc_device(zem_api.device)
        if str(validate_zem_status(zem_api, (options['microscopeHost'], options['microscopePort']))).lower() != 'idle':
            raise ValueError('电镜未空闲，不能读取导航图')
        _cache_zemrpc_interface(zem_api.device, options['microscopeHost'], options['microscopePort'])
        zem_api.sync_stage_and_scan_field()
        nav = zem_api.get_nav_img()
        if nav is None or getattr(nav, 'img', None) is None or getattr(nav, 'bounds', None) is None:
            raise ValueError('设备没有可用导航图，请先在电镜软件中准备导航图后重试')
        image = _navigation_to_bgr(nav.img)
        bounds = list(map(float, nav.bounds))
        nav_pixel_to_stage((0, 0), image.shape, bounds)
        scan = zem_api.device.m.scan_info
        field, mag, rotation = float(scan.scan_um), float(zem_api.device.m.mag), float(scan.rot_degree)
        if not all(math.isfinite(v) for v in (field, mag, rotation)) or min(field, mag) <= 0:
            raise ValueError('无法读取当前 FOV 或倍率')
        if abs((rotation+180) % 360-180) > .1:
            raise ValueError('直接 SEM 扫描目前要求扫描旋转为 0°，请在电镜端调整后重新读取')
        model = MODEL_FILES['particle']
        directory = TASK_ROOT/task_id
        directory.mkdir(parents=True)
        if not cv2.imwrite(str(directory/'navigation.png'), image):
            raise OSError('导航图保存失败')
        _task_config_path(directory, options)
        manifest = {'taskId': task_id, 'mode': 'sem_direct', 'status': 'DRAFT', 'createdAt': now(),
                    'input': {'workpieceName': '直接 SEM 分析', 'particleCount': 0, 'selectedCount': 0},
                    'request': {**options, 'mode': 'sem_direct'}, 'particles': [],
                    'navigation': {'boundsUm': bounds, 'imageShape': list(image.shape[:2]), 'path': 'navigation.png',
                                   'sha256': hash_file(directory/'navigation.png'), 'fieldUm': field, 'magnification': mag,
                                   'rotationDeg': rotation},
                    'model': {'backend': 'yolo', 'path': str(model), 'sha256': hash_file(model) if model.is_file() else None}}
        atomic_json(directory/'task.json', manifest)
        JOBS.update(job['id'], status='completed')
        return {**manifest, 'folderId': task_id, 'taskDirectory': str(directory), 'previewDataUrl': _image_data_url(image)}
    except Exception as error:
        JOBS.update(job['id'], status='failed', error=str(error))
        raise


def save_sem_plan(task_id, body):
    from backend.cleanliness_core.sem_direct import build_plan
    path, manifest = task_manifest(task_id)
    if manifest.get('mode') != 'sem_direct' or manifest['status'] != 'DRAFT' or JOBS.active_for_task(task_id):
        raise ValueError('只有未启动的直接 SEM 任务可以修改扫描区域')
    plan = build_plan(body, manifest['navigation']['boundsUm'])
    # The microscope sets integer magnification. Preview the realizable FOV,
    # then verify its actual readback at capture time.
    field_product = manifest['navigation']['fieldUm']*manifest['navigation']['magnification']
    magnification = max(1, round(field_product/plan['fieldUm']))
    plan = build_plan({**body, 'fieldUm': field_product/magnification}, manifest['navigation']['boundsUm'])
    selection = body.get('segmentation', manifest.get('model', {}))
    backend = selection.get('backend', 'yolo')
    if backend == 'sam3':
        from backend.cleanliness_core.sem_segmentation import sam3_config
        model = sam3_config(selection)
    elif backend == 'yolo':
        model_path = MODEL_FILES['particle']
        if not model_path.is_file():
            raise ValueError('YOLO 颗粒模型不存在')
        model = {'backend': 'yolo', 'path': str(model_path), 'sha256': hash_file(model_path)}
    else:
        raise ValueError('不支持的颗粒分割模型')
    manifest['model'] = model
    manifest['plan'] = plan
    manifest['request']['edsEnabled'] = body.get('edsEnabled', True) is True
    manifest['input']['workpieceName'] = str(body.get('name') or '直接 SEM 分析').strip()[:120]
    manifest['updatedAt'] = now()
    atomic_json(path, manifest)
    return {**manifest, 'folderId': task_id, 'taskDirectory': str(path.parent)}


def task_manifest(task_id: str) -> tuple[Path, dict[str, Any]]:
    task_dir = TASK_ROOT / task_id
    path = task_dir / "task.json"
    if not path.is_file():
        path = task_dir / "draft.json"
    if not path.is_file():
        raise FileNotFoundError(f"任务不存在：{task_id}")
    manifest = read_json(path)
    if manifest.get('mode') == 'sem_direct' and manifest.get('incrementalState'):
        from backend.cleanliness_core.sem_task_store import load_sem_task
        manifest = load_sem_task(path.parent)
    return path, manifest


@lru_cache(maxsize=128)
def _cached_task_list_summary(path: str, mtime_ns: int, size: int):
    # Cache metadata only, never the particle/spectrum graph. Atomic file
    # replacements invalidate this key through their timestamp and size.
    manifest=read_json(Path(path))
    if manifest.get('incrementalState'):
        return {'_needsReplay':True}
    summary={k:manifest[k] for k in ('mode','status','input','createdAt','completedAt') if k in manifest}
    summary['selection']={'count':(manifest.get('selection') or {}).get('count')}
    return summary


def _capture_level(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 7:
        raise ValueError("captureLevel 必须是 1-7 的整数")
    return value


def _capture_image_size(value: Any) -> int:
    from backend.cleanliness_core.imaging_settings import validate_capture_image_size
    return validate_capture_image_size(value)


def _eds_target_counts(value: Any) -> int | None:
    from backend.cleanliness_core.eds.client_v2 import validate_target_counts
    return validate_target_counts(value)


def _unit_interval(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是 0 与 1 之间的数值")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0 < normalized < 1:
        raise ValueError(f"{name} 必须是 0 与 1 之间的数值")
    return normalized


def _candidate_score_margin(value: Any) -> float:
    return _unit_interval(value, "candidateScoreMargin")


def _template_match_min_score(value: Any) -> float:
    return _unit_interval(value, "templateMatchMinScore")


def _contour_overlap_min(value: Any) -> float:
    return _unit_interval(value, "contourOverlapMin")


def _max_capture_attempts(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise ValueError("maxCaptureAttempts 必须是 1-5 的整数")
    return value


def _build_task_draft(task_id: str, body: dict[str, Any], previous: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    previous = previous or {}
    input_path = Path(str(body["inputPath"])).resolve()
    preview = _load_optical(input_path, body.get("filters"))
    selected = _selected_preview_particles(preview, body.get("selectedSequences"))
    selected_sequences = [int(item["sequence"]) for item in selected]
    defaults = app_settings()
    request_body = {key: value for key, value in body.items() if key != "confirmReset"}
    request_snapshot = {
        **(previous.get("request") or {}),
        **request_body,
        "microscopeHost": str(body.get("microscopeHost") or (previous.get("request") or {}).get("microscopeHost") or defaults["microscopeHost"]),
        "microscopePort": int(body.get("microscopePort") or (previous.get("request") or {}).get("microscopePort") or defaults["microscopePort"]),
        "edsHost": str(body.get("edsHost") or (previous.get("request") or {}).get("edsHost") or defaults["edsHost"]),
        "edsPort": int(body.get("edsPort") or (previous.get("request") or {}).get("edsPort") or defaults["edsPort"]),
        "edsEnabled": bool(body.get("edsEnabled", (previous.get("request") or {}).get("edsEnabled", defaults["edsEnabled"]))),
        "edsTargetCounts": _eds_target_counts(body.get("edsTargetCounts", (previous.get("request") or {}).get("edsTargetCounts", defaults["edsTargetCounts"]))),
        "restoreStartPosition": bool(body.get("restoreStartPosition", (previous.get("request") or {}).get("restoreStartPosition", defaults["restoreStartPosition"]))),
        "gunOffAfterTask": bool(body.get("gunOffAfterTask", (previous.get("request") or {}).get("gunOffAfterTask", defaults["gunOffAfterTask"]))),
        "showReportCoordinates": bool(body.get("showReportCoordinates", (previous.get("request") or {}).get("showReportCoordinates", defaults["showReportCoordinates"]))),
        "captureLevel": _capture_level(body.get("captureLevel", (previous.get("request") or {}).get("captureLevel", defaults["captureLevel"]))),
        "captureImageSize": _capture_image_size(body.get("captureImageSize", (previous.get("request") or {}).get("captureImageSize", defaults["captureImageSize"]))),
        "candidateScoreMargin": _candidate_score_margin(body.get("candidateScoreMargin", (previous.get("request") or {}).get("candidateScoreMargin", defaults["candidateScoreMargin"]))),
        "templateMatchMinScore": _template_match_min_score(body.get("templateMatchMinScore", (previous.get("request") or {}).get("templateMatchMinScore", defaults["templateMatchMinScore"]))),
        "contourOverlapMin": _contour_overlap_min(body.get("contourOverlapMin", (previous.get("request") or {}).get("contourOverlapMin", defaults["contourOverlapMin"]))),
        "maxCaptureAttempts": _max_capture_attempts(body.get("maxCaptureAttempts", (previous.get("request") or {}).get("maxCaptureAttempts", defaults["maxCaptureAttempts"]))),
    }
    manifest = {
        "schemaVersion": 1,
        "taskType": "CLEANLINESS_DETECTION",
        "taskId": task_id,
        "folderId": task_id,
        "status": "DRAFT",
        "createdAt": previous.get("createdAt") or now(),
        "updatedAt": now(),
        "revision": int(previous.get("revision") or 0),
        "revisionHistory": list(previous.get("revisionHistory") or []),
        "input": {"path": str(input_path), "sha256": preview["sha256"], "projectId": preview["projectId"], "workpieceId": preview["workpieceId"], "workpieceName": preview["workpieceName"], "particleCount": preview["particleCount"], "selectedCount": len(selected)},
        "selection": {"filter": body.get("filters") or {}, "sequences": selected_sequences, "count": len(selected)},
        "request": {**request_snapshot, "selectedSequences": selected_sequences, "inputSha256": preview["sha256"]},
        "settingsSnapshot": ".cleanliness-settings.json",
        "models": model_status(),
    }
    return manifest, preview


def _task_input_signature(manifest: dict[str, Any]) -> str:
    request = manifest.get("request") or {}
    calibration = manifest.get("calibration") or {}
    raw_filters = (manifest.get("selection") or {}).get("filter") or request.get("filters") or {}
    normalized_filters = {str(key): str(value) for key, value in raw_filters.items() if value not in (None, "")}
    return json.dumps({
        "inputSha256": (manifest.get("input") or {}).get("sha256") or request.get("inputSha256"),
        "filters": normalized_filters,
        "selectedSequences": [int(value) for value in ((manifest.get("selection") or {}).get("sequences") or request.get("selectedSequences") or [])],
        "mode": request.get("mode") or calibration.get("strategy") or "multi_particle",
    }, ensure_ascii=False, sort_keys=True)


def revise_task_input(task_id: str, body: dict[str, Any]) -> dict[str, Any]:
    path, manifest = task_manifest(task_id)
    active = JOBS.active_for_task(task_id)
    if active is not None:
        raise ValueError("当前任务仍有作业运行，请等待完成或停止后再修改")
    allowed = {"DRAFT", "CALIBRATION_PENDING_REVIEW", "CALIBRATION_ACCEPTED", "AB_POINTS_PENDING_SEM_REFINEMENT"}
    status = str(manifest.get("status") or "")
    if status not in allowed:
        raise ValueError(f"当前任务状态不能修改导入和筛选条件：{status}")

    draft_path = path.parent / "draft.json"
    original_draft = read_json(draft_path) if draft_path.is_file() else {}
    previous = {
        **original_draft,
        **manifest,
        "request": {**(original_draft.get("request") or {}), **(manifest.get("request") or {})},
        "revisionHistory": list(manifest.get("revisionHistory") or original_draft.get("revisionHistory") or []),
    }
    revised, _preview = _build_task_draft(task_id, body, previous)
    if _task_input_signature(previous) == _task_input_signature(revised):
        return {**manifest, "folderId": task_id, "taskDirectory": str(path.parent), "inputRevised": False}
    if status != "DRAFT" and not bool(body.get("confirmReset")):
        raise ValueError("修改输入会使当前坐标标定结果失效，请确认后重试")

    if path.name == "task.json":
        revisions_dir = path.parent / "revisions"
        revisions_dir.mkdir(parents=True, exist_ok=True)
        archive_name = f"task-before-input-revision-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.json"
        archive_path = revisions_dir / archive_name
        path.replace(archive_path)
        revised["revisionHistory"].append({"revisedAt": now(), "previousStatus": status, "archive": str(archive_path.relative_to(path.parent)).replace("\\", "/")})

    revised["revision"] = int(previous.get("revision") or 0) + 1
    atomic_json(draft_path, revised)
    AB_SESSIONS.pop(task_id, None)
    log_event("info", "task_input_revised", taskId=task_id, previousStatus=status, revision=revised["revision"], selectedCount=revised["selection"]["count"])
    return {**revised, "taskDirectory": str(path.parent), "inputRevised": True}


def _config_path() -> Path:
    return RESOURCE_ROOT / "conf" / "cleanliness_detection.json"


def _config_value(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted_path)
        value = value[part]
    return value


def _set_config_value(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    target = payload
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    target[parts[-1]] = value


def _engineer_overrides() -> dict[str, Any]:
    if not ENGINEER_OVERRIDES_PATH.is_file():
        return {}
    payload = read_json(ENGINEER_OVERRIDES_PATH)
    values = payload.get("values") or {}
    allowed = {field["path"] for field in ENGINEER_FIELDS}
    return {str(path): value for path, value in values.items() if path in allowed}


def effective_cleanliness_config() -> dict[str, Any]:
    from backend.cleanliness_core.cleanliness_detection import load_cleanliness_config

    config = copy.deepcopy(load_cleanliness_config(_config_path()))
    for path, value in _engineer_overrides().items():
        _set_config_value(config, path, value)

    settings = app_settings()
    config.setdefault("device", {}).update({
        "microscopeHost": settings["microscopeHost"],
        "microscopePort": settings["microscopePort"],
        "edsHost": settings["edsHost"],
        "edsPort": settings["edsPort"],
    })
    config.setdefault("task", {}).update({
        "edsEnabled": settings["edsEnabled"],
        "edsTargetCounts": settings["edsTargetCounts"],
        "restoreStartPosition": settings["restoreStartPosition"],
        "gunOnAtStart": False,
        "gunOffAfterTask": settings["gunOffAfterTask"],
        "captureLevel": settings["captureLevel"],
        "captureImageSize": settings["captureImageSize"],
        "contourOverlapMin": settings["contourOverlapMin"],
        "maxCaptureAttempts": settings["maxCaptureAttempts"],
    })
    config.setdefault("abParticle", {})["match_min_margin"] = settings["candidateScoreMargin"]
    config.setdefault("abParticle", {})["match_min_score"] = settings["templateMatchMinScore"]
    config.setdefault("report", {})["showCoordinates"] = settings["showReportCoordinates"]
    return config


def _task_config_path(output_dir: Path, options: dict[str, Any] | None = None) -> Path:
    destination = output_dir / ".cleanliness-settings.json"
    if destination.is_file():
        return destination
    config = effective_cleanliness_config()
    options = options or {}
    config.setdefault("task", {})["edsTargetCounts"] = _eds_target_counts(
        options.get("edsTargetCounts", (config.get("task") or {}).get("edsTargetCounts"))
    )
    if "gunOffAfterTask" in options:
        config["task"]["gunOffAfterTask"] = bool(options["gunOffAfterTask"])
    if "showReportCoordinates" in options:
        config["report"]["showCoordinates"] = bool(options["showReportCoordinates"])
    capture_level = _capture_level(options.get("captureLevel", (config.get("task") or {}).get("captureLevel", 7)))
    config.setdefault("task", {})["captureLevel"] = capture_level
    config.setdefault("abPoint", {}).setdefault("semRefine", {})["pixelAvg"] = capture_level
    config.setdefault("abParticle", {})["pixel_avg"] = capture_level
    image_size = _capture_image_size(options.get("captureImageSize", config["task"].get("captureImageSize", 512)))
    config["task"]["captureImageSize"] = image_size
    config["abPoint"]["semRefine"]["imageShape"] = [image_size, image_size]
    config["abParticle"]["image_size"] = image_size
    candidate_score_margin = _candidate_score_margin(
        options.get("candidateScoreMargin", config["abParticle"].get("match_min_margin", 0.07))
    )
    config["abParticle"]["match_min_margin"] = candidate_score_margin
    config.setdefault("task", {})["candidateScoreMargin"] = candidate_score_margin
    template_match_min_score = _template_match_min_score(
        options.get("templateMatchMinScore", config["abParticle"].get("match_min_score", 0.65))
    )
    config["abParticle"]["match_min_score"] = template_match_min_score
    config.setdefault("task", {})["templateMatchMinScore"] = template_match_min_score
    contour_overlap_min = _contour_overlap_min(
        options.get("contourOverlapMin", config["task"].get("contourOverlapMin", 0.5))
    )
    config["task"]["contourOverlapMin"] = contour_overlap_min
    max_capture_attempts = _max_capture_attempts(
        options.get("maxCaptureAttempts", config["task"].get("maxCaptureAttempts", 2))
    )
    config["task"]["maxCaptureAttempts"] = max_capture_attempts
    # Resolve bundled resources before moving the immutable task snapshot away
    # from the installation's conf directory.
    config["experienceCalibration"] = str(RESOURCE_ROOT / "conf" / "optical_particle_calibration.json")
    config.setdefault("abPoint", {})["modelDirectory"] = str(RESOURCE_ROOT / "models" / "cleanliness_ab")
    segmentation = config.setdefault("calibration", {}).setdefault("candidateSegmentation", {})
    segmentation["modelPath"] = str(MODEL_FILES["particle"])
    atomic_json(destination, config)
    return destination


def _clean_engineer_sessions() -> None:
    current = time.monotonic()
    for token, expires_at in list(ENGINEER_SESSIONS.items()):
        if expires_at <= current:
            ENGINEER_SESSIONS.pop(token, None)


def unlock_engineer(password: Any) -> dict[str, Any]:
    current = time.monotonic()
    ENGINEER_FAILURES[:] = [stamp for stamp in ENGINEER_FAILURES if current - stamp < 60]
    if len(ENGINEER_FAILURES) >= 5:
        retry_after = max(1, math.ceil(60 - (current - ENGINEER_FAILURES[0])))
        raise PermissionError(f"密码错误次数过多，请在 {retry_after} 秒后重试")
    submitted = hashlib.sha256(str(password).encode("utf-8")).hexdigest()
    if not hmac.compare_digest(submitted, ENGINEER_PASSWORD_SHA256):
        ENGINEER_FAILURES.append(current)
        raise PermissionError("工程师密码不正确")
    ENGINEER_FAILURES.clear()
    _clean_engineer_sessions()
    token = secrets.token_urlsafe(32)
    ENGINEER_SESSIONS[token] = current + ENGINEER_SESSION_SECONDS
    return {"token": token, "expiresInSeconds": ENGINEER_SESSION_SECONDS}


def _require_engineer(token: Any) -> str:
    _clean_engineer_sessions()
    normalized = str(token or "")
    expires_at = ENGINEER_SESSIONS.get(normalized)
    if expires_at is None:
        raise PermissionError("工程师面板未解锁或会话已过期")
    ENGINEER_SESSIONS[normalized] = time.monotonic() + ENGINEER_SESSION_SECONDS
    return normalized


def engineer_settings(token: Any) -> dict[str, Any]:
    _require_engineer(token)
    from backend.cleanliness_core.cleanliness_detection import load_cleanliness_config

    defaults_config = load_cleanliness_config(_config_path())
    effective = effective_cleanliness_config()
    definitions = []
    values: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    for raw in ENGINEER_FIELDS:
        field = dict(raw)
        path = str(field["path"])
        field["defaultValue"] = _config_value(defaults_config, path)
        definitions.append(field)
        defaults[path] = field["defaultValue"]
        values[path] = _config_value(effective, path)
    updated_at = None
    if ENGINEER_OVERRIDES_PATH.is_file():
        updated_at = read_json(ENGINEER_OVERRIDES_PATH).get("updatedAt")
    return {"definitions": definitions, "values": values, "defaults": defaults, "updatedAt": updated_at}


def save_engineer_settings(token: Any, submitted: Any) -> dict[str, Any]:
    normalized_token = _require_engineer(token)
    if not isinstance(submitted, dict):
        raise ValueError("工程师配置必须是键值对象")

    from backend.cleanliness_core.cleanliness_detection import load_cleanliness_config

    definitions = {str(field["path"]): field for field in ENGINEER_FIELDS}
    unknown = sorted(set(submitted) - set(definitions))
    if unknown:
        raise ValueError(f"包含不可编辑的配置项：{', '.join(unknown)}")
    missing = sorted(set(definitions) - set(submitted))
    if missing:
        raise ValueError(f"缺少配置项：{', '.join(missing)}")

    validated: dict[str, int | float] = {}
    for path, definition in definitions.items():
        raw_value = submitted[path]
        if isinstance(raw_value, bool):
            raise ValueError(f"{definition['label']} 必须是数字")
        try:
            number = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{definition['label']} 必须是数字") from error
        if not math.isfinite(number):
            raise ValueError(f"{definition['label']} 必须是有限数字")
        if number < float(definition["minimum"]) or number > float(definition["maximum"]):
            raise ValueError(
                f"{definition['label']} 必须在 {definition['minimum']} 到 {definition['maximum']} 之间"
            )
        if definition["kind"] == "integer":
            if not number.is_integer():
                raise ValueError(f"{definition['label']} 必须是整数")
            validated[path] = int(number)
        else:
            validated[path] = number

    if validated["calibration.acceptance.maxErrorUm"] < validated["calibration.acceptance.meanErrorUm"]:
        raise ValueError("最大误差上限不能小于平均误差上限")

    defaults_config = load_cleanliness_config(_config_path())
    overrides = {
        path: value for path, value in validated.items()
        if value != _config_value(defaults_config, path)
    }
    atomic_json(ENGINEER_OVERRIDES_PATH, {
        "schemaVersion": 1,
        "updatedAt": now(),
        "values": overrides,
    })
    return engineer_settings(normalized_token)


def lock_engineer(token: Any) -> dict[str, Any]:
    ENGINEER_SESSIONS.pop(str(token or ""), None)
    return {"locked": True}


def app_settings() -> dict[str, Any]:
    from backend.cleanliness_core.cleanliness_detection import load_cleanliness_config

    bundled = load_cleanliness_config(_config_path())
    device = bundled.get("device") or {}
    task = bundled.get("task") or {}
    defaults = {
        "microscopeHost": str(device.get("microscopeHost") or "192.168.0.15"),
        "microscopePort": int(device.get("microscopePort") or 61557),
        "edsHost": str(device.get("edsHost") or "192.168.0.15"),
        "edsPort": int(device.get("edsPort") or 6200),
        "edsEnabled": bool(task.get("edsEnabled", True)),
        "edsTargetCounts": None,
        "restoreStartPosition": bool(task.get("restoreStartPosition", True)),
        "taskRoot": str(TASK_ROOT),
        "gunOffAfterTask": True,
        "showReportCoordinates": True,
        "preflightEnabled": True,
        "captureLevel": 7,
        "captureImageSize": 512,
        "candidateScoreMargin": float((bundled.get("abParticle") or {}).get("match_min_margin", 0.07)),
        "templateMatchMinScore": float((bundled.get("abParticle") or {}).get("match_min_score", 0.65)),
        "contourOverlapMin": float((bundled.get("task") or {}).get("contourOverlapMin", 0.5)),
        "maxCaptureAttempts": int((bundled.get("task") or {}).get("maxCaptureAttempts", 2)),
        "settingsSchemaVersion": 8,
    }
    if SETTINGS_PATH.is_file():
        saved = read_json(SETTINGS_PATH)
        if saved.get("settingsSchemaVersion", 0) >= 8:
            defaults["captureImageSize"] = _capture_image_size(saved.get("captureImageSize", 512))
        if saved.get("settingsSchemaVersion", 0) >= 7:
            defaults["edsTargetCounts"] = _eds_target_counts(saved.get("edsTargetCounts"))
        for key in ("microscopeHost", "microscopePort", "edsHost", "edsPort", "edsEnabled", "restoreStartPosition"):
            if key in saved:
                defaults[key] = saved[key]
        # Previous releases always wrote False without offering a gun-off option.
        if saved.get("settingsSchemaVersion", 0) >= 2:
            for key in ("gunOffAfterTask", "showReportCoordinates", "preflightEnabled"):
                if isinstance(saved.get(key), bool):
                    defaults[key] = saved[key]
        if saved.get("settingsSchemaVersion", 0) >= 3:
            capture_level = saved.get("captureLevel")
            if isinstance(capture_level, int) and not isinstance(capture_level, bool) and 1 <= capture_level <= 7:
                defaults["captureLevel"] = capture_level
        if saved.get("settingsSchemaVersion", 0) >= 4:
            try:
                defaults["candidateScoreMargin"] = _candidate_score_margin(saved.get("candidateScoreMargin"))
            except ValueError:
                pass
        if saved.get("settingsSchemaVersion", 0) >= 5:
            try:
                defaults["templateMatchMinScore"] = _template_match_min_score(saved.get("templateMatchMinScore"))
            except ValueError:
                pass
        if saved.get("settingsSchemaVersion", 0) >= 6:
            try:
                defaults["contourOverlapMin"] = _contour_overlap_min(saved.get("contourOverlapMin"))
            except ValueError:
                pass
            try:
                defaults["maxCaptureAttempts"] = _max_capture_attempts(saved.get("maxCaptureAttempts"))
            except ValueError:
                pass
    defaults["taskRoot"] = str(TASK_ROOT)
    return defaults


def save_app_settings(payload: dict[str, Any]) -> dict[str, Any]:
    with SETTINGS_LOCK:
        return _save_app_settings(payload)


def _save_app_settings(payload: dict[str, Any]) -> dict[str, Any]:
    current = app_settings()
    for key in ("microscopeHost", "edsHost"):
        if key in payload:
            value = str(payload[key]).strip()
            if not value:
                raise ValueError(f"{key} 不能为空")
            current[key] = value
    for key in ("microscopePort", "edsPort"):
        if key in payload:
            value = int(payload[key])
            if not 1 <= value <= 65535:
                raise ValueError(f"{key} 必须在 1-65535 范围内")
            current[key] = value
    for key in ("edsEnabled", "restoreStartPosition"):
        if key in payload:
            current[key] = bool(payload[key])
    for key in ("gunOffAfterTask", "showReportCoordinates", "preflightEnabled"):
        if key in payload:
            if not isinstance(payload[key], bool):
                raise ValueError(f"{key} 必须为布尔值")
            current[key] = payload[key]
    if "captureLevel" in payload:
        current["captureLevel"] = _capture_level(payload["captureLevel"])
    if "captureImageSize" in payload:
        current["captureImageSize"] = _capture_image_size(payload["captureImageSize"])
    if "edsTargetCounts" in payload:
        current["edsTargetCounts"] = _eds_target_counts(payload["edsTargetCounts"])
    if "candidateScoreMargin" in payload:
        current["candidateScoreMargin"] = _candidate_score_margin(payload["candidateScoreMargin"])
    if "templateMatchMinScore" in payload:
        current["templateMatchMinScore"] = _template_match_min_score(payload["templateMatchMinScore"])
    if "contourOverlapMin" in payload:
        current["contourOverlapMin"] = _contour_overlap_min(payload["contourOverlapMin"])
    if "maxCaptureAttempts" in payload:
        current["maxCaptureAttempts"] = _max_capture_attempts(payload["maxCaptureAttempts"])
    control = MCP_CONTROL
    target = _mcp_endpoint(current["microscopeHost"], current["microscopePort"])
    current_endpoint = (
        _mcp_endpoint(control.device.host, control.device.port)
        if control is not None and not getattr(control.device, "simulated", False) else None
    )
    if control is not None and current_endpoint is not None and target != current_endpoint:
        # 换电镜地址后不继承上一台的采图/移动批准；mcp-device.json 仍保留，重启后按文件重新加载。
        with control.coordinator.operation("settings:microscope-endpoint", writes=False):
            atomic_json(SETTINGS_PATH, current)
            control.device.host, control.device.port = target[0], target[1]
            before = {
                "capture": control.allow_capture, "move": control.bounds is not None,
                "rpc": sorted(control.allow_rpc),
            }
            control.allow_capture = False
            control.bounds = None
            control.max_move_um = None
            control.allow_rpc = frozenset()
            control.rpc_limits = {}
            control.rpc_path_roots = ()
            configured = load_mcp_device_config()
            log_event(
                "warning", "mcp_device_permissions_cleared",
                reason="microscope_endpoint_changed",
                previousEndpoint={"host": current_endpoint[0], "port": current_endpoint[1]},
                newEndpoint={"host": target[0], "port": target[1]},
                before=before,
                mcpDeviceFileStillAllowsCapture=configured.get("allowCapture") is True,
                hint="权限仅在本进程内存清除；mcp-device.json 未改。重启软件后按文件重新加载。",
            )
    else:
        atomic_json(SETTINGS_PATH, current)
    return current


def _mcp_endpoint(host: Any, port: Any) -> tuple[str, int]:
    return str(host).strip(), int(port)


def mcp_device_config_path() -> Path:
    return TASK_ROOT.parent / "mcp-device.json"


def load_mcp_device_config() -> dict[str, Any]:
    path = mcp_device_config_path()
    return read_json(path) if path.is_file() else {}


def _make_config(input_path: Path, output_dir: Path, body: dict[str, Any], *, for_analysis: bool = False):
    from backend.cleanliness_core.cleanliness_detection import ParticleFilter
    from backend.cleanliness_core.cleanliness_task import CleanlinessTaskConfig

    filters = body.get("filters") or {}
    settings = app_settings()
    snapshot_path = _task_config_path(output_dir, body)
    snapshot = read_json(snapshot_path)
    eds_retry_count = int(body.get("edsRetryCount", 0))
    if not 0 <= eds_retry_count <= 3:
        raise ValueError("edsRetryCount 必须在 0 到 3 之间")
    return CleanlinessTaskConfig(
        input_path=input_path,
        output_dir=output_dir,
        settings_path=snapshot_path,
        particle_filter=ParticleFilter(
            min_length_um=float(filters["minLengthUm"]) if filters.get("minLengthUm") not in (None, "") else None,
            max_length_um=float(filters["maxLengthUm"]) if filters.get("maxLengthUm") not in (None, "") else None,
            min_ecd_um=float(filters["minEcdUm"]) if filters.get("minEcdUm") not in (None, "") else None,
            max_ecd_um=float(filters["maxEcdUm"]) if filters.get("maxEcdUm") not in (None, "") else None,
            min_area_um2=float(filters["minAreaUm2"]) if filters.get("minAreaUm2") not in (None, "") else None,
            max_area_um2=float(filters["maxAreaUm2"]) if filters.get("maxAreaUm2") not in (None, "") else None,
            labels=tuple(filters.get("labels") or ()),
        ),
        selected_sequences=tuple(int(value) for value in body.get("selectedSequences", [])) or None,
        experience_calibration_path=RESOURCE_ROOT / "conf" / "optical_particle_calibration.json",
        microscope_host=str(body.get("microscopeHost") or settings["microscopeHost"]),
        microscope_port=int(body.get("microscopePort") or settings["microscopePort"]),
        eds_host=str(body.get("edsHost") or settings["edsHost"]),
        eds_port=int(body.get("edsPort") or settings["edsPort"]),
        eds_enabled=bool(body.get("edsEnabled", settings["edsEnabled"])),
        eds_retry_count=eds_retry_count,
        pixel_avg=_capture_level((snapshot.get("task") or {}).get("captureLevel", 7)),
        max_capture_attempts=_max_capture_attempts((snapshot.get("task") or {}).get("maxCaptureAttempts", 2)),
        restore_start_position=bool(body.get("restoreStartPosition", settings["restoreStartPosition"])),
        auto_gun_on=bool(body.get("autoGunOn", False)),
        gun_off_after_task=bool((snapshot.get("task") or {}).get("gunOffAfterTask", False)) if for_analysis else False,
        calibration_strategy=str(body.get("mode") or "multi_particle"),
        expected_input_sha256=body.get("inputSha256"),
    )


class AcceptanceMockSpectrumAnalyzer:
    """Deterministic EDS substitute used only when ZEM_FAKE_EDS is explicit.

    The SEM image, stage movement, particle localization, and report generation
    remain real. Only the unavailable EDS acquisition is replaced, and every
    returned spectrum carries an audit marker so it cannot be confused with
    customer production data.
    """

    def __init__(self) -> None:
        from backend.cleanliness_core.spectrum_device import MockSpectrumDevice

        self.device = MockSpectrumDevice()

    def check_ready(self) -> dict[str, Any]:
        return {
            "status": "READY",
            "device": "Mock EDS (acceptance test)",
            "protocol": "deterministic_local_mock",
            "mock": True,
        }

    def analyze(
        self,
        particle: Any,
        candidate: Any,
        image_path: Path,
        safe_point: tuple[int, int] | None,
    ) -> dict[str, Any]:
        contour = getattr(getattr(candidate, "candidate", None), "contour", None)
        if contour is None:
            points = []
        elif hasattr(contour, "tolist"):
            points = contour.tolist()
        else:
            points = [list(point) for point in contour]
        spectrum = self.device.acquire_particle_spectrum(
            {
                "label": str(particle.sequence),
                "points": points,
                "area": float(getattr(particle, "area_um2", 0.0) or 0.0),
            },
            {"image_path": str(image_path)},
        )
        spectrum.update(
            {
                "device": "Mock EDS (acceptance test)",
                "target": {
                    "mode": "particle_safe_interior_point",
                    "x": int(safe_point[0]) if safe_point else None,
                    "y": int(safe_point[1]) if safe_point else None,
                },
                "mock": True,
                "notice": "模拟能谱，仅用于软件真机验收，不代表真实材料成分",
            }
        )
        return spectrum

    def close(self) -> None:
        return


def _navigation_to_bgr(image: Any):
    import cv2
    import numpy as np

    array = np.asarray(image)
    if array.dtype != np.uint8:
        minimum = float(array.min()) if array.size else 0.0
        maximum = float(array.max()) if array.size else 0.0
        if maximum <= minimum:
            array = np.zeros(array.shape, dtype=np.uint8)
        else:
            array = np.clip(np.rint((array.astype(np.float32) - minimum) / (maximum - minimum) * 255.0), 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim == 3 and array.shape[2] == 3:
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    if array.ndim == 3 and array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    raise ValueError(f"导航图数据维度不受支持：{array.shape}")


def _image_data_url(image: Any) -> str:
    import cv2

    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise OSError("导航图预览编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def fetch_and_detect_ab(task_id: str) -> dict[str, Any]:
    if not JOBS.allow_hardware_jobs:
        raise ValueError("模拟 MCP 模式不允许调用旧桌面硬件入口")
    with JOBS.control.operation(f"desktop:ab-navigation:{task_id}", writes=False):
        return _fetch_and_detect_ab(task_id)


def _fetch_and_detect_ab(task_id: str) -> dict[str, Any]:
    from backend.cleanliness_core.ab_point_calibration import annotate_navigation_ab, detect_navigation_ab, nav_pixel_to_stage, validate_ab_export
    from backend.cleanliness_core.cleanliness_detection import load_cleanliness_config
    from backend.cleanliness_core.microscope_discovery import validate_zem_status
    from backend.cleanliness_core.optical_particle_analysis import load_optical_export
    from zemrpc.zem.zem_api import zem_api

    path, manifest = task_manifest(task_id)
    if path.name != "draft.json":
        raise ValueError("当前任务已经生成正式标定结果，不能重新获取导航图")
    export = load_optical_export(Path(str(manifest["input"]["path"])))
    validate_ab_export(export)
    body = manifest.get("request") or {}
    settings = app_settings()
    host = str(body.get("microscopeHost") or settings["microscopeHost"])
    port = int(body.get("microscopePort") or settings["microscopePort"])
    status = str(validate_zem_status(zem_api, (host, port)))
    if status.lower() != "idle":
        raise RuntimeError(f"电镜当前状态不是 Idle：{status}")
    zem_api.sync_stage_and_scan_field()
    navigation = zem_api.get_nav_img()
    raw_image = getattr(navigation, "img", None)
    raw_bounds = getattr(navigation, "bounds", None)
    if raw_image is None or raw_bounds is None:
        raise ValueError("导航图 API 未返回图像或 bounds")
    image = _navigation_to_bgr(raw_image)
    bounds = tuple(float(value) for value in raw_bounds)
    nav_pixel_to_stage((image.shape[1] / 2, image.shape[0] / 2), image.shape[:2], bounds)

    config = load_cleanliness_config(_task_config_path(path.parent))
    ab = config.get("abPoint") or {}
    detection = detect_navigation_ab(
        image,
        fixture_model_path=MODEL_FILES["fixture"],
        hole_model_path=MODEL_FILES["hole"],
        input_size=int(ab.get("inputSize") or 640),
        fixture_confidence=float(ab.get("fixtureConfidence") or 0.1),
        hole_confidence=float(ab.get("holeConfidence") or 0.1),
        nms_iou_threshold=float(ab.get("nmsIouThreshold") or 0.45),
        crop_margin=float(ab.get("cropMargin") or 0.12),
        dedup_distance_px=float(ab.get("dedupDistancePx") or 30.0),
    )
    annotated = annotate_navigation_ab(image, detection)
    AB_SESSIONS[task_id] = {"image": image, "bounds": bounds, "detection": detection, "inputSha256": export.sha256}
    return {
        "taskId": task_id,
        "boundsUm": list(bounds),
        "imageShape": list(image.shape[:2]),
        "rawPreviewDataUrl": _image_data_url(image),
        "previewDataUrl": _image_data_url(annotated),
        "detection": detection.to_dict(),
    }


def save_manual_ab_assignment(task_id: str, body: dict[str, Any]) -> dict[str, Any]:
    from backend.cleanliness_core.cleanliness_task import save_ab_point_calibration

    if JOBS.active_for_task(task_id) is not None:
        raise ValueError("当前任务仍有作业运行，不能设置人工 A/B 坐标")
    path, manifest = task_manifest(task_id)
    if manifest.get("status") != "DRAFT":
        raise ValueError("仅草稿任务可提交手动 AB 坐标，请新建任务")
    if body.get("coordinateUnit") != "mm":
        raise ValueError("手动 AB 坐标单位必须为 mm")
    points = {}
    for label in ("a", "b"):
        point = body.get(label)
        if not isinstance(point, dict):
            raise ValueError("请完整填写 A/B 点 X/Y 坐标")
        values = [point.get("x"), point.get("y")]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
            raise ValueError("AB 坐标必须为数值")
        points[label] = dict(zip(("x", "y"), (v * 1000 for v in values)))
    snapshot = {**(manifest.get("request") or {}), "mode": "ab_points",
                "inputSha256": manifest["input"]["sha256"]}
    config = _make_config(Path(manifest["input"]["path"]), path.parent, snapshot)
    result = save_ab_point_calibration(config, manual_stage_points=points)
    result["taskId"] = task_id
    result["request"] = {**snapshot, "mode": "ab_manual"}
    atomic_json(path.parent / "task.json", result)
    AB_SESSIONS.pop(task_id, None)
    return {**result, "folderId": task_id, "taskDirectory": str(path.parent)}


def save_ab_assignment(task_id: str, body: dict[str, Any]) -> dict[str, Any]:
    from backend.cleanliness_core.cleanliness_task import save_ab_point_calibration

    session = AB_SESSIONS.get(task_id)
    if not session:
        raise ValueError("AB 导航图会话不存在，请重新获取导航图")
    path, manifest = task_manifest(task_id)
    request = manifest.get("request") or {}
    request["mode"] = str(body.get("mode") or ("ab_particle" if request.get("mode") == "ab_particle" else "ab_points"))
    if request["mode"] not in {"ab_points", "ab_particle"}:
        raise ValueError("AB 导航仅支持 AB 标定或 AB 导航 + 颗粒精校准")
    request["inputSha256"] = session["inputSha256"]
    config = _make_config(Path(str(manifest["input"]["path"])), path.parent, request)
    result = save_ab_point_calibration(
        config,
        navigation_image=session["image"],
        navigation_bounds=session["bounds"],
        detection=session["detection"],
        a_index=int(body.get("aIndex", 0)),
        b_index=int(body.get("bIndex", 1)),
        handedness=str(body.get("handedness") or "right"),
    )
    # The desktop task folder ID is the customer-facing identifier. The legacy
    # engine creates an internal UUID when it materializes task.json, so replace
    # it here before later analysis and report generation use the manifest.
    result["taskId"] = task_id
    result["request"] = request
    atomic_json(path.parent / "task.json", result)
    AB_SESSIONS.pop(task_id, None)
    return {**result, "folderId": task_id, "taskDirectory": str(path.parent)}


def retry_ab_refinement(task_id: str) -> dict[str, Any]:
    """Return a reviewed AB task to the explicit SEM-refinement step.

    This endpoint only changes task state. It never connects to the microscope,
    moves the stage, or starts a background job; the operator must click the
    SEM refinement button again before any hardware action can happen.
    """
    if JOBS.active_for_task(task_id) is not None:
        raise ValueError("当前任务仍在运行，不能重置精校准")
    path, manifest = task_manifest(task_id)
    calibration = manifest.get("calibration") or {}
    if str(calibration.get("strategy")) not in {"ab_points", "ab_particle"}:
        raise ValueError("当前任务不是 AB 点标定")
    if manifest.get("status") != "CALIBRATION_PENDING_REVIEW":
        raise ValueError(f"当前任务不能返回待精定位：{manifest.get('status')}")
    ab_result = calibration.get("abPointResult") or {}
    previous = ab_result.get("semRefinement")
    if previous:
        history = ab_result.setdefault("semRefinementHistory", [])
        history.append(copy.deepcopy(previous))
        ab_result.pop("semRefinement", None)
    ab_result["status"] = "PENDING_SEM_REFINEMENT"
    calibration["result"] = None
    calibration["refinedTransform"] = None
    calibration["review"] = {
        "accepted": False,
        "reviewer": "desktop",
        "reviewedAt": now(),
        "note": "光镜与 SEM 对比不一致，操作员要求重新精定位",
    }
    manifest["status"] = "AB_POINTS_PENDING_SEM_REFINEMENT"
    manifest["error"] = None
    atomic_json(path, manifest)
    atomic_json(path.parent / "ab_point_result.json", ab_result)
    return {**manifest, "folderId": task_id, "taskDirectory": str(path.parent)}


def analysis_resume_state(manifest: dict[str, Any]) -> dict[str, int]:
    status = str(manifest.get("status") or "")
    if status not in {"RUNNING", "STOPPED", "FAILED"}:
        raise ValueError(f"当前任务状态不能继续检测：{status}")
    review = ((manifest.get("calibration") or {}).get("review") or {})
    if not bool(review.get("accepted")):
        raise ValueError("必须先由用户确认本次标定结果")

    selected = [
        int(value)
        for value in ((manifest.get("selection") or {}).get("sequences") or [])
    ]
    selected_set = set(selected)
    execution_order = [int(value) for value in manifest.get("executionOrder") or []]
    if (
        not selected
        or len(execution_order) != len(selected)
        or len(execution_order) != len(set(execution_order))
        or set(execution_order) != selected_set
    ):
        raise ValueError("原检测执行顺序缺失或与当前颗粒列表不一致，不能安全继续")

    processed: set[int] = set()
    for record in manifest.get("particles") or []:
        if str(record.get("status") or "") not in {"SUCCEEDED", "FAILED"}:
            continue
        sequence = int(record.get("sequence"))
        if sequence not in selected_set:
            raise ValueError(f"历史检测结果包含未知颗粒：#{sequence}")
        if sequence in processed:
            raise ValueError(f"历史检测结果包含重复颗粒：#{sequence}")
        processed.add(sequence)

    remaining = [sequence for sequence in execution_order if sequence not in processed]
    if not remaining:
        raise ValueError("当前任务没有未处理颗粒，不能继续检测")
    return {
        "processed": len(processed),
        "total": len(execution_order),
        "remaining": len(remaining),
        "nextRank": execution_order.index(remaining[0]) + 1,
        "nextSequence": remaining[0],
    }


def reset_analysis(task_id: str) -> dict[str, Any]:
    """Reset an interrupted or failed analysis without touching the microscope.

    A desktop/backend process can be terminated before the hardware routine gets
    a chance to write its final STOPPED state. In that case task.json remains
    RUNNING forever. This explicit operator action archives an audit summary and
    returns the already accepted calibration to the ready-to-run state. It does
    not connect to a device, move the stage, or change the gun state.
    """
    path, manifest = task_manifest(task_id)
    if path.name != "task.json":
        raise ValueError("任务尚未完成坐标标定，不能重置检测")
    active = JOBS.active_for_task(task_id)
    if active is not None:
        raise ValueError("当前检测作业仍在运行，请先停止并等待作业结束")
    calibration = manifest.get("calibration") or {}
    review = calibration.get("review") or {}
    if not bool(review.get("accepted")):
        raise ValueError("必须先由用户确认本次标定结果")
    previous_status = str(manifest.get("status") or "")
    allowed = {"RUNNING", "STOPPED", "FAILED", "COMPLETED_WITH_FAILURES"}
    if previous_status not in allowed:
        raise ValueError(f"当前任务状态不能重新开始检测：{previous_status}")

    particles = list(manifest.get("particles") or [])
    manifest.setdefault("analysisHistory", []).append(
        {
            "resetAt": now(),
            "previousStatus": previous_status,
            "analysisStartedAt": manifest.get("analysisStartedAt"),
            "analysisCompletedAt": manifest.get("analysisCompletedAt"),
            "processed": len(particles),
            "succeeded": sum(item.get("status") == "SUCCEEDED" for item in particles),
            "failed": sum(item.get("status") != "SUCCEEDED" for item in particles),
            "reports": copy.deepcopy(manifest.get("reports")),
            "cleanup": copy.deepcopy(manifest.get("cleanup")),
            "reason": "operator_requested_analysis_restart",
        }
    )
    manifest["status"] = "CALIBRATION_ACCEPTED"
    manifest["particles"] = []
    manifest["reports"] = None
    manifest["cleanup"] = {}
    manifest["error"] = None
    manifest["completedAt"] = None
    manifest["analysisResetAt"] = now()
    for key in ("analysisStartedAt", "analysisCompletedAt", "executionOrder", "executionRoute", "baselineImaging", "reportError"):
        manifest.pop(key, None)
    atomic_json(path, manifest)
    return {**manifest, "folderId": task_id, "taskDirectory": str(path.parent)}


def _run_job(job_id: str, task_id: str, operation: str) -> None:
    try:
        _run_job_impl(job_id, task_id, operation)
    except Exception as error:
        log_event("error", "job_thread_crashed", jobId=job_id, taskId=task_id, operation=operation, error=error, traceback=traceback.format_exc())
        JOBS.update(job_id, status="failed", message=str(error), error=str(error), traceback=traceback.format_exc(), completedAt=now())


def _execute_preflight(config, manifest, operation, *, should_stop, progress):
    from backend.cleanliness_core.preflight import run_preflight
    from backend.cleanliness_core.cleanliness_detection import load_cleanliness_config, CandidateSegmentationConfig
    from backend.cleanliness_core.cleanliness_task import RealSpectrumAnalyzer
    enabled = bool(app_settings()['preflightEnabled'])
    if not enabled:
        return run_preflight(config, manifest, operation, enabled=False)
    settings = load_cleanliness_config(config.settings_path)
    segmentation = (settings.get('calibration') or {}).get('candidateSegmentation') or {}
    calibration = manifest.get('calibration') or {}
    manual = bool((calibration.get('abPointResult') or {}).get('manualStagePoints'))
    models = []
    if operation in {'analysis','analysis_resume','calibration'} and segmentation.get('backend') == 'yolo':
        models.append(CandidateSegmentationConfig.from_dict(segmentation, settings_path=config.settings_path).yolo_model_path)
    if operation == 'ab_refine' and not manual:
        models.extend([MODEL_FILES['fixture'], MODEL_FILES['hole']])
    def eds_check():
        if FAKE_EDS_ENABLED:
            return {'status':'READY', 'mock':True}
        analyzer = RealSpectrumAnalyzer(config.eds_host, config.eds_port, config.microscope_host, config.microscope_port, timeout=3)
        try: return analyzer.check_ready()
        finally: analyzer.close()
    report = run_preflight(config, manifest, operation, enabled=enabled, model_paths=models,
                           eds_check=eds_check, should_stop=should_stop, progress=progress)
    if FAKE_EDS_ENABLED and config.eds_enabled and report['enabled']:
        report['items'].append({'id':'fake_eds','label':'验收模式','status':'WARN','detail':'模拟能谱，不代表真实 EDS 服务就绪'})
    return report


def _run_job_impl(job_id: str, task_id: str, operation: str) -> None:
    path, manifest = task_manifest(task_id)
    task_dir = path.parent
    JOBS.update(job_id, status="running", startedAt=now())
    log_event("info", "job_started", jobId=job_id, taskId=task_id, operation=operation, taskDirectory=str(task_dir), inputPath=(manifest.get("input") or {}).get("path"))

    last_logged_progress = -1
    def progress(index: int, total: int, message: str) -> None:
        nonlocal last_logged_progress
        JOBS.update(job_id, progress=index, total=total, message=message)
        interval = max(1, total // 10) if total else 1
        if index == total or index == 0 or index - last_logged_progress >= interval:
            last_logged_progress = index
            log_event("info", "job_progress", jobId=job_id, taskId=task_id, operation=operation, progress=index, total=total, message=message)

    def stage_event(event: str, fields: dict[str, Any]) -> None:
        log_event(
            "info",
            event,
            jobId=job_id,
            taskId=task_id,
            operation=operation,
            **fields,
        )

    try:
        if operation == 'sem_report':
            from backend.cleanliness_core.sem_direct import write_reports
            if manifest.get('mode') != 'sem_direct' or manifest['status'] == 'RUNNING':
                raise ValueError('请先暂停扫描，再生成报告')
            progress(0, 1, '正在生成完整报告')
            started = time.perf_counter()
            try:
                write_reports(task_dir, manifest)
                manifest.update(reportState='READY')
                manifest.pop('reportError', None)
            except Exception as error:
                manifest.update(reportState='FAILED', reportError=str(error))
                raise
            finally:
                manifest.setdefault('manualReportTimings', []).append({'at':now(),'seconds':time.perf_counter()-started})
                atomic_json(path, manifest)
                from backend.cleanliness_core.sem_progress import refresh_inactive_header
                refresh_inactive_header(task_dir,manifest)
            JOBS.update(job_id, status='completed', progress=1, total=1, message='报告已生成', completedAt=now())
            return
        if operation == 'sem_analysis':
            from backend.cleanliness_core.sem_direct import run_task
            from backend.cleanliness_core.preflight import run_preflight
            from backend.cleanliness_core.cleanliness_task import RealSpectrumAnalyzer
            from types import SimpleNamespace
            options = manifest['request']
            config = SimpleNamespace(output_dir=task_dir, settings_path=task_dir/'.cleanliness-settings.json',
                                     microscope_host=options['microscopeHost'], microscope_port=options['microscopePort'],
                                     eds_enabled=options['edsEnabled'])
            def sem_eds_check():
                analyzer = RealSpectrumAnalyzer(options['edsHost'], options['edsPort'], options['microscopeHost'], options['microscopePort'], timeout=3)
                try:
                    return analyzer.check_ready()
                finally:
                    analyzer.close()
            preflight_started = time.perf_counter()
            preflight = run_preflight(config, manifest, operation, enabled=options['preflightEnabled'],
                                      model_paths=[] if manifest['model'].get('backend') == 'sam3' else [Path(manifest['model']['path'])], eds_check=sem_eds_check,
                                      should_stop=lambda: JOBS.cancelled(job_id), progress=progress)
            manifest['preflight'] = preflight
            manifest.setdefault('preflightTimings', []).append({'checkedAt':preflight.get('checkedAt'),'seconds':time.perf_counter()-preflight_started,'passed':preflight['passed']})
            atomic_json(path, manifest)
            JOBS.update(job_id, preflight=preflight)
            if not preflight['passed']:
                raise ValueError('开工检查未通过：'+'；'.join(i['detail'] for i in preflight['items'] if i['status']=='FAIL'))
            result = run_task(task_dir, progress=progress, should_stop=lambda: JOBS.cancelled(job_id))
            # The SEM page fetches the final task once through /tasks/{id}.
            # Do not retain a second full particle/spectrum graph in job history.
            JOBS.update(job_id, status='completed', message=result['status'], result={
                'taskId':task_id,'mode':'sem_direct','status':result['status'],'reports':result.get('reports'),
                'folderId':task_id,'taskDirectory':str(task_dir)}, completedAt=now())
            return
        draft_path = task_dir / "draft.json"
        draft = read_json(draft_path) if draft_path.is_file() else {}
        body = {**(draft.get("request") or {}), **(manifest.get("request") or {})}
        config = _make_config(Path(manifest["input"]["path"]), task_dir, body, for_analysis=operation in {"analysis", "analysis_resume"})
        from backend.cleanliness_core.cleanliness_task import run_cleanliness_analysis, run_cleanliness_calibration, review_cleanliness_calibration

        preflight = None
        if operation in {'calibration','ab_refine','analysis','analysis_resume'}:
            preflight = _execute_preflight(config, manifest, operation, should_stop=lambda:JOBS.cancelled(job_id), progress=progress)
            manifest['preflight'] = preflight
            manifest.setdefault('preflightHistory', []).append(preflight)
            atomic_json(path, manifest)
            JOBS.update(job_id, preflight=preflight)
            if not preflight['passed']:
                problems = '；'.join(f"{item['label']}：{item['detail']}" for item in preflight['items'] if item['status']=='FAIL')
                raise RuntimeError('开工检查未通过，未执行设备动作：'+problems)

        if operation == "calibration":
            result = run_cleanliness_calibration(
                config,
                progress=progress,
                should_stop=lambda: JOBS.cancelled(job_id),
                event_logger=stage_event,
            )
        elif operation in {"analysis", "analysis_resume"}:
            resume_analysis = operation == "analysis_resume"
            spectrum_analyzer = AcceptanceMockSpectrumAnalyzer() if FAKE_EDS_ENABLED else None
            if FAKE_EDS_ENABLED:
                acceptance_notice = "模拟能谱，仅用于软件真机验收，不代表真实材料成分"
                manifest.setdefault("acceptanceTest", {})["fakeEds"] = True
                manifest["acceptanceTest"]["notice"] = acceptance_notice
                atomic_json(task_dir / "task.json", manifest)
            result = run_cleanliness_analysis(
                task_dir,
                config=config,
                spectrum_analyzer=spectrum_analyzer,
                progress=progress,
                should_stop=lambda: JOBS.cancelled(job_id),
                event_logger=stage_event,
                resume=resume_analysis,
            )
            if FAKE_EDS_ENABLED:
                result.setdefault("acceptanceTest", {})["fakeEds"] = True
                result["acceptanceTest"]["notice"] = acceptance_notice
                atomic_json(task_dir / "task.json", result)
        elif operation == "review":
            result = review_cleanliness_calibration(task_dir, accepted=bool(body.get("accepted", True)), selected_group=int(body.get("selectedGroup", 0)), reviewer="desktop")
        elif operation == "ab_refine":
            if (manifest.get("calibration") or {}).get("strategy") == "ab_particle":
                from backend.cleanliness_core.ab_particle_calibration import run_ab_particle_calibration
                result = run_ab_particle_calibration(
                    task_dir,
                    config=config,
                    progress=progress,
                    should_stop=lambda: JOBS.cancelled(job_id),
                    event_logger=stage_event,
                )
            else:
                from backend.cleanliness_core.cleanliness_task import refine_ab_point_calibration
                result = refine_ab_point_calibration(
                    task_dir,
                    config=config,
                    progress=progress,
                    should_stop=lambda: JOBS.cancelled(job_id),
                    event_logger=stage_event,
                )
        else:
            raise ValueError(f"不支持的作业：{operation}")
        result["taskId"] = task_id
        if preflight is not None:
            result['preflight'] = preflight
            result['preflightHistory'] = manifest.get('preflightHistory', [])
        atomic_json(task_dir / "task.json", result)
        JOBS.update(job_id, status="completed", progress=JOBS.get(job_id).get("total", 0), message=result.get("status", "已完成"), result={**result, "folderId": task_id, "taskDirectory": str(task_dir)}, completedAt=now())
        log_event("info", "job_completed", jobId=job_id, taskId=task_id, operation=operation, status=result.get("status"), particleCount=len(result.get("particles") or []))
    except Exception as error:
        JOBS.update(job_id, status="failed", message=str(error), error=str(error), traceback=traceback.format_exc(), completedAt=now())
        log_event("error", "job_failed", jobId=job_id, taskId=task_id, operation=operation, error=error, traceback=traceback.format_exc())


class Handler(BaseHTTPRequestHandler):
    server_version = "ZEMCleanliness/0.1"

    def _begin_request(self) -> None:
        self._request_id = self.headers.get("X-Request-Id") or str(uuid.uuid4())
        self._request_started = time.monotonic()
        log_event("info", "http_request_started", requestId=self._request_id, method=self.command, route=urlparse(self.path).path, client=self.client_address[0] if self.client_address else None)

    def _send(self, status: int, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)
        log_event("info" if status < 400 else "warn", "http_request_finished", requestId=getattr(self, "_request_id", None), method=self.command, route=urlparse(self.path).path, status=status, durationMs=round((time.monotonic() - getattr(self, "_request_started", time.monotonic())) * 1000, 2), responseBytes=len(data))

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        if isinstance(body, dict):
            log_event("debug", "http_request_body", requestId=getattr(self, "_request_id", None), keys=list(body.keys()), inputPath=body.get("inputPath"), selectedCount=len(body.get("selectedSequences", [])) if isinstance(body.get("selectedSequences"), list) else None)
        return body

    def _send_file(self, path: Path) -> None:
        content_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".html": "text/html; charset=utf-8",
            ".pdf": "application/pdf",
            ".csv": "text/csv; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json; charset=utf-8",
        }
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_types.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)
        log_event("info", "file_response", requestId=getattr(self, "_request_id", None), path=str(path), bytes=len(data))

    def do_OPTIONS(self) -> None:
        self._begin_request()
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Request-Id")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        self._begin_request()
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/")
        try:
            if route == "" or route == "/health":
                return self._send(
                    200,
                    {
                        "ok": True,
                        "service": "zem-cleanliness",
                        "version": "0.1.0",
                        "capabilities": {"edsRetryCount": True, "abParticleCalibration": True},
                        "time": now(),
                    },
                )
            if route == "/models":
                return self._send(200, {"models": model_status()})
            if route == "/openapi.json":
                from backend.cleanliness_core.report_data_openapi import openapi_spec
                return self._send(200, openapi_spec())
            if route == "/docs" or route.startswith("/docs/"):
                relative = route[len("/docs/"):].strip("/") if route != "/docs" else "index.html"
                asset = (RESOURCE_ROOT / "swagger-ui" / relative).resolve()
                if (RESOURCE_ROOT / "swagger-ui") not in asset.parents or not asset.is_file():
                    return self._send(404, {"error": "接口文档资源不存在"})
                return self._send_file(asset)
            if route.startswith('/sem/active/'):
                return self._send(200, {'job': JOBS.active_for_task(route.split('/')[-1])})
            if route.startswith('/tasks/') and route.endswith('/report-data'):
                from backend.cleanliness_core.report_data import build_report_data
                task_id = route.split('/')[2]
                path, manifest = task_manifest(task_id)
                base_url = f"http://{self.headers.get('Host') or '127.0.0.1:8765'}"
                return self._send(200, build_report_data(path.parent, manifest, base_url=base_url))
            if route.startswith('/tasks/') and route.endswith('/sem/progress'):
                from backend.cleanliness_core.sem_progress import read_progress,header_view,particle_view,tile_view
                from collections import Counter
                task_id = route.split('/')[2]
                # Restrict this optimized reader to a single task-directory name.
                directory = (TASK_ROOT/task_id).resolve()
                if directory.parent != TASK_ROOT.resolve() or not (directory/'task.json').is_file():
                    raise ValueError('任务不存在')
                cursor = parse_qs(parsed.query).get('cursor',[''])[0]
                result = read_progress(directory,cursor)
                if result is None:
                    _, manifest = task_manifest(task_id)
                    if manifest.get('mode') != 'sem_direct' or not manifest.get('plan'):
                        raise ValueError('任务没有直接 SEM 扫描计划')
                    particles=manifest['particles']; tiles=manifest['plan']['tiles']
                    view=header_view(manifest,Counter(p['status'] for p in particles),sum(t['status']=='COMPLETED' for t in tiles))
                    view.update(particles=[particle_view(p) for p in particles[-100:]],taskDirectory=str(directory))
                    result={'cursor':'','reset':True,'task':view,'tiles':[tile_view(t) for t in tiles]}
                return self._send(200,result)
            if route == "/settings":
                return self._send(200, app_settings())
            if route == "/devices":
                return self._send(200, {"devices": device_status()})
            if route == "/tasks":
                tasks = []
                task_files = list(TASK_ROOT.glob("*/task.json"))
                formal_dirs = {item.parent for item in task_files}
                task_files.extend(item for item in TASK_ROOT.glob("*/draft.json") if item.parent not in formal_dirs)
                for task_file in sorted(task_files, key=lambda item: item.stat().st_mtime, reverse=True):
                    try:
                        from backend.cleanliness_core.sem_progress import read_summary
                        item = read_summary(task_file.parent) if task_file.name=='task.json' else None
                        if item is None:
                            stat=task_file.stat()
                            item=_cached_task_list_summary(str(task_file),stat.st_mtime_ns,stat.st_size)
                            if item.get('_needsReplay'):
                                _, item = task_manifest(task_file.parent.name)
                        input_summary = dict(item.get("input") or {})
                        selected_count = (item.get("selection") or {}).get("count")
                        if selected_count is not None:
                            input_summary["selectedCount"] = int(selected_count)
                        tasks.append({"id": task_file.parent.name, "mode": item.get("mode"), "status": item.get("status"), "input": input_summary, "updatedAt": item.get("completedAt") or item.get("createdAt")})
                    except Exception as error:
                        log_event("warn", "task_manifest_skipped", requestId=self._request_id, path=str(task_file), error=error, traceback=traceback.format_exc())
                        continue
                return self._send(200, {"tasks": tasks})
            if route.startswith("/jobs/"):
                job = JOBS.get(route.split("/", 2)[2])
                return self._send(200 if job else 404, job or {"error": "作业不存在"})
            if route == "/chat/file":
                # 给 chat 面板回显 agent 产出的图片；demo 阶段只校验扩展名。
                target = Path(unquote(parse_qs(parsed.query).get("path", [""])[0])).resolve()
                if target.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".gif"} or not target.is_file():
                    return self._send(404, {"error": "图片不存在"})
                return self._send_file(target)
            if route == "/chat/sessions":
                return self._send(200, {"sessions": _chat_manager().list_sessions()})
            if route == "/chat/runtime":
                from backend.chat_runner import model_configuration
                from backend import chat_mcp
                caps = MCP_CONTROL.capabilities() if MCP_CONTROL is not None else {}
                configured = load_mcp_device_config()
                return self._send(200, {**model_configuration(), "mcp": {
                    "available": MCP_CONTROL is not None, "mode": caps.get("mode", "unavailable"),
                    "captureEnabled": caps.get("capture_enabled", False), "moveEnabled": caps.get("move_enabled", False),
                    "captureConfigured": configured.get("allowCapture") is True,
                    "moveConfigured": configured.get("boundsUm") is not None,
                    "servers": chat_mcp.allowed_servers(), "uiMcp": chat_mcp.ui_available()}})
            if route.startswith("/chat/sessions/"):
                parts = route[len("/chat/sessions/"):].split("/")
                if len(parts) == 1:
                    after = int(parse_qs(parsed.query).get("after", ["0"])[0] or 0)
                    return self._send(200, _chat_manager().get_session(parts[0], after=after))
                return self._send(404, {"error": "接口不存在"})
            if route.startswith("/tasks/"):
                _, _, task_id = route.partition("/tasks/")
                if "/files/" in task_id:
                    task_id, relative = task_id.split("/files/", 1)
                    task_dir = (TASK_ROOT / task_id).resolve()
                    target = (task_dir / unquote(relative)).resolve()
                    if task_dir not in target.parents or not target.is_file():
                        return self._send(404, {"error": "任务文件不存在"})
                    return self._send_file(target)
                _, manifest = task_manifest(task_id)
                return self._send(200, {**manifest, "taskDirectory": str(TASK_ROOT / task_id), "folderId": task_id})
            return self._send(404, {"error": "接口不存在"})
        except ChatError as error:
            return self._send(error.status, {"error": str(error)})
        except Exception as error:
            log_event("error", "http_request_failed", requestId=self._request_id, method="GET", route=route, error=error, traceback=traceback.format_exc())
            return self._send(400, {"error": str(error)})

    def do_POST(self) -> None:
        self._begin_request()
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/")
        try:
            if route == '/mcp/call':
                return self._mcp_call()
            body = self._body()
            if route == '/sem/navigation':
                return self._send(201, create_sem_navigation())
            if route.startswith('/tasks/') and route.endswith('/sem/report'):
                task_id = route.split('/')[2]
                _, manifest = task_manifest(task_id)
                if manifest.get('mode') != 'sem_direct' or manifest['status'] == 'RUNNING':
                    raise ValueError('请先暂停扫描，再生成报告')
                job = JOBS.create(task_id, 'sem_report')
                threading.Thread(target=_run_job, args=(job['id'], task_id, 'sem_report'), daemon=True).start()
                return self._send(202, job)
            if route.startswith('/tasks/') and route.endswith('/sem/plan'):
                return self._send(200, save_sem_plan(route.split('/')[2], body))
            if route.startswith('/tasks/') and route.endswith('/sem/start'):
                task_id = route.split('/')[2]
                _, manifest = task_manifest(task_id)
                if manifest.get('mode') != 'sem_direct' or not manifest.get('plan'):
                    raise ValueError('请先保存直接 SEM 扫描区域')
                if manifest['status'] not in {'DRAFT', 'STOPPED', 'FAILED', 'RUNNING'}:
                    raise ValueError('该任务已完成，请新建任务')
                job = JOBS.create(task_id, 'sem_analysis')
                threading.Thread(target=_run_job, args=(job['id'], task_id, 'sem_analysis'), daemon=True).start()
                return self._send(202, job)
            if route == "/optical/preview":
                return self._send(200, _load_optical(Path(str(body["inputPath"])), body.get("filters")))
            if route == "/settings":
                return self._send(200, save_app_settings(body))
            if route == "/engineer/unlock":
                return self._send(200, unlock_engineer(body.get("password")))
            if route == "/engineer/settings":
                return self._send(200, engineer_settings(body.get("token")))
            if route == "/engineer/settings/save":
                return self._send(200, save_engineer_settings(body.get("token"), body.get("values")))
            if route == "/engineer/lock":
                return self._send(200, lock_engineer(body.get("token")))
            if route == "/tasks":
                task_id = str(uuid.uuid4())
                manifest, preview = _build_task_draft(task_id, body)
                task_dir = TASK_ROOT / task_id
                task_dir.mkdir(parents=True, exist_ok=False)
                _task_config_path(task_dir, manifest["request"])
                atomic_json(task_dir / "draft.json", manifest)
                log_event("info", "task_created", requestId=self._request_id, taskId=task_id, inputPath=manifest["input"]["path"], particleCount=preview["particleCount"], selectedCount=manifest["selection"]["count"], mode=body.get("mode"))
                return self._send(201, manifest)
            if route.endswith("/input") and route.startswith("/tasks/"):
                task_id = route.split("/")[2]
                return self._send(200, revise_task_input(task_id, body))
            if route.endswith("/ab/navigation") and route.startswith("/tasks/"):
                task_id = route.split("/")[2]
                result = fetch_and_detect_ab(task_id)
                return self._send(200, result)
            if route.endswith("/ab/assign") and route.startswith("/tasks/"):
                task_id = route.split("/")[2]
                return self._send(200, save_ab_assignment(task_id, body))
            if route.endswith("/ab/manual") and route.startswith("/tasks/"):
                return self._send(200, save_manual_ab_assignment(route.split("/")[2], body))
            if route.endswith("/ab/manual-stage-points") and route.startswith("/tasks/"):
                from backend.cleanliness_core.cleanliness_task import (
                    set_manual_ab_stage_points,
                )

                task_id = route.split("/")[2]
                if JOBS.active_for_task(task_id) is not None:
                    raise ValueError("当前任务仍有作业运行，不能设置人工 A/B 坐标")
                a = body.get("a") or {}
                b = body.get("b") or {}
                result = set_manual_ab_stage_points(
                    TASK_ROOT / task_id,
                    a_stage_um=(a.get("x"), a.get("y")),
                    b_stage_um=(b.get("x"), b.get("y")),
                    operator=str(body.get("operator") or "user"),
                )
                return self._send(200, result)
            if route.endswith("/ab/refine") and route.startswith("/tasks/"):
                task_id = route.split("/")[2]
                job = JOBS.create(task_id, "ab_refine")
                threading.Thread(target=_run_job, args=(job["id"], task_id, "ab_refine"), daemon=True).start()
                return self._send(202, job)
            if route.endswith("/ab/retry") and route.startswith("/tasks/"):
                task_id = route.split("/")[2]
                return self._send(200, retry_ab_refinement(task_id))
            if route.endswith("/calibration") and route.startswith("/tasks/"):
                task_id = route.split("/")[2]
                draft_path, draft = task_manifest(task_id)
                if draft_path.name == "draft.json":
                    draft["request"] = {**(draft.get("request") or {}), **body, "mode": "multi_particle"}
                    atomic_json(draft_path, draft)
                job = JOBS.create(task_id, "calibration")
                threading.Thread(target=_run_job, args=(job["id"], task_id, "calibration"), daemon=True).start()
                return self._send(202, job)
            if route.endswith("/review") and route.startswith("/tasks/"):
                task_id = route.split("/")[2]
                _, manifest = task_manifest(task_id)
                manifest["request"] = {**manifest.get("request", {}), **body}
                atomic_json(TASK_ROOT / task_id / "task.json", manifest)
                job = JOBS.create(task_id, "review")
                threading.Thread(target=_run_job, args=(job["id"], task_id, "review"), daemon=True).start()
                return self._send(202, job)
            if route.endswith("/analysis") and route.startswith("/tasks/"):
                task_id = route.split("/")[2]
                if JOBS.active_for_task(task_id) is not None:
                    raise ValueError("当前检测作业已经在运行")
                _, manifest = task_manifest(task_id)
                if manifest.get("status") != "CALIBRATION_ACCEPTED":
                    raise ValueError("必须先确认标定；异常中断的任务请选择继续检测或从头重检")
                job = JOBS.create(task_id, "analysis")
                threading.Thread(target=_run_job, args=(job["id"], task_id, "analysis"), daemon=True).start()
                return self._send(202, job)
            if route.endswith("/analysis/resume") and route.startswith("/tasks/"):
                task_id = route.split("/")[2]
                if JOBS.active_for_task(task_id) is not None:
                    raise ValueError("当前检测作业已经在运行")
                _, manifest = task_manifest(task_id)
                resume_state = analysis_resume_state(manifest)
                job = JOBS.create(task_id, "analysis_resume")
                JOBS.update(
                    job["id"],
                    progress=resume_state["processed"],
                    total=resume_state["total"],
                    message=(
                        f"准备从第 {resume_state['nextRank']} 个继续，"
                        f"剩余 {resume_state['remaining']} 个颗粒"
                    ),
                )
                threading.Thread(
                    target=_run_job,
                    args=(job["id"], task_id, "analysis_resume"),
                    daemon=True,
                ).start()
                return self._send(202, JOBS.get(job["id"]))
            if route.endswith("/analysis/reset") and route.startswith("/tasks/"):
                task_id = route.split("/")[2]
                return self._send(200, reset_analysis(task_id))
            if route.startswith("/jobs/") and route.endswith("/cancel"):
                JOBS.cancel(route.split("/")[2])
                return self._send(202, {"status": "cancelling"})
            if route == "/chat/sessions":
                return self._send(201, _chat_manager().create_session(str(body.get("title") or "") or None))
            if route.startswith("/chat/sessions/"):
                parts = route[len("/chat/sessions/"):].split("/")
                if len(parts) == 2 and parts[1] == "messages":
                    return self._send(202, _chat_manager().send_message(parts[0], str(body.get("text") or "")))
                if len(parts) == 2 and parts[1] == "cancel":
                    _chat_manager().cancel(parts[0])
                    return self._send(200, {"status": "cancelled"})
                if len(parts) == 2 and parts[1] == "delete":
                    _chat_manager().delete_session(parts[0])
                    return self._send(200, {"status": "deleted"})
                return self._send(404, {"error": "接口不存在"})
            return self._send(404, {"error": "接口不存在"})
        except ChatError as error:
            log_event("warn", "chat_request_rejected", requestId=self._request_id, method="POST", route=route, error=str(error))
            return self._send(error.status, {"error": str(error)})
        except Exception as error:
            log_event("error", "http_request_failed", requestId=self._request_id, method="POST", route=route, error=error, traceback=traceback.format_exc())
            return self._send(400, {"error": str(error), "traceback": traceback.format_exc()})

    def _mcp_call(self) -> None:
        if MCP_CONTROL is None:
            return self._send(503, {"error": {"code": "SERVICE_UNAVAILABLE", "message": "MCP 设备服务未启动"}})
        authorization = self.headers.get("Authorization", "")
        if (not MCP_TOKEN or not hmac.compare_digest(authorization, "Bearer " + MCP_TOKEN)
                or self.headers.get("Origin") is not None
                or self.client_address[0] not in {"127.0.0.1", "::1"}):
            return self._send(403, {"error": {"code": "FORBIDDEN", "message": "本机 MCP 访问凭据无效"}})
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 65536:
            return self._send(413, {"error": {"code": "INVALID_ARGUMENT", "message": "MCP 请求体大小无效"}})
        from backend.sem_control import SemError
        try:
            body = self._body()
            if not isinstance(body, dict) or set(body) != {"name", "arguments", "caller"}:
                raise SemError("INVALID_ARGUMENT", "MCP 请求结构无效")
            result = MCP_CONTROL.call(body["name"], body["arguments"], body["caller"])
            log_event("info", "mcp_tool_called", tool=body["name"], caller=body["caller"],
                      jobId=result.get("job_id"), simulated=result.get("simulated"))
            return self._send(200, {"result": result})
        except (SemError, DeviceBusy) as error:
            detail = error.record() if isinstance(error, SemError) else {"code": "DEVICE_BUSY", "message": str(error)}
            return self._send(200, {"error": detail})

    def log_message(self, _format: str, *_args: Any) -> None:
        log_event("debug", "http_server_message", message=_format % _args if _args else _format)


def device_status() -> list[dict[str, Any]]:
    device = app_settings()
    devices = [
        {"id": "microscope", "name": "ZEM 扫描电镜", "host": device["microscopeHost"], "port": device["microscopePort"], "status": "unknown", "detail": "尚未探测"},
        {"id": "eds", "name": "EDS 能谱服务", "host": device["edsHost"], "port": device["edsPort"], "status": "unknown", "detail": "尚未探测"},
    ]
    if MCP_CONTROL is not None and MCP_CONTROL.device.simulated:
        for item in devices:
            item.update(status="connected" if item['id'] == 'microscope' else "unknown",
                        detail="模拟 MCP 模式，未连接真实设备")
        return devices
    if JOBS.control.owner is not None:
        for item in devices:
            item.update(status="busy", detail="设备任务执行中，未插入额外通信")
        return devices
    try:
        devices[0].update(
            _microscope_status_bounded(
                str(devices[0]["host"]), int(devices[0]["port"])
            )
        )
    except Exception as error:
        devices[0].update(status="offline", detail=str(error))
        log_event("warn", "microscope_status_failed", host=devices[0]["host"], port=devices[0]["port"], error=error, traceback=traceback.format_exc())
    try:
        with JOBS.control.operation("desktop:eds-status", writes=False):
            with socket.create_connection((str(devices[1]["host"]), int(devices[1]["port"])), timeout=0.8):
                pass
            from backend.cleanliness_core.cleanliness_task import RealSpectrumAnalyzer
            analyzer = RealSpectrumAnalyzer(str(devices[1]["host"]), int(devices[1]["port"]), str(devices[0]["host"]), int(devices[0]["port"]), timeout=2.0)
            try:
                readiness = analyzer.check_ready()
                devices[1].update(status="connected", detail=str(readiness.get("status") or "READY"))
            finally:
                analyzer.close()
    except DeviceBusy as error:
        devices[1].update(status="busy", detail=str(error))
    except Exception as error:
        devices[1].update(status="offline", detail=str(error))
        log_event("warn", "eds_status_failed", host=devices[1]["host"], port=devices[1]["port"], error=error, traceback=traceback.format_exc())
    return devices


def _microscope_status(host: str, port: int) -> dict[str, str]:
    """Read the ZEM status through one lightweight UDP RPC command.

    Port 61557 is a UDP discovery/status endpoint. It is intentionally not
    preflighted with ``socket.create_connection`` because the microscope does
    not expose a TCP listener on that port.
    """
    from backend.cleanliness_core.eds.microscope_jsonrpc import (
        probe_microscope_status,
    )

    try:
        with JOBS.control.operation("desktop:status", writes=False):
            status = probe_microscope_status(str(host), int(port), timeout=1.0)
    except DeviceBusy as error:
        return {"status": "busy", "detail": str(error)}
    return {
        "status": "connected" if status.lower() == "idle" else "busy",
        "detail": status,
    }


def _microscope_status_bounded(
    host: str, port: int, *, timeout_seconds: float = 2.0
) -> dict[str, str]:
    """Keep the HTTP device-status endpoint responsive during UDP timeouts."""

    if not MICROSCOPE_STATUS_LOCK.acquire(blocking=False):
        return {"status": "unknown", "detail": "电镜状态探测进行中"}

    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def probe() -> None:
        try:
            result_queue.put((True, _microscope_status(host, port)))
        except Exception as error:
            result_queue.put((False, error))
        finally:
            MICROSCOPE_STATUS_LOCK.release()

    threading.Thread(target=probe, name="microscope-status", daemon=True).start()
    try:
        succeeded, value = result_queue.get(timeout=float(timeout_seconds))
    except queue.Empty:
        return {
            "status": "offline",
            "detail": f"电镜状态探测超时（{float(timeout_seconds):.1f}s）",
        }
    if not succeeded:
        raise value
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--check-runtime", action="store_true", help="Check packaged driver dependencies without hardware access")
    parser.add_argument("--mcp-stdio", action="store_true", help="Run the MCP bridge without opening a device connection")
    parser.add_argument("--mcp-simulated", action="store_true", default=os.environ.get("ZEM_MCP_SIMULATED") == "1",
                        help="Use simulated MCP devices; never contact hardware")
    parser.add_argument("--mcp-state-dir", type=Path, help="Directory for durable MCP operations and images")
    args = parser.parse_args()
    if args.mcp_stdio:
        from backend.sem_mcp import main as mcp_main
        mcp_main()
        return
    if args.check_runtime:
        from backend.cleanliness_core.runtime_check import check_runtime
        print(json.dumps(check_runtime()), flush=True)
        return
    from backend.sem_control import RealMicroscope, SemControl, SimulatedMicroscope
    from backend.sem_rpc_full import WRITE_COMMANDS as RPC_WRITES
    root = args.mcp_state_dir or TASK_ROOT.parent / "mcp"
    options = app_settings()
    simulated = args.mcp_simulated
    config = load_mcp_device_config()
    device = SimulatedMicroscope() if simulated else RealMicroscope(options["microscopeHost"], options["microscopePort"])
    owner_path = root / "simulated-backend.lock" if simulated else None
    with BackendOwnerLock(owner_path):
        global BACKEND_PORT, MCP_CONTROL, MCP_TOKEN
        JOBS.control = DeviceCoordinator(root / "control-state.json")
        MCP_CONTROL = SemControl(root, device, JOBS.control,
            allow_capture=True if simulated else config.get("allowCapture") is True,
            bounds=[-1000., -1000., 1000., 1000.] if simulated else config.get("boundsUm"),
            max_move_um=1000. if simulated else config.get("maxMoveUm"),
            allow_rpc=tuple(RPC_WRITES) if simulated else config.get("allowRpcCommands", ()),
            rpc_limits=config.get("rpcLimits", {}), rpc_timeout=config.get("rpcTimeoutSeconds", 120.),
            rpc_path_roots=config.get('rpcPathRoots', ()))
        JOBS.allow_hardware_jobs = not simulated
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        BACKEND_PORT = server.server_address[1]
        MCP_TOKEN = (os.environ.get("ZEM_MCP_TOKEN") if simulated else None) or secrets.token_urlsafe(32)
        os.environ["ZEM_MCP_BACKEND_URL"] = f"http://127.0.0.1:{BACKEND_PORT}"
        os.environ["ZEM_MCP_TOKEN"] = MCP_TOKEN
        log_event(
            "info", "backend_start", host=args.host, port=BACKEND_PORT, taskRoot=str(TASK_ROOT),
            resourceRoot=str(RESOURCE_ROOT), logDirectory=str(LOG_ROOT), fakeEds=FAKE_EDS_ENABLED,
            mcpMode="simulated" if simulated else "real",
            captureEnabled=MCP_CONTROL.allow_capture, moveEnabled=MCP_CONTROL.bounds is not None,
            mcpDeviceConfig=str(mcp_device_config_path()),
        )
        print(json.dumps({"service": "zem-cleanliness", "host": args.host, "port": BACKEND_PORT}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            log_event("info", "backend_stop", host=args.host, port=BACKEND_PORT)
            server.server_close()
            MCP_CONTROL.close(wait=False)


if __name__ == "__main__":
    main()
