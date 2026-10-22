"""Bounded, persisted AB navigation + particle refinement job.

Device access occurs only inside run_ab_particle_calibration after explicit start.
Tests inject a microscope; no RPC import is needed by the image/math primitives.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .ab_point_calibration import build_ab_preview_transform
from .atomic_json import write_json_atomic
from .cleanliness_detection import load_cleanliness_config, transform_solution_from_dict
from .optical_particle_analysis import decode_optical_image, load_calibration, load_optical_export, normalize_uint8
from .optical_particle_task import ZemMicroscope, _capture_signature, _validate_same_signature
from .particle_refinement import (
    RefinementSettings, check_linear_change, coverage, match_particle,
    measure_pixel_axes, metric_ab_initial, robust_affine, select_context_seeds, select_plan, validate_frozen, verify_particle_context, xy,
)


def _elapsed(started: float) -> float:
    return round(max(0.0, time.monotonic() - started), 3)


def _emit_event(callback, event: str, **fields):
    if callback is None:
        return
    try:
        callback(event, fields)
    except Exception:
        # Timing evidence must never interrupt microscope recovery or calibration.
        return


def settings_fingerprint(cfg: RefinementSettings) -> str:
    return hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()


def _now():
    return datetime.now().astimezone().isoformat()


def validate_field_metadata(capture, expected_field=None):
    """Reject stale cached pixel scales after SetMag; never repair metadata by assumption."""
    pixel = float(capture.get("pixelSizeUm") or 0)
    shape = capture.get("shape") or []
    if len(shape) < 2 or not np.isfinite(pixel) or pixel <= 0:
        raise RuntimeError("图像像素尺寸无效")
    width = pixel * int(shape[1])
    field = float((capture.get("meta") or {}).get("FieldSize") or width)
    fields = [field]
    if expected_field is not None:
        fields.append(float(expected_field))
    bounds = capture.get("bounds")
    if bounds is not None:
        if len(bounds) != 4 or not np.isfinite(bounds).all():
            raise RuntimeError("图像边界元数据无效")
        fields.extend([abs(float(bounds[2]) - float(bounds[0])),
                       abs(float(bounds[3]) - float(bounds[1])) * int(shape[1]) / int(shape[0])])
    if any(not np.isfinite(v) or abs(v - width) > max(.5, width * .002) for v in fields):
        raise RuntimeError(f"倍率切换后视场/像素尺寸/边界不一致：像素推算 {width:.3f} µm，其他回读 {fields}")


def move_verified(microscope, target, cfg, *, check=lambda: None):
    """Bounded commands + delayed physical readback; never trust a nominal encoder target."""
    target = np.asarray(target, float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("移台目标不是有限二维坐标")
    attempts = []
    for attempt in range(cfg.stage_move_attempts):
        check()
        move = microscope.move_xy(*target)
        measured = np.asarray(microscope.get_position(), float)
        if not np.isfinite(measured).all():
            raise RuntimeError("样品台回读不是有限坐标")
        error = float(np.linalg.norm(measured - target))
        record = {"attempt": attempt + 1, "move": move, "errorUm": error}
        attempts.append(record)
        if error <= cfg.stage_error_um:
            deadline = time.monotonic() + cfg.stage_settle_seconds
            while time.monotonic() < deadline:
                check()
                time.sleep(min(.1, max(0, deadline - time.monotonic())))
            measured = np.asarray(microscope.get_position(), float)
            error = float(np.linalg.norm(measured - target))
            record.update(recheckUm=measured.tolist(), recheckErrorUm=error)
            if np.isfinite(measured).all() and error <= cfg.stage_error_um:
                return {"target": {"x": float(target[0]), "y": float(target[1])},
                        "after": {"x": float(measured[0]), "y": float(measured[1])},
                        "positionErrorUm": error, "attempts": attempts}
    raise RuntimeError(f"样品台重复移台 {cfg.stage_move_attempts} 次仍未收敛到 {cfg.stage_error_um} µm：{attempts}")


class Acquisition:
    def __init__(self, microscope, task_dir, audit, cfg, roi, progress, should_stop, persist, event_logger=None):
        self.microscope, self.task_dir, self.audit = microscope, task_dir, audit
        self.cfg, self.roi = cfg, roi
        self.progress, self.should_stop, self.persist = progress, should_stop, persist
        self.event_logger = event_logger
        self.started = time.monotonic()
        self.signatures = {}
        self.phase = "coarse"

    def check(self):
        if self.should_stop():
            raise InterruptedError("用户停止颗粒精校准")
        if time.monotonic() - self.started > self.cfg.max_seconds:
            raise TimeoutError("颗粒精校准已达到总时间上限")

    def configure(self, phase, magnification):
        self.check()
        started = time.monotonic()
        _emit_event(
            self.event_logger,
            "calibration_stage_started",
            calibrationMode="ab_particle",
            stage="imaging_configure",
            phase=phase,
            magnification=magnification,
            attempt=None,
        )
        self.phase = phase
        self.microscope.imaging = {"magnification": magnification, "imageShape": [self.cfg.image_size] * 2}
        # Device imaging profile supplies a reference FOV, not a hardcoded Zem20F constant.
        # Passing an expected field activates ZemMicroscope's post-SetMag convergence check.
        reference_mag = float(self.roi.imaging.get("magnification") or 0)
        reference_field = float(self.roi.imaging.get("fieldSizeUm") or 0)
        if reference_mag > 0 and reference_field > 0:
            self.microscope.imaging["fieldSizeUm"] = reference_field * reference_mag / magnification
        self.microscope.pixel_avg = self.cfg.pixel_avg
        try:
            self.audit.setdefault("imagingConnections", {})[phase] = self.microscope.connect()
        except Exception as error:
            _emit_event(
                self.event_logger,
                "calibration_stage_finished",
                calibrationMode="ab_particle",
                stage="imaging_configure",
                phase=phase,
                magnification=magnification,
                attempt=None,
                status="failed",
                elapsedSeconds=_elapsed(started),
                error=str(error),
            )
            raise
        _emit_event(
            self.event_logger,
            "calibration_stage_finished",
            calibrationMode="ab_particle",
            stage="imaging_configure",
            phase=phase,
            magnification=magnification,
            attempt=None,
            status="succeeded",
            elapsedSeconds=_elapsed(started),
        )
        self.persist()

    def inside(self, target):
        approach = np.asarray(target) + self.cfg.approach_offset_um
        return (np.isfinite([target, approach]).all() and self.roi.target_inside_roi(tuple(target))
                and self.roi.target_inside_roi(tuple(approach)))

    def move(self, target):
        self.check()
        if not self.inside(target):
            raise ValueError("搜索/进近目标超出设备配置的安全 ROI；未执行移台")
        self.audit["stageMoved"] = True
        self.persist()
        approach = np.asarray(target) + self.cfg.approach_offset_um
        if np.linalg.norm(self.cfg.approach_offset_um) > 0:
            move_verified(self.microscope, approach, self.cfg, check=self.check)
        self.check()
        return move_verified(self.microscope, target, self.cfg, check=self.check)

    def capture(self, target, label):
        self.check()
        shots = self.audit["captures"]
        if len(shots) >= self.cfg.max_captures:
            raise TimeoutError("颗粒精校准已达到总拍摄张数上限")
        index = len(shots) + 1
        self.progress(index - 1, self.cfg.max_captures, f"{label} · 第 {index} 张（最多 {self.cfg.max_captures} 张）")
        shot = {"index": index, "label": label, "phase": self.phase, "targetUm": list(target), "status": "MOVING"}
        shots.append(shot)
        self.persist()
        move_started = time.monotonic()
        move_finished = False
        capture_started = None
        capture_finished = False
        common = {
            "calibrationMode": "ab_particle",
            "phase": self.phase,
            "captureIndex": index,
            "label": label,
            "targetStageUm": list(target),
        }
        _emit_event(
            self.event_logger,
            "calibration_stage_started",
            stage="stage_move",
            attempt=1,
            **common,
        )
        try:
            shot["move"] = self.move(target)
            move_finished = True
            _emit_event(
                self.event_logger,
                "calibration_stage_finished",
                stage="stage_move",
                attempt=1,
                status="succeeded",
                elapsedSeconds=_elapsed(move_started),
                actualStageUm=shot["move"].get("after"),
                positionErrorUm=shot["move"].get("positionErrorUm"),
                **common,
            )
            before = np.asarray(self.microscope.get_position(), float)
            self.check()
            raw_path = self.task_dir / self.audit["runDirectory"] / f"{index:04d}.png"
            capture_started = time.monotonic()
            _emit_event(
                self.event_logger,
                "calibration_stage_started",
                stage="capture",
                attempt=1,
                **common,
            )
            image, metadata = self.microscope.capture(raw_path)
            after = np.asarray(self.microscope.get_position(), float)
            shot.update(capture=metadata, captureStageUm=after.tolist(), status="CAPTURED")
            if not np.isfinite([before, after]).all() or np.linalg.norm(after - before) > self.cfg.stage_error_um:
                raise RuntimeError("拍摄期间样品台漂移超限")
            if np.linalg.norm(after - target) > self.cfg.stage_error_um:
                raise RuntimeError("拍摄后台坐标偏离目标，拒绝用名义位置计算")
            signature = _capture_signature(metadata)
            if signature["shape"][:2] != [self.cfg.image_size] * 2 or signature["pixelSizeUm"] <= 0:
                raise RuntimeError("采图尺寸/像素尺寸无效")
            validate_field_metadata(metadata, self.microscope.imaging.get("fieldSizeUm"))
            expected_mag = self.cfg.coarse_magnification if self.phase == "coarse" else self.cfg.fine_magnification
            if signature.get("magnification") is None or abs(float(signature["magnification"]) - expected_mag) > .1:
                raise RuntimeError("采图倍率与当前精校准阶段不一致")
            if self.phase in self.signatures:
                _validate_same_signature(self.signatures[self.phase], signature)
            else:
                self.signatures[self.phase] = signature
            self.check()
            capture_finished = True
            _emit_event(
                self.event_logger,
                "calibration_stage_finished",
                stage="capture",
                attempt=1,
                status="succeeded",
                elapsedSeconds=_elapsed(capture_started),
                imageShape=list(image.shape),
                pixelSizeUm=metadata.get("pixelSizeUm"),
                **common,
            )
            return image, metadata, after
        except Exception as exc:
            status = "stopped" if isinstance(exc, InterruptedError) else "failed"
            if not move_finished:
                _emit_event(
                    self.event_logger,
                    "calibration_stage_finished",
                    stage="stage_move",
                    attempt=1,
                    status=status,
                    elapsedSeconds=_elapsed(move_started),
                    error=str(exc),
                    **common,
                )
            if capture_started is not None and not capture_finished:
                _emit_event(
                    self.event_logger,
                    "calibration_stage_finished",
                    stage="capture",
                    attempt=1,
                    status=status,
                    elapsedSeconds=_elapsed(capture_started),
                    error=str(exc),
                    **common,
                )
            shot.update(status="FAILED", error=str(exc))
            raise
        finally:
            self.persist()

    def axes(self, center):
        # Stay below a quarter frame for reliable phase correlation, including coarse magnification.
        images, positions = {}, {}
        images["o1"], metadata, positions["o1"] = self.capture(center, f"{self.phase} 像素轴原点")
        pixel = float(metadata["pixelSizeUm"])
        step = min(self.cfg.axis_step_um, pixel * self.cfg.image_size * .15)
        for name, offset in (("xp", (step, 0)), ("xm", (-step, 0)), ("yp", (0, step)), ("ym", (0, -step)), ("o2", (0, 0))):
            images[name], _, positions[name] = self.capture(np.asarray(center) + offset, f"{self.phase} 像素轴 {name}")
        try:
            axis = measure_pixel_axes(images, positions, pixel, self.cfg)
        except ValueError as exc:
            self.audit.setdefault("axisFailureDetails", []).append({
                "phase": self.phase, "lastCaptureIndex": len(self.audit["captures"]),
                "error": str(exc), "retryable": getattr(exc, "retryable", True),
                "diagnostics": getattr(exc, "diagnostics", {})})
            self.persist()
            raise
        axis.update(signature=self.signatures[self.phase], positionsUm={k: list(v) for k, v in positions.items()})
        self.audit.setdefault("pixelAxes", {})[self.phase] = axis
        self.persist()
        return axis

    def locate(self, particle, target, axis, *, search=False, prior=None, role="anchor", context=None):
        cfg = self.cfg
        target = np.asarray(target, float)
        offsets = [(0., 0.)]
        if search:
            step = cfg.search_step_ratio * axis["pixelSizeUm"] * cfg.image_size
            rings = int(np.ceil(cfg.search_radius_um / step))
            offsets += sorted(((x * step, y * step) for x in range(-rings, rings + 1)
                               for y in range(-rings, rings + 1) if (x or y) and np.hypot(x * step, y * step) <= cfg.search_radius_um),
                              key=lambda o: (np.hypot(*o), o[1], o[0]))
            offsets = offsets[:cfg.max_search_captures]
        record = {"sequence": particle.sequence, "role": role, "lengthUm": particle.length_um,
                  "opticalAbUm": [particle.ab_x_um, particle.ab_y_um],
                  "targetStageUm": {"x": float(target[0]), "y": float(target[1])}, "attempts": []}
        self.audit["observations"].append(record)
        decode_optical_image(particle, self.task_dir / "optical" / f"{particle.sequence:04d}.png")
        for offset in offsets:
            point = target + offset
            if not self.inside(point):
                record["attempts"].append({"status": "OUTSIDE_ROI", "targetUm": point.tolist()})
                continue
            image, metadata, stage = self.capture(point, f"{role} #{particle.sequence}")
            match_started = time.monotonic()
            match_common = {
                "calibrationMode": "ab_particle",
                "stage": "particle_match",
                "phase": self.phase,
                "captureIndex": len(self.audit["captures"]),
                "particleSequence": particle.sequence,
                "role": role,
                "attempt": len(record["attempts"]) + 1,
            }
            _emit_event(self.event_logger, "calibration_stage_started", **match_common)
            try:
                match = match_particle(image, particle, float(metadata["pixelSizeUm"]), cfg, prior=prior,
                                       should_stop=lambda: self.check() or False)
                if context is not None and match["status"] == "MATCH":
                    verification = verify_particle_context(
                        image, particle, match, context["particles"], context["matrix"],
                        np.asarray(axis["pixelToStageMatrix"]), float(metadata["pixelSizeUm"]), cfg,
                        excluded_sequences=context["excludedSequences"], should_stop=lambda: self.check() or False)
                    match = {**match, "contextVerification": verification}
                    if verification["status"] != "VERIFIED":
                        match.update(status="CONTEXT_UNVERIFIED", reason=verification["reason"])
            except Exception as error:
                _emit_event(
                    self.event_logger,
                    "calibration_stage_finished",
                    status="stopped" if isinstance(error, InterruptedError) else "failed",
                    elapsedSeconds=_elapsed(match_started),
                    error=str(error),
                    **match_common,
                )
                raise
            _emit_event(
                self.event_logger,
                "calibration_stage_finished",
                status=str(match.get("status") or "unknown").lower(),
                elapsedSeconds=_elapsed(match_started),
                score=match.get("score"),
                margin=match.get("margin"),
                **match_common,
            )
            record["attempts"].append({"captureIndex": len(self.audit["captures"]), **match})
            # The summary belongs to THIS frame. Rejected-neighbor evidence and
            # coordinates remain in attempts, not attached to a later blank or
            # low-score frame that did not produce those values.
            for key in tuple(record):
                if key not in ("sequence", "role", "lengthUm", "opticalAbUm", "targetStageUm", "attempts"):
                    del record[key]
            record.update(match)
            record["captureStageUm"] = stage.tolist()
            record["moveErrorUm"] = float(np.linalg.norm(stage - point))
            display = cv2.cvtColor(normalize_uint8(image), cv2.COLOR_GRAY2BGR)
            h, w = image.shape[:2]
            cv2.drawMarker(display, (w // 2, h // 2), (0, 255, 0), cv2.MARKER_CROSS, 30, 1)
            if "centerPixel" in match:
                center = np.asarray(match["centerPixel"])
                cv2.circle(display, tuple(np.rint(center).astype(int)), 24, (0, 0, 255), 2)
                matrix = np.asarray(axis["pixelToStageMatrix"])
                measured = stage + matrix @ (center - [w / 2, h / 2])
                record["measuredStageUm"] = measured.tolist()
                record["predictionErrorUm"] = float(np.linalg.norm(measured - target))
            path = self.task_dir / self.audit["runDirectory"] / f"{len(self.audit['captures']):04d}_match.png"
            if not cv2.imwrite(str(path), display):
                raise OSError(f"不能保存标定证据图：{path}")
            record["images"] = {"opticalImage": f"optical/{particle.sequence:04d}.png",
                                "semRawImage": Path(metadata["rawPath"]).relative_to(self.task_dir).as_posix(),
                                "semDisplayImage": Path(metadata["displayPath"]).relative_to(self.task_dir).as_posix(),
                                "semAnnotatedImage": path.relative_to(self.task_dir).as_posix()}
            self.persist()
            if match["status"] == "MATCH":
                return record
            # An ambiguous primary field is not made unique by moving its
            # runner-up out of view. A UNIQUE primary rejected by independent
            # neighbor evidence, however, must not abort the entire grid: keep
            # the rejection and continue PRECOMPUTED offsets, never recenter on
            # the rejected candidate or update the navigation from it.
            if match["status"] == "AMBIGUOUS":
                break
        record.setdefault("status", "NO_MATCH")
        self.persist()
        return record


def run_ab_particle_calibration(
    task_dir: Path,
    *,
    config=None,
    microscope=None,
    progress=None,
    should_stop=None,
    event_logger=None,
):
    from .cleanliness_task import default_cleanliness_config_path, _resolve_experience_path
    calibration_started = time.monotonic()
    _emit_event(
        event_logger,
        "calibration_stage_started",
        calibrationMode="ab_particle",
        stage="calibration",
        attempt=None,
    )
    task_dir = Path(task_dir).resolve()
    manifest_path = task_dir / "task.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration = manifest["calibration"]
    if calibration.get("strategy") != "ab_particle" or manifest.get("status") != "AB_POINTS_PENDING_SEM_REFINEMENT":
        raise ValueError("当前任务不是待执行的 AB 导航 + 颗粒精校准任务")
    settings_path = Path(config.settings_path if config else default_cleanliness_config_path()).resolve()
    preflight_started = time.monotonic()
    _emit_event(
        event_logger,
        "calibration_stage_started",
        calibrationMode="ab_particle",
        stage="preflight",
        attempt=None,
    )
    try:
        settings = load_cleanliness_config(settings_path)
        cfg = RefinementSettings.from_dict(settings.get("abParticle"))
        export = load_optical_export(Path(manifest["input"]["path"]))
        if export.sha256 != manifest["input"]["sha256"]:
            raise ValueError("光镜 JSON 内容已变化，请重新创建标定任务")
        # Calibration uses the full optical specimen, independent of the analysis filter.
        plan = select_plan(export.particles, cfg)
        experience_path = Path(config.experience_calibration_path) if config and config.experience_calibration_path else _resolve_experience_path(settings_path, settings)
        roi = load_calibration(experience_path)  # Device safe ROI ONLY, never its old coordinate matrix.
        if not np.isfinite(roi.roi_center).all() or not np.isfinite(roi.roi_radius_um):
            raise ValueError("设备安全 ROI 包含非有限数值")
        device = manifest["device"]["microscope"]
        if ((roi.device.get("host") and roi.device["host"] != device["host"])
                or (roi.device.get("port") and int(roi.device["port"]) != int(device["port"]))):
            raise ValueError("安全 ROI 配置不属于当前电镜，请先配置该设备")
        if config and ((config.microscope_host and config.microscope_host != device["host"])
                       or (config.microscope_port and config.microscope_port != device["port"])):
            raise ValueError("设备与导航图绑定的设备不一致，请重新获取导航图")
        initial = transform_solution_from_dict(calibration["abPointResult"]["previewTransform"])
        if not np.isfinite(initial.matrix).all() or not np.isfinite(initial.translation).all() or abs(np.linalg.det(initial.matrix)) < 1e-9:
            raise ValueError("AB 初始矩阵无效，请重新保存导航图指派")
        source = calibration["abPointResult"]["sourceReferencePointsUm"]
        initial, navigation_initial = metric_ab_initial(initial, source["a"], source["b"])
        _emit_event(
            event_logger,
            "calibration_stage_finished",
            calibrationMode="ab_particle",
            stage="preflight",
            attempt=None,
            status="succeeded",
            elapsedSeconds=_elapsed(preflight_started),
        )
    except Exception as exc:
        ab = calibration["abPointResult"]
        if ab.get("semRefinement"):
            ab.setdefault("semRefinementHistory", []).append(copy.deepcopy(ab["semRefinement"]))
        ab["semRefinement"] = {"status": "FAILED", "phase": "preflight", "mode": "ab_particle",
                               "error": str(exc), "completedAt": _now(), "deviceActionsPerformed": False}
        manifest["error"] = str(exc)
        calibration.update(result=None, refinedTransform=None, review=None)
        write_json_atomic(manifest_path, manifest)
        write_json_atomic(task_dir / "ab_point_result.json", ab)
        _emit_event(
            event_logger,
            "calibration_stage_finished",
            calibrationMode="ab_particle",
            stage="preflight",
            attempt=None,
            status="failed",
            elapsedSeconds=_elapsed(preflight_started),
            error=str(exc),
        )
        _emit_event(
            event_logger,
            "calibration_stage_finished",
            calibrationMode="ab_particle",
            stage="calibration",
            attempt=None,
            status="failed",
            elapsedSeconds=_elapsed(calibration_started),
            error=str(exc),
        )
        raise
    progress, should_stop = progress or (lambda *_: None), should_stop or (lambda: False)
    if should_stop():
        _emit_event(
            event_logger,
            "calibration_stage_finished",
            calibrationMode="ab_particle",
            stage="calibration",
            attempt=None,
            status="stopped",
            elapsedSeconds=_elapsed(calibration_started),
            error="用户已停止；未连接电镜",
        )
        raise InterruptedError("用户已停止；未连接电镜")
    ab = calibration["abPointResult"]
    if ab.get("semRefinement"):
        ab.setdefault("semRefinementHistory", []).append(copy.deepcopy(ab["semRefinement"]))
    audit = {"status": "RUNNING", "mode": "ab_particle", "algorithmVersion": 5,
             "navigationInitial": navigation_initial,
             "startedAt": _now(), "settings": asdict(cfg), "settingsSha256": settings_fingerprint(cfg),
             "plan": plan, "captures": [], "observations": [], "anchors": [], "validation": [],
             "runDirectory": f"ab_particle/{uuid.uuid4().hex[:12]}"}
    ab["semRefinement"] = audit
    calibration.update(result=None, refinedTransform=None, review=None)
    manifest["error"] = None

    def persist():
        write_json_atomic(manifest_path, manifest)
        write_json_atomic(task_dir / "ab_point_result.json", ab)

    persist()
    # Initial connect does not request a magnification change; capture previous settings first.
    microscope = microscope or ZemMicroscope(device["host"], device["port"], {}, pixel_avg=cfg.pixel_avg)
    acquisition = Acquisition(
        microscope,
        task_dir,
        audit,
        cfg,
        roi,
        progress,
        should_stop,
        persist,
        event_logger=event_logger,
    )
    start, connection = None, None
    task_settings = settings.get("task") or {}
    restore = config.restore_start_position if config and config.restore_start_position is not None else task_settings.get("restoreStartPosition", True)
    gun_on = config.auto_gun_on if config and config.auto_gun_on is not None else task_settings.get("gunOnAtStart", False)
    gun_off = config.gun_off_after_task if config and config.gun_off_after_task is not None else task_settings.get("gunOffAfterTask", False)
    connected = False
    try:
        acquisition.check()
        connection_started = time.monotonic()
        _emit_event(
            event_logger,
            "calibration_stage_started",
            calibrationMode="ab_particle",
            stage="connect",
            attempt=1,
        )
        try:
            connection = microscope.connect()
        except Exception as exc:
            _emit_event(
                event_logger,
                "calibration_stage_finished",
                calibrationMode="ab_particle",
                stage="connect",
                attempt=1,
                status="failed",
                elapsedSeconds=_elapsed(connection_started),
                error=str(exc),
            )
            raise
        _emit_event(
            event_logger,
            "calibration_stage_finished",
            calibrationMode="ab_particle",
            stage="connect",
            attempt=1,
            status="succeeded",
            elapsedSeconds=_elapsed(connection_started),
        )
        connected = True
        device["connected"] = True
        start = tuple(microscope.get_position())
        reference_mag = float(roi.imaging.get("magnification") or 0)
        reference_field = float(roi.imaging.get("fieldSizeUm") or 0)
        original_mag = float(connection.get("magnification") or 0)
        # Save a validated intended restoration profile even if synchronization fails.
        if reference_mag > 0 and reference_field > 0 and original_mag > 0:
            connection["fieldSizeUm"] = reference_field * reference_mag / original_mag
            microscope.imaging = {"magnification": original_mag, "imageShape": [cfg.image_size] * 2,
                                  "fieldSizeUm": reference_field * reference_mag / original_mag}
            connection = microscope.connect()
        audit["connection"] = connection
        audit["startPositionUm"] = list(start)
        if gun_on:
            acquisition.check()
            audit["gunOnStatus"] = str(microscope.gun_on())
        if hasattr(microscope, "get_hv_status"):
            from .microscope_safety import hv_status_name
            audit["hvStatus"] = str(microscope.get_hv_status())
            if hv_status_name(audit["hvStatus"]) != "open":
                raise RuntimeError("电子枪未开启；请由操作员确认设备状态后重试")
        acquisition.configure("coarse", cfg.coarse_magnification)
        by_seq = export.by_sequence
        seed_candidates = [by_seq[s] for s in plan["anchors"]]
        axis = None
        for seed in seed_candidates[:3]:
            center = initial.transform_xy(seed.ab_x_um, seed.ab_y_um)
            if not acquisition.inside(center):
                continue
            try:
                axis = acquisition.axes(center)
                break
            except ValueError as exc:
                audit.setdefault("axisFailures", []).append(str(exc))
                persist()
                if not getattr(exc, "retryable", True):
                    raise
        if axis is None:
            raise ValueError("无法可靠测量像素轴；请检查视场纹理、样品范围或成像参数")

        # Two AB chiralities. A hypothesis is only usable after >=4 geometrically consistent seeds.
        source = ab["sourceReferencePointsUm"]
        assignment = ab["assignment"]
        opposite = build_ab_preview_transform(source["a"], source["b"], assignment["a"]["coarseStageUm"],
                                              assignment["b"]["coarseStageUm"],
                                              handedness="left" if np.linalg.det(initial.matrix) < 0 else "right")
        bootstrap = None
        opposite_initial, _ = metric_ab_initial(transform_solution_from_dict(opposite), source["a"], source["b"])
        for hypothesis_index, hypothesis in enumerate((initial, opposite_initial)):
            matches = []
            current = hypothesis
            visible_candidates = [p for p in seed_candidates if acquisition.inside(hypothesis.transform_xy(p.ab_x_um, p.ab_y_um))]
            planned_seeds, seed_plan = select_context_seeds(
                visible_candidates, export.particles, hypothesis.matrix, np.asarray(axis['pixelToStageMatrix']),
                axis['pixelSizeUm'], cfg, excluded_sequences=plan['validation'], should_stop=lambda: acquisition.check() or False)
            seed_plan['hypothesisIndex'] = hypothesis_index
            audit.setdefault('seedPlans', []).append(seed_plan)
            persist()
            for seed in planned_seeds:
                record = acquisition.locate(seed, current.transform_xy(seed.ab_x_um, seed.ab_y_um), axis,
                                            search=True, role=f"seed_h{hypothesis_index}",
                                            context={"particles": export.particles, "matrix": hypothesis.matrix,
                                                     "excludedSequences": plan["validation"]})
                if record["status"] != "MATCH":
                    continue
                matches.append(record)
                # Translation-only correction needs two consistent offsets. No free affine from 3 points.
                if len(matches) >= 2:
                    offsets = np.array([np.asarray(r["measuredStageUm"]) - hypothesis.transform_xy(*r["opticalAbUm"]) for r in matches])
                    delta = np.median(offsets, axis=0)
                    if np.max(np.linalg.norm(offsets - delta, axis=1)) <= cfg.inlier_threshold_um * 2 and np.linalg.norm(delta) <= cfg.search_radius_um:
                        current = replace(hypothesis, translation=hypothesis.translation + delta)
                if len(matches) >= 4:
                    try:
                        fit, robust = robust_affine(np.array([r["opticalAbUm"] for r in matches]),
                                                    np.array([r["measuredStageUm"] for r in matches]),
                                                    delta=cfg.bootstrap_huber_delta_um, threshold=cfg.bootstrap_inlier_um, min_inliers=4)
                        check_linear_change(fit, hypothesis, cfg.max_linear_change)
                        seed_coverage = coverage(np.array([r["opticalAbUm"] for r, ok in zip(matches, robust["inliers"]) if ok]), xy(export.particles))
                        if seed_coverage < cfg.min_coverage:
                            audit.setdefault("bootstrapRejections", []).append(f"种子内点覆盖 {seed_coverage:.1%} < {cfg.min_coverage:.1%}")
                            persist()
                            continue
                        bootstrap = fit, hypothesis, matches
                        break
                    except ValueError as exc:
                        audit.setdefault("bootstrapRejections", []).append(str(exc))
            if bootstrap is not None:
                break
        if bootstrap is None:
            raise ValueError("两种 AB 手性均未取得足够且分散的一致种子；未生成可用矩阵")
        current, initial, seeds = bootstrap
        audit["bootstrap"] = {"transform": current.to_dict(), "seedSequences": [r["sequence"] for r in seeds],
                              "robustFit": robust, "coverage": seed_coverage,
                              "handedness": "right" if np.linalg.det(initial.matrix) < 0 else "left"}
        acquisition.configure("fine", cfg.fine_magnification)
        fine_axis = None
        # A small feature may be excellent for correspondence but a poor phase-
        # correlation field (periodic substrate). Prefer larger consensus features,
        # then try at most three distinct regions, without lowering quality gates.
        axis_sites = sorted((r for r, ok in zip(seeds, robust["inliers"]) if ok),
                            key=lambda r: (-by_seq[r["sequence"]].area_um2, r["sequence"]))
        for site in axis_sites[:3]:
            try:
                fine_axis = acquisition.axes(site["measuredStageUm"])
                break
            except ValueError as exc:
                audit.setdefault("axisFailures", []).append(f"fine #{site['sequence']}: {exc}")
                persist()
                if not getattr(exc, "retryable", True):
                    raise
        if fine_axis is None:
            raise ValueError("多个区域都未通过高倍率像素轴质量检查，拒绝进行精校准")
        anchors = audit["anchors"]
        final, robust = None, None
        for sequence in plan["anchors"]:
            particle = by_seq[sequence]
            record = acquisition.locate(particle, current.transform_xy(particle.ab_x_um, particle.ab_y_um), fine_axis, role="anchor")
            if record["status"] != "MATCH":
                continue
            # Fresh centered capture verifies identity and image-to-stage correction before fitting.
            confirmed = acquisition.locate(particle, record["measuredStageUm"], fine_axis, prior=record, role="anchor_confirm")
            consistency = (float(np.linalg.norm(np.subtract(confirmed["measuredStageUm"], record["measuredStageUm"])))
                           if confirmed.get("measuredStageUm") else float("inf"))
            if confirmed["status"] != "MATCH" or consistency > max(15, cfg.stage_error_um * 3):
                record["rejection"] = "居中复拍未确认同一颗粒"
                persist()
                continue
            confirmed["reacquireConsistencyUm"] = consistency
            anchors.append(confirmed)
            if len(anchors) >= 6:
                try:
                    candidate, robust = robust_affine(np.array([r["opticalAbUm"] for r in anchors]),
                                                      np.array([r["measuredStageUm"] for r in anchors]),
                                                      delta=cfg.huber_delta_um, threshold=cfg.inlier_threshold_um,
                                                      min_inliers=min(cfg.min_inliers, len(anchors)))
                    check_linear_change(candidate, initial, cfg.max_linear_change)
                    current = candidate
                    if (len(anchors) >= cfg.anchor_count and robust["inlierCount"] >= cfg.min_inliers
                            and robust["inlierCount"] / len(anchors) >= cfg.min_inlier_fraction
                            and coverage(np.array([r["opticalAbUm"] for r, ok in zip(anchors, robust["inliers"]) if ok]), xy(export.particles)) >= cfg.min_coverage):
                        final = candidate
                        break
                except ValueError as exc:
                    audit.setdefault("fitRejections", []).append(str(exc))
            persist()
        if final is None:
            raise ValueError("可靠精校准锚点数量、几何一致性或空间覆盖不足；拒绝进入验收")
        for record, weight, error, inlier in zip(anchors, robust["weights"], robust["residualsUm"], robust["inliers"]):
            record.update(robustWeight=weight, fitResidualUm=error, inlier=inlier)
        audit["robustFit"] = robust
        audit["frozenAt"] = _now()
        audit["frozenTransform"] = final.to_dict()
        write_json_atomic(task_dir / audit["runDirectory"] / "frozen_transform.json", final.to_dict())
        persist()
        # No searching, recentering, replacements or refits on independent validation.
        for sequence in plan["validation"]:
            particle = by_seq[sequence]
            audit["validation"].append(acquisition.locate(particle, final.transform_xy(particle.ab_x_um, particle.ab_y_um),
                                                          fine_axis, role="validation"))
            persist()
        acceptance = validate_frozen(final, anchors, audit["validation"], plan["validation"], xy(export.particles), cfg)
        result = {**final.to_dict(), "validation": acceptance, "acceptance": acceptance}
        audit.update(status="SUCCEEDED" if acceptance["withinLimits"] else "VALIDATION_FAILED",
                     acceptance=acceptance, refinedTransform=result)
        calibration.update(result=result, refinedTransform=result)
        signature = acquisition.signatures["fine"]
        # This matrix already contains actual field rotation and all axis coupling.
        calibration["pixelAxisMatrix"] = (np.asarray(fine_axis["pixelToStageMatrix"]) / fine_axis["pixelSizeUm"]).tolist()
        calibration["pixelAxisIncludesRotation"] = True
        manifest["baselineImaging"] = signature
        payload = {"schemaVersion": 1, "calibrationId": f"ab-particle-{manifest['taskId']}",
                   "device": {"host": device["host"], "port": device["port"]},
                   "source": {"coordinateUnit": export.coordinate_unit, "coordinateSpace": export.coordinate_space,
                              "projectId": export.project_id, "workpieceId": export.workpiece_id,
                              "inputSha256": export.sha256, "requireExactInput": True},
                   "transform": result, "validStageRoi": {"centerUm": roi.roi_center.tolist(), "radiusUm": roi.roi_radius_um},
                   "imaging": {"magnification": cfg.fine_magnification, "imageShape": signature["shape"],
                               "pixelSizeUm": signature["pixelSizeUm"], "fieldSizeUm": signature["fieldSizeUm"],
                               "detector": signature["detector"], "fieldRotationDeg": signature["fieldRotationDeg"]},
                   "audit": {"algorithmVersion": 3, "frozenAt": audit["frozenAt"], "acceptance": acceptance,
                             "settingsSha256": audit["settingsSha256"], "pixelAxisMatrix": calibration["pixelAxisMatrix"]}}
        calibration_path = task_dir / audit["runDirectory"] / "calibration.json"
        write_json_atomic(calibration_path, payload)
        # Existing analysis API consumes this per-task calibration instead of the experience matrix.
        manifest["experienceCalibration"] = load_calibration(calibration_path).to_audit_dict()
        manifest["device"]["eds"] = {"host": (config.eds_host if config else None) or (settings.get("device") or {}).get("edsHost", device["host"]),
                                      "port": (config.eds_port if config else None) or (settings.get("device") or {}).get("edsPort", 6200)}
        ab["status"] = "PARTICLE_REFINED"
        manifest["status"] = "CALIBRATION_PENDING_REVIEW"
    except InterruptedError as exc:
        audit.update(status="STOPPED", error=str(exc))
    except Exception as exc:
        audit.update(status="FAILED", error=str(exc))
        manifest["error"] = str(exc)
    finally:
        cleanup = {"deviceActionsPerformed": connected, "stageMoved": bool(audit.get("stageMoved")),
                   "gunStateChanged": bool(audit.get("gunOnStatus"))}
        if connected:
            if restore and start is not None:
                try:
                    # Cancellation does not skip restoration of the known starting position.
                    approach = np.asarray(start) + cfg.approach_offset_um
                    if acquisition.inside(start):
                        move_verified(microscope, approach, cfg)
                    cleanup["stageRestore"] = move_verified(microscope, start, cfg)
                except Exception as exc:
                    cleanup["stageRestoreError"] = str(exc)
            if connection and connection.get("magnification"):
                try:
                    microscope.imaging = {"magnification": connection["magnification"], "imageShape": [cfg.image_size] * 2}
                    if connection.get("fieldSizeUm"):
                        microscope.imaging["fieldSizeUm"] = connection["fieldSizeUm"]
                    cleanup["imagingRestore"] = microscope.connect()
                except Exception as exc:
                    cleanup["imagingRestoreError"] = str(exc)
            try:
                microscope.restore_video()
            except Exception as exc:
                cleanup["videoRestoreError"] = str(exc)
            if gun_off:
                try:
                    cleanup["gunOffStatus"] = str(microscope.gun_off())
                    cleanup["gunStateChanged"] = True
                except Exception as exc:
                    cleanup["gunOffError"] = str(exc)
        cleanup_errors = [v for k, v in cleanup.items() if k.endswith("Error")]
        if cleanup_errors:
            audit["cleanupErrors"] = cleanup_errors
            if audit.get("acceptance"):
                audit["acceptance"]["withinLimits"] = False
                audit["acceptance"]["reasons"].append("设备恢复失败，需要操作员检查设备后重试")
            audit["status"] = "FAILED"
            audit["error"] = "；".join(cleanup_errors)
        manifest["cleanup"] = cleanup
        audit["completedAt"] = _now()
        try:
            from .particle_calibration_report import build_particle_calibration_report
            audit["htmlReport"] = build_particle_calibration_report(manifest, task_dir).relative_to(task_dir).as_posix()
        except Exception as exc:
            audit["reportError"] = str(exc)
        _emit_event(
            event_logger,
            "calibration_stage_finished",
            calibrationMode="ab_particle",
            stage="calibration",
            attempt=None,
            status=str(audit.get("status") or "unknown").lower(),
            elapsedSeconds=_elapsed(calibration_started),
            error=audit.get("error"),
            captureCount=len(audit.get("captures") or []),
        )
        persist()
    return manifest
