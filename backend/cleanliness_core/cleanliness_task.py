from __future__ import annotations

import json
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import cv2
import numpy as np
from loguru import logger
from backend.cleanliness_core.imaging_settings import calibration_with_image_size, validate_capture_image_size

from backend.cleanliness_core.ab_point_calibration import (
    NavigationABDetection,
    annotate_navigation_ab,
    build_ab_preview_transform,
    infer_ab_handedness,
    nav_pixel_to_stage,
    resolve_ab_reference_points,
    validate_ab_export,
)
from backend.cleanliness_core.atomic_json import write_json_atomic as _write_json_atomic
from backend.cleanliness_core.large_particle_center import LargeParticleCenterSettings, plan_large_particle_center
from backend.cleanliness_core.cleanliness_detection import (
    CalibrationMatchResult,
    CandidateGate,
    CandidateSegmentationConfig,
    ParticleFilter,
    SemAnchorCandidate,
    TransformSolution,
    build_sem_anchor_candidates,
    choose_calibration_matches,
    evaluate_transform,
    filter_optical_particles,
    load_cleanliness_config,
    pixel_to_stage_xy,
    select_calibration_groups,
    select_calibration_landmarks,
    select_validation_landmarks,
    transform_solution_from_dict,
)
from backend.cleanliness_core.optical_particle_analysis import (
    MatchDecision,
    OpticalParticle,
    annotate_sem_match,
    decode_optical_image,
    load_calibration,
    load_optical_export,
    normalize_uint8,
)
from backend.cleanliness_core.optical_particle_task import (
    ZemMicroscope,
    _capture_signature,
    _validate_calibrated_capture,
    _validate_same_signature,
)
from backend.cleanliness_core.route_optimizer import RoutePlan, optimize_route


ProgressCallback = Callable[[int, int, str], None]
StopCallback = Callable[[], bool]
StageEventCallback = Callable[[str, dict[str, Any]], None]


def default_cleanliness_config_path() -> Path:
    source_root = Path(__file__).resolve().parents[2]
    resource_root = Path(
        os.environ.get("ZEM_RESOURCE_ROOT", str(source_root / "resources"))
    )
    candidates = (
        resource_root / "conf" / "cleanliness_detection.json",
        Path(sys.prefix)
        / "data"
        / "zem_cleanliness"
        / "conf"
        / "cleanliness_detection.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _relative(path: Path, root: Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def _stage_xy(value: Any) -> tuple[float, float]:
    if isinstance(value, dict):
        return float(value["x"]), float(value["y"])
    return float(value[0]), float(value[1])


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _seconds_since(started: float) -> float:
    return round(max(0.0, time.monotonic() - started), 3)


def _emit_stage_event(
    callback: StageEventCallback | None,
    event: str,
    **fields: Any,
) -> None:
    """Emit structured timing evidence without allowing logging to stop hardware work."""
    if callback is not None:
        try:
            callback(event, fields)
            return
        except Exception as error:
            logger.warning("结构化阶段日志写入失败 event={} error={}", event, error)
    logger.bind(event=event, **fields).info(
        "阶段事件 event={} fields={}",
        event,
        json.dumps(fields, ensure_ascii=False, default=str),
    )


def _empty_particle_timing() -> dict[str, Any]:
    return {
        "stageMoveSeconds": None,
        "captureSeconds": 0.0,
        "matchSeconds": 0.0,
        "edsSeconds": 0.0,
        "totalSeconds": None,
    }


def _accumulate_timing(record: dict[str, Any], key: str, seconds: float | None) -> float:
    timing = record.setdefault("timing", _empty_particle_timing())
    if seconds is None:
        return float(timing.get(key) or 0.0)
    total = round(float(timing.get(key) or 0.0) + max(0.0, float(seconds)), 3)
    timing[key] = total
    return total


def _finalize_particle_timing(record: dict[str, Any], started: float) -> None:
    timing = record.setdefault("timing", _empty_particle_timing())
    timing["totalSeconds"] = _seconds_since(started)
    logger.info(
        "颗粒阶段耗时 sequence={} status={} 移台={}s 拍图={}s 匹配={}s 能谱={}s 合计={}s",
        record.get("sequence"),
        record.get("status"),
        timing.get("stageMoveSeconds"),
        timing.get("captureSeconds"),
        timing.get("matchSeconds"),
        timing.get("edsSeconds"),
        timing.get("totalSeconds"),
    )


def _timing_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "stageMoveSeconds",
        "captureSeconds",
        "matchSeconds",
        "edsSeconds",
        "totalSeconds",
    )
    samples: dict[str, list[float]] = {key: [] for key in keys}
    for record in records:
        timing = record.get("timing") or {}
        for key in keys:
            value = timing.get(key)
            if isinstance(value, (int, float)):
                samples[key].append(float(value))
    summary: dict[str, Any] = {"particleCount": len(records)}
    for key, values in samples.items():
        if not values:
            summary[key] = None
            continue
        summary[key] = {
            "count": len(values),
            "total": round(sum(values), 3),
            "mean": round(sum(values) / len(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }
    return summary


def _resolve_experience_path(settings_path: Path, settings: dict[str, Any]) -> Path:
    configured = Path(str(settings.get("experienceCalibration") or ""))
    if not configured.is_absolute():
        configured = settings_path.parent / configured
    return configured.resolve()


def _candidate_segmentation_config(
    settings_path: Path,
    calibration_settings: dict[str, Any],
    *,
    sam3_url: str | None = None,
) -> CandidateSegmentationConfig:
    config = CandidateSegmentationConfig.from_dict(
        calibration_settings.get("candidateSegmentation"),
        settings_path=settings_path,
        sam3_url=sam3_url,
    )
    config.validate()
    return config


def _config_value(
    explicit: Any,
    settings: dict[str, Any],
    section: str,
    key: str,
    fallback: Any,
) -> Any:
    if explicit is not None:
        return explicit
    return (settings.get(section) or {}).get(key, fallback)


@dataclass(frozen=True)
class CleanlinessTaskConfig:
    input_path: Path
    output_dir: Path
    settings_path: Path = field(default_factory=default_cleanliness_config_path)
    experience_calibration_path: Path | None = None
    particle_filter: ParticleFilter = field(default_factory=ParticleFilter)
    selected_sequences: tuple[int, ...] | None = None
    microscope_host: str | None = None
    microscope_port: int | None = None
    eds_host: str | None = None
    eds_port: int | None = None
    eds_enabled: bool | None = None
    eds_retry_count: int = 0
    anchor_sequences: tuple[int, ...] | None = None
    validation_sequences: tuple[int, ...] | None = None
    calibration_groups: list[dict] | None = None
    pixel_avg: int = 7
    max_capture_attempts: int = 2
    restore_start_position: bool | None = None
    auto_gun_on: bool | None = None
    gun_off_after_task: bool | None = None
    sam3_url: str | None = None
    calibration_strategy: str = "multi_particle"
    expected_input_sha256: str | None = None


def save_ab_point_calibration(
    config: CleanlinessTaskConfig,
    *,
    navigation_image: np.ndarray | None = None,
    navigation_bounds: Sequence[float] = (),
    detection: NavigationABDetection | None = None,
    a_index: int = 0,
    b_index: int = 1,
    handedness: str | None = None,
    manual_stage_points: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """保存导航检测或手动台坐标的 AB 结果；不连接设备、不移台。"""

    strategy = str(config.calibration_strategy).strip().lower()
    if strategy not in {"ab_points", "ab_particle"}:
        raise ValueError("保存 AB 点结果时 calibration_strategy 必须为 ab_points 或 ab_particle")
    settings_path = Path(config.settings_path).resolve()
    settings = load_cleanliness_config(settings_path)
    export = load_optical_export(config.input_path)
    if (
        config.expected_input_sha256 is not None
        and export.sha256 != config.expected_input_sha256
    ):
        raise ValueError(
            "光镜 JSON 内容已变化，请重新加载后再执行 AB 点定位"
        )
    validate_ab_export(export)
    manual = None
    if manual_stage_points is not None:
        if strategy != "ab_points":
            raise ValueError("手动 AB 坐标仅支持 AB 标定")
        manual = _validated_manual_ab_points(manual_stage_points)
        coarse_a = (manual["a"]["x"], manual["a"]["y"])
        coarse_b = (manual["b"]["x"], manual["b"]["y"])
    else:
        if detection is None:
            raise ValueError("缺少 AB 导航检测结果")
        image = np.asarray(navigation_image)
        if image.ndim not in (2, 3) or image.shape[:2] != detection.image_shape:
            raise ValueError(
                "导航图尺寸与检测结果不一致："
                f"{image.shape[:2]} != {detection.image_shape}"
            )
        bounds = tuple(float(value) for value in navigation_bounds)
        if len(bounds) != 4:
            raise ValueError("导航图 bounds 必须包含四个数值")
        if not (0 <= a_index < len(detection.holes)):
            raise ValueError(f"A 点候选索引无效：{a_index}")
        if not (0 <= b_index < len(detection.holes)):
            raise ValueError(f"B 点候选索引无效：{b_index}")
        if a_index == b_index:
            raise ValueError("A 点和 B 点不能选择同一个孔")

        a_detection = detection.holes[a_index]
        b_detection = detection.holes[b_index]
        coarse_a = nav_pixel_to_stage(
            a_detection.center_xy,
            detection.image_shape,
            bounds,
        )
        coarse_b = nav_pixel_to_stage(
            b_detection.center_xy,
            detection.image_shape,
            bounds,
        )
    ab_settings = settings.get("abPoint") or {}
    selected_handedness = str(
        handedness or ab_settings.get("handedness") or "right"
    )
    # 参考点可能是 sourceStage 坐标系（与颗粒 abPosition 不同系），先解析
    resolved_references = resolve_ab_reference_points(export)
    preview_transform = build_ab_preview_transform(
        resolved_references["a"],
        resolved_references["b"],
        coarse_a,
        coarse_b,
        handedness=selected_handedness,
    )
    if manual is not None:
        preview_transform["source"] = "operator_confirmed_stage_points"

    selected = (
        _select_by_sequences(export.particles, config.selected_sequences)
        if config.selected_sequences is not None
        else filter_optical_particles(export.particles, config.particle_filter)
    )
    if not selected:
        raise ValueError("当前筛选条件下没有待检测颗粒")

    task_dir = Path(config.output_dir).resolve()
    manifest_path = task_dir / "task.json"
    if manifest_path.exists():
        raise FileExistsError(f"任务目录已经存在 task.json：{task_dir}")
    task_dir.mkdir(parents=True, exist_ok=True)
    navigation_record = None
    if manual is None:
        navigation_dir = task_dir / "navigation"
        navigation_dir.mkdir(parents=True, exist_ok=True)
        original_path = navigation_dir / "navigation.png"
        annotated_path = navigation_dir / "navigation_ab_annotated.png"
        annotated = annotate_navigation_ab(
            image,
            detection,
            a_index=a_index,
            b_index=b_index,
        )
        if not cv2.imwrite(str(original_path), image):
            raise OSError(f"无法保存导航图：{original_path}")
        if not cv2.imwrite(str(annotated_path), annotated):
            raise OSError(f"无法保存 AB 标注导航图：{annotated_path}")

        navigation_record = {
            "image": _relative(original_path, task_dir),
            "annotatedImage": _relative(annotated_path, task_dir),
            "imageShape": list(detection.image_shape),
            "boundsUm": list(bounds),
        }
    device_settings = settings.get("device") or {}
    microscope_host = str(
        config.microscope_host
        or device_settings.get("microscopeHost")
        or ""
    )
    microscope_port = int(
        config.microscope_port
        or device_settings.get("microscopePort")
        or 0
    )
    assignment = {
        "a": {
            "candidateIndex": int(a_index) if manual is None else None,
            "pixel": list(a_detection.center_xy) if manual is None else None,
            "coarseStageUm": list(coarse_a),
        },
        "b": {
            "candidateIndex": int(b_index) if manual is None else None,
            "pixel": list(b_detection.center_xy) if manual is None else None,
            "coarseStageUm": list(coarse_b),
        },
    }
    ab_point_result = {
        "status": "PENDING_SEM_REFINEMENT",
        # ab 坐标系下的参考点（与颗粒 abPosition 同系，变换矩阵的源坐标）
        "sourceReferencePointsUm": {
            "a": list(resolved_references["a"]),
            "b": list(resolved_references["b"]),
        },
        "referenceResolution": {
            key: value
            for key, value in resolved_references.items()
            if key not in ("a", "b")
        },
        "coordinateSource": "manual_stage_points" if manual else "navigation_detection",
        "manualStagePoints": manual,
        "navigation": navigation_record,
        "detection": detection.to_dict() if manual is None else None,
        "assignment": assignment,
        "previewTransform": preview_transform,
    }
    manifest: dict[str, Any] = {
        "schemaVersion": 3,
        "taskType": "CLEANLINESS_DETECTION",
        "taskId": str(uuid.uuid4()),
        "status": "AB_POINTS_PENDING_SEM_REFINEMENT",
        "startedAt": _now(),
        "completedAt": None,
        "input": {
            "path": str(export.path),
            "sha256": export.sha256,
            "projectId": export.project_id,
            "workpieceId": export.workpiece_id,
            "workpieceName": export.workpiece_name,
            "particleCount": len(export.particles),
            "registrationMode": export.registration_mode,
            "referencePointsUm": {
                "a": list(export.reference_a_um),
                "b": list(export.reference_b_um),
            },
        },
        "selection": {
            "filter": config.particle_filter.to_dict(),
            "count": len(selected),
            "sequences": [particle.sequence for particle in selected],
        },
        "device": {
            "microscope": {
                "host": microscope_host,
                "port": microscope_port,
                "connected": False,
            },
        },
        "calibration": {
            "strategy": strategy,
            "experimental": True,
            "groups": [],
            "selectedGroup": None,
            "review": None,
            "abPointResult": ab_point_result,
            "previewTransform": preview_transform,
        },
        "particles": [],
        "reports": None,
        "cleanup": {
            "deviceActionsPerformed": False,
            "stageMoved": False,
            "gunStateChanged": False,
        },
        "error": None,
    }
    _write_json_atomic(task_dir / "ab_point_result.json", ab_point_result)
    _write_json_atomic(manifest_path, manifest)
    return manifest


def abandon_ab_point_calibration(
    task_dir: Path,
    *,
    reviewer: str = "user",
    note: str = "",
) -> dict[str, Any]:
    task_dir = Path(task_dir).resolve()
    manifest_path = task_dir / "task.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration = manifest.get("calibration") or {}
    status = manifest.get("status")
    is_ab_review = (
        status == "CALIBRATION_PENDING_REVIEW"
        and str(calibration.get("strategy")) in {"ab_points", "ab_particle"}
    )
    if status != "AB_POINTS_PENDING_SEM_REFINEMENT" and not is_ab_review:
        raise ValueError(
            f"当前任务状态不能放弃 AB 粗定位：{status}"
        )
    calibration["review"] = {
        "accepted": False,
        "reviewer": str(reviewer),
        "reviewedAt": _now(),
        "note": str(note),
    }
    manifest["status"] = "AB_POINTS_ABANDONED"
    manifest["completedAt"] = _now()
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _ab_refine_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """归一化 abPoint.semRefine 配置，缺省值来自 Zem20F 真机实测。"""
    refine = (settings.get("abPoint") or {}).get("semRefine") or {}
    axis = refine.get("pixelAxisMatrix") or ((-1.0, 0.0), (0.0, 1.0))
    approach = refine.get("approachOffsetUm") or (-150.0, -150.0)
    shape = refine.get("imageShape") or (1024, 1024)
    return {
        "magnification": int(refine.get("magnification") or 100),
        "imageShape": [int(value) for value in shape],
        "pixelAvg": int(refine.get("pixelAvg") or 7),
        "holeDiameterUm": float(refine.get("holeDiameterUm") or 910.0),
        "radiusTolerance": float(refine.get("radiusTolerance") or 0.15),
        "maxSearchRadiusUm": float(refine.get("maxSearchRadiusUm") or 2000.0),
        "searchStepRatio": float(refine.get("searchStepRatio") or 0.60),
        "maxCaptures": int(refine.get("maxCaptures") or 25),
        "validationCount": int(refine.get("validationCount") or 2),
        "stageErrorUm": float(refine.get("stageErrorUm") or 2.0),
        "fieldRotationDeg": float(refine.get("fieldRotationDeg") or 0.0),
        "pixelAxisMatrix": [[float(v) for v in row] for row in axis],
        "approachOffsetUm": [float(v) for v in approach],
    }


def _stage_to_pixel(
    stage_xy: Sequence[float],
    stage_center: Sequence[float],
    image_shape: Sequence[int],
    pixel_size_um: float,
    axis_signs: Sequence[float],
    pixel_axis_matrix: Sequence[Sequence[float]],
    field_rotation_deg: float,
) -> tuple[float, float]:
    """台坐标 → 图像像素坐标（hole_center_locator._pixel_offset_to_stage 的逆）。"""
    height, width = int(image_shape[0]), int(image_shape[1])
    axis = np.diag(axis_signs) @ np.asarray(pixel_axis_matrix, dtype=np.float64)
    radians = math.radians(float(field_rotation_deg))
    rotation = np.asarray(
        [
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ]
    )
    delta = np.asarray(stage_xy, dtype=np.float64) - np.asarray(
        stage_center, dtype=np.float64
    )
    offset_px = np.linalg.solve(rotation @ axis, delta) / float(pixel_size_um)
    return width / 2.0 + float(offset_px[0]), height / 2.0 + float(offset_px[1])


def _pick_spread_particles(
    particles: Sequence[OpticalParticle],
    count: int,
) -> list[OpticalParticle]:
    """最远点采样：挑选 AB 空间上相互距离最大的若干颗粒（覆盖全样品面）。"""
    pool = list(particles)
    count = max(0, int(count))
    if count == 0 or not pool:
        return []
    points = np.asarray(
        [[p.ab_x_um, p.ab_y_um] for p in pool], dtype=np.float64
    )
    chosen = [int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))]
    while len(chosen) < min(count, len(pool)):
        distances = np.min(
            np.stack(
                [np.linalg.norm(points - points[i], axis=1) for i in chosen]
            ),
            axis=0,
        )
        distances[chosen] = -1.0
        chosen.append(int(np.argmax(distances)))
    return [pool[i] for i in chosen]


def _annotate_ab_validation_capture(
    capture: dict[str, Any],
    particle: OpticalParticle,
    output_path: Path,
) -> Path:
    """在验证颗粒 SEM 实拍图上标出预测位置（视场中心）与期望颗粒尺寸。"""
    display_path = capture.get("displayPath") or capture.get("rawPath")
    image = cv2.imread(str(display_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        image = cv2.imread(str(capture["rawPath"]), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"无法读取验证颗粒 SEM 图：{display_path}")
    canvas = cv2.cvtColor(
        cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(image),
        cv2.COLOR_GRAY2BGR,
    )
    height, width = image.shape[:2]
    pixel_size = float(capture.get("pixelSizeUm") or 1.0)
    center = (width // 2, height // 2)
    cyan = (255, 220, 80)
    radius = max(8, int(round((particle.length_um / 2.0) / pixel_size)) + 6)
    cv2.drawMarker(canvas, center, cyan, cv2.MARKER_CROSS, 42, 2, cv2.LINE_AA)
    cv2.circle(canvas, center, radius, cyan, 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"#{particle.sequence} expected here"
        f" (L={particle.length_um:.0f}um)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        cyan,
        2,
        cv2.LINE_AA,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"无法保存验证颗粒标注图：{output_path}")
    return output_path


def _capture_ab_validation_particles(
    microscope: Any,
    task_dir: Path,
    particles: Sequence[OpticalParticle],
    solution: TransformSolution,
    experience: Any,
    *,
    stage_error_um: float,
    progress: ProgressCallback,
    progress_offset: int,
    progress_total: int,
    should_stop: StopCallback,
) -> list[dict[str, Any]]:
    """矩阵验证实拍：按精定位矩阵移台到验证颗粒的预测位置，用经验标定的
    分析成像参数拍 SEM 图，供人工与光镜颗粒照片肉眼对比（同多颗粒法证据图）。

    单颗失败只记录 error 不中断——矩阵本身已建好，验证图少一张不影响人工审核。
    """
    if not particles:
        return []
    # 孔精定位用低倍大视场，颗粒在其中不可见；切换到经验标定的分析成像参数。
    # ZemMicroscope.connect() 会按 imaging["magnification"] 重新设置倍率。
    microscope.imaging = dict(experience.imaging)
    microscope.connect()
    records: list[dict[str, Any]] = []
    for offset, particle in enumerate(particles):
        if should_stop():
            raise InterruptedError("用户停止了 AB SEM 精定位")
        sequence = particle.sequence
        target = solution.transform_xy(particle.ab_x_um, particle.ab_y_um)
        progress(
            progress_offset + offset,
            progress_total,
            f"[验证] 移台到颗粒 #{sequence} 预测位置 "
            f"({target[0]:.1f}, {target[1]:.1f}) µm 实拍",
        )
        record: dict[str, Any] = {
            "sequence": sequence,
            "lengthUm": particle.length_um,
            "opticalAbPositionUm": {"x": particle.ab_x_um, "y": particle.ab_y_um},
            "targetStageUm": {"x": target[0], "y": target[1]},
            "images": {},
            "error": None,
        }
        try:
            optical_path = task_dir / "optical" / f"{sequence:04d}.png"
            if not optical_path.is_file():
                decode_optical_image(particle, optical_path)
            record["images"]["opticalImage"] = _relative(optical_path, task_dir)
            _require_target_inside_roi(experience, target, particle=particle)
            move = _move_stage_checked(
                microscope, target, maximum_error_um=stage_error_um
            )
            record["moveErrorUm"] = float(move["positionErrorUm"])
            raw_path = task_dir / "ab_sem" / "validation" / f"{sequence:04d}.png"
            _, capture = microscope.capture(raw_path)
            post_capture = microscope.get_position()
            record["captureStageUm"] = {"x": float(post_capture[0]), "y": float(post_capture[1])}
            record["postCaptureErrorUm"] = math.hypot(post_capture[0] - target[0], post_capture[1] - target[1])
            record["pixelSizeUm"] = float(capture["pixelSizeUm"])
            record["imageShape"] = capture.get("shape")
            if record["postCaptureErrorUm"] > stage_error_um:
                raise RuntimeError(f"验证采图期间台位置超差：{record['postCaptureErrorUm']:.2f} μm")
            annotated_path = _annotate_ab_validation_capture(
                capture,
                particle,
                raw_path.with_name(f"{raw_path.stem}.annotated.png"),
            )
            record["images"].update(
                {
                    "semRawImage": _relative_or_none(capture["rawPath"], task_dir),
                    "semDisplayImage": _relative_or_none(
                        capture["displayPath"], task_dir
                    ),
                    "semAnnotatedImage": _relative_or_none(annotated_path, task_dir),
                }
            )
        except InterruptedError:
            raise
        except Exception as error:
            logger.warning(f"验证颗粒 #{sequence} 实拍失败: {error}")
            record["error"] = str(error)
        records.append(record)
    return records


def _relative_or_none(path: Any, root: Path) -> str | None:
    if not path:
        return None
    try:
        return _relative(Path(str(path)), root)
    except ValueError:
        return str(path)


def _ab_hole_summary(
    locate_result: dict[str, Any],
    task_dir: Path,
) -> dict[str, Any]:
    """把 locate_hole_center 的结果压缩成 manifest 友好的摘要（路径转相对）。"""
    final = locate_result["finalCapture"]
    captures = [
        {
            "index": item.get("index"),
            "stageTarget": item.get("stageTarget"),
            "stageAfter": item.get("stageAfter"),
            "positionErrorUm": item.get("positionErrorUm"),
            "warning": item.get("warning"),
            "status": (item.get("detection") or {}).get("status"),
            "rawPath": _relative_or_none(item.get("rawPath"), task_dir),
            "annotatedPath": _relative_or_none(item.get("annotatedPath"), task_dir),
        }
        for item in locate_result.get("captures") or []
    ]
    return {
        "holeStageUm": locate_result["holeStage"],
        "coarseStageUm": locate_result["coarseStage"],
        "offsetFromCoarseUm": locate_result["offsetFromCoarseUm"],
        "sigmaUm": locate_result.get("sigmaUm"),
        "diameterUm": locate_result.get("diameterUm"),
        "coverageDeg": locate_result.get("coverageDeg"),
        "rmsPx": locate_result.get("rmsPx"),
        "pixelSizeUm": locate_result.get("pixelSizeUm"),
        "axisSigns": locate_result.get("axisSigns"),
        "captureCount": locate_result.get("captureCount"),
        "centerMoves": locate_result.get("centerMoves"),
        "warnings": locate_result.get("warnings") or [],
        "detection": final.get("detection"),
        "finalImage": _relative_or_none(final.get("rawPath"), task_dir),
        "finalDisplayImage": _relative_or_none(final.get("displayPath"), task_dir),
        "finalAnnotatedImage": _relative_or_none(
            final.get("annotatedPath"), task_dir
        ),
        "finalStageUm": final.get("stageAfter"),
        "captures": captures,
    }


def _validated_manual_ab_points(points: dict[str, Any]) -> dict[str, Any]:
    parsed = {}
    for label in ("a", "b"):
        point = points.get(label)
        if not isinstance(point, dict):
            raise ValueError(f"{label.upper()} 点必须包含 X/Y 坐标")
        values = [point.get("x"), point.get("y")]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError(f"{label.upper()} 点坐标必须包含两个有限数值")
        if any(abs(v) > 50000 for v in values):
            raise ValueError("AB 台坐标必须在 ±50 mm 内，请检查单位和小数点")
        parsed[label] = dict(zip(("x", "y"), map(float, values)))
    baseline = math.hypot(parsed["b"]["x"] - parsed["a"]["x"], parsed["b"]["y"] - parsed["a"]["y"])
    if baseline < 1000:
        raise ValueError("A/B 点间距必须至少为 1 mm")
    return {**parsed, "baselineUm": baseline, "coordinateUnit": "um",
            "operator": str(points.get("operator") or "user"), "recordedAt": _now()}


def set_manual_ab_stage_points(
    task_dir: Path,
    *,
    a_stage_um: Sequence[float],
    b_stage_um: Sequence[float],
    operator: str = "user",
) -> dict[str, Any]:
    """记录操作员确认的 A/B 孔台坐标，供精定位流程跳过自动找孔并实拍验证。"""
    task_dir = Path(task_dir).resolve()
    manifest_path = task_dir / "task.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "AB_POINTS_PENDING_SEM_REFINEMENT":
        raise ValueError(
            f"当前任务状态不能设置人工 A/B 坐标：{manifest.get('status')}"
        )

    def point(name: str, values: Sequence[float]) -> tuple[float, float]:
        parsed = tuple(float(value) for value in values)
        if len(parsed) != 2 or not all(math.isfinite(value) for value in parsed):
            raise ValueError(f"{name} 孔坐标必须包含两个有限数值")
        return parsed

    a_stage = point("A", a_stage_um)
    b_stage = point("B", b_stage_um)
    baseline = math.hypot(b_stage[0] - a_stage[0], b_stage[1] - a_stage[1])
    if baseline < 1000.0:
        raise ValueError(f"A/B 孔台坐标间距异常：{baseline:.1f} µm")

    calibration = manifest.get("calibration") or {}
    ab_result = calibration.get("abPointResult") or {}
    ab_result["manualStagePoints"] = {
        "a": {"x": a_stage[0], "y": a_stage[1]},
        "b": {"x": b_stage[0], "y": b_stage[1]},
        "baselineUm": baseline,
        "operator": str(operator),
        "recordedAt": _now(),
    }
    ab_result["status"] = "MANUAL_STAGE_POINTS_READY"
    ab_result.pop("semRefinement", None)
    calibration["abPointResult"] = ab_result
    manifest["calibration"] = calibration
    manifest["error"] = None
    _write_json_atomic(task_dir / "ab_point_result.json", ab_result)
    _write_json_atomic(manifest_path, manifest)
    return manifest


def refine_ab_point_calibration(
    task_dir: Path,
    *,
    config: CleanlinessTaskConfig | None = None,
    microscope: Any | None = None,
    progress: ProgressCallback | None = None,
    should_stop: StopCallback | None = None,
    event_logger: StageEventCallback | None = None,
) -> dict[str, Any]:
    """AB 点法第二阶段：连接电镜，对 A/B 两孔做 SEM 闭环圆心精定位。

    成功后用精确圆心重建 AB 变换矩阵（与 Tab3 分析兼容），再按新矩阵移台到
    若干空间分散的验证颗粒的预测位置、切换到分析成像参数实拍——供人工与
    光镜颗粒照片肉眼对比（同多颗粒法证据图），状态推进到
    ``CALIBRATION_PENDING_REVIEW`` 等待用户确认；失败/停止时状态保持
    ``AB_POINTS_PENDING_SEM_REFINEMENT``，错误记录在
    ``calibration.abPointResult.semRefinement`` 里，可重试。
    """
    from backend.cleanliness_core.hole_center_locator import (
        HoleLocatorConfig,
        locate_hole_center,
    )

    progress = progress or (lambda _index, _total, _message: None)
    should_stop = should_stop or (lambda: False)
    task_dir = Path(task_dir).resolve()
    manifest_path = task_dir / "task.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "AB_POINTS_PENDING_SEM_REFINEMENT":
        raise ValueError(
            f"当前任务状态不能执行 SEM 精定位：{manifest.get('status')}"
        )
    calibration = manifest["calibration"]
    ab_result = calibration.get("abPointResult") or {}
    assignment = ab_result.get("assignment") or {}
    if calibration.get("strategy") == "ab_particle":
        raise ValueError("AB 导航 + 颗粒精校准不可执行 SEM 找孔，请使用颗粒精校准流程")
    coarse_a = tuple(float(v) for v in (assignment.get("a") or {}).get("coarseStageUm") or ())
    coarse_b = tuple(float(v) for v in (assignment.get("b") or {}).get("coarseStageUm") or ())
    if len(coarse_a) != 2 or len(coarse_b) != 2:
        raise ValueError("AB 粗定位结果缺少 A/B 坐标，无法执行 SEM 精定位")
    manual_points = ab_result.get("manualStagePoints") or {}
    manual_a = tuple(
        float(v)
        for v in (
            (manual_points.get("a") or {}).get("x"),
            (manual_points.get("a") or {}).get("y"),
        )
        if v is not None
    )
    manual_b = tuple(
        float(v)
        for v in (
            (manual_points.get("b") or {}).get("x"),
            (manual_points.get("b") or {}).get("y"),
        )
        if v is not None
    )
    use_manual_points = len(manual_a) == 2 and len(manual_b) == 2

    settings_path = (
        Path(config.settings_path).resolve()
        if config is not None
        else default_cleanliness_config_path().resolve()
    )
    settings = load_cleanliness_config(settings_path)
    refine_settings = _ab_refine_settings(settings)
    task_settings = settings.get("task") or {}
    host = str(
        _config_value(
            config.microscope_host if config else None,
            settings, "device", "microscopeHost", "192.168.0.15",
        )
    )
    port = int(
        _config_value(
            config.microscope_port if config else None,
            settings, "device", "microscopePort", 61557,
        )
    )
    eds_host = str(
        _config_value(config.eds_host if config else None, settings, "device", "edsHost", host)
    )
    eds_port = int(
        _config_value(config.eds_port if config else None, settings, "device", "edsPort", 6200)
    )
    restore_position = bool(
        _config_value(
            config.restore_start_position if config else None,
            settings, "task", "restoreStartPosition", True,
        )
    )
    auto_gun_on = bool(
        _config_value(
            config.auto_gun_on if config else None,
            settings, "task", "gunOnAtStart", False,
        )
    )
    gun_off_after_phase = bool(
        _config_value(
            config.gun_off_after_task if config else None,
            settings, "task", "gunOffAfterTask", True,
        )
    )
    handedness = str(
        (ab_result.get("previewTransform") or {}).get("handedness")
        or (settings.get("abPoint") or {}).get("handedness")
        or "right"
    )
    experience_path = (
        Path(config.experience_calibration_path).resolve()
        if config is not None and config.experience_calibration_path is not None
        else _resolve_experience_path(settings_path, settings)
    )
    export = load_optical_export(Path(manifest["input"]["path"]))
    if export.sha256 != manifest["input"]["sha256"]:
        raise ValueError("光镜 JSON 内容已变化，请重新执行 AB 粗定位")
    selected = _select_by_sequences(
        export.particles,
        manifest.get("selection", {}).get("sequences") or (),
    )
    # 参考点每次从导出实时解析到 ab 坐标系（不信任旧任务里可能错的保存值）；
    # 导出缺 referencePoints 的旧数据回退到清单保存值
    if export.reference_a_um is not None and export.reference_b_um is not None:
        resolved_references = resolve_ab_reference_points(export)
    else:
        saved_points = ab_result.get("sourceReferencePointsUm") or {}
        saved_a = tuple(float(v) for v in saved_points.get("a") or ())
        saved_b = tuple(float(v) for v in saved_points.get("b") or ())
        if len(saved_a) != 2 or len(saved_b) != 2:
            raise ValueError("光镜导出与任务清单都缺少 A/B 参考点坐标")
        resolved_references = {
            "a": saved_a,
            "b": saved_b,
            "method": "manifestSaved",
            "sourceToAbDet": None,
        }
    source_a = resolved_references["a"]
    source_b = resolved_references["b"]
    ab_result["sourceReferencePointsUm"] = {
        "a": list(source_a),
        "b": list(source_b),
    }
    ab_result["referenceResolution"] = {
        key: value
        for key, value in resolved_references.items()
        if key not in ("a", "b")
    }

    locator_config = HoleLocatorConfig(
        hole_diameter_um=refine_settings["holeDiameterUm"],
        radius_tolerance=refine_settings["radiusTolerance"],
        search_step_ratio=refine_settings["searchStepRatio"],
        max_search_radius_um=refine_settings["maxSearchRadiusUm"],
        max_captures=refine_settings["maxCaptures"],
        approach_offset_um=tuple(refine_settings["approachOffsetUm"]),
    )
    axis_matrix = refine_settings["pixelAxisMatrix"]
    field_rotation = refine_settings["fieldRotationDeg"]
    microscope = microscope or ZemMicroscope(
        host,
        port,
        {
            "imageShape": refine_settings["imageShape"],
            "magnification": refine_settings["magnification"],
        },
        pixel_avg=refine_settings["pixelAvg"],
    )

    validation_targets = _pick_spread_particles(
        selected, refine_settings["validationCount"]
    )
    total_steps = 2 + len(validation_targets)

    refinement: dict[str, Any] = {
        "status": "RUNNING",
        "startedAt": _now(),
        "settings": refine_settings,
        "mode": "manual_stage_points" if use_manual_points else "automatic_hole_search",
    }
    ab_result["semRefinement"] = refinement
    start_position: tuple[float, float] | None = None
    calibration_mode = "ab_manual" if use_manual_points else "ab_points"
    calibration_started = time.monotonic()
    _emit_stage_event(
        event_logger,
        "calibration_stage_started",
        calibrationMode=calibration_mode,
        stage="calibration",
        attempt=None,
        validationCount=len(validation_targets),
    )
    try:
        progress(0, total_steps, f"连接电镜 {host}:{port}（倍率 {refine_settings['magnification']}x）")
        connection_started = time.monotonic()
        _emit_stage_event(
            event_logger,
            "calibration_stage_started",
            calibrationMode=calibration_mode,
            stage="connect",
            attempt=1,
        )
        try:
            connection = microscope.connect()
        except Exception as error:
            _emit_stage_event(
                event_logger,
                "calibration_stage_finished",
                calibrationMode=calibration_mode,
                stage="connect",
                attempt=1,
                status="failed",
                elapsedSeconds=_seconds_since(connection_started),
                error=str(error),
            )
            raise
        _emit_stage_event(
            event_logger,
            "calibration_stage_finished",
            calibrationMode=calibration_mode,
            stage="connect",
            attempt=1,
            status="succeeded",
            elapsedSeconds=_seconds_since(connection_started),
        )
        refinement["connection"] = connection
        manifest["device"]["microscope"]["connected"] = True
        start_position = microscope.get_position()
        if auto_gun_on and hasattr(microscope, "gun_on"):
            progress(0, total_steps, "开启电子枪并等待高压稳定")
            refinement["gunOnStatus"] = str(microscope.gun_on())

        locate_results: dict[str, dict[str, Any]] = {}
        if use_manual_points:
            refined_a = manual_a
            refined_b = manual_b
            progress(
                2,
                total_steps,
                "使用操作员确认的 A/B 孔台坐标，跳过自动找孔并重建变换矩阵",
            )
        else:
            plan = (("A", coarse_a), ("B", coarse_b))
            for index, (label, coarse) in enumerate(plan):
                if should_stop():
                    raise InterruptedError("用户停止了 AB SEM 精定位")
                progress(
                    index,
                    total_steps,
                    f"[{label}] 从粗坐标 ({coarse[0]:.1f}, {coarse[1]:.1f}) µm 开始闭环精定位",
                )
                locate_started = time.monotonic()
                _emit_stage_event(
                    event_logger,
                    "calibration_stage_started",
                    calibrationMode=calibration_mode,
                    stage="hole_localization",
                    holeLabel=label,
                    holeRank=index + 1,
                    attempt=None,
                    coarseStageUm={"x": coarse[0], "y": coarse[1]},
                )
                try:
                    locate_results[label] = locate_hole_center(
                        microscope,
                        coarse,
                        task_dir / "ab_sem" / label.lower(),
                        config=locator_config,
                        pixel_axis_matrix=axis_matrix,
                        field_rotation_deg=field_rotation,
                        stage_error_um=refine_settings["stageErrorUm"],
                        label=label,
                        progress=lambda message, _i=index: progress(
                            _i, total_steps, message
                        ),
                        should_stop=should_stop,
                    )
                except Exception as error:
                    _emit_stage_event(
                        event_logger,
                        "calibration_stage_finished",
                        calibrationMode=calibration_mode,
                        stage="hole_localization",
                        holeLabel=label,
                        holeRank=index + 1,
                        attempt=None,
                        status="stopped" if isinstance(error, InterruptedError) else "failed",
                        elapsedSeconds=_seconds_since(locate_started),
                        error=str(error),
                    )
                    raise
                _emit_stage_event(
                    event_logger,
                    "calibration_stage_finished",
                    calibrationMode=calibration_mode,
                    stage="hole_localization",
                    holeLabel=label,
                    holeRank=index + 1,
                    attempt=None,
                    status="succeeded",
                    elapsedSeconds=_seconds_since(locate_started),
                    captureCount=locate_results[label].get("captureCount"),
                    holeStageUm=locate_results[label].get("holeStage"),
                )

            refined_a = (
                float(locate_results["A"]["holeStage"]["x"]),
                float(locate_results["A"]["holeStage"]["y"]),
            )
            refined_b = (
                float(locate_results["B"]["holeStage"]["x"]),
                float(locate_results["B"]["holeStage"]["y"]),
            )
            progress(2, total_steps, "A/B 精定位完成，重建变换矩阵")
        # 经验标定提前加载：手性推断、验证实拍成像参数与 ROI、Tab3 分析都需要
        try:
            experience = calibration_with_image_size(load_calibration(experience_path), task_settings)
            manifest["experienceCalibration"] = experience.to_audit_dict()
        except Exception as error:
            raise RuntimeError(
                f"经验标定文件不可用（Tab3 分析必需）：{error}"
            ) from error

        # 手性链推断：ab→SEM台 反射性 = 光镜(source→ab)反射性 XOR 经验矩阵反射性。
        # 两点标定本身定不出镜像方向，配置值可能错；推断值可用时以推断为准。
        handedness_source = "configured"
        inferred_handedness = infer_ab_handedness(
            resolved_references.get("sourceToAbDet"),
            getattr(experience, "matrix", None),
        )
        if inferred_handedness is not None:
            if inferred_handedness != handedness:
                progress(
                    2,
                    total_steps,
                    f"手性自动修正：配置 {handedness} → 推断 {inferred_handedness}"
                    "（光镜坐标手性 × 经验矩阵手性）",
                )
            handedness = inferred_handedness
            handedness_source = "chirality_chain"

        transform_started = time.monotonic()
        _emit_stage_event(
            event_logger,
            "calibration_stage_started",
            calibrationMode=calibration_mode,
            stage="transform_fit",
            attempt=None,
        )
        try:
            refined_transform = build_ab_preview_transform(
                source_a,
                source_b,
                refined_a,
                refined_b,
                handedness=handedness,
            )
        except Exception as error:
            _emit_stage_event(
                event_logger,
                "calibration_stage_finished",
                calibrationMode=calibration_mode,
                stage="transform_fit",
                attempt=None,
                status="failed",
                elapsedSeconds=_seconds_since(transform_started),
                error=str(error),
            )
            raise
        sigma_values = (
            [
                float(locate_results[key].get("sigmaUm") or 0.0)
                for key in ("A", "B")
            ]
            if locate_results
            else [0.0, 0.0]
        )
        refined_transform.update(
            {
                "source": (
                    "operator_confirmed_stage_points"
                    if use_manual_points
                    else "sem_refined"
                ),
                "verified": True,
                "handednessSource": handedness_source,
                "fit": {
                    "residualsUm": [0.0, 0.0],
                    "rmsErrorUm": max(sigma_values),
                    "maxErrorUm": max(sigma_values),
                },
            }
        )
        solution = transform_solution_from_dict(refined_transform)
        _emit_stage_event(
            event_logger,
            "calibration_stage_finished",
            calibrationMode=calibration_mode,
            stage="transform_fit",
            attempt=None,
            status="succeeded",
            elapsedSeconds=_seconds_since(transform_started),
            handedness=handedness,
            handednessSource=handedness_source,
        )

        validation_started = time.monotonic()
        _emit_stage_event(
            event_logger,
            "calibration_stage_started",
            calibrationMode=calibration_mode,
            stage="validation_capture",
            attempt=None,
            particleSequences=[particle.sequence for particle in validation_targets],
        )
        try:
            validation_records = _capture_ab_validation_particles(
                microscope,
                task_dir,
                validation_targets,
                solution,
                experience,
                stage_error_um=refine_settings["stageErrorUm"],
                progress=progress,
                progress_offset=2,
                progress_total=total_steps,
                should_stop=should_stop,
            )
        except Exception as error:
            _emit_stage_event(
                event_logger,
                "calibration_stage_finished",
                calibrationMode=calibration_mode,
                stage="validation_capture",
                attempt=None,
                status="stopped" if isinstance(error, InterruptedError) else "failed",
                elapsedSeconds=_seconds_since(validation_started),
                error=str(error),
            )
            raise
        _emit_stage_event(
            event_logger,
            "calibration_stage_finished",
            calibrationMode=calibration_mode,
            stage="validation_capture",
            attempt=None,
            status="succeeded",
            elapsedSeconds=_seconds_since(validation_started),
            succeeded=sum(not record.get("error") for record in validation_records),
            failed=sum(bool(record.get("error")) for record in validation_records),
        )

        refinement.update(
            {
                "status": "SUCCEEDED",
                "completedAt": _now(),
                "a": (
                    {
                        "holeStageUm": {"x": refined_a[0], "y": refined_a[1]},
                        "source": "operator_confirmed",
                        "captureCount": 0,
                    }
                    if use_manual_points
                    else _ab_hole_summary(locate_results["A"], task_dir)
                ),
                "b": (
                    {
                        "holeStageUm": {"x": refined_b[0], "y": refined_b[1]},
                        "source": "operator_confirmed",
                        "captureCount": 0,
                    }
                    if use_manual_points
                    else _ab_hole_summary(locate_results["B"], task_dir)
                ),
                "warnings": [
                    warning
                    for label in ("A", "B")
                    for warning in (
                        (locate_results.get(label) or {}).get("warnings") or []
                    )
                ],
                "validation": validation_records,
                "refinedTransform": refined_transform,
            }
        )
        ab_result["status"] = "SEM_REFINED"
        calibration["result"] = refined_transform
        calibration["refinedTransform"] = refined_transform
        manifest["device"]["eds"] = {"host": eds_host, "port": eds_port}
        manifest["status"] = "CALIBRATION_PENDING_REVIEW"
        _write_json_atomic(task_dir / "ab_point_result.json", ab_result)
    except InterruptedError as error:
        refinement["status"] = "STOPPED"
        refinement["error"] = str(error)
    except Exception as error:
        refinement["status"] = "FAILED"
        refinement["error"] = str(error)
    finally:
        cleanup = manifest.setdefault("cleanup", {})
        cleanup["deviceActionsPerformed"] = True
        cleanup["stageMoved"] = True
        cleanup.update(
            _restore_and_maybe_gun_off(
                microscope,
                start_position,
                restore_position=restore_position,
                gun_off=gun_off_after_phase,
            )
        )
        _emit_stage_event(
            event_logger,
            "calibration_stage_finished",
            calibrationMode=calibration_mode,
            stage="calibration",
            attempt=None,
            status=str(refinement.get("status") or "unknown").lower(),
            elapsedSeconds=_seconds_since(calibration_started),
            error=refinement.get("error"),
        )
        _write_json_atomic(manifest_path, manifest)
    return manifest


class SpectrumAnalyzer(Protocol):
    def check_ready(self) -> dict[str, Any]: ...

    def analyze(
        self,
        particle: OpticalParticle,
        candidate: SemAnchorCandidate | None,
        image_path: Path,
        safe_point: tuple[int, int] | None,
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


class RealSpectrumAnalyzer:
    def __init__(
        self,
        eds_host: str,
        eds_port: int,
        microscope_host: str,
        microscope_port: int,
        *,
        timeout: float = 60.0,
        target_counts: int | None = None,
    ) -> None:
        from backend.cleanliness_core.eds.client_v2 import EdsV2Client

        self.client = EdsV2Client(eds_host, eds_port, timeout=timeout, target_counts=target_counts)
        self.microscope_host = microscope_host
        self.microscope_port = int(microscope_port)

    def check_ready(self) -> dict[str, Any]:
        session_id = self.client.open_session(
            client_name="ZEM Cleanliness Preflight",
            client_version="1.0",
        )
        try:
            return {
                "status": "READY",
                "device": self.client.target,
                "protocol": "grpc_v2",
            }
        finally:
            self.client.close_session(session_id)

    def analyze(
        self,
        particle: OpticalParticle,
        candidate: SemAnchorCandidate | None,
        image_path: Path,
        safe_point: tuple[int, int] | None,
    ) -> dict[str, Any]:
        from backend.cleanliness_core.energy_spectrum import (
            process_energy_spectrum_particle,
        )

        if candidate is None and safe_point is None:
            raise ValueError("中心采谱必须提供明确的图像采样点")
        mask = {
            "label": str(particle.sequence),
            "points": [list(point) for point in candidate.candidate.contour] if candidate is not None else [],
        }
        options = {"target_point": safe_point}
        if candidate is None:
            options["target_mode"] = "large_particle_center_region"
        return process_energy_spectrum_particle(
            self.client,
            mask,
            str(image_path),
            self.microscope_host,
            self.microscope_port,
            **options,
        )

    def close(self) -> None:
        self.client.close()


def _select_by_sequences(
    particles: Sequence[OpticalParticle],
    sequences: Sequence[int],
) -> tuple[OpticalParticle, ...]:
    by_sequence = {particle.sequence: particle for particle in particles}
    missing = [int(value) for value in sequences if int(value) not in by_sequence]
    if missing:
        raise ValueError(f"光镜 JSON 中不存在颗粒 sequence：{missing}")
    return tuple(by_sequence[int(value)] for value in sequences)


def _decision_for_candidate(
    candidate: SemAnchorCandidate,
    candidate_count: int,
    *,
    reason_prefix: str,
    gate: CandidateGate | None = None,
) -> MatchDecision:
    item = candidate.candidate
    gate = gate or CandidateGate()
    rejection = gate.rejection(item)
    if rejection is None and candidate.shape_score > 2.5:
        rejection = f"形貌差 {candidate.shape_score:.3f} 超过 2.5"
    matched = rejection is None
    detail = (
        f"检测={candidate.detector_score:.3f}，形貌差={candidate.shape_score:.3f}"
        if matched
        else f"候选不可信（{rejection}）"
    )
    return MatchDecision(
        status="MATCHED_HIGH" if matched else "FAILED_LOW_CONFIDENCE",
        confidence="high" if matched else "low",
        reason=f"{reason_prefix}；{detail}；光镜轮廓来源={candidate.optical_shape_source}",
        best=item,
        second_score=None,
        score_margin=None,
        candidate_count=int(candidate_count),
    )


def _verify_candidate_mask_identity(
    image,
    particle,
    pixel_size_um,
    decision,
    *,
    refinement_settings=None,
    should_stop=lambda: False,
):
    """Require independent mask-template agreement before EDS for masked exports.

    Correlation and runner-up margin are scores, not calibrated probabilities.
    The existing candidate supplies the EDS contour; a template peak alone never
    substitutes a new contour or authorizes acquisition.
    """
    if not particle.mask_data_base64 or not decision.matched:
        return decision, None
    from .particle_refinement import RefinementSettings, match_particle
    match_settings = refinement_settings or RefinementSettings()
    verification = match_particle(image, particle, pixel_size_um, match_settings, should_stop=should_stop)
    tolerance = max(8.0, particle.width_um / pixel_size_um * .25)
    verification["maximumCenterDifferencePx"] = tolerance
    accepted = verification["status"] == "MATCH"
    if "centerPixel" in verification:
        difference = float(np.linalg.norm(np.subtract(verification["centerPixel"], decision.best.center)))
        verification["candidateCenterDifferencePx"] = difference
        accepted = accepted and difference <= tolerance
    else:
        accepted = False
    verification["accepted"] = accepted
    detail = "光镜掩码模板与候选位置一致" if accepted else "光镜掩码身份复核未通过（分数不足、候选歧义或位置不一致）"
    return replace(decision, status=decision.status if accepted else "FAILED_MASK_IDENTITY",
                   confidence=decision.confidence if accepted else "low",
                   reason=f"{decision.reason}；{detail}"), verification


def _contour_overlap_min(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("contourOverlapMin 必须是 0 与 1 之间的数值")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0 < normalized < 1:
        raise ValueError("contourOverlapMin 必须是 0 与 1 之间的数值")
    return normalized


def _select_mask_guided_candidate(image, particle, pixel_size_um, candidates, *,
                                  stage_center_um, field_rotation_deg=0.0,
                                  pixel_axis_matrix=((1.0, 0.0), (0.0, 1.0)),
                                  segmentation=None, gate=None, should_stop=lambda: False,
                                  calibration_location=None, refinement_settings=None,
                                  minimum_mask_iou=None):
    """Locate by mask, then segment observed SEM pixels around that location.

    A warped optical mask is only a reference, never the measured SEM contour.
    Template uniqueness or conservative two-detector shape corroboration, plus
    the configured final mask IoU (default 0.5), are required alongside the
    physical/shape gates.
    """
    first = candidates[0] if candidates else None
    decision = (_decision_for_candidate(first, len(candidates), reason_prefix="离线/在线统一候选", gate=gate)
                if first else MatchDecision("FAILED_LOW_CONFIDENCE", "low", "没有颗粒候选", None, None, None, 0))
    if not particle.mask_data_base64:
        return first, decision, None
    try:
        overlap_floor = 0.5 if minimum_mask_iou is None else _contour_overlap_min(minimum_mask_iou)
    except ValueError as error:
        return first, replace(decision, status="FAILED_MASK_IDENTITY", confidence="low", reason=str(error)), None
    from .particle_refinement import RefinementSettings, match_particle, optical_mask
    match_settings = refinement_settings or RefinementSettings()
    audit = match_particle(image, particle, pixel_size_um, match_settings, should_stop=should_stop)
    audit["minimumCandidateScoreMargin"] = match_settings.match_min_margin
    audit["minimumTemplateMatchScore"] = match_settings.match_min_score
    audit.update(accepted=False, minimumMaskIoU=overlap_floor,
                 initialCandidateCenterPx=list(first.pixel_xy) if first else None)
    if audit["status"] == "AMBIGUOUS":
        from .shape_consensus import review_ambiguous_shapes
        review = review_ambiguous_shapes(image, particle, pixel_size_um, audit,
            segmentation=segmentation, gate=gate, should_stop=should_stop,
            minimum_shape_margin=match_settings.match_min_margin,
            minimum_template_score=match_settings.match_min_score)
        audit["shapeConsensus"] = review
        if review["corroborated"]:
            audit.update(templateStatus=audit["status"], templateReason=audit.get("reason"),
                         status="MATCH_BY_SHAPE_CONSENSUS", reason=review["reason"])
        elif calibration_location is not None:
            from .calibrated_match_location import review_calibrated_location
            location_review = review_calibrated_location(image.shape,particle,pixel_size_um,audit,
                calibration_location=calibration_location,stage_center_um=stage_center_um,
                field_rotation_deg=field_rotation_deg,pixel_axis_matrix=pixel_axis_matrix,
                minimum_template_score=match_settings.match_min_score)
            audit["calibratedLocation"] = location_review
            if location_review["accepted"]:
                audit.update(templateStatus=audit["status"],templateReason=audit.get("reason"),
                             status="MATCH_BY_CALIBRATED_LOCATION",reason=location_review["reason"])
    if audit["status"] not in {"MATCH", "MATCH_BY_SHAPE_CONSENSUS", "MATCH_BY_CALIBRATED_LOCATION"}:
        return first, replace(decision, status="FAILED_MASK_IDENTITY", confidence="low",
                              reason=f"掩码定位复核未通过：{audit.get('reason') or audit['status']}；不能采谱"), audit
    if should_stop():
        raise InterruptedError("用户停止颗粒匹配")
    mask, optical_pixel, _ = optical_mask(particle)
    mh, mw = mask.shape
    scale = optical_pixel / pixel_size_um * audit["scaleFactor"]
    n = int(math.ceil(math.hypot(mh, mw) * scale)) + 20
    warp = cv2.getRotationMatrix2D(((mw-1)/2, (mh-1)/2), audit["angleDeg"], scale)
    warp[:, 2] += np.array([(n-mw)/2, (n-mh)/2])
    warp[:, 2] += np.asarray(audit["centerPixel"]) - (n-1)/2
    reference = cv2.warpAffine(np.fliplr(mask).astype(np.float32) if audit["mirrored"] else mask.astype(np.float32),
                               warp, (image.shape[1], image.shape[0])) > .5
    yy, xx = np.nonzero(reference)
    if len(xx) == 0:
        return first, replace(decision, status="FAILED_MASK_IDENTITY", confidence="low", reason="模板超出视场"), audit
    padding = max(16, int(math.ceil(particle.width_um/pixel_size_um)))
    x0, y0 = max(0, int(xx.min())-padding), max(0, int(yy.min())-padding)
    x1, y1 = min(image.shape[1], int(xx.max())+padding+1), min(image.shape[0], int(yy.max())+padding+1)
    audit["searchCropXyxy"] = [x0,y0,x1,y1]
    local = build_sem_anchor_candidates(particle, image[y0:y1,x0:x1], pixel_size_um=pixel_size_um,
        stage_center_um=(0,0), maximum_candidates=64, segmentation=segmentation, gate=gate)
    pool = list(candidates)
    for c in local:
        item = c.candidate
        center = (item.center[0]+x0,item.center[1]+y0)
        contour = tuple((x+x0,y+y0) for x,y in item.contour)
        box = (item.box[0]+x0,item.box[1]+y0,item.box[2]+x0,item.box[3]+y0)
        observed = replace(item, center=center, contour=contour, box=box,
                           distance_px=float(np.linalg.norm(np.subtract(center,np.array(image.shape[:2][::-1])/2))))
        pool.append(replace(c,candidate=observed,pixel_xy=center,
            stage_xy_um=pixel_to_stage_xy(center,image.shape,pixel_size_um,stage_center_um,
                                         field_rotation_deg=field_rotation_deg,pixel_axis_matrix=pixel_axis_matrix)))
    scored = []
    shape_center = None
    shape_center_tolerance = max(8.0, particle.width_um/pixel_size_um*.25)
    if audit["status"] == "MATCH_BY_SHAPE_CONSENSUS":
        shape_center = np.asarray(audit["shapeConsensus"]["targets"][0]["yoloCenterPx"], float)
        audit["maximumConsensusCenterDifferencePx"] = shape_center_tolerance
    elif audit["status"] == "MATCH_BY_CALIBRATED_LOCATION":
        shape_center = np.asarray(audit["centerPixel"],float)
        audit["maximumConsensusCenterDifferencePx"] = shape_center_tolerance
    for c in pool:
        measured = np.zeros(image.shape[:2],np.uint8)
        cv2.fillPoly(measured,[np.asarray(c.candidate.contour,np.int32)],1)
        intersection = int(np.logical_and(reference,measured).sum())
        union = int(np.logical_or(reference,measured).sum())
        iou = intersection/max(union,1)
        d = _decision_for_candidate(c,len(pool),reason_prefix="掩码定位后从 SEM 像素分割",gate=gate)
        agrees_with_review = shape_center is None or np.linalg.norm(np.subtract(c.pixel_xy,shape_center)) <= shape_center_tolerance
        scored.append((d.matched and iou>=overlap_floor and agrees_with_review,iou,c,d))
    if not scored:
        return first, replace(decision,status="FAILED_MASK_IDENTITY",confidence="low",
                              reason="模板位置没有可用 SEM 轮廓"), audit
    accepted,iou,chosen,decision = max(scored,key=lambda x:(x[0],x[1]))
    audit.update(accepted=bool(accepted),maskIoU=iou,selectedCenterPx=list(chosen.pixel_xy),
                 candidateCount=len(pool),relocated=bool(first is None or np.linalg.norm(np.subtract(first.pixel_xy,chosen.pixel_xy))>5))
    if shape_center is not None:
        audit["consensusCenterDifferencePx"] = float(np.linalg.norm(np.subtract(chosen.pixel_xy,shape_center)))
    reason = f"{decision.reason}；模板与 SEM 轮廓 IoU={iou:.3f}"
    if not accepted:
        reason += "；轮廓一致性或物理门槛未通过"
    return chosen, replace(decision,status=decision.status if accepted else "FAILED_MASK_IDENTITY",
                            confidence=decision.confidence if accepted else "low",reason=reason), audit


def _save_match_images(
    task_dir: Path,
    particle: OpticalParticle,
    image: np.ndarray,
    candidate: SemAnchorCandidate,
    candidate_count: int,
    prefix: str,
    reason: str,
    gate: CandidateGate | None = None,
) -> tuple[dict[str, Any], MatchDecision]:
    decision = _decision_for_candidate(
        candidate,
        candidate_count,
        reason_prefix=reason,
        gate=gate,
    )
    annotated_path = task_dir / f"{prefix}_annotated" / f"{particle.sequence:04d}.png"
    crop_path = task_dir / f"{prefix}_particles" / f"{particle.sequence:04d}.png"
    outputs = annotate_sem_match(
        image,
        particle,
        decision,
        annotated_path,
        crop_path,
    )
    return outputs, decision


def _save_center_sampling_images(task_dir: Path, particle: OpticalParticle, image: np.ndarray, plan: dict) -> dict:
    """Show a sampling region/point without inventing a matched particle contour."""
    annotated_path=task_dir/'analysis_annotated'/f'{particle.sequence:04d}.png'
    crop_path=task_dir/'analysis_particles'/f'{particle.sequence:04d}.png'
    annotated_path.parent.mkdir(parents=True,exist_ok=True)
    crop_path.parent.mkdir(parents=True,exist_ok=True)
    gray=normalize_uint8(image)
    marked=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    x0,y0,x1,y1=plan['centerRegionXyxy']
    point=tuple(plan['pointPx'])
    cv2.rectangle(marked,(x0,y0),(x1-1,y1-1),(255,210,60),1)
    cv2.drawMarker(marked,tuple(plan['imageCenterPx']),(0,220,255),cv2.MARKER_CROSS,22,1)
    cv2.drawMarker(marked,point,(255,80,220),cv2.MARKER_TILTED_CROSS,20,2)
    for path,array in [(annotated_path,marked),(crop_path,gray[y0:y1,x0:x1])]:
        ok,encoded=cv2.imencode('.png',array)
        if not ok:raise RuntimeError("中心采谱标注图写入失败")
        path.write_bytes(encoded.tobytes())
    return {'annotatedImage':annotated_path,'particleImage':crop_path,'safePointPx':list(point),
            'safeBoundaryDistancePx':None}


def _calibration_record(
    task_dir: Path,
    particle: OpticalParticle,
    predicted: tuple[float, float],
    move: dict[str, Any],
    capture: dict[str, Any],
    candidate: SemAnchorCandidate,
    candidate_count: int,
    image: np.ndarray,
    prefix: str,
    reason: str,
    capture_attempts: Sequence[dict[str, Any]] = (),
    gate: CandidateGate | None = None,
) -> dict[str, Any]:
    outputs, decision = _save_match_images(
        task_dir,
        particle,
        image,
        candidate,
        candidate_count,
        prefix,
        reason,
        gate,
    )
    return {
        "sequence": particle.sequence,
        "particleId": particle.particle_id,
        "opticalAbUm": [particle.ab_x_um, particle.ab_y_um],
        "predictedStageUm": {"x": predicted[0], "y": predicted[1]},
        "captureStageUm": move.get("after"),
        "stagePositionErrorUm": move.get("positionErrorUm"),
        "observedStageUm": {
            "x": candidate.stage_xy_um[0],
            "y": candidate.stage_xy_um[1],
        },
        "match": decision.to_dict(),
        "shapeScore": candidate.shape_score,
        "detectorScore": candidate.detector_score,
        "captureAttempts": list(capture_attempts),
        "images": {
            "opticalImage": f"optical/{particle.sequence:04d}.png",
            "semRawImage": _relative(Path(capture["rawPath"]), task_dir),
            "semDisplayImage": _relative(Path(capture["displayPath"]), task_dir),
            "semAnnotatedImage": _relative(Path(outputs["annotatedImage"]), task_dir),
            "semParticleImage": _relative(Path(outputs["particleImage"]), task_dir),
        },
    }


def _restore_and_maybe_gun_off(
    microscope: Any,
    start_position: tuple[float, float] | None,
    *,
    restore_position: bool,
    gun_off: bool,
) -> dict[str, Any]:
    cleanup: dict[str, Any] = {}
    if restore_position and start_position is not None:
        try:
            cleanup["stageRestore"] = microscope.move_xy(*start_position)
        except Exception as error:
            cleanup["stageRestore"] = {"error": str(error)}
    try:
        microscope.restore_video()
    except Exception as error:
        cleanup["videoRestoreError"] = str(error)
    cleanup.update(gunOffRequested=bool(gun_off), gunOffVerified=False, gunOffStatus=None, gunOffError=None)
    if gun_off:
        cleanup["gunOffVerified"] = False
        try:
            if not callable(getattr(microscope, "gun_off", None)):
                raise RuntimeError("设备驱动不支持关枪")
            cleanup["gunOffStatus"] = str(microscope.gun_off())
            from .microscope_safety import hv_status_name
            if hv_status_name(cleanup["gunOffStatus"]) != "close":
                raise RuntimeError(f"关枪未确认：{cleanup['gunOffStatus']}")
            cleanup["gunOffVerified"] = True
        except Exception as error:
            cleanup["gunOffError"] = str(error)
    return cleanup


def _move_stage_checked(
    microscope: Any,
    target: tuple[float, float],
    *,
    maximum_error_um: float = 2.0,
    raise_on_excess: bool = True,
) -> dict[str, Any]:
    move = microscope.move_xy(*target)
    after = _stage_xy(move.get("after"))
    error_um = float(
        move.get("positionErrorUm")
        if move.get("positionErrorUm") is not None
        else math.hypot(after[0] - target[0], after[1] - target[1])
    )
    move["positionErrorUm"] = error_um
    if error_um > float(maximum_error_um):
        warning = f"样品台回读误差超过 {maximum_error_um:.1f} μm：{error_um:.3f} μm"
        move["warning"] = warning
        if raise_on_excess:
            raise RuntimeError(warning)
    return move


def _require_target_inside_roi(
    experience: Any,
    target: tuple[float, float],
    *,
    particle: OpticalParticle,
) -> None:
    if not experience.target_inside_roi(target):
        raise RuntimeError(
            f"颗粒 #{particle.sequence} 的预测样品台坐标超出经验标定有效 ROI："
            f"({target[0]:.3f}, {target[1]:.3f}) μm"
        )


def _capture_candidates_with_retries(
    microscope: Any,
    task_dir: Path,
    folder: str,
    particle: OpticalParticle,
    stage_center: tuple[float, float],
    experience: Any,
    *,
    field_rotation: float,
    axis_matrix: Sequence[Sequence[float]],
    maximum_candidates: int,
    maximum_attempts: int,
    baseline_signature: dict[str, Any] | None,
    should_stop: StopCallback,
    segmentation: CandidateSegmentationConfig | None = None,
    gate: CandidateGate | None = None,
    event_logger: StageEventCallback | None = None,
    particle_rank: int | None = None,
) -> tuple[
    np.ndarray,
    dict[str, Any],
    tuple[SemAnchorCandidate, ...],
    dict[str, Any],
    list[dict[str, Any]],
]:
    attempts: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for attempt in range(1, max(1, int(maximum_attempts)) + 1):
        if should_stop():
            raise InterruptedError("用户停止了清洁度检测任务")
        raw_path = (
            task_dir
            / folder
            / f"{particle.sequence:04d}_attempt{attempt}.png"
        )
        capture_started = time.monotonic()
        capture_finished = False
        detection_started: float | None = None
        detection_finished = False
        _emit_stage_event(
            event_logger,
            "calibration_stage_started",
            calibrationMode="multi_particle",
            stage="capture",
            particleSequence=particle.sequence,
            particleRank=particle_rank,
            attempt=attempt,
        )
        try:
            image, capture = microscope.capture(raw_path)
            _validate_calibrated_capture(capture, experience)
            signature = _capture_signature(capture)
            if baseline_signature is not None:
                _validate_same_signature(baseline_signature, signature)
            capture_finished = True
            _emit_stage_event(
                event_logger,
                "calibration_stage_finished",
                calibrationMode="multi_particle",
                stage="capture",
                particleSequence=particle.sequence,
                particleRank=particle_rank,
                attempt=attempt,
                status="succeeded",
                elapsedSeconds=_seconds_since(capture_started),
                imageShape=list(image.shape),
                pixelSizeUm=capture.get("pixelSizeUm"),
            )
            detection_started = time.monotonic()
            _emit_stage_event(
                event_logger,
                "calibration_stage_started",
                calibrationMode="multi_particle",
                stage="candidate_detection",
                particleSequence=particle.sequence,
                particleRank=particle_rank,
                attempt=attempt,
            )
            candidates = build_sem_anchor_candidates(
                particle,
                image,
                pixel_size_um=float(capture["pixelSizeUm"]),
                stage_center_um=stage_center,
                field_rotation_deg=field_rotation,
                pixel_axis_matrix=axis_matrix,
                maximum_candidates=maximum_candidates,
                segmentation=segmentation,
                gate=gate,
            )
            if not candidates:
                raise RuntimeError("SEM 视场中没有找到颗粒候选")
            detection_finished = True
            _emit_stage_event(
                event_logger,
                "calibration_stage_finished",
                calibrationMode="multi_particle",
                stage="candidate_detection",
                particleSequence=particle.sequence,
                particleRank=particle_rank,
                attempt=attempt,
                status="succeeded",
                elapsedSeconds=_seconds_since(detection_started),
                candidateCount=len(candidates),
            )
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "SUCCEEDED",
                    "rawPath": _relative(Path(capture["rawPath"]), task_dir),
                    "displayPath": _relative(
                        Path(capture["displayPath"]), task_dir
                    ),
                    "candidateCount": len(candidates),
                }
            )
            return image, capture, candidates, signature, attempts
        except InterruptedError as error:
            if not capture_finished:
                _emit_stage_event(
                    event_logger,
                    "calibration_stage_finished",
                    calibrationMode="multi_particle",
                    stage="capture",
                    particleSequence=particle.sequence,
                    particleRank=particle_rank,
                    attempt=attempt,
                    status="stopped",
                    elapsedSeconds=_seconds_since(capture_started),
                    error=str(error),
                )
            if detection_started is not None and not detection_finished:
                _emit_stage_event(
                    event_logger,
                    "calibration_stage_finished",
                    calibrationMode="multi_particle",
                    stage="candidate_detection",
                    particleSequence=particle.sequence,
                    particleRank=particle_rank,
                    attempt=attempt,
                    status="stopped",
                    elapsedSeconds=_seconds_since(detection_started),
                    error=str(error),
                )
            raise
        except Exception as error:
            if not capture_finished:
                _emit_stage_event(
                    event_logger,
                    "calibration_stage_finished",
                    calibrationMode="multi_particle",
                    stage="capture",
                    particleSequence=particle.sequence,
                    particleRank=particle_rank,
                    attempt=attempt,
                    status="failed",
                    elapsedSeconds=_seconds_since(capture_started),
                    error=str(error),
                )
            if detection_started is not None and not detection_finished:
                _emit_stage_event(
                    event_logger,
                    "calibration_stage_finished",
                    calibrationMode="multi_particle",
                    stage="candidate_detection",
                    particleSequence=particle.sequence,
                    particleRank=particle_rank,
                    attempt=attempt,
                    status="failed",
                    elapsedSeconds=_seconds_since(detection_started),
                    error=str(error),
                )
            last_error = error
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "FAILED",
                    "error": str(error),
                }
            )
    raise RuntimeError(
        f"颗粒 #{particle.sequence} 连续 {len(attempts)} 次采图/识别失败："
        f"{last_error or '未知错误'}"
    ) from last_error


def run_cleanliness_calibration(
    config: CleanlinessTaskConfig,
    *,
    microscope: Any | None = None,
    progress: ProgressCallback | None = None,
    should_stop: StopCallback | None = None,
    event_logger: StageEventCallback | None = None,
) -> dict[str, Any]:
    progress = progress or (lambda _index, _total, _message: None)
    should_stop = should_stop or (lambda: False)
    settings_path = Path(config.settings_path).resolve()
    settings = load_cleanliness_config(settings_path)
    calibration_settings = settings.get("calibration") or {}
    segmentation = _candidate_segmentation_config(
        settings_path,
        calibration_settings,
        sam3_url=config.sam3_url,
    )
    host = str(
        _config_value(
            config.microscope_host,
            settings,
            "device",
            "microscopeHost",
            "192.168.0.15",
        )
    )
    port = int(
        _config_value(
            config.microscope_port,
            settings,
            "device",
            "microscopePort",
            61557,
        )
    )
    restore_position = bool(
        _config_value(
            config.restore_start_position,
            settings,
            "task",
            "restoreStartPosition",
            True,
        )
    )
    auto_gun_on = bool(
        _config_value(config.auto_gun_on, settings, "task", "gunOnAtStart", False)
    )
    gun_off_after_phase = bool(
        _config_value(
            config.gun_off_after_task,
            settings,
            "task",
            "gunOffAfterTask",
            True,
        )
    )
    eds_host = str(
        _config_value(config.eds_host, settings, "device", "edsHost", host)
    )
    eds_port = int(
        _config_value(config.eds_port, settings, "device", "edsPort", 6200)
    )
    experience_path = (
        Path(config.experience_calibration_path).resolve()
        if config.experience_calibration_path is not None
        else _resolve_experience_path(settings_path, settings)
    )
    task_dir = Path(config.output_dir).resolve()
    task_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = task_dir / "task.json"
    if manifest_path.exists():
        raise FileExistsError(f"任务目录已经存在 task.json：{task_dir}")

    task_settings = settings.get("task") or {}
    export = load_optical_export(config.input_path)
    experience = calibration_with_image_size(load_calibration(experience_path), task_settings)
    experience.validate_export(export)
    selected = (
        _select_by_sequences(export.particles, config.selected_sequences)
        if config.selected_sequences is not None
        else filter_optical_particles(export.particles, config.particle_filter)
    )
    if not selected:
        raise ValueError("当前筛选条件下没有待检测颗粒")

    group_count = int(calibration_settings.get("groupCount") or 3)
    model = str(calibration_settings.get("model") or "reflected_similarity")
    gate = CandidateGate.from_dict(calibration_settings.get("candidateGate"))

    if config.calibration_groups:
        groups = []
        for g in config.calibration_groups:
            a_seqs = tuple(g.get("anchorSequences") or ())
            v_seqs = tuple(g.get("validationSequences") or ())
            anchors_g = _select_by_sequences(export.particles, a_seqs)
            if len(anchors_g) < 3:
                raise ValueError(f"标定组锚点不足 3 个（实际 {len(anchors_g)} 个）")
            validation_g = _select_by_sequences(export.particles, v_seqs) if v_seqs else []
            groups.append((anchors_g, validation_g))
    elif config.anchor_sequences:
        anchors = _select_by_sequences(export.particles, config.anchor_sequences)
        if len(anchors) < 3:
            raise ValueError("人工指定的标定颗粒至少需要三个")
        used = {p.sequence for p in anchors}
        if config.validation_sequences:
            validation_particles = _select_by_sequences(
                export.particles, config.validation_sequences,
            )
        else:
            validation_particles = select_validation_landmarks(
                export, anchors,
                validation_count=int(calibration_settings.get("validationCount") or 2),
                pool_size=int(calibration_settings.get("anchorPoolSize") or 30),
                excluded_sequences=used,
            )
        groups = [(anchors, validation_particles)]
    else:
        groups = select_calibration_groups(
            export,
            group_count=group_count,
            anchor_count=int(calibration_settings.get("anchorCount") or 3),
            validation_count=int(calibration_settings.get("validationCount") or 2),
            pool_size=int(calibration_settings.get("anchorPoolSize") or 30),
        )

    all_particles_map: dict[int, OpticalParticle] = {}
    for anchors_g, validation_g in groups:
        for p in (*anchors_g, *validation_g):
            all_particles_map[p.sequence] = p
    all_cal_particles = list(all_particles_map.values())

    all_images = {p.sequence: p for p in selected}
    all_images.update(all_particles_map)
    for particle in all_images.values():
        decode_optical_image(particle, task_dir / "optical" / f"{particle.sequence:04d}.png")

    manifest: dict[str, Any] = {
        "schemaVersion": 2,
        "taskType": "CLEANLINESS_DETECTION",
        "taskId": str(uuid.uuid4()),
        "status": "CALIBRATING",
        "startedAt": _now(),
        "completedAt": None,
        "input": {
            "path": str(export.path),
            "sha256": export.sha256,
            "projectId": export.project_id,
            "workpieceId": export.workpiece_id,
            "workpieceName": export.workpiece_name,
            "particleCount": len(export.particles),
        },
        "selection": {
            "filter": config.particle_filter.to_dict(),
            "count": len(selected),
            "sequences": [p.sequence for p in selected],
        },
        "device": {
            "microscope": {"host": host, "port": port},
            "eds": {"host": eds_host, "port": eds_port},
        },
        "experienceCalibration": experience.to_audit_dict(),
        "calibration": {
            "model": model,
            "candidateSegmentation": segmentation.to_dict(),
            "groups": [],
            "selectedGroup": None,
            "review": None,
        },
        "particles": [],
        "reports": None,
        "cleanup": {},
        "error": None,
    }
    _write_json_atomic(manifest_path, manifest)

    microscope = microscope or ZemMicroscope(
        host, port, experience.imaging, pixel_avg=config.pixel_avg,
    )
    start_position: tuple[float, float] | None = None
    total = len(all_cal_particles)
    calibration_started = time.monotonic()
    active_group: tuple[int, float] | None = None
    _emit_stage_event(
        event_logger,
        "calibration_stage_started",
        calibrationMode="multi_particle",
        stage="calibration",
        particleCount=total,
        attempt=None,
    )
    try:
        progress(0, total, f"连接电镜 {host}:{port}")
        connection_started = time.monotonic()
        _emit_stage_event(
            event_logger,
            "calibration_stage_started",
            calibrationMode="multi_particle",
            stage="connect",
            attempt=1,
        )
        try:
            connection = microscope.connect()
        except Exception as error:
            _emit_stage_event(
                event_logger,
                "calibration_stage_finished",
                calibrationMode="multi_particle",
                stage="connect",
                attempt=1,
                status="failed",
                elapsedSeconds=_seconds_since(connection_started),
                error=str(error),
            )
            raise
        _emit_stage_event(
            event_logger,
            "calibration_stage_finished",
            calibrationMode="multi_particle",
            stage="connect",
            attempt=1,
            status="succeeded",
            elapsedSeconds=_seconds_since(connection_started),
        )
        manifest["device"]["microscope"]["connection"] = connection
        start_position = microscope.get_position()
        manifest["startStageUm"] = {"x": start_position[0], "y": start_position[1]}
        route_started = time.monotonic()
        _emit_stage_event(
            event_logger,
            "calibration_stage_started",
            calibrationMode="multi_particle",
            stage="route_planning",
            particleCount=total,
            attempt=None,
        )
        try:
            route_plan = optimize_route(
                all_cal_particles,
                experience.target_stage,
                start_position,
                return_to_start=restore_position,
            )
        except Exception as error:
            _emit_stage_event(
                event_logger,
                "calibration_stage_finished",
                calibrationMode="multi_particle",
                stage="route_planning",
                particleCount=total,
                attempt=None,
                status="failed",
                elapsedSeconds=_seconds_since(route_started),
                error=str(error),
            )
            raise
        all_cal_particles = list(route_plan.items)
        manifest["calibration"]["executionOrder"] = [
            particle.sequence for particle in all_cal_particles
        ]
        manifest["calibration"]["executionRoute"] = route_plan.to_audit_dict()
        _emit_stage_event(
            event_logger,
            "calibration_stage_finished",
            calibrationMode="multi_particle",
            stage="route_planning",
            particleCount=total,
            attempt=None,
            status="succeeded",
            elapsedSeconds=_seconds_since(route_started),
            **route_plan.to_audit_dict(),
        )
        _write_json_atomic(manifest_path, manifest)
        if auto_gun_on and hasattr(microscope, "gun_on"):
            progress(0, total, "开启电子枪并等待高压稳定")
            manifest["device"]["microscope"]["gunOnStatus"] = str(microscope.gun_on())

        # Phase 1: 一次性采集所有标定/验证颗粒的 SEM
        axis_matrix = calibration_settings.get("pixelAxisMatrix") or ((1.0, 0.0), (0.0, 1.0))
        maximum_candidates = int(calibration_settings.get("semCandidatesPerAnchor") or 6)
        field_rotation = float(experience.imaging.get("fieldRotationDeg") or 0.0)
        baseline_signature: dict[str, Any] | None = None
        candidates_by_sequence: dict[int, tuple[SemAnchorCandidate, ...]] = {}
        captures_by_sequence: dict[int, tuple[np.ndarray, dict[str, Any], dict[str, Any], tuple[float, float], list[dict[str, Any]]]] = {}

        for index, particle in enumerate(all_cal_particles, start=1):
            particle_started = time.monotonic()
            _emit_stage_event(
                event_logger,
                "calibration_stage_started",
                calibrationMode="multi_particle",
                stage="particle_acquisition",
                particleSequence=particle.sequence,
                particleRank=index,
                attempt=None,
            )
            try:
                if should_stop():
                    raise InterruptedError("用户停止了标定任务")
                predicted = experience.target_stage(particle)
                _require_target_inside_roi(experience, predicted, particle=particle)
                progress(index - 1, total, f"[{index}/{total}] 采集颗粒 #{particle.sequence} SEM")
                move_started = time.monotonic()
                _emit_stage_event(
                    event_logger,
                    "calibration_stage_started",
                    calibrationMode="multi_particle",
                    stage="stage_move",
                    particleSequence=particle.sequence,
                    particleRank=index,
                    attempt=1,
                    targetStageUm={"x": predicted[0], "y": predicted[1]},
                )
                try:
                    move = _move_stage_checked(microscope, predicted)
                except Exception as error:
                    _emit_stage_event(
                        event_logger,
                        "calibration_stage_finished",
                        calibrationMode="multi_particle",
                        stage="stage_move",
                        particleSequence=particle.sequence,
                        particleRank=index,
                        attempt=1,
                        status="stopped" if isinstance(error, InterruptedError) else "failed",
                        elapsedSeconds=_seconds_since(move_started),
                        error=str(error),
                    )
                    raise
                _emit_stage_event(
                    event_logger,
                    "calibration_stage_finished",
                    calibrationMode="multi_particle",
                    stage="stage_move",
                    particleSequence=particle.sequence,
                    particleRank=index,
                    attempt=1,
                    status="succeeded",
                    elapsedSeconds=_seconds_since(move_started),
                    actualStageUm=move.get("after"),
                    positionErrorUm=move.get("positionErrorUm"),
                )
                stage_center = _stage_xy(move["after"])
                image, capture, candidates, signature, capture_attempts = (
                    _capture_candidates_with_retries(
                        microscope, task_dir, "calibration_sem", particle, stage_center,
                        experience, field_rotation=field_rotation, axis_matrix=axis_matrix,
                        maximum_candidates=maximum_candidates,
                        maximum_attempts=config.max_capture_attempts,
                        baseline_signature=baseline_signature, should_stop=should_stop,
                        segmentation=segmentation, gate=gate,
                        event_logger=event_logger, particle_rank=index,
                    )
                )
            except Exception as error:
                _emit_stage_event(
                    event_logger,
                    "calibration_stage_finished",
                    calibrationMode="multi_particle",
                    stage="particle_acquisition",
                    particleSequence=particle.sequence,
                    particleRank=index,
                    attempt=None,
                    status="stopped" if isinstance(error, InterruptedError) else "failed",
                    elapsedSeconds=_seconds_since(particle_started),
                    error=str(error),
                )
                raise
            if baseline_signature is None:
                baseline_signature = signature
                manifest["baselineImaging"] = signature
            candidates_by_sequence[particle.sequence] = candidates
            captures_by_sequence[particle.sequence] = (image, capture, move, predicted, capture_attempts)
            _emit_stage_event(
                event_logger,
                "calibration_stage_finished",
                calibrationMode="multi_particle",
                stage="particle_acquisition",
                particleSequence=particle.sequence,
                particleRank=index,
                attempt=None,
                status="succeeded",
                elapsedSeconds=_seconds_since(particle_started),
                candidateCount=len(candidates),
            )

        progress(total, total, "SEM 采集完成，正在逐组拟合标定矩阵")

        # Phase 2: 逐组拟合
        acceptance = calibration_settings.get("acceptance") or {}
        mean_limit = float(acceptance.get("meanErrorUm") or math.inf)
        max_limit = float(acceptance.get("maxErrorUm") or math.inf)
        residual_scale_um = float(
            calibration_settings.get("residualScaleUm") or 60.0
        )

        for group_index, (anchors_g, validation_g) in enumerate(groups):
            group_started = time.monotonic()
            active_group = (group_index, group_started)
            _emit_stage_event(
                event_logger,
                "calibration_stage_started",
                calibrationMode="multi_particle",
                stage="matrix_fit",
                groupIndex=group_index,
                anchorSequences=[particle.sequence for particle in anchors_g],
                validationSequences=[particle.sequence for particle in validation_g],
                attempt=None,
            )
            group_data: dict[str, Any] = {
                "groupIndex": group_index,
                "anchorSequences": [p.sequence for p in anchors_g],
                "validationSequences": [p.sequence for p in validation_g],
                "anchors": [],
                "validation": [],
                "result": None,
            }
            match_result: CalibrationMatchResult = choose_calibration_matches(
                anchors_g, candidates_by_sequence, model=model, experience=experience,
                gate=gate, residual_scale_um=residual_scale_um,
            )
            prefix = f"g{group_index}_cal"
            for particle, chosen in zip(anchors_g, match_result.matches, strict=True):
                image, capture, move, predicted, capture_attempts = captures_by_sequence[particle.sequence]
                group_data["anchors"].append(
                    _calibration_record(
                        task_dir, particle, predicted, move, capture, chosen,
                        len(candidates_by_sequence[particle.sequence]),
                        image, prefix,
                        "三颗粒形貌与三角几何联合匹配", capture_attempts, gate,
                    )
                )

            solution = match_result.solution
            validation_source: list[list[float]] = []
            validation_observed: list[tuple[float, float]] = []
            val_prefix = f"g{group_index}_val"
            for particle in validation_g:
                cands = candidates_by_sequence.get(particle.sequence)
                if not cands:
                    continue
                chosen = next(
                    (item for item in cands if gate.accepts(item.candidate)),
                    cands[0],
                )
                image, capture, move, predicted, capture_attempts = captures_by_sequence[particle.sequence]
                group_data["validation"].append(
                    _calibration_record(
                        task_dir, particle, predicted, move, capture, chosen,
                        len(cands), image, val_prefix,
                        "独立验证集形貌匹配", capture_attempts, gate,
                    )
                )
                validation_source.append([particle.ab_x_um, particle.ab_y_um])
                validation_observed.append(chosen.stage_xy_um)

            validation_metrics = evaluate_transform(solution, validation_source, validation_observed)
            unreliable = [
                record["sequence"]
                for record in (*group_data["anchors"], *group_data["validation"])
                if not (record.get("match") or {}).get("status", "").startswith("MATCHED")
            ]
            within_limits = bool(
                validation_metrics["count"] > 0
                and not unreliable
                and float(validation_metrics["meanErrorUm"]) <= mean_limit
                and float(validation_metrics["maxErrorUm"]) <= max_limit
            )
            group_data["unreliableSequences"] = unreliable
            group_data["result"] = {
                **solution.to_dict(),
                "hypothesisScore": match_result.hypothesis_score,
                "validation": validation_metrics,
                "unreliableSequences": unreliable,
                "acceptance": {
                    "meanErrorLimitUm": mean_limit,
                    "maxErrorLimitUm": max_limit,
                    "withinLimits": within_limits,
                },
            }
            manifest["calibration"]["groups"].append(group_data)
            _emit_stage_event(
                event_logger,
                "calibration_stage_finished",
                calibrationMode="multi_particle",
                stage="matrix_fit",
                groupIndex=group_index,
                attempt=None,
                status="accepted" if within_limits else "rejected",
                elapsedSeconds=_seconds_since(group_started),
                meanErrorUm=validation_metrics.get("meanErrorUm"),
                maxErrorUm=validation_metrics.get("maxErrorUm"),
                unreliableSequences=unreliable,
            )
            active_group = None

        manifest["status"] = "CALIBRATION_PENDING_REVIEW"
        _write_json_atomic(task_dir / "calibration_result.json", manifest["calibration"])
    except InterruptedError as error:
        if active_group is not None:
            group_index, group_started = active_group
            _emit_stage_event(
                event_logger,
                "calibration_stage_finished",
                calibrationMode="multi_particle",
                stage="matrix_fit",
                groupIndex=group_index,
                attempt=None,
                status="stopped",
                elapsedSeconds=_seconds_since(group_started),
                error=str(error),
            )
        manifest["status"] = "STOPPED"
        manifest["error"] = str(error)
    except Exception as error:
        if active_group is not None:
            group_index, group_started = active_group
            _emit_stage_event(
                event_logger,
                "calibration_stage_finished",
                calibrationMode="multi_particle",
                stage="matrix_fit",
                groupIndex=group_index,
                attempt=None,
                status="failed",
                elapsedSeconds=_seconds_since(group_started),
                error=str(error),
            )
        manifest["status"] = "CALIBRATION_FAILED"
        manifest["error"] = str(error)
    finally:
        manifest["cleanup"].update(
            _restore_and_maybe_gun_off(
                microscope, start_position,
                restore_position=restore_position, gun_off=gun_off_after_phase,
            )
        )
        _emit_stage_event(
            event_logger,
            "calibration_stage_finished",
            calibrationMode="multi_particle",
            stage="calibration",
            particleCount=total,
            attempt=None,
            status=str(manifest.get("status") or "unknown").lower(),
            elapsedSeconds=_seconds_since(calibration_started),
            error=manifest.get("error"),
        )
        _write_json_atomic(manifest_path, manifest)
    return manifest


def review_cleanliness_calibration(
    task_dir: Path,
    *,
    accepted: bool,
    selected_group: int = 0,
    reviewer: str = "user",
    note: str = "",
) -> dict[str, Any]:
    task_dir = Path(task_dir).resolve()
    manifest_path = task_dir / "task.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "CALIBRATION_PENDING_REVIEW":
        raise ValueError(f"当前任务状态不能审核标定：{manifest.get('status')}")
    calibration = manifest["calibration"]
    if accepted:
        if calibration.get("strategy") == "ab_particle":
            refinement = (calibration.get("abPointResult") or {}).get("semRefinement") or {}
            if (refinement.get("status") != "SUCCEEDED"
                    or not (refinement.get("acceptance") or {}).get("withinLimits")
                    or not (calibration.get("result") or {}).get("acceptance", {}).get("withinLimits")):
                raise ValueError("颗粒精校准未通过独立数值验收，不能确认标定")
            if load_optical_export(Path(manifest["input"]["path"])).sha256 != manifest["input"]["sha256"]:
                raise ValueError("光镜 JSON 已变化，请重新标定")
        group_list = calibration.get("groups") or []
        if group_list:
            if not (0 <= selected_group < len(group_list)):
                raise ValueError(f"所选标定组索引无效：{selected_group}，共 {len(group_list)} 组")
            calibration["selectedGroup"] = int(selected_group)
        else:
            calibration["selectedGroup"] = 0
    calibration["review"] = {
        "accepted": bool(accepted),
        "selectedGroup": int(selected_group) if accepted else None,
        "reviewer": str(reviewer),
        "reviewedAt": _now(),
        "note": str(note),
    }
    manifest["status"] = "CALIBRATION_ACCEPTED" if accepted else "CALIBRATION_REJECTED"
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _optimized_particle_route(
    particles: Sequence[OpticalParticle],
    solution: TransformSolution,
    start_position: tuple[float, float],
    *,
    return_to_start: bool,
) -> RoutePlan[OpticalParticle]:
    return optimize_route(
        particles,
        lambda particle: solution.transform_xy(particle.ab_x_um, particle.ab_y_um),
        start_position,
        return_to_start=return_to_start,
    )


def _shape_metrics(candidate: SemAnchorCandidate) -> dict[str, float]:
    contour = np.asarray(candidate.candidate.contour, dtype=np.int32)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    (_, _), (width, height), _ = cv2.minAreaRect(contour)
    long_side = max(float(width), float(height), 1.0)
    short_side = max(min(float(width), float(height)), 1.0)
    return {
        "aspect": long_side / short_side,
        "roundness": (
            4.0 * math.pi * area / (perimeter * perimeter)
            if perimeter > 0
            else 0.0
        ),
    }


def _is_transient_eds_error(error: Exception) -> bool:
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    message = str(error).casefold()
    transient_markers = (
        "stopedspectrumacquisition failed",
        "current acquisition time is 0.0",
        "deadline exceeded",
        "timed out",
        "timeout",
        "statuscode.unavailable",
        "connection reset",
        "connection refused",
        "transport is closing",
        "socket closed",
    )
    return any(marker in message for marker in transient_markers)


def run_cleanliness_analysis(
    task_dir: Path,
    *,
    config: CleanlinessTaskConfig | None = None,
    microscope: Any | None = None,
    spectrum_analyzer: SpectrumAnalyzer | None = None,
    progress: ProgressCallback | None = None,
    should_stop: StopCallback | None = None,
    event_logger: StageEventCallback | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    progress = progress or (lambda _index, _total, _message: None)
    should_stop = should_stop or (lambda: False)
    task_dir = Path(task_dir).resolve()
    manifest_path = task_dir / "task.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    initial_status = str(manifest.get("status") or "")
    calibration_review = (manifest.get("calibration") or {}).get("review") or {}
    if resume:
        if initial_status not in {"RUNNING", "STOPPED", "FAILED"}:
            raise ValueError(f"当前任务状态不能继续检测：{initial_status}")
        if not bool(calibration_review.get("accepted")):
            raise ValueError("必须先由用户确认本次标定结果")
    elif initial_status != "CALIBRATION_ACCEPTED":
        raise ValueError("必须先由用户确认本次标定结果")
    settings_path = (
        Path(config.settings_path).resolve()
        if config is not None
        else default_cleanliness_config_path().resolve()
    )
    settings = load_cleanliness_config(settings_path)
    task_settings = settings.get("task") or {}
    from .particle_refinement import RefinementSettings
    identity_settings = RefinementSettings.from_dict({
        "match_min_margin": (settings.get("abParticle") or {}).get("match_min_margin", 0.07),
        "match_min_score": (settings.get("abParticle") or {}).get("match_min_score", 0.65),
    })
    manifest["candidateScoreMargin"] = identity_settings.match_min_margin
    manifest["templateMatchMinScore"] = identity_settings.match_min_score
    contour_overlap_min = _contour_overlap_min(task_settings.get("contourOverlapMin", 0.5))
    manifest["contourOverlapMin"] = contour_overlap_min
    maximum_attempts = config.max_capture_attempts if config is not None else 2
    manifest["maxCaptureAttempts"] = maximum_attempts
    center_settings = LargeParticleCenterSettings.from_dict(task_settings.get("largeParticleCenterSampling"))
    manifest["largeParticleCenterSamplingPolicy"] = center_settings.to_dict()
    calibration_settings = settings.get("calibration") or {}
    segmentation = _candidate_segmentation_config(
        settings_path,
        calibration_settings,
        sam3_url=config.sam3_url if config is not None else None,
    )
    device = manifest["device"]
    microscope_host = str(device["microscope"]["host"])
    microscope_port = int(device["microscope"]["port"])
    eds_host = str(device["eds"]["host"])
    eds_port = int(device["eds"]["port"])
    eds_enabled = bool(
        config.eds_enabled
        if config is not None and config.eds_enabled is not None
        else task_settings.get("edsEnabled", True)
    )
    eds_retry_count = max(
        0,
        int(
            config.eds_retry_count
            if config is not None
            else task_settings.get("edsRetryCount", 0)
        ),
    )
    restore_position = bool(
        config.restore_start_position
        if config is not None and config.restore_start_position is not None
        else task_settings.get("restoreStartPosition", True)
    )
    auto_gun_on = bool(
        config.auto_gun_on
        if config is not None and config.auto_gun_on is not None
        else task_settings.get("gunOnAtStart", False)
    )
    gun_off_after_task = bool(
        config.gun_off_after_task
        if config is not None and config.gun_off_after_task is not None
        else task_settings.get("gunOffAfterTask", True)
    )
    export = load_optical_export(Path(manifest["input"]["path"]))
    is_particle_refinement = (manifest.get("calibration") or {}).get("strategy") == "ab_particle"
    if (resume or is_particle_refinement) and export.sha256 != str((manifest.get("input") or {}).get("sha256") or ""):
        raise ValueError("光镜 JSON 内容已变化，不能继续上次检测，请重新创建任务")
    if config is not None and config.selected_sequences:
        selected = _select_by_sequences(export.particles, config.selected_sequences)
        requested_sequences = [particle.sequence for particle in selected]
        saved_sequences = [
            int(value) for value in (manifest.get("selection") or {}).get("sequences") or []
        ]
        if resume and requested_sequences != saved_sequences:
            raise ValueError("颗粒筛选结果已变化，不能继续上次检测")
        if not resume:
            manifest["selection"]["sequences"] = requested_sequences
            manifest["selection"]["count"] = len(selected)
            manifest["selection"]["filter"] = config.particle_filter.to_dict()
    else:
        selected = _select_by_sequences(
            export.particles,
            manifest["selection"]["sequences"],
        )
    # AB-point calibration only materializes the few particles used for visual
    # validation. Reports, however, reference an optical image for every
    # analyzed particle. Ensure the complete selected set exists before any
    # hardware action so HTML/PDF export can never contain broken image links.
    for particle in selected:
        optical_path = task_dir / "optical" / f"{particle.sequence:04d}.png"
        if not optical_path.is_file():
            decode_optical_image(particle, optical_path)
    calibration = manifest["calibration"]
    group_list = calibration.get("groups") or []
    if group_list:
        selected_idx = int(calibration.get("selectedGroup") or 0)
        solution = transform_solution_from_dict(group_list[selected_idx]["result"])
    else:
        solution = transform_solution_from_dict(calibration["result"])
    experience = load_calibration(
        Path(manifest["experienceCalibration"]["path"])
    )
    if not is_particle_refinement:
        experience = calibration_with_image_size(experience, task_settings)
    elif task_settings.get("captureImageSize") is not None:
        requested_size = validate_capture_image_size(task_settings["captureImageSize"])
        if list(experience.imaging.get("imageShape") or ()) != [requested_size, requested_size]:
            raise ValueError("图幅与已冻结的颗粒精校准不一致，请重新标定")
    if is_particle_refinement:
        from .ab_particle_calibration import settings_fingerprint
        from .particle_refinement import RefinementSettings
        refinement = calibration["abPointResult"]["semRefinement"]
        if refinement.get("status") != "SUCCEEDED" or not refinement.get("acceptance", {}).get("withinLimits"):
            raise ValueError("颗粒精校准未通过数值验收")
        experience.validate_export(export)
        if experience.sha256 != manifest["experienceCalibration"]["sha256"]:
            raise ValueError("精校准文件已变化，请重新标定")
        if settings_fingerprint(RefinementSettings.from_dict(settings.get("abParticle"))) != refinement.get("settingsSha256"):
            raise ValueError("精校准设置已变化，请重新标定")
        if config and ((config.microscope_host and config.microscope_host != microscope_host)
                       or (config.microscope_port and config.microscope_port != microscope_port)):
            raise ValueError("电镜设备与精校准不一致，请重新标定")
        if not np.allclose(solution.matrix, experience.matrix) or not np.allclose(solution.translation, experience.translation):
            raise ValueError("任务中的精校准矩阵与冻结文件不一致")
        if not np.allclose(calibration["pixelAxisMatrix"], experience.audit["pixelAxisMatrix"]):
            raise ValueError("像素轴与冻结文件不一致")
        manifest["baselineImaging"] = refinement["pixelAxes"]["fine"]["signature"]
    microscope = microscope or ZemMicroscope(
        microscope_host,
        microscope_port,
        experience.imaging,
        pixel_avg=(config.pixel_avg if config is not None else 7),
    )
    if eds_enabled and spectrum_analyzer is None:
        spectrum_analyzer = RealSpectrumAnalyzer(
            eds_host,
            eds_port,
            microscope_host,
            microscope_port,
            timeout=float(task_settings.get("edsTimeoutSeconds") or 60.0),
            target_counts=task_settings.get("edsTargetCounts"),
        )
    selected_by_sequence = {particle.sequence: particle for particle in selected}
    total_count = len(selected)
    completed_count = 0
    work_items: list[tuple[int, OpticalParticle]] | None = None

    if resume:
        execution_order = [int(value) for value in manifest.get("executionOrder") or []]
        selected_sequences = [particle.sequence for particle in selected]
        if (
            len(execution_order) != len(selected_sequences)
            or len(execution_order) != len(set(execution_order))
            or set(execution_order) != set(selected_sequences)
        ):
            raise ValueError("原检测执行顺序缺失或与当前颗粒列表不一致，不能安全继续")

        preserved_records: list[dict[str, Any]] = []
        processed_sequences: set[int] = set()
        for record in manifest.get("particles") or []:
            status = str(record.get("status") or "")
            if status not in {"SUCCEEDED", "FAILED"}:
                continue
            sequence = int(record.get("sequence"))
            if sequence not in selected_by_sequence:
                raise ValueError(f"历史检测结果包含未知颗粒：#{sequence}")
            if sequence in processed_sequences:
                raise ValueError(f"历史检测结果包含重复颗粒：#{sequence}")
            processed_sequences.add(sequence)
            preserved_records.append(record)

        pending_sequences = [
            sequence for sequence in execution_order if sequence not in processed_sequences
        ]
        if not pending_sequences:
            raise ValueError("当前任务没有未处理颗粒，不能继续检测")

        resume_at = _now()
        archive_name = (
            "task-before-analysis-resume-"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        )
        archive_path = task_dir / "revisions" / archive_name
        _write_json_atomic(archive_path, manifest)
        completed_count = len(processed_sequences)
        next_rank = execution_order.index(pending_sequences[0]) + 1
        manifest.setdefault("analysisResumeHistory", []).append(
            {
                "resumedAt": resume_at,
                "previousStatus": initial_status,
                "previousError": manifest.get("error"),
                "previousAnalysisCompletedAt": manifest.get("analysisCompletedAt"),
                "processed": completed_count,
                "succeeded": sum(
                    record.get("status") == "SUCCEEDED"
                    for record in preserved_records
                ),
                "failed": sum(
                    record.get("status") == "FAILED"
                    for record in preserved_records
                ),
                "remaining": len(pending_sequences),
                "nextRank": next_rank,
                "nextSequence": pending_sequences[0],
                "previousReports": manifest.get("reports"),
                "previousCleanup": manifest.get("cleanup"),
                "archive": archive_path.relative_to(task_dir).as_posix(),
            }
        )
        manifest["particles"] = preserved_records
        manifest["analysisResumedAt"] = resume_at
        manifest["reports"] = None
        manifest["cleanup"] = {}
        manifest["error"] = None
        manifest["completedAt"] = None
        manifest.pop("analysisCompletedAt", None)
        manifest.pop("reportError", None)
        work_items = [
            (rank, selected_by_sequence[sequence])
            for rank, sequence in enumerate(execution_order, start=1)
            if sequence not in processed_sequences
        ]
    else:
        manifest["particles"] = []
        manifest["analysisStartedAt"] = _now()

    start_position: tuple[float, float] | None = None
    manifest["status"] = "RUNNING"
    _write_json_atomic(manifest_path, manifest)
    try:
        connection = microscope.connect()
        manifest["device"]["microscope"]["analysisConnection"] = connection
        start_position = microscope.get_position()
        if resume:
            progress(
                completed_count,
                total_count,
                f"恢复检测：已保留 {completed_count} 个结果，继续剩余 {total_count - completed_count} 个颗粒",
            )
        if eds_enabled and spectrum_analyzer is not None:
            progress(
                completed_count,
                total_count,
                f"检查能谱服务 {eds_host}:{eds_port}",
            )
            check_ready = getattr(spectrum_analyzer, "check_ready", None)
            if callable(check_ready):
                manifest["device"]["eds"]["readiness"] = check_ready()
        if auto_gun_on and hasattr(microscope, "gun_on"):
            progress(completed_count, total_count, "开启电子枪并等待高压稳定")
            manifest["device"]["microscope"]["analysisGunOnStatus"] = str(
                microscope.gun_on()
            )
        if work_items is None:
            route_plan = _optimized_particle_route(
                selected,
                solution,
                start_position,
                return_to_start=restore_position,
            )
            ordered = list(route_plan.items)
            manifest["executionOrder"] = [particle.sequence for particle in ordered]
            manifest["executionRoute"] = route_plan.to_audit_dict()
            work_items = list(enumerate(ordered, start=1))
        axis_matrix = calibration_settings.get("pixelAxisMatrix") or (
            (1.0, 0.0),
            (0.0, 1.0),
        )
        field_rotation = float(experience.imaging.get("fieldRotationDeg") or 0.0)
        if is_particle_refinement:
            axis_matrix = calibration["pixelAxisMatrix"]
            # Measured pixel axes include field rotation; applying it again would rotate twice.
            field_rotation = 0.0
        maximum_candidates = int(
            calibration_settings.get("semCandidatesPerAnchor") or 6
        )
        gate = CandidateGate.from_dict(calibration_settings.get("candidateGate"))
        baseline_signature = manifest.get("baselineImaging")
        for index, particle in work_items:
            if should_stop():
                manifest["status"] = "STOPPED"
                break
            target = solution.transform_xy(particle.ab_x_um, particle.ab_y_um)
            record: dict[str, Any] = {
                "rank": index,
                "sequence": particle.sequence,
                "particleId": particle.particle_id,
                "label": particle.label,
                "status": "RUNNING",
                "lengthUm": particle.length_um,
                "widthUm": particle.width_um,
                "areaUm2": particle.area_um2,
                "equivalentDiameterUm": particle.equivalent_diameter_um,
                "opticalAbPositionUm": {"x": particle.ab_x_um, "y": particle.ab_y_um},
                "targetStageUm": {"x": target[0], "y": target[1]},
                "images": {"opticalImage": f"optical/{particle.sequence:04d}.png"},
                "captureAttempts": [],
                "energySpectrum": None,
                "error": None,
                "timing": _empty_particle_timing(),
            }
            manifest["particles"].append(record)
            progress(
                index - 1,
                total_count,
                f"[{index}/{total_count}] 定位颗粒 #{particle.sequence}",
            )
            particle_started = time.monotonic()
            _emit_stage_event(
                event_logger,
                "particle_stage_started",
                particleSequence=particle.sequence,
                particleRank=index,
                stage="particle",
                attempt=None,
            )
            try:
                _require_target_inside_roi(
                    experience,
                    target,
                    particle=particle,
                )
                move_started = time.monotonic()
                _emit_stage_event(
                    event_logger,
                    "particle_stage_started",
                    particleSequence=particle.sequence,
                    particleRank=index,
                    stage="stage_move",
                    attempt=1,
                    targetStageUm={"x": target[0], "y": target[1]},
                )
                try:
                    if is_particle_refinement:
                        from .ab_particle_calibration import move_verified
                        refinement_cfg = RefinementSettings.from_dict(refinement["settings"])
                        approach = tuple(np.asarray(target) + refinement_cfg.approach_offset_um)
                        _require_target_inside_roi(experience, approach, particle=particle)
                        def check_refinement_stop():
                            if should_stop():
                                raise InterruptedError("用户停止了清洁度检测任务")
                        move_verified(microscope, approach, refinement_cfg, check=check_refinement_stop)
                        move = move_verified(microscope, target, refinement_cfg, check=check_refinement_stop)
                    else:
                        move = _move_stage_checked(microscope, target)
                except Exception as error:
                    move_seconds = _seconds_since(move_started)
                    record["timing"]["stageMoveSeconds"] = move_seconds
                    _emit_stage_event(
                        event_logger,
                        "particle_stage_finished",
                        particleSequence=particle.sequence,
                        particleRank=index,
                        stage="stage_move",
                        attempt=1,
                        status="failed",
                        elapsedSeconds=move_seconds,
                        error=str(error),
                    )
                    raise
                record["timing"]["stageMoveSeconds"] = _seconds_since(move_started)
                _emit_stage_event(
                    event_logger,
                    "particle_stage_finished",
                    particleSequence=particle.sequence,
                    particleRank=index,
                    stage="stage_move",
                    attempt=1,
                    status="succeeded",
                    elapsedSeconds=record["timing"]["stageMoveSeconds"],
                    actualStageUm=move.get("after"),
                    positionErrorUm=move.get("positionErrorUm"),
                )
                stage_center = _stage_xy(move["after"])
                record["actualStageUm"] = move["after"]
                record["stagePositionErrorUm"] = move.get("positionErrorUm")
                selected_attempt: tuple[
                    np.ndarray,
                    dict[str, Any],
                    tuple[SemAnchorCandidate, ...],
                    SemAnchorCandidate | None,
                    MatchDecision,
                    dict[str, Any],
                ] | None = None
                best_failed: tuple[SemAnchorCandidate, MatchDecision] | None = None
                last_capture_error: Exception | None = None
                for attempt in range(1, max(1, int(maximum_attempts)) + 1):
                    if should_stop():
                        raise InterruptedError("用户停止了清洁度检测任务")
                    raw_path = (
                        task_dir
                        / "sem"
                        / f"{particle.sequence:04d}_attempt{attempt}.png"
                    )
                    capture_seconds = None
                    match_seconds = None
                    match_started = None
                    capture_started = time.monotonic()
                    _emit_stage_event(
                        event_logger,
                        "particle_stage_started",
                        particleSequence=particle.sequence,
                        particleRank=index,
                        stage="capture",
                        attempt=attempt,
                    )
                    try:
                        image, capture = microscope.capture(raw_path)
                        _validate_calibrated_capture(capture, experience)
                        if is_particle_refinement:
                            from .ab_particle_calibration import validate_field_metadata
                            validate_field_metadata(capture, experience.imaging.get("fieldSizeUm"))
                            after_capture = np.asarray(microscope.get_position(), float)
                            if not np.isfinite(after_capture).all() or np.linalg.norm(after_capture - stage_center) > refinement_cfg.stage_error_um:
                                raise RuntimeError("拍摄期间样品台漂移超限，不能使用旧台坐标")
                            stage_center = tuple(after_capture)
                        signature = _capture_signature(capture)
                        if list(image.shape[:2]) != list(signature["shape"][:2]):
                            raise RuntimeError("采图数组尺寸与采集元数据不一致，不能使用该图定位或采谱")
                        if baseline_signature is not None:
                            _validate_same_signature(baseline_signature, signature)
                        else:
                            baseline_signature = signature
                            manifest["baselineImaging"] = signature
                        capture_seconds = _seconds_since(capture_started)
                        _accumulate_timing(record, "captureSeconds", capture_seconds)
                        _emit_stage_event(
                            event_logger,
                            "particle_stage_finished",
                            particleSequence=particle.sequence,
                            particleRank=index,
                            stage="capture",
                            attempt=attempt,
                            status="succeeded",
                            elapsedSeconds=capture_seconds,
                            imageShape=list(image.shape),
                            pixelSizeUm=capture.get("pixelSizeUm"),
                        )
                        if should_stop():
                            raise InterruptedError("用户停止了清洁度检测任务")
                        match_started = time.monotonic()
                        _emit_stage_event(
                            event_logger,
                            "particle_stage_started",
                            particleSequence=particle.sequence,
                            particleRank=index,
                            stage="match",
                            attempt=attempt,
                        )
                        center_plan = plan_large_particle_center(image,particle,float(capture["pixelSizeUm"]),
                            field_width_um=signature.get("fieldSizeUm"),settings=center_settings)
                        center_ready = center_plan["status"] == "READY"
                        if center_plan["status"] != "NOT_APPLICABLE":
                            candidates, chosen, mask_verification = (), None, None
                            decision = MatchDecision(
                                "BYPASSED_LARGE_PARTICLE_CENTER" if center_ready else "FAILED_CENTER_SAMPLING",
                                "unverified", center_plan["reason"], None, None, None, 0)
                            record.update(centerSampling=center_plan,match=decision.to_dict())
                        else:
                            candidates = build_sem_anchor_candidates(
                                particle, image, pixel_size_um=float(capture["pixelSizeUm"]),
                                stage_center_um=stage_center, field_rotation_deg=field_rotation,
                                pixel_axis_matrix=axis_matrix, maximum_candidates=maximum_candidates,
                                segmentation=segmentation, gate=gate,
                            )
                            if not candidates and not particle.mask_data_base64:
                                raise RuntimeError("SEM 视场中没有找到颗粒候选")
                            chosen, decision, mask_verification = _select_mask_guided_candidate(
                                image, particle, float(capture["pixelSizeUm"]), candidates,
                                stage_center_um=stage_center,field_rotation_deg=field_rotation,
                                pixel_axis_matrix=axis_matrix,segmentation=segmentation,gate=gate,should_stop=should_stop,
                                refinement_settings=identity_settings,
                                minimum_mask_iou=contour_overlap_min,
                                calibration_location={"reviewAccepted":calibration_review.get("accepted") is True,
                                    "maximumCenterErrorUm":(calibration_settings.get("acceptance") or {}).get("maxErrorUm"),
                                    "targetStageUm":list(target)},
                            )
                        match_seconds = _seconds_since(match_started)
                        _accumulate_timing(record, "matchSeconds", match_seconds)
                        _emit_stage_event(
                            event_logger,
                            "particle_stage_finished",
                            particleSequence=particle.sequence,
                            particleRank=index,
                            stage="match",
                            attempt=attempt,
                            status="accepted" if decision.matched or center_ready else "rejected",
                            elapsedSeconds=match_seconds,
                            decisionStatus=decision.status,
                            candidateCount=len(candidates),
                            acquisitionMode="large_particle_center_region" if center_ready else "matched_particle",
                        )
                        record["captureAttempts"].append(
                            {
                                "attempt": attempt,
                                "status": decision.status,
                                "rawPath": _relative(
                                    Path(capture["rawPath"]), task_dir
                                ),
                                "displayPath": _relative(
                                    Path(capture["displayPath"]), task_dir
                                ),
                                "match": decision.to_dict(),
                                "maskVerification": mask_verification,
                                "centerSampling": center_plan if center_plan["status"] != "NOT_APPLICABLE" else None,
                                "captureSeconds": capture_seconds,
                                "matchSeconds": match_seconds,
                            }
                        )
                        if decision.matched or center_ready:
                            selected_attempt = (
                                image,
                                capture,
                                candidates,
                                chosen,
                                decision,
                                center_plan,
                            )
                            break
                        if chosen is not None and (
                            best_failed is None
                            or chosen.combined_score < best_failed[0].combined_score
                        ):
                            best_failed = (chosen, decision)
                        last_capture_error = RuntimeError(decision.reason)
                    except InterruptedError as error:
                        if capture_seconds is None:
                            capture_seconds = _seconds_since(capture_started)
                            _accumulate_timing(record, "captureSeconds", capture_seconds)
                            _emit_stage_event(
                                event_logger,
                                "particle_stage_finished",
                                particleSequence=particle.sequence,
                                particleRank=index,
                                stage="capture",
                                attempt=attempt,
                                status="stopped",
                                elapsedSeconds=capture_seconds,
                                error=str(error),
                            )
                        if match_started is not None and match_seconds is None:
                            match_seconds = _seconds_since(match_started)
                            _accumulate_timing(record, "matchSeconds", match_seconds)
                            _emit_stage_event(
                                event_logger,
                                "particle_stage_finished",
                                particleSequence=particle.sequence,
                                particleRank=index,
                                stage="match",
                                attempt=attempt,
                                status="stopped",
                                elapsedSeconds=match_seconds,
                                error=str(error),
                            )
                        raise
                    except Exception as error:
                        last_capture_error = error
                        if capture_seconds is None:
                            capture_seconds = _seconds_since(capture_started)
                            _accumulate_timing(record, "captureSeconds", capture_seconds)
                            _emit_stage_event(
                                event_logger,
                                "particle_stage_finished",
                                particleSequence=particle.sequence,
                                particleRank=index,
                                stage="capture",
                                attempt=attempt,
                                status="failed",
                                elapsedSeconds=capture_seconds,
                                error=str(error),
                            )
                        if match_started is not None and match_seconds is None:
                            match_seconds = _seconds_since(match_started)
                            _accumulate_timing(record, "matchSeconds", match_seconds)
                            _emit_stage_event(
                                event_logger,
                                "particle_stage_finished",
                                particleSequence=particle.sequence,
                                particleRank=index,
                                stage="match",
                                attempt=attempt,
                                status="failed",
                                elapsedSeconds=match_seconds,
                                error=str(error),
                            )
                        record["captureAttempts"].append(
                            {
                                "attempt": attempt,
                                "status": "FAILED",
                                "error": str(error),
                                "captureSeconds": capture_seconds,
                                "matchSeconds": match_seconds,
                            }
                        )
                if selected_attempt is None:
                    if best_failed is not None:
                        record["match"] = best_failed[1].to_dict()
                    raise RuntimeError(
                        f"连续 {max(1, int(maximum_attempts))} 次采图/匹配失败："
                        f"{last_capture_error or '未知错误'}"
                    )
                image, capture, candidates, chosen, decision, center_plan = selected_attempt
                center_ready = center_plan["status"] == "READY"
                save_started = time.monotonic()
                _emit_stage_event(
                    event_logger,
                    "particle_stage_started",
                    particleSequence=particle.sequence,
                    particleRank=index,
                    stage="match_output",
                    attempt=None,
                )
                try:
                    if center_ready:
                        outputs = _save_center_sampling_images(task_dir,particle,image,center_plan)
                        sample_stage = pixel_to_stage_xy(center_plan["pointPx"],image.shape,float(capture["pixelSizeUm"]),
                            stage_center,field_rotation_deg=field_rotation,pixel_axis_matrix=axis_matrix)
                        center_error, metrics, observed_stage = None, {"aspect":None,"roundness":None}, None
                        record.update(acquisitionMode="large_particle_center_region",centerSampling=center_plan,
                            samplingStageUm={"x":sample_stage[0],"y":sample_stage[1]})
                    else:
                        outputs, _image_decision = _save_match_images(
                            task_dir, particle, image, chosen, len(candidates), "analysis",
                            "经验矩阵粗定位后结合形貌完成精定位", gate,
                        )
                        center_error = math.hypot(chosen.stage_xy_um[0]-target[0],chosen.stage_xy_um[1]-target[1])
                        metrics = _shape_metrics(chosen)
                        observed_stage = {"x":chosen.stage_xy_um[0],"y":chosen.stage_xy_um[1]}
                        record["acquisitionMode"] = "matched_particle"
                except Exception as error:
                    save_seconds = _seconds_since(save_started)
                    _accumulate_timing(record, "matchSeconds", save_seconds)
                    _emit_stage_event(
                        event_logger,
                        "particle_stage_finished",
                        particleSequence=particle.sequence,
                        particleRank=index,
                        stage="match_output",
                        attempt=None,
                        status="failed",
                        elapsedSeconds=save_seconds,
                        error=str(error),
                    )
                    raise
                save_seconds = _seconds_since(save_started)
                _accumulate_timing(record, "matchSeconds", save_seconds)
                _emit_stage_event(
                    event_logger,
                    "particle_stage_finished",
                    particleSequence=particle.sequence,
                    particleRank=index,
                    stage="match_output",
                    attempt=None,
                    status="succeeded",
                    elapsedSeconds=save_seconds,
                )
                record.update(
                    {
                        "status": "SUCCEEDED",
                        "match": decision.to_dict(),
                        "maskVerification": mask_verification,
                        "localization": {
                            "observedStageUm": observed_stage,
                            "centerErrorUm": center_error,
                            "safePointPx": outputs.get("safePointPx"),
                            "safeBoundaryDistancePx": outputs.get(
                                "safeBoundaryDistancePx"
                            ),
                        },
                        "aspect": metrics["aspect"],
                        "roundness": metrics["roundness"],
                    }
                )
                record["images"].update(
                    {
                        "semRawImage": _relative(Path(capture["rawPath"]), task_dir),
                        "semDisplayImage": _relative(Path(capture["displayPath"]), task_dir),
                        "semAnnotatedImage": _relative(
                            Path(outputs["annotatedImage"]), task_dir
                        ),
                        "semParticleImage": _relative(
                            Path(outputs["particleImage"]), task_dir
                        ),
                    }
                )
                if eds_enabled and spectrum_analyzer is not None:
                    progress(
                        index - 1,
                        total_count,
                        f"[{index}/{total_count}] 颗粒 #{particle.sequence} 电子束偏转与 EDS",
                    )
                    safe = outputs.get("safePointPx")
                    safe_point = (
                        (int(safe[0]), int(safe[1])) if safe is not None else None
                    )
                    record["edsAttempts"] = []
                    for eds_attempt in range(1, eds_retry_count + 2):
                        if should_stop():
                            raise InterruptedError("用户停止了清洁度检测任务")
                        eds_started = time.monotonic()
                        _emit_stage_event(
                            event_logger,
                            "particle_stage_started",
                            particleSequence=particle.sequence,
                            particleRank=index,
                            stage="eds",
                            attempt=eds_attempt,
                            targetPointPx=list(safe_point) if safe_point is not None else None,
                        )
                        try:
                            record["energySpectrum"] = spectrum_analyzer.analyze(
                                particle,
                                chosen,
                                Path(capture["displayPath"]),
                                safe_point,
                            )
                            eds_seconds = _seconds_since(eds_started)
                            _accumulate_timing(record, "edsSeconds", eds_seconds)
                            _emit_stage_event(
                                event_logger,
                                "particle_stage_finished",
                                particleSequence=particle.sequence,
                                particleRank=index,
                                stage="eds",
                                attempt=eds_attempt,
                                status="succeeded",
                                elapsedSeconds=eds_seconds,
                            )
                            record["edsAttempts"].append(
                                {
                                    "attempt": eds_attempt,
                                    "status": "SUCCEEDED",
                                    "elapsedSeconds": eds_seconds,
                                }
                            )
                            break
                        except Exception as error:
                            eds_seconds = _seconds_since(eds_started)
                            _accumulate_timing(record, "edsSeconds", eds_seconds)
                            transient = _is_transient_eds_error(error)
                            _emit_stage_event(
                                event_logger,
                                "particle_stage_finished",
                                particleSequence=particle.sequence,
                                particleRank=index,
                                stage="eds",
                                attempt=eds_attempt,
                                status="failed",
                                elapsedSeconds=eds_seconds,
                                transient=transient,
                                error=str(error),
                            )
                            record["edsAttempts"].append(
                                {
                                    "attempt": eds_attempt,
                                    "status": "FAILED",
                                    "transient": transient,
                                    "error": str(error),
                                    "elapsedSeconds": eds_seconds,
                                }
                            )
                            if not transient or eds_attempt > eds_retry_count:
                                raise
                            progress(
                                index - 1,
                                total_count,
                                (
                                    f"[{index}/{total_count}] 颗粒 #{particle.sequence} "
                                    f"EDS 临时失败，自动重试 {eds_attempt}/{eds_retry_count}"
                                ),
                            )
            except InterruptedError as error:
                record["status"] = "STOPPED"
                record["error"] = str(error)
                _finalize_particle_timing(record, particle_started)
                _emit_stage_event(
                    event_logger,
                    "particle_stage_finished",
                    particleSequence=particle.sequence,
                    particleRank=index,
                    stage="particle",
                    attempt=None,
                    status="stopped",
                    elapsedSeconds=record["timing"]["totalSeconds"],
                    error=str(error),
                    timing=record["timing"],
                )
                manifest["status"] = "STOPPED"
                _write_json_atomic(manifest_path, manifest)
                break
            except Exception as error:
                record["status"] = "FAILED"
                record["error"] = str(error)
            _finalize_particle_timing(record, particle_started)
            _emit_stage_event(
                event_logger,
                "particle_stage_finished",
                particleSequence=particle.sequence,
                particleRank=index,
                stage="particle",
                attempt=None,
                status=record["status"].lower(),
                elapsedSeconds=record["timing"]["totalSeconds"],
                error=record.get("error"),
                timing=record["timing"],
            )
            _write_json_atomic(manifest_path, manifest)
            progress(index, total_count, f"颗粒 #{particle.sequence} {record['status']}")
        if manifest["status"] == "RUNNING":
            failed = sum(record["status"] != "SUCCEEDED" for record in manifest["particles"])
            manifest["status"] = "COMPLETED" if failed == 0 else "COMPLETED_WITH_FAILURES"
    except Exception as error:
        manifest["status"] = "FAILED"
        manifest["error"] = str(error)
    finally:
        if spectrum_analyzer is not None:
            try:
                spectrum_analyzer.close()
            except Exception as error:
                manifest["cleanup"]["edsCloseError"] = str(error)
        manifest["cleanup"].update(
            _restore_and_maybe_gun_off(
                microscope,
                start_position,
                restore_position=restore_position,
                gun_off=gun_off_after_task,
            )
        )
        manifest["timingSummary"] = _timing_summary(manifest.get("particles") or [])
        manifest["analysisCompletedAt"] = _now()
        if manifest["cleanup"].get("gunOffError"):
            shutdown_error = f"自动关枪失败，请人工确认电子枪状态：{manifest['cleanup']['gunOffError']}"
            manifest["error"] = f"{manifest['error']}；{shutdown_error}" if manifest.get("error") else shutdown_error
        manifest["completedAt"] = _now()
        _write_json_atomic(manifest_path, manifest)

    try:
        from backend.cleanliness_core.cleanliness_report import build_cleanliness_reports

        reports = build_cleanliness_reports(
            task_dir,
            manifest,
            report_config=settings.get("report"),
        )
        manifest["reports"] = {
            key: _relative(Path(value), task_dir) for key, value in reports.items()
        }
    except Exception as error:
        manifest["reportError"] = str(error)
    _write_json_atomic(manifest_path, manifest)
    return manifest
