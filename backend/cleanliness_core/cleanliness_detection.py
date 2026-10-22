from __future__ import annotations

import base64
import hashlib
import itertools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from loguru import logger

from backend.cleanliness_core.optical_particle_analysis import (
    Calibration,
    OpticalExport,
    OpticalParticle,
    ParticleCandidate,
    detect_particle_candidates,
    normalize_uint8,
)


@dataclass(frozen=True)
class CandidateSegmentationConfig:
    """Optional segmentation backend used to supplement classic SEM detection."""

    backend: str = "classic"
    yolo_model_path: Path | None = None
    input_size: int = 1024
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    mask_threshold: float = 0.5
    maximum_detections: int = 100
    sam3_url: str | None = None

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any] | None,
        *,
        settings_path: Path,
        sam3_url: str | None = None,
    ) -> "CandidateSegmentationConfig":
        if not payload:
            return cls(
                backend="sam3" if sam3_url else "classic",
                sam3_url=sam3_url,
            )
        backend = str(payload.get("backend") or "classic").strip().lower()
        if backend not in {"classic", "yolo", "sam3"}:
            raise ValueError(
                "calibration.candidateSegmentation.backend "
                f"must be yolo, sam3 or classic, got {backend!r}"
            )
        model_path: Path | None = None
        configured_model = str(payload.get("modelPath") or "").strip()
        if configured_model:
            model_path = Path(configured_model)
            if not model_path.is_absolute():
                model_path = Path(settings_path).resolve().parent / model_path
            model_path = model_path.resolve()
        if backend == "yolo" and model_path is None:
            raise ValueError(
                "candidateSegmentation.backend=yolo requires modelPath"
            )
        configured_sam3_url = str(payload.get("sam3Url") or "").strip() or None
        return cls(
            backend=backend,
            yolo_model_path=model_path,
            input_size=max(32, int(payload.get("inputSize") or 1024)),
            confidence_threshold=float(
                payload.get("confidenceThreshold")
                if payload.get("confidenceThreshold") is not None
                else 0.25
            ),
            iou_threshold=float(
                payload.get("iouThreshold")
                if payload.get("iouThreshold") is not None
                else 0.45
            ),
            mask_threshold=float(
                payload.get("maskThreshold")
                if payload.get("maskThreshold") is not None
                else 0.5
            ),
            maximum_detections=max(
                1, int(payload.get("maxDetections") or 100)
            ),
            sam3_url=configured_sam3_url or sam3_url,
        )

    def validate(self) -> None:
        if self.backend == "yolo":
            if self.yolo_model_path is None:
                raise ValueError("YOLO segmentation model path is not configured")
            if not self.yolo_model_path.is_file():
                raise FileNotFoundError(
                    f"YOLO segmentation model not found: {self.yolo_model_path}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "modelPath": (
                str(self.yolo_model_path) if self.yolo_model_path is not None else None
            ),
            "inputSize": self.input_size,
            "confidenceThreshold": self.confidence_threshold,
            "iouThreshold": self.iou_threshold,
            "maskThreshold": self.mask_threshold,
            "maxDetections": self.maximum_detections,
            "sam3Url": self.sam3_url,
        }


@dataclass(frozen=True)
class ParticleFilter:
    """清洁度任务使用的光镜物理量筛选条件，单位均来自光镜 JSON。"""

    min_length_um: float | None = None
    max_length_um: float | None = None
    min_width_um: float | None = None
    max_width_um: float | None = None
    min_area_um2: float | None = None
    max_area_um2: float | None = None
    min_ecd_um: float | None = None
    max_ecd_um: float | None = None
    labels: tuple[str, ...] = ()
    max_particles: int | None = None

    def matches(self, particle: OpticalParticle) -> bool:
        checks = (
            (particle.length_um, self.min_length_um, self.max_length_um),
            (particle.width_um, self.min_width_um, self.max_width_um),
            (particle.area_um2, self.min_area_um2, self.max_area_um2),
            (
                particle.equivalent_diameter_um,
                self.min_ecd_um,
                self.max_ecd_um,
            ),
        )
        for value, minimum, maximum in checks:
            if minimum is not None and value < minimum:
                return False
            if maximum is not None and value > maximum:
                return False
        if self.labels and particle.label not in self.labels:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "minLengthUm": self.min_length_um,
            "maxLengthUm": self.max_length_um,
            "minWidthUm": self.min_width_um,
            "maxWidthUm": self.max_width_um,
            "minAreaUm2": self.min_area_um2,
            "maxAreaUm2": self.max_area_um2,
            "minEcdUm": self.min_ecd_um,
            "maxEcdUm": self.max_ecd_um,
            "labels": list(self.labels),
            "maxParticles": self.max_particles,
        }


def filter_optical_particles(
    particles: Iterable[OpticalParticle],
    criteria: ParticleFilter,
    *,
    largest_first: bool = True,
) -> tuple[OpticalParticle, ...]:
    selected = [particle for particle in particles if criteria.matches(particle)]
    if largest_first:
        selected.sort(
            key=lambda particle: (
                particle.length_um,
                particle.area_um2,
                -particle.sequence,
            ),
            reverse=True,
        )
    else:
        selected.sort(key=lambda particle: particle.sequence)
    if criteria.max_particles is not None:
        selected = selected[: max(0, int(criteria.max_particles))]
    return tuple(selected)


def decode_optical_image_array(particle: OpticalParticle) -> np.ndarray:
    encoded = particle.image_data_base64
    if "," in encoded and encoded.lstrip().lower().startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    raw = base64.b64decode(encoded, validate=True)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"颗粒 #{particle.sequence} 的光镜图片无法解码")
    return image


@dataclass(frozen=True)
class OpticalShape:
    contour: tuple[tuple[int, int], ...]
    aspect: float
    solidity: float
    area_fraction: float
    source: str = "image_inferred"


def extract_optical_shape(particle: OpticalParticle) -> OpticalShape | None:
    """优先使用导出的颗粒掩码；仅无掩码的旧数据从照片提取轮廓。"""

    if particle.mask_data_base64:
        encoded = particle.mask_data_base64.split(",", 1)[-1]
        try:
            raw = base64.b64decode(encoded, validate=True)
            mask = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        except (ValueError, cv2.error) as exc:
            raise ValueError(f"颗粒 #{particle.sequence} 的光镜掩码无法解码") from exc
        if mask is None:
            raise ValueError(f"颗粒 #{particle.sequence} 的光镜掩码无法解码")
        height, width = mask.shape
        if ((particle.mask_width and width != particle.mask_width)
                or (particle.mask_height and height != particle.mask_height)
                or (particle.image_width and width != particle.image_width)
                or (particle.image_height and height != particle.image_height)):
            raise ValueError(f"颗粒 #{particle.sequence} 的光镜掩码尺寸不符")
        binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or binary.all():
            raise ValueError(f"颗粒 #{particle.sequence} 的光镜掩码为空或全前景")
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < 8:
            raise ValueError(f"颗粒 #{particle.sequence} 的光镜掩码有效轮廓过小")
        (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(contour)
        short_side = max(1.0, min(rect_width, rect_height))
        return OpticalShape(
            contour=tuple((int(p[0][0]), int(p[0][1])) for p in contour),
            aspect=max(short_side, rect_width, rect_height) / short_side,
            solidity=area / max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0),
            area_fraction=area / float(height * width),
            source="customer_mask",
        )

    image = decode_optical_image_array(particle)
    height, width = image.shape[:2]
    if min(height, width) < 6:
        return None
    work = cv2.GaussianBlur(image.astype(np.float32), (3, 3), 0)
    border = np.concatenate((work[0], work[-1], work[:, 0], work[:, -1]))
    background = np.median(border, axis=0)
    spread = np.maximum(
        np.median(np.abs(border - background), axis=0) * 1.4826,
        4.0,
    )
    distance = np.sqrt(np.sum(((work - background) / spread) ** 2, axis=2))
    mask = np.where(distance >= 2.4, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    image_center = np.asarray([width / 2.0, height / 2.0])

    def rank(contour: np.ndarray) -> float:
        area = float(cv2.contourArea(contour))
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            return -1.0
        center = np.asarray(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
        )
        return area - 1.5 * float(np.linalg.norm(center - image_center)) * math.sqrt(
            max(area, 1.0)
        )

    contour = max(contours, key=rank)
    area = float(cv2.contourArea(contour))
    if area < 8.0:
        return None
    (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(contour)
    short_side = max(1.0, min(rect_width, rect_height))
    long_side = max(short_side, rect_width, rect_height)
    hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
    return OpticalShape(
        contour=tuple((int(point[0][0]), int(point[0][1])) for point in contour),
        aspect=long_side / short_side,
        solidity=area / hull_area,
        area_fraction=area / float(height * width),
    )


def shape_dissimilarity(
    optical_shape: OpticalShape | None,
    candidate: ParticleCandidate,
) -> float:
    if optical_shape is None:
        return 0.75
    contour = np.asarray(candidate.contour, dtype=np.int32)
    if len(contour) < 5:
        return 2.0
    reference = np.asarray(optical_shape.contour, dtype=np.int32)
    hu = min(
        float(cv2.matchShapes(reference, contour, cv2.CONTOURS_MATCH_I1, 0.0)),
        3.0,
    )
    area = max(float(cv2.contourArea(contour)), 1.0)
    (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(contour)
    short_side = max(1.0, min(rect_width, rect_height))
    aspect = max(rect_width, rect_height, short_side) / short_side
    hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
    solidity = area / hull_area
    return (
        0.55 * hu
        + 0.75 * abs(math.log(max(aspect, 0.2) / max(optical_shape.aspect, 0.2)))
        + 1.10 * abs(solidity - optical_shape.solidity)
    )


def optical_salience_score(particle: OpticalParticle) -> float:
    image = decode_optical_image_array(particle)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    return (
        1.6 * math.log1p(max(particle.area_um2, 0.0))
        + 0.9 * math.log1p(max(particle.length_um, 0.0))
        + 0.025 * contrast
    )


def _normalized_positions(
    particles: Sequence[OpticalParticle],
) -> dict[int, np.ndarray]:
    points = np.asarray(
        [[particle.ab_x_um, particle.ab_y_um] for particle in particles],
        dtype=np.float64,
    )
    minimum = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - minimum, 1.0)
    return {
        particle.sequence: (point - minimum) / span
        for particle, point in zip(particles, points, strict=True)
    }


def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    return abs(float(ab[0] * ac[1] - ab[1] * ac[0])) * 0.5


def select_calibration_landmarks(
    export: OpticalExport,
    *,
    anchor_count: int = 3,
    validation_count: int = 2,
    pool_size: int = 30,
    excluded_sequences: Iterable[int] = (),
) -> tuple[tuple[OpticalParticle, ...], tuple[OpticalParticle, ...]]:
    """选择大、显著且空间分散的标定颗粒，并独立选择验证颗粒。"""

    required = max(3, int(anchor_count)) + max(0, int(validation_count))
    excluded = {int(value) for value in excluded_sequences}
    available = [
        particle for particle in export.particles if particle.sequence not in excluded
    ]
    if len(available) < required:
        raise ValueError(f"有效颗粒不足：至少需要 {required} 个")
    qualities = {particle.sequence: optical_salience_score(particle) for particle in available}
    ranked = sorted(
        available,
        key=lambda particle: (qualities[particle.sequence], particle.length_um),
        reverse=True,
    )[: max(required, int(pool_size))]
    positions = _normalized_positions(ranked)
    quality_values = np.asarray([qualities[p.sequence] for p in ranked], dtype=np.float64)
    quality_min = float(quality_values.min())
    quality_span = max(float(np.ptp(quality_values)), 1e-9)

    def quality(particle: OpticalParticle) -> float:
        return (qualities[particle.sequence] - quality_min) / quality_span

    chosen: list[OpticalParticle] = [ranked[0]]
    while len(chosen) < max(3, int(anchor_count)):
        best: OpticalParticle | None = None
        best_score = -math.inf
        for particle in ranked:
            if particle in chosen:
                continue
            point = positions[particle.sequence]
            min_distance = min(
                float(np.linalg.norm(point - positions[item.sequence]))
                for item in chosen
            )
            area = 0.0
            if len(chosen) >= 2:
                area = max(
                    _triangle_area(
                        positions[first.sequence],
                        positions[second.sequence],
                        point,
                    )
                    for first, second in itertools.combinations(chosen, 2)
                )
                if len(chosen) == 2 and area < 0.01:
                    continue
            # 优先使用最大、最亮、形貌清晰的颗粒；空间距离和三角形面积
            # 主要用于避免锚点过近或接近共线。真实 Zem20F 数据回放表明，
            # 过度强调几何跨度会选中较小颗粒并放大跨区域定位误差。
            score = 1.00 * quality(particle) + 0.15 * min_distance + 0.30 * area
            if score > best_score:
                best = particle
                best_score = score
        if best is None:
            raise ValueError("无法选择足够的非共线标定颗粒")
        chosen.append(best)

    first_three = [positions[particle.sequence] for particle in chosen[:3]]
    if _triangle_area(*first_three) < 0.01:
        raise ValueError("自动选择的三个标定颗粒接近共线，请人工指定标定颗粒")

    validation = list(
        select_validation_landmarks(
            export,
            chosen,
            validation_count=validation_count,
            pool_size=pool_size,
            excluded_sequences=excluded,
        )
    )
    return tuple(chosen), tuple(validation)


def select_calibration_groups(
    export: OpticalExport,
    *,
    group_count: int = 3,
    anchor_count: int = 3,
    validation_count: int = 2,
    pool_size: int = 30,
    excluded_sequences: Iterable[int] = (),
) -> list[tuple[tuple[OpticalParticle, ...], tuple[OpticalParticle, ...]]]:
    """选出多组互不重叠的标定方案，每组独立包含标定颗粒和验证颗粒。

    循环调用 select_calibration_landmarks，每轮将已选颗粒排除，保证组间
    不重叠且每组内部仍满足空间分散和显著性要求。
    """
    group_count = max(1, int(group_count))
    excluded = {int(v) for v in excluded_sequences}
    groups: list[tuple[tuple[OpticalParticle, ...], tuple[OpticalParticle, ...]]] = []
    for _ in range(group_count):
        anchors, validation = select_calibration_landmarks(
            export,
            anchor_count=anchor_count,
            validation_count=validation_count,
            pool_size=pool_size,
            excluded_sequences=excluded,
        )
        excluded.update(p.sequence for p in anchors)
        excluded.update(p.sequence for p in validation)
        groups.append((anchors, validation))
    return groups


def select_validation_landmarks(
    export: OpticalExport,
    anchors: Sequence[OpticalParticle],
    *,
    validation_count: int = 2,
    pool_size: int = 30,
    excluded_sequences: Iterable[int] = (),
) -> tuple[OpticalParticle, ...]:
    """围绕已确定的标定集选择空间分散、易识别的独立验证颗粒。"""

    count = max(0, int(validation_count))
    if count == 0:
        return ()
    anchors = tuple(anchors)
    if not anchors:
        raise ValueError("选择验证颗粒前至少需要一个标定颗粒")
    excluded = {int(value) for value in excluded_sequences}
    excluded.update(particle.sequence for particle in anchors)
    available = [
        particle
        for particle in export.particles
        if particle.sequence not in excluded
    ]
    if len(available) < count:
        raise ValueError(f"独立验证颗粒不足：需要 {count} 个，实际 {len(available)} 个")
    ranked = sorted(
        available,
        key=lambda particle: (
            optical_salience_score(particle),
            particle.length_um,
        ),
        reverse=True,
    )[: max(count, int(pool_size))]
    positions = _normalized_positions((*anchors, *ranked))
    salience = {
        particle.sequence: optical_salience_score(particle)
        for particle in ranked
    }
    values = np.asarray(list(salience.values()), dtype=np.float64)
    minimum = float(values.min())
    span = max(float(np.ptp(values)), 1e-9)
    chosen: list[OpticalParticle] = []
    while len(chosen) < count:
        used = [*anchors, *chosen]
        candidates = [particle for particle in ranked if particle not in chosen]
        best = max(
            candidates,
            key=lambda particle: (
                0.72
                * min(
                    float(
                        np.linalg.norm(
                            positions[particle.sequence]
                            - positions[item.sequence]
                        )
                    )
                    for item in used
                )
                + 0.28
                * (salience[particle.sequence] - minimum)
                / span,
                particle.length_um,
            ),
        )
        chosen.append(best)
    return tuple(chosen)


@dataclass(frozen=True)
class TransformSolution:
    model: str
    matrix: np.ndarray
    translation: np.ndarray
    residuals_um: tuple[float, ...]
    rms_error_um: float
    max_error_um: float
    scale: float | None = None
    reflected: bool | None = None

    def transform_xy(self, x: float, y: float) -> tuple[float, float]:
        target = self.matrix @ np.asarray([x, y], dtype=np.float64) + self.translation
        return float(target[0]), float(target[1])

    def equations(self) -> tuple[str, str]:
        return (
            f"stageX = {self.matrix[0, 0]:.10f}*abX {self.matrix[0, 1]:+.10f}*abY {self.translation[0]:+.6f}",
            f"stageY = {self.matrix[1, 0]:.10f}*abX {self.matrix[1, 1]:+.10f}*abY {self.translation[1]:+.6f}",
        )

    def to_dict(self) -> dict[str, Any]:
        equation_x, equation_y = self.equations()
        return {
            "model": self.model,
            "matrix": self.matrix.tolist(),
            "translation": self.translation.tolist(),
            "equation": {"stageX": equation_x, "stageY": equation_y},
            "scale": self.scale,
            "reflected": self.reflected,
            "fit": {
                "residualsUm": list(self.residuals_um),
                "rmsErrorUm": self.rms_error_um,
                "maxErrorUm": self.max_error_um,
            },
        }


def transform_solution_from_dict(payload: dict[str, Any]) -> TransformSolution:
    fit = payload.get("fit") or {}
    matrix = np.asarray(payload.get("matrix"), dtype=np.float64)
    translation = np.asarray(payload.get("translation"), dtype=np.float64)
    if matrix.shape != (2, 2) or translation.shape != (2,):
        raise ValueError("标定结果中的矩阵或平移向量无效")
    residuals = tuple(float(value) for value in fit.get("residualsUm") or ())
    return TransformSolution(
        model=str(payload.get("model") or "reflected_similarity"),
        matrix=matrix,
        translation=translation,
        residuals_um=residuals,
        rms_error_um=float(fit.get("rmsErrorUm") or 0.0),
        max_error_um=float(fit.get("maxErrorUm") or 0.0),
        scale=(
            float(payload["scale"])
            if payload.get("scale") is not None
            else None
        ),
        reflected=(
            bool(payload["reflected"])
            if payload.get("reflected") is not None
            else None
        ),
    )


def _solution_from_matrix(
    model: str,
    matrix: np.ndarray,
    translation: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    *,
    scale: float | None = None,
    reflected: bool | None = None,
) -> TransformSolution:
    predicted = (matrix @ source.T).T + translation
    residuals = np.linalg.norm(predicted - target, axis=1)
    return TransformSolution(
        model=model,
        matrix=matrix,
        translation=translation,
        residuals_um=tuple(float(value) for value in residuals),
        rms_error_um=float(math.sqrt(float(np.mean(residuals**2)))),
        max_error_um=float(np.max(residuals)),
        scale=scale,
        reflected=reflected,
    )


def _fit_similarity(
    source: np.ndarray,
    target: np.ndarray,
    *,
    determinant: int,
    model: str,
) -> TransformSolution:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    x = source - source_center
    y = target - target_center
    covariance = x.T @ y
    u, _singular, vt = np.linalg.svd(covariance)
    v = vt.T
    rotation = v @ u.T
    if int(round(float(np.linalg.det(rotation)))) != int(determinant):
        correction = np.eye(2)
        correction[-1, -1] = -1.0
        rotation = v @ correction @ u.T
    rotated = (rotation @ x.T).T
    denominator = float(np.sum(x * x))
    if denominator <= 1e-12:
        raise ValueError("标定源坐标没有足够跨度")
    scale = float(np.sum(rotated * y) / denominator)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("标定相似变换缩放系数无效")
    matrix = scale * rotation
    translation = target_center - matrix @ source_center
    return _solution_from_matrix(
        model,
        matrix,
        translation,
        source,
        target,
        scale=scale,
        reflected=determinant < 0,
    )


def fit_coordinate_transform(
    source_points: Sequence[Sequence[float]],
    target_points: Sequence[Sequence[float]],
    *,
    model: str = "reflected_similarity",
) -> TransformSolution:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("标定源点和目标点必须是相同数量的二维坐标")
    if source.shape[0] < 3:
        raise ValueError("重新标定至少需要三个颗粒")
    centered = source - source.mean(axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError("三个标定颗粒不能共线")
    normalized_model = str(model).strip().lower()
    if normalized_model == "similarity":
        return _fit_similarity(source, target, determinant=1, model=normalized_model)
    if normalized_model == "reflected_similarity":
        return _fit_similarity(source, target, determinant=-1, model=normalized_model)
    if normalized_model == "auto_similarity":
        normal = _fit_similarity(source, target, determinant=1, model="similarity")
        reflected = _fit_similarity(
            source,
            target,
            determinant=-1,
            model="reflected_similarity",
        )
        return min((normal, reflected), key=lambda solution: solution.rms_error_um)
    if normalized_model == "affine":
        design = np.column_stack((source, np.ones(source.shape[0])))
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        matrix = coefficients[:2].T
        translation = coefficients[2]
        if abs(float(np.linalg.det(matrix))) < 1e-12:
            raise ValueError("标定仿射矩阵不可逆")
        return _solution_from_matrix(
            normalized_model,
            matrix,
            translation,
            source,
            target,
        )
    raise ValueError(f"不支持的标定模型：{model}")


def candidate_appearance_penalty(candidate: ParticleCandidate) -> float:
    """候选与光镜实测尺寸、信号强度的吻合度，不含"距视场中心多近"这一项。

    detect_particle_candidates 的综合评分里带有位置先验。三点标定的组合择优本来就
    用几何残差衡量位置是否自洽，再把位置先验算进外观项就成了重复计分，会把靠近视场
    中心的弱候选抬到真实颗粒前面。
    """
    return (
        0.72 * abs(math.log(max(candidate.length_ratio, 0.03)))
        + 0.24 * abs(math.log(max(candidate.width_ratio, 0.03)))
        + 0.10 * abs(math.log(max(candidate.area_ratio, 0.02)))
        - 0.035 * min(candidate.signal_z, 20.0)
    )


@dataclass(frozen=True)
class SemAnchorCandidate:
    sequence: int
    stage_xy_um: tuple[float, float]
    pixel_xy: tuple[float, float]
    detector_score: float
    shape_score: float
    candidate: ParticleCandidate
    optical_shape_source: str = "unknown"

    @property
    def combined_score(self) -> float:
        return self.detector_score + 1.35 * self.shape_score

    @property
    def appearance_score(self) -> float:
        return candidate_appearance_penalty(self.candidate) + 1.35 * self.shape_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "stageUm": {"x": self.stage_xy_um[0], "y": self.stage_xy_um[1]},
            "pixel": {"x": self.pixel_xy[0], "y": self.pixel_xy[1]},
            "detectorScore": self.detector_score,
            "shapeScore": self.shape_score,
            "opticalShapeSource": self.optical_shape_source,
            "combinedScore": self.combined_score,
            "appearanceScore": self.appearance_score,
            "candidate": self.candidate.to_dict(),
        }


def pixel_to_stage_xy(
    pixel_xy: Sequence[float],
    image_shape: Sequence[int],
    pixel_size_um: float,
    stage_center_um: Sequence[float],
    *,
    field_rotation_deg: float = 0.0,
    pixel_axis_matrix: Sequence[Sequence[float]] = ((1.0, 0.0), (0.0, 1.0)),
) -> tuple[float, float]:
    height, width = int(image_shape[0]), int(image_shape[1])
    offset = np.asarray(
        [float(pixel_xy[0]) - width / 2.0, float(pixel_xy[1]) - height / 2.0],
        dtype=np.float64,
    ) * float(pixel_size_um)
    axis_matrix = np.asarray(pixel_axis_matrix, dtype=np.float64)
    if axis_matrix.shape != (2, 2):
        raise ValueError("pixelAxisMatrix 必须是 2×2 矩阵")
    radians = math.radians(float(field_rotation_deg))
    rotation = np.asarray(
        [[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]],
        dtype=np.float64,
    )
    target = np.asarray(stage_center_um, dtype=np.float64) + rotation @ axis_matrix @ offset
    return float(target[0]), float(target[1])


def sam3_text_candidates(
    image: np.ndarray,
    *,
    expected_length_px: float,
    expected_width_px: float,
    expected_area_px: float,
    image_id: str = "",
    url: str,
    score_threshold: float = 0.25,
    timeout: float = 30.0,
) -> list[ParticleCandidate]:
    """调用 SAM3 ``/segment_text`` 获取颗粒候选，channel 标记为 ``sam3``。"""
    import urllib.error
    import urllib.request

    gray = normalize_uint8(image)
    height, width = gray.shape
    image_center = np.asarray([width / 2.0, height / 2.0], dtype=np.float64)

    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    success, buf = cv2.imencode(".png", rgb)
    if not success:
        raise ValueError("SAM3: 图片编码失败")
    png_b64 = base64.b64encode(buf).decode("ascii")

    # SAM3 服务按 image_id 复用图像缓存，固定 id 会让后续每张图都拿到第一张图的
    # 分割结果，因此必须拼上图像内容摘要保证不同画面拿到不同 id。
    digest = hashlib.sha1(buf.tobytes()).hexdigest()[:16]
    payload = json.dumps({
        "image_id": f"{image_id or 'cleanliness'}-{digest}",
        "image_png_b64": png_b64,
        "texts": ["particle", "debris"],
        "score_threshold": float(score_threshold),
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{url.rstrip('/')}/segment_text",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    if "error" in result:
        raise RuntimeError(f"SAM3 服务返回错误: {result['error']}")

    detections = result.get("detections", [])
    candidates: list[ParticleCandidate] = []
    for det in detections:
        mask_b64 = det.get("mask_png_b64", "")
        if not mask_b64:
            continue
        mask_bytes = base64.b64decode(mask_b64)
        mask_img = cv2.imdecode(
            np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE,
        )
        if mask_img is None or not mask_img.any():
            continue
        _, mask_bin = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < 4.0 or area > height * width * 0.22 or len(contour) < 3:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        center = (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )
        bx, by, bw, bh = cv2.boundingRect(contour)
        padding = max(2, min(8, int(round(expected_width_px * 0.04))))
        x1 = max(0, bx - padding)
        y1 = max(0, by - padding)
        x2 = min(width - 1, bx + bw - 1 + padding)
        y2 = min(height - 1, by + bh - 1 + padding)
        distance_px = float(np.linalg.norm(np.asarray(center) - image_center))
        (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(contour)
        observed_length = float(max(rect_w, rect_h, bw, bh))
        observed_width = float(max(1.0, min(rect_w, rect_h, bw, bh)))
        length_ratio = observed_length / max(expected_length_px, 2.0)
        width_ratio = observed_width / max(expected_width_px, 2.0)
        area_ratio = area / max(expected_area_px, 2.0)
        distance_scale = max(22.0, min(70.0, expected_length_px * 0.25 + 22.0))
        score = (
            distance_px / distance_scale
            + 0.72 * abs(math.log(max(length_ratio, 0.03)))
            + 0.24 * abs(math.log(max(width_ratio, 0.03)))
            + 0.10 * abs(math.log(max(area_ratio, 0.02)))
        )
        contour_points = tuple(
            (int(pt[0][0]), int(pt[0][1])) for pt in contour
        )
        candidates.append(
            ParticleCandidate(
                box=(x1, y1, x2, y2),
                center=center,
                contour=contour_points,
                distance_px=distance_px,
                length_ratio=length_ratio,
                width_ratio=width_ratio,
                area_ratio=area_ratio,
                contrast_z=0.0,
                signal_z=float(det.get("score", 0.5)) * 10.0,
                channel="sam3",
                score=score,
                threshold_z=0.0,
                close_kernel=0,
            )
        )
    candidates.sort(key=lambda c: c.score)
    return candidates


_YOLO_SESSION_CACHE: dict[str, Any] = {}


def _yolo_session(model_path: Path):
    cache_key = str(Path(model_path).resolve())
    session = _YOLO_SESSION_CACHE.get(cache_key)
    if session is not None:
        return session
    import onnxruntime as ort

    from backend.cleanliness_core.onnx_crypto import load_onnx_model

    available = set(ort.get_available_providers())
    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available
    ]
    model_data = load_onnx_model(cache_key)
    session_options = None
    thread_count = os.environ.get("ZEM_ORT_THREADS", "").strip()
    if thread_count:
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = max(1, int(thread_count))
        session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(model_data, sess_options=session_options, providers=providers or None)
    _YOLO_SESSION_CACHE[cache_key] = session
    return session


def _nms_indices(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    maximum_detections: int,
) -> np.ndarray:
    if not len(boxes):
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size and len(keep) < maximum_detections:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[index], x1[rest])
        yy1 = np.maximum(y1[index], y1[rest])
        xx2 = np.minimum(x2[index], x2[rest])
        yy2 = np.minimum(y2[index], y2[rest])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[index] + areas[rest] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def yolo_seg_candidates(
    image: np.ndarray,
    *,
    expected_length_px: float,
    expected_width_px: float,
    expected_area_px: float,
    model_path: Path,
    input_size: int = 1024,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    mask_threshold: float = 0.5,
    maximum_detections: int = 100,
) -> list[ParticleCandidate]:
    """Run the local YOLO segmentation ONNX model and return SEM candidates."""

    gray = normalize_uint8(image)
    height, width = gray.shape
    image_center = np.asarray([width / 2.0, height / 2.0], dtype=np.float64)
    size = max(32, int(input_size))
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    input_tensor = (
        resized.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    )
    session = _yolo_session(Path(model_path))
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    if len(outputs) < 2:
        raise ValueError("YOLO segmentation model must return predictions and prototypes")
    prediction = np.asarray(outputs[0])[0]
    prototypes = np.asarray(outputs[1])[0]
    if prediction.ndim != 2 or prototypes.ndim != 3:
        raise ValueError(
            f"Unsupported YOLO segmentation output: {prediction.shape}, {prototypes.shape}"
        )
    if prediction.shape[0] < prediction.shape[1]:
        prediction = prediction.T
    mask_channels = int(prototypes.shape[0])
    class_count = prediction.shape[1] - 4 - mask_channels
    if class_count < 1:
        raise ValueError(
            f"Cannot infer YOLO class count from output shape {prediction.shape}"
        )
    class_scores = prediction[:, 4 : 4 + class_count]
    confidences = class_scores.max(axis=1)
    selected = confidences >= float(confidence_threshold)
    prediction = prediction[selected]
    confidences = confidences[selected]
    if not len(prediction):
        return []

    xywh = prediction[:, :4].astype(np.float32)
    boxes = np.empty_like(xywh)
    boxes[:, :2] = xywh[:, :2] - xywh[:, 2:] / 2.0
    boxes[:, 2:] = xywh[:, :2] + xywh[:, 2:] / 2.0
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, size)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, size)
    keep = _nms_indices(
        boxes,
        confidences,
        float(iou_threshold),
        max(1, int(maximum_detections)),
    )
    boxes = boxes[keep]
    confidences = confidences[keep]
    coefficients = prediction[keep, 4 + class_count :]
    proto_height, proto_width = prototypes.shape[1:]
    logits = coefficients @ prototypes.reshape(mask_channels, -1)
    masks = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    masks = masks.reshape(-1, proto_height, proto_width)

    candidates: list[ParticleCandidate] = []
    for confidence, box, mask in zip(confidences, boxes, masks, strict=True):
        proto_box = box.copy()
        proto_box[[0, 2]] *= proto_width / float(size)
        proto_box[[1, 3]] *= proto_height / float(size)
        px1, py1, px2, py2 = proto_box
        columns = np.arange(proto_width)[None, :]
        rows = np.arange(proto_height)[:, None]
        mask = mask * (
            (columns >= px1)
            & (columns < px2)
            & (rows >= py1)
            & (rows < py2)
        )
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        mask_bin = np.where(mask >= float(mask_threshold), 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < 4.0 or area > height * width * 0.22 or len(contour) < 3:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        center = (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )
        bx, by, bw, bh = cv2.boundingRect(contour)
        padding = max(2, min(8, int(round(expected_width_px * 0.04))))
        x1 = max(0, bx - padding)
        y1 = max(0, by - padding)
        x2 = min(width - 1, bx + bw - 1 + padding)
        y2 = min(height - 1, by + bh - 1 + padding)
        distance_px = float(np.linalg.norm(np.asarray(center) - image_center))
        (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(contour)
        observed_length = float(max(rect_width, rect_height, bw, bh))
        observed_width = float(max(1.0, min(rect_width, rect_height, bw, bh)))
        length_ratio = observed_length / max(expected_length_px, 2.0)
        width_ratio = observed_width / max(expected_width_px, 2.0)
        area_ratio = area / max(expected_area_px, 2.0)
        distance_scale = max(22.0, min(70.0, expected_length_px * 0.25 + 22.0))
        score = (
            distance_px / distance_scale
            + 0.72 * abs(math.log(max(length_ratio, 0.03)))
            + 0.24 * abs(math.log(max(width_ratio, 0.03)))
            + 0.10 * abs(math.log(max(area_ratio, 0.02)))
        )
        candidates.append(
            ParticleCandidate(
                box=(x1, y1, x2, y2),
                center=center,
                contour=tuple(
                    (int(point[0][0]), int(point[0][1])) for point in contour
                ),
                distance_px=distance_px,
                length_ratio=length_ratio,
                width_ratio=width_ratio,
                area_ratio=area_ratio,
                contrast_z=0.0,
                signal_z=float(confidence) * 10.0,
                channel="yolo",
                score=score,
                threshold_z=float(confidence_threshold),
                close_kernel=0,
            )
        )
    candidates.sort(key=lambda candidate: candidate.score)
    return candidates


def _merge_candidates(
    classic: list[ParticleCandidate],
    extra: list[ParticleCandidate],
    expected_width_px: float,
) -> list[ParticleCandidate]:
    """合并经典检测和额外候选，经典优先，按中心距离去重。"""
    cluster_radius = max(8.0, min(36.0, expected_width_px * 0.20))
    merged = list(classic)
    for candidate in extra:
        duplicate = any(
            math.hypot(
                candidate.center[0] - kept.center[0],
                candidate.center[1] - kept.center[1],
            )
            <= cluster_radius
            for kept in merged
        )
        if not duplicate:
            merged.append(candidate)
    return merged


@dataclass(frozen=True)
class CandidateGate:
    """候选颗粒的物理可信度门槛。

    检测器的综合评分里含有"距视场中心多近"这一项，视场正中心的几个像素噪点因此
    很容易压过真实颗粒。这些噪点的尺寸相对光镜实测值小了一两个数量级，用比例门槛
    可以在参与标定拟合之前把它们直接剔除。
    """

    min_length_ratio: float = 0.35
    max_length_ratio: float = 2.8
    min_width_ratio: float = 0.18
    max_width_ratio: float = 4.5
    min_area_ratio: float = 0.12
    max_area_ratio: float = 8.0
    min_signal_z: float = 1.6
    min_contour_area_px: float = 30.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CandidateGate":
        if not payload:
            return cls()
        mapping = {
            "minLengthRatio": "min_length_ratio",
            "maxLengthRatio": "max_length_ratio",
            "minWidthRatio": "min_width_ratio",
            "maxWidthRatio": "max_width_ratio",
            "minAreaRatio": "min_area_ratio",
            "maxAreaRatio": "max_area_ratio",
            "minSignalZ": "min_signal_z",
            "minContourAreaPx": "min_contour_area_px",
        }
        values = {
            field: float(payload[key])
            for key, field in mapping.items()
            if payload.get(key) is not None
        }
        return cls(**values)

    def rejection(self, candidate: ParticleCandidate) -> str | None:
        """返回拒绝原因；None 表示候选通过门槛。"""
        if not (self.min_length_ratio <= candidate.length_ratio <= self.max_length_ratio):
            return f"长度比 {candidate.length_ratio:.3f} 超出 [{self.min_length_ratio}, {self.max_length_ratio}]"
        if not (self.min_width_ratio <= candidate.width_ratio <= self.max_width_ratio):
            return f"宽度比 {candidate.width_ratio:.3f} 超出 [{self.min_width_ratio}, {self.max_width_ratio}]"
        if not (self.min_area_ratio <= candidate.area_ratio <= self.max_area_ratio):
            return f"面积比 {candidate.area_ratio:.3f} 超出 [{self.min_area_ratio}, {self.max_area_ratio}]"
        if candidate.signal_z < self.min_signal_z:
            return f"信号强度 z={candidate.signal_z:.2f} 低于 {self.min_signal_z}"
        contour = np.asarray(candidate.contour, dtype=np.int32)
        if len(contour) < 3:
            return "轮廓点数不足"
        contour_area = float(cv2.contourArea(contour))
        if contour_area < self.min_contour_area_px:
            return f"轮廓面积 {contour_area:.0f}px² 小于 {self.min_contour_area_px:.0f}px²"
        return None

    def accepts(self, candidate: ParticleCandidate) -> bool:
        return self.rejection(candidate) is None


DEFAULT_CANDIDATE_GATE = CandidateGate()


def build_sem_anchor_candidates(
    particle: OpticalParticle,
    image: np.ndarray,
    *,
    pixel_size_um: float,
    stage_center_um: Sequence[float],
    field_rotation_deg: float = 0.0,
    pixel_axis_matrix: Sequence[Sequence[float]] = ((1.0, 0.0), (0.0, 1.0)),
    maximum_candidates: int = 6,
    segmentation: CandidateSegmentationConfig | None = None,
    gate: CandidateGate | None = None,
) -> tuple[SemAnchorCandidate, ...]:
    expected_length_px = particle.length_um / pixel_size_um
    expected_width_px = particle.width_um / pixel_size_um
    expected_area_px = particle.area_um2 / (pixel_size_um**2)
    candidates = detect_particle_candidates(
        image,
        expected_length_px=expected_length_px,
        expected_width_px=expected_width_px,
        expected_area_px=expected_area_px,
    )
    segmentation = segmentation or CandidateSegmentationConfig()
    if segmentation.backend == "sam3" and segmentation.sam3_url:
        try:
            sam3 = sam3_text_candidates(
                image,
                expected_length_px=expected_length_px,
                expected_width_px=expected_width_px,
                expected_area_px=expected_area_px,
                image_id=f"cleanliness-{particle.sequence:04d}",
                url=segmentation.sam3_url,
                score_threshold=segmentation.confidence_threshold,
            )
            candidates = _merge_candidates(candidates, sam3, expected_width_px)
            logger.debug(
                "SAM3 补充候选：经典 {} + SAM3 {} → 合并 {}",
                len(candidates) - len(sam3), len(sam3), len(candidates),
            )
        except Exception as error:
            logger.warning("SAM3 候选获取失败，降级为仅经典检测: {}", error)
    elif segmentation.backend == "yolo":
        segmentation.validate()
        try:
            yolo = yolo_seg_candidates(
                image,
                expected_length_px=expected_length_px,
                expected_width_px=expected_width_px,
                expected_area_px=expected_area_px,
                model_path=segmentation.yolo_model_path,
                input_size=segmentation.input_size,
                confidence_threshold=segmentation.confidence_threshold,
                iou_threshold=segmentation.iou_threshold,
                mask_threshold=segmentation.mask_threshold,
                maximum_detections=segmentation.maximum_detections,
            )
            classic_count = len(candidates)
            candidates = _merge_candidates(candidates, yolo, expected_width_px)
            logger.debug(
                "YOLO 补充候选：经典 {} + YOLO {} → 合并 {}",
                classic_count,
                len(yolo),
                len(candidates),
            )
        except Exception as error:
            logger.warning("YOLO 候选获取失败，降级为仅经典检测: {}", error)
    optical_shape = extract_optical_shape(particle)
    gate = gate or DEFAULT_CANDIDATE_GATE
    scored = [
        (shape_dissimilarity(optical_shape, candidate), candidate)
        for candidate in candidates
    ]
    # 通过物理门槛的候选整体排在前面，保证截断后交给标定匹配的都是真实颗粒。
    scored.sort(
        key=lambda item: (
            not gate.accepts(item[1]),
            item[1].score + 1.35 * item[0],
        )
    )
    return tuple(
        SemAnchorCandidate(
            sequence=particle.sequence,
            stage_xy_um=pixel_to_stage_xy(
                candidate.center,
                image.shape,
                pixel_size_um,
                stage_center_um,
                field_rotation_deg=field_rotation_deg,
                pixel_axis_matrix=pixel_axis_matrix,
            ),
            pixel_xy=candidate.center,
            detector_score=candidate.score,
            shape_score=shape_score,
            candidate=candidate,
            optical_shape_source=optical_shape.source if optical_shape else "unavailable",
        )
        for shape_score, candidate in scored[: max(1, int(maximum_candidates))]
    )


@dataclass(frozen=True)
class CalibrationMatchResult:
    solution: TransformSolution
    matches: tuple[SemAnchorCandidate, ...]
    hypothesis_score: float
    unreliable_sequences: tuple[int, ...] = ()


def choose_calibration_matches(
    anchors: Sequence[OpticalParticle],
    candidates_by_sequence: dict[int, Sequence[SemAnchorCandidate]],
    *,
    model: str = "reflected_similarity",
    experience: Calibration | TransformSolution | None = None,
    maximum_combinations: int = 1000,
    gate: CandidateGate | None = None,
    residual_scale_um: float = 60.0,
) -> CalibrationMatchResult:
    if len(anchors) < 3:
        raise ValueError("重新标定至少需要三个锚点颗粒")
    residual_scale_um = max(float(residual_scale_um), 1.0)
    gate = gate or DEFAULT_CANDIDATE_GATE
    candidate_groups: list[Sequence[SemAnchorCandidate]] = []
    unreliable: list[int] = []
    for particle in anchors:
        group = tuple(candidates_by_sequence.get(particle.sequence) or ())
        if not group:
            raise ValueError(f"标定颗粒 #{particle.sequence} 没有 SEM 候选")
        # 只让物理上说得通的候选参与拟合，否则视场中心的噪点会让"锚点就落在预测
        # 位置上"这一退化解同时拿到近乎为零的残差和零漂移，把真实颗粒挤掉。
        plausible = tuple(item for item in group if gate.accepts(item.candidate))
        if not plausible:
            unreliable.append(particle.sequence)
            logger.warning(
                "标定颗粒 #{} 的 {} 个 SEM 候选全部未通过物理门槛，本组标定结果不可信",
                particle.sequence, len(group),
            )
            plausible = group
        candidate_groups.append(plausible)
    combination_count = math.prod(len(group) for group in candidate_groups)
    if combination_count > maximum_combinations:
        keep = max(1, int(round(maximum_combinations ** (1 / len(candidate_groups)))))
        candidate_groups = [group[:keep] for group in candidate_groups]

    source = [[particle.ab_x_um, particle.ab_y_um] for particle in anchors]
    best: CalibrationMatchResult | None = None
    experience_matrix = None
    if isinstance(experience, Calibration):
        experience_matrix = experience.matrix
    elif isinstance(experience, TransformSolution):
        experience_matrix = experience.matrix

    for combination in itertools.product(*candidate_groups):
        target = [candidate.stage_xy_um for candidate in combination]
        try:
            solution = fit_coordinate_transform(source, target, model=model)
        except ValueError:
            continue
        descriptor_score = float(np.mean([item.appearance_score for item in combination]))
        drift_penalty = 0.0
        if experience_matrix is not None:
            denominator = max(float(np.linalg.norm(experience_matrix)), 1e-9)
            drift_penalty = 35.0 * float(
                np.linalg.norm(solution.matrix - experience_matrix) / denominator
            )
        # 残差按经验标定自身的定位噪声量级归一。经验矩阵实测平均误差就有几十微米，
        # 直接把微米数加进来会让几何项彻底压过形貌项。
        residual_penalty = 6.0 * solution.rms_error_um / residual_scale_um
        hypothesis_score = residual_penalty + 8.0 * descriptor_score + drift_penalty
        result = CalibrationMatchResult(
            solution=solution,
            matches=tuple(combination),
            hypothesis_score=hypothesis_score,
            unreliable_sequences=tuple(unreliable),
        )
        if best is None or result.hypothesis_score < best.hypothesis_score:
            best = result
    if best is None:
        raise ValueError("无法从 SEM 候选中构造有效的三颗粒标定组合")
    return best


def evaluate_transform(
    solution: TransformSolution,
    source_points: Sequence[Sequence[float]],
    observed_points: Sequence[Sequence[float]],
) -> dict[str, Any]:
    source = np.asarray(source_points, dtype=np.float64)
    observed = np.asarray(observed_points, dtype=np.float64)
    if source.shape != observed.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("验证源点和观测点必须是相同数量的二维坐标")
    predicted = (solution.matrix @ source.T).T + solution.translation
    errors = np.linalg.norm(predicted - observed, axis=1)
    return {
        "count": int(len(errors)),
        "errorsUm": [float(value) for value in errors],
        "meanErrorUm": float(np.mean(errors)) if len(errors) else None,
        "rmsErrorUm": (
            float(math.sqrt(float(np.mean(errors**2)))) if len(errors) else None
        ),
        "maxErrorUm": float(np.max(errors)) if len(errors) else None,
        "predictedStageUm": [
            {"x": float(point[0]), "y": float(point[1])} for point in predicted
        ],
    }


def _strip_json_comments(text: str) -> str:
    """去掉 JSONC 风格注释（// 与 /* */），字符串字面量内容原样保留。"""
    result: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    escape = False
    while i < n:
        ch = text[i]
        if in_string:
            result.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            result.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "/":
                i += 2
                while i < n and text[i] not in "\r\n":
                    i += 1
                continue
            if nxt == "*":
                i += 2
                while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i = min(i + 2, n)
                continue
        result.append(ch)
        i += 1
    return "".join(result)


def load_cleanliness_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    payload = json.loads(_strip_json_comments(raw))
    if not isinstance(payload, dict):
        raise ValueError("清洁度检测配置文件顶层必须是 JSON 对象")
    if int(payload.get("schemaVersion") or 0) != 1:
        raise ValueError("不支持的清洁度检测配置版本")
    return payload
