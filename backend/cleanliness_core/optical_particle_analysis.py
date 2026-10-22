from __future__ import annotations

import base64
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


SIMULATED_ELEMENTS_DISCLAIMER = "模拟元素数据，仅用于验证光镜颗粒到电镜颗粒的定位链路，不代表真实 EDS 测量结果。"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class OpticalParticle:
    sequence: int
    particle_id: str
    label: str
    ab_x_um: float
    ab_y_um: float
    length_um: float
    width_um: float
    area_um2: float
    equivalent_diameter_um: float
    image_data_base64: str
    image_width: int
    image_height: int
    # 光镜源台坐标（position 字段，sourceStage 坐标系）；旧导出可能没有
    source_x_um: float | None = None
    source_y_um: float | None = None
    mask_data_base64: str = ""
    mask_width: int = 0
    mask_height: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "particleId": self.particle_id,
            "label": self.label,
            "abPositionUm": {"x": self.ab_x_um, "y": self.ab_y_um},
            "lengthUm": self.length_um,
            "widthUm": self.width_um,
            "areaUm2": self.area_um2,
            "equivalentDiameterUm": self.equivalent_diameter_um,
            "imageWidth": self.image_width,
            "imageHeight": self.image_height,
        }


@dataclass(frozen=True)
class OpticalExport:
    path: Path
    sha256: str
    project_id: str
    workpiece_id: str
    workpiece_name: str
    coordinate_unit: str
    coordinate_space: str
    registration_mode: str
    reference_a_um: tuple[float, float] | None
    reference_b_um: tuple[float, float] | None
    particles: tuple[OpticalParticle, ...]

    @property
    def by_sequence(self) -> dict[int, OpticalParticle]:
        return {particle.sequence: particle for particle in self.particles}

    @property
    def reference_points(self) -> dict[str, tuple[float, float]]:
        points: dict[str, tuple[float, float]] = {}
        if self.reference_a_um is not None:
            points["a"] = self.reference_a_um
        if self.reference_b_um is not None:
            points["b"] = self.reference_b_um
        return points


def load_optical_export(path: Path) -> OpticalExport:
    path = Path(path).resolve()
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("光镜导出文件顶层必须是 JSON 对象")
    raw_particles = payload.get("particles")
    if not isinstance(raw_particles, list):
        raise ValueError("光镜导出文件缺少 particles 数组")

    raw_reference_points = payload.get("referencePoints")
    if raw_reference_points is not None and not isinstance(raw_reference_points, dict):
        raise ValueError("光镜导出文件的 referencePoints 必须是对象")

    def optional_reference_point(name: str) -> tuple[float, float] | None:
        raw = (raw_reference_points or {}).get(name)
        if raw is None:
            return None
        if not isinstance(raw, dict) or "x" not in raw or "y" not in raw:
            raise ValueError(f"光镜导出文件的参考点 {name.upper()} 格式无效")
        point = (float(raw["x"]), float(raw["y"]))
        if not all(math.isfinite(value) for value in point):
            raise ValueError(f"光镜导出文件的参考点 {name.upper()} 坐标无效")
        return point

    reference_a_um = optional_reference_point("a")
    reference_b_um = optional_reference_point("b")

    particles: list[OpticalParticle] = []
    seen_sequences: set[int] = set()
    for item in raw_particles:
        if not isinstance(item, dict):
            continue
        info = item.get("particleInfo") or {}
        if not bool(info.get("isValid", True)):
            continue
        sequence = int(item["sequence"])
        if sequence in seen_sequences:
            raise ValueError(f"光镜导出文件存在重复 sequence：{sequence}")
        seen_sequences.add(sequence)
        ab = item.get("abPosition") or {}
        image_data = str(item.get("imageDataBase64") or "")
        if not image_data:
            raise ValueError(f"颗粒 #{sequence} 缺少 imageDataBase64")
        source_position = item.get("position") or {}
        try:
            source_x_um = float(source_position["x"])
            source_y_um = float(source_position["y"])
        except (KeyError, TypeError, ValueError):
            source_x_um = None
            source_y_um = None
        particles.append(
            OpticalParticle(
                sequence=sequence,
                particle_id=str(item.get("particleId") or sequence),
                label=str(item.get("label") or "particle"),
                ab_x_um=float(ab["x"]),
                ab_y_um=float(ab["y"]),
                source_x_um=source_x_um,
                source_y_um=source_y_um,
                length_um=float(info.get("length") or 0.0),
                width_um=float(info.get("width") or info.get("orthoWidth") or 0.0),
                area_um2=float(info.get("area") or 0.0),
                equivalent_diameter_um=float(
                    info.get("equivalentDiameter") or info.get("diameter") or 0.0
                ),
                image_data_base64=image_data,
                image_width=int(item.get("imageWidth") or 0),
                image_height=int(item.get("imageHeight") or 0),
                mask_data_base64=str(item.get("maskDataBase64") or ""),
                mask_width=int(item.get("maskWidth") or 0),
                mask_height=int(item.get("maskHeight") or 0),
            )
        )
    if not particles:
        raise ValueError("光镜导出文件没有有效颗粒")

    return OpticalExport(
        path=path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        project_id=str(payload.get("projectId") or ""),
        workpiece_id=str(payload.get("workpieceId") or ""),
        workpiece_name=str(payload.get("workpieceName") or ""),
        coordinate_unit=str(payload.get("coordinateUnit") or ""),
        coordinate_space=str(payload.get("coordinateSpace") or ""),
        registration_mode=str(payload.get("registrationMode") or ""),
        reference_a_um=reference_a_um,
        reference_b_um=reference_b_um,
        particles=tuple(sorted(particles, key=lambda particle: particle.sequence)),
    )


@dataclass(frozen=True)
class Calibration:
    path: Path
    sha256: str
    calibration_id: str
    matrix: np.ndarray
    translation: np.ndarray
    roi_center: np.ndarray
    roi_radius_um: float
    device: dict[str, Any]
    source: dict[str, Any]
    imaging: dict[str, Any]
    audit: dict[str, Any]

    def target_stage(self, particle: OpticalParticle) -> tuple[float, float]:
        target = self.matrix @ np.asarray(
            [particle.ab_x_um, particle.ab_y_um],
            dtype=np.float64,
        ) + self.translation
        return float(target[0]), float(target[1])

    def target_inside_roi(self, target: tuple[float, float]) -> bool:
        return bool(
            np.linalg.norm(np.asarray(target, dtype=np.float64) - self.roi_center)
            <= self.roi_radius_um
        )

    def validate_export(self, export: OpticalExport) -> None:
        expected_unit = str(self.source.get("coordinateUnit") or "um")
        if export.coordinate_unit != expected_unit:
            raise ValueError(
                f"光镜坐标单位不匹配：{export.coordinate_unit!r} != {expected_unit!r}"
            )
        expected_space = str(self.source.get("coordinateSpace") or "")
        if expected_space and export.coordinate_space != expected_space:
            raise ValueError(
                f"光镜坐标空间不匹配：{export.coordinate_space!r} != {expected_space!r}"
            )
        expected_workpiece = str(self.source.get("workpieceId") or "")
        if expected_workpiece and export.workpiece_id != expected_workpiece:
            raise ValueError(
                f"标定工件不匹配：{export.workpiece_id!r} != {expected_workpiece!r}"
            )
        expected_project = str(self.source.get("projectId") or "")
        if expected_project and export.project_id != expected_project:
            raise ValueError(
                f"标定项目不匹配：{export.project_id!r} != {expected_project!r}"
            )
        expected_sha256 = str(self.source.get("inputSha256") or "")
        if bool(self.source.get("requireExactInput", False)) and expected_sha256:
            if export.sha256.lower() != expected_sha256.lower():
                raise ValueError("光镜输入文件校验值与标定文件不一致")

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "calibrationId": self.calibration_id,
            "path": str(self.path),
            "sha256": self.sha256,
            "matrix": self.matrix.tolist(),
            "translation": self.translation.tolist(),
            "validStageRoi": {
                "centerUm": self.roi_center.tolist(),
                "radiusUm": self.roi_radius_um,
            },
            "device": self.device,
            "source": self.source,
            "imaging": self.imaging,
            "audit": self.audit,
        }


def load_calibration(path: Path) -> Calibration:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    transform = payload.get("transform") or {}
    matrix = np.asarray(transform.get("matrix"), dtype=np.float64)
    translation = np.asarray(transform.get("translation"), dtype=np.float64)
    roi = payload.get("validStageRoi") or {}
    roi_center = np.asarray(roi.get("centerUm"), dtype=np.float64)
    roi_radius = float(roi.get("radiusUm") or 0.0)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("标定矩阵必须是有限的 2×2 矩阵")
    if translation.shape != (2,) or not np.all(np.isfinite(translation)):
        raise ValueError("标定平移必须是有限的二维向量")
    if roi_center.shape != (2,) or roi_radius <= 0:
        raise ValueError("标定文件的有效 Stage ROI 无效")
    if abs(float(np.linalg.det(matrix))) < 1e-9:
        raise ValueError("标定矩阵不可逆")
    return Calibration(
        path=path,
        sha256=sha256_file(path),
        calibration_id=str(payload.get("calibrationId") or path.stem),
        matrix=matrix,
        translation=translation,
        roi_center=roi_center,
        roi_radius_um=roi_radius,
        device=dict(payload.get("device") or {}),
        source=dict(payload.get("source") or {}),
        imaging=dict(payload.get("imaging") or {}),
        audit=dict(payload.get("audit") or {}),
    )


def decode_optical_image(particle: OpticalParticle, output_path: Path) -> Path:
    encoded = particle.image_data_base64
    if "," in encoded and encoded.lstrip().lower().startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as error:
        raise ValueError(f"颗粒 #{particle.sequence} 的光镜图片 Base64 无效") from error
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ValueError(f"颗粒 #{particle.sequence} 的光镜图片无法解码")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"保存光镜颗粒图失败：{output_path}")
    return output_path


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
        else:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if array.dtype == np.uint8:
        return array.copy()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape[:2], dtype=np.uint8)
    low, high = np.percentile(finite.astype(np.float64), [0.5, 99.5])
    if high <= low:
        return np.zeros(array.shape[:2], dtype=np.uint8)
    scaled = (array.astype(np.float64) - low) * 255.0 / (high - low)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _odd_kernel(value: float, maximum: int = 17) -> int:
    rounded = max(3, min(maximum, int(round(value))))
    return rounded if rounded % 2 else rounded + 1


def _box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left + 1) * max(0, bottom - top + 1)
    first_area = max(0, first[2] - first[0] + 1) * max(
        0, first[3] - first[1] + 1
    )
    second_area = max(0, second[2] - second[0] + 1) * max(
        0, second[3] - second[1] + 1
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class ParticleCandidate:
    box: tuple[int, int, int, int]
    center: tuple[float, float]
    contour: tuple[tuple[int, int], ...]
    distance_px: float
    length_ratio: float
    width_ratio: float
    area_ratio: float
    contrast_z: float
    signal_z: float
    channel: str
    score: float
    threshold_z: float
    close_kernel: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "boxPx": list(self.box),
            "centerPx": list(self.center),
            "distancePx": self.distance_px,
            "lengthRatio": self.length_ratio,
            "widthRatio": self.width_ratio,
            "areaRatio": self.area_ratio,
            "contrastZ": self.contrast_z,
            "signalZ": self.signal_z,
            "channel": self.channel,
            "score": self.score,
            "thresholdZ": self.threshold_z,
            "closeKernelPx": self.close_kernel,
        }


def _robust_sigma(values: np.ndarray) -> float:
    median = float(np.median(values))
    return max(
        1e-6,
        1.4826 * float(np.median(np.abs(values - median))),
    )


def _channel_responses(
    gray: np.ndarray,
    expected_width_px: float,
) -> list[tuple[str, np.ndarray]]:
    """构建亮目标、暗目标和纹理缺失三个检测响应通道。

    - bright：金属等高原子序数颗粒在 BSE 下比背景亮。
    - dark：低原子序数颗粒（纤维、盐、塑料薄片）比滤膜背景暗。
    - flat：颗粒内部比编织滤膜纹理更平滑，局部高频能量显著低于背景。

    每个通道都归一化成稳健 z 分数图，>0 表示"更像颗粒"。
    """
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    background_kernel = _odd_kernel(
        max(expected_width_px * 2.5, 63.0),
        maximum=201,
    )
    background = cv2.medianBlur(smooth, background_kernel)
    signed = smooth.astype(np.float32) - background.astype(np.float32)

    region_kernel = _odd_kernel(
        max(3.0, expected_width_px * 0.22),
        maximum=31,
    )
    region_signed = cv2.blur(signed, (region_kernel, region_kernel))

    bright = signed / _robust_sigma(signed)
    dark = -region_signed / _robust_sigma(region_signed)

    highpass = gray.astype(np.float32) - cv2.GaussianBlur(gray, (0, 0), 2.5)
    texture = np.sqrt(
        cv2.blur(highpass * highpass, (region_kernel, region_kernel))
    )
    flat_response = float(np.median(texture)) - texture
    flat = flat_response / _robust_sigma(flat_response)
    return [("bright", bright), ("dark", dark), ("flat", flat)]


def detect_particle_candidates(
    image: np.ndarray,
    *,
    expected_length_px: float,
    expected_width_px: float,
    expected_area_px: float,
    maximum_distance_px: float | None = None,
) -> list[ParticleCandidate]:
    gray = normalize_uint8(image)
    height, width = gray.shape
    image_center = np.asarray([width / 2.0, height / 2.0], dtype=np.float64)
    maximum_distance_px = (
        float(maximum_distance_px)
        if maximum_distance_px is not None
        else 0.48 * min(height, width)
    )
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    smooth_median = float(np.median(smooth))
    smooth_sigma = max(
        1.0,
        _robust_sigma(smooth.astype(np.float64)),
    )
    close_kernels = sorted(
        {
            3,
            _odd_kernel(expected_width_px * 0.16, maximum=17),
            _odd_kernel(expected_width_px * 0.30, maximum=35),
        }
    )
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    minimum_component = max(4, int(round(expected_area_px * 0.002)))
    candidates: list[ParticleCandidate] = []
    seen: set[tuple[str, int, int, int, int, int]] = set()

    for channel, response in _channel_responses(gray, expected_width_px):
        response_sigma = _robust_sigma(response)
        for threshold_z in (6.0, 4.2, 3.0, 2.2, 1.6):
            base_mask = np.where(
                response >= threshold_z * response_sigma,
                255,
                0,
            ).astype(np.uint8)
            if channel != "bright":
                base_mask = cv2.morphologyEx(
                    base_mask,
                    cv2.MORPH_OPEN,
                    open_kernel,
                )
            for kernel_size in close_kernels:
                if kernel_size > 1:
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (kernel_size, kernel_size),
                    )
                    mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, kernel)
                else:
                    mask = base_mask
                component_count, labels, stats, centroids = (
                    cv2.connectedComponentsWithStats(mask, connectivity=8)
                )
                for component_label in range(1, component_count):
                    x, y, box_width, box_height, area = map(
                        int, stats[component_label]
                    )
                    if (
                        area < minimum_component
                        or area > height * width * 0.22
                    ):
                        continue
                    padding = max(
                        2,
                        min(8, int(round(expected_width_px * 0.04))),
                    )
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(width - 1, x + box_width - 1 + padding)
                    y2 = min(height - 1, y + box_height - 1 + padding)
                    signature = (channel, x1, y1, x2, y2, kernel_size)
                    if signature in seen:
                        continue
                    seen.add(signature)

                    center = (
                        float(centroids[component_label][0]),
                        float(centroids[component_label][1]),
                    )
                    distance_px = float(
                        np.linalg.norm(np.asarray(center) - image_center)
                    )
                    if distance_px > maximum_distance_px:
                        continue

                    label_roi = labels[y : y + box_height, x : x + box_width]
                    component_roi = label_roi == component_label
                    component_mask = np.where(component_roi, 255, 0).astype(
                        np.uint8
                    )
                    contours, _ = cv2.findContours(
                        component_mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    if not contours:
                        continue
                    local_contour = max(contours, key=cv2.contourArea)
                    if len(local_contour) < 3:
                        continue
                    contour = local_contour + np.asarray(
                        [[[x, y]]],
                        dtype=np.int32,
                    )
                    (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(
                        contour
                    )
                    observed_length = float(
                        max(rect_width, rect_height, box_width, box_height)
                    )
                    observed_width = float(
                        max(
                            1.0,
                            min(
                                rect_width,
                                rect_height,
                                box_width,
                                box_height,
                            ),
                        )
                    )
                    length_ratio = observed_length / max(
                        expected_length_px, 2.0
                    )
                    width_ratio = observed_width / max(expected_width_px, 2.0)
                    area_ratio = float(area) / max(expected_area_px, 2.0)
                    smooth_roi = smooth[y : y + box_height, x : x + box_width]
                    component_pixels = smooth_roi[component_roi]
                    contrast_z = (
                        (float(np.mean(component_pixels)) - smooth_median)
                        / smooth_sigma
                        if component_pixels.size
                        else 0.0
                    )
                    response_roi = response[
                        y : y + box_height, x : x + box_width
                    ]
                    signal_z = (
                        float(np.mean(response_roi[component_roi]))
                        / response_sigma
                        if component_pixels.size
                        else 0.0
                    )
                    distance_scale = max(
                        22.0,
                        min(70.0, expected_length_px * 0.25 + 22.0),
                    )
                    score = (
                        distance_px / distance_scale
                        + 0.72 * abs(math.log(max(length_ratio, 0.03)))
                        + 0.24 * abs(math.log(max(width_ratio, 0.03)))
                        + 0.10 * abs(math.log(max(area_ratio, 0.02)))
                        - 0.035 * min(signal_z, 20.0)
                    )
                    contour_points = tuple(
                        (int(point[0][0]), int(point[0][1]))
                        for point in contour
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
                            contrast_z=contrast_z,
                            signal_z=signal_z,
                            channel=channel,
                            score=score,
                            threshold_z=threshold_z,
                            close_kernel=kernel_size,
                        )
                    )

    candidates.sort(key=lambda candidate: candidate.score)
    deduplicated: list[ParticleCandidate] = []
    cluster_radius = max(8.0, min(36.0, expected_width_px * 0.20))
    for candidate in candidates:
        duplicate = False
        for kept in deduplicated:
            center_distance = math.hypot(
                candidate.center[0] - kept.center[0],
                candidate.center[1] - kept.center[1],
            )
            if center_distance <= cluster_radius and _box_iou(
                candidate.box, kept.box
            ) >= 0.15:
                duplicate = True
                break
        if not duplicate:
            deduplicated.append(candidate)
        if len(deduplicated) >= 40:
            break
    return deduplicated


@dataclass(frozen=True)
class MatchDecision:
    status: str
    confidence: str
    reason: str
    best: ParticleCandidate | None
    second_score: float | None
    score_margin: float | None
    candidate_count: int

    @property
    def matched(self) -> bool:
        return self.status in {"MATCHED_HIGH", "MATCHED_MEDIUM"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "confidence": self.confidence,
            "reason": self.reason,
            "best": self.best.to_dict() if self.best is not None else None,
            "secondScore": self.second_score,
            "scoreMargin": self.score_margin,
            "candidateCount": self.candidate_count,
        }


def decide_particle_match(
    candidates: list[ParticleCandidate],
) -> MatchDecision:
    if not candidates:
        return MatchDecision(
            status="FAILED_NOT_FOUND",
            confidence="low",
            reason="搜索图中没有产生可用颗粒候选",
            best=None,
            second_score=None,
            score_margin=None,
            candidate_count=0,
        )
    best = candidates[0]
    second_score = candidates[1].score if len(candidates) > 1 else None
    margin = second_score - best.score if second_score is not None else None
    high_size = (
        0.30 <= best.length_ratio <= 2.8
        and 0.12 <= best.width_ratio <= 4.5
        and 0.04 <= best.area_ratio <= 8.0
    )
    medium_size = (
        0.25 <= best.length_ratio <= 3.5
        and 0.08 <= best.width_ratio <= 5.5
        and 0.02 <= best.area_ratio <= 12.0
    )
    ambiguous = margin is not None and margin < 0.08
    if (
        high_size
        and best.signal_z >= 3.0
        and best.score <= 4.0
        and not ambiguous
    ):
        return MatchDecision(
            status="MATCHED_HIGH",
            confidence="high",
            reason=(
                "位置、物理尺寸和检测通道信号强度综合评分通过高置信度门限"
                f"（{best.channel} 通道）"
            ),
            best=best,
            second_score=second_score,
            score_margin=margin,
            candidate_count=len(candidates),
        )
    if medium_size and best.signal_z >= 1.6 and best.score <= 4.8:
        reason = (
            "第一、第二候选分差较小，已保守降为中置信度"
            if ambiguous
            else f"候选通过宽松尺寸和信号强度门限（{best.channel} 通道）"
        )
        return MatchDecision(
            status="MATCHED_MEDIUM",
            confidence="medium",
            reason=reason,
            best=best,
            second_score=second_score,
            score_margin=margin,
            candidate_count=len(candidates),
        )
    return MatchDecision(
        status="FAILED_LOW_CONFIDENCE",
        confidence="low",
        reason="最佳候选的尺寸、信号强度或综合评分不足，未自动确认目标身份",
        best=best,
        second_score=second_score,
        score_margin=margin,
        candidate_count=len(candidates),
    )


def candidate_safe_point(
    image_shape: tuple[int, int],
    candidate: ParticleCandidate,
) -> tuple[tuple[int, int], float]:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    contour = np.asarray(candidate.contour, dtype=np.int32)
    cv2.fillPoly(mask, [contour], 255)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, maximum, _, maximum_location = cv2.minMaxLoc(distance)
    return (int(maximum_location[0]), int(maximum_location[1])), float(maximum)


def annotate_sem_match(
    image: np.ndarray,
    particle: OpticalParticle,
    decision: MatchDecision,
    annotated_path: Path,
    crop_path: Path,
) -> dict[str, Any]:
    gray = normalize_uint8(image)
    clean_sem = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canvas = clean_sem.copy()
    height, width = gray.shape
    image_center = (int(round(width / 2.0)), int(round(height / 2.0)))
    cv2.drawMarker(
        canvas,
        image_center,
        (0, 220, 255),
        cv2.MARKER_CROSS,
        24,
        2,
        cv2.LINE_AA,
    )
    safe_point: tuple[int, int] | None = None
    safe_distance_px: float | None = None
    crop_box = (0, 0, width - 1, height - 1)
    if decision.best is not None:
        candidate = decision.best
        x1, y1, x2, y2 = candidate.box
        crop_box = candidate.box
        color = (
            (80, 230, 120)
            if decision.matched
            else (60, 80, 255)
        )
        contour = np.asarray(candidate.contour, dtype=np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [contour], color)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
        cv2.polylines(canvas, [contour], True, color, 2)
        particle_center = (
            int(round(candidate.center[0])),
            int(round(candidate.center[1])),
        )
        # 只有几个像素的候选画出来几乎不可见，补一个外接框让人一眼看出选中的是什么。
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        if max(x2 - x1, y2 - y1) < 24:
            cv2.circle(canvas, particle_center, 16, color, 1, cv2.LINE_AA)
        cv2.drawMarker(
            canvas,
            particle_center,
            color,
            cv2.MARKER_CROSS,
            20,
            2,
            cv2.LINE_AA,
        )
        cv2.arrowedLine(
            canvas,
            image_center,
            particle_center,
            (80, 150, 255),
            2,
            cv2.LINE_AA,
            tipLength=0.12,
        )
        safe_point, safe_distance_px = candidate_safe_point(
            gray.shape,
            candidate,
        )
        eds_color = (255, 80, 220)
        cv2.drawMarker(
            canvas,
            safe_point,
            eds_color,
            cv2.MARKER_TILTED_CROSS,
            16,
            2,
            cv2.LINE_AA,
        )
        # 标注电子束实际打点位置，报告中可据此核对 EDS 偏转落点。
        label_xy = (safe_point[0] + 10, max(14, safe_point[1] - 8))
        cv2.putText(
            canvas,
            "EDS",
            label_xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            eds_color,
            1,
            cv2.LINE_AA,
        )

    lines = [
        f"Optical #{particle.sequence}  {decision.status}  "
        f"candidates={decision.candidate_count}"
    ]
    if decision.best is not None:
        best = decision.best
        lines.append(
            f"ch={best.channel} len={best.length_ratio:.2f}x "
            f"area={best.area_ratio:.3f}x signalZ={best.signal_z:.1f} "
            f"dist={best.distance_px:.0f}px"
        )
    banner_height = 26 * len(lines) + 8
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width - 1, banner_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (10, 24 + 24 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

    annotated_path = Path(annotated_path)
    crop_path = Path(crop_path)
    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(annotated_path), canvas):
        raise OSError(f"保存 SEM 标注图失败：{annotated_path}")

    x1, y1, x2, y2 = crop_box
    padding = max(24, int(round(max(x2 - x1 + 1, y2 - y1 + 1) * 0.35)))
    cx1 = max(0, x1 - padding)
    cy1 = max(0, y1 - padding)
    cx2 = min(width, x2 + 1 + padding)
    cy2 = min(height, y2 + 1 + padding)
    # 裁剪带轮廓/EDS 打点的标注图，便于报告核对算法识别结果与打点位置。
    crop = canvas[cy1:cy2, cx1:cx2]
    if crop.size == 0 or not cv2.imwrite(str(crop_path), crop):
        raise OSError(f"保存 SEM 颗粒图失败：{crop_path}")
    return {
        "annotatedImage": str(annotated_path),
        "particleImage": str(crop_path),
        "safePointPx": list(safe_point) if safe_point is not None else None,
        "safeBoundaryDistancePx": safe_distance_px,
    }


def deterministic_fake_elements(particle: OpticalParticle) -> dict[str, Any]:
    seed_bytes = hashlib.sha256(
        f"{particle.particle_id}:{particle.sequence}".encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
    recipes = (
        ("模拟不锈钢", (("Fe", 68.0), ("Cr", 18.0), ("Ni", 10.0), ("Mn", 4.0))),
        ("模拟铝合金", (("Al", 86.0), ("Mg", 6.0), ("Si", 5.0), ("Fe", 3.0))),
        ("模拟铜合金", (("Cu", 73.0), ("Zn", 20.0), ("Sn", 5.0), ("Fe", 2.0))),
    )
    classification, recipe = recipes[seed_bytes[8] % len(recipes)]
    values = [
        max(0.2, baseline * (1.0 + rng.uniform(-0.10, 0.10)))
        for _, baseline in recipe
    ]
    total = sum(values)
    rounded = [round(value * 100.0 / total, 2) for value in values]
    rounded[0] = round(rounded[0] + (100.0 - sum(rounded)), 2)
    elements = [
        {"symbol": symbol, "value": value, "unit": "Weight %"}
        for (symbol, _), value in zip(recipe, rounded, strict=True)
    ]
    return {
        "mode": "SIMULATED",
        "classification": classification,
        "elements": elements,
        "disclaimer": SIMULATED_ELEMENTS_DISCLAIMER,
    }


def select_particles(
    export: OpticalExport,
    *,
    sequences: Iterable[int] | None = None,
    sample_count: int | None = 10,
    exclude_sequences: Iterable[int] = (1, 2),
) -> list[OpticalParticle]:
    by_sequence = export.by_sequence
    if sequences is not None:
        selected: list[OpticalParticle] = []
        for sequence in sequences:
            sequence = int(sequence)
            if sequence not in by_sequence:
                raise KeyError(f"光镜导出文件不存在颗粒 #{sequence}")
            selected.append(by_sequence[sequence])
        if not selected:
            raise ValueError("显式颗粒编号列表不能为空")
        return selected

    excluded = {int(value) for value in exclude_sequences}
    candidates = [
        particle
        for particle in export.particles
        if particle.sequence not in excluded
    ]
    candidates.sort(
        key=lambda particle: (particle.area_um2, particle.length_um),
        reverse=True,
    )
    if sample_count is None or sample_count <= 0:
        return candidates
    return candidates[: min(int(sample_count), len(candidates))]


def nearest_neighbor_order(
    particles: Iterable[OpticalParticle],
    calibration: Calibration,
    start_position: tuple[float, float],
) -> list[OpticalParticle]:
    remaining = list(particles)
    ordered: list[OpticalParticle] = []
    current = np.asarray(start_position, dtype=np.float64)
    while remaining:
        next_particle = min(
            remaining,
            key=lambda particle: float(
                np.linalg.norm(
                    np.asarray(
                        calibration.target_stage(particle),
                        dtype=np.float64,
                    )
                    - current
                )
            ),
        )
        ordered.append(next_particle)
        current = np.asarray(
            calibration.target_stage(next_particle),
            dtype=np.float64,
        )
        remaining.remove(next_particle)
    return ordered
