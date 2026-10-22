from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import onnxruntime as ort

from backend.cleanliness_core.optical_particle_analysis import OpticalExport


@dataclass(frozen=True)
class DetectionBox:
    class_id: int
    confidence: float
    box_xyxy: tuple[float, float, float, float]

    @property
    def center_xy(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box_xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def translated(self, dx: float, dy: float) -> DetectionBox:
        x1, y1, x2, y2 = self.box_xyxy
        return DetectionBox(
            class_id=self.class_id,
            confidence=self.confidence,
            box_xyxy=(x1 + dx, y1 + dy, x2 + dx, y2 + dy),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "classId": self.class_id,
            "confidence": self.confidence,
            "boxXyxy": list(self.box_xyxy),
            "centerPixel": list(self.center_xy),
        }


@dataclass(frozen=True)
class NavigationABDetection:
    image_shape: tuple[int, int]
    fixture: DetectionBox
    crop_box_xyxy: tuple[int, int, int, int]
    holes: tuple[DetectionBox, ...]
    membranes: tuple[DetectionBox, ...]
    fixture_model: dict[str, Any]
    hole_model: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "imageShape": list(self.image_shape),
            "fixture": self.fixture.to_dict(),
            "cropBoxXyxy": list(self.crop_box_xyxy),
            "holes": [item.to_dict() for item in self.holes],
            "membranes": [item.to_dict() for item in self.membranes],
            "models": {
                "fixture": dict(self.fixture_model),
                "hole": dict(self.hole_model),
            },
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bgr_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    elif array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"导航图维度无效：{array.shape}")

    if array.dtype != np.uint8:
        float_array = array.astype(np.float32)
        minimum = float(np.min(float_array)) if float_array.size else 0.0
        maximum = float(np.max(float_array)) if float_array.size else 0.0
        if maximum <= minimum:
            array = np.zeros(float_array.shape, dtype=np.uint8)
        else:
            normalized = (float_array - minimum) / (maximum - minimum)
            array = np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _letterbox(
    image: np.ndarray,
    input_size: int,
) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("导航图尺寸无效")
    ratio = min(input_size / height, input_size / width)
    resized_width = max(1, int(round(width * ratio)))
    resized_height = max(1, int(round(height * ratio)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    delta_width = (input_size - resized_width) / 2.0
    delta_height = (input_size - resized_height) / 2.0
    left = int(round(delta_width - 0.1))
    right = int(round(delta_width + 0.1))
    top = int(round(delta_height - 0.1))
    bottom = int(round(delta_height + 0.1))
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    if padded.shape[:2] != (input_size, input_size):
        raise RuntimeError(f"letterbox 输出尺寸无效：{padded.shape[:2]}")
    tensor = (
        padded[:, :, ::-1]
        .transpose(2, 0, 1)[None]
        .astype(np.float32)
        / 255.0
    )
    return np.ascontiguousarray(tensor), ratio, left, top


def _box_iou(reference: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(reference[0], boxes[:, 0])
    y1 = np.maximum(reference[1], boxes[:, 1])
    x2 = np.minimum(reference[2], boxes[:, 2])
    y2 = np.minimum(reference[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    reference_area = max(
        0.0,
        float(reference[2] - reference[0]),
    ) * max(0.0, float(reference[3] - reference[1]))
    box_areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0,
        boxes[:, 3] - boxes[:, 1],
    )
    union = reference_area + box_areas - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def _class_aware_nms(
    detections: Sequence[DetectionBox],
    iou_threshold: float,
) -> tuple[DetectionBox, ...]:
    kept: list[DetectionBox] = []
    class_ids = sorted({item.class_id for item in detections})
    for class_id in class_ids:
        class_items = sorted(
            (item for item in detections if item.class_id == class_id),
            key=lambda item: item.confidence,
            reverse=True,
        )
        while class_items:
            best = class_items.pop(0)
            kept.append(best)
            if not class_items:
                break
            remaining_boxes = np.asarray(
                [item.box_xyxy for item in class_items],
                dtype=np.float64,
            )
            overlap = _box_iou(
                np.asarray(best.box_xyxy, dtype=np.float64),
                remaining_boxes,
            )
            class_items = [
                item
                for item, iou in zip(class_items, overlap, strict=True)
                if float(iou) <= iou_threshold
            ]
    return tuple(sorted(kept, key=lambda item: item.confidence, reverse=True))


class OnnxYoloDetector:
    """运行固定 640 输入、输出为 ``[x1,y1,x2,y2,score,class]`` 的检测模型。"""

    def __init__(self, model_path: Path, input_size: int = 640) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX 模型不存在：{self.model_path}")
        self.input_size = int(input_size)
        if self.input_size <= 0:
            raise ValueError("模型输入尺寸必须大于 0")

        if self.model_path.name.endswith(".enc.onnx"):
            from backend.cleanliness_core.onnx_crypto import load_onnx_model

            model_data: bytes | str = load_onnx_model(str(self.model_path))
        else:
            model_data = str(self.model_path)
        available = set(ort.get_available_providers())
        providers = [
            provider
            for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in available
        ]
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            model_data,
            providers=providers or None,
            sess_options=options,
        )
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"检测模型应只有一个输入，实际为 {len(inputs)} 个")
        self.input_name = inputs[0].name
        self.sha256 = _sha256_file(self.model_path)

    @property
    def audit(self) -> dict[str, Any]:
        return {
            "path": str(self.model_path),
            "name": self.model_path.name,
            "sha256": self.sha256,
            "inputSize": self.input_size,
            "runtime": "onnxruntime",
        }

    def predict(
        self,
        image: np.ndarray,
        *,
        confidence_threshold: float,
        nms_iou_threshold: float,
    ) -> tuple[DetectionBox, ...]:
        source = _as_bgr_uint8(image)
        tensor, ratio, pad_x, pad_y = _letterbox(source, self.input_size)
        output = np.asarray(
            self.session.run(None, {self.input_name: tensor})[0]
        )
        if output.ndim != 3 or output.shape[0] != 1:
            raise ValueError(f"检测模型输出维度不受支持：{output.shape}")
        rows = output[0]
        if rows.shape[-1] != 6 and rows.shape[0] == 6:
            rows = rows.T
        if rows.ndim != 2 or rows.shape[1] != 6:
            raise ValueError(
                "检测模型必须输出 Nx6 的 "
                f"[x1,y1,x2,y2,score,class]，实际为 {output.shape}"
            )

        height, width = source.shape[:2]
        candidates: list[DetectionBox] = []
        for row in rows:
            if not np.all(np.isfinite(row)):
                continue
            score = float(row[4])
            if score < confidence_threshold:
                continue
            x1 = float(np.clip((row[0] - pad_x) / ratio, 0.0, width))
            y1 = float(np.clip((row[1] - pad_y) / ratio, 0.0, height))
            x2 = float(np.clip((row[2] - pad_x) / ratio, 0.0, width))
            y2 = float(np.clip((row[3] - pad_y) / ratio, 0.0, height))
            if x2 <= x1 or y2 <= y1:
                continue
            candidates.append(
                DetectionBox(
                    class_id=int(round(float(row[5]))),
                    confidence=score,
                    box_xyxy=(x1, y1, x2, y2),
                )
            )
        return _class_aware_nms(candidates, float(nms_iou_threshold))


@lru_cache(maxsize=4)
def _cached_detector(model_path: str, input_size: int) -> OnnxYoloDetector:
    return OnnxYoloDetector(Path(model_path), input_size)


def _expanded_crop_box(
    box_xyxy: Sequence[float],
    image_width: int,
    image_height: int,
    margin: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    margin_x = (x2 - x1) * margin
    margin_y = (y2 - y1) * margin
    return (
        max(0, int(round(x1 - margin_x))),
        max(0, int(round(y1 - margin_y))),
        min(image_width, int(round(x2 + margin_x))),
        min(image_height, int(round(y2 + margin_y))),
    )


def _deduplicate_by_center(
    detections: Sequence[DetectionBox],
    minimum_distance_px: float,
) -> tuple[DetectionBox, ...]:
    chosen: list[DetectionBox] = []
    for detection in sorted(
        detections,
        key=lambda item: item.confidence,
        reverse=True,
    ):
        center_x, center_y = detection.center_xy
        if all(
            math.hypot(center_x - other.center_xy[0], center_y - other.center_xy[1])
            >= minimum_distance_px
            for other in chosen
        ):
            chosen.append(detection)
    return tuple(chosen)


def detect_navigation_ab(
    image: np.ndarray,
    *,
    fixture_model_path: Path,
    hole_model_path: Path,
    input_size: int = 640,
    fixture_confidence: float = 0.7,
    hole_confidence: float = 0.25,
    nms_iou_threshold: float = 0.45,
    crop_margin: float = 0.12,
    dedup_distance_px: float = 30.0,
) -> NavigationABDetection:
    source = _as_bgr_uint8(image)
    height, width = source.shape[:2]
    fixture_detector = _cached_detector(
        str(Path(fixture_model_path).resolve()),
        int(input_size),
    )
    hole_detector = _cached_detector(
        str(Path(hole_model_path).resolve()),
        int(input_size),
    )
    fixture_detections = fixture_detector.predict(
        source,
        confidence_threshold=float(fixture_confidence),
        nms_iou_threshold=float(nms_iou_threshold),
    )
    if not fixture_detections:
        raise ValueError("导航图中未检测到夹具")
    fixture = fixture_detections[0]
    crop_box = _expanded_crop_box(
        fixture.box_xyxy,
        width,
        height,
        float(crop_margin),
    )
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise ValueError("夹具检测框无法生成有效裁剪区域")
    crop = source[crop_y1:crop_y2, crop_x1:crop_x2]
    local_detections = hole_detector.predict(
        crop,
        confidence_threshold=float(hole_confidence),
        nms_iou_threshold=float(nms_iou_threshold),
    )
    original_detections = tuple(
        item.translated(crop_x1, crop_y1) for item in local_detections
    )
    holes = _deduplicate_by_center(
        [item for item in original_detections if item.class_id == 1],
        float(dedup_distance_px),
    )
    membranes = tuple(
        item for item in original_detections if item.class_id == 0
    )
    if len(holes) < 2:
        raise ValueError(f"AB 孔候选不足两个（实际 {len(holes)} 个）")
    return NavigationABDetection(
        image_shape=(height, width),
        fixture=fixture,
        crop_box_xyxy=crop_box,
        holes=holes,
        membranes=membranes,
        fixture_model=fixture_detector.audit,
        hole_model=hole_detector.audit,
    )


def nav_pixel_to_stage(
    pixel_xy: Sequence[float],
    image_shape: Sequence[int],
    bounds: Sequence[float],
) -> tuple[float, float]:
    if len(pixel_xy) != 2 or len(image_shape) < 2 or len(bounds) != 4:
        raise ValueError("导航图坐标换算参数格式无效")
    pixel_x, pixel_y = (float(value) for value in pixel_xy)
    height, width = float(image_shape[0]), float(image_shape[1])
    x_min, y_min, x_max, y_max = (float(value) for value in bounds)
    values = (
        pixel_x,
        pixel_y,
        height,
        width,
        x_min,
        y_min,
        x_max,
        y_max,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("导航图坐标换算参数包含无效数值")
    if width <= 0 or height <= 0 or x_max <= x_min or y_max <= y_min:
        raise ValueError("导航图尺寸或电机边界无效")
    if not (0.0 <= pixel_x <= width and 0.0 <= pixel_y <= height):
        raise ValueError(
            f"导航图像素坐标超出范围：({pixel_x:.3f}, {pixel_y:.3f})"
        )
    normalized_x = pixel_x / width
    normalized_y = pixel_y / height
    stage_x = x_max + normalized_x * (x_min - x_max)
    stage_y = y_min + normalized_y * (y_max - y_min)
    return float(stage_x), float(stage_y)


def validate_ab_export(export: OpticalExport) -> None:
    if export.registration_mode != "twoPointAB":
        raise ValueError(
            "AB 点法要求光镜导出 registrationMode 为 twoPointAB，"
            f"当前为 {export.registration_mode or '未提供'}"
        )
    if export.coordinate_unit.lower() != "um":
        raise ValueError(
            f"AB 点法要求坐标单位为 um，当前为 {export.coordinate_unit or '未提供'}"
        )
    if export.coordinate_space != "sourceStage":
        raise ValueError(
            "AB 点法要求 coordinateSpace 为 sourceStage，"
            f"当前为 {export.coordinate_space or '未提供'}"
        )
    if export.reference_a_um is None or export.reference_b_um is None:
        raise ValueError("AB 点法要求光镜导出包含 referencePoints.a/b")
    baseline = float(
        np.linalg.norm(
            np.subtract(export.reference_b_um, export.reference_a_um)
        )
    )
    if not math.isfinite(baseline) or baseline <= 1e-9:
        raise ValueError("光镜导出的 A/B 参考点距离过近")


def fit_source_to_ab_transform(
    particles: Sequence[Any],
) -> dict[str, Any] | None:
    """用颗粒自带的 position(sourceStage) ↔ abPosition 对应关系拟合两坐标系
    间的相似变换（Umeyama，允许反射）。

    返回 None 表示导出里没有可用的 position 字段或几何退化。
    真机数据实测该拟合残差为 0（ab 系就是光镜软件从 sourceStage 刚性变换来的）。
    """
    pairs = [
        (
            (particle.source_x_um, particle.source_y_um),
            (particle.ab_x_um, particle.ab_y_um),
        )
        for particle in particles
        if particle.source_x_um is not None and particle.source_y_um is not None
    ]
    if len(pairs) < 2:
        return None
    source = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    ab = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    source_center = source.mean(axis=0)
    ab_center = ab.mean(axis=0)
    source_centered = source - source_center
    ab_centered = ab - ab_center
    source_variance = float((source_centered**2).sum())
    if source_variance <= 1e-9:
        return None
    u, singular, vt = np.linalg.svd(ab_centered.T @ source_centered)
    rotation = u @ vt  # 不修正 det，允许反射
    scale = float(singular.sum() / source_variance)
    if not math.isfinite(scale) or scale <= 1e-12:
        return None
    matrix = scale * rotation
    translation = ab_center - matrix @ source_center
    residuals = np.linalg.norm(source @ matrix.T + translation - ab, axis=1)
    return {
        "matrix": matrix,
        "translation": translation,
        "scale": scale,
        "det": float(np.linalg.det(rotation)),
        "residualP95Um": float(np.percentile(residuals, 95)),
        "residualMaxUm": float(residuals.max()),
        "pairCount": len(pairs),
    }


def resolve_ab_reference_points(export: OpticalExport) -> dict[str, Any]:
    """按 PhotoelectricExport 规范 6.2 节把 A/B 参考点解析到 abPosition 系。

    规范定义（PDF 3.1/6.2）：referencePoints 是光镜平台（sourceStage）坐标；
    abPosition 以 A 为原点、A→B 为 +X 轴、A→B 的左法向为 +Y 轴。因此在
    ab 系中参考点恒为 A=(0,0)、B=(|AB|,0)，|AB| 用 sourceStage 原始值计算
    （刚性变换保长）。直接拿 sourceStage 原始值当 ab 坐标用会让矩阵整体
    错位（历史 bug：颗粒预测坐标整体偏约 48mm、飞出 ROI）。

    颗粒若带 position 字段，则用 position↔abPosition 拟合做交叉校验：
    换算出的参考点应与定义值一致，否则说明导出数据自相矛盾，直接报错。
    """
    if export.reference_a_um is None or export.reference_b_um is None:
        raise ValueError("光镜导出缺少 referencePoints.a/b")
    raw_a = np.asarray(export.reference_a_um, dtype=np.float64)
    raw_b = np.asarray(export.reference_b_um, dtype=np.float64)
    baseline = float(np.linalg.norm(raw_b - raw_a))
    if not math.isfinite(baseline) or baseline <= 1e-9:
        raise ValueError("光镜导出的 A/B 参考点距离过近")
    resolved_a = np.zeros(2)
    resolved_b = np.asarray([baseline, 0.0])

    fit = fit_source_to_ab_transform(export.particles)
    fit_usable = fit is not None and fit["residualP95Um"] <= 5.0
    fit_consistency_um: float | None = None
    if fit_usable:
        converted_a = fit["matrix"] @ raw_a + fit["translation"]
        converted_b = fit["matrix"] @ raw_b + fit["translation"]
        fit_consistency_um = float(
            max(
                np.linalg.norm(converted_a - resolved_a),
                np.linalg.norm(converted_b - resolved_b),
            )
        )
        if fit_consistency_um > max(10.0, 0.001 * baseline):
            raise ValueError(
                "光镜导出自相矛盾：按规范定义的 AB 参考点与 position↔abPosition "
                f"拟合换算结果相差 {fit_consistency_um:.1f} µm，请检查导出文件"
            )
    return {
        "a": (0.0, 0.0),
        "b": (baseline, 0.0),
        "method": "abDefinition",
        "rawA": (float(raw_a[0]), float(raw_a[1])),
        "rawB": (float(raw_b[0]), float(raw_b[1])),
        "baselineUm": baseline,
        "fitConsistencyUm": fit_consistency_um,
        "sourceToAbDet": float(fit["det"]) if fit_usable else None,
        "sourceToAbScale": float(fit["scale"]) if fit_usable else None,
        "sourceToAbResidualP95Um": (
            float(fit["residualP95Um"]) if fit_usable else None
        ),
    }


def infer_ab_handedness(
    source_to_ab_det: float | None,
    experience_matrix: Sequence[Sequence[float]] | None,
) -> str | None:
    """由坐标系手性链推断 ab→SEM台 的 handedness。

    经验标定矩阵（sourceStage→SEM台）的反射性是仪器组合的固有属性，
    不随样品摆放旋转变化；ab→sourceStage 按规范 6.2 定义（左法向建系）
    恒为保向变换（det=+1），颗粒 position 拟合可用时以拟合值为准。
    两者异或即 ab→SEM台 是否需要反射：反射 ↔ handedness="right"
    （对应把规范公式中的左法向 n' 换成右法向）。
    """
    if experience_matrix is None:
        return None
    if source_to_ab_det is None:
        source_to_ab_det = 1.0  # 规范 6.2：ab 系用左法向建立，保向
    experience_det = float(np.linalg.det(np.asarray(experience_matrix)))
    if abs(experience_det) < 1e-12 or abs(source_to_ab_det) < 1e-12:
        return None
    reflected = (experience_det < 0.0) != (source_to_ab_det < 0.0)
    return "right" if reflected else "left"


def build_ab_preview_transform(
    source_a_um: Sequence[float],
    source_b_um: Sequence[float],
    coarse_target_a_um: Sequence[float],
    coarse_target_b_um: Sequence[float],
    *,
    handedness: str = "right",
) -> dict[str, Any]:
    source_a = np.asarray(source_a_um, dtype=np.float64)
    source_b = np.asarray(source_b_um, dtype=np.float64)
    target_a = np.asarray(coarse_target_a_um, dtype=np.float64)
    target_b = np.asarray(coarse_target_b_um, dtype=np.float64)
    if any(point.shape != (2,) for point in (source_a, source_b, target_a, target_b)):
        raise ValueError("AB 参考点必须是二维坐标")
    if not all(
        np.all(np.isfinite(point))
        for point in (source_a, source_b, target_a, target_b)
    ):
        raise ValueError("AB 参考点包含无效数值")

    source_delta = source_b - source_a
    target_delta = target_b - target_a
    source_length = float(np.linalg.norm(source_delta))
    target_length = float(np.linalg.norm(target_delta))
    if source_length <= 1e-9 or target_length <= 1e-9:
        raise ValueError("A/B 两点距离过近，无法构造预览矩阵")

    normalized_handedness = str(handedness).strip().lower()
    if normalized_handedness in {
        "right",
        "reflected",
        "reflected_similarity",
    }:
        normalized_handedness = "right"
        reflected = True
    elif normalized_handedness in {"left", "similarity"}:
        normalized_handedness = "left"
        reflected = False
    else:
        raise ValueError("AB 坐标手性必须为 left 或 right")

    # 通用两点相似变换：不假设 A 在 AB 坐标系原点、B 在 +X 轴上。
    # 在源/目标各建一组以基线为 X 轴的正交基，matrix = scale·B_t·B_sᵀ；
    # 反射手性通过翻转目标基的 Y 轴实现。
    # 当源参考点取规范定义 A=(0,0)/B=(L,0) 时，本式退化为
    # PhotoelectricExport 规范 6.2 的 P' = A' + x·s·u' + y·s·n'：
    # handedness="left" 即规范的左法向 n'=(-u'.y, u'.x)，
    # handedness="right" 即右法向 (u'.y, -u'.x)（Zem 电镜台实际所需）。
    unit_source = source_delta / source_length
    perp_source = np.asarray([-unit_source[1], unit_source[0]], dtype=np.float64)
    unit_x = target_delta / target_length
    unit_y = (
        np.asarray([unit_x[1], -unit_x[0]], dtype=np.float64)
        if reflected
        else np.asarray([-unit_x[1], unit_x[0]], dtype=np.float64)
    )
    scale = target_length / source_length
    basis_source = np.column_stack((unit_source, perp_source))
    basis_target = np.column_stack((unit_x, unit_y))
    matrix = scale * basis_target @ basis_source.T
    translation = target_a - matrix @ source_a
    equation_x = (
        f"stageX = {matrix[0, 0]:.10f}*abX "
        f"{matrix[0, 1]:+.10f}*abY {translation[0]:+.6f}"
    )
    equation_y = (
        f"stageY = {matrix[1, 0]:.10f}*abX "
        f"{matrix[1, 1]:+.10f}*abY {translation[1]:+.6f}"
    )
    return {
        "model": "reflected_similarity" if reflected else "similarity",
        "source": "navigation_coarse",
        "verified": False,
        "inputCoordinate": "abPosition",
        "handedness": normalized_handedness,
        "reflected": reflected,
        "matrix": matrix.tolist(),
        "translation": translation.tolist(),
        "scale": scale,
        "rotationDeg": math.degrees(math.atan2(unit_x[1], unit_x[0])),
        "sourceReferencePointsUm": {
            "a": source_a.tolist(),
            "b": source_b.tolist(),
        },
        "coarseTargetReferencePointsUm": {
            "a": target_a.tolist(),
            "b": target_b.tolist(),
        },
        "sourceBaselineUm": source_length,
        "targetBaselineUm": target_length,
        "equation": {
            "stageX": equation_x,
            "stageY": equation_y,
        },
    }


def annotate_navigation_ab(
    image: np.ndarray,
    detection: NavigationABDetection,
    *,
    a_index: int | None = None,
    b_index: int | None = None,
) -> np.ndarray:
    canvas = _as_bgr_uint8(image).copy()
    fixture_box = tuple(int(round(value)) for value in detection.fixture.box_xyxy)
    cv2.rectangle(
        canvas,
        fixture_box[:2],
        fixture_box[2:],
        (0, 220, 255),
        3,
    )
    crop_x1, crop_y1, crop_x2, crop_y2 = detection.crop_box_xyxy
    cv2.rectangle(
        canvas,
        (crop_x1, crop_y1),
        (crop_x2, crop_y2),
        (255, 180, 0),
        2,
    )
    for index, hole in enumerate(detection.holes):
        center = tuple(int(round(value)) for value in hole.center_xy)
        if index == a_index:
            color, label = (40, 40, 240), "A"
        elif index == b_index:
            color, label = (240, 120, 40), "B"
        else:
            color, label = (60, 210, 80), f"H{index + 1}"
        radius = max(
            10,
            int(
                round(
                    max(
                        hole.box_xyxy[2] - hole.box_xyxy[0],
                        hole.box_xyxy[3] - hole.box_xyxy[1],
                    )
                    / 2.0
                )
            ),
        )
        cv2.circle(canvas, center, radius, color, 3, cv2.LINE_AA)
        cv2.drawMarker(
            canvas,
            center,
            color,
            cv2.MARKER_CROSS,
            max(18, radius),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{label} {hole.confidence:.2f}",
            (center[0] + radius + 5, center[1] - radius - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas
