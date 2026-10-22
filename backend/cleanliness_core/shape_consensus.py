"""Conservative additional evidence for two distinct ambiguous template targets.

This corroborates the original winner; it never substitutes a new location or
authorizes EDS. The task must still obtain and validate its final SEM contour.
"""
from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np

from .cleanliness_detection import CandidateGate, CandidateSegmentationConfig, yolo_seg_candidates
from .optical_particle_analysis import OpticalParticle, normalize_uint8
from .particle_refinement import RefinementSettings, optical_mask
from .shape_comparison import compare_shapes


MINIMUM_MODEL_SHAPE_IOU = .85
MINIMUM_DETECTOR_AGREEMENT_IOU = .75
MINIMUM_SHAPE_MARGIN = .07


def _mask(contour, shape):
    result = np.zeros(shape, np.uint8)
    cv2.fillPoly(result, [np.asarray(contour, np.int32)], 1)
    return result > 0


def _iou(a, b):
    return float(np.logical_and(a, b).sum() / max(1, np.logical_or(a, b).sum()))


def _failed(reason, **details):
    return {"status": "UNVERIFIED", "corroborated": False, "allowsEds": False,
            "reason": reason, **details}


def review_ambiguous_shapes(
    image: np.ndarray,
    particle: OpticalParticle,
    pixel_size_um: float,
    template_audit: dict,
    *,
    segmentation: CandidateSegmentationConfig | None,
    gate: CandidateGate | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    minimum_shape_margin: float = MINIMUM_SHAPE_MARGIN,
    minimum_template_score: float = RefinementSettings().match_min_score,
) -> dict:
    """Require complete original search, two measured targets, and agreement.

    YOLO and a separately thresholded grayscale contour must both rank the
    original template winner first by the configured candidate margin. The YOLO foreground must agree
    with the optical shape at >= .85. For BOTH targets the independently
    measured masks must overlap by >= .75. Missing competitors stay ambiguous.
    """
    if should_stop():
        raise InterruptedError("用户停止外形复核")
    if (not math.isfinite(minimum_shape_margin) or not 0 < minimum_shape_margin < 1
            or not math.isfinite(minimum_template_score) or not 0 < minimum_template_score < 1):
        raise ValueError("候选分差或模板分数阈值无效")
    if (template_audit.get("status") != "AMBIGUOUS"
            or template_audit.get("searchTruncated") is not False
            or template_audit.get("distinctHypothesisCount") != 2
            or template_audit.get("runnerUpOverlapIoU") != 0
            or not isinstance(template_audit.get("runnerUp"), dict)):
        return _failed("缺少完整的两个独立竞争位置，保留原拒绝")
    if segmentation is None or segmentation.backend != "yolo":
        return _failed("外形交叉复核需要已配置的本地 YOLO 分割")
    if not math.isfinite(pixel_size_um) or pixel_size_um <= 0:
        return _failed("SEM 像素尺寸无效")
    if np.asarray(image).ndim != 2 or not np.isfinite(image).all() or float(np.std(image)) < 1e-6:
        return _failed("SEM 图像为空白或无效")
    height, width = image.shape
    targets = [template_audit, template_audit["runnerUp"]]
    try:
        centers = [np.asarray(t["centerPixel"], float) for t in targets]
        scores = [float(t["score"]) for t in targets]
        if any(c.shape != (2,) or not np.isfinite(c).all()
               or not (0 <= c[0] < width and 0 <= c[1] < height) for c in centers):
            return _failed("模板位置不是有效的全图坐标")
        if any(not math.isfinite(s) or s < minimum_template_score for s in scores):
            return _failed("原模板相关分数不足，不能用外形替代低分拒绝")
        if np.linalg.norm(centers[0] - centers[1]) <= max(8., particle.width_um/pixel_size_um*.6):
            return _failed("两个位置尚不能视为分离的实测对象")
        mask, optical_pixel, mask_source = optical_mask(particle)
        if mask_source != "customer_mask":
            return _failed("缺少客户原始光镜掩码")
        segmentation.validate()
        detected = yolo_seg_candidates(
            image, expected_length_px=particle.length_um/pixel_size_um,
            expected_width_px=particle.width_um/pixel_size_um,
            expected_area_px=particle.area_um2/pixel_size_um**2,
            model_path=segmentation.yolo_model_path, input_size=segmentation.input_size,
            confidence_threshold=segmentation.confidence_threshold,
            iou_threshold=segmentation.iou_threshold, mask_threshold=segmentation.mask_threshold,
            maximum_detections=segmentation.maximum_detections,
        )
        if len(detected) >= segmentation.maximum_detections:
            return _failed("分割达到候选数量上限，不能完整复核竞争目标")
        gray = normalize_uint8(image)
        candidate_gate = gate or CandidateGate()
        association_radius = max(8., particle.width_um/pixel_size_um*.4)
        radius = max(40, int(math.ceil(particle.length_um/pixel_size_um*1.25)))
        selected_indices, observed_masks, contours, evidence = [], [], [], []
        for center in centers:
            if should_stop():
                raise InterruptedError("用户停止外形复核")
            nearby = [(i, c) for i, c in enumerate(detected)
                      if np.linalg.norm(np.asarray(c.center)-center) <= association_radius
                      and candidate_gate.accepts(c)]
            if len(nearby) != 1:
                return _failed("至少一个竞争位置缺少唯一、可信的 YOLO 轮廓", targets=evidence)
            index, yolo = nearby[0]
            if index in selected_indices:
                return _failed("两个位置关联到同一条分割轮廓，不能当作独立证据")
            selected_indices.append(index)
            points = np.asarray(yolo.contour, np.int32)
            if (points[:, 0].min() <= 1 or points[:, 1].min() <= 1
                    or points[:, 0].max() >= width-2 or points[:, 1].max() >= height-2):
                return _failed("竞争位置的 YOLO 轮廓被视场边缘截断")
            yolo_mask = _mask(points, image.shape)
            if any(np.logical_and(yolo_mask, previous).any() for previous in observed_masks):
                return _failed("两个竞争位置的实测分割重叠，不能排除实例混淆")
            observed_masks.append(yolo_mask)
            x0, y0 = np.maximum(0, np.floor(center-radius).astype(int))
            x1, y1 = np.minimum((width, height), np.ceil(center+radius).astype(int))
            crop = gray[y0:y1, x0:x1]
            smoothed = cv2.GaussianBlur(crop, (0, 0), .7)
            threshold, binary = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            found, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            observed = []
            for item in found:
                if cv2.contourArea(item) < candidate_gate.min_contour_area_px:
                    continue
                local = item.reshape(-1, 2)
                # Reject truncation in the crop as well as the original frame.
                if (local.min() <= 1 or local[:, 0].max() >= crop.shape[1]-2
                        or local[:, 1].max() >= crop.shape[0]-2):
                    continue
                full = local + np.asarray([x0, y0])
                if cv2.pointPolygonTest(full.astype(np.float32), tuple(map(float, yolo.center)), False) >= 0:
                    observed.append(full)
            if len(observed) != 1:
                return _failed("至少一个竞争位置缺少独立、完整的灰度轮廓", targets=evidence)
            intensity = observed[0]
            overlap = _iou(yolo_mask, _mask(intensity, image.shape))
            if overlap < MINIMUM_DETECTOR_AGREEMENT_IOU:
                return _failed("两种分割对竞争颗粒的轮廓不一致", detectorAgreementIoU=overlap, targets=evidence)
            contours.extend([points, intensity])
            evidence.append({"templateCenterPx": center.tolist(), "yoloCenterPx": list(yolo.center),
                             "detectorAgreementIoU": overlap, "grayscaleThreshold": float(threshold),
                             "cropXyxy": [int(x0), int(y0), int(x1), int(y1)]})
        comparison = compare_shapes(mask, optical_pixel, pixel_size_um, contours, image.shape, should_stop=should_stop)
        if comparison["unavailableCandidates"] or len(comparison["candidates"]) != 4:
            return _failed("竞争轮廓的外形证据不足", targets=evidence)
        measured = {item["index"]: item for item in comparison["candidates"]}
        for i, target in enumerate(evidence):
            target.update(yoloShapeIoU=measured[i*2]["shapeIoU"], grayscaleShapeIoU=measured[i*2+1]["shapeIoU"])
        model_margin = evidence[0]["yoloShapeIoU"] - evidence[1]["yoloShapeIoU"]
        intensity_margin = evidence[0]["grayscaleShapeIoU"] - evidence[1]["grayscaleShapeIoU"]
        details = {"method": "two_target_shape_consensus_v1", "targets": evidence,
                   "yoloShapeMargin": model_margin, "grayscaleShapeMargin": intensity_margin,
                   "minimumModelShapeIoU": MINIMUM_MODEL_SHAPE_IOU,
                   "minimumDetectorAgreementIoU": MINIMUM_DETECTOR_AGREEMENT_IOU,
                   "minimumShapeMargin": minimum_shape_margin}
        if evidence[0]["yoloShapeIoU"] < MINIMUM_MODEL_SHAPE_IOU:
            return _failed("原第一候选的实测外形吻合度不足", **details)
        if model_margin < minimum_shape_margin or intensity_margin < minimum_shape_margin:
            return _failed("两种实测轮廓未同时支持原第一候选明显领先", **details)
        return {"status": "CORROBORATED", "corroborated": True, "allowsEds": False,
                "reason": "两种独立实测轮廓均支持原模板位置，仍须最终轮廓及采谱安全点复核", **details}
    except InterruptedError:
        raise
    except Exception as exc:
        return _failed("外形复核异常，保留原拒绝", error=f"{type(exc).__name__}: {exc}")
