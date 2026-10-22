"""Offline shape comparison experiment. Never authorizes identification or EDS.

Compare measured SEM foreground with the customer mask using physical-size
binary IoU. Brightness, stage distance, particle IDs and human labels are not
inputs. Raw ranking never authorizes acquisition; the separate shape_consensus
review also requires complete template evidence and independent measurements.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class ShapeComparisonSettings:
    scale_factors: tuple[float, ...] = (0.9, 1.0, 1.1)
    coarse_angle_step: int = 5
    fine_angle_step: int = 1
    minimum_area_px: float = 30.0
    minimum_side_px: float = 5.0
    maximum_canvas_side: int = 1400


def _features(contour: np.ndarray) -> dict:
    area = float(cv2.contourArea(contour))
    sides = cv2.minAreaRect(contour)[1]
    perimeter = float(cv2.arcLength(contour, True))
    return {
        "areaPx": area,
        "aspect": float(max(sides) / max(min(sides), 1e-9)),
        "solidity": area / max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0),
        "circularity": 4 * math.pi * area / max(perimeter * perimeter, 1.0),
    }


def compare_shapes(
    optical_mask: np.ndarray,
    optical_pixel_size_um: float,
    sem_pixel_size_um: float,
    contours: Sequence[Sequence[Sequence[float]]],
    image_shape: tuple[int, int],
    *,
    settings: ShapeComparisonSettings | None = None,
    should_stop: Callable[[], bool] = lambda: False,
) -> dict:
    """Rank complete observed contours without changing production acceptance.

    Every contour gets the same translation/rotation/mirror/scale search. The
    optical mask is a reference only, never a substitute for a measured contour.
    Incomplete or tiny measurements are retained as explicit unevaluated items.
    There is deliberately no acceptance threshold or EDS point in this result.
    """
    cfg = settings or ShapeComparisonSettings()
    mask = np.asarray(optical_mask)
    if mask.ndim != 2 or not np.isfinite(mask).all():
        raise ValueError("Invalid optical mask")
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() < 12 or mask.mean() >= .95:
        raise ValueError("Optical mask has insufficient foreground or background")
    if (not math.isfinite(optical_pixel_size_um) or optical_pixel_size_um <= 0
            or not math.isfinite(sem_pixel_size_um) or sem_pixel_size_um <= 0):
        raise ValueError("Pixel sizes must be finite and positive")
    if len(image_shape) != 2 or any(int(v) != v or v < 8 for v in image_shape):
        raise ValueError("Invalid SEM image shape")
    if (cfg.coarse_angle_step != 5 or cfg.fine_angle_step != 1
            or cfg.scale_factors != (.9, 1., 1.1)):
        raise ValueError("The experiment uses a frozen search policy")
    base_scale = optical_pixel_size_um / sem_pixel_size_um
    mh, mw = mask.shape
    maximum_template = math.ceil(math.hypot(mh, mw) * base_scale * 1.1) + 20
    if maximum_template * 2 > cfg.maximum_canvas_side:
        raise ValueError("Physical scale exceeds the bounded comparison canvas")
    reference_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    optical_contour = max(reference_contours, key=cv2.contourArea)
    features = _features(optical_contour)
    template_cache = {}

    def template(mirror, angle, factor, phase=(0.0, 0.0)):
        key = (mirror, int((angle + 180) % 360 - 180), factor, *phase)
        if key not in template_cache:
            scale = base_scale * factor
            side = int(math.ceil(math.hypot(mh, mw) * scale)) + 20
            warp = cv2.getRotationMatrix2D(((mw-1)/2, (mh-1)/2), key[1], scale)
            warp[:, 2] += np.array([(side-mw)/2, (side-mh)/2])
            warp[:, 2] += phase
            source = np.fliplr(mask) if mirror else mask
            binary = cv2.warpAffine(source.astype(np.float32), warp, (side, side)) > .5
            template_cache[key] = (binary.astype(np.uint8), int(binary.sum()))
        return key, template_cache[key]

    scored, unavailable = [], []
    height, width = map(int, image_shape)
    for index, points in enumerate(contours):
        if should_stop():
            raise InterruptedError("Stopped offline shape comparison")
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (2,) or not np.isfinite(points).all():
            raise ValueError("Malformed contour coordinates")
        if len(points) < 3:
            unavailable.append({"index": index, "reason": "insufficient_contour_points"})
            continue
        if (np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width)
                or np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height)):
            raise ValueError("Contour coordinates are outside the original SEM frame")
        contour = np.rint(points).astype(np.int32)
        low, high = contour.min(axis=0), contour.max(axis=0)
        if low.min() <= 1 or high[0] >= width-2 or high[1] >= height-2:
            unavailable.append({"index": index, "reason": "frame_edge_partial_contour"})
            continue
        measured_features = _features(contour)
        if (measured_features["areaPx"] < cfg.minimum_area_px
                or min(cv2.minAreaRect(contour)[1]) < cfg.minimum_side_px):
            unavailable.append({"index": index, "reason": "insufficient_contour_support"})
            continue
        canvas_side = 2 * max(maximum_template, int(math.ceil(np.linalg.norm(high-low))) + 20)
        if canvas_side > cfg.maximum_canvas_side:
            unavailable.append({"index": index, "reason": "comparison_canvas_limit"})
            continue
        offset = ((canvas_side - (high-low)) // 2).astype(int)
        measured = np.zeros((canvas_side, canvas_side), np.uint8)
        cv2.fillPoly(measured, [contour-low+offset], 1)
        measured_area = int(measured.sum())
        best, evaluated = None, {}

        def evaluate(mirror, angle, factor, phase=(0.0, 0.0)):
            nonlocal best
            if should_stop():
                raise InterruptedError("Stopped offline shape comparison")
            key, (binary, area) = template(mirror, angle, factor, phase)
            if key in evaluated:
                return
            # The exact IoU cannot exceed this upper bound: do not truncate
            # competitors or search poses based on a heuristic score.
            upper = min(area, measured_area) / max(area, measured_area, 1)
            if best is not None and upper <= best["iou"]:
                # Retain the upper bound for the subsequent fine-search set;
                # skipping this convolution must not hide a plausible pose.
                evaluated[key] = float(upper)
                return
            intersections = np.clip(np.rint(cv2.matchTemplate(measured, binary, cv2.TM_CCORR)), 0, min(area, measured_area))
            _, overlap, _, location = cv2.minMaxLoc(intersections)
            iou = overlap / max(1, area + measured_area - overlap)
            evaluated[key] = float(iou)
            if best is None or iou > best["iou"]:
                best = {"iou": float(iou), "mirrored": bool(mirror), "angleDeg": key[1],
                        "scaleFactor": factor, "location": location, "templateKey": key,
                        "subpixelPhasePx": list(phase)}

        for mirror in (False, True):
            for angle in range(-180, 180, cfg.coarse_angle_step):
                for factor in cfg.scale_factors:
                    evaluate(mirror, angle, factor)
        plausible = [key for key, score in evaluated.items() if score >= best["iou"] - .07]
        fine_poses = sorted({(mirror, angle + delta, factor) for mirror, angle, factor, *_ in plausible
                             for delta in (-2, -1, 1, 2)})
        for mirror, angle, factor in fine_poses:
            evaluate(mirror, angle, factor)
        # A half-pixel warp can erode a thresholded binary mask even for an
        # otherwise identical object. Search both grid phases symmetrically;
        # do not weaken the identity/invariance tests to hide rasterization.
        plausible = [key for key, score in evaluated.items() if score >= best["iou"] - .07]
        for mirror, angle, factor, *_ in plausible:
            for phase in ((.5, 0.), (0., .5), (.5, .5)):
                evaluate(mirror, angle, factor, phase)
        moments = cv2.moments(contour)
        center = [moments["m10"]/moments["m00"], moments["m01"]/moments["m00"]]
        binary = template_cache[best["templateKey"]][0].astype(np.uint8)
        reference, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shift = np.asarray(best["location"]) + low - offset
        best_reference = [(c.reshape(-1, 2) + shift).tolist() for c in reference]
        scored.append({"index": index, "centerPx": center, "shapeIoU": best["iou"],
                       "angleDeg": best["angleDeg"], "mirrored": best["mirrored"],
                       "scaleFactor": best["scaleFactor"], "features": measured_features,
                       "subpixelPhasePx": best["subpixelPhasePx"],
                       "huDifference": float(cv2.matchShapes(optical_contour, contour, cv2.CONTOURS_MATCH_I1, 0)),
                       "referenceContours": best_reference,
                       "observedContour": contour.tolist(), "evaluatedPoseCount": len(evaluated)})
    scored.sort(key=lambda item: (-item["shapeIoU"], item["index"]))
    runner = scored[1] if len(scored) > 1 else None
    return {"method": "offline_physical_binary_iou_v1", "allowsEds": False,
            "status": "RANKED" if scored else "INSUFFICIENT_CONTOUR" if unavailable else "NO_TARGET",
            "opticalFeatures": features, "candidates": scored, "unavailableCandidates": unavailable,
            "scoreMargin": scored[0]["shapeIoU"]-runner["shapeIoU"] if runner else None,
            "distinctTargetCheckPerformed": False,
            "notice": "Research ranking only; model detections do not establish complete target coverage or permission to acquire EDS."}
