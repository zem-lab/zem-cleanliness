"""Local interior EDS planning for oversized or visibly clipped large particles.

No optical contour matching, YOLO inference, or invented SEM particle contour.
The intensity check only keeps the sampling point off obvious background/edges;
it does not establish particle identity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .optical_particle_analysis import OpticalParticle, normalize_uint8


@dataclass(frozen=True)
class LargeParticleCenterSettings:
    enabled: bool = True
    minimum_length_to_field_width: float = 1.0
    center_region_radius_fraction: float = .05
    minimum_center_coverage: float = .90
    minimum_intensity_separation: float = 20.0
    minimum_point_clearance_px: float = 3.0
    allow_clipped_near_field: bool = True

    @classmethod
    def from_dict(cls, payload: dict | None):
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError("大颗粒中心采谱设置必须为对象，开关使用 enabled 字段")
        mapping = {"enabled": "enabled", "allowClippedNearField": "allow_clipped_near_field",
                   "minimumLengthToFieldWidth": "minimum_length_to_field_width",
                   "centerRegionRadiusFraction": "center_region_radius_fraction",
                   "minimumCenterCoverage": "minimum_center_coverage",
                   "minimumIntensitySeparation": "minimum_intensity_separation",
                   "minimumPointClearancePx": "minimum_point_clearance_px"}
        result = cls(**{field: payload[key] for key, field in mapping.items() if key in payload})
        if not isinstance(result.enabled, bool) or not isinstance(result.allow_clipped_near_field, bool):
            raise ValueError("大颗粒中心采谱开关必须为布尔值")
        values = [getattr(result, field) for field in mapping.values() if field not in {"enabled", "allow_clipped_near_field"}]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
            raise ValueError("大颗粒中心采谱设置必须为有限数值")
        if (result.minimum_length_to_field_width < 1 or not 0 < result.center_region_radius_fraction <= .2
                or not .8 <= result.minimum_center_coverage <= 1
                or not 0 < result.minimum_intensity_separation <= 255
                or result.minimum_point_clearance_px < 1):
            raise ValueError("大颗粒中心采谱设置超出有效范围")
        return result

    def to_dict(self):
        return {"enabled": self.enabled, "allowClippedNearField": self.allow_clipped_near_field,
                "minimumLengthToFieldWidth": self.minimum_length_to_field_width,
                "centerRegionRadiusFraction": self.center_region_radius_fraction,
                "minimumCenterCoverage": self.minimum_center_coverage,
                "minimumIntensitySeparation": self.minimum_intensity_separation,
                "minimumPointClearancePx": self.minimum_point_clearance_px}


def plan_large_particle_center(image: np.ndarray, particle: OpticalParticle, pixel_size_um: float,
                               *, field_width_um=None, settings: LargeParticleCenterSettings | None = None) -> dict:
    cfg = settings or LargeParticleCenterSettings()
    result = {"mode": "large_particle_center_region", "status": "NOT_APPLICABLE", "eligible": False,
              "identityVerified": False, "contourMatchingSkipped": False, "policy": cfg.to_dict(),
              "algorithmVersion": "local_interior_with_clipped_field_v2"}
    if not cfg.enabled:
        return result
    if (np.asarray(image).ndim not in (2,3) or min(image.shape[:2]) < 16
            or not math.isfinite(pixel_size_um) or pixel_size_um <= 0):
        return {**result, "status": "REJECTED", "reason": "SEM 图像尺寸或像素尺寸无效"}
    height, width = image.shape[:2]
    if (not all(math.isfinite(v) and v > 0 for v in [particle.length_um, particle.width_um, particle.area_um2])
            or particle.width_um > particle.length_um * 1.1):
        return {**result, "status": "REJECTED", "reason": "光镜颗粒尺寸无效"}
    physical_width = width * pixel_size_um
    ratio = particle.length_um / physical_width
    result.update(fieldWidthUm=physical_width, fieldHeightUm=height*pixel_size_um,
                  particleLengthUm=particle.length_um, lengthToFieldWidth=ratio)
    oversize = ratio > cfg.minimum_length_to_field_width
    # A full particle needs room for bounded centering error on both sides.
    # Near-field routing additionally requires actual large foreground reaching
    # an image edge; size alone does not turn ordinary particles into a bypass.
    margin_um = min(height, width)*cfg.center_region_radius_fraction*pixel_size_um
    near_field = (cfg.allow_clipped_near_field and ratio <= cfg.minimum_length_to_field_width
                  and particle.length_um + 2*margin_um > physical_width*cfg.minimum_length_to_field_width)
    if not oversize and not near_field:
        return result
    result.update(eligible=oversize, contourMatchingSkipped=oversize,
                  eligibilityReason="length_exceeds_field" if oversize else "pending_clipped_field_evidence",
                  centeringMarginUm=margin_um)
    try:
        recorded_field = float(field_width_um)
    except (TypeError, ValueError):
        recorded_field = float("nan")
    if (not math.isfinite(recorded_field) or recorded_field <= 0
            or abs(recorded_field-physical_width) > max(1., physical_width*.01)):
        return {**result, "status": "REJECTED", "reason": "SEM 视场元数据与像素尺寸不一致，不能中心采谱"}
    if not np.isfinite(image).all() or float(np.std(image)) < 1e-6:
        return {**result, "status": "REJECTED", "reason": "SEM 图像为空白或无效，不能中心采谱"}
    gray = normalize_uint8(image)
    smooth = cv2.GaussianBlur(gray, (0, 0), 1)
    threshold, binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    foreground = binary > 0
    if not foreground.any() or foreground.all():
        return {**result, "status": "REJECTED", "reason": "图像缺少可区分的颗粒信号与背景"}
    separation = float(np.median(smooth[foreground])-np.median(smooth[~foreground]))
    cx, cy = width//2, height//2
    radius = max(4, int(math.ceil(min(height, width)*cfg.center_region_radius_fraction)))
    x0, y0, x1, y1 = max(0,cx-radius), max(0,cy-radius), min(width,cx+radius+1), min(height,cy+radius+1)
    coverage = float(foreground[y0:y1,x0:x1].mean())
    result.update(imageCenterPx=[cx,cy], centerRegionXyxy=[x0,y0,x1,y1], centerCoverage=coverage,
                  intensityThreshold=float(threshold), intensitySeparation=separation)
    if separation < cfg.minimum_intensity_separation:
        return {**result, "status": "REJECTED", "reason": "中心区域缺少可靠颗粒信号，未执行中心采谱"}
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # A small isolated bright fleck is not sufficient evidence for a large
    # particle. The component must span at least the optical minor dimension
    # (bounded by the field), and provide measurable two-dimensional support.
    required_span = min(particle.width_um/pixel_size_um, min(height,width)/2)
    required_area = min(particle.area_um2/pixel_size_um**2, height*width)*.12
    component_ok = np.zeros(len(stats), bool)
    for index, (left, top, cw, ch, area) in enumerate(stats[1:], 1):
        reaches_edge = left == 0 or top == 0 or left+cw == width or top+ch == height
        large_support = max(cw,ch) >= required_span and area >= required_area
        component_ok[index] = large_support and (oversize or reaches_edge)
    if near_field and not oversize:
        if not component_ok[labels[y0:y1,x0:x1]].any():
            return {**result, "eligibilityReason": "no_large_clipped_center_component"}
        result.update(eligible=True, contourMatchingSkipped=True, eligibilityReason="near_field_and_visibly_clipped")
    # Find a point near the exact field center; do not seek a target elsewhere.
    point_signal = np.where(foreground & (gray > threshold),255,0).astype(np.uint8)
    clearance = cv2.distanceTransform(point_signal, cv2.DIST_L2, 5)
    yy, xx = np.mgrid[y0:y1,x0:x1]
    # Keep the coverage threshold, but apply it to the neighborhood of the
    # actual beam point, not the entire search box. Border and crack pixels
    # retain their raw intensity; the search remains bounded around the center.
    point_radius = max(1, int(math.ceil(cfg.minimum_point_clearance_px)))
    point_coverage = cv2.boxFilter((point_signal>0).astype(np.float32), -1,
                                  (2*point_radius+1, 2*point_radius+1), borderType=cv2.BORDER_CONSTANT)
    allowed = ((clearance[y0:y1,x0:x1] >= cfg.minimum_point_clearance_px)
               & (point_coverage[y0:y1,x0:x1] >= cfg.minimum_center_coverage)
               & component_ok[labels[y0:y1,x0:x1]])
    distances = np.where(allowed, (xx-cx)**2+(yy-cy)**2, np.inf)
    if not np.isfinite(distances).any():
        return {**result, "status": "REJECTED", "reason": "中心区域没有远离背景的采样点"}
    iy, ix = np.unravel_index(np.argmin(distances), distances.shape)
    point = [int(xx[iy,ix]), int(yy[iy,ix])]
    return {**result, "status": "READY", "pointPx": point,
            "pointNeighborhoodRadiusPx": point_radius,
            "pointCoverage": float(point_coverage[point[1],point[0]]),
            "coverageScope": "selected_point_neighborhood",
            "pointOffsetFromCenterPx": float(math.sqrt(distances[iy,ix])),
            "foregroundClearancePx": float(clearance[point[1],point[0]]),
            "reason": "大颗粒超出或接近视场且实拍被截断，在中心附近颗粒内部采谱；未进行轮廓身份匹配"}
