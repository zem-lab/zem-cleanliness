"""Local, deterministic AB-navigation/particle calibration primitives.

No device imports or inference services. Coordinates are always optical AB µm.
Template centers use the optical ROI center (not its intensity centroid).
"""
from __future__ import annotations

import base64
import concurrent.futures
import itertools
import math
import os
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import cv2
import numpy as np

from .cleanliness_detection import TransformSolution, extract_optical_shape, optical_salience_score
from .optical_particle_analysis import OpticalParticle


@dataclass(frozen=True)
class RefinementSettings:
    coarse_magnification: int = 60
    fine_magnification: int = 120
    image_size: int = 1024
    pixel_avg: int = 7
    anchor_count: int = 12
    min_inliers: int = 8
    candidate_count: int = 24
    validation_count: int = 6
    pool_size: int = 160
    seed_attempts: int = 8
    max_search_captures: int = 9
    max_captures: int = 200
    max_seconds: float = 5400
    search_radius_um: float = 1800
    search_step_ratio: float = 0.55
    stage_error_um: float = 2
    stage_move_attempts: int = 6
    stage_settle_seconds: float = 1
    approach_offset_um: tuple[float, float] = (-150, -150)
    axis_step_um: float = 150
    axis_min_response: float = 0.15
    axis_max_return_um: float = 5
    match_min_score: float = 0.65
    match_min_margin: float = 0.07
    context_min_neighbors: int = 2
    context_max_neighbors: int = 6
    context_error_um: float = 60
    huber_delta_um: float = 30
    bootstrap_inlier_um: float = 150
    bootstrap_huber_delta_um: float = 60
    inlier_threshold_um: float = 75
    min_inlier_fraction: float = 0.7
    min_coverage: float = 0.25
    max_linear_change: float = 0.15
    mean_error_um: float = 30
    max_error_um: float = 60

    @classmethod
    def from_dict(cls, value: dict | None) -> "RefinementSettings":
        data = dict(value or {})
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"未知颗粒精校准配置：{sorted(unknown)}")
        if "approach_offset_um" in data:
            data["approach_offset_um"] = tuple(data["approach_offset_um"])
        for key, value in list(data.items()):
            if cls.__dataclass_fields__[key].type in (int, "int"):
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or int(value) != value:
                    raise ValueError(f"颗粒精校准配置 {key} 必须是整数")
                data[key] = int(value)
        result = cls(**data)
        for key, val in asdict(result).items():
            if key == "approach_offset_um":
                if len(val) != 2 or not np.isfinite(val).all():
                    raise ValueError("approach_offset_um 必须是两个有限数值")
            elif not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0:
                raise ValueError(f"颗粒精校准配置 {key} 必须是正数")
            elif cls.__dataclass_fields__[key].type in (int, "int") and int(val) != val:
                raise ValueError(f"颗粒精校准配置 {key} 必须是整数")
        if not (result.min_inliers >= 6 and result.anchor_count >= result.min_inliers
                and result.candidate_count >= result.anchor_count and result.validation_count >= 4):
            raise ValueError("至少需要 6 个内点、4 个独立验证点，候选数不得少于锚点数")
        if (result.image_size not in (512, 1024, 2048) or result.seed_attempts < 4
                or result.pool_size < result.candidate_count + result.validation_count
                or result.coarse_magnification > result.fine_magnification
                or result.context_min_neighbors < 2
                or result.context_max_neighbors < result.context_min_neighbors):
            raise ValueError("颗粒精校准采图/选点配置无效")
        for key in ("search_step_ratio", "axis_min_response", "match_min_score",
                    "match_min_margin", "min_inlier_fraction", "min_coverage", "max_linear_change"):
            if not 0 < getattr(result, key) < 1:
                raise ValueError(f"{key} 必须在 0 与 1 之间")
        return result


def optical_mask(particle: OpticalParticle) -> tuple[np.ndarray, float, str]:
    """Use the supplied PNG mask; malformed supplied masks are never silently ignored."""
    if particle.mask_data_base64:
        encoded = particle.mask_data_base64
        if encoded.startswith("data:"):
            encoded = encoded.split(",", 1)[1]
        raw = base64.b64decode(encoded, validate=True)
        mask = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"#{particle.sequence} 光镜掩膜无法解码")
        if (particle.mask_width and mask.shape[1] != particle.mask_width
                or particle.mask_height and mask.shape[0] != particle.mask_height):
            raise ValueError(f"#{particle.sequence} 光镜掩膜尺寸不一致")
        mask = (mask > 0).astype(np.uint8)
        source = "customer_mask"
    else:
        shape = extract_optical_shape(particle)
        if shape is None or particle.image_height <= 0 or particle.image_width <= 0:
            raise ValueError(f"#{particle.sequence} 无可用光镜轮廓")
        mask = np.zeros((particle.image_height, particle.image_width), np.uint8)
        cv2.fillPoly(mask, [np.asarray(shape.contour, np.int32)], 1)
        source = "local_contour"
    area = int(mask.sum())
    if area < 12 or area >= mask.size * 0.95 or not math.isfinite(particle.area_um2) or particle.area_um2 <= 0:
        raise ValueError(f"#{particle.sequence} 光镜掩膜面积无效")
    return mask, math.sqrt(particle.area_um2 / area), source


def xy(particles: Sequence[OpticalParticle]) -> np.ndarray:
    return np.asarray([[p.ab_x_um, p.ab_y_um] for p in particles], dtype=float)


def hull_area(points: np.ndarray) -> float:
    return float(cv2.contourArea(cv2.convexHull(np.asarray(points, np.float32)))) if len(points) >= 3 else 0.0


def coverage(points: np.ndarray, population: np.ndarray) -> float:
    # One common isotropic scale avoids loss of precision for large coordinates.
    origin = np.mean(population, axis=0)
    scale = max(float(np.ptp(population, axis=0).max()), 1)
    return hull_area((points - origin) / scale) / max(hull_area((population - origin) / scale), 1e-12)


def select_plan(particles: Sequence[OpticalParticle], cfg: RefinementSettings) -> dict:
    """Optical-only split, fixed before any SEM acquisition. Prefer large distinctive shapes."""
    population = xy(particles)
    if not np.isfinite(population).all() or hull_area(population) <= 1:
        raise ValueError("光镜颗粒坐标无效或共线，不能做二维精校准")
    # Bound expensive image decoding. No sequence, sample, coordinate or offset hardcodes.
    ranked = sorted(particles, key=lambda p: (-p.area_um2, -p.length_um, p.sequence))[:cfg.pool_size]
    eligible, qualities, rejected = [], {}, []
    for p in ranked:
        try:
            mask, _, _ = optical_mask(p)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(contour)[1]
            aspect = max(rect) / max(min(rect), 1)
            solidity = cv2.contourArea(contour) / max(hull_area(contour.reshape(-1, 2)), 1)
            # Contrast and physical size dominate; concavity is only a small bonus.
            # Otherwise thin irregular fibers can outrank a large clear seed.
            qualities[p.sequence] = optical_salience_score(p) * (1 + .03 * min(aspect, 5) + .1 * (1 - solidity))
            eligible.append(p)
        except (ValueError, cv2.error) as exc:
            rejected.append({"sequence": p.sequence, "reason": str(exc)})
    required = cfg.anchor_count + cfg.validation_count
    if len(eligible) < required:
        raise ValueError(f"可用光镜轮廓不足：需要 {required} 个，实际 {len(eligible)} 个")
    coords = {p.sequence: np.array([p.ab_x_um, p.ab_y_um]) for p in eligible}
    span = max(float(np.ptp(population, axis=0).max()), 1)
    qmax = max(qualities.values())

    def spread(pool, count):
        selected = [max(pool, key=lambda p: (qualities[p.sequence], -p.sequence))]
        while len(selected) < min(count, len(pool)):
            remaining = [p for p in pool if p not in selected]
            def rank(p):
                distance = min(np.linalg.norm(coords[p.sequence] - coords[s.sequence]) for s in selected) / span
                return distance * (.75 + .25 * qualities[p.sequence] / qmax), -p.sequence
            selected.append(max(remaining, key=rank))
        return selected

    # Keep the strongest seed for navigation. Then reserve validation BEFORE any SEM
    # action, excluding initial training anchors; replacements never consume it.
    initial_anchors = spread(eligible, cfg.anchor_count)
    validation = spread([p for p in eligible if p not in initial_anchors], cfg.validation_count)
    anchors = spread([p for p in eligible if p not in validation], cfg.candidate_count)
    for group in (validation, anchors[:cfg.anchor_count]):
        if coverage(xy(group), population) < cfg.min_coverage:
            raise ValueError("显著颗粒空间覆盖不足，请增加候选池或扩大光镜颗粒范围")
    return {"anchors": [p.sequence for p in anchors], "validation": [p.sequence for p in validation],
            "rejectedOptical": rejected, "selection": "optical_only_spatial_salience_v1"}


def robust_affine(source: np.ndarray, target: np.ndarray, *, delta: float = 30,
                  threshold: float = 75, min_inliers: int = 6) -> tuple[TransformSolution, dict]:
    """Deterministic consensus initialization + Huber IRLS, not a 3-point exact-fit gate."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("标定对应点必须是同样大小的 N×2 数组")
    if len(source) < max(4, min_inliers) or not np.isfinite([source, target]).all():
        raise ValueError("有效对应点不足或包含非有限坐标")
    origin, scale = source.mean(axis=0), max(float(np.std(source, axis=0).max()), 1)
    design = np.column_stack(((source - origin) / scale, np.ones(len(source))))
    if np.linalg.cond(design) > 100:
        raise ValueError("锚点接近共线，二维仿射不可辨识")
    # All triples for small sets; fixed RNG and a fixed cap for larger sets.
    combos = list(itertools.islice(itertools.combinations(range(len(source)), 3), 2001))
    if len(combos) > 2000:
        rng = np.random.default_rng(0)
        combos = sorted({tuple(sorted(rng.choice(len(source), 3, replace=False))) for _ in range(2000)})
    best = None
    for indices in combos:
        sub = design[list(indices)]
        if np.linalg.cond(sub) > 100:
            continue
        coef = np.linalg.solve(sub, target[list(indices)])
        residuals = np.linalg.norm(design @ coef - target, axis=1)
        key = (int((residuals <= threshold).sum()), -float(np.minimum(residuals, threshold).sum()))
        if best is None or key > best[0]:
            best = key, coef, residuals
    if best is None or best[0][0] < min_inliers:
        raise ValueError("对应点无法形成足够的几何一致内点")
    coef = best[1]
    # Freeze consensus exclusions for gross mismatches. Huber alone has unbounded
    # leverage: a far-away wrong seed can drag a small otherwise valid consensus.
    support = best[2] <= threshold * 2
    if support.sum() < min_inliers or np.linalg.cond(design[support]) > 100:
        raise ValueError("几何共识的有效覆盖不足")
    for _ in range(30):
        residuals = np.linalg.norm(design @ coef - target, axis=1)
        weights = np.minimum(1, delta / np.maximum(residuals, 1e-9)) * support
        updated = np.linalg.lstsq(design * np.sqrt(weights[:, None]), target * np.sqrt(weights[:, None]), rcond=None)[0]
        if np.linalg.norm(updated - coef) < 1e-8:
            coef = updated
            break
        coef = updated
    residuals = np.linalg.norm(design @ coef - target, axis=1)
    inliers = (residuals <= threshold) & support
    if inliers.sum() < min_inliers or np.linalg.cond(design[inliers]) > 100:
        raise ValueError("稳健拟合内点不足或空间分布退化")
    matrix = coef[:2].T / scale
    translation = coef[2] - matrix @ origin
    solution = TransformSolution("affine", matrix, translation, tuple(residuals),
                                 float(np.sqrt(np.mean(residuals**2))), float(residuals.max()),
                                 reflected=bool(np.linalg.det(matrix) < 0))
    return solution, {"method": "deterministic_consensus_huber_irls", "weights": (np.minimum(1, delta / np.maximum(residuals, 1e-9)) * support).tolist(),
                      "grossMismatchExcluded": (~support).tolist(),
                      "inliers": inliers.tolist(), "inlierCount": int(inliers.sum()), "residualsUm": residuals.tolist()}


def check_linear_change(solution: TransformSolution, initial: TransformSolution, limit: float) -> None:
    relative = solution.matrix @ np.linalg.inv(initial.matrix)
    if not np.isfinite(relative).all() or np.linalg.norm(relative - np.eye(2), ord=2) > limit:
        raise ValueError("拟合的比例/旋转/手性偏离 AB 初值过大，拒绝该匹配组合")


def metric_ab_initial(initial: TransformSolution, source_a, source_b) -> tuple[TransformSolution, dict]:
    """Task-local search hypothesis: physical um-to-um scale, AB midpoint preserved.

    Navigation camera scale is NOT a specimen dimensional measurement. Keep its
    orientation/handedness and midpoint as approximate guesses; translation must
    still be established by actual particles. Never updates device navigation.
    """
    endpoints = np.asarray([source_a, source_b], float)
    if endpoints.shape != (2, 2) or not np.isfinite(endpoints).all() or np.linalg.norm(endpoints[1]-endpoints[0]) < 1000:
        raise ValueError("AB 物理基线无效")
    u, singular, vt = np.linalg.svd(initial.matrix)
    if not np.isfinite(singular).all() or singular.min() < .5 or singular.max() > 2 or singular.max()/singular.min() > 1.05:
        raise ValueError("导航粗矩阵尺度异常，请核对 AB 孔身份、坐标单位和导航参数")
    matrix = u @ vt
    midpoint = endpoints.mean(axis=0)
    translation = initial.matrix @ midpoint + initial.translation - matrix @ midpoint
    result = TransformSolution("affine", matrix, translation, (), 0., 0., reflected=bool(np.linalg.det(matrix) < 0))
    return result, {"method": "metric_ab_midpoint_search_hypothesis", "verified": False,
                    "rawNavigationTransform": initial.to_dict(), "searchTransform": result.to_dict(),
                    "rawNavigationScale": singular.tolist(), "sourceBaselineUm": float(np.linalg.norm(endpoints[1]-endpoints[0])),
                    "notice": "Only a bounded-search initial guess. Device navigation unchanged; real particle verification required."}


def flatten_sem(image: np.ndarray, sigma: float) -> np.ndarray:
    raw = np.asarray(image, np.float32)
    if raw.ndim != 2 or not np.isfinite(raw).all() or float(raw.std()) < 1e-6:
        raise ValueError("SEM 图像为空白或无效")
    raw = cv2.GaussianBlur(raw, (0, 0), 1)
    high = raw - cv2.GaussianBlur(raw, (0, 0), sigma)
    return np.clip(high / np.sqrt(cv2.GaussianBlur(high**2, (0, 0), sigma) + 1), -2, 8).astype(np.float32)


def _natural_template_peaks(scores, cutoff, limit=64):
    """Find maxima before suppression; carving a hole must never create peaks."""
    maxima = (scores >= cv2.dilate(scores, np.ones((3, 3), np.uint8))) & (scores >= cutoff)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(maxima.astype(np.uint8), connectivity=8)
    peaks = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area == 1:
            peaks.append((float(scores[y, x]), (int(x), int(y))))
        else:
            patch = np.where(labels[y:y+h, x:x+w] == label, scores[y:y+h, x:x+w], -np.inf)
            iy, ix = np.unravel_index(np.argmax(patch), patch.shape)
            peaks.append((float(patch[iy, ix]), (int(x+ix), int(y+iy))))
    if not peaks:
        _, maximum, _, location = cv2.minMaxLoc(scores)
        peaks = [(float(maximum), location)]
    peaks.sort(key=lambda x: x[0], reverse=True)
    return peaks[:limit], len(peaks) > limit


def _template_overlap(first, second):
    ax, ay, ar, ab = first['_box']
    bx, by, br, bb = second['_box']
    x, y, right, bottom = max(ax,bx), max(ay,by), min(ar,br), min(ab,bb)
    if x >= right or y >= bottom:
        return 0.0
    intersection = int(np.logical_and(first['_support'][y-ay:bottom-ay, x-ax:right-ax],
                                     second['_support'][y-by:bottom-by, x-bx:right-bx]).sum())
    return intersection / max(1, first['_area'] + second['_area'] - intersection)


def _distinct_template_hypotheses(peaks, center_radius=0.0):
    # Compare with representatives only: transitive overlap chains must not join
    # two distinct particles via a sequence of intermediate translations.
    representatives = []
    spatial_index = {}
    cell_size = 16

    def occupied_cells(peak):
        left, top, right, bottom = peak['_box']
        return ((x, y)
                for y in range(top // cell_size, (bottom - 1) // cell_size + 1)
                for x in range(left // cell_size, (right - 1) // cell_size + 1))

    def same_target(peak, old):
        ax, ay, ar, ab = peak['_box']
        bx, by, br, bb = old['_box']
        intersection_bound = max(0, min(ar, br) - max(ax, bx)) * max(0, min(ab, bb) - max(ay, by))
        nearby_pose = False
        if center_radius > 0:
            dx = peak['centerPixel'][0] - old['centerPixel'][0]
            dy = peak['centerPixel'][1] - old['centerPixel'][1]
            nearby_pose = dx * dx + dy * dy <= center_radius * center_radius
        # If even the overlapping bounding boxes cannot contain enough shared
        # support for IoU 0.5, skip the substantially dearer pixel intersection.
        if not nearby_pose and intersection_bound * 3 < peak['_area'] + old['_area']:
            return False
        overlap = _template_overlap(peak, old)
        # Keep the existing near-center pose equivalence, but require actual
        # shared support so disjoint targets are never merged just for proximity.
        return overlap >= .5 or (overlap > 0 and nearby_pose)
    for peak in sorted(peaks, key=lambda p: p['score'], reverse=True):
        # Actual support overlap requires bounding-box overlap, so only existing
        # representatives sharing a grid cell can be equivalent.  Sorting the
        # indices retains the original representative comparison order exactly.
        possible = sorted({index for cell in occupied_cells(peak)
                           for index in spatial_index.get(cell, ())})
        if not any(same_target(peak, representatives[index]) for index in possible):
            index = len(representatives)
            representatives.append(peak)
            for cell in occupied_cells(peak):
                spatial_index.setdefault(cell, []).append(index)
    return representatives


def match_particle(image: np.ndarray, particle: OpticalParticle, pixel_size: float,
                   cfg: RefinementSettings, *, prior: dict | None = None,
                   should_stop: Callable[[], bool] = lambda: False) -> dict:
    """Physical-size mask matching with footprint-aware, coarse-to-fine search.

    Scores are correlations, NOT probabilities. Both image chiralities are tested;
    stage-coordinate chirality is separately checked using multiple correspondences.
    """
    if not math.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("SEM 像素尺寸无效")
    mask, optical_px, mask_source = optical_mask(particle)
    try:
        flat = flatten_sem(image, max(25, particle.length_um / pixel_size * .5))
    except ValueError as exc:
        return {"status": "NO_MATCH", "reason": str(exc)}
    height, width = flat.shape
    mh, mw = mask.shape
    base_scale = optical_px / pixel_size
    if math.sqrt(mask.sum()) * base_scale < 5:
        return {"status": "NO_MATCH", "reason": "颗粒在当前倍率下像素过少"}
    peaks = []
    visited = set()
    search_truncated = False
    angles = range(-180, 180, 5) if prior is None else [prior["angleDeg"] + d for d in (-15, -10, -5, 0, 5, 10, 15)]
    mirrors = (False, True) if prior is None else (prior["mirrored"],)
    masks = {False: mask.astype(np.float32), True: np.fliplr(mask).astype(np.float32)}

    def evaluate(key):
        mirrored, angle, factor = key
        if should_stop():
            raise InterruptedError("用户停止颗粒匹配")
        scale = base_scale * factor
        n = int(math.ceil(math.hypot(mh, mw) * scale)) + 20
        if n >= min(height, width) * .9:
            return [], False
        warp = cv2.getRotationMatrix2D(((mw - 1) / 2, (mh - 1) / 2), angle, scale)
        warp[:, 2] += np.array([(n - mw) / 2, (n - mh) / 2])
        template = cv2.warpAffine(masks[mirrored], warp, (n, n))
        support = template > .5
        yy, xx = np.nonzero(support)
        if len(xx) == 0:
            return [], False
        left, top, right, bottom = int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)
        support = support[top:bottom, left:right].copy()
        template = cv2.GaussianBlur(template, (0, 0), 1)
        scores = cv2.matchTemplate(flat, template, cv2.TM_CCOEFF_NORMED)
        scores[~np.isfinite(scores)] = -1
        maxima, truncated = _natural_template_peaks(scores, cfg.match_min_score - cfg.match_min_margin)
        candidates = []
        support_area = int(support.sum())
        for score, loc in maxima:
            center = [loc[0] + (n - 1) / 2, loc[1] + (n - 1) / 2]
            candidate = {"score": float(score), "centerPixel": center, "angleDeg": angle,
                         "mirrored": mirrored, "scaleFactor": factor, "maskSource": mask_source,
                         '_support': support, '_area': support_area,
                         '_box': (loc[0]+left, loc[1]+top, loc[0]+right, loc[1]+bottom)}
            candidates.append(candidate)
        return candidates, truncated

    def evaluate_many(poses):
        nonlocal search_truncated
        keys = []
        for mirrored, angle, factor in poses:
            angle = float((angle + 180) % 360 - 180)
            key = (mirrored, angle, factor)
            if key not in visited:
                visited.add(key)
                keys.append(key)
        if not keys:
            return
        # OpenCV releases the GIL for the expensive correlation and image
        # operations.  map() preserves pose order, so the result is identical
        # to the former serial scan while independent poses use separate cores.
        worker_count = min(8, len(keys), max(1, os.cpu_count() or 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            for candidates, truncated in executor.map(evaluate, keys):
                peaks.extend(candidates)
                search_truncated |= truncated

    evaluate_many(itertools.product(mirrors, angles, (.9, 1., 1.1)))
    if not peaks:
        return {"status": "NO_MATCH", "reason": "颗粒模板无法完整放入视场"}
    # Refine plausible competitors as well as the winner, so fine search does
    # not manufacture a lead by only optimizing one hypothesis.
    coarse_best = max(p['score'] for p in peaks)
    fine_angles = {(p['mirrored'], p['angleDeg']+delta) for p in peaks
                   if p['score'] >= coarse_best - cfg.match_min_margin - .05
                   for delta in (-2, -1, 1, 2)}
    if prior is not None:
        fine_angles = {(mirror, angle) for mirror, angle in fine_angles
                       if abs((angle-prior['angleDeg']+180) % 360-180) <= 15}
    evaluate_many((mirrored, angle, factor) for mirrored, angle in sorted(fine_angles)
                  for factor in (.9, 1., 1.1))
    distinct = _distinct_template_hypotheses(peaks, max(8, particle.width_um / pixel_size * .6))
    best = distinct[0]
    runner = distinct[1] if len(distinct) > 1 else None
    second = runner['score'] if runner else -1.
    margin = best["score"] - second
    status = "MATCH" if best["score"] >= cfg.match_min_score and margin >= cfg.match_min_margin and not search_truncated else (
        "AMBIGUOUS" if best["score"] >= cfg.match_min_score else "NO_MATCH")
    public = lambda p: {k:v for k,v in p.items() if not k.startswith('_')}
    reason = ("" if status == 'MATCH' else f"模板相关分数 {best['score']:.3f} 低于 {cfg.match_min_score:.3f}"
              if status == 'NO_MATCH' else "候选过多，未能完整排除竞争目标" if search_truncated
              else f"去除重叠匹配后仍有不同位置的相近候选，差距 {margin:.3f} 小于 {cfg.match_min_margin:.3f}")
    return {**public(best), "status": status, "runnerUpScore": second, "margin": margin,
            "runnerUp": public(runner) if runner else None, "hypothesisCount": len(peaks),
            "runnerUpOverlapIoU": _template_overlap(best, runner) if runner else None,
            "distinctHypothesisCount": len(distinct), "overlapSuppressedCount": len(peaks)-len(distinct),
            "searchTruncated": search_truncated, "coarseAngleStepDeg": 5, "fineAngleStepDeg": 1,
            "reason": reason}


def context_neighbors(seed, neighbors, stage_linear, pixel_axis, center, image_shape, pixel_size, cfg,
                      *, excluded_sequences=(), should_stop=lambda: False):
    """Shared optical visibility rule for seed planning and actual context checks."""
    matrix, axes, center = np.asarray(stage_linear, float), np.asarray(pixel_axis, float), np.asarray(center, float)
    if (matrix.shape != (2, 2) or axes.shape != (2, 2) or not np.isfinite([matrix, axes]).all()
            or np.linalg.cond(matrix) > 5 or np.linalg.cond(axes) > 5
            or not math.isfinite(pixel_size) or pixel_size <= 0):
        raise ValueError("邻域复核坐标轴无效")
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("主颗粒图像中心无效")
    height, width = image_shape[:2]
    local = np.linalg.solve(axes, matrix)
    seed_ab = np.array([seed.ab_x_um, seed.ab_y_um])
    excluded = set(excluded_sequences) | {seed.sequence}
    choices = []
    for p in neighbors:
        if should_stop():
            raise InterruptedError("用户停止邻近颗粒复核")
        if p.sequence in excluded:
            continue
        delta = np.array([p.ab_x_um, p.ab_y_um]) - seed_ab
        expected = center + local @ delta
        radius = (p.length_um * .7 + cfg.context_error_um) / pixel_size
        if (not np.isfinite([*expected, radius, p.area_um2]).all() or p.length_um <= 0
                or p.area_um2 <= 0 or math.sqrt(p.area_um2) / pixel_size < 5
                or not (radius <= expected[0] < width - radius and radius <= expected[1] < height - radius)):
            continue
        if np.linalg.norm(matrix @ delta) <= (seed.length_um + p.length_um) * .6:
            continue
        choices.append((p, expected))
    return sorted(choices, key=lambda item: (-item[0].area_um2, item[0].sequence))


def select_context_seeds(candidates, particles, stage_linear, pixel_axis, pixel_size, cfg,
                         *, excluded_sequences=(), should_stop=lambda: False):
    """Prefer resolvable, spatially spread constellations over isolated large seeds.

    Uses optical metadata/masks and measured coordinate axes, not SEM particle
    identities. Reserved validation IDs are excluded before any mask is decoded.
    Visibility at a centered field is only a planning estimate, never acceptance.
    """
    excluded = set(excluded_sequences)
    audit = {'method': 'optical_centered_neighborhood_and_spread_v1', 'candidates': [],
             'excludedSequences': sorted(excluded), 'sequences': [], 'acceptedTransform': False}
    eligible, quality = [], {}
    for seed in candidates:
        if should_stop():
            raise InterruptedError("用户停止邻域种子选点")
        if seed.sequence in excluded:
            continue
        choices = context_neighbors(seed, particles, stage_linear, pixel_axis,
                                    [cfg.image_size/2]*2, [cfg.image_size]*2, pixel_size, cfg,
                                    excluded_sequences=excluded, should_stop=should_stop)
        valid, rejected = [], []
        for p, _ in choices[:cfg.context_max_neighbors]:
            try:
                optical_mask(p)
                valid.append(p)
            except (ValueError, cv2.error) as exc:
                rejected.append({'sequence': p.sequence, 'reason': str(exc)})
        row = {'sequence': seed.sequence, 'centeredNeighborSequences': [p.sequence for p in valid],
               'centeredNeighborCount': len(valid), 'rejectedOpticalNeighbors': rejected}
        audit['candidates'].append(row)
        if len(valid) < cfg.context_min_neighbors:
            row['status'] = 'INSUFFICIENT_CENTERED_CONTEXT'
            continue
        # Resolution of the weakest required feature limits a constellation.
        areas = sorted((p.area_um2 for p in valid), reverse=True)
        q = math.sqrt(min(seed.area_um2, areas[cfg.context_min_neighbors-1])) / pixel_size
        row.update(status='ELIGIBLE', qualityPixels=q)
        eligible.append(seed)
        quality[seed.sequence] = q
    if len(eligible) < 4:
        audit['reason'] = '居中视场中具备足够可用邻居的种子少于 4 个；不启动无证据基础的搜索'
        return [], audit
    pool = sorted(eligible, key=lambda p: p.sequence)
    selected = [max(pool, key=lambda p: (quality[p.sequence], p.area_um2, -p.sequence))]
    span = max(float(np.ptp(xy(pool), axis=0).max()), 1)
    qmax = max(quality.values())
    while len(selected) < min(cfg.seed_attempts, len(pool)):
        remaining = [p for p in pool if p not in selected]
        def rank(p):
            distance = min(np.linalg.norm(np.subtract([p.ab_x_um, p.ab_y_um], [s.ab_x_um, s.ab_y_um])) for s in selected)/span
            return distance*(.5+.5*quality[p.sequence]/qmax), quality[p.sequence], -p.sequence
        selected.append(max(remaining, key=rank))
    audit['sequences'] = [p.sequence for p in selected]
    return selected, audit


def verify_particle_context(image: np.ndarray, seed: OpticalParticle, seed_match: dict,
                            neighbors: Sequence[OpticalParticle], stage_linear: np.ndarray,
                            pixel_axis: np.ndarray, pixel_size: float, cfg: RefinementSettings,
                            *, excluded_sequences: Sequence[int] = (),
                            should_stop: Callable[[], bool] = lambda: False) -> dict:
    """Check local constellation in the SAME frame before using a seed to move/fit.

    Translation cancels: neighbor_pixel = matched_seed_pixel + inv(pixel_axis) @
    AB_linear @ (neighbor_AB - seed_AB). No historical translation, sample ID or
    final matrix is consumed. Reserved validation particles are excluded BEFORE
    looking at their masks or pixels. No searching outside this captured frame.
    """
    result = {"status": "UNVERIFIED", "method": "same_frame_neighbor_geometry_v1",
              "requiredCount": cfg.context_min_neighbors, "confirmedCount": 0,
              "neighbors": [], "excludedSequences": sorted(set(excluded_sequences))}
    if seed_match.get("status") != "MATCH":
        result["reason"] = "主颗粒尚未取得唯一匹配"
        return result
    matrix, axes = np.asarray(stage_linear, float), np.asarray(pixel_axis, float)
    if (matrix.shape != (2, 2) or axes.shape != (2, 2) or not np.isfinite([matrix, axes]).all()
            or np.linalg.cond(matrix) > 5 or np.linalg.cond(axes) > 5):
        raise ValueError("邻域复核坐标轴无效")
    center = np.asarray(seed_match["centerPixel"], float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("主颗粒图像中心无效")
    choices = context_neighbors(seed, neighbors, matrix, axes, center, image.shape, pixel_size, cfg,
                                excluded_sequences=excluded_sequences, should_stop=should_stop)
    result["eligibleCount"] = len(choices)
    confirmed_pixels = [center]
    confirmed_particles = [seed]
    for p, expected in choices[:cfg.context_max_neighbors]:
        if should_stop():
            raise InterruptedError("用户停止邻近颗粒复核")
        record = {"sequence": p.sequence, "opticalAbUm": [p.ab_x_um, p.ab_y_um],
                  "expectedCenterPixel": expected.tolist()}
        try:
            # Keep a full-frame uniqueness check. A small predicted crop must not
            # hide a competing particle and manufacture a unique correspondence.
            match = match_particle(image, p, pixel_size, cfg, should_stop=should_stop)
            record["match"] = match
            if match.get("centerPixel") is not None:
                candidate = np.asarray(match["centerPixel"], float)
                error = float(np.linalg.norm(axes @ (candidate - expected)))
                distinct = all(np.linalg.norm(axes @ (candidate - other)) >
                               (p.length_um + other_p.length_um) * .4
                               for other, other_p in zip(confirmed_pixels, confirmed_particles))
                record.update(relativeErrorUm=error, distinctObject=distinct)
                if match["status"] == "MATCH" and error <= cfg.context_error_um and distinct:
                    record["confirmed"] = True
                    confirmed_pixels.append(candidate)
                    confirmed_particles.append(p)
        except (ValueError, cv2.error) as exc:
            record["error"] = str(exc)
        result["neighbors"].append(record)
        if len(confirmed_particles) - 1 >= cfg.context_min_neighbors:
            break
    result["confirmedCount"] = len(confirmed_particles) - 1
    result["status"] = "VERIFIED" if result["confirmedCount"] >= cfg.context_min_neighbors else "UNVERIFIED"
    result["reason"] = ("" if result["status"] == "VERIFIED" else
                        "同图邻近颗粒的形状/相对位置证据不足，单颗粒高分不用于更新导航")
    return result


class AxisMeasurementError(ValueError):
    """Quality failures can retry another field; image/stage contradictions cannot."""
    def __init__(self, message: str, diagnostics: dict, *, retryable: bool = True):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.retryable = retryable


def measure_pixel_axes(images: dict[str, np.ndarray], positions: dict[str, Sequence[float]],
                       pixel_size: float, cfg: RefinementSettings) -> dict:
    """Six physical captures: origin, ±X, ±Y, repeated origin. Uses measured stage deltas."""
    names = ("o1", "xp", "xm", "yp", "ym", "o2")
    if (not math.isfinite(pixel_size) or pixel_size <= 0 or any(n not in images or n not in positions for n in names)
            or any(np.asarray(positions[n]).shape != (2,) or not np.isfinite(positions[n]).all() for n in names)):
        raise ValueError("像素轴图像/实测台坐标/像素尺寸缺失或无效")
    prepared = {}
    for name in names:
        raw = np.asarray(images[name], np.float32)
        if raw.ndim != 2 or raw.shape != images["o1"].shape or not np.isfinite(raw).all() or raw.std() < 1e-6:
            raise ValueError("像素轴测量图像为空白或尺寸不一致")
        prepared[name] = cv2.GaussianBlur(raw, (0, 0), 1.5) - cv2.GaussianBlur(raw, (0, 0), 30)
    window = cv2.createHanningWindow((prepared["o1"].shape[1], prepared["o1"].shape[0]), cv2.CV_32F)
    shifts = {}
    unreliable = []
    for name in names[1:]:
        shift, response = cv2.phaseCorrelate(prepared["o1"].copy(), prepared[name].copy(), window)
        if not np.isfinite([*shift, response]).all() or response < cfg.axis_min_response:
            unreliable.append(name)
        shifts[name] = {"shiftPx": [float(v) if math.isfinite(v) else None for v in shift],
                        "response": float(response) if math.isfinite(response) else None}
    diagnostics = {"shifts": shifts, "unreliableFrames": unreliable,
                   "positionsUm": {n: list(positions[n]) for n in names}, "pixelSizeUm": pixel_size}
    # Inspect the repeated origin even if an earlier frame has a poor phase peak.
    # Previously 'ym response low' hid a 352-pixel displacement at identical stage
    # coordinates. Do not retry another region to mask a capture/stage contradiction.
    if "o2" not in unreliable:
        repeat_px = float(np.linalg.norm(shifts["o2"]["shiftPx"]))
        stage_return = float(np.linalg.norm(np.subtract(positions["o2"], positions["o1"])))
        diagnostics.update(repeatImageShiftPx=repeat_px, stageReturnDeltaUm=stage_return)
        # Conservative 0.7 is the same lower scale bound checked below, not a
        # newly fitted axis or a claimed physical measurement of this failed set.
        if unreliable and .7 * pixel_size * repeat_px - stage_return > cfg.axis_max_return_um:
            raise AxisMeasurementError(
                f"回原点图像与台坐标不一致：图像位移 {repeat_px:.2f} px，台坐标差 {stage_return:.2f} µm；停止标定，不自动换区域重试",
                diagnostics, retryable=False)
    if unreliable:
        raise AxisMeasurementError("像素轴相位相关不可靠：" + ", ".join(
            f"{n} response={shifts[n]['response']}" for n in unreliable), diagnostics)
    dp = np.array([np.subtract(shifts[a]["shiftPx"], shifts[b]["shiftPx"]) for a, b in (("xp", "xm"), ("yp", "ym"))]).T
    ds = np.array([np.subtract(positions[a], positions[b]) for a, b in (("xp", "xm"), ("yp", "ym"))]).T
    if np.linalg.cond(dp) > 5 or np.linalg.cond(ds) > 5:
        raise AxisMeasurementError("像素轴位移退化或轴耦合过大", diagnostics)
    matrix = -ds @ np.linalg.inv(dp)
    singular = np.linalg.svd(matrix / pixel_size)[1]
    errors = [float(np.linalg.norm(matrix @ shifts[n]["shiftPx"] + np.subtract(positions[n], positions["o1"]))) for n in names[1:]]
    return_error = float(np.linalg.norm(matrix @ shifts["o2"]["shiftPx"]))
    diagnostics.update(pixelToStageMatrix=matrix.tolist(), closureErrorsUm=errors,
                       returnErrorUm=return_error, relativeSingularValues=singular.tolist())
    measured_pixel = float(np.sqrt(abs(np.linalg.det(matrix))))
    diagnostics["measuredPixelSizeUm"] = measured_pixel
    if min(singular) < .7 or max(singular) > 1.3:
        raise AxisMeasurementError(
            f"图像尺度与实测台位移不一致：元数据 {pixel_size:.4f} µm/px，实测约 {measured_pixel:.4f} µm/px；停止标定，需核查倍率/采图链路",
            diagnostics, retryable=False)
    if not np.isfinite(matrix).all() or max(errors) > cfg.axis_max_return_um or return_error > cfg.axis_max_return_um:
        raise AxisMeasurementError(f"像素轴尺度/回原点/闭环残差超限：return={return_error:.2f} µm, closure={max(errors):.2f} µm", diagnostics)
    return {"pixelToStageMatrix": matrix.tolist(), "pixelSizeUm": pixel_size,
            "shifts": shifts, "closureErrorsUm": errors, "returnErrorUm": return_error}


def validate_frozen(solution: TransformSolution, anchors: list[dict], validation: list[dict],
                    validation_sequences: Sequence[int], population: np.ndarray,
                    cfg: RefinementSettings) -> dict:
    """Missing, ambiguous or overlapping validation records fail closed."""
    anchor_ids = {a["sequence"] for a in anchors}
    observed = [v["sequence"] for v in validation]
    reasons = []
    if len(validation_sequences) < cfg.validation_count or len(set(validation_sequences)) != len(validation_sequences):
        reasons.append("独立验证计划数量不足或重复")
    if set(validation_sequences) & anchor_ids or set(observed) != set(validation_sequences) or len(set(observed)) != len(observed):
        reasons.append("训练/验证重叠或验证记录不完整")
    errors, points = [], []
    for record in validation:
        if record.get("status") != "MATCH" or "measuredStageUm" not in record or record.get("error"):
            reasons.append(f"#{record['sequence']} 未取得唯一可靠匹配")
            continue
        source = np.asarray(record["opticalAbUm"], float)
        target = np.asarray(record["measuredStageUm"], float)
        predicted = np.asarray(solution.transform_xy(*source))
        if not np.isfinite([source, target, predicted]).all():
            reasons.append("验证记录包含非有限坐标")
            continue
        error = float(np.linalg.norm(predicted - target))
        record["predictionErrorUm"] = error
        errors.append(error)
        points.append(source)
    cov = coverage(np.asarray(points), population) if len(points) >= 3 else 0
    if cov < cfg.min_coverage:
        reasons.append("独立验证空间覆盖不足")
    mean = float(np.mean(errors)) if errors else None
    maximum = max(errors) if errors else None
    if mean is None or mean > cfg.mean_error_um or maximum > cfg.max_error_um:
        reasons.append("独立验证平均或最大误差超限")
    return {"withinLimits": not reasons, "reasons": reasons, "count": len(errors),
            "requiredCount": len(validation_sequences), "meanErrorUm": mean, "maxErrorUm": maximum,
            "coverage": cov, "limits": {"meanErrorUm": cfg.mean_error_um, "maxErrorUm": cfg.max_error_um}}
