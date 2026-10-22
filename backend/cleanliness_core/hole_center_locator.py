"""SEM 孔（hole）圆心精定位：单图识别 + 电镜闭环搜索。

算法来源于 tmp/ab_sem_probe 下的探针实验（已用 23 张实拍图验证）：

1. ``flatten``            —— 高通滤波去暗角/低频背景，把细边缘拉出来；
2. ``vote_with_prior``    —— 边缘像素沿梯度方向、按先验半径投票选出粗圆心；
3. ``polar_edge_points``  —— 从粗圆心发射线，抛物线插值取亚像素边缘点；
4. ``ransac_circle``      —— 自由半径 RANSAC + Taubin 拟合（整圆场景）；
5. ``robust_center_fixed_radius`` —— 固定半径（孔径先验）只解圆心（短弧场景，
   实验数据：45° 弧圆心中位误差 3.6 µm / P90 9.3 µm）。

单图识别结果分三态：

- ``full``：拟合质量通过三道自检（直径 vs 先验、角覆盖、RMS），圆心可信；
- ``arc`` ：只看到一段弧，固定半径拟合给出圆心估计（精度足够引导移台）；
- ``none``：视场内没有可信的孔特征。

闭环流程 ``locate_hole_center``：

    粗坐标 → 移台 → 拍图 → 识别
      ├─ full → 若明显偏心则再居中一次 → 输出精确圆心台坐标
      ├─ arc  → 移台到弧估计的圆心 → 重拍确认
      └─ none → 在粗坐标周围方形螺旋扩环搜索（步长 = 视场 × 比例）

安全约束：所有移台目标被限制在粗坐标 ``max_search_radius_um`` 半径内；
移台回读超差时记录警告并按实际回读坐标继续。首次带圆心的两张图之间会自动校验像素轴与
电机轴的方向符号，方向接反时自动翻转（记录在结果里）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

ProgressCallback = Callable[[str], None]
StopCallback = Callable[[], bool]


# =========================================================================
# 基础图像处理与圆拟合
# =========================================================================
def flatten(image: np.ndarray, sigma: float = 21.0) -> np.ndarray:
    """去暗角/低频背景：原图减去高斯模糊后按 99.5 分位归一化到 [-1, 1]。"""
    source = np.asarray(image, dtype=np.float32)
    blurred = cv2.GaussianBlur(source, (0, 0), sigma)
    high = source - blurred
    scale = float(np.percentile(np.abs(high), 99.5)) or 1.0
    return np.clip(high / scale, -1.0, 1.0)


def taubin_fit(points: np.ndarray) -> tuple[float, float, float]:
    """Taubin 代数圆拟合，返回 (cx, cy, r)。"""
    x, y = points[:, 0], points[:, 1]
    mx, my = x.mean(), y.mean()
    u, v = x - mx, y - my
    z = u * u + v * v
    zmean = z.mean()
    z0 = (z - zmean) / (2.0 * math.sqrt(zmean)) if zmean > 0 else z
    matrix = np.column_stack([z0, u, v])
    _, _, vt = np.linalg.svd(matrix, full_matrices=False)
    a = vt[-1]
    a0 = a[0] / (2.0 * math.sqrt(zmean)) if zmean > 0 else a[0]
    solution = np.array([a0, a[1], a[2], -zmean * a0])
    if abs(solution[0]) < 1e-12:
        raise ValueError("圆拟合退化（三点近共线）")
    cx = -solution[1] / (2 * solution[0])
    cy = -solution[2] / (2 * solution[0])
    r2 = (solution[1] ** 2 + solution[2] ** 2) / (4 * solution[0] ** 2) - solution[3] / solution[0]
    return cx + mx, cy + my, math.sqrt(max(r2, 1e-9))


def ransac_circle(
    points: np.ndarray,
    tolerance_px: float,
    iterations: int = 300,
    rng: np.random.Generator | None = None,
) -> tuple[tuple[float, float, float], np.ndarray] | None:
    """自由半径 RANSAC 圆拟合，返回 ((cx, cy, r), 内点掩码)；失败返回 None。"""
    if len(points) < 8:
        return None
    rng = rng or np.random.default_rng(12345)
    best_inliers: np.ndarray | None = None
    best_model: tuple[float, float, float] | None = None
    for _ in range(iterations):
        idx = rng.choice(len(points), size=3, replace=False)
        try:
            model = taubin_fit(points[idx])
        except (ValueError, np.linalg.LinAlgError):
            continue
        cx, cy, r = model
        if not all(map(math.isfinite, model)) or r <= 1:
            continue
        residual = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - r)
        inliers = residual < tolerance_px
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers, best_model = inliers, model
    if best_inliers is None or best_inliers.sum() < 8:
        return None
    for _ in range(3):
        try:
            best_model = taubin_fit(points[best_inliers])
        except (ValueError, np.linalg.LinAlgError):
            break
        cx, cy, r = best_model
        residual = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - r)
        best_inliers = residual < tolerance_px
        if best_inliers.sum() < 8:
            return None
    assert best_model is not None
    return best_model, best_inliers


def fit_center_fixed_radius(
    points: np.ndarray,
    radius: float,
    initial: tuple[float, float],
) -> tuple[float, float] | None:
    """半径固定时求圆心：高斯牛顿最小化 sum((|p-c| - r)^2)，只有 2 个自由度。"""
    center = np.asarray(initial, dtype=np.float64)
    for _ in range(60):
        delta = points - center
        distance = np.hypot(delta[:, 0], delta[:, 1])
        if np.any(distance < 1e-9):
            return None
        residual = distance - radius
        jacobian = -delta / distance[:, None]
        try:
            step = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        center = center + step
        if np.linalg.norm(step) < 1e-7:
            break
    return float(center[0]), float(center[1])


def robust_center_fixed_radius(
    points: np.ndarray,
    radius: float,
    initial: tuple[float, float],
    tolerance_px: float = 1.5,
) -> tuple[tuple[float, float], np.ndarray] | None:
    """固定半径拟合 + 迭代剔野点，返回 (圆心, 内点掩码)。"""
    keep = np.ones(len(points), dtype=bool)
    center = initial
    for _ in range(5):
        fitted = fit_center_fixed_radius(points[keep], radius, center)
        if fitted is None:
            return None
        center = fitted
        residual = np.abs(
            np.hypot(points[:, 0] - center[0], points[:, 1] - center[1]) - radius
        )
        new_keep = residual < tolerance_px
        if new_keep.sum() < 8:
            return None
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return center, keep


def vote_with_prior(
    flat: np.ndarray,
    r_expect: float,
    tolerance: float,
) -> tuple[float, float, float]:
    """边缘像素沿梯度方向、限制在先验半径窗内投票，返回 (cx, cy, 峰值对比度)。

    圆心可以落在图像外（弧场景），累加器四周留了 r_hi 的边距。
    """
    r_lo, r_hi = r_expect * (1 - tolerance), r_expect * (1 + tolerance)
    gx = cv2.Scharr(flat, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(flat, cv2.CV_32F, 0, 1)
    magnitude = np.hypot(gx, gy)
    threshold = np.quantile(magnitude, 0.94)
    ys, xs = np.nonzero(magnitude >= threshold)
    if len(xs) < 50:
        return float("nan"), float("nan"), 0.0
    weights = magnitude[ys, xs]
    ux = gx[ys, xs] / (weights + 1e-9)
    uy = gy[ys, xs] / (weights + 1e-9)

    height, width = flat.shape
    pad = int(r_hi) + 2
    accumulator = np.zeros((height + 2 * pad, width + 2 * pad), np.float32)
    step = max(0.5, (r_hi - r_lo) / 12.0)
    for radius in np.arange(r_lo, r_hi + 1e-6, step):
        for sign in (1.0, -1.0):
            tx = np.rint(xs + sign * radius * ux).astype(np.int32) + pad
            ty = np.rint(ys + sign * radius * uy).astype(np.int32) + pad
            valid = (
                (tx >= 0)
                & (ty >= 0)
                & (tx < accumulator.shape[1])
                & (ty < accumulator.shape[0])
            )
            np.add.at(accumulator, (ty[valid], tx[valid]), weights[valid])
    accumulator = cv2.GaussianBlur(accumulator, (0, 0), 2.5)
    peak_y, peak_x = np.unravel_index(int(np.argmax(accumulator)), accumulator.shape)
    contrast = float(accumulator[peak_y, peak_x] / (accumulator.mean() + 1e-9))
    return float(peak_x - pad), float(peak_y - pad), contrast


def polar_edge_points(
    flat: np.ndarray,
    center: tuple[float, float],
    r_lo: float,
    r_hi: float,
    n_rays: int = 1080,
    polarity: float = 0.0,
    smooth_sigma: float = 1.5,
) -> np.ndarray:
    """从粗圆心做极坐标展开，每条射线取梯度峰值的亚像素半径。

    返回 (N, 3)：[x, y, 角度]。圆心在图像外时自动跳过出界射线。

    ``polarity``：0 取无符号最强梯度；-1 只取"亮→暗"（孔内亮时的内缘）；
    +1 只取"暗→亮"（环带外缘）。真机上孔边缘是有宽度的环带，按极性分离
    才能把同一条缘的点聚到一个圆上（混取会一半内缘一半外缘，拟合崩溃）。
    """
    cx, cy = center
    if smooth_sigma > 0:
        flat = cv2.GaussianBlur(flat, (0, 0), smooth_sigma)
    radii = np.arange(max(r_lo - 25, 3.0), r_hi + 25, 0.5, dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False, dtype=np.float32)
    sample_x = cx + np.outer(np.cos(angles), radii)
    sample_y = cy + np.sin(angles)[:, None] * radii
    profile = cv2.remap(
        flat,
        sample_x.astype(np.float32),
        sample_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    inside = (
        (sample_x >= 1)
        & (sample_x < flat.shape[1] - 1)
        & (sample_y >= 1)
        & (sample_y < flat.shape[0] - 1)
    )
    derivative = np.gradient(profile, axis=1)
    derivative[~inside] = 0.0
    strength = (
        np.abs(derivative)
        if polarity == 0.0
        else np.clip(derivative * polarity, 0.0, None)
    )

    points: list[tuple[float, float, float]] = []
    for ray in range(n_rays):
        column = int(np.argmax(strength[ray]))
        if column <= 0 or column >= len(radii) - 1 or strength[ray, column] <= 0:
            continue
        left = strength[ray, column - 1]
        mid = strength[ray, column]
        right = strength[ray, column + 1]
        denominator = left - 2 * mid + right
        offset = 0.5 * (left - right) / denominator if abs(denominator) > 1e-9 else 0.0
        radius = radii[column] + float(np.clip(offset, -1, 1)) * (radii[1] - radii[0])
        points.append(
            (
                cx + radius * math.cos(angles[ray]),
                cy + radius * math.sin(angles[ray]),
                float(angles[ray]),
            )
        )
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def center_uncertainty(residual_rms: float, angles: np.ndarray) -> float:
    """由残差与角度覆盖推出圆心标准误差（px）。覆盖越完整越稳。"""
    n = len(angles)
    if n < 3:
        return float("inf")
    design = np.column_stack([np.cos(angles), np.sin(angles), np.ones(n)])
    try:
        covariance = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(residual_rms * math.sqrt(max(covariance[0, 0], covariance[1, 1])))


# =========================================================================
# 单图识别
# =========================================================================
@dataclass(frozen=True)
class HoleLocatorConfig:
    """孔定位参数。

    直径先验默认 910 µm：2026-08 在 Zem20F（192.168.0.15）100x 下实测的
    孔内缘直径（环带外缘约 1071 µm，标称 1 mm 量的是环带）。像素尺寸来自
    设备扫描视场读数，是权威值。
    """

    hole_diameter_um: float = 910.0
    radius_tolerance: float = 0.15
    # full（整圆可信）三道自检
    full_min_visible_deg: float = 180.0     # 圆在视场内的理论可见角下限
    full_coverage_ratio: float = 0.55       # 实际覆盖角 / 理论可见角 下限（环带脏污会遮断边缘，
                                            # 真机实测脏污可遮 1/3 圈；直径+RMS 两道门兜底）
    full_max_rms_px: float = 1.5
    diameter_tolerance_ratio: float = 0.08  # 拟合直径 vs 先验的相对容差
    # arc（弧可用）判据
    arc_min_inliers: int = 25
    arc_min_coverage_deg: float = 30.0
    arc_max_rms_px: float = 2.5
    arc_min_density: float = 0.75          # 内点角向连续性下限（真弧≈1，噪声≈0.5）
    # 环带脏污或信噪比较低时，真实孔边缘可能断续。仅在覆盖、RMS、内点数都很强时，
    # 允许较低连续密度通过；不能直接下调上面的常规门限，否则容易把噪声拼成短弧。
    fragmented_arc_min_inliers: int = 35
    fragmented_arc_min_coverage_deg: float = 120.0
    fragmented_arc_max_rms_px: float = 1.5
    fragmented_arc_min_density: float = 0.35
    fragmented_arc_min_radius_px: float = 200.0
    # 闭环
    search_step_ratio: float = 0.60         # 搜索步长 = 视场短边 × 比例
    max_search_radius_um: float = 2000.0    # 所有移台目标距粗坐标的上限
    max_captures: int = 25
    max_center_moves: int = 6               # 弧追踪 + 居中 移台次数上限
    recenter_offset_ratio: float = 0.20     # full 但偏心超视场短边该比例 → 再居中
    accept_offset_ratio: float = 0.35       # 最终接受时圆心离图中心的上限比例
    # 防回程差：每次移台先到 目标+offset 再正向进入目标，统一逼近方向。
    # Zem20F 实测换向回程差 X≈85µm / Y≈17µm，同向重复性 ≤2µm。
    approach_offset_um: tuple[float, float] = (-150.0, -150.0)


@dataclass(frozen=True)
class HoleDetection:
    status: str                              # "full" | "arc" | "none"
    center_px: tuple[float, float] | None = None
    radius_px: float = float("nan")
    diameter_um: float = float("nan")
    coverage_deg: float = 0.0
    visible_deg: float = 0.0
    span_deg: float = 0.0
    density: float = 0.0
    rms_px: float = float("nan")
    sigma_px: float = float("nan")
    inlier_count: int = 0
    vote_contrast: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "centerPx": list(self.center_px) if self.center_px else None,
            "radiusPx": _json_float(self.radius_px),
            "diameterUm": _json_float(self.diameter_um),
            "coverageDeg": self.coverage_deg,
            "visibleDeg": self.visible_deg,
            "spanDeg": self.span_deg,
            "density": self.density,
            "rmsPx": _json_float(self.rms_px),
            "sigmaPx": _json_float(self.sigma_px),
            "inlierCount": self.inlier_count,
            "voteContrast": self.vote_contrast,
            "note": self.note,
        }


def _json_float(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _visible_arc_deg(
    center: tuple[float, float],
    radius: float,
    shape: tuple[int, int],
    margin: float = 2.0,
) -> float:
    """圆周落在图像内的理论角度（度），用于判断"整圆自检"是否可能通过。"""
    height, width = shape
    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    px = center[0] + radius * np.cos(angles)
    py = center[1] + radius * np.sin(angles)
    inside = (
        (px >= margin) & (px < width - margin) & (py >= margin) & (py < height - margin)
    )
    return float(inside.mean() * 360.0)


def _coverage_deg(angles: np.ndarray, bin_deg: float = 5.0) -> float:
    occupied = np.unique((np.degrees(angles) // bin_deg).astype(int))
    return float(len(occupied) * bin_deg)


def _span_and_density(angles: np.ndarray, bin_deg: float = 5.0) -> tuple[float, float]:
    """内点的角向跨度与连续密度。真实弧是连续曲线（密度≈1），噪声拼弧是稀疏散点。"""
    bins = np.unique((np.degrees(angles) % 360.0 // bin_deg).astype(int))
    if len(bins) < 2:
        return float(len(bins) * bin_deg), 1.0
    positions = np.sort(bins * bin_deg)
    gaps = np.diff(np.concatenate([positions, [positions[0] + 360.0]]))
    span = 360.0 - float(gaps.max()) + bin_deg
    coverage = float(len(bins) * bin_deg)
    return span, coverage / max(span, bin_deg)


def _arc_quality_ok(candidate: HoleDetection, config: HoleLocatorConfig) -> bool:
    """接受连续弧，或几何证据足够强的断续长弧。"""
    basic_quality = (
        candidate.inlier_count >= config.arc_min_inliers
        and candidate.coverage_deg >= config.arc_min_coverage_deg
        and candidate.rms_px <= config.arc_max_rms_px
    )
    if not basic_quality:
        return False
    if candidate.density >= config.arc_min_density:
        return True
    return (
        candidate.inlier_count >= config.fragmented_arc_min_inliers
        and candidate.coverage_deg >= config.fragmented_arc_min_coverage_deg
        and candidate.rms_px <= config.fragmented_arc_max_rms_px
        and candidate.density >= config.fragmented_arc_min_density
        and candidate.radius_px >= config.fragmented_arc_min_radius_px
    )


def _iterative_free_fit(
    flat: np.ndarray,
    start_center: tuple[float, float],
    r_lo: float,
    r_hi: float,
    polarity: float,
) -> tuple[tuple[float, float, float], np.ndarray, np.ndarray] | None:
    """迭代精化的自由半径拟合：极坐标取点 → 先验窗过滤 → RANSAC，×3 轮。"""
    center = start_center
    model: tuple[float, float, float] | None = None
    points: np.ndarray | None = None
    inliers: np.ndarray | None = None
    for _ in range(3):
        candidates = polar_edge_points(flat, center, r_lo, r_hi, polarity=polarity)
        if len(candidates) < 30:
            return None
        radius_of_point = np.hypot(
            candidates[:, 0] - center[0], candidates[:, 1] - center[1]
        )
        candidates = candidates[(radius_of_point > r_lo) & (radius_of_point < r_hi)]
        if len(candidates) < 30:
            return None
        # 真机 BSE 边缘较毛糙，容差 2.0px 提高内点回收率（RMS 门仍保证拟合质量）
        fitted = ransac_circle(candidates[:, :2], tolerance_px=2.0)
        if fitted is None:
            return None
        model, inliers = fitted
        points = candidates
        center = (model[0], model[1])
    assert model is not None and points is not None and inliers is not None
    return model, points, inliers


def detect_hole(
    image: np.ndarray,
    pixel_size_um: float,
    *,
    config: HoleLocatorConfig = HoleLocatorConfig(),
    info_bar_px: int = 0,
) -> HoleDetection:
    """在单张 SEM 图上识别孔，返回 full / arc / none 三态结果。"""
    array = np.asarray(image)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if info_bar_px > 0:
        array = array[: array.shape[0] - info_bar_px]
    height, width = array.shape
    pixel_size = float(pixel_size_um)
    if not math.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError(f"像素尺寸无效：{pixel_size_um}")

    r_expect = (config.hole_diameter_um / 2.0) / pixel_size
    if r_expect < 20:
        return HoleDetection(
            status="none",
            note=f"倍率过低：先验半径仅 {r_expect:.1f} px",
        )
    diagonal = math.hypot(height, width)
    if r_expect * (1 - config.radius_tolerance) > diagonal:
        return HoleDetection(
            status="none",
            note=f"倍率过高：先验半径 {r_expect:.0f} px 超出图像对角线",
        )

    flat = flatten(array)
    coarse_x, coarse_y, contrast = vote_with_prior(
        flat, r_expect, config.radius_tolerance
    )
    if not math.isfinite(coarse_x):
        return HoleDetection(status="none", vote_contrast=contrast, note="投票无峰")

    r_lo = r_expect * (1 - config.radius_tolerance)
    r_hi = r_expect * (1 + config.radius_tolerance)
    polarities = (-1.0, 1.0, 0.0)  # 内缘（亮→暗）、外缘（暗→亮）、无符号兜底

    # ---- 路线 1：自由半径整圆拟合（按极性分别取点，质量门选优） -----------
    full_candidates: list[HoleDetection] = []
    mismatch_candidates: list[HoleDetection] = []
    for polarity in polarities:
        fit = _iterative_free_fit(flat, (coarse_x, coarse_y), r_lo, r_hi, polarity)
        if fit is None:
            continue
        (cx, cy, r), points, inliers = fit
        kept = points[inliers]
        residual = np.hypot(kept[:, 0] - cx, kept[:, 1] - cy) - r
        rms = float(np.sqrt(np.mean(residual**2)))
        angles = np.arctan2(kept[:, 1] - cy, kept[:, 0] - cx)
        coverage = _coverage_deg(angles)
        span, density = _span_and_density(angles)
        visible = _visible_arc_deg((cx, cy), r, (height, width))
        diameter_um = 2.0 * r * pixel_size
        diameter_ok = (
            abs(diameter_um - config.hole_diameter_um)
            <= config.hole_diameter_um * config.diameter_tolerance_ratio
        )
        coverage_ok = (
            visible >= config.full_min_visible_deg
            and coverage >= visible * config.full_coverage_ratio
        )
        if not (coverage_ok and rms <= config.full_max_rms_px):
            continue
        candidate = HoleDetection(
            status="full" if diameter_ok else "none",
            center_px=(float(cx), float(cy)),
            radius_px=float(r),
            diameter_um=diameter_um,
            coverage_deg=coverage,
            visible_deg=visible,
            span_deg=span,
            density=density,
            rms_px=rms,
            sigma_px=center_uncertainty(rms, angles),
            inlier_count=int(inliers.sum()),
            vote_contrast=contrast,
            note=(
                f"自由半径拟合通过三道自检（极性 {polarity:+.0f}）"
                if diameter_ok
                else f"拟合直径 {diameter_um:.0f} µm 与先验 "
                f"{config.hole_diameter_um:.0f} µm 不符，拒绝"
            ),
        )
        if diameter_ok:
            full_candidates.append(candidate)
        else:
            mismatch_candidates.append(candidate)

    if full_candidates:
        return max(
            full_candidates,
            key=lambda item: (item.coverage_deg, item.inlier_count),
        )
    if mismatch_candidates:
        # 结构上是一个清晰的圆，但尺寸与先验对不上：说明倍率/先验有问题，
        # 此时用假先验去做固定半径弧拟合只会给出自信的错误答案，必须拒绝。
        best = max(
            mismatch_candidates,
            key=lambda item: (item.coverage_deg, item.inlier_count),
        )
        return HoleDetection(
            status="none",
            coverage_deg=best.coverage_deg,
            visible_deg=best.visible_deg,
            span_deg=best.span_deg,
            density=best.density,
            rms_px=best.rms_px,
            diameter_um=best.diameter_um,
            inlier_count=best.inlier_count,
            vote_contrast=contrast,
            note=best.note,
        )

    # ---- 路线 2：固定半径弧拟合（同样按极性选优） ---------------------------
    arc_candidates: list[HoleDetection] = []
    rejected_arcs: list[HoleDetection] = []
    for polarity in polarities:
        arc_points = polar_edge_points(
            flat, (coarse_x, coarse_y), r_lo, r_hi, polarity=polarity
        )
        if len(arc_points) < config.arc_min_inliers:
            continue
        radius_of_point = np.hypot(
            arc_points[:, 0] - coarse_x, arc_points[:, 1] - coarse_y
        )
        arc_points = arc_points[(radius_of_point > r_lo) & (radius_of_point < r_hi)]
        if len(arc_points) < config.arc_min_inliers:
            continue
        fitted = robust_center_fixed_radius(
            arc_points[:, :2],
            r_expect,
            (coarse_x, coarse_y),
            tolerance_px=config.arc_max_rms_px,
        )
        if fitted is None:
            continue
        (cx, cy), keep = fitted
        kept = arc_points[keep]
        residual = np.hypot(kept[:, 0] - cx, kept[:, 1] - cy) - r_expect
        rms = float(np.sqrt(np.mean(residual**2)))
        angles = np.arctan2(kept[:, 1] - cy, kept[:, 0] - cx)
        coverage = _coverage_deg(angles)
        span, density = _span_and_density(angles)
        candidate = HoleDetection(
            status="arc",
            center_px=(float(cx), float(cy)),
            radius_px=float(r_expect),
            diameter_um=config.hole_diameter_um,
            coverage_deg=coverage,
            visible_deg=_visible_arc_deg((cx, cy), r_expect, (height, width)),
            span_deg=span,
            density=density,
            rms_px=rms,
            sigma_px=center_uncertainty(rms, angles),
            inlier_count=int(keep.sum()),
            vote_contrast=contrast,
            note=f"固定半径弧拟合（圆心为估计值，极性 {polarity:+.0f}）",
        )
        if _arc_quality_ok(candidate, config):
            arc_candidates.append(candidate)
        else:
            rejected_arcs.append(candidate)

    if arc_candidates:
        return max(
            arc_candidates,
            key=lambda item: (item.coverage_deg, item.density),
        )
    if rejected_arcs:
        best = max(rejected_arcs, key=lambda item: item.coverage_deg)
        return HoleDetection(
            status="none",
            coverage_deg=best.coverage_deg,
            span_deg=best.span_deg,
            density=best.density,
            rms_px=best.rms_px,
            inlier_count=best.inlier_count,
            vote_contrast=contrast,
            note=f"弧未通过质量门（密度 {best.density:.2f}，覆盖 {best.coverage_deg:.0f}°）",
        )

    return HoleDetection(
        status="none",
        vote_contrast=contrast,
        note="边缘点不足或拟合未通过质量门",
    )


def annotate_hole_detection(
    image: np.ndarray,
    detection: HoleDetection,
    *,
    info_bar_px: int = 0,
) -> np.ndarray:
    """生成识别结果叠加图（CLAHE 增强 + 圆/圆心标注），用于人工复核。"""
    array = np.asarray(image)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if info_bar_px > 0:
        array = array[: array.shape[0] - info_bar_px]
    if array.dtype != np.uint8:
        source = array.astype(np.float32)
        low, high = float(source.min()), float(source.max())
        array = (
            np.zeros(source.shape, np.uint8)
            if high <= low
            else np.clip((source - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
        )
    canvas = cv2.cvtColor(
        cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8)).apply(array),
        cv2.COLOR_GRAY2BGR,
    )
    color = {"full": (0, 230, 0), "arc": (0, 180, 255), "none": (0, 0, 255)}[
        detection.status
    ]
    if detection.center_px is not None and math.isfinite(detection.radius_px):
        center = (
            int(round(detection.center_px[0])),
            int(round(detection.center_px[1])),
        )
        cv2.circle(canvas, center, int(round(detection.radius_px)), color, 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, center, color, cv2.MARKER_CROSS, 40, 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"{detection.status} cov={detection.coverage_deg:.0f} "
        f"rms={detection.rms_px:.2f} d={detection.diameter_um:.0f}um",
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    return canvas


# =========================================================================
# 闭环定位
# =========================================================================
def _search_offsets(step_um: float, max_radius_um: float) -> list[tuple[float, float]]:
    """方形螺旋扩环的搜索偏移序列（含原点），按距离排序并限制在半径内。"""
    max_index = int(math.ceil(max_radius_um / step_um)) + 1
    offsets: list[tuple[float, float]] = []
    for ix in range(-max_index, max_index + 1):
        for iy in range(-max_index, max_index + 1):
            dx, dy = ix * step_um, iy * step_um
            if math.hypot(dx, dy) <= max_radius_um + 1e-6:
                offsets.append((dx, dy))
    offsets.sort(key=lambda item: (math.hypot(*item), abs(item[1]), abs(item[0])))
    return offsets


def _pixel_offset_to_stage(
    center_px: tuple[float, float],
    image_shape: tuple[int, int],
    pixel_size_um: float,
    stage_xy: tuple[float, float],
    axis_signs: Sequence[float],
    pixel_axis_matrix: Sequence[Sequence[float]],
    field_rotation_deg: float,
) -> tuple[float, float]:
    """像素圆心 → 电机坐标，约定与 cleanliness_detection.pixel_to_stage_xy 一致。"""
    height, width = image_shape
    offset = np.asarray(
        [center_px[0] - width / 2.0, center_px[1] - height / 2.0],
        dtype=np.float64,
    ) * float(pixel_size_um)
    axis = np.diag(axis_signs) @ np.asarray(pixel_axis_matrix, dtype=np.float64)
    radians = math.radians(float(field_rotation_deg))
    rotation = np.asarray(
        [
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ]
    )
    target = np.asarray(stage_xy, dtype=np.float64) + rotation @ axis @ offset
    return float(target[0]), float(target[1])


def _clamp_to_search_disc(
    target: tuple[float, float],
    origin: tuple[float, float],
    max_radius_um: float,
) -> tuple[float, float]:
    dx, dy = target[0] - origin[0], target[1] - origin[1]
    distance = math.hypot(dx, dy)
    if distance <= max_radius_um:
        return target
    scale = max_radius_um / distance
    return origin[0] + dx * scale, origin[1] + dy * scale


@dataclass
class _LoopState:
    axis_signs: list[float] = field(default_factory=lambda: [1.0, 1.0])
    axis_verified: list[bool] = field(default_factory=lambda: [False, False])
    last_center_px: tuple[float, float] | None = None
    last_stage: tuple[float, float] | None = None


def _update_axis_signs(
    state: _LoopState,
    new_center_px: tuple[float, float],
    new_stage: tuple[float, float],
    pixel_size_um: float,
    pixel_axis_matrix: Sequence[Sequence[float]],
    field_rotation_deg: float,
    minimum_shift_px: float = 15.0,
) -> str | None:
    """两张都识别到圆心的图之间，用移台量校验像素轴与电机轴的方向符号。

    电机移动 Δs 后，圆心在图上应当反向移动 M⁻¹·R⁻¹·Δs / pixel_size。
    若观测位移与预期显著反号，翻转该轴符号（每轴只翻一次）。
    """
    if state.last_center_px is None or state.last_stage is None:
        return None
    delta_stage = np.asarray(new_stage) - np.asarray(state.last_stage)
    radians = math.radians(float(field_rotation_deg))
    rotation = np.asarray(
        [
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ]
    )
    axis = np.diag(state.axis_signs) @ np.asarray(pixel_axis_matrix, dtype=np.float64)
    try:
        expected_px = -np.linalg.solve(rotation @ axis, delta_stage) / pixel_size_um
    except np.linalg.LinAlgError:
        return None
    observed_px = np.asarray(new_center_px) - np.asarray(state.last_center_px)
    notes: list[str] = []
    for index in range(2):
        if state.axis_verified[index]:
            continue
        if abs(expected_px[index]) < minimum_shift_px:
            continue
        if observed_px[index] * expected_px[index] < 0:
            state.axis_signs[index] = -state.axis_signs[index]
            notes.append(
                f"轴{'XY'[index]}方向与预期相反，符号翻转为 {state.axis_signs[index]:+.0f}"
            )
        state.axis_verified[index] = True
    return "；".join(notes) if notes else None


def locate_hole_center(
    microscope: Any,
    coarse_stage_xy: tuple[float, float],
    work_dir: Path,
    *,
    config: HoleLocatorConfig = HoleLocatorConfig(),
    pixel_axis_matrix: Sequence[Sequence[float]] = ((1.0, 0.0), (0.0, 1.0)),
    field_rotation_deg: float = 0.0,
    stage_error_um: float = 2.0,
    label: str = "hole",
    progress: ProgressCallback | None = None,
    should_stop: StopCallback | None = None,
) -> dict[str, Any]:
    """闭环精定位孔圆心，返回 JSON 可序列化的结果字典。

    ``microscope`` 需要提供 ``move_xy(x, y)`` 与 ``capture(path)``
    （与 ``ZemMicroscope`` 相同的约定）。失败抛 ``RuntimeError``。
    """
    from backend.cleanliness_core.cleanliness_task import _move_stage_checked

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    def stopped() -> bool:
        return bool(should_stop()) if should_stop is not None else False

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    origin = (float(coarse_stage_xy[0]), float(coarse_stage_xy[1]))
    state = _LoopState()
    captures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    capture_index = 0
    center_moves = 0
    search_queue: list[tuple[float, float]] | None = None
    search_cursor = 0
    pending_target: tuple[float, float] = origin
    step_um: float | None = None

    while capture_index < config.max_captures:
        if stopped():
            raise InterruptedError("用户停止了孔定位")

        target = _clamp_to_search_disc(pending_target, origin, config.max_search_radius_um)
        report(
            f"[{label}] 移台到 ({target[0]:.1f}, {target[1]:.1f}) µm 并拍图"
            f"（第 {capture_index + 1} 张）"
        )
        # 防回程差：先到偏置点，再统一方向进入目标
        approach_dx, approach_dy = config.approach_offset_um
        if approach_dx or approach_dy:
            microscope.move_xy(target[0] + approach_dx, target[1] + approach_dy)
        move = _move_stage_checked(
            microscope,
            target,
            maximum_error_um=stage_error_um,
            raise_on_excess=False,
        )
        stage_after = (
            float(move["after"]["x"]),
            float(move["after"]["y"]),
        )
        stage_warning = move.get("warning")
        if stage_warning:
            warning_record = {
                "hole": label,
                "captureIndex": capture_index + 1,
                "message": str(stage_warning),
                "thresholdUm": float(stage_error_um),
                "positionErrorUm": float(move["positionErrorUm"]),
                "stageTarget": {"x": target[0], "y": target[1]},
                "stageAfter": {"x": stage_after[0], "y": stage_after[1]},
            }
            warnings.append(warning_record)
            report(
                f"[{label}] 警告：{stage_warning}；"
                f"将按实际回读坐标 ({stage_after[0]:.1f}, {stage_after[1]:.1f}) µm 继续"
            )

        capture_index += 1
        raw_path = work_dir / f"{label}_{capture_index:02d}.png"
        image, capture = microscope.capture(raw_path)
        pixel_size = float(capture["pixelSizeUm"])
        shape = tuple(int(v) for v in np.asarray(image).shape[:2])
        if step_um is None:
            step_um = min(shape) * pixel_size * config.search_step_ratio

        detection = detect_hole(image, pixel_size, config=config)
        annotated = annotate_hole_detection(image, detection)
        annotated_path = raw_path.with_name(f"{raw_path.stem}.annotated.png")
        cv2.imwrite(str(annotated_path), annotated)

        record: dict[str, Any] = {
            "index": capture_index,
            "stageTarget": {"x": target[0], "y": target[1]},
            "stageAfter": {"x": stage_after[0], "y": stage_after[1]},
            "positionErrorUm": move.get("positionErrorUm"),
            "pixelSizeUm": pixel_size,
            "rawPath": str(raw_path),
            "displayPath": capture.get("displayPath"),
            "annotatedPath": str(annotated_path),
            "detection": detection.to_dict(),
        }
        if stage_warning:
            record["warning"] = str(stage_warning)
        captures.append(record)
        report(
            f"[{label}] 识别结果：{detection.status}"
            f"（覆盖 {detection.coverage_deg:.0f}°，RMS {detection.rms_px:.2f} px）"
        )

        if detection.status in {"full", "arc"} and detection.center_px is not None:
            axis_note = _update_axis_signs(
                state,
                detection.center_px,
                stage_after,
                pixel_size,
                pixel_axis_matrix,
                field_rotation_deg,
            )
            if axis_note:
                record["axisNote"] = axis_note
                report(f"[{label}] {axis_note}")
            state.last_center_px = detection.center_px
            state.last_stage = stage_after

            hole_stage = _pixel_offset_to_stage(
                detection.center_px,
                shape,
                pixel_size,
                stage_after,
                state.axis_signs,
                pixel_axis_matrix,
                field_rotation_deg,
            )
            record["holeStageEstimate"] = {"x": hole_stage[0], "y": hole_stage[1]}
            offset_px = math.hypot(
                detection.center_px[0] - shape[1] / 2.0,
                detection.center_px[1] - shape[0] / 2.0,
            )
            offset_ratio = offset_px / min(shape)

            if detection.status == "full" and offset_ratio <= config.accept_offset_ratio:
                if (
                    offset_ratio > config.recenter_offset_ratio
                    and center_moves < config.max_center_moves
                ):
                    center_moves += 1
                    pending_target = hole_stage
                    report(
                        f"[{label}] 整圆可信但偏心 {offset_ratio * 100:.0f}%，"
                        f"再居中一次（第 {center_moves} 次移心）"
                    )
                    continue
                result = {
                    "ok": True,
                    "label": label,
                    "holeStage": {"x": hole_stage[0], "y": hole_stage[1]},
                    "sigmaUm": _json_float(detection.sigma_px * pixel_size),
                    "diameterUm": _json_float(detection.diameter_um),
                    "coverageDeg": detection.coverage_deg,
                    "rmsPx": _json_float(detection.rms_px),
                    "pixelSizeUm": pixel_size,
                    "coarseStage": {"x": origin[0], "y": origin[1]},
                    "offsetFromCoarseUm": math.hypot(
                        hole_stage[0] - origin[0], hole_stage[1] - origin[1]
                    ),
                    "axisSigns": list(state.axis_signs),
                    "searchStepUm": step_um,
                    "captureCount": capture_index,
                    "centerMoves": center_moves,
                    "finalCapture": record,
                    "captures": captures,
                    "warnings": warnings,
                }
                report(
                    f"[{label}] 定位完成：({hole_stage[0]:.2f}, {hole_stage[1]:.2f}) µm，"
                    f"σ≈{(detection.sigma_px * pixel_size):.2f} µm，共 {capture_index} 张图"
                )
                return result

            # arc 或 full-但太偏：向估计圆心追一步；追踪次数用尽则退回扩环搜索
            if center_moves < config.max_center_moves:
                center_moves += 1
                pending_target = hole_stage
                report(
                    f"[{label}] 向估计圆心追踪（第 {center_moves} 次移心，"
                    f"目标 ({hole_stage[0]:.1f}, {hole_stage[1]:.1f}) µm）"
                )
                continue
            report(f"[{label}] 移心追踪次数用尽，退回扩环搜索")

        # none：进入/继续扩环搜索
        if search_queue is None:
            assert step_um is not None
            search_queue = _search_offsets(step_um, config.max_search_radius_um)
            search_cursor = 0
        advanced = False
        while search_cursor < len(search_queue):
            dx, dy = search_queue[search_cursor]
            search_cursor += 1
            candidate = (origin[0] + dx, origin[1] + dy)
            if math.hypot(candidate[0] - target[0], candidate[1] - target[1]) < 1e-3:
                continue  # 跳过当前已经拍过的位置
            pending_target = candidate
            advanced = True
            break
        if not advanced:
            raise RuntimeError(
                f"[{label}] 搜索范围（半径 {config.max_search_radius_um:.0f} µm）内未找到孔"
            )
        report(f"[{label}] 视场内没有孔特征，扩环搜索下一个位置")

    raise RuntimeError(
        f"[{label}] 拍图次数达到上限（{config.max_captures}）仍未完成定位"
    )
