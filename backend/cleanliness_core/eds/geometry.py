from __future__ import annotations

from typing import Iterable


def particle_geometric_center(points: Iterable[Iterable[float]]) -> tuple[float, float]:
    """Return the polygon centroid, falling back to point average for degenerate input."""
    pts = [(float(p[0]), float(p[1])) for p in points or []]
    if not pts:
        return 0.0, 0.0
    if len(pts) < 3:
        return _mean_point(pts)

    signed_area = 0.0
    cx = 0.0
    cy = 0.0
    for idx, (x0, y0) in enumerate(pts):
        x1, y1 = pts[(idx + 1) % len(pts)]
        cross = x0 * y1 - x1 * y0
        signed_area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross

    signed_area *= 0.5
    if abs(signed_area) < 1e-9:
        return _mean_point(pts)

    return cx / (6.0 * signed_area), cy / (6.0 * signed_area)


def particle_center_pixel(points: Iterable[Iterable[float]]) -> tuple[int, int]:
    x, y = particle_geometric_center(points)
    return max(0, int(round(x))), max(0, int(round(y)))


def _mean_point(points: list[tuple[float, float]]) -> tuple[float, float]:
    return sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points)
