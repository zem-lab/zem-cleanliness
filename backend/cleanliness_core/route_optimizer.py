"""Deterministic stage-route optimization for particle acquisition.

The stage starts at a fixed physical position.  Small routes are solved exactly
with Held-Karp dynamic programming; larger routes use multiple nearest-neighbor
seeds followed by 2-opt improvement.  The original nearest-neighbor route is
always retained as a candidate, so optimization can never increase travel.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Generic, Iterable, TypeVar


T = TypeVar("T")
_EPSILON = 1e-9


@dataclass(frozen=True)
class RoutePlan(Generic[T]):
    items: tuple[T, ...]
    indices: tuple[int, ...]
    algorithm: str
    exact: bool
    return_to_start: bool
    baseline_distance_um: float
    optimized_distance_um: float

    def to_audit_dict(self) -> dict[str, object]:
        saved = max(0.0, self.baseline_distance_um - self.optimized_distance_um)
        improvement = (
            saved * 100.0 / self.baseline_distance_um
            if self.baseline_distance_um > _EPSILON
            else 0.0
        )
        return {
            "algorithm": self.algorithm,
            "exact": self.exact,
            "returnToStart": self.return_to_start,
            "baselineAlgorithm": "nearest_neighbor",
            "baselineDistanceUm": self.baseline_distance_um,
            "optimizedDistanceUm": self.optimized_distance_um,
            "savedDistanceUm": saved,
            "improvementPercent": improvement,
        }


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _route_distance(
    route: tuple[int, ...] | list[int],
    start_distances: list[float],
    distances: list[list[float]],
    *,
    return_to_start: bool,
) -> float:
    if not route:
        return 0.0
    total = start_distances[route[0]]
    total += sum(distances[left][right] for left, right in zip(route, route[1:]))
    if return_to_start:
        total += start_distances[route[-1]]
    return total


def _nearest_neighbor(
    count: int,
    start_distances: list[float],
    distances: list[list[float]],
    *,
    first: int | None = None,
) -> tuple[int, ...]:
    remaining = set(range(count))
    route: list[int] = []
    current: int | None = None
    if first is not None:
        route.append(first)
        remaining.remove(first)
        current = first
    while remaining:
        next_index = min(
            remaining,
            key=lambda index: (
                start_distances[index]
                if current is None
                else distances[current][index],
                index,
            ),
        )
        route.append(next_index)
        remaining.remove(next_index)
        current = next_index
    return tuple(route)


def _held_karp(
    count: int,
    start_distances: list[float],
    distances: list[list[float]],
    *,
    return_to_start: bool,
) -> tuple[int, ...]:
    costs: dict[tuple[int, int], float] = {
        (1 << index, index): start_distances[index] for index in range(count)
    }
    parents: dict[tuple[int, int], int] = {}
    full_mask = (1 << count) - 1
    for mask in range(1, full_mask + 1):
        if mask & (mask - 1) == 0:
            continue
        for last in range(count):
            last_bit = 1 << last
            if not mask & last_bit:
                continue
            previous_mask = mask ^ last_bit
            best_cost = math.inf
            best_previous = -1
            for previous in range(count):
                if not previous_mask & (1 << previous):
                    continue
                candidate = costs[(previous_mask, previous)] + distances[previous][last]
                if candidate < best_cost - _EPSILON or (
                    abs(candidate - best_cost) <= _EPSILON
                    and (best_previous < 0 or previous < best_previous)
                ):
                    best_cost = candidate
                    best_previous = previous
            costs[(mask, last)] = best_cost
            parents[(mask, last)] = best_previous

    best_last = min(
        range(count),
        key=lambda last: (
            costs[(full_mask, last)]
            + (start_distances[last] if return_to_start else 0.0),
            last,
        ),
    )
    reverse_route = [best_last]
    mask = full_mask
    while mask & (mask - 1):
        previous = parents[(mask, reverse_route[-1])]
        mask ^= 1 << reverse_route[-1]
        reverse_route.append(previous)
    return tuple(reversed(reverse_route))


def _two_opt(
    route: tuple[int, ...],
    start_distances: list[float],
    distances: list[list[float]],
    *,
    return_to_start: bool,
) -> tuple[int, ...]:
    improved = list(route)
    count = len(improved)
    # Best-improvement passes are deterministic.  The bound prevents a
    # pathological large data set from spending unbounded time in planning.
    for _pass in range(min(50, max(1, count))):
        best_delta = -_EPSILON
        best_pair: tuple[int, int] | None = None
        for left in range(count - 1):
            previous = improved[left - 1] if left else None
            for right in range(left + 1, count):
                following = improved[right + 1] if right + 1 < count else None
                old_left = (
                    start_distances[improved[left]]
                    if previous is None
                    else distances[previous][improved[left]]
                )
                new_left = (
                    start_distances[improved[right]]
                    if previous is None
                    else distances[previous][improved[right]]
                )
                old_right = (
                    distances[improved[right]][following]
                    if following is not None
                    else start_distances[improved[right]] if return_to_start else 0.0
                )
                new_right = (
                    distances[improved[left]][following]
                    if following is not None
                    else start_distances[improved[left]] if return_to_start else 0.0
                )
                delta = new_left + new_right - old_left - old_right
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (left, right)
        if best_pair is None:
            break
        left, right = best_pair
        improved[left : right + 1] = reversed(improved[left : right + 1])
    return tuple(improved)


def _seed_indices(
    points: list[tuple[float, float]], start: tuple[float, float], start_distances: list[float]
) -> tuple[int, ...]:
    count = len(points)
    if count <= 32:
        return tuple(range(count))
    nearest = sorted(range(count), key=lambda index: (start_distances[index], index))[:16]
    extremes = {
        min(range(count), key=lambda index: (points[index][0], index)),
        max(range(count), key=lambda index: (points[index][0], -index)),
        min(range(count), key=lambda index: (points[index][1], index)),
        max(range(count), key=lambda index: (points[index][1], -index)),
        max(range(count), key=lambda index: (start_distances[index], -index)),
    }
    angular = []
    for sector in range(8):
        target_angle = -math.pi + (2.0 * math.pi * sector / 8.0)
        angular.append(
            min(
                range(count),
                key=lambda index: (
                    abs(
                        math.atan2(
                            math.sin(math.atan2(points[index][1] - start[1], points[index][0] - start[0]) - target_angle),
                            math.cos(math.atan2(points[index][1] - start[1], points[index][0] - start[0]) - target_angle),
                        )
                    ),
                    start_distances[index],
                    index,
                ),
            )
        )
    return tuple(dict.fromkeys([*nearest, *sorted(extremes), *angular]))


def optimize_route(
    items: Iterable[T],
    position: Callable[[T], tuple[float, float]],
    start_position: tuple[float, float],
    *,
    return_to_start: bool = False,
    exact_limit: int = 12,
) -> RoutePlan[T]:
    materialized = tuple(items)
    start = (float(start_position[0]), float(start_position[1]))
    if not all(math.isfinite(value) for value in start):
        raise ValueError("路线起点必须是有限二维坐标")
    points: list[tuple[float, float]] = []
    for item in materialized:
        raw = position(item)
        point = (float(raw[0]), float(raw[1]))
        if not all(math.isfinite(value) for value in point):
            raise ValueError("颗粒路线包含无效台坐标")
        points.append(point)

    count = len(points)
    if count == 0:
        return RoutePlan((), (), "empty", True, return_to_start, 0.0, 0.0)
    start_distances = [_distance(start, point) for point in points]
    distances = [
        [_distance(left, right) for right in points]
        for left in points
    ]
    baseline = _nearest_neighbor(count, start_distances, distances)
    baseline_distance = _route_distance(
        baseline, start_distances, distances, return_to_start=return_to_start
    )

    if count <= max(1, int(exact_limit)):
        optimized = _held_karp(
            count,
            start_distances,
            distances,
            return_to_start=return_to_start,
        )
        algorithm = "held_karp_exact"
        exact = True
    else:
        raw_candidates = {baseline}
        for first in _seed_indices(points, start, start_distances):
            raw_candidates.add(
                _nearest_neighbor(count, start_distances, distances, first=first)
            )
        ranked = sorted(
            raw_candidates,
            key=lambda route: (
                _route_distance(
                    route,
                    start_distances,
                    distances,
                    return_to_start=return_to_start,
                ),
                route,
            ),
        )
        optimized_candidates = {
            _two_opt(
                route,
                start_distances,
                distances,
                return_to_start=return_to_start,
            )
            for route in ranked[:6]
        }
        optimized_candidates.add(baseline)
        optimized = min(
            optimized_candidates,
            key=lambda route: (
                _route_distance(
                    route,
                    start_distances,
                    distances,
                    return_to_start=return_to_start,
                ),
                route,
            ),
        )
        algorithm = "multi_start_nearest_neighbor_2opt"
        exact = False

    optimized_distance = _route_distance(
        optimized, start_distances, distances, return_to_start=return_to_start
    )
    if optimized_distance > baseline_distance + _EPSILON:
        optimized = baseline
        optimized_distance = baseline_distance
    return RoutePlan(
        tuple(materialized[index] for index in optimized),
        optimized,
        algorithm,
        exact,
        bool(return_to_start),
        baseline_distance,
        optimized_distance,
    )
