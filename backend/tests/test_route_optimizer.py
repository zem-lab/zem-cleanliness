from __future__ import annotations

import itertools
import math
import random

import pytest

from backend.cleanliness_core.route_optimizer import optimize_route


def distance(order, points, start=(0.0, 0.0), *, return_to_start=False):
    current = start
    total = 0.0
    for index in order:
        total += math.dist(current, points[index])
        current = points[index]
    if return_to_start:
        total += math.dist(current, start)
    return total


@pytest.mark.parametrize("return_to_start", [False, True])
def test_small_route_is_globally_optimal(return_to_start):
    points = [(7.0, 1.0), (2.0, 8.0), (-4.0, 5.0), (3.0, 3.0), (-1.0, -6.0)]
    plan = optimize_route(
        range(len(points)),
        lambda index: points[index],
        (0.0, 0.0),
        return_to_start=return_to_start,
    )
    expected = min(
        distance(order, points, return_to_start=return_to_start)
        for order in itertools.permutations(range(len(points)))
    )
    assert plan.exact is True
    assert plan.algorithm == "held_karp_exact"
    assert plan.optimized_distance_um == pytest.approx(expected)
    assert set(plan.indices) == set(range(len(points)))


def test_large_route_never_regresses_from_current_greedy_and_is_deterministic():
    generator = random.Random(20260921)
    points = [(generator.uniform(-1000, 1000), generator.uniform(-1000, 1000)) for _ in range(80)]
    first = optimize_route(range(len(points)), lambda index: points[index], (50.0, -25.0))
    second = optimize_route(range(len(points)), lambda index: points[index], (50.0, -25.0))
    assert first.exact is False
    assert first.indices == second.indices
    assert first.optimized_distance_um <= first.baseline_distance_um + 1e-9
    assert first.to_audit_dict()["savedDistanceUm"] >= 0
    assert set(first.indices) == set(range(len(points)))


def test_invalid_start_or_particle_coordinate_is_rejected():
    with pytest.raises(ValueError, match="起点"):
        optimize_route([0], lambda _item: (1.0, 2.0), (float("nan"), 0.0))
    with pytest.raises(ValueError, match="无效台坐标"):
        optimize_route([0], lambda _item: (float("inf"), 2.0), (0.0, 0.0))
