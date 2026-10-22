from pathlib import Path

import numpy as np
import pytest

from backend.cleanliness_core.cleanliness_task import _move_stage_checked
from backend.cleanliness_core import hole_center_locator


def test_fragmented_long_arc_passes_when_geometric_fit_is_strong():
    config = hole_center_locator.HoleLocatorConfig()
    detection = hole_center_locator.HoleDetection(
        status="arc",
        radius_px=366.9,
        coverage_deg=140.0,
        span_deg=325.0,
        density=0.43,
        rms_px=1.15,
        inlier_count=42,
    )

    assert hole_center_locator._arc_quality_ok(detection, config)


@pytest.mark.parametrize(
    ("radius_px", "coverage_deg", "density", "rms_px", "inlier_count"),
    [
        (199.0, 140.0, 0.43, 1.15, 42),
        (366.9, 115.0, 0.43, 1.15, 42),
        (366.9, 140.0, 0.30, 1.15, 42),
        (366.9, 140.0, 0.43, 1.60, 42),
        (366.9, 140.0, 0.43, 1.15, 34),
    ],
)
def test_fragmented_arc_still_rejects_weak_or_noisy_fits(
    radius_px: float,
    coverage_deg: float,
    density: float,
    rms_px: float,
    inlier_count: int,
):
    config = hole_center_locator.HoleLocatorConfig()
    detection = hole_center_locator.HoleDetection(
        status="arc",
        radius_px=radius_px,
        coverage_deg=coverage_deg,
        density=density,
        rms_px=rms_px,
        inlier_count=inlier_count,
    )

    assert not hole_center_locator._arc_quality_ok(detection, config)


class _MisalignedMicroscope:
    def __init__(self, error_um: float = 80.304):
        self.error_um = float(error_um)
        self.position = (0.0, 0.0)

    def move_xy(self, x: float, y: float) -> dict:
        before = self.position
        target = (float(x), float(y))
        self.position = (target[0] + self.error_um, target[1])
        return {
            "before": {"x": before[0], "y": before[1]},
            "target": {"x": target[0], "y": target[1]},
            "after": {"x": self.position[0], "y": self.position[1]},
            "positionErrorUm": self.error_um,
        }

    def capture(self, raw_path: Path):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        return image, {"pixelSizeUm": 1.0, "displayPath": str(raw_path)}


def test_move_stage_checked_remains_strict_by_default():
    microscope = _MisalignedMicroscope()

    with pytest.raises(RuntimeError, match="80.304"):
        _move_stage_checked(microscope, (10.0, 20.0), maximum_error_um=2.0)


def test_hole_locator_warns_and_uses_actual_stage_position(monkeypatch, tmp_path):
    microscope = _MisalignedMicroscope()
    messages: list[str] = []
    detection = hole_center_locator.HoleDetection(
        status="full",
        center_px=(50.0, 50.0),
        radius_px=10.0,
        diameter_um=20.0,
        coverage_deg=360.0,
        rms_px=0.5,
        sigma_px=0.2,
    )
    monkeypatch.setattr(hole_center_locator, "detect_hole", lambda *_args, **_kwargs: detection)
    monkeypatch.setattr(
        hole_center_locator,
        "annotate_hole_detection",
        lambda image, _detection: image,
    )

    result = hole_center_locator.locate_hole_center(
        microscope,
        (100.0, 200.0),
        tmp_path,
        config=hole_center_locator.HoleLocatorConfig(approach_offset_um=(0.0, 0.0)),
        stage_error_um=2.0,
        label="A",
        progress=messages.append,
    )

    assert result["ok"] is True
    assert result["holeStage"] == pytest.approx({"x": 180.304, "y": 200.0})
    assert result["warnings"][0]["positionErrorUm"] == pytest.approx(80.304)
    assert result["captures"][0]["warning"].endswith("80.304 μm")
    assert any("警告" in message and "按实际回读坐标" in message for message in messages)
