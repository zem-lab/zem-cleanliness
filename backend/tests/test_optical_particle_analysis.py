import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.cleanliness_core.optical_particle_analysis import (
    annotate_sem_match,
    decide_particle_match,
    detect_particle_candidates,
    deterministic_fake_elements,
    load_calibration,
    load_optical_export,
)
from backend.cleanliness_core.optical_particle_task import (
    TaskConfig,
    default_calibration_path,
    run_optical_particle_task,
)


PIXEL_SIZE_UM = 2.0670572916666665


def _encoded_png() -> str:
    image = np.full((40, 70, 3), 180, dtype=np.uint8)
    cv2.rectangle(image, (12, 10), (58, 30), (30, 30, 30), -1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _write_export(path: Path) -> Path:
    length_um = 58 * PIXEL_SIZE_UM
    width_um = 39 * PIXEL_SIZE_UM
    area_um2 = 58 * 39 * PIXEL_SIZE_UM**2
    payload = {
        "projectId": "example-project-id",
        "workpieceId": "Test",
        "workpieceName": "Test",
        "coordinateUnit": "um",
        "coordinateSpace": "sourceStage",
        "particles": [
            {
                "sequence": 3,
                "particleId": "particle-three",
                "label": "metal",
                "particleInfo": {
                    "isValid": True,
                    "length": length_um,
                    "width": width_um,
                    "area": area_um2,
                    "equivalentDiameter": 100.0,
                },
                "imageDataBase64": _encoded_png(),
                "imageWidth": 70,
                "imageHeight": 40,
                "abPosition": {
                    "x": 14003.192318732523,
                    "y": 13775.332732850906,
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _synthetic_sem() -> np.ndarray:
    rng = np.random.default_rng(7)
    image = rng.normal(42, 3, size=(512, 512)).clip(0, 255).astype(np.uint8)
    cv2.rectangle(image, (232, 236), (289, 274), 235, -1)
    return image


def test_load_export_and_calibration_transform(tmp_path):
    export = load_optical_export(_write_export(tmp_path / "optical.json"))
    calibration = load_calibration(default_calibration_path())

    calibration.validate_export(export)
    target = calibration.target_stage(export.particles[0])

    assert len(export.particles) == 1
    assert target == pytest.approx((-325.1008832546322, -21858.45930135609))
    assert calibration.target_inside_roi(target)


def test_detect_dark_low_contrast_particle():
    rng = np.random.default_rng(11)
    image = rng.normal(120, 9, size=(512, 512)).clip(0, 255).astype(np.uint8)
    dark = rng.normal(106, 4, size=(80, 110)).clip(0, 255).astype(np.uint8)
    image[215:295, 200:310] = dark

    candidates = detect_particle_candidates(
        image,
        expected_length_px=110.0,
        expected_width_px=80.0,
        expected_area_px=110.0 * 80.0,
    )
    decision = decide_particle_match(candidates)

    assert decision.matched
    assert decision.best is not None
    assert decision.best.channel in {"dark", "flat"}
    assert decision.best.center == pytest.approx((255.0, 255.0), abs=12.0)


def test_detect_decide_annotate_and_fake_elements(tmp_path):
    export = load_optical_export(_write_export(tmp_path / "optical.json"))
    particle = export.particles[0]
    image = _synthetic_sem()

    candidates = detect_particle_candidates(
        image,
        expected_length_px=particle.length_um / PIXEL_SIZE_UM,
        expected_width_px=particle.width_um / PIXEL_SIZE_UM,
        expected_area_px=particle.area_um2 / PIXEL_SIZE_UM**2,
    )
    decision = decide_particle_match(candidates)

    assert candidates
    assert decision.matched
    assert decision.best is not None
    assert decision.best.center == pytest.approx((260.5, 255.0), abs=3.0)

    outputs = annotate_sem_match(
        image,
        particle,
        decision,
        tmp_path / "annotated.png",
        tmp_path / "particle.png",
    )
    assert Path(outputs["annotatedImage"]).is_file()
    assert Path(outputs["particleImage"]).is_file()
    assert outputs["safePointPx"] is not None
    annotated = cv2.imread(str(outputs["annotatedImage"]))
    particle_crop = cv2.imread(str(outputs["particleImage"]))
    assert annotated is not None and particle_crop is not None
    # 标注图与颗粒裁剪图都应保留彩色叠加（轮廓/EDS），不能是纯灰度拷贝。
    assert annotated.ndim == 3 and not np.allclose(annotated[:, :, 0], annotated[:, :, 1])
    assert particle_crop.ndim == 3 and not np.allclose(
        particle_crop[:, :, 0], particle_crop[:, :, 1]
    )

    first = deterministic_fake_elements(particle)
    second = deterministic_fake_elements(particle)
    assert first == second
    assert first["mode"] == "SIMULATED"
    assert sum(item["value"] for item in first["elements"]) == pytest.approx(100.0)


class _FakeMicroscope:
    host = "192.168.0.15"
    port = 61557

    def __init__(self):
        self.position = (100.0, 200.0)
        self.restored_video = False

    def connect(self):
        return {
            "status": "Idle",
            "position": {"x": self.position[0], "y": self.position[1]},
            "magnification": 120,
            "fieldSizeUm": 1058.3333333333333,
        }

    def get_position(self):
        return self.position

    def move_xy(self, x, y):
        before = self.position
        self.position = (float(x), float(y))
        return {
            "before": {"x": before[0], "y": before[1]},
            "target": {"x": float(x), "y": float(y)},
            "after": {"x": float(x), "y": float(y)},
            "positionErrorUm": 0.0,
        }

    def capture(self, raw_path):
        image = _synthetic_sem()
        raw_path = Path(raw_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(raw_path), image)
        display_path = raw_path.with_name(f"{raw_path.stem}.u8.png")
        assert cv2.imwrite(str(display_path), image)
        return image, {
            "rawPath": str(raw_path.resolve()),
            "displayPath": str(display_path.resolve()),
            "shape": [512, 512],
            "pixelSizeUm": PIXEL_SIZE_UM,
            "bounds": None,
            "meta": {
                "Detector": "BSE",
                "Magnification": 120,
                "FieldSize": 1058.3333333333333,
                "FieldRotation": 0,
                "KV": 20,
            },
        }

    def restore_video(self):
        self.restored_video = True


def test_full_task_generates_dual_image_pdf_and_json(tmp_path):
    input_path = _write_export(tmp_path / "optical.json")
    task_dir = tmp_path / "task"
    microscope = _FakeMicroscope()
    config = TaskConfig(
        input_path=input_path,
        output_dir=task_dir,
        calibration_path=default_calibration_path(),
        sequences=(3,),
        sample_count=1,
        max_capture_attempts=1,
    )

    manifest = run_optical_particle_task(config, microscope=microscope)

    assert manifest["status"] == "COMPLETED"
    assert manifest["summary"]["succeeded"] == 1
    assert manifest["particles"][0]["status"] == "SUCCEEDED"
    assert manifest["particles"][0]["energySpectrum"]["mode"] == "SIMULATED"
    assert (task_dir / "optical" / "0003.png").is_file()
    assert (task_dir / "particles" / "0003.png").is_file()
    assert (task_dir / "annotated" / "0003.png").is_file()
    assert (task_dir / "report.pdf").stat().st_size > 1000
    assert (task_dir / "report.html").is_file()
    assert (task_dir / "task.json").is_file()
    assert microscope.position == pytest.approx((100.0, 200.0))
    assert microscope.restored_video
