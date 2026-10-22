import base64
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.cleanliness_core.ab_point_calibration import (
    DetectionBox,
    NavigationABDetection,
)
from backend.cleanliness_core.cleanliness_detection import ParticleFilter
from backend.cleanliness_core.cleanliness_task import (
    CleanlinessTaskConfig,
    abandon_ab_point_calibration,
    review_cleanliness_calibration,
    run_cleanliness_analysis,
    run_cleanliness_calibration,
    save_ab_point_calibration,
)


PIXEL_SIZE_UM = 2.0


def _encoded_particle(width_px: int, height_px: int) -> str:
    image = np.full((48, 72, 3), 220, dtype=np.uint8)
    cv2.ellipse(
        image,
        (36, 24),
        (max(4, width_px // 2), max(3, height_px // 2)),
        12,
        0,
        360,
        (24, 28, 32),
        -1,
    )
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _write_export(path: Path) -> Path:
    coordinates = ((0.0, 0.0), (5000.0, 200.0), (900.0, 4200.0), (5200.0, 4500.0), (2700.0, 2600.0))
    particles = []
    for index, (x, y) in enumerate(coordinates, start=1):
        length_px = 32 + index
        width_px = 17 + index
        particles.append(
            {
                "sequence": index,
                "particleId": f"particle-{index}",
                "label": "metal",
                "particleInfo": {
                    "isValid": True,
                    "length": length_px * PIXEL_SIZE_UM,
                    "width": width_px * PIXEL_SIZE_UM,
                    "area": np.pi * (length_px / 2) * (width_px / 2) * PIXEL_SIZE_UM**2,
                    "equivalentDiameter": 45.0 + index,
                },
                "imageDataBase64": _encoded_particle(length_px, width_px),
                "imageWidth": 72,
                "imageHeight": 48,
                "abPosition": {"x": x, "y": y},
            }
        )
    path.write_text(
        json.dumps(
            {
                "projectId": "project",
                "workpieceId": "sample",
                "workpieceName": "sample",
                "coordinateUnit": "um",
                "coordinateSpace": "sourceStage",
                "registrationMode": "twoPointAB",
                "referencePoints": {
                    "a": {"x": 0.0, "y": 0.0},
                    "b": {"x": 5000.0, "y": 0.0},
                },
                "particles": particles,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_experience(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "calibrationId": "test-experience",
                "device": {"host": "fake", "port": 61557, "model": "Fake"},
                "source": {
                    "coordinateUnit": "um",
                    "coordinateSpace": "sourceStage",
                    "workpieceId": "sample",
                    "projectId": "project",
                    "requireExactInput": False,
                },
                "transform": {
                    "model": "2D reflected similarity",
                    "matrix": [[1.0, 0.0], [0.0, -1.0]],
                    "translation": [0.0, 0.0],
                },
                "validStageRoi": {"centerUm": [0.0, 0.0], "radiusUm": 100000.0},
                "imaging": {
                    "magnification": 120,
                    "fieldSizeUm": 1024.0,
                    "imageShape": [512, 512],
                    "pixelSizeUm": PIXEL_SIZE_UM,
                    "detector": "BSE",
                    "fieldRotationDeg": 0,
                },
                "audit": {},
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_settings(path: Path, experience: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "device": {
                    "microscopeHost": "fake",
                    "microscopePort": 61557,
                    "edsHost": "fake",
                    "edsPort": 6200,
                },
                "experienceCalibration": str(experience),
                "calibration": {
                    "model": "reflected_similarity",
                    "candidateSegmentation": {"backend": "classic"},
                    "anchorCount": 3,
                    "validationCount": 1,
                    "anchorPoolSize": 5,
                    "semCandidatesPerAnchor": 4,
                    "pixelAxisMatrix": [[1.0, 0.0], [0.0, 1.0]],
                    "acceptance": {"meanErrorUm": 10.0, "maxErrorUm": 15.0},
                },
                "task": {
                    "restoreStartPosition": True,
                    "gunOnAtStart": False,
                    "gunOffAfterTask": True,
                    "edsEnabled": True,
                    "edsTimeoutSeconds": 10,
                },
                "report": {"elementColumns": 5, "excludeElements": ["O", "C"]},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_ab_point_strategy_saves_pending_result_without_device_actions(tmp_path):
    export_path = _write_export(tmp_path / "optical.json")
    experience_path = _write_experience(tmp_path / "experience.json")
    settings_path = _write_settings(tmp_path / "settings.json", experience_path)
    task_dir = tmp_path / "ab-task"
    detection = NavigationABDetection(
        image_shape=(100, 200),
        fixture=DetectionBox(
            class_id=0,
            confidence=0.98,
            box_xyxy=(5.0, 5.0, 195.0, 95.0),
        ),
        crop_box_xyxy=(0, 0, 200, 100),
        holes=(
            DetectionBox(
                class_id=1,
                confidence=0.91,
                box_xyxy=(10.0, 10.0, 30.0, 30.0),
            ),
            DetectionBox(
                class_id=1,
                confidence=0.89,
                box_xyxy=(170.0, 70.0, 190.0, 90.0),
            ),
        ),
        membranes=(),
        fixture_model={"name": "fixture.onnx", "sha256": "fixture"},
        hole_model={"name": "hole.onnx", "sha256": "hole"},
    )
    stale_config = CleanlinessTaskConfig(
        input_path=export_path,
        output_dir=tmp_path / "stale-ab-task",
        settings_path=settings_path,
        particle_filter=ParticleFilter(),
        calibration_strategy="ab_points",
        expected_input_sha256="stale",
    )
    with pytest.raises(ValueError, match="内容已变化"):
        save_ab_point_calibration(
            stale_config,
            navigation_image=np.zeros((100, 200, 3), dtype=np.uint8),
            navigation_bounds=(0.0, 0.0, 1000.0, 500.0),
            detection=detection,
            a_index=0,
            b_index=1,
            handedness="right",
        )

    config = CleanlinessTaskConfig(
        input_path=export_path,
        output_dir=task_dir,
        settings_path=settings_path,
        particle_filter=ParticleFilter(),
        calibration_strategy="ab_points",
    )

    manifest = save_ab_point_calibration(
        config,
        navigation_image=np.zeros((100, 200, 3), dtype=np.uint8),
        navigation_bounds=(0.0, 0.0, 1000.0, 500.0),
        detection=detection,
        a_index=0,
        b_index=1,
        handedness="right",
    )

    assert manifest["status"] == "AB_POINTS_PENDING_SEM_REFINEMENT"
    assert manifest["calibration"]["strategy"] == "ab_points"
    assert manifest["calibration"]["previewTransform"]["verified"] is False
    assert manifest["cleanup"] == {
        "deviceActionsPerformed": False,
        "stageMoved": False,
        "gunStateChanged": False,
    }
    assignment = manifest["calibration"]["abPointResult"]["assignment"]
    assert assignment["a"]["pixel"] == pytest.approx((20.0, 20.0))
    assert assignment["a"]["coarseStageUm"] == pytest.approx((900.0, 100.0))
    assert assignment["b"]["coarseStageUm"] == pytest.approx((100.0, 400.0))
    assert (task_dir / "task.json").is_file()
    assert (task_dir / "ab_point_result.json").is_file()
    assert (task_dir / "navigation" / "navigation.png").is_file()
    assert (
        task_dir / "navigation" / "navigation_ab_annotated.png"
    ).is_file()
    abandoned = abandon_ab_point_calibration(
        task_dir,
        reviewer="test",
    )
    assert abandoned["status"] == "AB_POINTS_ABANDONED"
    assert abandoned["calibration"]["review"]["accepted"] is False


class _FakeMicroscope:
    host = "fake"
    port = 61557

    def __init__(self, particles):
        self.particles = particles
        self.position = (100.0, 100.0)
        self.start_position = self.position
        self.gun_is_on = True
        self.video_restored = False
        self.move_targets = []
        self.capture_failures_remaining = 0
        self.true_matrix = np.asarray([[1.0, 0.0], [0.0, -1.0]])
        self.true_translation = np.asarray([36.0, -24.0])

    def _experience_target(self, particle):
        return np.asarray([particle["x"], -particle["y"]])

    def _true_target(self, particle):
        source = np.asarray([particle["x"], particle["y"]])
        return self.true_matrix @ source + self.true_translation

    def connect(self):
        return {"status": "Idle", "position": self.position, "magnification": 120}

    def get_position(self):
        return self.position

    def move_xy(self, x, y):
        before = self.position
        self.position = (float(x), float(y))
        self.move_targets.append(self.position)
        return {
            "before": {"x": before[0], "y": before[1]},
            "target": {"x": float(x), "y": float(y)},
            "after": {"x": float(x), "y": float(y)},
            "positionErrorUm": 0.0,
        }

    def capture(self, raw_path):
        if self.capture_failures_remaining > 0:
            self.capture_failures_remaining -= 1
            raise RuntimeError("simulated capture failure")
        current = np.asarray(self.position)
        particle = min(
            self.particles,
            key=lambda item: min(
                np.linalg.norm(self._experience_target(item) - current),
                np.linalg.norm(self._true_target(item) - current),
            ),
        )
        true_target = self._true_target(particle)
        offset = (true_target - current) / PIXEL_SIZE_UM
        center = np.asarray([256.0, 256.0]) + offset
        image = np.full((512, 512), 45, dtype=np.uint8)
        axes = (int(round(particle["length_px"] / 2)), int(round(particle["width_px"] / 2)))
        cv2.ellipse(
            image,
            (int(round(center[0])), int(round(center[1]))),
            axes,
            12,
            0,
            360,
            235,
            -1,
        )
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
                "FieldSize": 1024.0,
                "FieldRotation": 0,
                "KV": 20,
            },
        }

    def restore_video(self):
        self.video_restored = True

    def gun_on(self):
        self.gun_is_on = True
        return "EnumHvStatus.Open"

    def gun_off(self):
        self.gun_is_on = False
        return "EnumHvStatus.Close"


class _FakeSpectrumAnalyzer:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.ready_checked = False

    def check_ready(self):
        self.ready_checked = True
        return {"status": "READY", "device": "fake:6200", "protocol": "grpc_v2"}

    def analyze(self, particle, candidate, image_path, safe_point):
        self.calls.append((particle.sequence, safe_point))
        return {
            "device": "fake:6200",
            "target": {"mode": "particle_safe_interior_point", "x": safe_point[0], "y": safe_point[1]},
            "elements": [
                {"symbol": "Fe", "value": 72.0},
                {"symbol": "Cr", "value": 18.0},
                {"symbol": "Ni", "value": 10.0},
            ],
        }

    def close(self):
        self.closed = True


class _SequencedSpectrumAnalyzer(_FakeSpectrumAnalyzer):
    def __init__(self, errors):
        super().__init__()
        self.errors = list(errors)

    def analyze(self, particle, candidate, image_path, safe_point):
        if self.errors:
            self.calls.append((particle.sequence, safe_point))
            raise RuntimeError(self.errors.pop(0))
        return super().analyze(particle, candidate, image_path, safe_point)


@pytest.mark.parametrize("reject_mask_identity", [False, True])
def test_full_cleanliness_flow_recalibrates_validates_eds_reports_and_closes_gun(tmp_path, monkeypatch, reject_mask_identity):
    input_path = _write_export(tmp_path / "optical.json")
    experience = _write_experience(tmp_path / "experience.json")
    settings = _write_settings(tmp_path / "settings.json", experience)
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    particles = [
        {
            "sequence": item["sequence"],
            "x": item["abPosition"]["x"],
            "y": item["abPosition"]["y"],
            "length_px": item["particleInfo"]["length"] / PIXEL_SIZE_UM,
            "width_px": item["particleInfo"]["width"] / PIXEL_SIZE_UM,
        }
        for item in raw["particles"]
    ]
    microscope = _FakeMicroscope(particles)
    task_dir = tmp_path / "task"
    config = CleanlinessTaskConfig(
        input_path=input_path,
        output_dir=task_dir,
        settings_path=settings,
        particle_filter=ParticleFilter(min_length_um=68.0, max_particles=2),
        anchor_sequences=(1, 2, 4),
        validation_sequences=(3,),
        auto_gun_on=False,
        gun_off_after_task=True,
    )

    calibration_events = []
    calibration = run_cleanliness_calibration(
        config,
        microscope=microscope,
        event_logger=lambda event, fields: calibration_events.append((event, fields)),
    )

    assert calibration["status"] == "CALIBRATION_PENDING_REVIEW"
    route = calibration["calibration"]["executionRoute"]
    assert route["algorithm"] == "held_karp_exact"
    assert route["exact"] is True
    assert route["optimizedDistanceUm"] <= route["baselineDistanceUm"] + 1e-9
    assert set(calibration["calibration"]["executionOrder"]) == {1, 2, 3, 4}
    for stage in ("calibration", "connect", "route_planning", "particle_acquisition", "stage_move", "capture", "candidate_detection", "matrix_fit"):
        stage_events = [item for item in calibration_events if item[1]["stage"] == stage]
        assert stage_events, stage
        assert stage_events[0][0] == "calibration_stage_started"
        assert stage_events[-1][0] == "calibration_stage_finished"
        assert all(item[1].get("elapsedSeconds", 0) >= 0 for item in stage_events if item[0].endswith("finished"))
    groups = calibration["calibration"]["groups"]
    assert len(groups) == 1
    result = groups[0]["result"]
    assert np.asarray(result["matrix"]) == pytest.approx(
        np.asarray([[1.0, 0.0], [0.0, -1.0]]), abs=0.08
    )
    assert np.asarray(result["translation"]) == pytest.approx(
        np.asarray([36.0, -24.0]), abs=5.0
    )
    assert result["validation"]["maxErrorUm"] < 8.0
    assert result["acceptance"]["withinLimits"] is True
    review_cleanliness_calibration(task_dir, accepted=True, selected_group=0, reviewer="pytest")

    # Simulate the AB-point path, which previously created only its validation
    # optical images. Analysis must repair the full selected set before reports.
    for optical_path in (task_dir / "optical").glob("*.png"):
        optical_path.unlink()

    analyzer = _FakeSpectrumAnalyzer()
    if reject_mask_identity:
        from dataclasses import replace
        def reject_identity(image, particle, pixel_size_um, candidates, **kwargs):
            from backend.cleanliness_core.cleanliness_task import _decision_for_candidate
            decision = _decision_for_candidate(candidates[0],len(candidates),reason_prefix="test")
            return candidates[0], replace(decision, status="FAILED_MASK_IDENTITY", confidence="low",
                           reason="ambiguous mask identity"), {"status":"AMBIGUOUS", "accepted":False}
        monkeypatch.setattr("backend.cleanliness_core.cleanliness_task._select_mask_guided_candidate", reject_identity)
    manifest = run_cleanliness_analysis(
        task_dir,
        config=config,
        microscope=microscope,
        spectrum_analyzer=analyzer,
    )

    if reject_mask_identity:
        assert manifest["status"] == "COMPLETED_WITH_FAILURES"
        assert len(manifest["particles"]) == 2
        assert not analyzer.calls
        assert all(p["status"] == "FAILED" and not p["energySpectrum"] for p in manifest["particles"])
        assert all("ambiguous mask identity" in p["error"] for p in manifest["particles"])
        assert microscope.gun_is_on is False
        return
    assert manifest["status"] == "COMPLETED"
    assert len(manifest["particles"]) == 2
    assert len(analyzer.calls) == 2
    assert analyzer.ready_checked
    assert all(call[1] is not None for call in analyzer.calls)
    assert analyzer.closed
    assert microscope.position == pytest.approx(microscope.start_position)
    assert microscope.gun_is_on is False
    assert "Close" in manifest["cleanup"]["gunOffStatus"]
    assert (task_dir / "cleanliness_report.pdf").stat().st_size > 1000
    report_html = (task_dir / "cleanliness_report.html").read_text(encoding="utf-8")
    assert "光镜颗粒" in report_html
    assert "电镜图像" in report_html
    assert "识别轮廓 + EDS 打点" in report_html
    assert "analysis_annotated" in report_html
    assert "电镜标注图图例" in report_html
    assert "EDS 电子束打点位置" in report_html
    assert "谱峰" not in report_html
    for record in manifest["particles"]:
        optical = task_dir / str((record.get("images") or {}).get("opticalImage") or "")
        assert optical.is_file()
        assert cv2.imread(str(optical)) is not None
        annotated = task_dir / str((record.get("images") or {}).get("semAnnotatedImage") or "")
        assert annotated.is_file()
        image = cv2.imread(str(annotated))
        assert image is not None
        assert not np.allclose(image[:, :, 0], image[:, :, 1])


def _calibrated_task(tmp_path, *, max_particles=1):
    input_path = _write_export(tmp_path / "optical.json")
    experience = _write_experience(tmp_path / "experience.json")
    settings = _write_settings(tmp_path / "settings.json", experience)
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    particles = [
        {
            "sequence": item["sequence"],
            "x": item["abPosition"]["x"],
            "y": item["abPosition"]["y"],
            "length_px": item["particleInfo"]["length"] / PIXEL_SIZE_UM,
            "width_px": item["particleInfo"]["width"] / PIXEL_SIZE_UM,
        }
        for item in raw["particles"]
    ]
    microscope = _FakeMicroscope(particles)
    task_dir = tmp_path / "task"
    config = CleanlinessTaskConfig(
        input_path=input_path,
        output_dir=task_dir,
        settings_path=settings,
        particle_filter=ParticleFilter(max_particles=max_particles),
        anchor_sequences=(1, 2, 4),
        validation_sequences=(3,),
        auto_gun_on=False,
        gun_off_after_task=True,
    )
    calibration = run_cleanliness_calibration(config, microscope=microscope)
    assert calibration["status"] == "CALIBRATION_PENDING_REVIEW"
    review_cleanliness_calibration(task_dir, accepted=True, reviewer="pytest")
    return task_dir, config, microscope


def test_analysis_uses_frozen_candidate_margin_and_rejection_never_enters_eds(tmp_path, monkeypatch):
    from backend.cleanliness_core import cleanliness_task as engine
    task_dir, config, microscope = _calibrated_task(tmp_path)
    settings = json.loads(config.settings_path.read_text(encoding="utf-8"))
    settings.setdefault("abParticle", {})["match_min_margin"] = .12
    settings["abParticle"]["match_min_score"] = .5
    settings.setdefault("task", {})["contourOverlapMin"] = .4
    config.settings_path.write_text(json.dumps(settings), encoding="utf-8")
    config = replace(config, max_capture_attempts=1)
    analyzer = _FakeSpectrumAnalyzer()
    seen = []

    def reject_with_config(_image, _particle, _pixel, candidates, **kwargs):
        seen.append((
            kwargs["refinement_settings"].match_min_margin,
            kwargs["refinement_settings"].match_min_score,
            kwargs["minimum_mask_iou"],
        ))
        chosen = candidates[0]
        decision = engine._decision_for_candidate(
            chosen, len(candidates), reason_prefix="configured margin test"
        )
        return chosen, replace(
            decision,
            status="FAILED_MASK_IDENTITY",
            confidence="low",
            reason="候选分差低于已冻结阈值",
        ), {"status": "AMBIGUOUS", "accepted": False, "minimumCandidateScoreMargin": .12}

    monkeypatch.setattr(engine, "_select_mask_guided_candidate", reject_with_config)
    result = run_cleanliness_analysis(
        task_dir,
        config=config,
        microscope=microscope,
        spectrum_analyzer=analyzer,
    )

    assert result["candidateScoreMargin"] == .12
    assert result["templateMatchMinScore"] == .5
    assert result["contourOverlapMin"] == .4
    assert result["maxCaptureAttempts"] == 1
    assert seen == [(.12, .5, .4)] * config.max_capture_attempts
    assert result["particles"][0]["status"] == "FAILED"
    assert len(result["particles"][0]["captureAttempts"]) == config.max_capture_attempts
    assert not analyzer.calls


def test_contour_overlap_threshold_uses_configured_floor_and_defaults_to_half(monkeypatch):
    from backend.cleanliness_core import cleanliness_task as engine
    from backend.cleanliness_core.cleanliness_detection import SemAnchorCandidate
    from backend.cleanliness_core.optical_particle_analysis import OpticalParticle, ParticleCandidate

    mask = np.zeros((40, 40), np.uint8)
    mask[8:32, 8:32] = 255
    encoded_mask = base64.b64encode(cv2.imencode(".png", mask)[1]).decode()
    encoded_photo = base64.b64encode(cv2.imencode(".png", np.full((40, 40, 3), 200, np.uint8))[1]).decode()
    particle = OpticalParticle(
        1, "1", "metal", 0.0, 0.0, 20.0, 20.0, 576.0, 16.0, encoded_photo, 40, 40,
        mask_data_base64=encoded_mask, mask_width=40, mask_height=40,
    )
    sem = np.full((220, 220), 30, np.uint8)

    def fake_match(*_args, **_kwargs):
        return {
            "status": "MATCH", "score": 0.9, "centerPixel": [110.0, 110.0],
            "angleDeg": 0.0, "mirrored": False, "scaleFactor": 1.0, "margin": 0.2, "reason": "",
        }

    monkeypatch.setattr(engine, "build_sem_anchor_candidates", lambda *_args, **_kwargs: ())
    monkeypatch.setattr("backend.cleanliness_core.particle_refinement.match_particle", fake_match)
    optical = (mask > 0).astype(np.float32)
    height, width = optical.shape
    span = int(np.ceil(np.hypot(height, width))) + 20
    warp = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), 0, 1)
    warp[:, 2] += np.array([(span - width) / 2, (span - height) / 2])
    warp[:, 2] += np.array([110.0, 110.0]) - (span - 1) / 2
    reference = cv2.warpAffine(optical, warp, (sem.shape[1], sem.shape[0])) > 0.5
    shifted = np.zeros(reference.shape, dtype=bool)
    shifted[:, 10:] = reference[:, :-10]
    contours, _hierarchy = cv2.findContours(shifted.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = max(contours, key=cv2.contourArea)
    contour = tuple((int(point[0][0]), int(point[0][1])) for point in outline)
    box = cv2.boundingRect(outline)
    moments = cv2.moments(outline)
    center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    item = ParticleCandidate(box, center, contour, 0.0, 1.0, 1.0, 1.0, 3.0, 3.0, "gray", 1.0, 1.0, 1)
    candidate = SemAnchorCandidate(1, (0.0, 0.0), center, 1.0, 0.1, item, "customer_mask")

    def select(floor):
        return engine._select_mask_guided_candidate(
            sem, particle, 1.0, (candidate,), stage_center_um=(0.0, 0.0),
            **({} if floor is None else {"minimum_mask_iou": floor}),
        )

    _chosen, default_decision, default_audit = select(None)
    _chosen, loose_decision, loose_audit = select(0.01)
    _chosen, strict_decision, strict_audit = select(0.99)
    _chosen, invalid_decision, invalid_audit = select(1)
    assert default_audit["minimumMaskIoU"] == 0.5
    assert loose_audit["maskIoU"] == pytest.approx(default_audit["maskIoU"])
    assert 0.01 < default_audit["maskIoU"] < 0.99
    assert default_decision.matched == (default_audit["maskIoU"] >= 0.5)
    assert loose_decision.matched and loose_audit["accepted"]
    assert not strict_decision.matched and not strict_audit["accepted"]
    assert not invalid_decision.matched and invalid_audit is None


@pytest.mark.parametrize("shape_corroborated", [False, True])
def test_shape_review_never_bypasses_final_contour_or_eds_gate(tmp_path, monkeypatch, shape_corroborated):
    """Exercise the actual new branch and task loop with fake device services."""
    from dataclasses import replace
    from backend.cleanliness_core import cleanliness_task as engine
    task_dir, config, microscope = _calibrated_task(tmp_path)
    analyzer = _FakeSpectrumAnalyzer()
    original_selector = engine._select_mask_guided_candidate
    original_builder = engine.build_sem_anchor_candidates
    mask = np.zeros((48,72),np.uint8)
    cv2.ellipse(mask,(36,24),(18,10),0,0,360,255,-1)
    encoded=base64.b64encode(cv2.imencode('.png',mask)[1]).decode()
    def with_customer_mask(image, particle, pixel, candidates, **kwargs):
        return original_selector(image,replace(particle,mask_data_base64=encoded,mask_width=72,mask_height=48),
                                 pixel,candidates,**kwargs)
    monkeypatch.setattr(engine,'_select_mask_guided_candidate',with_customer_mask)
    monkeypatch.setattr('backend.cleanliness_core.particle_refinement.match_particle',lambda *a,**k:
        {'status':'AMBIGUOUS','score':.95,'margin':.02,'centerPixel':[256.,256.],
         'angleDeg':0,'mirrored':False,'scaleFactor':1.})
    reviews=[]
    def reviewed(*args,**kwargs):
        reviews.append(True)
        return {'status':'CORROBORATED' if shape_corroborated else 'UNVERIFIED',
                'corroborated':shape_corroborated,'allowsEds':False,'reason':'controlled cross-check',
                'targets':[{'yoloCenterPx':[256.,256.]}]}
    monkeypatch.setattr('backend.cleanliness_core.shape_consensus.review_ambiguous_shapes',reviewed)
    def missing_local_contour(particle,image,**kwargs):
        if kwargs.get('maximum_candidates')==64:
            return ()
        pool=original_builder(particle,image,**kwargs)
        # A plausible candidate at another location must not become the final
        # contour merely because the preceding shape review succeeded.
        return tuple(replace(c,pixel_xy=(30.,30.),candidate=replace(c.candidate,
            center=(30.,30.),box=(20,20,40,40),contour=((20,20),(40,20),(40,40),(20,40)))) for c in pool)
    monkeypatch.setattr(engine,'build_sem_anchor_candidates',missing_local_contour)
    result=run_cleanliness_analysis(task_dir,config=config,microscope=microscope,spectrum_analyzer=analyzer)
    assert reviews
    assert result['status']=='COMPLETED_WITH_FAILURES'
    assert not analyzer.calls
    assert all(p['status']=='FAILED' and p['energySpectrum'] is None for p in result['particles'])
    for record in result['particles']:
        for attempt in record['captureAttempts']:
            audit=attempt.get('maskVerification')
            if audit:
                assert audit['accepted'] is False
                assert audit['shapeConsensus']['corroborated'] is shape_corroborated


def test_analysis_resume_preserves_completed_records_and_runs_only_remaining(tmp_path):
    task_dir, config, microscope = _calibrated_task(tmp_path, max_particles=2)
    first_analyzer = _FakeSpectrumAnalyzer()
    stop_requested = False

    def progress(index, _total, _message):
        nonlocal stop_requested
        if index >= 1:
            stop_requested = True

    interrupted = run_cleanliness_analysis(
        task_dir,
        config=config,
        microscope=microscope,
        spectrum_analyzer=first_analyzer,
        progress=progress,
        should_stop=lambda: stop_requested,
    )

    assert interrupted["status"] == "STOPPED"
    assert len(interrupted["particles"]) == 1
    preserved_record = json.loads(json.dumps(interrupted["particles"][0]))
    original_started_at = interrupted["analysisStartedAt"]
    manifest_path = task_dir / "task.json"
    failed_checkpoint = json.loads(manifest_path.read_text(encoding="utf-8"))
    failed_checkpoint["status"] = "FAILED"
    failed_checkpoint["error"] = "[WinError 5] task.json.tmp -> task.json"
    manifest_path.write_text(
        json.dumps(failed_checkpoint, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    second_analyzer = _FakeSpectrumAnalyzer()

    resumed = run_cleanliness_analysis(
        task_dir,
        config=config,
        microscope=microscope,
        spectrum_analyzer=second_analyzer,
        resume=True,
    )

    assert resumed["status"] == "COMPLETED"
    assert len(resumed["particles"]) == 2
    assert resumed["particles"][0] == preserved_record
    assert [record["rank"] for record in resumed["particles"]] == [1, 2]
    assert len({record["sequence"] for record in resumed["particles"]}) == 2
    assert len(first_analyzer.calls) == 1
    assert len(second_analyzer.calls) == 1
    assert second_analyzer.calls[0][0] != first_analyzer.calls[0][0]
    assert resumed["analysisStartedAt"] == original_started_at
    history = resumed["analysisResumeHistory"][-1]
    assert history["previousStatus"] == "FAILED"
    assert history["previousError"].startswith("[WinError 5]")
    assert history["processed"] == 1
    assert history["remaining"] == 1
    assert history["nextRank"] == 2
    assert (task_dir / history["archive"]).is_file()
    assert resumed["reports"]["pdf"] == "cleanliness_report.pdf"


def test_analysis_resume_rejects_changed_input_before_hardware_action(tmp_path):
    task_dir, config, microscope = _calibrated_task(tmp_path, max_particles=2)
    stop_requested = False

    def progress(index, _total, _message):
        nonlocal stop_requested
        if index >= 1:
            stop_requested = True

    run_cleanliness_analysis(
        task_dir,
        config=config,
        microscope=microscope,
        spectrum_analyzer=_FakeSpectrumAnalyzer(),
        progress=progress,
        should_stop=lambda: stop_requested,
    )
    input_path = Path(config.input_path)
    input_path.write_text(
        input_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    microscope.move_targets.clear()

    with pytest.raises(ValueError, match="光镜 JSON 内容已变化"):
        run_cleanliness_analysis(
            task_dir,
            config=config,
            microscope=microscope,
            spectrum_analyzer=_FakeSpectrumAnalyzer(),
            resume=True,
        )

    assert microscope.move_targets == []


def test_analysis_retries_capture_before_eds(tmp_path):
    task_dir, config, microscope = _calibrated_task(tmp_path)
    microscope.capture_failures_remaining = 1
    analyzer = _FakeSpectrumAnalyzer()

    manifest = run_cleanliness_analysis(
        task_dir,
        config=config,
        microscope=microscope,
        spectrum_analyzer=analyzer,
    )

    assert manifest["status"] == "COMPLETED"
    attempts = manifest["particles"][0]["captureAttempts"]
    assert [item["status"] for item in attempts[:2]] == ["FAILED", "MATCHED_HIGH"]
    assert attempts[0]["captureSeconds"] >= 0
    assert attempts[0]["matchSeconds"] is None
    assert attempts[1]["captureSeconds"] >= 0
    assert attempts[1]["matchSeconds"] >= 0
    timing = manifest["particles"][0]["timing"]
    assert timing["captureSeconds"] == pytest.approx(
        attempts[0]["captureSeconds"] + attempts[1]["captureSeconds"]
    )
    assert len(analyzer.calls) == 1
    assert "Close" in manifest["cleanup"]["gunOffStatus"]


def _assert_particle_timing(record, *, expect_eds: bool):
    timing = record["timing"]
    for key in ("captureSeconds", "matchSeconds", "edsSeconds", "totalSeconds"):
        assert isinstance(timing[key], (int, float))
        assert timing[key] >= 0
    assert timing["stageMoveSeconds"] is None or timing["stageMoveSeconds"] >= 0
    assert timing["totalSeconds"] >= timing["captureSeconds"]
    assert timing["totalSeconds"] >= timing["matchSeconds"]
    assert timing["totalSeconds"] >= timing["edsSeconds"]
    if expect_eds:
        assert timing["edsSeconds"] >= 0
        assert record["edsAttempts"]
        assert all("elapsedSeconds" in item for item in record["edsAttempts"])
    else:
        assert timing["edsSeconds"] == 0


def test_particle_stage_timing_recorded_on_success_and_match_failure(tmp_path, monkeypatch):
    task_dir, config, microscope = _calibrated_task(tmp_path)
    analyzer = _FakeSpectrumAnalyzer()
    success_events = []
    manifest = run_cleanliness_analysis(
        task_dir,
        config=config,
        microscope=microscope,
        spectrum_analyzer=analyzer,
        event_logger=lambda event, fields: success_events.append((event, fields)),
    )
    record = manifest["particles"][0]
    assert manifest["status"] == "COMPLETED"
    _assert_particle_timing(record, expect_eds=True)
    summary = manifest["timingSummary"]
    assert summary["particleCount"] == 1
    assert summary["edsSeconds"]["count"] == 1
    assert summary["totalSeconds"]["mean"] == pytest.approx(record["timing"]["totalSeconds"])
    for stage in ("particle", "stage_move", "capture", "match", "match_output", "eds"):
        stage_events = [item for item in success_events if item[1]["stage"] == stage]
        assert [item[0] for item in stage_events] == [
            "particle_stage_started",
            "particle_stage_finished",
        ]
        assert stage_events[-1][1]["elapsedSeconds"] >= 0

    fail_root = tmp_path / "fail"
    fail_root.mkdir()
    task_dir, config, microscope = _calibrated_task(fail_root)
    analyzer = _FakeSpectrumAnalyzer()

    def reject_identity(image, particle, pixel_size_um, candidates, **kwargs):
        from backend.cleanliness_core.cleanliness_task import _decision_for_candidate

        decision = _decision_for_candidate(
            candidates[0], len(candidates), reason_prefix="test"
        )
        return (
            candidates[0],
            replace(
                decision,
                status="FAILED_MASK_IDENTITY",
                confidence="low",
                reason="ambiguous mask identity",
            ),
            {"status": "AMBIGUOUS", "accepted": False},
        )

    monkeypatch.setattr(
        "backend.cleanliness_core.cleanliness_task._select_mask_guided_candidate",
        reject_identity,
    )
    failed_events = []
    failed = run_cleanliness_analysis(
        task_dir,
        config=config,
        microscope=microscope,
        spectrum_analyzer=analyzer,
        event_logger=lambda event, fields: failed_events.append((event, fields)),
    )
    assert failed["status"] == "COMPLETED_WITH_FAILURES"
    failed_record = failed["particles"][0]
    _assert_particle_timing(failed_record, expect_eds=False)
    assert failed_record["energySpectrum"] is None
    assert failed_record["timing"]["stageMoveSeconds"] is not None
    assert failed_record["timing"]["captureSeconds"] > 0 or failed_record["captureAttempts"]
    assert failed["timingSummary"]["edsSeconds"]["total"] == 0
    assert not [item for item in failed_events if item[1]["stage"] == "eds"]
    match_events = [item for item in failed_events if item[1]["stage"] == "match"]
    assert len(match_events) == 4
    assert [item[0] for item in match_events] == [
        "particle_stage_started",
        "particle_stage_finished",
        "particle_stage_started",
        "particle_stage_finished",
    ]
    assert all(item[1]["status"] == "rejected" for item in match_events[1::2])


def test_analysis_retries_one_transient_eds_failure(tmp_path):
    task_dir, config, microscope = _calibrated_task(tmp_path)
    analyzer = _SequencedSpectrumAnalyzer(
        ["Particle 1 EDS job failed: StopEdSpectrumAcquisition failed"]
    )

    manifest = run_cleanliness_analysis(
        task_dir,
        config=replace(config, eds_retry_count=1),
        microscope=microscope,
        spectrum_analyzer=analyzer,
    )

    record = manifest["particles"][0]
    assert manifest["status"] == "COMPLETED"
    assert len(analyzer.calls) == 2
    assert [item["status"] for item in record["edsAttempts"]] == [
        "FAILED",
        "SUCCEEDED",
    ]
    assert record["edsAttempts"][0]["transient"] is True
    assert all(item["elapsedSeconds"] >= 0 for item in record["edsAttempts"])
    assert record["timing"]["edsSeconds"] == pytest.approx(
        record["edsAttempts"][0]["elapsedSeconds"]
        + record["edsAttempts"][1]["elapsedSeconds"]
    )


@pytest.mark.parametrize('mode', ['complete','partial','stop','fatal','disabled','shutdown_error','shutdown_unconfirmed'])
def test_analysis_final_shutdown_order_and_failures(tmp_path, monkeypatch, mode):
    task_dir, config, microscope = _calibrated_task(tmp_path)
    microscope.gun_is_on = True
    analyzer = _FakeSpectrumAnalyzer()
    events = []
    original_close = analyzer.close
    original_off = microscope.gun_off
    def close():
        original_close(); events.append('eds_closed')
    def off():
        assert analyzer.closed, 'EDS must be cleaned up before switching off the gun'
        events.append('gun_off')
        if mode == 'shutdown_error': raise RuntimeError('HV timeout')
        if mode == 'shutdown_unconfirmed': return 'EnumHvStatus.Open'
        return original_off()
    monkeypatch.setattr(analyzer, 'close', close)
    monkeypatch.setattr(microscope, 'gun_off', off)
    if mode == 'partial':
        monkeypatch.setattr(analyzer, 'analyze', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('EDS failed')))
    if mode == 'fatal':
        monkeypatch.setattr(microscope, 'connect', lambda: (_ for _ in ()).throw(RuntimeError('connection failed')))
    result = run_cleanliness_analysis(task_dir, config=replace(config, gun_off_after_task=mode!='disabled'),
        microscope=microscope, spectrum_analyzer=analyzer, should_stop=lambda: mode=='stop')
    if mode == 'disabled':
        assert events == ['eds_closed'] and microscope.gun_is_on
        assert result['cleanup']['gunOffRequested'] is False
    else:
        assert events == ['eds_closed','gun_off']
        if mode.startswith('shutdown_'):
            assert result['cleanup']['gunOffVerified'] is False
            assert '自动关枪失败' in result['error']
            assert result['particles'][0]['status'] == 'SUCCEEDED'
        else:
            assert result['cleanup']['gunOffVerified'] is True
            assert microscope.gun_is_on is False
    expected = {'partial':'COMPLETED_WITH_FAILURES','stop':'STOPPED','fatal':'FAILED'}.get(mode,'COMPLETED')
    assert result['status'] == expected


def test_analysis_stops_retrying_after_transient_eds_retry_is_exhausted(tmp_path):
    task_dir, config, microscope = _calibrated_task(tmp_path)
    analyzer = _SequencedSpectrumAnalyzer(
        [
            "Particle 1 EDS job failed: StopEdSpectrumAcquisition failed",
            "Particle 1 EDS job failed: StopEdSpectrumAcquisition failed",
        ]
    )

    manifest = run_cleanliness_analysis(
        task_dir,
        config=replace(config, eds_retry_count=1),
        microscope=microscope,
        spectrum_analyzer=analyzer,
    )

    record = manifest["particles"][0]
    assert manifest["status"] == "COMPLETED_WITH_FAILURES"
    assert len(analyzer.calls) == 2
    assert [item["status"] for item in record["edsAttempts"]] == [
        "FAILED",
        "FAILED",
    ]
    assert "StopEdSpectrumAcquisition failed" in record["error"]


def test_analysis_does_not_retry_non_transient_eds_failure(tmp_path):
    task_dir, config, microscope = _calibrated_task(tmp_path)
    analyzer = _SequencedSpectrumAnalyzer(["EDS configuration is invalid"])

    manifest = run_cleanliness_analysis(
        task_dir,
        config=replace(config, eds_retry_count=1),
        microscope=microscope,
        spectrum_analyzer=analyzer,
    )

    record = manifest["particles"][0]
    assert manifest["status"] == "COMPLETED_WITH_FAILURES"
    assert len(analyzer.calls) == 1
    assert record["edsAttempts"][0]["transient"] is False
    assert record["error"] == "EDS configuration is invalid"


def test_analysis_blocks_out_of_roi_target_before_stage_move(tmp_path):
    task_dir, config, microscope = _calibrated_task(tmp_path)
    manifest_path = task_dir / "task.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = manifest["calibration"]["groups"]
    groups[0]["result"]["translation"] = [500000.0, 500000.0]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    microscope.move_targets.clear()
    analyzer = _FakeSpectrumAnalyzer()

    result = run_cleanliness_analysis(
        task_dir,
        config=config,
        microscope=microscope,
        spectrum_analyzer=analyzer,
    )

    assert result["status"] == "COMPLETED_WITH_FAILURES"
    assert "超出经验标定有效 ROI" in result["particles"][0]["error"]
    assert len(microscope.move_targets) == 1  # finally 中仅恢复起始位置
    assert analyzer.calls == []
    assert "Close" in result["cleanup"]["gunOffStatus"]
