"""Real-image behavior regression, not an accuracy benchmark or hardware test."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.cleanliness_core.cleanliness_detection import CandidateSegmentationConfig, build_sem_anchor_candidates
from backend.cleanliness_core.cleanliness_task import _select_mask_guided_candidate
from backend.cleanliness_core.optical_particle_analysis import load_optical_export

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures/mask_matching_sample2"
BASELINE = json.loads((FIXTURE / "baseline.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def replay_inputs():
    optical = FIXTURE / "optical.json"
    assert hashlib.sha256(optical.read_bytes()).hexdigest() == BASELINE["opticalSubsetSha256"]
    model = ROOT / BASELINE["model"]
    assert model.is_file(), "Regression model must be available; do not silently skip real-image tests"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == BASELINE["modelSha256"]
    segmentation = CandidateSegmentationConfig(backend="yolo", yolo_model_path=model)
    return load_optical_export(optical), segmentation


@pytest.mark.parametrize("case", BASELINE["cases"], ids=lambda c: f"rank-{c['rank']}-seq-{c['sequence']}")
def test_real_particle_matching_does_not_regress(replay_inputs, case):
    export, segmentation = replay_inputs
    particle = export.by_sequence[case["sequence"]]
    assert particle.particle_id == case["particleId"]
    raw = (FIXTURE / case["image"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == case["imageSha256"]
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    candidates = build_sem_anchor_candidates(particle, image, pixel_size_um=BASELINE["pixelSizeUm"],
                                             stage_center_um=(0, 0), segmentation=segmentation)
    chosen, decision, audit = _select_mask_guided_candidate(image, particle, BASELINE["pixelSizeUm"], candidates,
                                                            stage_center_um=(0, 0), segmentation=segmentation)
    assert decision.matched == case["accepted"], f"Unexpected acceptance change for rank {case['rank']}: {audit}"
    assert decision.status == case["status"]
    assert audit["accepted"] == case["accepted"]
    if case["accepted"]:
        assert np.linalg.norm(np.subtract(chosen.pixel_xy, case["newCenterPx"])) <= BASELINE["centerTolerancePx"]
        assert audit["maskIoU"] >= case["minimumIoU"]
        if case["knownWrongCenterPx"]:
            assert np.linalg.norm(np.subtract(chosen.pixel_xy, case["knownWrongCenterPx"])) > 40


def test_regression_corpus_covers_acceptance_and_rejection():
    cases = BASELINE["cases"]
    assert len(cases) == len({c["sequence"] for c in cases}) == 49
    assert sum(c["accepted"] for c in cases) == 40
    assert sum(not c["accepted"] for c in cases) == 9
    assert {u['rank'] for u in BASELINE['baselineUpdates20260910']} == {1,664,673}
    assert {c["rank"] for c in cases if c["knownWrongCenterPx"]} == {597, 631, 722}
