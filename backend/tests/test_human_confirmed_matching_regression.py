"""Human identity labels remain separate from the preserved historical rejection."""
import base64
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.cleanliness_core.cleanliness_detection import (
    CandidateSegmentationConfig,
    build_sem_anchor_candidates,
)
from backend.cleanliness_core.cleanliness_task import _select_mask_guided_candidate
from backend.cleanliness_core.optical_particle_analysis import load_optical_export


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures/human_confirmed_20260918"
CORPUS = json.loads((FIXTURE / "cases.json").read_text(encoding="utf-8"))
CASES = CORPUS["cases"]
CONFIRMED = [case for case in CASES if case["reviewStatus"] == "human_confirmed"]


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(scope="module")
def inputs():
    path = FIXTURE / "optical.json"
    assert _sha(path.read_bytes()) == CORPUS["opticalSubsetSha256"]
    model = ROOT / CORPUS["model"]["path"]
    assert model.is_file(), "The pinned regression model must not be silently skipped"
    assert _sha(model.read_bytes()) == CORPUS["model"]["sha256"]
    return load_optical_export(path), CandidateSegmentationConfig(
        backend="yolo", yolo_model_path=model
    )


def test_feedback_scope_is_distinct_from_historical_behavior():
    assert CORPUS["sourceTaskId"] == "82599546-2f1d-40a8-b986-7fca42d99133"
    assert len(CASES) == len({case["caseId"] for case in CASES}) == 4
    assert {(c["reportRank"], c["sequence"], c["attempt"]) for c in CONFIRMED} == {
        (48, 53, 2), (125, 143, 2)
    }
    assert all(c["reviewStatus"] == "linked_repeat_capture" for c in CASES if c["attempt"] == 1)
    assert CORPUS["feedback"]["source"] == "user_in_conversation"
    assert CORPUS["feedback"]["verbatim"] == "我可以 肯定 sem图上两个图的 A就是 正确的。"
    for case in CASES:
        assert case["desiredBehavior"]["accepted"] is True
        assert case["desiredBehavior"]["targetLabel"] == "A"
        assert case["desiredBehavior"]["mustNotSelect"] == ["B"]
        assert case["historicalBehavior"]["accepted"] is False


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["caseId"])
def test_evidence_fingerprints_and_object_identity(inputs, case):
    export, _ = inputs
    particle = export.by_sequence[case["sequence"]]
    assert particle.particle_id == case["particleId"]
    for filename, fingerprint in [
        ("image", "imageSha256"), ("mask", "maskSha256"),
        ("historicalRecord", "historicalRecordSha256"),
    ]:
        assert _sha((FIXTURE / case[filename]).read_bytes()) == case[fingerprint]
    mask_bytes = base64.b64decode(particle.mask_data_base64.split(",")[-1])
    assert mask_bytes == (FIXTURE / case["mask"]).read_bytes()
    raw = (FIXTURE / case["image"]).read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    assert image.shape == (512, 512) and image.dtype == np.uint8
    history = json.loads((FIXTURE / case["historicalRecord"]).read_text(encoding="utf-8"))
    assert history["particleId"] == case["particleId"]
    assert history["rank"] == case["reportRank"]
    assert history["sequence"] == case["sequence"]
    assert history["energySpectrum"] is None
    audit = history["captureAttempts"][case["attempt"] - 1]["maskVerification"]
    assert audit["centerPixel"] == case["correctTarget"]["referenceCenterPx"]
    assert audit["runnerUp"]["centerPixel"] == case["wrongTarget"]["referenceCenterPx"]
    assert np.linalg.norm(np.subtract(audit["centerPixel"], audit["runnerUp"]["centerPixel"])) > 100


@pytest.fixture(scope="module")
def replay(inputs):
    export, segmentation = inputs
    cache = {}

    def run(case):
        if case["caseId"] not in cache:
            particle = export.by_sequence[case["sequence"]]
            raw = (FIXTURE / case["image"]).read_bytes()
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
            kwargs = dict(pixel_size_um=CORPUS["pixelSizeUm"], stage_center_um=(0, 0), segmentation=segmentation)
            candidates = build_sem_anchor_candidates(particle, image, **kwargs)
            cache[case["caseId"]] = _select_mask_guided_candidate(
                image, particle, CORPUS["pixelSizeUm"], candidates,
                stage_center_um=(0, 0), segmentation=segmentation,
            )
        return cache[case["caseId"]]

    return run


def _assert_target_a(center, case):
    correct = case["correctTarget"]["referenceCenterPx"]
    wrong = case["wrongTarget"]["referenceCenterPx"]
    assert np.linalg.norm(np.subtract(center, correct)) <= CORPUS["associationTolerancePx"]
    assert np.linalg.norm(np.subtract(center, wrong)) > 40


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["caseId"])
def test_matching_must_never_accept_competitor_b(replay, case):
    chosen, decision, audit = replay(case)
    _assert_target_a(audit["centerPixel"], case)
    assert decision.matched, audit
    assert chosen is not None and audit["accepted"] is True
    _assert_target_a(chosen.pixel_xy, case)
    assert audit["maskIoU"] >= 0.5
    assert audit["status"] == "MATCH_BY_SHAPE_CONSENSUS"
    assert audit["templateStatus"] == "AMBIGUOUS" and audit["margin"] < .07
    assert audit["shapeConsensus"]["corroborated"] is True
    assert audit["shapeConsensus"]["allowsEds"] is False
    assert audit["shapeConsensus"]["yoloShapeMargin"] >= .07
    assert audit["shapeConsensus"]["grayscaleShapeMargin"] >= .07
    assert audit["consensusCenterDifferencePx"] <= audit["maximumConsensusCenterDifferencePx"]


@pytest.mark.parametrize("case", CONFIRMED, ids=lambda c: c["caseId"])
def test_human_confirmed_target_a_should_be_accepted(replay, case):
    chosen, decision, audit = replay(case)
    _assert_target_a(audit["centerPixel"], case)
    assert decision.matched, audit
    assert chosen is not None and audit["accepted"] is True
    _assert_target_a(chosen.pixel_xy, case)
    assert audit["maskIoU"] >= 0.5
