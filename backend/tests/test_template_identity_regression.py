"""Reported false rejections and a real distinct-location competitor; no devices."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.cleanliness_core.cleanliness_detection import CandidateSegmentationConfig, build_sem_anchor_candidates
from backend.cleanliness_core.cleanliness_task import _select_mask_guided_candidate
from backend.cleanliness_core.optical_particle_analysis import load_optical_export

ROOT=Path(__file__).resolve().parents[2]
FIXTURE=Path(__file__).parent/'fixtures/template_identity_cases'
CASES=json.loads((FIXTURE/'cases.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def inputs():
    path=FIXTURE/'optical.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==CASES['opticalSha256']
    return load_optical_export(path),CandidateSegmentationConfig(backend='yolo',
        yolo_model_path=ROOT/'resources/models/cleanliness/particle_yolo11n_seg_reann_20260811_best.onnx')


@pytest.mark.parametrize('case',CASES['cases'],ids=lambda c:f"rank{c['rank']}-capture{c['attempt']}")
def test_reported_template_rejection_regression(inputs,case):
    export,segmentation=inputs
    raw=(FIXTURE/case['image']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==case['imageSha256']
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),-1)
    particle=export.by_sequence[case['sequence']]
    candidates=build_sem_anchor_candidates(particle,image,pixel_size_um=CASES['pixelSizeUm'],
        stage_center_um=(0,0),segmentation=segmentation)
    chosen,decision,audit=_select_mask_guided_candidate(image,particle,CASES['pixelSizeUm'],candidates,
        stage_center_um=(0,0),segmentation=segmentation)
    assert decision.matched==case['expectedAccepted'],audit
    if case['expectedAccepted']:
        assert audit['score']>=.65 and audit['margin']>=.07
        assert audit['maskIoU']>=.5
        assert np.linalg.norm(np.subtract(chosen.pixel_xy,case['expectedCenterPx']))<=5
    else:
        assert audit['status']=='AMBIGUOUS'
        assert audit['runnerUpOverlapIoU']==0
        assert np.linalg.norm(np.subtract(audit['runnerUp']['centerPixel'],audit['centerPixel']))>40
        assert audit['margin']<.07
