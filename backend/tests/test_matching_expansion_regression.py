"""Additional real captures; historical matching is compatibility, not identity truth."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.cleanliness_core.cleanliness_detection import CandidateSegmentationConfig, build_sem_anchor_candidates
from backend.cleanliness_core.cleanliness_task import _select_mask_guided_candidate
from backend.cleanliness_core.optical_particle_analysis import load_optical_export

ROOT=Path(__file__).resolve().parents[2]
FIXTURE=Path(__file__).parent/'fixtures/matching_expansion_20260918'
CORPUS=json.loads((FIXTURE/'cases.json').read_text(encoding='utf-8'))
LEGACY_REVIEW=json.loads((FIXTURE/'legacy_version_review.json').read_text(encoding='utf-8'))
LEGACY_CASES={c['caseId']:c for c in LEGACY_REVIEW['cases']}


@pytest.fixture(scope='module')
def replay_inputs():
    model=ROOT/CORPUS['model']
    assert model.is_file(), 'The real-image regression model must not be skipped'
    assert hashlib.sha256(model.read_bytes()).hexdigest()==CORPUS['modelSha256']
    sources={}
    for source in CORPUS['sources']:
        optical=FIXTURE/source['optical']
        assert hashlib.sha256(optical.read_bytes()).hexdigest()==source['opticalSubsetSha256']
        parameters=source['replaySegmentation']
        assert parameters['backend']=='yolo' and parameters['modelPath']==CORPUS['model']
        segmentation=CandidateSegmentationConfig(backend='yolo',yolo_model_path=model,
            input_size=parameters['inputSize'],confidence_threshold=parameters['confidenceThreshold'],
            iou_threshold=parameters['iouThreshold'],mask_threshold=parameters['maskThreshold'],
            maximum_detections=parameters['maxDetections'])
        assert np.isfinite(source['pixelSizeUm']) and source['pixelSizeUm']>0
        sources[source['id']]=(load_optical_export(optical),segmentation,source['pixelSizeUm'])
    return sources


def test_expansion_has_new_fingerprinted_frames_and_explicit_label_scope():
    cases=CORPUS['cases']
    assert len(cases)==len({c['caseId'] for c in cases})==len({c['imageSha256'] for c in cases})==78
    assert Counter(c['source'] for c in cases)=={
        'sample7':34,'sample2_stable':26,'sample2_low_signal':10,'legacy_without_mask':8}
    assert sum(c['expectedAccepted'] for c in cases)==36
    assert all(c['labelKind']=='historical_behavior_only' for c in cases)
    assert set(c['category'] for c in cases)=={
        'accepted_behavior','template_does_not_fit','matched_despite_eds_failure',
        'below_template_score','insufficient_final_contour','distinct_competition',
        'legacy_no_mask_acceptance','legacy_physical_or_shape_rejection'}
    existing=set()
    for name,filename in [('mask_matching_sample2','baseline.json'),('template_identity_cases','cases.json'),
                          ('human_confirmed_20260918','cases.json')]:
        previous=json.loads((FIXTURE.parent/name/filename).read_text(encoding='utf-8'))
        existing.update(c['imageSha256'] for c in previous['cases'])
    assert not existing.intersection(c['imageSha256'] for c in cases)
    eds_failed=[c for c in cases if c['category']=='matched_despite_eds_failure']
    assert len(eds_failed)==2
    assert all(c['historicalTaskStatus']=='FAILED' and c['expectedAccepted'] for c in eds_failed)


def test_legacy_version_reference_preserves_history_and_does_not_claim_identity():
    assert hashlib.sha256((FIXTURE/'cases.json').read_bytes()).hexdigest()==LEGACY_REVIEW['importedHistoricalManifestSha256']
    assert LEGACY_REVIEW['referenceCommit']=='7cb754459a86632547e98d5c7ce34da6b8106705'
    assert len(LEGACY_CASES)==8
    assert LEGACY_REVIEW['summary']=={'reviewed':8,'historicalDifferences':6,'committedAccepted':7}
    assert LEGACY_REVIEW['identityReviewStatus'].startswith('unconfirmed')
    for case in CORPUS['cases']:
        if case['source']!='legacy_without_mask':
            assert case['caseId'] not in LEGACY_CASES
            continue
        review=LEGACY_CASES[case['caseId']]
        assert review['historicalExpected']['accepted']==case['expectedAccepted']
        assert review['historicalExpected']['status']==case['expectedDecisionStatus']
        assert review['workingTreeReproducesCommitted'] is True


@pytest.mark.parametrize('case',CORPUS['cases'],ids=lambda c:c['caseId'])
def test_added_real_frame_preserves_matching_behavior(replay_inputs,case):
    export,segmentation,pixel=replay_inputs[case['source']]
    particle=export.by_sequence[case['sequence']]
    assert particle.particle_id==case['particleId']
    assert bool(particle.mask_data_base64)==case['hasCustomerMask']
    raw=(FIXTURE/case['image']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==case['imageSha256']
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED)
    assert image is not None and image.ndim==2
    candidates=build_sem_anchor_candidates(particle,image,pixel_size_um=pixel,
        stage_center_um=(0,0),segmentation=segmentation)
    chosen,decision,audit=_select_mask_guided_candidate(image,particle,pixel,candidates,
        stage_center_um=(0,0),segmentation=segmentation)
    detail={'caseId':case['caseId'],'historical':case['historicalMatch'],'actual':decision.to_dict(),'audit':audit}
    expected={'accepted':case['expectedAccepted'],'status':case['expectedDecisionStatus'],
              'centerPx':case['expectedCenterPx'],'centerTolerancePx':case['centerTolerancePx']}
    if case['source']=='legacy_without_mask':
        # This independent committed-version baseline was audited separately;
        # the original historical expectations remain immutable above.
        expected=LEGACY_CASES[case['caseId']]['committedExpected']
        detail['versionReference']=LEGACY_REVIEW['referenceCommit']
    assert decision.matched==expected['accepted'],detail
    assert decision.status==expected['status'],detail
    if case['hasCustomerMask']:
        assert audit['accepted']==expected['accepted'],detail
    else:
        assert audit is None
    if expected['accepted']:
        assert chosen is not None
        assert np.linalg.norm(np.subtract(chosen.pixel_xy,expected['centerPx']))<=expected['centerTolerancePx'],detail
        if case['minimumMaskIoU'] is not None:
            assert audit['maskIoU']>=case['minimumMaskIoU'],detail
    elif case['hasCustomerMask']:
        assert audit['status']==case['historicalMaskVerification']['status'],detail
        assert not (audit.get('shapeConsensus') or {}).get('corroborated',False),detail
