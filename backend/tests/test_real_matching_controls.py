"""Derived real-image positives and safety negatives; failures are not auto-xfailed."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.cleanliness_core.cleanliness_detection import CandidateSegmentationConfig,build_sem_anchor_candidates
from backend.cleanliness_core.cleanliness_task import _select_mask_guided_candidate
from backend.cleanliness_core.optical_particle_analysis import load_optical_export
from backend.cleanliness_core.shape_consensus import review_ambiguous_shapes

ROOT=Path(__file__).resolve().parents[2]
FIXTURE=Path(__file__).parent/'fixtures/real_matching_controls_20260918'
CORPUS=json.loads((FIXTURE/'cases.json').read_text(encoding='utf-8'))
SOURCE=FIXTURE.parent/CORPUS['sourceCorpus']


@pytest.fixture(scope='module')
def inputs():
    optical=SOURCE/'optical.json';model=ROOT/CORPUS['model']['path']
    assert hashlib.sha256((SOURCE/'cases.json').read_bytes()).hexdigest()==CORPUS['sourceCorpusSha256']
    assert hashlib.sha256(optical.read_bytes()).hexdigest()==CORPUS['sourceOpticalSha256']
    assert hashlib.sha256(model.read_bytes()).hexdigest()==CORPUS['model']['sha256']
    return load_optical_export(optical),CandidateSegmentationConfig(backend='yolo',yolo_model_path=model)


def load_case(inputs,case):
    export,segmentation=inputs
    particle=export.by_sequence[case['sequence']]
    assert particle.particle_id==case['particleId']
    assert hashlib.sha256((SOURCE/case['sourceImage']).read_bytes()).hexdigest()==case['sourceImageSha256']
    raw=(FIXTURE/case['image']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==case['imageSha256']
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED)
    return image,particle,segmentation


def run_case(inputs,case):
    image,particle,segmentation=load_case(inputs,case)
    pixel=CORPUS['pixelSizeUm']
    candidates=build_sem_anchor_candidates(particle,image,pixel_size_um=pixel,stage_center_um=(0,0),segmentation=segmentation)
    return _select_mask_guided_candidate(image,particle,pixel,candidates,stage_center_um=(0,0),segmentation=segmentation)


def test_derived_controls_do_not_inflate_raw_corpus_or_human_labels():
    cases=CORPUS['cases']
    assert len(cases)==len({c['imageSha256'] for c in cases})==16
    assert sum(c['expectedAccepted'] for c in cases)==8
    assert all(c['labelKind']=='derived_control_not_raw_acquisition' for c in cases)
    assert {c['rank'] for c in cases}=={48,125}
    source_spec=json.loads((SOURCE/'cases.json').read_text(encoding='utf-8'))
    by_id={c['caseId']:c for c in source_spec['cases']}
    for case in cases:
        assert by_id[case['sourceCaseId']]['reviewStatus']=='human_confirmed'
        if case['condition']=='target_erased_competitor_remains':
            assert case['expectedAccepted'] is False
            assert case['knownWrongCenterPx'] is not None
            original=cv2.imdecode(np.frombuffer((SOURCE/case['sourceImage']).read_bytes(),np.uint8),-1)
            derived=cv2.imdecode(np.frombuffer((FIXTURE/case['image']).read_bytes(),np.uint8),-1)
            left,top,right,bottom=case['derivation']['replaceXyxy']
            assert np.array_equal(derived[top:bottom,left:right],original[top:bottom,left-120:right-120])
            outside=np.ones(original.shape,bool);outside[top:bottom,left:right]=False
            assert np.array_equal(derived[outside],original[outside])
            bx,by=np.rint(case['knownWrongCenterPx']).astype(int)
            assert np.array_equal(derived[by-10:by+10,bx-10:bx+10],original[by-10:by+10,bx-10:bx+10])


@pytest.mark.parametrize('case',CORPUS['cases'],ids=lambda c:c['caseId'])
def test_derived_real_image_preserves_target_or_rejects(inputs,case):
    chosen,decision,audit=run_case(inputs,case)
    detail={'caseId':case['caseId'],'expectedAccepted':case['expectedAccepted'],
            'knownWrongCenterPx':case['knownWrongCenterPx'],'decision':decision.to_dict(),'audit':audit}
    assert decision.matched==case['expectedAccepted'],detail
    if case['expectedAccepted']:
        assert np.linalg.norm(np.subtract(chosen.pixel_xy,case['expectedTargetCenterPx']))<=12,detail
        assert audit['accepted'] and audit['maskIoU']>=.5
    else:
        assert not audit['accepted'],detail


@pytest.mark.parametrize('rank',[48,125])
@pytest.mark.parametrize('bad_center',[[600,200],[200,-20]])
def test_real_audit_wrong_frame_coordinates_are_rejected_before_segmentation(inputs,monkeypatch,rank,bad_center):
    case=next(c for c in CORPUS['cases'] if c['rank']==rank and c['condition']=='contrast')
    image,particle,segmentation=load_case(inputs,case)
    history=json.loads((SOURCE/'history'/f"{case['sequence']:04d}.json").read_text(encoding='utf-8'))
    audit=history['captureAttempts'][1]['maskVerification'].copy()
    audit['centerPixel']=bad_center
    calls=[]
    def forbidden(*a,**k):calls.append(True);raise AssertionError('Invalid coordinates must stop before inference')
    monkeypatch.setattr('backend.cleanliness_core.shape_consensus.yolo_seg_candidates',forbidden)
    result=review_ambiguous_shapes(image,particle,CORPUS['pixelSizeUm'],audit,segmentation=segmentation)
    assert not calls
    assert not result['corroborated'] and not result['allowsEds']


@pytest.mark.parametrize('rank',[48,125])
@pytest.mark.parametrize('condition',['identical_competitor','background_only','target_erased_competitor_remains'])
def test_rejected_real_control_never_enters_eds(inputs,tmp_path,monkeypatch,rank,condition):
    from backend.tests.test_cleanliness_task import _calibrated_task,_FakeSpectrumAnalyzer
    from backend.cleanliness_core import cleanliness_task as engine
    case=next(c for c in CORPUS['cases'] if c['rank']==rank and c['condition']==condition)
    image,particle,segmentation=load_case(inputs,case)
    task_dir,config,microscope=_calibrated_task(tmp_path)
    original=engine._select_mask_guided_candidate
    calls=[]
    def real_control(*args,**kwargs):
        calls.append(True)
        candidates=build_sem_anchor_candidates(particle,image,pixel_size_um=CORPUS['pixelSizeUm'],stage_center_um=(0,0),segmentation=segmentation)
        return original(image,particle,CORPUS['pixelSizeUm'],candidates,stage_center_um=(0,0),segmentation=segmentation)
    monkeypatch.setattr(engine,'_select_mask_guided_candidate',real_control)
    analyzer=_FakeSpectrumAnalyzer()
    result=engine.run_cleanliness_analysis(task_dir,config=config,microscope=microscope,spectrum_analyzer=analyzer)
    assert calls
    assert not analyzer.calls
    assert result['status']=='COMPLETED_WITH_FAILURES'
    assert all(r['status']=='FAILED' and r['energySpectrum'] is None for r in result['particles'])
