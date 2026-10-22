"""Real rerun fixes with frozen source evidence and explicit refusal controls."""
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.cleanliness_core.calibrated_match_location import review_calibrated_location,validate_selected_location
from backend.cleanliness_core.cleanliness_detection import CandidateSegmentationConfig, build_sem_anchor_candidates
from backend.cleanliness_core.cleanliness_task import _select_mask_guided_candidate
from backend.cleanliness_core.large_particle_center import plan_large_particle_center
from backend.cleanliness_core.optical_particle_analysis import load_optical_export

ROOT=Path(__file__).resolve().parents[2]
FIXTURE=Path(__file__).parent/'fixtures/rerun_failures_20260920'
CORPUS=json.loads((FIXTURE/'cases.json').read_text(encoding='utf-8'))
CASES=CORPUS['cases']
SMALL=[c for c in CASES if c['sequence']==131]
LARGE=[c for c in CASES if c['sequence']!=131]


@pytest.fixture(scope='module')
def inputs():
    model=ROOT/CORPUS['model']
    assert hashlib.sha256(model.read_bytes()).hexdigest()==CORPUS['modelSha256']
    exports={}
    for s in CORPUS['sources']:
        path=FIXTURE/s['optical']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==s['opticalSha256']
        assert hashlib.sha256((FIXTURE/s['id']/'settings.json').read_bytes()).hexdigest()==s['settingsSha256']
        exports[s['id']]=load_optical_export(path)
    return exports,CandidateSegmentationConfig(backend='yolo',yolo_model_path=model)


def load(inputs,case):
    exports,seg=inputs
    particle=exports[case['source']].by_sequence[case['sequence']]
    assert particle.particle_id==case['particleId']
    raw=(FIXTURE/case['image']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==case['imageSha256']
    hpath=FIXTURE/case['history']
    assert hashlib.sha256(hpath.read_bytes()).hexdigest()==case['historySha256']
    history=json.loads(hpath.read_text(encoding='utf-8'))
    source=next(s for s in CORPUS['sources'] if s['id']==case['source'])
    return cv2.imdecode(np.frombuffer(raw,np.uint8),-1),particle,source,seg,history


def prior(history):
    return {'reviewAccepted':True,'maximumCenterErrorUm':160.,
            'targetStageUm':[history['targetStageUm'][axis] for axis in ['x','y']]}


def replay(inputs,case,image_override=None):
    image,p,source,seg,history=load(inputs,case)
    if image_override is not None:image=image_override
    stage=tuple(history['actualStageUm'][axis] for axis in ['x','y'])
    pixel=source['pixelSizeUm']
    candidates=build_sem_anchor_candidates(p,image,pixel_size_um=pixel,stage_center_um=stage,segmentation=seg)
    return _select_mask_guided_candidate(image,p,pixel,candidates,stage_center_um=stage,segmentation=seg,
                                         calibration_location=prior(history))


@pytest.fixture(scope='module')
def small_replays(inputs):
    return {c['caseId']:replay(inputs,c) for c in SMALL}


def test_corpus_preserves_six_failures_and_previous_success_without_relabeling_history(inputs):
    assert len(CASES)==7 and len({c['imageSha256'] for c in CASES})==7
    assert sum(c['source']=='rerun' for c in CASES)==6
    assert 'Historical success alone is not identity truth' in CORPUS['labelMeaning']
    for case in CASES:
        _,_,_,_,history=load(inputs,case)
        if case['source']=='rerun':
            assert history['status']=='FAILED' and history['energySpectrum'] is None
        else:assert history['status']=='SUCCEEDED'


@pytest.mark.parametrize('case',SMALL,ids=lambda c:c['caseId'])
def test_small_target_survives_competitor_entering_field(inputs,small_replays,case):
    chosen,decision,audit=small_replays[case['caseId']]
    assert decision.matched,audit
    assert audit['accepted'] and audit['maskIoU']>=.5
    assert np.linalg.norm(np.subtract(chosen.pixel_xy,case['expectedTargetCenterPx']))<=case['centerTolerancePx']
    assert np.linalg.norm(np.subtract(chosen.pixel_xy,case['competitorCenterPx']))>100
    if case['source']=='rerun':
        assert audit['status']=='MATCH_BY_CALIBRATED_LOCATION'
        assert audit['calibratedLocation']['accepted']
        assert audit['calibratedLocation']['allowsEds'] is False
        assert audit['shapeConsensus']['corroborated'] is False
        # The strict existing shape route remains refused, rather than lowering its gates.
        assert audit['calibratedLocation']['candidateDistancesUm'][1]>160


@pytest.mark.parametrize('case',LARGE,ids=lambda c:c['caseId'])
def test_large_rerun_has_a_local_interior_point_and_skips_matching(inputs,case):
    image,p,s,_,_=load(inputs,case)
    result=plan_large_particle_center(image,p,s['pixelSizeUm'],field_width_um=s['fieldWidthUm'])
    assert result['status']=='READY',result
    assert result['identityVerified'] is False and result['contourMatchingSkipped']
    assert result['pointCoverage']>=.9 and result['foregroundClearancePx']>=3
    assert result['pointOffsetFromCenterPx']<=8
    x,y=result['pointPx'];assert image[y,x]>result['intensityThreshold']
    if case['sequence']==7:
        assert result['eligibilityReason']=='near_field_and_visibly_clipped'
        assert result['lengthToFieldWidth']<1
        assert result['pointPx']==[256,256]
    else:
        assert result['centerCoverage']<.9
        assert result['pointPx']!=[256,256]


@pytest.mark.parametrize('problem',['duplicate_in_range','winner_outside','wrong_stage','bad_axes','unreviewed',
    'unbounded_error','nan_center','truncated','third_target','low_score','only_competitor'])
def test_calibration_prior_cannot_override_competition_or_invalid_evidence(inputs,problem):
    case=SMALL[0];image,p,s,_,h=load(inputs,case)
    audit=copy.deepcopy(h['captureAttempts'][0]['maskVerification']);location=prior(h)
    stage=[h['actualStageUm'][axis] for axis in ['x','y']];axes=((1,0),(0,1))
    if problem=='duplicate_in_range':audit['runnerUp']['centerPixel']=[228,229]
    elif problem=='winner_outside':audit['centerPixel'],audit['runnerUp']['centerPixel']=audit['runnerUp']['centerPixel'],audit['centerPixel']
    elif problem=='wrong_stage':stage[0]+=40
    elif problem=='bad_axes':axes=((1,0),(0,0))
    elif problem=='unreviewed':location['reviewAccepted']=False
    elif problem=='unbounded_error':location['maximumCenterErrorUm']=1000
    elif problem=='nan_center':audit['centerPixel']=[float('nan'),220]
    elif problem=='truncated':audit['searchTruncated']=True
    elif problem=='third_target':audit['distinctHypothesisCount']=3
    elif problem=='low_score':audit['score']=.5
    elif problem=='only_competitor':audit.update(status='MATCH',distinctHypothesisCount=1)
    result=review_calibrated_location(image.shape,p,s['pixelSizeUm'],audit,calibration_location=location,
                                      stage_center_um=stage,pixel_axis_matrix=axes)
    assert not result['accepted'] and not result['allowsEds'],result


@pytest.mark.parametrize('problem',['background','center_hole','isolated_speck','thin_fragment','wrong_field'])
@pytest.mark.parametrize('sequence',[7,6])
def test_large_local_point_still_refuses_missing_or_insufficient_particle(inputs,sequence,problem):
    case=next(c for c in LARGE if c['sequence']==sequence)
    image,p,s,_,_=load(inputs,case);field=s['fieldWidthUm']
    if problem=='background':image[:]=25
    elif problem=='center_hole':image[210:303,210:303]=25
    elif problem=='isolated_speck':image[:]=25;image[249:264,249:264]=210
    elif problem=='thin_fragment':image[:]=25;image[255:258,:]=210
    elif problem=='wrong_field':field/=2
    result=plan_large_particle_center(image,p,s['pixelSizeUm'],field_width_um=field)
    assert result['status']!='READY' and 'pointPx' not in result,result


@pytest.mark.parametrize('sequence',[7,6,131])
def test_real_images_through_full_task_choose_correct_mode_and_point(inputs,tmp_path,monkeypatch,sequence):
    from backend.tests.test_large_particle_center import large_task
    from backend.tests.test_cleanliness_task import _FakeSpectrumAnalyzer
    from backend.cleanliness_core import cleanliness_task as engine
    case=next(c for c in CASES if c['source']=='rerun' and c['sequence']==sequence)
    image,p,s,seg,_=load(inputs,case)
    task,config,scope=large_task(tmp_path,monkeypatch,p,image)
    settings=json.loads(config.settings_path.read_text(encoding='utf-8'))
    settings['calibration']['acceptance']['maxErrorUm']=160
    settings['calibration']['candidateSegmentation'].update(backend='yolo',modelPath=str(seg.yolo_model_path))
    config.settings_path.write_text(json.dumps(settings),encoding='utf-8')
    if sequence!=131:
        def forbidden(*a,**k):raise AssertionError('Large-particle route must skip candidate and template matching')
        monkeypatch.setattr(engine,'build_sem_anchor_candidates',forbidden)
        monkeypatch.setattr(engine,'_select_mask_guided_candidate',forbidden)
    analyzer=_FakeSpectrumAnalyzer()
    result=engine.run_cleanliness_analysis(task,config=config,microscope=scope,spectrum_analyzer=analyzer)
    record=result['particles'][0]
    assert result['status']=='COMPLETED',record
    assert len(analyzer.calls)==1
    point=analyzer.calls[0][1]
    if sequence==131:
        assert record['acquisitionMode']=='matched_particle'
        assert np.linalg.norm(np.subtract(point,case['expectedTargetCenterPx']))<20
        assert record['maskVerification']['calibratedLocation']['accepted']
    else:
        assert record['acquisitionMode']=='large_particle_center_region'
        assert np.linalg.norm(np.subtract(point,[256,256]))<=8
    assert scope.gun_is_on is False


@pytest.mark.parametrize('problem',['background','wrong_field','center_hole'])
def test_large_refusal_in_full_task_never_enters_eds(inputs,tmp_path,monkeypatch,problem):
    from backend.tests.test_large_particle_center import large_task
    from backend.tests.test_cleanliness_task import _FakeSpectrumAnalyzer
    from backend.cleanliness_core import cleanliness_task as engine
    case=next(c for c in LARGE if c['sequence']==6)
    image,p,_,_,_=load(inputs,case)
    if problem=='background':image[:]=25
    elif problem=='center_hole':image[210:303,210:303]=25
    task,config,scope=large_task(tmp_path,monkeypatch,p,image)
    if problem=='wrong_field':
        original=scope.capture
        def bad_capture(path):
            frame,metadata=original(path);metadata['meta']['FieldSize']/=2
            return frame,metadata
        scope.capture=bad_capture
    analyzer=_FakeSpectrumAnalyzer()
    result=engine.run_cleanliness_analysis(task,config=config,microscope=scope,spectrum_analyzer=analyzer)
    assert result['status']=='COMPLETED_WITH_FAILURES'
    assert not analyzer.calls and result['particles'][0]['energySpectrum'] is None
    assert scope.gun_is_on is False


@pytest.mark.parametrize('problem',['target_erased','duplicate_in_range','noise','wrong_stage'])
def test_real_small_negative_is_rejected_and_never_enters_eds(inputs,tmp_path,monkeypatch,problem):
    from backend.tests.test_cleanliness_task import _calibrated_task,_FakeSpectrumAnalyzer
    from backend.cleanliness_core import cleanliness_task as engine
    case=SMALL[0];image,p,s,seg,history=load(inputs,case)
    if problem=='target_erased':image[185:273,241:329]=image[185:273,131:219]
    elif problem=='duplicate_in_range':image[185:273,160:248]=image[185:273,241:329]
    elif problem=='noise':image=np.random.default_rng(31).normal(35,1,image.shape).clip(0,255).astype(np.uint8)
    stage=[history['actualStageUm'][axis] for axis in ['x','y']]
    if problem=='wrong_stage':stage[0]+=40
    candidates=build_sem_anchor_candidates(p,image,pixel_size_um=s['pixelSizeUm'],stage_center_um=stage,segmentation=seg)
    chosen,decision,audit=_select_mask_guided_candidate(image,p,s['pixelSizeUm'],candidates,
        stage_center_um=stage,segmentation=seg,calibration_location=prior(history))
    if problem=='target_erased':
        assert audit['status']=='MATCH'
        assert 'calibratedLocationGuard' not in audit
        assert decision.matched and audit['accepted'],audit
        return
    assert not decision.matched and not audit['accepted'],audit
    task,config,scope=_calibrated_task(tmp_path)
    calls=[]
    def saved_rejection(*a,**k):calls.append(True);return chosen,decision,audit
    monkeypatch.setattr(engine,'_select_mask_guided_candidate',saved_rejection)
    analyzer=_FakeSpectrumAnalyzer()
    result=engine.run_cleanliness_analysis(task,config=config,microscope=scope,spectrum_analyzer=analyzer)
    assert calls and not analyzer.calls
    assert result['status']=='COMPLETED_WITH_FAILURES'
    assert all(r['status']=='FAILED' and r['energySpectrum'] is None for r in result['particles'])


def test_location_corroboration_never_replaces_final_measured_contour(inputs,small_replays,monkeypatch):
    from backend.cleanliness_core import cleanliness_task as engine
    case=SMALL[0];image,p,s,seg,h=load(inputs,case)
    original,_,_=small_replays[case['caseId']]
    competitor=replace(original.candidate,center=(468.,124.),
        contour=((460,116),(476,116),(476,132),(460,132)),box=(460,116,476,132))
    wrong=replace(original,candidate=competitor,pixel_xy=(468.,124.))
    monkeypatch.setattr(engine,'build_sem_anchor_candidates',lambda *a,**k:())
    _,decision,audit=engine._select_mask_guided_candidate(image,p,s['pixelSizeUm'],(wrong,),
        stage_center_um=tuple(h['actualStageUm'][axis] for axis in ['x','y']),segmentation=seg,
        calibration_location=prior(h))
    assert audit['calibratedLocation']['accepted']
    assert not decision.matched and not audit['accepted']


def test_unconfirmed_distant_historical_acceptance_cannot_override_calibration(inputs):
    spec=json.loads((FIXTURE/'location_guard_case.json').read_text(encoding='utf-8'))
    for name,key in [('location_guard_sem.png','imageSha256'),('location_guard_optical.json','opticalSha256'),
                     ('location_guard_history.json','historySha256')]:
        assert hashlib.sha256((FIXTURE/name).read_bytes()).hexdigest()==spec[key]
    assert spec['labelKind']=='unconfirmed_identity_geometric_refusal'
    history=json.loads((FIXTURE/'location_guard_history.json').read_text(encoding='utf-8'))
    image=cv2.imdecode(np.frombuffer((FIXTURE/'location_guard_sem.png').read_bytes(),np.uint8),-1)
    particle=load_optical_export(FIXTURE/'location_guard_optical.json').by_sequence[spec['sequence']]
    seg=inputs[1];pixel=CORPUS['sources'][0]['pixelSizeUm']
    stage=tuple(history['actualStageUm'][axis] for axis in ['x','y'])
    candidates=build_sem_anchor_candidates(particle,image,pixel_size_um=pixel,stage_center_um=stage,segmentation=seg)
    _,decision,audit=_select_mask_guided_candidate(image,particle,pixel,candidates,stage_center_um=stage,
                                                   segmentation=seg,calibration_location=prior(history))
    assert audit['status']=='MATCH'
    assert 'calibratedLocationGuard' not in audit
    assert decision.matched and audit['accepted'],audit


@pytest.mark.parametrize('axes',[[[-1.05,.04],[.02,.96]],[[.98,.07],[-.04,1.03]]])
def test_measured_pixel_axes_use_physical_distance_without_assuming_orthogonality(inputs,axes):
    image,p,s,_,h=load(inputs,SMALL[0])
    audit=copy.deepcopy(h['captureAttempts'][0]['maskVerification'])
    stage=tuple(h['actualStageUm'][axis] for axis in ['x','y'])
    result=review_calibrated_location(image.shape,p,s['pixelSizeUm'],audit,
        calibration_location=prior(h),stage_center_um=stage,pixel_axis_matrix=axes,field_rotation_deg=17)
    assert result['accepted'],result
    guard=validate_selected_location(image.shape,p,s['pixelSizeUm'],[285,229],
        calibration_location=prior(h),stage_center_um=stage,pixel_axis_matrix=axes,field_rotation_deg=17,
        measured_contour=((273,213),(298,213),(298,243),(273,243)))
    assert guard['accepted'] and guard['supportSource']=='measured_sem_contour'
    assert guard['distanceToParticleUm']<guard['candidateDistanceUm']
