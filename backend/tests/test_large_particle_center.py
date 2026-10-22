"""Oversized particles use explicit center sampling, never pretend to match a contour."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.cleanliness_core.large_particle_center import LargeParticleCenterSettings,plan_large_particle_center
from backend.cleanliness_core.optical_particle_analysis import load_optical_export

FIXTURE=Path(__file__).parent/'fixtures/matching_expansion_20260918'
CORPUS=json.loads((FIXTURE/'cases.json').read_text(encoding='utf-8'))
POLICY_CASES=json.loads((FIXTURE.parent/'large_particle_center_20260918/cases.json').read_text(encoding='utf-8'))
SOURCE=next(s for s in CORPUS['sources'] if s['id']=='sample7')
CASES=[c for c in CORPUS['cases'] if c['caseId'] in POLICY_CASES['cases']]


def real_case(case):
    optical=FIXTURE/SOURCE['optical']
    assert hashlib.sha256(optical.read_bytes()).hexdigest()==SOURCE['opticalSubsetSha256']
    particle=load_optical_export(optical).by_sequence[case['sequence']]
    assert particle.particle_id==case['particleId']
    raw=(FIXTURE/case['image']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==case['imageSha256']
    return cv2.imdecode(np.frombuffer(raw,np.uint8),-1),particle


@pytest.mark.parametrize('case',CASES,ids=lambda c:f"rank{c['rank']}-capture{c['attempt']}")
def test_real_large_particle_center_is_ready_without_identity_claim(case):
    image,particle=real_case(case);pixel=SOURCE['pixelSizeUm']
    assert SOURCE['sourceTaskSha256']==POLICY_CASES['sourceTaskSha256']
    assert pixel==POLICY_CASES['recordedPixelSizeUm']
    result=plan_large_particle_center(image,particle,pixel,field_width_um=POLICY_CASES['recordedFieldWidthUm'])
    assert result['status']=='READY',result
    assert result['pointPx']==[256,256]
    assert result['contourMatchingSkipped'] and result['identityVerified'] is False
    assert result['lengthToFieldWidth']>1


@pytest.mark.parametrize('length',[70.,512.,1024.])
def test_original_length_only_policy_keeps_its_boundary_when_clipped_route_disabled(length):
    image,p=real_case(CASES[0]);p=replace(p,length_um=length,width_um=min(length*.5,p.width_um))
    # v2 adds an independently evidenced clipped-field route. Preserve the old
    # length-only boundary as a compatibility option instead of erasing it.
    settings=LargeParticleCenterSettings(allow_clipped_near_field=False)
    assert plan_large_particle_center(image,p,2,field_width_um=1024,settings=settings)['status']=='NOT_APPLICABLE'


@pytest.mark.parametrize('length',[70.,512.,900.])
def test_ordinary_size_does_not_bypass_matching_even_on_large_bright_image(length):
    image,p=real_case(CASES[0]);p=replace(p,length_um=length,width_um=min(length*.5,p.width_um))
    assert plan_large_particle_center(image,p,2,field_width_um=1024)['status']=='NOT_APPLICABLE'


def test_exact_field_size_requires_visible_clipping_to_use_new_route():
    image,p=real_case(CASES[0]);p=replace(p,length_um=1024,width_um=500)
    assert plan_large_particle_center(image,p,2,field_width_um=1024)['status']=='READY'
    image=np.full_like(image,25);image[80:440,80:440]=210
    assert plan_large_particle_center(image,p,2,field_width_um=1024)['status']=='NOT_APPLICABLE'


def test_disabled_option_keeps_the_original_matching_path():
    image,p=real_case(CASES[0])
    assert plan_large_particle_center(image,p,2,field_width_um=1024,settings=LargeParticleCenterSettings(enabled=False))['status']=='NOT_APPLICABLE'


def test_color_encoded_sem_has_the_same_center_sampling_point():
    image,p=real_case(CASES[0]);color=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    assert plan_large_particle_center(color,p,2,field_width_um=1024)['pointPx']==[256,256]


def test_small_color_encoded_sem_stays_in_normal_matching():
    image,p=real_case(CASES[0]);p=replace(p,length_um=70,width_um=50)
    color=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    assert plan_large_particle_center(color,p,2,field_width_um=1024)['status']=='NOT_APPLICABLE'


@pytest.mark.parametrize('problem',['blank','noise','off_center','central_background','invalid_field','invalid_pixel','nonfinite'])
def test_invalid_center_evidence_cannot_become_a_sampling_point(problem):
    image,p=real_case(CASES[0]);pixel=2.;field=1024.
    if problem=='blank':image=np.zeros_like(image)
    elif problem=='noise':image=np.random.default_rng(9).normal(40,3,image.shape).clip(0,255).astype(np.uint8)
    elif problem=='off_center':
        image=np.full_like(image,25);image[20:150,:]=210
    elif problem=='central_background':image[220:292,220:292]=25
    elif problem=='invalid_field':field=512.
    elif problem=='invalid_pixel':pixel=0.
    elif problem=='nonfinite':image=image.astype(float);image[256,256]=np.nan
    result=plan_large_particle_center(image,p,pixel,field_width_um=field)
    assert result['status']=='REJECTED'
    assert 'pointPx' not in result


def test_center_region_avoids_a_narrow_dark_crack_without_searching_elsewhere():
    image,p=real_case(CASES[0]);image[180:330,256]=0
    result=plan_large_particle_center(image,p,2,field_width_um=1024)
    assert result['status']=='READY',result
    x,y=result['pointPx']
    assert image[y,x]>result['intensityThreshold']
    assert result['pointOffsetFromCenterPx']<=8 and x!=256


@pytest.mark.parametrize('config',[{'enabled':'yes'},{'minimumLengthToFieldWidth':.5},
    {'minimumCenterCoverage':0},{'minimumIntensitySeparation':float('nan')},
    {'centerRegionRadiusFraction':1},{'minimumPointClearancePx':0},False,[]])
def test_settings_reject_invalid_or_unbounded_sampling(config):
    with pytest.raises(ValueError):LargeParticleCenterSettings.from_dict(config)


def large_task(tmp_path,monkeypatch,particle,image):
    from backend.tests import test_cleanliness_task as helpers
    from backend.cleanliness_core.cleanliness_detection import ParticleFilter
    from backend.cleanliness_core.cleanliness_task import CleanlinessTaskConfig,run_cleanliness_calibration,review_cleanliness_calibration
    pixel=SOURCE['pixelSizeUm']
    monkeypatch.setattr(helpers,'PIXEL_SIZE_UM',pixel)
    input_path=helpers._write_export(tmp_path/'optical.json')
    payload=json.loads(input_path.read_text(encoding='utf-8'))
    item=payload['particles'][-1]
    item['particleInfo'].update(length=particle.length_um,width=particle.width_um,area=particle.area_um2,
                                equivalentDiameter=particle.equivalent_diameter_um)
    item.update(particleId=particle.particle_id,imageDataBase64=particle.image_data_base64,
                imageWidth=particle.image_width,imageHeight=particle.image_height,maskDataBase64=particle.mask_data_base64,
                maskWidth=particle.mask_width,maskHeight=particle.mask_height)
    input_path.write_text(json.dumps(payload),encoding='utf-8')
    experience=helpers._write_experience(tmp_path/'experience.json')
    exp=json.loads(experience.read_text(encoding='utf-8'));exp['imaging']['fieldSizeUm']=512*pixel
    experience.write_text(json.dumps(exp),encoding='utf-8')
    settings=helpers._write_settings(tmp_path/'settings.json',experience)
    samples=[dict(sequence=p['sequence'],x=p['abPosition']['x'],y=p['abPosition']['y'],
                  length_px=p['particleInfo']['length']/pixel,width_px=p['particleInfo']['width']/pixel) for p in payload['particles']]
    scope=helpers._FakeMicroscope(samples);original_capture=scope.capture
    def calibrated_capture(path):
        frame,metadata=original_capture(path);metadata['meta']['FieldSize']=512*pixel
        return frame,metadata
    scope.capture=calibrated_capture
    task_dir=tmp_path/'task'
    config=CleanlinessTaskConfig(input_path=input_path,output_dir=task_dir,settings_path=settings,
        particle_filter=ParticleFilter(max_particles=1),anchor_sequences=(1,2,4),validation_sequences=(3,),
        auto_gun_on=False,gun_off_after_task=True)
    calibration=run_cleanliness_calibration(config,microscope=scope)
    assert calibration['status']=='CALIBRATION_PENDING_REVIEW'
    review_cleanliness_calibration(task_dir,accepted=True,reviewer='pytest')
    def captured(path):
        _,metadata=calibrated_capture(path)
        for key in ['rawPath','displayPath']:Path(metadata[key]).write_bytes(cv2.imencode('.png',image)[1].tobytes())
        return image.copy(),metadata
    scope.capture=captured
    return task_dir,config,scope


@pytest.mark.parametrize('case',[c for c in CASES if c['attempt']==1],ids=lambda c:f"rank{c['rank']}")
@pytest.mark.parametrize('background',[False,True])
def test_full_task_large_center_bypasses_matching_and_rejects_background(tmp_path,monkeypatch,case,background):
    from backend.cleanliness_core import cleanliness_task as engine
    from backend.tests.test_cleanliness_task import _FakeSpectrumAnalyzer
    image,p=real_case(case)
    if background:image[:]=25
    task,config,scope=large_task(tmp_path,monkeypatch,p,image)
    def forbidden(*a,**k):raise AssertionError('Large-center mode must not segment or contour-match')
    monkeypatch.setattr(engine,'build_sem_anchor_candidates',forbidden)
    monkeypatch.setattr(engine,'_select_mask_guided_candidate',forbidden)
    class Analyzer(_FakeSpectrumAnalyzer):
        def analyze(self,particle,candidate,image_path,safe_point):
            assert candidate is None
            return super().analyze(particle,candidate,image_path,safe_point)
    analyzer=Analyzer()
    manifest=engine.run_cleanliness_analysis(task,config=config,microscope=scope,spectrum_analyzer=analyzer)
    record=manifest['particles'][0]
    if background:
        assert manifest['status']=='COMPLETED_WITH_FAILURES'
        assert not analyzer.calls and record['energySpectrum'] is None
        assert record['centerSampling']['status']=='REJECTED'
    else:
        assert manifest['status']=='COMPLETED',record
        assert analyzer.calls==[(5,(256,256))]
        assert record['acquisitionMode']=='large_particle_center_region'
        assert record['match']['status']=='BYPASSED_LARGE_PARTICLE_CENTER' and record['match']['best'] is None
        assert record['maskVerification'] is None
        assert record['localization']['observedStageUm'] is None and record['localization']['centerErrorUm'] is None
        assert record['samplingStageUm']==record['actualStageUm']
        assert record['aspect'] is None and record['roundness'] is None
        report=(task/'cleanliness_report.html').read_text(encoding='utf-8')
        assert '中心区域采谱，未做轮廓匹配' in report
        assert '电镜采样点（中心区域）' in report
    assert scope.gun_is_on is False


def test_cancel_after_center_selection_stops_before_eds(tmp_path,monkeypatch):
    from backend.cleanliness_core import cleanliness_task as engine
    from backend.tests.test_cleanliness_task import _FakeSpectrumAnalyzer
    image,p=real_case(CASES[0]);task,config,scope=large_task(tmp_path,monkeypatch,p,image)
    stopped=False
    def progress(index,total,message):
        nonlocal stopped
        if '电子束偏转与 EDS' in message:stopped=True
    analyzer=_FakeSpectrumAnalyzer()
    result=engine.run_cleanliness_analysis(task,config=config,microscope=scope,spectrum_analyzer=analyzer,
        progress=progress,should_stop=lambda:stopped)
    assert stopped and result['status']=='STOPPED'
    assert not analyzer.calls and scope.gun_is_on is False


def test_actual_capture_shape_must_match_metadata_before_center_planning(tmp_path,monkeypatch):
    from backend.cleanliness_core import cleanliness_task as engine
    from backend.tests.test_cleanliness_task import _FakeSpectrumAnalyzer
    image,p=real_case(CASES[0]);task,config,scope=large_task(tmp_path,monkeypatch,p,image)
    capture=scope.capture
    def wrong_shape(path):
        frame,metadata=capture(path)
        return frame[:,:-1],metadata
    scope.capture=wrong_shape
    plans=[]
    def forbidden(*a,**k):plans.append(True);raise AssertionError('Capture dimensions must be checked first')
    monkeypatch.setattr(engine,'plan_large_particle_center',forbidden)
    analyzer=_FakeSpectrumAnalyzer()
    result=engine.run_cleanliness_analysis(task,config=config,microscope=scope,spectrum_analyzer=analyzer)
    assert result['status']=='COMPLETED_WITH_FAILURES'
    assert not plans and not analyzer.calls
    assert '采图数组尺寸与采集元数据不一致' in result['particles'][0]['error']
