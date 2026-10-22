"""Cross-check real mask measurements; absence of a competitor is not proof."""
import base64
from dataclasses import replace

import cv2
import numpy as np
import pytest

from backend.cleanliness_core.cleanliness_detection import CandidateSegmentationConfig
from backend.cleanliness_core.optical_particle_analysis import OpticalParticle, ParticleCandidate
from backend.cleanliness_core.shape_consensus import review_ambiguous_shapes


def scene(*, duplicate=False, pixels_duplicate=False):
    mask=np.zeros((56,48),np.uint8)
    cv2.fillPoly(mask,[np.array([[7,5],[39,5],[39,17],[31,17],[36,28],[20,50],[12,36],[12,20],[7,20]])],1)
    other=mask.copy() if duplicate else np.zeros_like(mask)
    if not duplicate:cv2.rectangle(other,(6,12),(40,40),1,-1)
    image=np.full((256,256),25,np.uint8)
    image[50:106,40:88]+=mask*180
    image[165:221,160:208]+=(mask if pixels_duplicate else other)*180
    encoded=base64.b64encode(cv2.imencode('.png',mask*255)[1]).decode()
    particle=OpticalParticle(sequence=1,particle_id='synthetic',label='particle',ab_x_um=0,ab_y_um=0,
        length_um=45,width_um=33,area_um2=float(mask.sum()),equivalent_diameter_um=35,
        image_data_base64=encoded,image_width=48,image_height=56,mask_data_base64=encoded,mask_width=48,mask_height=56)
    candidates=[]
    for binary,offset in [(mask,(40,50)),(other,(160,165))]:
        contour=max(cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea).reshape(-1,2)+offset
        m=cv2.moments(contour);center=(m['m10']/m['m00'],m['m01']/m['m00'])
        lo,hi=contour.min(axis=0),contour.max(axis=0)
        candidates.append(ParticleCandidate(box=(*map(int,lo),*map(int,hi)),center=center,
            contour=tuple(map(tuple,contour.tolist())),distance_px=20,length_ratio=1,width_ratio=1,
            area_ratio=1,contrast_z=10,signal_z=9,channel='yolo',score=1,threshold_z=.25,close_kernel=0))
    audit={'status':'AMBIGUOUS','searchTruncated':False,'distinctHypothesisCount':2,
           'runnerUpOverlapIoU':0.,'score':.95,'centerPixel':[63.5,77.5],
           'runnerUp':{'score':.93,'centerPixel':[183.5,192.5]},'margin':.02}
    return image,particle,candidates,audit


@pytest.fixture
def segmentation(tmp_path):
    model=tmp_path/'model.onnx';model.write_bytes(b'inference is stubbed in these synthetic tests')
    return CandidateSegmentationConfig(backend='yolo',yolo_model_path=model)


def run(monkeypatch,segmentation,*,duplicate=False,pixels_duplicate=False,modify=None,minimum_shape_margin=.07):
    image,particle,candidates,audit=scene(duplicate=duplicate,pixels_duplicate=pixels_duplicate)
    if modify:image,particle,candidates,audit=modify(image,particle,candidates,audit)
    monkeypatch.setattr('backend.cleanliness_core.shape_consensus.yolo_seg_candidates',lambda *a,**k:candidates)
    return review_ambiguous_shapes(
        image,particle,1,audit,segmentation=segmentation,
        minimum_shape_margin=minimum_shape_margin,
    )


def test_two_measurements_can_corroborate_original_winner(monkeypatch,segmentation):
    result=run(monkeypatch,segmentation)
    assert result['corroborated'],result
    assert result['allowsEds'] is False
    assert result['yoloShapeMargin']>=.07 and result['grayscaleShapeMargin']>=.07


def test_configured_margin_also_controls_shape_consensus_override(monkeypatch,segmentation):
    baseline=run(monkeypatch,segmentation)
    threshold=min(baseline['yoloShapeMargin'],baseline['grayscaleShapeMargin'])+.001
    blocked=run(monkeypatch,segmentation,minimum_shape_margin=threshold)
    assert baseline['corroborated']
    assert not blocked['corroborated'] and not blocked['allowsEds']
    assert blocked['minimumShapeMargin']==pytest.approx(threshold)


def test_two_identical_particles_remain_ambiguous(monkeypatch,segmentation):
    result=run(monkeypatch,segmentation,duplicate=True)
    assert not result['corroborated'] and not result['allowsEds']
    assert result['yoloShapeMargin']==pytest.approx(0,abs=1e-6)


def test_grayscale_competition_blocks_apparently_unique_model_shape(monkeypatch,segmentation):
    result=run(monkeypatch,segmentation,pixels_duplicate=True)
    assert not result['corroborated'] and not result['allowsEds'],result


def test_missing_competitor_mask_is_not_used_to_manufacture_a_margin(monkeypatch,segmentation):
    result=run(monkeypatch,segmentation,modify=lambda im,p,cs,a:(im,p,cs[:1],a))
    assert not result['corroborated'] and not result['allowsEds']


def test_model_hallucination_without_grayscale_object_is_rejected(monkeypatch,segmentation):
    def erased(im,p,cs,a):
        im[155:231,150:218]=25
        return im,p,cs,a
    result=run(monkeypatch,segmentation,modify=erased)
    assert not result['corroborated'] and not result['allowsEds']


@pytest.mark.parametrize('changes',[
    {'searchTruncated':True},{'searchTruncated':None},{'distinctHypothesisCount':3},
    {'runnerUp':None},{'runnerUpOverlapIoU':.5},{'score':.3},
    {'centerPixel':[-1,10]},{'centerPixel':[float('nan'),10]},
])
def test_incomplete_or_invalid_template_evidence_never_runs_inference(monkeypatch,segmentation,changes):
    im,p,_,a=scene();a.update(changes)
    def forbidden(*args,**kwargs):raise AssertionError('Inference should not be needed')
    monkeypatch.setattr('backend.cleanliness_core.shape_consensus.yolo_seg_candidates',forbidden)
    result=review_ambiguous_shapes(im,p,1,a,segmentation=segmentation)
    assert not result['corroborated'] and not result['allowsEds']
    assert 'error' not in result


def test_truncated_model_detections_are_not_complete_competition_evidence(monkeypatch,segmentation):
    result=run(monkeypatch,replace(segmentation,maximum_detections=2))
    assert not result['corroborated'] and not result['allowsEds']


@pytest.mark.parametrize('problem',['tiny','outside','empty'])
def test_insufficient_contour_or_bad_coordinate_is_rejected(monkeypatch,segmentation,problem):
    def change(im,p,cs,a):
        points={'tiny':((62,76),(64,76),(64,78),(62,78)),
                'outside':((-3,75),(60,75),(60,95)), 'empty':()}[problem]
        cs[0]=replace(cs[0],contour=points)
        return im,p,cs,a
    result=run(monkeypatch,segmentation,modify=change)
    assert not result['corroborated'] and not result['allowsEds']


@pytest.mark.parametrize('pixel',[0,-1,.1,float('nan')])
def test_bad_physical_scale_never_corroborates(monkeypatch,segmentation,pixel):
    im,p,cs,a=scene()
    monkeypatch.setattr('backend.cleanliness_core.shape_consensus.yolo_seg_candidates',lambda *args,**kwargs:cs)
    result=review_ambiguous_shapes(im,p,pixel,a,segmentation=segmentation)
    assert not result['corroborated'] and not result['allowsEds']


def test_detector_failure_is_explicit_and_fail_closed(monkeypatch,segmentation):
    im,p,_,a=scene()
    def broken(*args,**kwargs):raise RuntimeError('model failed')
    monkeypatch.setattr('backend.cleanliness_core.shape_consensus.yolo_seg_candidates',broken)
    result=review_ambiguous_shapes(im,p,1,a,segmentation=segmentation)
    assert not result['corroborated'] and 'model failed' in result['error']


def test_cancel_is_not_swallowed_as_detector_failure(segmentation):
    im,p,_,a=scene()
    with pytest.raises(InterruptedError):
        review_ambiguous_shapes(im,p,1,a,segmentation=segmentation,should_stop=lambda:True)
