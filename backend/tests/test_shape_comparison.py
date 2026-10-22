"""Offline research assertions, including ambiguous and unusable measurements."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.cleanliness_core.shape_comparison import compare_shapes

REAL_FIXTURE=Path(__file__).parent/'fixtures/human_confirmed_20260918'
REAL_CASES=json.loads((REAL_FIXTURE/'cases.json').read_text(encoding='utf-8'))


def target():
    mask=np.zeros((46,40),np.uint8)
    cv2.fillPoly(mask,[np.array([[4,4],[31,4],[31,16],[36,19],[23,40],[13,33],[9,20],[4,20]])],1)
    contour=max(cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea).reshape(-1,2)
    return mask,contour


def test_actual_shape_is_invariant_to_translation_and_mirror():
    mask,contour=target()
    one=compare_shapes(mask,1,1,[contour+[25,30]],(160,160))
    mirrored=contour.copy();mirrored[:,0]=39-mirrored[:,0]
    two=compare_shapes(mask,1,1,[mirrored+[80,60]],(160,160))
    assert one['candidates'][0]['shapeIoU']>.98
    assert two['candidates'][0]['shapeIoU']>.98
    assert one['allowsEds'] is two['allowsEds'] is False
    assert one['scoreMargin'] is None


def test_two_separate_identical_particles_keep_equal_scores():
    mask,contour=target()
    result=compare_shapes(mask,1,1,[contour+[20,25],contour+[95,90]],(180,180))
    assert len(result['candidates'])==2
    assert result['scoreMargin']==pytest.approx(0,abs=1e-6)
    assert np.linalg.norm(np.subtract(*[c['centerPx'] for c in result['candidates']]))>60
    assert result['allowsEds'] is False


@pytest.mark.parametrize('contours,expected',[
    ([], 'NO_TARGET'),
    ([[[10,10],[11,11]]], 'INSUFFICIENT_CONTOUR'),
    ([[[10,10],[12,10],[12,12],[10,12]]], 'INSUFFICIENT_CONTOUR'),
    ([[[0,10],[20,10],[20,40],[0,40]]], 'INSUFFICIENT_CONTOUR'),
])
def test_absent_small_or_partial_observations_are_not_replaced_by_optical_mask(contours,expected):
    mask,_=target()
    result=compare_shapes(mask,1,1,contours,(100,100))
    assert result['status']==expected
    assert not result['candidates'] and result['allowsEds'] is False


@pytest.mark.parametrize('contour',[
    [[-1,10],[20,10],[20,40]], [[100,10],[20,10],[20,40]],
    [[float('nan'),10],[20,10],[20,40]], [[1,2,3],[2,3,4],[3,4,5]],
])
def test_wrong_full_frame_coordinates_raise(contour):
    mask,_=target()
    with pytest.raises(ValueError):
        compare_shapes(mask,1,1,[contour],(100,100))


@pytest.mark.parametrize('pixel',[0,-1,float('nan'),float('inf'),.00001])
def test_invalid_pixel_scale_cannot_create_a_shape_match(pixel):
    mask,contour=target()
    with pytest.raises(ValueError):
        compare_shapes(mask,1,pixel,[contour+[20,20]],(100,100))


def test_incorrect_physical_size_is_not_normalized_away():
    mask,contour=target()
    result=compare_shapes(mask,1,1,[contour*2+[20,20]],(160,160))
    assert result['candidates'][0]['shapeIoU']<.4
    assert result['allowsEds'] is False


def test_cancellation_is_honored():
    mask,contour=target()
    with pytest.raises(InterruptedError):
        compare_shapes(mask,1,1,[contour+[20,20]],(100,100),should_stop=lambda:True)


def test_production_runner_does_not_use_raw_shape_ranking_as_identity():
    # Only the separately verified consensus gate may consume raw shape scores.
    import ast
    source=Path(__file__).resolve().parents[1]/'cleanliness_core/cleanliness_task.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):
            assert not (node.module or '').endswith('shape_comparison')


@pytest.mark.parametrize('case',REAL_CASES['cases'],ids=lambda c:c['caseId'])
def test_real_shape_experiment_ranks_a_above_b_without_authorizing_eds(case):
    import hashlib
    from backend.cleanliness_core.cleanliness_detection import yolo_seg_candidates
    from backend.cleanliness_core.optical_particle_analysis import load_optical_export
    from backend.cleanliness_core.particle_refinement import optical_mask
    root=Path(__file__).resolve().parents[2]
    model=root/REAL_CASES['model']['path']
    assert hashlib.sha256(model.read_bytes()).hexdigest()==REAL_CASES['model']['sha256']
    optical=REAL_FIXTURE/'optical.json'
    assert hashlib.sha256(optical.read_bytes()).hexdigest()==REAL_CASES['opticalSubsetSha256']
    particle=load_optical_export(optical).by_sequence[case['sequence']]
    assert particle.particle_id==case['particleId']
    raw=(REAL_FIXTURE/case['image']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==case['imageSha256']
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED)
    pixel=REAL_CASES['pixelSizeUm']
    observations=yolo_seg_candidates(image,expected_length_px=particle.length_um/pixel,
        expected_width_px=particle.width_um/pixel,expected_area_px=particle.area_um2/pixel**2,model_path=model)
    mask,opx,_=optical_mask(particle)
    result=compare_shapes(mask,opx,pixel,[c.contour for c in observations],image.shape)
    winner=result['candidates'][0]
    assert np.linalg.norm(np.subtract(winner['centerPx'],case['correctTarget']['referenceCenterPx']))<=REAL_CASES['associationTolerancePx']
    assert np.linalg.norm(np.subtract(winner['centerPx'],case['wrongTarget']['referenceCenterPx']))>40
    # Regression bounds for these measured frames, not a production gate.
    assert winner['shapeIoU']>.85
    assert result['scoreMargin']>.07
    assert result['allowsEds'] is False
