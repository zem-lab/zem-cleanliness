"""Offline-only tests: no microscope, RPC, network or model services."""
import base64
import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from backend.cleanliness_core.optical_particle_analysis import OpticalParticle, load_optical_export
from backend.cleanliness_core.particle_refinement import (
    AxisMeasurementError, RefinementSettings, check_linear_change, coverage, match_particle, measure_pixel_axes,
    metric_ab_initial, optical_mask, robust_affine, select_context_seeds, select_plan, validate_frozen, verify_particle_context, xy,
    _natural_template_peaks, _distinct_template_hypotheses,
)
from backend.cleanliness_core.cleanliness_detection import transform_solution_from_dict


def encoded(image):
    ok, raw = cv2.imencode('.png', image)
    assert ok
    return base64.b64encode(raw).decode()


def particle(sequence=1, position=(0, 0)):
    mask = np.zeros((60, 80), np.uint8)
    cv2.fillPoly(mask, [np.array([[9, 22], [61, 15], [71, 24], [43, 35], [22, 46], [18, 30]])], 255)
    image = np.full((*mask.shape, 3), 220, np.uint8)
    image[mask > 0] = 30
    return OpticalParticle(sequence, str(sequence), 'metal', *position, 64., 30., float((mask > 0).sum()),
                           35., encoded(image), 80, 60, mask_data_base64=encoded(mask), mask_width=80, mask_height=60)


def test_mask_roundtrip_and_bad_mask_rejected(tmp_path):
    p = particle()
    mask, pixel, source = optical_mask(p)
    assert pixel == 1 and source == 'customer_mask' and mask.sum() == p.area_um2
    with pytest.raises(ValueError, match='尺寸'):
        optical_mask(replace(p, mask_width=100))
    with pytest.raises(ValueError):
        optical_mask(replace(p, mask_data_base64='broken'))
    _, fallback_pixel, source = optical_mask(replace(p, mask_data_base64=''))
    assert source == 'local_contour' and fallback_pixel > 0
    path = tmp_path / 'export.json'
    path.write_text(json.dumps({'particles': [{'sequence': 1, 'particleInfo': {'area': p.area_um2},
                     'abPosition': {'x': 0, 'y': 0}, 'imageDataBase64': p.image_data_base64,
                     'maskDataBase64': p.mask_data_base64, 'maskWidth': 80, 'maskHeight': 60}]}))
    restored = load_optical_export(path).particles[0]
    assert restored.mask_data_base64 == p.mask_data_base64


@pytest.mark.parametrize('changes', [{'min_inliers': 3}, {'validation_count': 0}, {'max_captures': float('nan')},
                                   {'match_min_score': 1.1}, {'image_size': 123}, {'max_seconds': -1},
                                   {'anchor_count': 8.5}, {'unknownOption': 42}])
def test_settings_fail_closed(changes):
    with pytest.raises(ValueError):
        RefinementSettings.from_dict(changes)


def test_plan_disjoint_deterministic_spatial_and_no_sequence_hardcodes():
    particles = [particle(i + 100, (x * 3000, y * 3000)) for i, (x, y) in enumerate(np.ndindex(7, 7))]
    cfg = RefinementSettings()
    plan = select_plan(particles, cfg)
    assert plan == select_plan(list(reversed(particles)), cfg)
    assert not set(plan['anchors']) & set(plan['validation'])
    assert len(plan['anchors']) == cfg.candidate_count
    mapping = {p.sequence: p for p in particles}
    assert coverage(xy([mapping[s] for s in plan['validation']]), xy(particles)) >= cfg.min_coverage
    with pytest.raises(ValueError, match='共线'):
        select_plan([replace(p, ab_y_um=0) for p in particles], cfg)


def test_strongest_optical_seed_is_not_consumed_by_validation_pool():
    particles = [particle(i + 900, (x * 3000, y * 3000)) for i, (x, y) in enumerate(np.ndindex(7, 7))]
    particles[24] = replace(particles[24], area_um2=particles[24].area_um2 * 10, length_um=300)
    plan = select_plan(particles, RefinementSettings())
    assert plan['anchors'][0] == particles[24].sequence
    assert particles[24].sequence not in plan['validation']


def test_robust_affine_rejects_outlier_without_distorting_other_points():
    rng = np.random.default_rng(22)
    source = rng.uniform(-20000, 20000, (14, 2))
    matrix, translation = np.array([[.99, .015], [.012, -1.01]]), np.array([2300., -8000.])
    target = source @ matrix.T + translation + rng.normal(0, 3, source.shape)
    target[5] += [230, -180]
    solution, audit = robust_affine(source, target, min_inliers=8)
    assert not audit['inliers'][5] and audit['weights'][5] < .2
    assert audit['inlierCount'] == 13
    validation = rng.uniform(-20000, 20000, (20, 2))
    errors = np.linalg.norm(validation @ solution.matrix.T + solution.translation - (validation @ matrix.T + translation), axis=1)
    assert errors.max() < 12
    assert np.linalg.det(solution.matrix) < 0
    with pytest.raises(ValueError, match='共线'):
        robust_affine(np.column_stack((np.arange(10), np.zeros(10))), np.zeros((10, 2)))
    with pytest.raises(ValueError):
        robust_affine(source[:3], target[:3])


def test_linear_guard_rejects_wrong_handedness():
    initial = transform_solution_from_dict({'matrix': np.eye(2).tolist(), 'translation': [0, 0]})
    check_linear_change(initial, initial, .15)
    with pytest.raises(ValueError, match='手性'):
        check_linear_change(replace(initial, matrix=np.diag([1, -1])), initial, .15)


@pytest.mark.parametrize('sign', [-1, 1])
def test_metric_ab_search_preserves_midpoint_handedness_and_does_not_change_input(sign):
    matrix = np.array([[.8, -.6], [.6, .8]]) @ np.diag([.834, sign*.834])
    initial = transform_solution_from_dict({'matrix': matrix.tolist(), 'translation': [3400, -5100]})
    a, b = np.array([1300., 800.]), np.array([42000., -12000.])
    result, audit = metric_ab_initial(initial, a, b)
    assert np.allclose(np.linalg.svd(result.matrix)[1], 1)
    assert np.sign(np.linalg.det(result.matrix)) == sign
    assert np.allclose(result.transform_xy(*((a+b)/2)), initial.transform_xy(*((a+b)/2)))
    assert np.allclose(initial.matrix, matrix) and audit['verified'] is False
    assert np.isclose(np.linalg.norm(np.subtract(result.transform_xy(*a), result.transform_xy(*b))), np.linalg.norm(a-b))


def test_match_unique_no_match_and_ambiguous():
    p = particle()
    mask, _, _ = optical_mask(p)
    cfg = RefinementSettings()
    rng = np.random.default_rng(1)
    background = rng.normal(50, 1, (320, 320)).astype(np.float32)
    unique = background.copy()
    unique[70:130, 80:160] += mask * 180
    result = match_particle(unique, p, 1, cfg)
    assert result['status'] == 'MATCH', result
    assert np.linalg.norm(np.subtract(result['centerPixel'], [119.5, 99.5])) < 2
    ambiguous = unique.copy()
    ambiguous[200:260, 200:280] += mask * 180
    result = match_particle(ambiguous, p, 1, cfg)
    assert result['status'] == 'AMBIGUOUS', result
    assert match_particle(np.zeros((256, 256)), p, 1, cfg)['status'] == 'NO_MATCH'
    assert match_particle(background, p, 1, cfg)['status'] == 'NO_MATCH'
    with pytest.raises(InterruptedError):
        match_particle(unique, p, 1, cfg, should_stop=lambda: True)


def test_parallel_pose_search_is_bit_exact_for_match_competition_and_no_target(monkeypatch):
    from backend.cleanliness_core import particle_refinement as module

    p = particle()
    mask, _, _ = optical_mask(p)
    background = np.random.default_rng(71).normal(50, 1, (240, 240)).astype(np.float32)
    unique = background.copy()
    unique[35:95, 45:125] += mask * 180
    competing = unique.copy()
    competing[140:200, 145:225] += mask * 180
    prior = {"angleDeg": 0.0, "mirrored": False}

    monkeypatch.setattr(module.os, "cpu_count", lambda: 1)
    serial = [match_particle(scene, p, 1, RefinementSettings(), prior=prior)
              for scene in (unique, competing, background)]
    monkeypatch.setattr(module.os, "cpu_count", lambda: 8)
    parallel = [match_particle(scene, p, 1, RefinementSettings(), prior=prior)
                for scene in (unique, competing, background)]

    assert parallel == serial
    assert [result["status"] for result in parallel] == ["MATCH", "AMBIGUOUS", "NO_MATCH"]


def test_candidate_score_margin_controls_close_competitors_without_changing_default():
    p = particle()
    mask, _, _ = optical_mask(p)
    image = np.random.default_rng(11).normal(50, 1, (340, 340)).astype(np.float32)
    image[50:110, 50:130] += mask * 180
    competitor = mask.copy()
    competitor[:, -int(competitor.shape[1] * .3):] = 0
    image[210:270, 210:290] += competitor * 180

    permissive = match_particle(
        image, p, 1, replace(RefinementSettings(), match_min_margin=.05)
    )
    default = match_particle(image, p, 1, RefinementSettings())
    stricter = match_particle(
        image, p, 1, replace(RefinementSettings(), match_min_margin=.08)
    )

    assert permissive["margin"] == pytest.approx(default["margin"], abs=1e-6)
    assert .05 < default["margin"] < .07
    assert permissive["status"] == "MATCH"
    assert default["status"] == stricter["status"] == "AMBIGUOUS"


def test_template_score_threshold_rejects_weak_match_and_default_stays_065():
    p = particle()
    mask, _, _ = optical_mask(p)
    image = np.random.default_rng(19).normal(40, 18, (320, 320)).astype(np.float32)
    partial = mask.copy()
    partial[:, : partial.shape[1] // 2] = 0
    image[70:130, 80:160] += partial * 52
    default = match_particle(image, p, 1, RefinementSettings())
    lowered = match_particle(image, p, 1, replace(RefinementSettings(), match_min_score=.5))
    assert default["score"] == pytest.approx(lowered["score"], abs=1e-6)
    assert 0.5 <= default["score"] < 0.65
    assert default["status"] == "NO_MATCH"
    assert lowered["status"] != "NO_MATCH"


def test_peak_extraction_does_not_turn_one_peak_shoulder_into_runner_up():
    yy, xx = np.mgrid[:81, :121]
    scores = np.exp(-((xx-60)**2/800 + (yy-40)**2/20)).astype(np.float32)
    peaks, truncated = _natural_template_peaks(scores, .58)
    assert not truncated
    assert len(peaks) == 1
    assert peaks[0][1] == (60,40)
    # The old circle suppression would create a high-scoring shoulder here.
    old = scores.copy(); cv2.circle(old,(60,40),8,-1.,-1)
    assert cv2.minMaxLoc(old)[1] > .8


def test_footprint_dedup_keeps_distinct_targets_and_avoids_overlap_chains():
    def peak(x,score):
        return {'score':score,'_box':(x,0,x+100,10),'_support':np.ones((10,100),bool),'_area':1000}
    a,b,c,d=peak(0,1.),peak(25,.95),peak(50,.9),peak(200,.85)
    result=_distinct_template_hypotheses([d,c,b,a])
    assert [p['_box'][0] for p in result] == [0,50,200]


def test_nearby_disjoint_particles_are_not_merged_by_center_radius():
    a={'score':.9,'centerPixel':[2,2],'_box':(0,0,5,5),'_support':np.ones((5,5),bool),'_area':25}
    b={'score':.85,'centerPixel':[8,2],'_box':(6,0,11,5),'_support':np.ones((5,5),bool),'_area':25}
    assert len(_distinct_template_hypotheses([a,b],center_radius=8))==2


def test_peak_limit_is_reported_instead_of_silently_dropping_competitors():
    scores=np.zeros((41,41),np.float32);scores[::4,::4]=.9
    peaks,truncated=_natural_template_peaks(scores,.58)
    assert len(peaks)==64 and truncated


def test_fine_angle_search_recovers_long_thin_target_without_relaxing_threshold():
    mask=np.zeros((220,80),np.uint8)
    cv2.polylines(mask,[np.array([[20,15],[55,70],[25,145],[45,200]])],False,255,5)
    p=replace(particle(),length_um=200,width_um=40,area_um2=float((mask>0).sum()),
              mask_data_base64=encoded(mask),mask_width=80,mask_height=220,image_width=80,image_height=220)
    scene=np.random.default_rng(5).normal(50,1,(400,400)).astype(np.float32)
    warp=cv2.getRotationMatrix2D((39.5,109.5),8,1)
    warp[:,2]+=np.array([160.,90.])
    scene+=cv2.warpAffine((mask>0).astype(np.float32),warp,(400,400))*180
    result=match_particle(scene,p,1,RefinementSettings())
    assert result['status']=='MATCH',result
    assert abs(result['angleDeg']-8)<=1
    assert result['score']>=.65 and result['margin']>=.07
    assert np.linalg.norm(np.subtract(result['centerPixel'],[199.5,199.5]))<3
    json.dumps(result)  # Internal footprint arrays must never enter task JSON.


def context_scene():
    seed = particle(9001)
    neighbors = [particle(9002, (-130, -90)), particle(9003, (130, 90))]
    masks = []
    first = np.zeros((60, 80), np.uint8)
    cv2.fillPoly(first, [np.array([[12, 12], [66, 12], [66, 23], [27, 23], [27, 49], [12, 49]])], 255)
    masks.append(first)
    second = np.zeros((60, 80), np.uint8)
    cv2.fillPoly(second, [np.array([[12, 11], [65, 17], [50, 49], [17, 39]])], 255)
    masks.append(second)
    neighbors = [replace(p, mask_data_base64=encoded(m), area_um2=float((m > 0).sum())) for p, m in zip(neighbors, masks)]
    scene = np.random.default_rng(9).normal(50, 1, (512, 512)).astype(np.float32)
    for p in [seed, *neighbors]:
        mask, _, _ = optical_mask(p)
        x, y = np.rint([256 + p.ab_x_um - 39.5, 256 + p.ab_y_um - 29.5]).astype(int)
        scene[y:y+60, x:x+80] += mask * 180
    return seed, neighbors, scene


def test_same_frame_context_uses_relative_geometry_and_excludes_validation():
    seed, neighbors, scene = context_scene()
    cfg = replace(RefinementSettings(), context_error_um=10)
    match = match_particle(scene, seed, 1, cfg)
    assert match['status'] == 'MATCH', match
    result = verify_particle_context(scene, seed, match, [seed, *neighbors], np.eye(2), np.eye(2), 1, cfg)
    assert result['status'] == 'VERIFIED' and result['confirmedCount'] == 2, result
    assert all(r['relativeErrorUm'] < 2 for r in result['neighbors'])
    excluded = verify_particle_context(scene, seed, match, neighbors, np.eye(2), np.eye(2), 1, cfg,
                                       excluded_sequences=[neighbors[0].sequence])
    assert excluded['status'] == 'UNVERIFIED'
    assert neighbors[0].sequence not in [r['sequence'] for r in excluded['neighbors']]
    # A perfect seed at the wrong location/with wrong neighbors is NOT navigation evidence.
    wrong = dict(match, centerPixel=(np.array(match['centerPixel']) + [30, 0]).tolist(), score=.999)
    rejected = verify_particle_context(scene, seed, wrong, neighbors, np.eye(2), np.eye(2), 1, cfg)
    assert rejected['status'] == 'UNVERIFIED', rejected
    with pytest.raises(InterruptedError):
        verify_particle_context(scene, seed, match, neighbors, np.eye(2), np.eye(2), 1, cfg, should_stop=lambda: True)


def neighborhood_seed_fixture():
    seed, neighbors, _ = context_scene()
    candidates, population = [], []
    for i, (x, y) in enumerate([(0, 0), (4000, 0), (0, 4000), (4000, 4000), (2000, 6000), (-2000, 0)]):
        center = replace(seed, sequence=1000+i*3, ab_x_um=x, ab_y_um=y)
        candidates.append(center)
        population.append(center)
        population.extend(replace(p, sequence=center.sequence+j+1, ab_x_um=x+p.ab_x_um, ab_y_um=y+p.ab_y_um)
                          for j, p in enumerate(neighbors))
    isolated = replace(seed, sequence=999, ab_x_um=20000, ab_y_um=20000, area_um2=60000, length_um=800)
    return [isolated, *candidates], [isolated, *population]


def test_context_seed_plan_prefers_testable_clusters_over_large_isolated_particle():
    candidates, population = neighborhood_seed_fixture()
    cfg = replace(RefinementSettings(), image_size=512, context_error_um=10, seed_attempts=4)
    selected, audit = select_context_seeds(candidates, population, np.eye(2), np.eye(2), 1, cfg)
    again, _ = select_context_seeds(candidates[::-1], population[::-1], np.eye(2), np.eye(2), 1, cfg)
    assert [p.sequence for p in selected] == [p.sequence for p in again] == audit['sequences']
    assert len(selected) == 4 and 999 not in audit['sequences']
    assert audit['candidates'][0]['status'] == 'INSUFFICIENT_CENTERED_CONTEXT'
    assert coverage(xy(selected), xy(candidates[1:])) > .5


def test_context_seed_plan_excludes_reserved_masks_and_fails_with_too_few_clusters():
    candidates, population = neighborhood_seed_fixture()
    cfg = replace(RefinementSettings(), image_size=512, context_error_um=10, seed_attempts=4)
    reserved = 1001
    population = [replace(p, mask_data_base64='invalid and must never be decoded') if p.sequence == reserved else p for p in population]
    selected, audit = select_context_seeds(candidates, population, np.eye(2), np.eye(2), 1, cfg, excluded_sequences=[reserved])
    assert 1000 not in [p.sequence for p in selected]
    assert all(reserved not in r['centeredNeighborSequences'] for r in audit['candidates'])
    selected, audit = select_context_seeds(candidates[1:4], population, np.eye(2), np.eye(2), 1, cfg, excluded_sequences=[reserved])
    assert selected == [] and '少于 4' in audit['reason']
    with pytest.raises(InterruptedError):
        select_context_seeds(candidates, population, np.eye(2), np.eye(2), 1, cfg, should_stop=lambda: True)


def test_context_does_not_count_the_same_sem_object_twice(monkeypatch):
    from backend.cleanliness_core import particle_refinement as module
    seed, neighbors, scene = context_scene()
    cfg = replace(RefinementSettings(), context_error_um=80)
    match = {'status': 'MATCH', 'centerPixel': [256, 256], 'score': .99}
    # Two optical neighbors claim a single SEM object. Even if both errors fit
    # the spatial tolerance, they cannot count as two pieces of evidence.
    neighbors = [replace(p, ab_x_um=110+i*10, ab_y_um=0) for i, p in enumerate(neighbors)]
    monkeypatch.setattr(module, 'match_particle', lambda *args, **kwargs:
                        {'status': 'MATCH', 'centerPixel': [366, 256], 'score': .99})
    result = verify_particle_context(scene, seed, match, neighbors, np.eye(2), np.eye(2), 1, cfg)
    assert result['status'] == 'UNVERIFIED' and result['confirmedCount'] == 1


def test_axis_sign_coupling_repeated_origin_and_quality_gate():
    cfg = RefinementSettings()
    rng = np.random.default_rng(4)
    base = cv2.GaussianBlur(rng.normal(500, 200, (512, 512)).astype(np.float32), (0, 0), 2)
    matrix = np.array([[-.98, .22], [.23, .96]])
    positions = {'o1': [0, 0], 'xp': [75, 0], 'xm': [-75, 0], 'yp': [0, 75], 'ym': [0, -75], 'o2': [0, 0]}
    images = {}
    for key, stage in positions.items():
        shift = -np.linalg.solve(matrix, stage)
        images[key] = cv2.warpAffine(base, np.column_stack((np.eye(2), shift)), (512, 512), borderMode=cv2.BORDER_REFLECT)
    result = measure_pixel_axes(images, positions, 1, cfg)
    assert np.allclose(result['pixelToStageMatrix'], matrix, atol=.006)
    assert result['returnErrorUm'] < .1
    images['o2'] = np.roll(images['o2'], 30, axis=0)
    with pytest.raises(ValueError, match='回原点'):
        measure_pixel_axes(images, positions, 1, cfg)
    with pytest.raises(ValueError, match='空白'):
        measure_pixel_axes({k: np.zeros((512, 512)) for k in positions}, positions, 1, cfg)


def test_return_image_contradiction_not_hidden_by_earlier_weak_phase(monkeypatch):
    cfg = RefinementSettings()
    base = np.random.default_rng(55).normal(50, 10, (128, 128)).astype(np.float32)
    positions = {'o1': [0, 0], 'xp': [20, 0], 'xm': [-20, 0], 'yp': [0, 20], 'ym': [0, -20], 'o2': [0, 0]}
    phases = iter([((-20, 0), .9), ((20, 0), .9), ((0, -20), .9), ((0, 20), .08), ((-37, 350), .23)])
    monkeypatch.setattr(cv2, 'phaseCorrelate', lambda *args: next(phases))
    with pytest.raises(AxisMeasurementError, match='回原点图像与台坐标不一致') as error:
        measure_pixel_axes({n: base for n in positions}, positions, 1., cfg)
    assert not error.value.retryable
    assert error.value.diagnostics['unreliableFrames'] == ['ym']
    assert len(error.value.diagnostics['shifts']) == 5
    assert error.value.diagnostics['repeatImageShiftPx'] > 350


def test_wrong_pixel_metadata_not_misreported_as_sample_return_motion(monkeypatch):
    cfg = RefinementSettings()
    base = np.random.default_rng(55).normal(50, 10, (128, 128)).astype(np.float32)
    positions = {'o1': [0, 0], 'xp': [150, 0], 'xm': [-150, 0], 'yp': [0, 150], 'ym': [0, -150], 'o2': [0, 0]}
    phases = iter([((-100, 0), .9), ((100, 0), .9), ((0, -100), .9), ((0, 100), .9), ((1.45, 0), .9)])
    monkeypatch.setattr(cv2, 'phaseCorrelate', lambda *args: next(phases))
    with pytest.raises(AxisMeasurementError, match='图像尺度与实测台位移不一致') as error:
        measure_pixel_axes({n: base for n in positions}, positions, 6.2, cfg)
    assert not error.value.retryable
    assert np.isclose(error.value.diagnostics['measuredPixelSizeUm'], 1.5)
    assert error.value.diagnostics['returnErrorUm'] < cfg.axis_max_return_um


def test_validation_requires_every_reserved_point_and_never_changes_fit():
    solution = transform_solution_from_dict({'matrix': np.eye(2).tolist(), 'translation': [0, 0]})
    cfg = replace(RefinementSettings(), validation_count=4)
    points = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]])
    records = [{'sequence': i + 10, 'status': 'MATCH', 'opticalAbUm': point.tolist(),
                'measuredStageUm': (point + [10, 0]).tolist()} for i, point in enumerate(points)]
    ids = [r['sequence'] for r in records]
    gate = validate_frozen(solution, [{'sequence': 1}], records, ids, points, cfg)
    assert gate['withinLimits'] and gate['meanErrorUm'] == 10
    assert not validate_frozen(solution, [], records[:-1], ids, points, cfg)['withinLimits']
    assert not validate_frozen(solution, [{'sequence': 10}], records, ids, points, cfg)['withinLimits']
    records[-1]['status'] = 'AMBIGUOUS'
    assert not validate_frozen(solution, [], records, ids, points, cfg)['withinLimits']
    assert np.array_equal(solution.matrix, np.eye(2))
