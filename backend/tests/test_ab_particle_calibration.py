import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.cleanliness_core import ab_particle_calibration as job
from backend.cleanliness_core import cleanliness_task
from backend.cleanliness_core.cleanliness_task import CleanlinessTaskConfig, review_cleanliness_calibration, run_cleanliness_analysis
from backend.cleanliness_core.optical_particle_analysis import load_calibration, load_optical_export
from backend.tests.test_particle_refinement import particle


class OfflineMicroscope:
    """No RPC available. Fixed physical ground truth; lifecycle only, matcher tested separately."""
    def __init__(self):
        self.imaging = {}
        self.position = np.array([0., 0.])
        self.moves, self.connections, self.frames = [], [], []
        self.pixel_avg = 1
        self.restored = False
        self.fail_capture = False

    def connect(self):
        self.connections.append(dict(self.imaging))
        return {'magnification': self.imaging.get('magnification', 120), 'status': 'Idle'}

    def get_position(self):
        return tuple(self.position)

    def move_xy(self, x, y):
        self.position = np.array([x, y])
        self.moves.append(self.position.copy())
        return {'after': {'x': x, 'y': y}, 'positionErrorUm': 0.}

    def capture(self, path):
        if self.fail_capture:
            raise RuntimeError('offline injected capture failure')
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.zeros((512, 512), np.uint16)
        cv2.imwrite(str(path), image)
        display = path.with_suffix('.u8.png')
        cv2.imwrite(str(display), image.astype(np.uint8))
        self.frames.append((str(path), self.get_position()))
        pixel = 4. if self.imaging['magnification'] == 60 else 2.
        return image, {'rawPath': str(path), 'displayPath': str(display), 'pixelSizeUm': pixel,
                       'shape': [512, 512], 'meta': {'Magnification': self.imaging['magnification'],
                       'FieldSize': pixel * 512, 'Detector': 'BSE', 'FieldRotation': 13., 'KV': 15.}}

    def restore_video(self):
        self.restored = True

    def gun_on(self):
        raise AssertionError('No automatic gun change configured')

    gun_off = gun_on


def prepare(tmp_path, monkeypatch):
    optical = []
    for i, (x, y) in enumerate(np.ndindex(5, 5)):
        p = particle(i + 101, (x * 4000, y * 4000))
        optical.append({'sequence': p.sequence, 'particleInfo': {'area': p.area_um2, 'length': p.length_um, 'width': p.width_um},
                        'imageDataBase64': p.image_data_base64, 'imageWidth': 80, 'imageHeight': 60,
                        'maskDataBase64': p.mask_data_base64, 'maskWidth': 80, 'maskHeight': 60,
                        'abPosition': {'x': p.ab_x_um, 'y': p.ab_y_um}})
    input_path = tmp_path / 'optical.json'
    input_path.write_text(json.dumps({'coordinateUnit': 'um', 'coordinateSpace': 'sourceStage', 'registrationMode': 'twoPointAB', 'particles': optical}))
    roi_path = tmp_path / 'roi.json'
    roi_path.write_text(json.dumps({'device': {'host': 'offline', 'port': 1},
                                  'transform': {'matrix': [[1, 0], [0, 1]], 'translation': [0, 0]},
                                  'validStageRoi': {'centerUm': [0, 0], 'radiusUm': 100000}}))
    settings_path = tmp_path / 'settings.json'
    settings_path.write_text(json.dumps({'schemaVersion': 1, 'experienceCalibration': str(roi_path),
                          'abParticle': {'image_size': 512, 'stage_settle_seconds': .001, 'anchor_count': 8, 'min_inliers': 6,
                                         'candidate_count': 12, 'validation_count': 4, 'pool_size': 25},
                          'calibration': {'candidateSegmentation': {'backend': 'classic'}},
                          'task': {'gunOnAtStart': False, 'gunOffAfterTask': False, 'edsEnabled': False}}))
    export = load_optical_export(input_path)
    manifest = {'taskId': 'offline-job', 'status': 'AB_POINTS_PENDING_SEM_REFINEMENT',
                'input': {'path': str(input_path), 'sha256': export.sha256}, 'selection': {'sequences': [p.sequence for p in export.particles]},
                'device': {'microscope': {'host': 'offline', 'port': 1}},
                'calibration': {'strategy': 'ab_particle', 'groups': [], 'abPointResult': {
                    'previewTransform': {'matrix': [[1, 0], [0, 1]], 'translation': [0, 0]},
                    'sourceReferencePointsUm': {'a': [0, 0], 'b': [20000, 0]},
                    'assignment': {'a': {'coarseStageUm': [0, 0]}, 'b': {'coarseStageUm': [20000, 0]}}}},
                'particles': []}
    (tmp_path / 'task.json').write_text(json.dumps(manifest))
    scope = OfflineMicroscope()
    def axes(_images, _positions, pixel, _cfg):
        return {'pixelToStageMatrix': (np.eye(2) * pixel).tolist(), 'pixelSizeUm': pixel, 'returnErrorUm': .2}
    monkeypatch.setattr(job, 'measure_pixel_axes', axes)
    def match(_image, p, pixel, _cfg, **_kwargs):
        measured = np.array([p.ab_x_um, p.ab_y_um]) + [200., -150.]
        center = (measured - scope.position) / pixel + [256, 256]
        return {'status': 'MATCH', 'centerPixel': center.tolist(), 'score': .9, 'margin': .2,
                'angleDeg': 0., 'mirrored': False, 'scaleFactor': 1.}
    monkeypatch.setattr(job, 'match_particle', match)
    monkeypatch.setattr(job, 'verify_particle_context', lambda *args, **kwargs:
                        {'status': 'VERIFIED', 'confirmedCount': 2, 'requiredCount': 2})
    # Lifecycle fixture has a sparse 4 mm grid; physical neighborhood selection
    # is tested separately with real optical constellations.
    monkeypatch.setattr(job, 'select_context_seeds', lambda candidates, _population, _matrix, _axis, _pixel, cfg, **kwargs:
                        (list(candidates)[:cfg.seed_attempts], {'method': 'offline_lifecycle_fixture', 'sequences': [p.sequence for p in candidates[:cfg.seed_attempts]]}))
    config = CleanlinessTaskConfig(input_path=input_path, output_dir=tmp_path, settings_path=settings_path,
                                    experience_calibration_path=roi_path, microscope_host='offline', microscope_port=1,
                                    calibration_strategy='ab_particle', auto_gun_on=False, gun_off_after_task=False)
    return config, scope


def test_full_offline_lifecycle_frozen_validation_and_analysis_binding(tmp_path, monkeypatch):
    config, scope = prepare(tmp_path, monkeypatch)
    events = []
    result = job.run_ab_particle_calibration(
        tmp_path,
        config=config,
        microscope=scope,
        event_logger=lambda event, fields: events.append((event, fields)),
    )
    audit = result['calibration']['abPointResult']['semRefinement']
    assert audit['status'] == 'SUCCEEDED', audit.get('error')
    assert result['status'] == 'CALIBRATION_PENDING_REVIEW'
    assert np.allclose(result['calibration']['result']['translation'], [200, -150])
    assert len(audit['validation']) == 4 and audit['acceptance']['withinLimits']
    assert not set(audit['plan']['validation']) & {r['sequence'] for r in audit['anchors']}
    assert all(len(r['attempts']) == 1 and r['role'] == 'validation' for r in audit['validation'])
    assert all(c['phase'] == 'fine' for c in audit['captures'][-4:])
    frozen = json.loads((tmp_path / audit['runDirectory'] / 'frozen_transform.json').read_text())
    assert frozen['matrix'] == result['calibration']['result']['matrix']
    assert scope.restored and np.array_equal(scope.position, [0, 0])
    assert result['calibration']['pixelAxisIncludesRotation']
    assert np.allclose(result['calibration']['pixelAxisMatrix'], np.eye(2))
    for stage in ('calibration', 'preflight', 'connect', 'imaging_configure', 'stage_move', 'capture', 'particle_match'):
        stage_events = [item for item in events if item[1]['stage'] == stage]
        assert stage_events, stage
        assert stage_events[0][0] == 'calibration_stage_started'
        assert stage_events[-1][0] == 'calibration_stage_finished'
        assert all(item[1].get('elapsedSeconds', 0) >= 0 for item in stage_events if item[0].endswith('finished'))
    calibration = load_calibration(Path(result['experienceCalibration']['path']))
    assert calibration.imaging['fieldRotationDeg'] == 13
    review_cleanliness_calibration(tmp_path, accepted=True)
    seen = []
    def no_candidates(_particle, _image, **kwargs):
        seen.append(kwargs)
        return ()
    monkeypatch.setattr(cleanliness_task, 'build_sem_anchor_candidates', no_candidates)
    from backend.cleanliness_core import cleanliness_report
    monkeypatch.setattr(cleanliness_report, 'build_cleanliness_reports', lambda *args, **kwargs: {})
    analyzed = run_cleanliness_analysis(tmp_path, config=replace(config, selected_sequences=(101,), max_capture_attempts=1), microscope=scope)
    assert seen and seen[0]['field_rotation_deg'] == 0
    assert np.allclose(seen[0]['pixel_axis_matrix'], np.eye(2))
    assert np.allclose(list(analyzed['particles'][0]['targetStageUm'].values()), [200, -150])
    # Restore review state for checking initial-analysis input invalidation below.
    analyzed['status'] = 'CALIBRATION_ACCEPTED'
    (tmp_path / 'task.json').write_text(json.dumps(analyzed))
    # Input edits must invalidate both first analysis and resume BEFORE any device call.
    config.input_path.write_text(config.input_path.read_text() + ' ')
    with pytest.raises(ValueError, match='JSON'):
        run_cleanliness_analysis(tmp_path, config=config, microscope=object())


def test_context_unverified_high_score_never_updates_navigation(tmp_path, monkeypatch):
    config, scope = prepare(tmp_path, monkeypatch)
    seen = []
    def unverified(*args, **kwargs):
        seen.append(kwargs['excluded_sequences'])
        return {'status': 'UNVERIFIED', 'confirmedCount': 0, 'reason': 'offline context rejection'}
    monkeypatch.setattr(job, 'verify_particle_context', unverified)
    result = job.run_ab_particle_calibration(tmp_path, config=config, microscope=scope)
    audit = result['calibration']['abPointResult']['semRefinement']
    assert audit['status'] == 'FAILED'
    assert 'bootstrap' not in audit and not audit['anchors'] and not audit['validation']
    assert all(r['status'] == 'CONTEXT_UNVERIFIED' for r in audit['observations'])
    assert all(ids == audit['plan']['validation'] for ids in seen)
    assert all(len(r['attempts']) == audit['settings']['max_search_captures'] for r in audit['observations'])
    assert result['calibration']['refinedTransform'] is None
    assert scope.restored and np.array_equal(scope.position, [0, 0])


def test_failed_neighbor_context_continues_fixed_grid_without_recentering(tmp_path, monkeypatch):
    config, scope = prepare(tmp_path, monkeypatch)
    calls = 0
    def context(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {'status': 'UNVERIFIED' if calls == 1 else 'VERIFIED',
                'confirmedCount': 0 if calls == 1 else 2, 'requiredCount': 2,
                'reason': 'first candidate lacks corroboration' if calls == 1 else ''}
    monkeypatch.setattr(job, 'verify_particle_context', context)
    result = job.run_ab_particle_calibration(tmp_path, config=config, microscope=scope)
    audit = result['calibration']['abPointResult']['semRefinement']
    assert audit['status'] == 'SUCCEEDED', audit.get('error')
    seed = audit['observations'][0]
    assert [a['status'] for a in seed['attempts']] == ['CONTEXT_UNVERIFIED', 'MATCH']
    shots = {s['index']: s for s in audit['captures']}
    first, second = [shots[a['captureIndex']] for a in seed['attempts']]
    step = audit['settings']['search_step_ratio'] * 4 * 512
    assert np.allclose(np.subtract(second['targetUm'], first['targetUm']), [0, -step])
    assert np.allclose(list(seed['targetStageUm'].values()), first['targetUm'])
    assert all(len(r['attempts']) == 1 for r in audit['validation'])


def test_ambiguous_primary_still_stops_grid_without_hiding_competitor(tmp_path, monkeypatch):
    config, scope = prepare(tmp_path, monkeypatch)
    nominal = job.match_particle
    def ambiguous(*args, **kwargs):
        return {**nominal(*args, **kwargs), 'status': 'AMBIGUOUS'}
    monkeypatch.setattr(job, 'match_particle', ambiguous)
    result = job.run_ab_particle_calibration(tmp_path, config=config, microscope=scope)
    audit = result['calibration']['abPointResult']['semRefinement']
    assert audit['status'] == 'FAILED'
    assert all(len(r['attempts']) == 1 and r['status'] == 'AMBIGUOUS' for r in audit['observations'])
    assert 'bootstrap' not in audit and not audit['validation']


def test_later_blank_frame_does_not_inherit_rejected_candidate_metadata(tmp_path, monkeypatch):
    config, scope = prepare(tmp_path, monkeypatch)
    settings = json.loads(config.settings_path.read_text())
    settings['abParticle']['max_search_captures'] = 3
    config.settings_path.write_text(json.dumps(settings))
    nominal = job.match_particle
    calls = 0
    def match(*args, **kwargs):
        nonlocal calls
        calls += 1
        return nominal(*args, **kwargs) if calls == 1 else {'status': 'NO_MATCH', 'reason': 'blank frame'}
    monkeypatch.setattr(job, 'match_particle', match)
    monkeypatch.setattr(job, 'verify_particle_context', lambda *args, **kwargs:
                        {'status': 'UNVERIFIED', 'confirmedCount': 0, 'reason': 'wrong constellation'})
    result = job.run_ab_particle_calibration(tmp_path, config=config, microscope=scope)
    audit = result['calibration']['abPointResult']['semRefinement']
    record = audit['observations'][0]
    assert record['status'] == 'NO_MATCH'
    assert record['attempts'][0]['status'] == 'CONTEXT_UNVERIFIED'
    assert record['attempts'][0]['contextVerification']['confirmedCount'] == 0
    assert not {'contextVerification', 'measuredStageUm', 'predictionErrorUm', 'centerPixel', 'score'} & record.keys()
    assert 'bootstrap' not in audit


def test_return_image_contradiction_stops_without_axis_site_retry(tmp_path, monkeypatch):
    from backend.cleanliness_core.particle_refinement import AxisMeasurementError
    config, scope = prepare(tmp_path, monkeypatch)
    def inconsistent(*args):
        raise AxisMeasurementError('回原点图像与台坐标不一致', {'repeatImageShiftPx': 352}, retryable=False)
    monkeypatch.setattr(job, 'measure_pixel_axes', inconsistent)
    result = job.run_ab_particle_calibration(tmp_path, config=config, microscope=scope)
    audit = result['calibration']['abPointResult']['semRefinement']
    assert audit['status'] == 'FAILED' and len(audit['captures']) == 6
    assert len(audit['axisFailureDetails']) == 1
    assert not audit['axisFailureDetails'][0]['retryable']
    assert not audit['observations'] and scope.restored


def test_cancel_preflight_never_connects(tmp_path, monkeypatch):
    config, scope = prepare(tmp_path, monkeypatch)
    with pytest.raises(InterruptedError):
        job.run_ab_particle_calibration(tmp_path, config=config, microscope=scope, should_stop=lambda: True)
    assert not scope.connections and not scope.moves


def test_capture_failure_and_retry_audit_preserved(tmp_path, monkeypatch):
    config, scope = prepare(tmp_path, monkeypatch)
    scope.fail_capture = True
    result = job.run_ab_particle_calibration(tmp_path, config=config, microscope=scope)
    assert result['status'] == 'AB_POINTS_PENDING_SEM_REFINEMENT'
    audit = result['calibration']['abPointResult']['semRefinement']
    assert audit['status'] == 'FAILED' and 'injected' in audit['error']
    assert scope.restored and np.array_equal(scope.position, [0, 0])
    scope.fail_capture = False
    result = job.run_ab_particle_calibration(tmp_path, config=config, microscope=scope)
    ab = result['calibration']['abPointResult']
    assert ab['semRefinementHistory'][0]['status'] == 'FAILED'
    assert ab['semRefinementHistory'][0]['runDirectory'] != ab['semRefinement']['runDirectory']
    assert ab['semRefinement']['status'] == 'SUCCEEDED'


def test_independent_validation_ambiguity_blocks_review(tmp_path, monkeypatch):
    config, scope = prepare(tmp_path, monkeypatch)
    original = job.match_particle
    validation_ids = set(job.select_plan(load_optical_export(config.input_path).particles,
                                         job.RefinementSettings.from_dict(json.loads(config.settings_path.read_text())['abParticle']))['validation'])
    def ambiguous(image, p, pixel, cfg, **kwargs):
        result = original(image, p, pixel, cfg, **kwargs)
        if p.sequence in validation_ids:
            result['status'] = 'AMBIGUOUS'
        return result
    monkeypatch.setattr(job, 'match_particle', ambiguous)
    result = job.run_ab_particle_calibration(tmp_path, config=config, microscope=scope)
    assert result['calibration']['abPointResult']['semRefinement']['status'] == 'VALIDATION_FAILED'
    assert not result['calibration']['result']['acceptance']['withinLimits']
    with pytest.raises(ValueError, match='独立数值验收'):
        review_cleanliness_calibration(tmp_path, accepted=True)


def test_capture_budget_and_cancel_mid_run_restore(tmp_path, monkeypatch):
    config, scope = prepare(tmp_path, monkeypatch)
    settings = json.loads(config.settings_path.read_text())
    settings['abParticle']['max_captures'] = 3
    config.settings_path.write_text(json.dumps(settings))
    result = job.run_ab_particle_calibration(tmp_path, config=config, microscope=scope)
    audit = result['calibration']['abPointResult']['semRefinement']
    assert audit['status'] == 'FAILED' and len(audit['captures']) == 3
    assert '上限' in audit['error'] and scope.restored
    result = job.run_ab_particle_calibration(tmp_path, config=config, microscope=scope, should_stop=lambda: len(scope.frames) > 3)
    assert result['calibration']['abPointResult']['semRefinement']['status'] == 'STOPPED'
    assert np.array_equal(scope.position, [0, 0])


def test_wrong_device_rejected_before_connect(tmp_path, monkeypatch):
    config, scope = prepare(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='设备'):
        job.run_ab_particle_calibration(tmp_path, config=replace(config, microscope_host='other'), microscope=scope)
    assert not scope.connections
    saved = json.loads((tmp_path / 'task.json').read_text(encoding='utf-8'))
    assert saved['calibration']['abPointResult']['semRefinement']['phase'] == 'preflight'
    assert not saved['calibration']['abPointResult']['semRefinement']['deviceActionsPerformed']


def test_stale_pixel_metadata_rejected_after_magnification_change():
    capture = {'pixelSizeUm': 1., 'shape': [1024, 1024], 'meta': {'FieldSize': 1024},
               'bounds': [-1024, -1024, 1024, 1024]}
    with pytest.raises(RuntimeError, match='视场/像素尺寸/边界不一致'):
        job.validate_field_metadata(capture, 2048)
    capture['pixelSizeUm'] = 2.
    capture['meta']['FieldSize'] = 2048
    job.validate_field_metadata(capture, 2048)


def test_stage_readback_not_nominal_return_controls_convergence():
    scope = OfflineMicroscope()
    nominal_move = scope.move_xy
    calls = 0
    def undershoot(x, y):
        nonlocal calls
        calls += 1
        result = nominal_move(x, y)
        if calls == 1:
            scope.position += [80., 0.]
        return result  # Lying nominal 'after' / error=0 must not pass.
    scope.move_xy = undershoot
    result = job.move_verified(scope, [100, 100], replace(job.RefinementSettings(), stage_settle_seconds=.001))
    assert calls == 2 and result['positionErrorUm'] == 0


def test_cleanup_failure_prevents_numeric_acceptance(tmp_path, monkeypatch):
    config, scope = prepare(tmp_path, monkeypatch)
    def fail_restore():
        raise RuntimeError('cannot restore video')
    scope.restore_video = fail_restore
    result = job.run_ab_particle_calibration(tmp_path, config=config, microscope=scope)
    assert result['calibration']['abPointResult']['semRefinement']['status'] == 'FAILED'
    with pytest.raises(ValueError, match='独立数值验收'):
        review_cleanliness_calibration(tmp_path, accepted=True)


def test_fine_axis_retries_different_region_without_lowering_threshold(tmp_path, monkeypatch):
    config, scope = prepare(tmp_path, monkeypatch)
    original = job.measure_pixel_axes
    sites = []
    def unreliable_first_region(images, positions, pixel, cfg):
        if pixel == 2:
            sites.append(tuple(positions['o1']))
            assert cfg.axis_min_response == .15
            if len(sites) == 1:
                raise ValueError('periodic substrate: insufficient phase response')
        return original(images, positions, pixel, cfg)
    monkeypatch.setattr(job, 'measure_pixel_axes', unreliable_first_region)
    result = job.run_ab_particle_calibration(tmp_path, config=config, microscope=scope)
    audit = result['calibration']['abPointResult']['semRefinement']
    assert audit['status'] == 'SUCCEEDED'
    assert len(sites) == 2 and sites[0] != sites[1]
    assert len(audit['axisFailures']) == 1


def test_per_task_html_report_includes_failure_and_escapes_input(tmp_path, monkeypatch):
    config, scope = prepare(tmp_path, monkeypatch)
    scope.fail_capture = True
    manifest = job.run_ab_particle_calibration(tmp_path, config=config, microscope=scope)
    audit = manifest['calibration']['abPointResult']['semRefinement']
    path = tmp_path / audit['htmlReport']
    assert path.is_file()
    content = path.read_text(encoding='utf-8')
    assert '不可用于自动检测' in content and 'offline injected capture failure' in content


def test_server_preserves_third_mode_in_ab_assignment(tmp_path, monkeypatch):
    monkeypatch.setenv('ZEM_TASK_ROOT', str(tmp_path / 'isolated-server-tasks'))
    monkeypatch.setenv('ZEM_LOG_DIR', str(tmp_path / 'isolated-server-logs'))
    from backend import server
    config, _ = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(server, 'TASK_ROOT', tmp_path)
    monkeypatch.setattr(server, 'task_manifest', lambda _: (tmp_path / 'draft.json', {'input': {'path': str(config.input_path)}, 'request': {'mode': 'multi_particle'}}))
    monkeypatch.setattr(server, 'AB_SESSIONS', {'demo': {'inputSha256': 'test', 'image': None, 'bounds': [], 'detection': None}})
    captured = {}
    def make_config(_input, _output, request):
        captured.update(request)
        return config
    monkeypatch.setattr(server, '_make_config', make_config)
    monkeypatch.setattr(cleanliness_task, 'save_ab_point_calibration', lambda *_args, **_kwargs: {'taskId': 'new-id'})
    result = server.save_ab_assignment('demo', {'mode': 'ab_particle', 'aIndex': 0, 'bIndex': 1})
    assert captured['mode'] == 'ab_particle' and result['request']['mode'] == 'ab_particle'
    assert result['taskId'] == 'demo'


def test_job_manager_prevents_concurrent_device_work(tmp_path, monkeypatch):
    monkeypatch.setenv('ZEM_TASK_ROOT', str(tmp_path / 'isolated-server-tasks'))
    monkeypatch.setenv('ZEM_LOG_DIR', str(tmp_path / 'isolated-server-logs'))
    from backend import server
    monkeypatch.setattr(server, 'log_event', lambda *_args, **_kwargs: None)
    manager = server.JobManager()
    active = manager.create('sample1', 'ab_refine')
    with pytest.raises(ValueError, match='重复启动'):
        manager.create('sample2', 'analysis')
    with pytest.raises(ValueError):
        manager.create('sample1', 'review')
    manager.update(active['id'], status='completed')
    assert manager.create('sample2', 'analysis')['status'] == 'queued'
