import json
from dataclasses import replace

import numpy as np
import pytest

from backend.cleanliness_core import cleanliness_task as engine
from backend.tests.test_cleanliness_task import _write_export, _write_experience, _write_settings


POINTS = {'a': {'x': -20100., 'y': -18890.}, 'b': {'x': 20380., 'y': 20060.}}


def prepare(tmp_path):
    source = _write_export(tmp_path / 'optical.json')
    experience = _write_experience(tmp_path / 'experience.json')
    settings = _write_settings(tmp_path / 'settings.json', experience)
    return engine.CleanlinessTaskConfig(input_path=source, output_dir=tmp_path / 'task',
        settings_path=settings, experience_calibration_path=experience,
        calibration_strategy='ab_points', auto_gun_on=False, gun_off_after_task=False)


def test_manual_points_create_without_navigation_and_map_endpoints(tmp_path):
    config = prepare(tmp_path)
    result = engine.save_ab_point_calibration(config, manual_stage_points=POINTS)
    ab = result['calibration']['abPointResult']
    assert ab['navigation'] is None and ab['detection'] is None
    assert not (config.output_dir / 'navigation').exists()
    assert result['cleanup']['deviceActionsPerformed'] is False
    matrix = np.array(ab['previewTransform']['matrix'])
    translation = ab['previewTransform']['translation']
    for label in ('a', 'b'):
        np.testing.assert_allclose(matrix @ ab['sourceReferencePointsUm'][label] + translation,
                                   list(POINTS[label].values()), atol=1e-8)
    assert ab['manualStagePoints']['baselineUm'] == pytest.approx(56175.9103175)
    assert ab['previewTransform']['verified'] is False
    with pytest.raises(FileExistsError):
        engine.save_ab_point_calibration(config, manual_stage_points=POINTS)


@pytest.mark.parametrize('bad', [None, '', True, float('nan'), float('inf'), 50001])
def test_invalid_manual_points_leave_no_task(tmp_path, bad):
    config = prepare(tmp_path)
    with pytest.raises(ValueError):
        engine.save_ab_point_calibration(config, manual_stage_points={**POINTS, 'a': {'x': bad, 'y': 0}})
    assert not (config.output_dir / 'task.json').exists()


def test_manual_rejects_same_point_stale_input_and_wrong_strategy(tmp_path):
    config = prepare(tmp_path)
    for cfg, points in [(config, {'a': POINTS['a'], 'b': POINTS['a']}),
                        (replace(config, expected_input_sha256='stale'), POINTS),
                        (replace(config, calibration_strategy='ab_particle'), POINTS)]:
        with pytest.raises(ValueError):
            engine.save_ab_point_calibration(cfg, manual_stage_points=points)


def test_manual_refinement_skips_holes_and_requires_review(tmp_path, monkeypatch):
    config = prepare(tmp_path)
    engine.save_ab_point_calibration(config, manual_stage_points=POINTS)
    from backend.cleanliness_core import hole_center_locator
    monkeypatch.setattr(hole_center_locator, 'locate_hole_center', lambda *a, **k: pytest.fail('manual mode searched for holes'))
    calls = []
    monkeypatch.setattr(engine, '_capture_ab_validation_particles', lambda *a, **k: calls.append(a[3]) or [{'sequence': 1}])
    monkeypatch.setattr(engine, '_restore_and_maybe_gun_off', lambda *a, **k: {'restored': True})
    class Scope:
        def connect(self): return {'status': 'Idle'}
        def get_position(self): return (0., 0.)
    events = []
    result = engine.refine_ab_point_calibration(
        config.output_dir,
        config=config,
        microscope=Scope(),
        event_logger=lambda event, fields: events.append((event, fields)),
    )
    assert result['status'] == 'CALIBRATION_PENDING_REVIEW'
    assert len(calls) == 1
    refinement = result['calibration']['abPointResult']['semRefinement']
    assert refinement['mode'] == 'manual_stage_points'
    assert refinement['a']['captureCount'] == refinement['b']['captureCount'] == 0
    assert refinement['a']['holeStageUm'] == POINTS['a']
    assert not [item for item in events if item[1]['stage'] == 'hole_localization']
    for stage in ('calibration', 'connect', 'transform_fit', 'validation_capture'):
        stage_events = [item for item in events if item[1]['stage'] == stage]
        assert [item[0] for item in stage_events] == [
            'calibration_stage_started', 'calibration_stage_finished'
        ]
        assert stage_events[-1][1]['elapsedSeconds'] >= 0


def test_manual_api_units_id_reopen_and_lock(tmp_path, monkeypatch):
    from backend import server
    config = prepare(tmp_path)
    monkeypatch.setattr(server, 'TASK_ROOT', tmp_path)
    monkeypatch.setattr(server, '_make_config', lambda *a: config)
    config.output_dir.mkdir()
    from backend.cleanliness_core.optical_particle_analysis import load_optical_export
    sha = load_optical_export(config.input_path).sha256
    (config.output_dir / 'draft.json').write_text(json.dumps({'status': 'DRAFT', 'input': {
        'path': str(config.input_path), 'sha256': sha}, 'request': {}}), encoding='utf-8')
    body = {'coordinateUnit': 'mm', 'a': {'x': -20.1, 'y': -18.89}, 'b': {'x': 20.38, 'y': 20.06}}
    with pytest.raises(ValueError): server.save_manual_ab_assignment('task', {**body, 'coordinateUnit': 'um'})
    result = server.save_manual_ab_assignment('task', body)
    assert result['taskId'] == result['folderId'] == 'task'
    assert result['request']['mode'] == 'ab_manual'
    assert result['calibration']['abPointResult']['manualStagePoints']['a'] == POINTS['a']
    with pytest.raises(ValueError): server.save_manual_ab_assignment('task', body)
