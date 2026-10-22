import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.cleanliness_core.ab_point_calibration import (
    build_ab_preview_transform,
    infer_ab_handedness,
    nav_pixel_to_stage,
    resolve_ab_reference_points,
    validate_ab_export,
)
from backend.cleanliness_core.cleanliness_detection import (
    CandidateGate,
    CandidateSegmentationConfig,
    ParticleFilter,
    SemAnchorCandidate,
    choose_calibration_matches,
    evaluate_transform,
    filter_optical_particles,
    fit_coordinate_transform,
    optical_salience_score,
    pixel_to_stage_xy,
    sam3_text_candidates,
    select_calibration_landmarks,
    select_validation_landmarks,
    yolo_seg_candidates,
    extract_optical_shape,
)
from backend.cleanliness_core.optical_particle_analysis import (
    ParticleCandidate,
    load_optical_export,
)


def _encoded_particle(index: int) -> str:
    image = np.full((36, 64, 3), 215, dtype=np.uint8)
    cv2.ellipse(
        image,
        (32, 18),
        (10 + index % 8, 5 + index % 4),
        index * 13,
        0,
        360,
        (20 + index, 25 + index, 30 + index),
        -1,
    )
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _export(path: Path, count: int = 12):
    particles = []
    for index in range(count):
        particles.append(
            {
                "sequence": index + 1,
                "particleId": f"p-{index + 1}",
                "label": "metal" if index % 2 == 0 else "fiber",
                "particleInfo": {
                    "isValid": True,
                    "length": 40.0 + index * 12.0,
                    "width": 12.0 + index,
                    "area": 300.0 + index * 90.0,
                    "equivalentDiameter": 18.0 + index * 2.0,
                },
                "imageDataBase64": _encoded_particle(index),
                "imageWidth": 64,
                "imageHeight": 36,
                "abPosition": {
                    "x": float((index % 4) * 12000 + index * 300),
                    "y": float((index // 4) * 14000 - index * 170),
                },
            }
        )
    path.write_text(
        json.dumps(
            {
                "projectId": "project",
                "workpieceId": "sample",
                "coordinateUnit": "um",
                "coordinateSpace": "sourceStage",
                "particles": particles,
            }
        ),
        encoding="utf-8",
    )
    return load_optical_export(path)


def test_customer_mask_controls_shape_even_when_photo_is_unreadable(tmp_path):
    particle = _export(tmp_path / "optical.json", count=1).particles[0]
    mask = np.zeros((36, 64), np.uint8)
    # Asymmetric L shape, deliberately different from the ellipse photo.
    mask[6:30, 8:16] = 255
    mask[23:30, 8:50] = 255
    encoded = base64.b64encode(cv2.imencode(".png", mask)[1]).decode()
    shape = extract_optical_shape(replace(particle, image_data_base64="unreadable",
        mask_data_base64="data:image/png;base64,"+encoded, mask_width=64, mask_height=36))
    assert shape.source == "customer_mask"
    reconstructed = np.zeros_like(mask)
    cv2.fillPoly(reconstructed, [np.asarray(shape.contour, np.int32)], 255)
    assert np.array_equal(reconstructed, mask)
    assert shape.solidity < .7


@pytest.mark.parametrize("problem", ["broken", "size", "empty", "full"])
def test_invalid_supplied_mask_never_silently_falls_back_to_photo(tmp_path, problem):
    particle = _export(tmp_path / "optical.json", count=1).particles[0]
    mask = np.zeros((36, 64), np.uint8)
    mask[6:30, 8:50] = 255
    if problem == "empty":
        mask[:] = 0
    elif problem == "full":
        mask[:] = 255
    encoded = base64.b64encode(cv2.imencode(".png", mask)[1]).decode()
    with pytest.raises(ValueError, match="掩码"):
        extract_optical_shape(replace(particle, mask_data_base64="bad!" if problem=="broken" else encoded,
                                     mask_width=63 if problem=="size" else 64, mask_height=36))


def test_legacy_photo_without_mask_keeps_shape_fallback(tmp_path):
    particle = _export(tmp_path / "optical.json", count=1).particles[0]
    assert extract_optical_shape(particle).source == "image_inferred"


def test_filter_uses_physical_units_and_limit(tmp_path):
    export = _export(tmp_path / "optical.json")
    selected = filter_optical_particles(
        export.particles,
        ParticleFilter(
            min_length_um=100.0,
            max_length_um=170.0,
            labels=("metal",),
            max_particles=2,
        ),
    )
    assert len(selected) == 2
    assert all(100.0 <= particle.length_um <= 170.0 for particle in selected)
    assert all(particle.label == "metal" for particle in selected)
    assert selected[0].length_um >= selected[1].length_um


def test_landmark_selection_is_spread_and_independent(tmp_path):
    export = _export(tmp_path / "optical.json")
    anchors, validation = select_calibration_landmarks(
        export,
        anchor_count=3,
        validation_count=3,
    )
    assert len(anchors) == 3
    assert len(validation) == 3
    assert not ({item.sequence for item in anchors} & {item.sequence for item in validation})
    salience_rank = sorted(
        export.particles,
        key=lambda item: (optical_salience_score(item), item.length_um),
        reverse=True,
    )
    assert {item.sequence for item in anchors} <= {
        item.sequence for item in salience_rank[:6]
    }
    points = np.asarray([[item.ab_x_um, item.ab_y_um] for item in anchors])
    assert np.linalg.matrix_rank(points - points.mean(axis=0)) == 2


def test_manual_anchors_only_require_requested_validation_particles(tmp_path):
    export = _export(tmp_path / "optical.json", count=5)
    anchors = export.particles[:3]
    validation = select_validation_landmarks(
        export,
        anchors,
        validation_count=2,
    )
    assert {item.sequence for item in validation} == {4, 5}


def test_reflected_similarity_recovers_known_transform():
    source = np.asarray([[0.0, 0.0], [1000.0, 50.0], [120.0, 900.0], [850.0, 720.0]])
    matrix = np.asarray([[0.71, 0.69], [0.69, -0.71]])
    translation = np.asarray([-19000.0, -22000.0])
    target = (matrix @ source.T).T + translation

    solution = fit_coordinate_transform(source, target, model="reflected_similarity")

    assert solution.matrix == pytest.approx(matrix, abs=1e-10)
    assert solution.translation == pytest.approx(translation, abs=1e-8)
    assert solution.reflected is True
    assert solution.rms_error_um < 1e-8
    assert "stageX" in solution.equations()[0]


def test_affine_and_validation_errors():
    source = np.asarray([[0.0, 0.0], [1000.0, 0.0], [0.0, 800.0], [900.0, 700.0]])
    matrix = np.asarray([[0.8, 0.2], [-0.1, 1.1]])
    translation = np.asarray([40.0, -70.0])
    target = (matrix @ source.T).T + translation
    solution = fit_coordinate_transform(source[:3], target[:3], model="affine")
    validation = evaluate_transform(solution, source[3:], target[3:])
    assert solution.matrix == pytest.approx(matrix)
    assert validation["maxErrorUm"] == pytest.approx(0.0, abs=1e-8)


def test_pixel_to_stage_mapping():
    assert pixel_to_stage_xy(
        (300.0, 220.0),
        (512, 512),
        2.0,
        (1000.0, -500.0),
    ) == pytest.approx((1088.0, -572.0))


def test_navigation_pixel_to_stage_uses_existing_axis_directions():
    bounds = (10.0, 20.0, 30.0, 60.0)

    assert nav_pixel_to_stage((0.0, 0.0), (100, 200), bounds) == pytest.approx(
        (30.0, 20.0)
    )
    assert nav_pixel_to_stage(
        (200.0, 100.0),
        (100, 200),
        bounds,
    ) == pytest.approx((10.0, 60.0))
    assert nav_pixel_to_stage(
        (100.0, 50.0),
        (100, 200),
        bounds,
    ) == pytest.approx((20.0, 40.0))


def test_ab_export_metadata_and_reflected_preview_transform(tmp_path):
    path = tmp_path / "optical_ab.json"
    _export(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        {
            "registrationMode": "twoPointAB",
            "referencePoints": {
                "a": {"x": 1000.0, "y": 2000.0},
                "b": {"x": 5000.0, "y": 2000.0},
            },
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    export = load_optical_export(path)

    validate_ab_export(export)
    assert export.registration_mode == "twoPointAB"
    assert export.reference_points == {
        "a": pytest.approx((1000.0, 2000.0)),
        "b": pytest.approx((5000.0, 2000.0)),
    }

    preview = build_ab_preview_transform(
        export.reference_a_um,
        export.reference_b_um,
        (10.0, 20.0),
        (10.0, 4020.0),
        handedness="right",
    )
    assert preview["verified"] is False
    assert preview["reflected"] is True
    assert preview["scale"] == pytest.approx(1.0)
    assert np.asarray(preview["matrix"]) == pytest.approx(
        np.asarray([[0.0, 1.0], [1.0, 0.0]])
    )
    # 变换的本质属性：A、B 参考点必须精确映射到各自的目标坐标
    matrix = np.asarray(preview["matrix"], dtype=np.float64)
    translation = np.asarray(preview["translation"], dtype=np.float64)
    mapped_a = matrix @ np.asarray(export.reference_a_um) + translation
    mapped_b = matrix @ np.asarray(export.reference_b_um) + translation
    assert mapped_a == pytest.approx((10.0, 20.0))
    assert mapped_b == pytest.approx((10.0, 4020.0))


def _write_ab_export(path: Path, ab_positions, reference_a, reference_b, source_of=None):
    """写一个 twoPointAB 导出：可选地给颗粒附加 position(sourceStage) 字段。"""
    particles = []
    for index, ab in enumerate(ab_positions):
        item = {
            "sequence": index + 1,
            "particleId": f"p-{index + 1}",
            "label": "metal",
            "particleInfo": {"isValid": True, "length": 50.0, "width": 10.0},
            "imageDataBase64": _encoded_particle(index),
            "imageWidth": 64,
            "imageHeight": 36,
            "abPosition": {"x": float(ab[0]), "y": float(ab[1])},
        }
        if source_of is not None:
            source = source_of(ab)
            item["position"] = {"x": float(source[0]), "y": float(source[1])}
        particles.append(item)
    path.write_text(
        json.dumps(
            {
                "projectId": "project",
                "workpieceId": "sample",
                "coordinateUnit": "um",
                "coordinateSpace": "sourceStage",
                "registrationMode": "twoPointAB",
                "referencePoints": {
                    "a": {"x": float(reference_a[0]), "y": float(reference_a[1])},
                    "b": {"x": float(reference_b[0]), "y": float(reference_b[1])},
                },
                "particles": particles,
            }
        ),
        encoding="utf-8",
    )
    return load_optical_export(path)


def test_resolve_ab_reference_points_follows_spec_definition(tmp_path):
    """复刻真机：referencePoints 是 sourceStage 系，颗粒带 position 字段。

    规范 6.2：ab 系以 A 为原点、A→B 为 +X 轴（左法向为 +Y），
    所以解析结果恒为 A=(0,0)、B=(|AB|,0)，并用 position↔abPosition
    拟合交叉校验（ab = R(90°) @ source + t，保向）。
    """
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    translation = np.asarray([62207.0, -63745.0])

    def to_source(ab_xy):
        return rotation.T @ (np.asarray(ab_xy, dtype=np.float64) - translation)

    ab_positions = [
        (8000.0, -9000.0),
        (28000.0, 11000.0),
        (48000.0, -2000.0),
        (30000.0, 3000.0),
    ]
    export = _write_ab_export(
        tmp_path / "optical_source_frame.json",
        ab_positions,
        to_source((0.0, 0.0)),
        to_source((56000.0, 0.0)),
        source_of=to_source,
    )
    resolved = resolve_ab_reference_points(export)
    assert resolved["method"] == "abDefinition"
    assert resolved["a"] == pytest.approx((0.0, 0.0), abs=1e-6)
    assert resolved["b"] == pytest.approx((56000.0, 0.0), abs=1e-6)
    assert resolved["fitConsistencyUm"] == pytest.approx(0.0, abs=1e-6)
    assert resolved["sourceToAbDet"] == pytest.approx(1.0)
    assert resolved["sourceToAbScale"] == pytest.approx(1.0)


def test_resolve_ab_reference_points_without_position_fields(tmp_path):
    """旧导出无 position 字段：仍按规范定义解析，只是没有交叉校验。"""
    ab_positions = [
        (8000.0, -9000.0),
        (28000.0, 11000.0),
        (48000.0, -2000.0),
        (30000.0, 3400.0),
    ]
    export = _write_ab_export(
        tmp_path / "optical_no_position.json",
        ab_positions,
        (1000.0, 500.0),
        (55000.0, 1200.0),
        source_of=None,
    )
    resolved = resolve_ab_reference_points(export)
    baseline = float(np.hypot(55000.0 - 1000.0, 1200.0 - 500.0))
    assert resolved["method"] == "abDefinition"
    assert resolved["a"] == pytest.approx((0.0, 0.0))
    assert resolved["b"] == pytest.approx((baseline, 0.0))
    assert resolved["fitConsistencyUm"] is None
    assert resolved["sourceToAbDet"] is None


def test_resolve_ab_reference_points_rejects_inconsistent_export(tmp_path):
    """参考点与 position↔abPosition 拟合矛盾（导出数据自相矛盾）时报错。"""
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    translation = np.asarray([62207.0, -63745.0])

    def to_source(ab_xy):
        return rotation.T @ (np.asarray(ab_xy, dtype=np.float64) - translation)

    ab_positions = [
        (8000.0, -9000.0),
        (28000.0, 11000.0),
        (48000.0, -2000.0),
        (30000.0, 3000.0),
    ]
    with pytest.raises(ValueError, match="自相矛盾"):
        resolve_ab_reference_points(
            _write_ab_export(
                tmp_path / "optical_inconsistent.json",
                ab_positions,
                to_source((900.0, -700.0)),  # 参考点不是 ab 系原点，偏了 1.1mm
                to_source((56000.0, 0.0)),
                source_of=to_source,
            )
        )


def test_infer_ab_handedness_chirality_chain():
    reflected_experience = [[1.0, 0.0], [0.0, -1.0]]
    rotation_experience = [[0.0, -1.0], [1.0, 0.0]]
    # 光镜纯旋转 + 经验矩阵反射 → 需要反射（right，即规范公式取右法向）
    assert infer_ab_handedness(1.0, reflected_experience) == "right"
    # 光镜纯旋转 + 经验矩阵纯旋转 → 不需要反射（left，即规范原式左法向）
    assert infer_ab_handedness(1.0, rotation_experience) == "left"
    # 光镜带反射 + 经验矩阵反射 → 抵消（left）
    assert infer_ab_handedness(-1.0, reflected_experience) == "left"
    # 拟合不可用时按规范 6.2 默认保向（det=+1）
    assert infer_ab_handedness(None, reflected_experience) == "right"
    assert infer_ab_handedness(None, rotation_experience) == "left"
    assert infer_ab_handedness(1.0, None) is None


def _candidate(sequence: int, stage_xy, detector_score: float):
    particle_candidate = ParticleCandidate(
        box=(10, 10, 30, 30),
        center=(20.0, 20.0),
        contour=((10, 10), (30, 10), (30, 30), (10, 30)),
        distance_px=0.0,
        length_ratio=1.0,
        width_ratio=1.0,
        area_ratio=1.0,
        contrast_z=8.0,
        signal_z=8.0,
        channel="bright",
        score=detector_score,
        threshold_z=3.0,
        close_kernel=3,
    )
    return SemAnchorCandidate(
        sequence=sequence,
        stage_xy_um=tuple(stage_xy),
        pixel_xy=(20.0, 20.0),
        detector_score=detector_score,
        shape_score=0.0,
        candidate=particle_candidate,
    )


@pytest.mark.parametrize("status,center,accepted", [
    ("MATCH", [20,20], True), ("AMBIGUOUS", [20,20], False),
    ("MATCH", [80,80], False), ("NO_MATCH", None, False),
])
def test_mask_identity_requires_unique_peak_at_selected_contour(tmp_path, monkeypatch, status, center, accepted):
    from backend.cleanliness_core.cleanliness_task import _decision_for_candidate, _verify_candidate_mask_identity
    particle = replace(_export(tmp_path/"optical.json", count=1).particles[0], mask_data_base64="supplied")
    result = {"status":status,"score":.8,"margin":.04 if status=="AMBIGUOUS" else .2}
    if center is not None:
        result["centerPixel"] = center
    monkeypatch.setattr("backend.cleanliness_core.particle_refinement.match_particle", lambda *args, **kwargs:dict(result))
    original = _decision_for_candidate(_candidate(1,(0,0),1),1,reason_prefix="test")
    decision,audit = _verify_candidate_mask_identity(np.zeros((128,128),np.uint8),particle,1,original)
    assert decision.matched is accepted
    assert audit["accepted"] is accepted
    assert decision.best is original.best


@pytest.mark.parametrize("support", [True, False])
def test_mask_guided_search_uses_observed_contour_and_restores_full_frame_coordinates(tmp_path, monkeypatch, support):
    from backend.cleanliness_core.cleanliness_task import _select_mask_guided_candidate
    mask=np.zeros((21,21),np.uint8);mask[1:20,1:20]=255
    encoded=base64.b64encode(cv2.imencode('.png',mask)[1]).decode()
    particle=replace(_export(tmp_path/'optical.json',count=1).particles[0],
                     mask_data_base64=encoded,mask_width=21,mask_height=21,image_width=21,image_height=21,
                     area_um2=361,length_um=19,width_um=19)
    monkeypatch.setattr('backend.cleanliness_core.particle_refinement.match_particle',lambda *a,**k:
        dict(status='MATCH',centerPixel=[120.,90.],angleDeg=0,mirrored=False,scaleFactor=1.,score=.9,margin=.3))
    def local_candidates(particle,image,**kwargs):
        # With 19px padding, template bounding box [111..129,81..99]
        # gives crop origin (92,62). This independently defined SEM contour
        # either covers the target or sits at an unrelated location.
        c=_candidate(1,(0,0),1)
        low,high=(19,37) if support else (0,10)
        center=((low+high)/2,)*2
        return (replace(c,pixel_xy=center,candidate=replace(c.candidate,center=center,
            box=(low,low,high,high),contour=((low,low),(high,low),(high,high),(low,high)))),)
    monkeypatch.setattr('backend.cleanliness_core.cleanliness_task.build_sem_anchor_candidates',local_candidates)
    chosen,decision,audit=_select_mask_guided_candidate(np.zeros((256,256),np.uint8),particle,1,
        (_candidate(1,(0,0),1),),stage_center_um=(1000,2000))
    assert decision.matched is support
    json.dumps(audit)  # Includes numpy-derived comparisons; must remain serializable.
    if support:
        assert chosen.pixel_xy==pytest.approx((120,90))
        assert chosen.stage_xy_um==pytest.approx((992,1962))
        assert chosen.candidate.box==(111,81,129,99)
        assert audit['maskIoU']==pytest.approx(1)
        assert audit['relocated'] is True
    else:
        assert audit['accepted'] is False


def test_mask_guided_ambiguous_peak_never_triggers_local_segmentation(tmp_path, monkeypatch):
    from backend.cleanliness_core.cleanliness_task import _select_mask_guided_candidate
    particle=replace(_export(tmp_path/'optical.json',count=1).particles[0],mask_data_base64='present')
    monkeypatch.setattr('backend.cleanliness_core.particle_refinement.match_particle',lambda *a,**k:
        dict(status='AMBIGUOUS',centerPixel=[120.,90.],score=.9,margin=.01))
    def forbidden(*a,**k):raise AssertionError('Must not segment or accept an ambiguous template')
    monkeypatch.setattr('backend.cleanliness_core.cleanliness_task.build_sem_anchor_candidates',forbidden)
    _,decision,audit=_select_mask_guided_candidate(np.zeros((256,256),np.uint8),particle,1,
        (_candidate(1,(0,0),1),),stage_center_um=(0,0))
    assert not decision.matched and audit['accepted'] is False


def test_mask_guided_matching_uses_configured_candidate_margin_for_all_competition_checks(tmp_path, monkeypatch):
    from backend.cleanliness_core.cleanliness_task import _select_mask_guided_candidate
    from backend.cleanliness_core.particle_refinement import RefinementSettings
    particle=replace(_export(tmp_path/'optical.json',count=1).particles[0],mask_data_base64='present')
    seen={}
    def ambiguous(_image,_particle,_pixel,cfg,**_kwargs):
        seen['templateMargin']=cfg.match_min_margin
        return dict(status='AMBIGUOUS',centerPixel=[20.,20.],score=.9,margin=.08)
    def review(*_args,**kwargs):
        seen['shapeMargin']=kwargs['minimum_shape_margin']
        seen['templateScore']=kwargs['minimum_template_score']
        return dict(status='UNVERIFIED',corroborated=False,allowsEds=False,reason='controlled rejection')
    monkeypatch.setattr('backend.cleanliness_core.particle_refinement.match_particle',ambiguous)
    monkeypatch.setattr('backend.cleanliness_core.shape_consensus.review_ambiguous_shapes',review)
    settings=replace(RefinementSettings(),match_min_margin=.12)
    _,decision,audit=_select_mask_guided_candidate(
        np.zeros((128,128),np.uint8),particle,1,(_candidate(1,(0,0),1),),
        stage_center_um=(0,0),refinement_settings=settings)
    assert seen=={'templateMargin':.12,'shapeMargin':.12,'templateScore':settings.match_min_score}
    assert audit['minimumCandidateScoreMargin']==.12
    assert not decision.matched and audit['accepted'] is False


def test_shape_corroboration_cannot_switch_to_a_different_final_center(tmp_path, monkeypatch):
    from backend.cleanliness_core.cleanliness_task import _select_mask_guided_candidate
    mask=np.zeros((63,63),np.uint8);mask[12:51,12:51]=255
    encoded=base64.b64encode(cv2.imencode('.png',mask)[1]).decode()
    particle=replace(_export(tmp_path/'optical.json',count=1).particles[0],
        mask_data_base64=encoded,mask_width=63,mask_height=63,image_width=63,image_height=63,
        area_um2=1521,length_um=39,width_um=39)
    monkeypatch.setattr('backend.cleanliness_core.particle_refinement.match_particle',lambda *a,**k:
        dict(status='AMBIGUOUS',centerPixel=[120.,120.],angleDeg=0,mirrored=False,scaleFactor=1.,score=.95,margin=.02))
    monkeypatch.setattr('backend.cleanliness_core.shape_consensus.review_ambiguous_shapes',lambda *a,**k:
        dict(corroborated=True,allowsEds=False,reason='test',targets=[{'yoloCenterPx':[120.,120.]}]))
    candidate=_candidate(1,(1000,2000),1)
    shifted=replace(candidate,pixel_xy=(131.,120.),candidate=replace(candidate.candidate,
        center=(131.,120.),box=(112,101,150,139),contour=((112,101),(150,101),(150,139),(112,139))))
    monkeypatch.setattr('backend.cleanliness_core.cleanliness_task.build_sem_anchor_candidates',lambda *a,**k:())
    _,decision,audit=_select_mask_guided_candidate(np.zeros((256,256),np.uint8),particle,1,(shifted,),stage_center_um=(1000,2000))
    assert audit['maskIoU']>.5
    assert audit['consensusCenterDifferencePx']>audit['maximumConsensusCenterDifferencePx']
    assert not decision.matched and audit['accepted'] is False


def _center_speck(sequence: int, stage_xy):
    """视场正中心的几像素噪点：距心为零所以检测评分极佳，但尺寸完全不符。"""
    particle_candidate = ParticleCandidate(
        box=(254, 254, 258, 258),
        center=(256.0, 256.0),
        contour=((255, 255), (257, 255), (257, 257), (255, 257)),
        distance_px=0.0,
        length_ratio=0.05,
        width_ratio=0.06,
        area_ratio=0.004,
        contrast_z=3.0,
        signal_z=3.0,
        channel="bright",
        score=-0.5,
        threshold_z=2.2,
        close_kernel=3,
    )
    return SemAnchorCandidate(
        sequence=sequence,
        stage_xy_um=tuple(stage_xy),
        pixel_xy=(256.0, 256.0),
        detector_score=-0.5,
        shape_score=0.0,
        candidate=particle_candidate,
    )


def test_candidate_gate_rejects_center_speck():
    gate = CandidateGate()
    assert gate.accepts(_candidate(1, (0.0, 0.0), 0.2).candidate)
    rejection = gate.rejection(_center_speck(1, (0.0, 0.0)).candidate)
    assert rejection is not None
    assert "长度比" in rejection


def test_triangle_combination_rejects_individually_attractive_wrong_candidate(tmp_path):
    export = _export(tmp_path / "optical.json", count=6)
    anchors = (export.particles[0], export.particles[1], export.particles[4])
    source = np.asarray([[p.ab_x_um, p.ab_y_um] for p in anchors])
    matrix = np.asarray([[0.7, 0.7], [0.7, -0.7]])
    translation = np.asarray([-20000.0, -21000.0])
    target = (matrix @ source.T).T + translation
    candidates = {
        anchors[0].sequence: (
            _candidate(anchors[0].sequence, target[0] + np.asarray([400.0, -500.0]), -1.0),
            _candidate(anchors[0].sequence, target[0], 0.2),
        ),
        anchors[1].sequence: (
            _candidate(anchors[1].sequence, target[1], 0.2),
        ),
        anchors[2].sequence: (
            _candidate(anchors[2].sequence, target[2], 0.2),
        ),
    }
    result = choose_calibration_matches(
        anchors,
        candidates,
        model="reflected_similarity",
    )
    assert result.matches[0].stage_xy_um == pytest.approx(target[0])
    assert result.solution.rms_error_um < 1e-6


def test_calibration_prefers_real_particles_over_center_specks(tmp_path):
    """"锚点恰好落在预测位置"是零残差、零漂移的退化解，不能让它压过真实颗粒。"""
    export = _export(tmp_path / "optical.json", count=6)
    anchors = (export.particles[0], export.particles[1], export.particles[4])
    source = np.asarray([[p.ab_x_um, p.ab_y_um] for p in anchors])
    matrix = np.asarray([[0.7, 0.7], [0.7, -0.7]])
    translation = np.asarray([-20000.0, -21000.0])
    # 样品台每次都移动到经验矩阵预测的位置，所以预测点就是视场中心。
    predicted = (matrix @ source.T).T + translation
    # 真实颗粒相对预测位置有几十微米的经验标定残差，形态上略有差异。
    offsets = np.asarray([[80.0, -50.0], [72.0, -58.0], [86.0, -44.0]])
    real = predicted + offsets

    candidates = {
        anchor.sequence: (
            _center_speck(anchor.sequence, predicted[index]),
            _candidate(anchor.sequence, real[index], 1.4),
        )
        for index, anchor in enumerate(anchors)
    }
    experience = fit_coordinate_transform(
        source, predicted, model="reflected_similarity"
    )

    result = choose_calibration_matches(
        anchors,
        candidates,
        model="reflected_similarity",
        experience=experience,
    )

    assert result.unreliable_sequences == ()
    for index, match in enumerate(result.matches):
        assert match.stage_xy_um == pytest.approx(real[index])


def test_calibration_flags_anchor_without_any_plausible_candidate(tmp_path):
    export = _export(tmp_path / "optical.json", count=6)
    anchors = (export.particles[0], export.particles[1], export.particles[4])
    source = np.asarray([[p.ab_x_um, p.ab_y_um] for p in anchors])
    matrix = np.asarray([[0.7, 0.7], [0.7, -0.7]])
    target = (matrix @ source.T).T + np.asarray([-20000.0, -21000.0])
    candidates = {
        anchors[0].sequence: (_center_speck(anchors[0].sequence, target[0]),),
        anchors[1].sequence: (_candidate(anchors[1].sequence, target[1], 0.2),),
        anchors[2].sequence: (_candidate(anchors[2].sequence, target[2], 0.2),),
    }

    result = choose_calibration_matches(
        anchors,
        candidates,
        model="reflected_similarity",
    )

    assert result.unreliable_sequences == (anchors[0].sequence,)


def test_sam3_image_id_is_unique_per_image_content(monkeypatch):
    """SAM3 服务按 image_id 复用缓存，不同画面必须拿到不同 id。"""
    import io
    import urllib.request

    requested_ids: list[str] = []

    def fake_urlopen(request, timeout=None):
        requested_ids.append(json.loads(request.data.decode("utf-8"))["image_id"])
        return io.BytesIO(json.dumps({"detections": []}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    first = np.zeros((64, 64), dtype=np.uint8)
    first[10:30, 10:40] = 220
    second = np.zeros((64, 64), dtype=np.uint8)
    second[35:55, 20:50] = 200

    for image in (first, second, first):
        sam3_text_candidates(
            image,
            expected_length_px=30.0,
            expected_width_px=20.0,
            expected_area_px=600.0,
            image_id="cleanliness-0001",
            url="http://sam3.invalid",
        )

    assert requested_ids[0] != requested_ids[1]
    assert requested_ids[0] == requested_ids[2]


def test_candidate_segmentation_config_switches_backends(tmp_path):
    model_path = tmp_path / "particle.onnx"
    model_path.write_bytes(b"onnx")
    settings_path = tmp_path / "cleanliness.json"

    yolo = CandidateSegmentationConfig.from_dict(
        {
            "backend": "yolo",
            "modelPath": "particle.onnx",
            "inputSize": 1024,
        },
        settings_path=settings_path,
    )
    sam3 = CandidateSegmentationConfig.from_dict(
        {"backend": "sam3"},
        settings_path=settings_path,
        sam3_url="http://sam3.invalid",
    )

    yolo.validate()
    assert yolo.yolo_model_path == model_path.resolve()
    assert sam3.backend == "sam3"
    assert sam3.sam3_url == "http://sam3.invalid"


def test_legacy_candidate_config_keeps_sam3_when_url_is_supplied(tmp_path):
    config = CandidateSegmentationConfig.from_dict(
        None,
        settings_path=tmp_path / "cleanliness.json",
        sam3_url="http://sam3.invalid",
    )

    assert config.backend == "sam3"
    assert config.sam3_url == "http://sam3.invalid"


def test_local_yolo_onnx_segments_recorded_sem_image():
    root = Path(__file__).resolve().parents[2]
    fixture = Path(__file__).parent/'fixtures/human_confirmed_20260918'
    corpus=json.loads((fixture/'cases.json').read_text(encoding='utf-8'))
    case=next(c for c in corpus['cases'] if c['reportRank']==125 and c['reviewStatus']=='human_confirmed')
    model_path=root/corpus['model']['path']
    assert model_path.is_file(), 'Pinned YOLO model must be available'
    assert hashlib.sha256(model_path.read_bytes()).hexdigest()==corpus['model']['sha256']
    assert hashlib.sha256((fixture/'optical.json').read_bytes()).hexdigest()==corpus['opticalSubsetSha256']
    particle=load_optical_export(fixture/'optical.json').by_sequence[case['sequence']]
    assert particle.particle_id==case['particleId']
    raw=(fixture/case['image']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==case['imageSha256']
    image = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_UNCHANGED)
    assert image is not None

    candidates = yolo_seg_candidates(
        image,
        expected_length_px=particle.length_um/corpus['pixelSizeUm'],
        expected_width_px=particle.width_um/corpus['pixelSizeUm'],
        expected_area_px=particle.area_um2/corpus['pixelSizeUm']**2,
        model_path=model_path,
        input_size=1024,
        confidence_threshold=0.25,
    )

    assert candidates
    assert all(candidate.channel == "yolo" for candidate in candidates)
    assert all(len(candidate.contour) >= 3 for candidate in candidates)
    assert any(np.linalg.norm(np.subtract(c.center,case['correctTarget']['referenceCenterPx']))<=corpus['associationTolerancePx'] for c in candidates)
