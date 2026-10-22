"""报告原始数据导出接口（GET /tasks/{id}/report-data）的防退化测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.cleanliness_core.report_data import (
    REPORT_DATA_API_VERSION,
    build_report_data,
)


def _succeeded_particle() -> dict:
    return {
        "rank": 1,
        "sequence": 6,
        "particleId": "p-6",
        "label": "metal",
        "status": "SUCCEEDED",
        "lengthUm": 336.3,
        "widthUm": 196.8,
        "areaUm2": 26823.7,
        "equivalentDiameterUm": 184.8,
        "aspect": 1.3855,
        "roundness": 0.3443,
        "opticalAbPositionUm": {"x": 28446.5, "y": 17188.2},
        "targetStageUm": {"x": 12673.7, "y": -11745.3},
        "actualStageUm": {"x": 12680.0, "y": -11740.0},
        "stagePositionErrorUm": 8.2,
        "match": {"status": "MATCHED_HIGH", "confidence": "high"},
        "images": {
            "opticalImage": "optical/0006.png",
            "semRawImage": "sem/0006_attempt1.png",
            "semDisplayImage": "sem/0006_attempt1.u8.png",
            "semAnnotatedImage": "analysis_annotated/0006.png",
            "semParticleImage": "analysis_particles/0006.png",
        },
        "energySpectrum": {
            "acquired_at": "2026-10-10T16:04:20",
            "quantMethod": "default",
            "unit": "percent",
            "liveTimeSeconds": 4.55,
            "realTimeSeconds": 4.959,
            "materialClassification": {
                "materialClass": "Si/Si-C/Si-N",
                "parentClass": "Hard particles(HV>1000)",
                "hardnessLabel": "HV>1000",
            },
            "vdaClassification": {"materialClass": "unalloyed steel", "hardnessLabel": "V2"},
            "elements": [
                {"symbol": "C", "value": 80.80926537513733},
                {"symbol": "O", "value": 15.088513493537903},
                {"symbol": "Fe", "value": 3.7699613720174604},
            ],
        },
        "timing": {
            "stageMoveSeconds": 1.281,
            "captureSeconds": 3.0,
            "matchSeconds": 3.094,
            "edsSeconds": 5.563,
            "totalSeconds": 12.953,
        },
        "localization": {"centerErrorUm": 5.5},
    }


def _failed_particle() -> dict:
    return {
        "rank": 2,
        "sequence": 9,
        "particleId": "p-9",
        "label": "fiber",
        "status": "FAILED_LOW_CONFIDENCE",
        "lengthUm": 120.0,
        "widthUm": 40.0,
        "areaUm2": 3800.0,
        "equivalentDiameterUm": 69.5,
        "opticalAbPositionUm": {"x": 1.0, "y": 2.0},
        "targetStageUm": {"x": 3.0, "y": 4.0},
        "images": {"opticalImage": "optical/0009.png"},
        "error": "未找到颗粒候选",
    }


def _manifest(tmp_path: Path, *, task_type: str = "CLEANLINESS_DETECTION", with_reports: bool = True) -> tuple[Path, dict]:
    manifest = {
        "schemaVersion": 1,
        "taskType": task_type,
        "taskId": "task-0001",
        "status": "COMPLETED",
        "input": {
            "projectId": "proj-1",
            "workpieceId": "w-1",
            "workpieceName": "样品A",
            "particleCount": 389,
            "registrationMode": "twoPointAB",
        },
        "selection": {
            "filter": {"minEcdUm": 112.3},
            "sequences": [6, 9],
            "count": 2,
        },
        "calibration": {
            "strategy": "multi_particle",
            "review": {"accepted": True},
        },
        "particles": [_succeeded_particle(), _failed_particle()],
    }
    if with_reports:
        manifest["reports"] = {
            "html": "cleanliness_report.html",
            "pdf": "cleanliness_report.pdf",
            "csv": "cleanliness_results.csv",
            "json": "cleanliness_report.json",
        }
        (tmp_path / "cleanliness_report.json").write_text(
            json.dumps({"generatedAt": "2026-10-10T16:12:34.527332+08:00"}),
            encoding="utf-8",
        )
    return tmp_path, manifest


def test_report_data_builds_stable_payload(tmp_path):
    task_dir, manifest = _manifest(tmp_path)

    payload = build_report_data(task_dir, manifest, base_url="http://127.0.0.1:8765/")

    assert payload["apiVersion"] == REPORT_DATA_API_VERSION
    assert payload["taskId"] == "task-0001"
    assert payload["taskType"] == "CLEANLINESS_DETECTION"
    assert payload["taskStatus"] == "COMPLETED"
    assert payload["reportedAt"] == "2026-10-10T16:12:34.527332+08:00"
    assert payload["workpiece"]["workpieceName"] == "样品A"
    assert payload["workpiece"]["opticalParticleCount"] == 389
    assert payload["selection"]["selectedCount"] == 2
    assert payload["selection"]["filters"] == {"minEcdUm": 112.3}
    assert payload["calibration"] == {"strategy": "multi_particle", "accepted": True}

    # summary 与报告口径一致：成功 1、失败 1。
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["succeeded"] == 1
    assert payload["summary"]["failed"] == 1
    assert payload["summary"]["matchedHigh"] == 1
    assert payload["summary"]["centerErrorUm"]["mean"] == 5.5

    assert len(payload["particles"]) == 2
    first = payload["particles"][0]
    assert first["sequence"] == 6
    assert first["status"] == "SUCCEEDED"
    assert first["dimensions"]["lengthUm"] == 336.3
    assert first["dimensions"]["aspectRatio"] == 1.3855
    assert first["match"] == {"status": "MATCHED_HIGH", "confidence": "high"}
    assert first["eds"]["acquired"] is True
    assert first["eds"]["elements"] == [
        {"symbol": "C", "weightPercent": 80.8093},
        {"symbol": "O", "weightPercent": 15.0885},
        {"symbol": "Fe", "weightPercent": 3.77},
    ]
    assert first["eds"]["materialClassification"]["materialClass"] == "Si/Si-C/Si-N"
    assert first["eds"]["vdaClassification"]["hardnessLabel"] == "V2"
    assert first["timingSeconds"]["total"] == 12.953

    # 图片与报告文件一律是可下载 URL，指向任务目录相对路径。
    assert first["images"]["opticalImage"] == (
        "http://127.0.0.1:8765/tasks/task-0001/files/optical/0006.png"
    )
    assert first["images"]["semParticleImage"].endswith(
        "/tasks/task-0001/files/analysis_particles/0006.png"
    )
    second = payload["particles"][1]
    assert second["eds"] == {"acquired": False, "elements": []}
    assert second["images"]["semRawImage"] is None
    assert second["match"] is None

    assert payload["reports"]["pdf"] == (
        "http://127.0.0.1:8765/tasks/task-0001/files/cleanliness_report.pdf"
    )
    assert payload["reports"]["csv"].endswith("/files/cleanliness_results.csv")


def test_report_data_requires_generated_report(tmp_path):
    _, manifest = _manifest(tmp_path, with_reports=False)
    with pytest.raises(ValueError, match="报告尚未生成"):
        build_report_data(tmp_path, manifest, base_url="http://127.0.0.1:8765")


def test_report_data_rejects_non_correlation_tasks(tmp_path):
    _, manifest = _manifest(tmp_path, task_type=None)
    with pytest.raises(ValueError, match="仅支持光镜关联分析"):
        build_report_data(tmp_path, manifest, base_url="http://127.0.0.1:8765")


def test_report_data_includes_full_manifest(tmp_path):
    task_dir, manifest = _manifest(tmp_path)
    payload = build_report_data(task_dir, manifest, base_url="http://127.0.0.1:8765")
    # manifest 原文完整透传，第三方需要全部内部字段时可直接使用。
    assert payload["manifest"] == manifest
    assert payload["manifest"]["particles"][0]["particleId"] == "p-6"


def test_openapi_spec_declares_report_data_route():
    from backend.cleanliness_core.report_data_openapi import openapi_spec

    spec = openapi_spec()
    assert spec["openapi"].startswith("3.")
    path = spec["paths"]["/tasks/{taskId}/report-data"]["get"]
    assert path["parameters"][0]["name"] == "taskId"
    assert "200" in path["responses"]
    schemas = spec["components"]["schemas"]
    assert {"ReportData", "Particle", "Eds", "Summary", "Error"} <= set(schemas)
    assert "manifest" in schemas["ReportData"]["properties"]
