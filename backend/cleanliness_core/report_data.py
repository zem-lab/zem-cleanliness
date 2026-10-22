"""Stable third-party report data export for the optical-SEM correlation flow.

The endpoint returns the structured data behind the generated report so an
external consumer can render its own PDF. The payload is deliberately a
cleaned projection of the task manifest: internal manifest fields may evolve,
but this contract must stay backward compatible within an apiVersion.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .optical_particle_report import summarize_manifest

REPORT_DATA_API_VERSION = "1"

_IMAGE_KEYS = (
    "opticalImage",
    "semRawImage",
    "semDisplayImage",
    "semAnnotatedImage",
    "semParticleImage",
)


def _file_url(base_url: str, task_id: str, relative: Any) -> str | None:
    if not relative:
        return None
    return f"{base_url.rstrip('/')}/tasks/{task_id}/files/{relative}"


def _round(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _eds_view(energy_spectrum: dict[str, Any] | None) -> dict[str, Any]:
    if not energy_spectrum:
        return {"acquired": False, "elements": []}
    elements = [
        {
            "symbol": str(item.get("symbol")),
            "weightPercent": _round(item.get("value")),
        }
        for item in (energy_spectrum.get("elements") or [])
        if isinstance(item, dict) and item.get("symbol")
    ]
    material = energy_spectrum.get("materialClassification") or {}
    vda = energy_spectrum.get("vdaClassification") or {}
    return {
        "acquired": True,
        "acquiredAt": energy_spectrum.get("acquired_at"),
        "quantMethod": energy_spectrum.get("quantMethod"),
        "unit": energy_spectrum.get("unit") or "percent",
        "liveTimeSeconds": _round(energy_spectrum.get("liveTimeSeconds")),
        "realTimeSeconds": _round(energy_spectrum.get("realTimeSeconds")),
        "materialClassification": {
            "materialClass": material.get("materialClass"),
            "parentClass": material.get("parentClass"),
            "hardnessLabel": material.get("hardnessLabel"),
        } if material else None,
        "vdaClassification": {
            "materialClass": vda.get("materialClass"),
            "hardnessLabel": vda.get("hardnessLabel"),
        } if vda else None,
        "elements": elements,
    }


def _particle_view(
    particle: dict[str, Any],
    *,
    task_id: str,
    base_url: str,
) -> dict[str, Any]:
    images = particle.get("images") or {}
    match = particle.get("match") or {}
    timing = particle.get("timing") or {}
    eds = particle.get("energySpectrum") or {}
    return {
        "sequence": particle.get("sequence"),
        "particleId": particle.get("particleId"),
        "label": particle.get("label"),
        "rank": particle.get("rank"),
        "status": particle.get("status"),
        "dimensions": {
            "lengthUm": _round(particle.get("lengthUm")),
            "widthUm": _round(particle.get("widthUm")),
            "areaUm2": _round(particle.get("areaUm2")),
            "equivalentDiameterUm": _round(particle.get("equivalentDiameterUm")),
            "aspectRatio": _round(particle.get("aspect")),
            "roundness": _round(particle.get("roundness")),
        },
        "opticalPositionUm": particle.get("opticalAbPositionUm"),
        "targetStageUm": particle.get("targetStageUm"),
        "actualStageUm": particle.get("actualStageUm"),
        "stagePositionErrorUm": _round(particle.get("stagePositionErrorUm")),
        "match": {
            "status": match.get("status"),
            "confidence": match.get("confidence"),
        } if match else None,
        "eds": _eds_view(eds if eds else None),
        "images": {
            key: _file_url(base_url, task_id, images.get(key))
            for key in _IMAGE_KEYS
        },
        "timingSeconds": {
            "stageMove": _round(timing.get("stageMoveSeconds")),
            "capture": _round(timing.get("captureSeconds")),
            "match": _round(timing.get("matchSeconds")),
            "eds": _round(timing.get("edsSeconds")),
            "total": _round(timing.get("totalSeconds")),
        },
    }


def _reported_at(task_dir: Path, manifest: dict[str, Any]) -> str | None:
    reports = manifest.get("reports") or {}
    report_json = reports.get("json")
    if not report_json:
        return None
    path = Path(task_dir) / str(report_json)
    if not path.is_file():
        return None
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    generated_at = payload.get("generatedAt")
    return str(generated_at) if generated_at else None


def build_report_data(
    task_dir: Path | str,
    manifest: dict[str, Any],
    *,
    base_url: str,
) -> dict[str, Any]:
    """Build the third-party report payload from a completed task manifest.

    Raises ValueError when the task is not an optical-SEM correlation task or
    its report has not been generated yet.
    """
    task_dir = Path(task_dir)
    task_id = str(manifest.get("taskId") or task_dir.name)
    if manifest.get("taskType") != "CLEANLINESS_DETECTION":
        raise ValueError("report-data 仅支持光镜关联分析任务")
    reports = manifest.get("reports") or {}
    if not reports:
        raise ValueError("报告尚未生成，请先在软件中导出报告后再调用该接口")

    selection = manifest.get("selection") or {}
    calibration = manifest.get("calibration") or {}
    input_info = manifest.get("input") or {}
    sequences = selection.get("sequences") or []

    return {
        "apiVersion": REPORT_DATA_API_VERSION,
        "taskId": task_id,
        "taskType": manifest.get("taskType"),
        "taskStatus": manifest.get("status"),
        "exportedAt": datetime.now().astimezone().isoformat(),
        "reportedAt": _reported_at(task_dir, manifest),
        "workpiece": {
            "projectId": input_info.get("projectId"),
            "workpieceId": input_info.get("workpieceId"),
            "workpieceName": input_info.get("workpieceName"),
            "opticalParticleCount": input_info.get("particleCount"),
            "registrationMode": input_info.get("registrationMode"),
        },
        "selection": {
            "filters": selection.get("filter") or {},
            "selectedCount": len(sequences) if sequences else input_info.get("selectedCount"),
        },
        "calibration": {
            "strategy": calibration.get("strategy"),
            "accepted": bool((calibration.get("review") or {}).get("accepted")),
        },
        "summary": summarize_manifest(manifest),
        "particles": [
            _particle_view(particle, task_id=task_id, base_url=base_url)
            for particle in (manifest.get("particles") or [])
        ],
        "reports": {
            key: _file_url(base_url, task_id, reports.get(key))
            for key in ("pdf", "html", "csv", "json")
        },
        # 完整任务清单原文：需要全部内部字段的第三方可直接使用；该对象结构
        # 跟随任务文件演进，不做向后兼容承诺（顶层清洗字段才有稳定契约）。
        "manifest": manifest,
    }
