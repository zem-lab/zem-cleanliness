"""Headless EDS point-analysis flow used by cleanliness tasks.

This module intentionally contains no Qt or other desktop-UI dependencies.
"""

from __future__ import annotations

from datetime import datetime
import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

from backend.cleanliness_core.eds.client_v2 import EdsV2Client, EdsV2Error
from backend.cleanliness_core.eds.geometry import particle_center_pixel
from backend.cleanliness_core.eds.microscope_jsonrpc import start_scan_points
from backend.cleanliness_core.eds.generated.v2 import job_events_pb2


def _classification(material):
    return {'materialClass':material.material_class,'parentClass':material.parent_class,
            'hardnessLabel':material.hardness_label,'hardnessBand':material.hardness_band,
            'confidence':float(material.confidence),'matchedCriteria':list(material.matched_criteria),
            'warnings':list(material.warnings),'isUnclassified':bool(material.is_unclassified),
            'boundaryNotes':list(material.boundary_notes)}


def _completed_details(completed):
    # proto3 scalar zero means either absent or explicitly zero. Do not invent
    # zero acquisition durations for responses from older servers.
    present = {field.name for field, _ in completed.ListFields()}
    details = {'finalResultOnly':True}
    for source, target in [('real_time_seconds','realTimeSeconds'),('live_time_seconds','liveTimeSeconds'),
                           ('target_counts','targetCounts'),('started_at','startedAt'),('ended_at','endedAt'),
                           ('quant_method','quantMethod'),('classification_report','classificationReport')]:
        if source in present:
            details[target] = getattr(completed, source)
    for source,target in [('material_classification','materialClassification'),('vda_classification','vdaClassification')]:
        if completed.HasField(source): details[target] = _classification(getattr(completed,source))
    for source,target in [('oxford_weight_percent','oxfordWeightPercent'),('vda_analysis_composition','vdaAnalysisComposition')]:
        if source in present: details[target] = dict(getattr(completed,source))
    if completed.element_quantifications:
        details['elementQuantifications'] = [{'symbol':q.element,'value':float(q.weight_percent),
            **({'sigma':float(q.weight_percent_sigma)} if q.HasField('weight_percent_sigma') else {})}
            for q in completed.element_quantifications]
    code = int(completed.stop_reason)
    try: reason = job_events_pb2.SpectrumStopReason.Name(code)
    except ValueError: reason = f'UNKNOWN_{code}'
    details.update(stopReason=reason,stopReasonCode=code)
    target = details.get('targetCounts')
    details['targetReached'] = completed.accumulated_counts >= target if target and 'accumulated_counts' in present else None
    return details


def _event_elements(event: Any) -> list[dict[str, Any]]:
    payload_name = event.WhichOneof("payload")
    rows: list[dict[str, Any]] = []
    if payload_name == "point_measured":
        result = event.point_measured.result
        for concentration in result.concentrations if result else ():
            rows.append(
                {
                    "symbol": concentration.element,
                    "value": round(float(concentration.concentration), 2),
                }
            )
    elif payload_name == "completed":
        completed = event.completed
        if completed.quantify_result_map:
            rows.extend(
                {"symbol": symbol, "value": float(value)}
                for symbol, value in completed.quantify_result_map.items()
            )
        elif completed.index_concentrations:
            for indexed in completed.index_concentrations:
                rows.extend(
                    {
                        "symbol": concentration.element,
                        "value": round(float(concentration.concentration), 2),
                    }
                    for concentration in indexed.concentrations
                )
    return rows


def process_energy_spectrum_particle(
    eds_v2: EdsV2Client,
    mask: dict[str, Any],
    image_path: str,
    zem_host: str,
    zem_port: int,
    *,
    target_point: tuple[int, int] | None = None,
    target_mode: str | None = None,
) -> dict[str, Any]:
    """Move the beam and run one complete EDS V2 point-analysis job."""

    label = str(mask.get("label", ""))
    points = mask.get("points", [])
    if target_mode not in {None, "large_particle_center_region"}:
        raise ValueError("Unsupported EDS target mode")
    if target_mode is not None and target_point is None:
        raise ValueError("Center-region EDS requires an explicit target point")
    if target_point is None:
        x, y = particle_center_pixel(points)
        target_method = "particle_geometric_center"
    else:
        x, y = int(target_point[0]), int(target_point[1])
        target_method = target_mode or "particle_safe_interior_point"
        if target_mode == "large_particle_center_region":
            from PIL import Image
            with Image.open(image_path) as captured:
                width,height=captured.size
            if target_point[0]!=x or target_point[1]!=y or not (0<=x<width and 0<=y<height):
                raise ValueError("Center-region EDS point is outside the captured image or is not an integer pixel")

    logger.info(
        "EDS particle analysis start label={} point=({}, {}) microscope={}:{} eds={}",
        label,
        x,
        y,
        zem_host,
        zem_port,
        eds_v2.target,
    )
    call_started = time.perf_counter()
    start_scan_points(zem_host, zem_port, [(x, y)])

    session_id = ""
    elements: list[dict[str, Any]] = []
    result_text = ""
    spectrum_png = b""
    spectrum_image_info: dict[str, Any] = {}
    material_classification: dict[str, Any] | None = None
    elapsed_seconds: float | None = None
    accumulated_counts: int | None = None
    job_id = ''
    event_path = None
    event_handle = None
    final_details = {}
    try:
        session_id = eds_v2.open_session()
        job_id = eds_v2.start_spectrum_job(session_id)
        if Path(image_path).is_file():
            event_dir = Path(image_path).resolve().parent.parent/'eds_events'
            event_dir.mkdir(parents=True,exist_ok=True)
            safe_job = ''.join(c for c in str(job_id) if c.isalnum() or c in '-_')[:100]
            event_path = event_dir/f'{label if label.isdecimal() else "particle"}-{safe_job}.jsonl'
            event_handle = event_path.open('a',encoding='utf-8')
        job_finished = False
        for event in eds_v2.watch_job(session_id, job_id):
            payload_name = event.WhichOneof("payload")
            if event_path is not None:
                from google.protobuf.json_format import MessageToDict
                log_event = event
                omitted_png = False
                if payload_name in {'snapshot','completed'}:
                    payload_message = getattr(event,payload_name)
                    if payload_message.HasField('spectrum_image') and payload_message.spectrum_image.png:
                        log_event = type(event)()
                        log_event.CopyFrom(event)
                        getattr(log_event,payload_name).spectrum_image.ClearField('png')
                        omitted_png = True
                event_record = MessageToDict(log_event,preserving_proto_field_name=True)
                # Preserve status, phase and errors without repeating PNG blobs
                # in each streamed update. Completed spectrum PNG is saved below.
                payload = event_record.get(payload_name,{})
                if omitted_png:
                    payload.setdefault('spectrum_image',{})['png'] = '[omitted from event log]'
                event_handle.write(json.dumps(event_record,ensure_ascii=False)+'\n')
                event_handle.flush()
            if payload_name == "snapshot" and event.snapshot.HasField("spectrum_image"):
                image = event.snapshot.spectrum_image
                if image.png:
                    spectrum_png = bytes(image.png)
                    spectrum_image_info = {
                        "mediaType": image.media_type or "image/png",
                        "width": int(image.width),
                        "height": int(image.height),
                    }
            if payload_name == "completed":
                completed = event.completed
                final_details = _completed_details(completed)
                elements = ([{'symbol':q.element,'value':float(q.weight_percent)} for q in completed.element_quantifications]
                            if completed.element_quantifications else _event_elements(event))
                # A preview is never promoted to the final spectrum image.
                spectrum_png = b''
                spectrum_image_info = {}
                result_text = completed.result_text
                present = {field.name for field, _ in completed.ListFields()}
                elapsed_seconds = float(completed.elapsed_seconds) if 'elapsed_seconds' in present else None
                accumulated_counts = int(completed.accumulated_counts) if 'accumulated_counts' in present else None
                if completed.HasField("spectrum_image") and completed.spectrum_image.png:
                    image = completed.spectrum_image
                    spectrum_png = bytes(image.png)
                    spectrum_image_info = {
                        "mediaType": image.media_type or "image/png",
                        "width": int(image.width),
                        "height": int(image.height),
                    }
                if completed.HasField("material_classification"):
                    material_classification = final_details['materialClassification']
                requested_counts = getattr(eds_v2, "target_counts", None)
                if requested_counts is not None:
                    final_details["requestedTargetCounts"] = requested_counts
                    if final_details.get("targetCounts") != requested_counts:
                        raise EdsV2Error(
                            f"EDS 目标计数回读不一致：请求 {requested_counts}，返回 {final_details.get('targetCounts')}",
                            terminal=True, code="TARGET_COUNTS_MISMATCH", session_id=session_id, job_id=job_id,
                        )
                job_finished = True
                break
            if payload_name == "failed":
                raise EdsV2Error(
                    f"Particle {label} EDS job failed: "
                    f"code={event.failed.error_code}, msg={event.failed.error_message}",
                    terminal=True, code=event.failed.error_code, session_id=session_id, job_id=job_id,
                )
            if payload_name == "cancelled":
                raise EdsV2Error(
                    f"Particle {label} EDS job cancelled: {event.cancelled.reason}"
                )
        if not job_finished:
            raise EdsV2Error(
                f"Particle {label} EDS job ended without a completed result"
            )
    except EdsV2Error as error:
        error.session_id = error.session_id or session_id
        error.job_id = error.job_id or job_id
        if event_path is not None and spectrum_png:
            event_path.with_suffix('.partial.png').write_bytes(spectrum_png)
        raise
    finally:
        try:
            if event_handle is not None:
                event_handle.close()
        finally:
            if session_id:
                eds_v2.close_session(session_id)

    unique_by_symbol: dict[str, dict[str, Any]] = {}
    for element in elements:
        unique_by_symbol[str(element["symbol"])] = element
    unique_elements = sorted(
        unique_by_symbol.values(), key=lambda element: element["value"], reverse=True
    )
    logger.info(
        "EDS particle analysis done label={} elements={}", label, len(unique_elements)
    )
    result = {
        "schema_version": "1.0",
        "source": "grpc_v2",
        "mock": False,
        "device": eds_v2.target,
        "method": "EDS",
        "unit": "percent",
        "unit_label": "元素含量(%)",
        "particle_label": label,
        "image_path": image_path,
        "acquired_at": datetime.now().isoformat(timespec="seconds"),
        "sessionId": session_id,
        "jobId": job_id,
        "target": {"mode": target_method, "x": x, "y": y},
        "zem_target": f"{zem_host}:{zem_port}",
        "result_text": result_text,
        "elements": unique_elements,
        **final_details,
    }
    if elapsed_seconds is not None:
        result["elapsedSeconds"] = elapsed_seconds
    if accumulated_counts is not None:
        result["accumulatedCounts"] = accumulated_counts
    if material_classification is not None:
        result["materialClassification"] = material_classification
    if spectrum_png:
        task_dir = Path(image_path).resolve().parent.parent
        spectrum_dir = task_dir / "spectra"
        spectrum_dir.mkdir(parents=True, exist_ok=True)
        try:
            spectrum_name = f"{int(label):04d}.png"
        except ValueError:
            spectrum_name = "spectrum.png"
        spectrum_path = spectrum_dir / spectrum_name
        spectrum_path.write_bytes(spectrum_png)
        result["spectrumImage"] = {
            **spectrum_image_info,
            "path": spectrum_path.relative_to(task_dir).as_posix(),
            "bytes": len(spectrum_png),
        }
    result['clientCallSeconds'] = time.perf_counter()-call_started
    return result
