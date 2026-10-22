from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
COARSE_MATRIX = np.asarray(
    [
        [0.9881286283083116, -0.0021702850846471167],
        [0.009980298756535467, -1.0176428402411026],
    ],
    dtype=np.float64,
)
COARSE_TRANSLATION = np.asarray(
    [-26175.027570812865, 27550.87565867192], dtype=np.float64
)
ROI_CENTER_UM = np.asarray([1013.6051769934911, 869.1706924849896])
ROI_RADIUS_UM = 25798.31079908176


def parse_pair(value: str) -> tuple[float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected x,y")
    return parts[0], parts[1]


APPROACH_OFFSET_UM = (-150.0, -150.0)


def move_converged(
    microscope: Any,
    target: tuple[float, float],
    *,
    tolerance_um: float = 2.0,
    max_attempts: int = 6,
    anti_backlash: bool = True,
) -> dict[str, Any]:
    """闭环移动直到实测位置收敛到目标。

    Zem20F 的 ensure_stage_idle 在长距离移动中途的短暂停顿会误判为到位，
    所以必须重复下发 goto 并复核实测位置。
    anti_backlash: 先到 target+(-150,-150) 再进入目标，统一最终接近方向，
    消除长距离移动的迟滞（实测可达50um量级）。
    """
    import time as _time

    attempts: list[dict[str, Any]] = []
    before = microscope.get_position()
    if anti_backlash:
        approach = (target[0] + APPROACH_OFFSET_UM[0], target[1] + APPROACH_OFFSET_UM[1])
        for attempt in range(1, max_attempts + 1):
            result = microscope.move_xy(float(approach[0]), float(approach[1]))
            if float(result["positionErrorUm"]) <= 25.0:
                break
        attempts.append({"phase": "approach", "afterUm": result["after"]})
    for attempt in range(1, max_attempts + 1):
        result = microscope.move_xy(float(target[0]), float(target[1]))
        error = float(result["positionErrorUm"])
        attempts.append({"attempt": attempt, "afterUm": result["after"], "errorUm": error})
        if error <= tolerance_um:
            # 再等1秒复读，确认没有继续漂移
            _time.sleep(1.0)
            check = microscope.get_position()
            drift = math.hypot(check[0] - target[0], check[1] - target[1])
            attempts[-1]["recheckUm"] = list(check)
            attempts[-1]["recheckErrorUm"] = drift
            if drift <= tolerance_um:
                return {
                    "before": {"x": before[0], "y": before[1]},
                    "target": {"x": float(target[0]), "y": float(target[1])},
                    "after": {"x": check[0], "y": check[1]},
                    "positionErrorUm": drift,
                    "attempts": attempts,
                }
    raise RuntimeError(f"舞台移动{max_attempts}次仍未收敛到目标: {attempts}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    item = next(
        (entry for entry in payload.get("particles") or [] if int(entry["sequence"]) == args.sequence),
        None,
    )
    if item is None:
        raise ValueError(f"Particle #{args.sequence} is not present in the JSON")
    position = item.get("position") or {}
    source_xy = np.asarray([float(position["x"]), float(position["y"])])
    stage_offset = np.asarray(args.stage_offset_um, dtype=np.float64)
    if args.center_um is not None:
        target = np.asarray(args.center_um, dtype=np.float64) + stage_offset
    else:
        target = COARSE_MATRIX @ source_xy + COARSE_TRANSLATION + stage_offset
    distance = float(np.linalg.norm(target - ROI_CENTER_UM))
    if distance > ROI_RADIUS_UM:
        raise RuntimeError(
            f"Target is outside the safe ROI: {distance:.3f} > {ROI_RADIUS_UM:.3f} um"
        )

    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from backend.cleanliness_core.optical_particle_task import ZemMicroscope

    field_size_um = 127000.0 / args.magnification
    microscope = ZemMicroscope(
        args.host,
        args.port,
        {
            "imageShape": [1024, 1024],
            "magnification": args.magnification,
            "fieldSizeUm": field_size_um,
        },
        pixel_avg=args.pixel_avg,
    )
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "sequence": args.sequence,
        "startedAt": datetime.now().astimezone().isoformat(),
        "sourcePositionUm": source_xy.tolist(),
        "coarseTargetUm": (COARSE_MATRIX @ source_xy + COARSE_TRANSLATION).tolist(),
        "appliedStageOffsetUm": stage_offset.tolist(),
        "targetUm": target.tolist(),
        "requestedImaging": {
            "magnification": args.magnification,
            "fieldSizeUm": field_size_um,
            "imageShape": [1024, 1024],
        },
        "safeRoi": {
            "centerUm": ROI_CENTER_UM.tolist(),
            "radiusUm": ROI_RADIUS_UM,
            "targetDistanceUm": distance,
        },
        "gunStateChanged": False,
    }
    start_position: tuple[float, float] | None = None
    try:
        result["connection"] = microscope.connect()
        device = microscope.api.device
        detector = str(getattr(device.m, "ed_chanel", "BSE")).upper()
        result["detector"] = detector
        if detector != "BSE":
            raise RuntimeError(f"BSE is required, but the active detector is {detector}")
        start_position = microscope.get_position()
        result["startStageUm"] = list(start_position)
        result["hvStatusBefore"] = microscope.get_hv_status()
        result["moveToTarget"] = move_converged(
            microscope, (float(target[0]), float(target[1]))
        )
        raw_path = output_dir / f"{args.sequence:04d}_attempt1.png"
        _image, capture = microscope.capture(raw_path)
        result["capture"] = capture
        # 拍摄后复读位置，确认拍摄期间舞台没有移动
        post = microscope.get_position()
        result["postCaptureStageUm"] = list(post)
        result["postCaptureDriftUm"] = math.hypot(
            post[0] - float(target[0]), post[1] - float(target[1])
        )
        result["status"] = "CAPTURED"
    except Exception as error:
        result["status"] = "FAILED"
        result["error"] = str(error)
        raise
    finally:
        if start_position is not None:
            try:
                result["restoreMove"] = move_converged(microscope, start_position)
            except Exception as restore_error:
                result["restoreError"] = str(restore_error)
        try:
            microscope.restore_video()
        except Exception:
            pass
        result["completedAt"] = datetime.now().astimezone().isoformat()
        write_json(output_dir / f"{args.sequence:04d}_capture.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one A2081 SEM calibration point")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--host", default="192.168.0.15")
    parser.add_argument("--port", type=int, default=61557)
    parser.add_argument("--pixel-avg", type=int, default=7)
    parser.add_argument("--magnification", type=float, default=120.0)
    parser.add_argument("--stage-offset-um", type=parse_pair, default=(-88.7, -2.6))
    parser.add_argument(
        "--center-um",
        type=parse_pair,
        default=None,
        help="直接指定拍摄中心舞台坐标(um)，跳过粗矩阵预测；offset仍会叠加",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
