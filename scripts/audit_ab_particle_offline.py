"""Replay saved calibration frames only; no microscope/client imports or network.

Saved correspondences are comparison references, not new ground truth. Output must
be outside all source evidence directories, so acquisition logs stay immutable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def deny_network(event, _args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto"}:
        raise RuntimeError("Offline replay forbids network access")


sys.addaudithook(deny_network)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from backend.cleanliness_core.atomic_json import write_json_atomic
from backend.cleanliness_core.optical_particle_analysis import load_optical_export
from backend.cleanliness_core.particle_refinement import RefinementSettings, match_particle, measure_pixel_axes, select_plan, verify_particle_context


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_image(path):
    # imdecode supports non-ASCII Windows paths, unlike some OpenCV imread builds.
    result = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if result is None:
        raise ValueError(f"Cannot decode saved image: {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, action="append", default=[])
    parser.add_argument("--historical", type=Path, help="Saved final_validation_120 directory")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", action="store_true", help="Also replay new same-frame seed context gate")
    parser.add_argument("--axes-only", action="store_true", help="Only replay six-frame axis groups")
    args = parser.parse_args()
    output = args.output.resolve()
    sources = [p.resolve() for p in args.task] + ([args.historical.resolve()] if args.historical else [])
    if not sources or any(output.is_relative_to(p) for p in sources) or output == args.input.resolve():
        parser.error("Output must be separate from input/evidence directories")
    export = load_optical_export(args.input)
    cfg = RefinementSettings()
    result = {"kind": "offline_saved_frame_replay", "networkAccess": "blocked",
              "inputSha256": export.sha256, "acceptedTransform": False,
              "notice": "Development replay, not blind end-to-end validation or a new accuracy claim.",
              "plan": select_plan(export.particles, cfg), "matches": [], "axes": []}

    def replay(raw_path, sequence, pixel, reference, source, stage=None, axis=None, context=None):
        image = read_image(raw_path)
        match = match_particle(image, export.by_sequence[sequence], pixel, cfg)
        if args.context and context and axis and match["status"] == "MATCH":
            verification = verify_particle_context(
                image, export.by_sequence[sequence], match, export.particles, context["matrix"],
                np.asarray(axis["pixelToStageMatrix"]), pixel, cfg, excluded_sequences=context["excluded"])
            match["contextVerification"] = verification
            if verification["status"] != "VERIFIED":
                match.update(status="CONTEXT_UNVERIFIED", reason=verification["reason"])
        record = {"source": source, "sequence": sequence, "rawPath": str(raw_path.resolve()),
                  "previous": reference, "current": match}
        if match.get("centerPixel") and reference.get("centerPixel"):
            record["centerChangePx"] = float(np.linalg.norm(np.subtract(match["centerPixel"], reference["centerPixel"])))
        if axis and stage is not None and match.get("centerPixel"):
            record["currentMeasuredStageUm"] = (np.asarray(stage) + np.asarray(axis["pixelToStageMatrix"]) @
                                                  (np.asarray(match["centerPixel"]) - np.array(image.shape[::-1]) / 2)).tolist()
        result["matches"].append(record)
        write_json_atomic(output, result)
        print(f"{source} #{sequence}: {reference.get('status', 'historical')} -> {match['status']} "
              f"score={match.get('score', 0):.3f} margin={match.get('margin', 0):.3f}", flush=True)

    for task in args.task:
        manifest = read_json(task / "task.json")
        if manifest["input"]["sha256"] != export.sha256:
            raise ValueError("Saved task/input hash mismatch")
        audit = manifest["calibration"]["abPointResult"]["semRefinement"]
        shots = {s["index"]: s for s in audit["captures"]}
        for observation in ([] if args.axes_only else audit["observations"]):
            initial = audit.get("navigationInitial", {}).get("searchTransform")
            linear = np.array((initial or manifest["calibration"]["abPointResult"]["previewTransform"])["matrix"])
            if observation.get("role") == "seed_h1":
                ab_source = manifest["calibration"]["abPointResult"]["sourceReferencePointsUm"]
                direction = np.subtract(ab_source["b"], ab_source["a"])
                direction = direction / np.linalg.norm(direction)
                linear = linear @ (2 * np.outer(direction, direction) - np.eye(2))
            for attempt in observation["attempts"]:
                if "captureIndex" not in attempt:
                    continue
                shot = shots[attempt["captureIndex"]]
                raw_path = task / audit["runDirectory"] / Path(shot["capture"]["rawPath"]).name
                replay(raw_path, observation["sequence"], shot["capture"]["pixelSizeUm"], attempt,
                       f"{task.name}/{shot['index']:04d}", shot["captureStageUm"], audit.get("pixelAxes", {}).get(shot["phase"]),
                       {"matrix": linear, "excluded": audit["plan"]["validation"]} if observation["role"].startswith("seed") else None)
        # Each axis trial is six consecutive labeled captures, including failed trials.
        values = list(shots.values())
        for index, shot in enumerate(values):
            if "像素轴原点" not in shot["label"]:
                continue
            group = values[index:index + 6]
            if len(group) != 6 or any("capture" not in s or "像素轴" not in s["label"] for s in group):
                continue
            axis_record = {"source": f"{task.name}/{shot['index']:04d}", "phase": shot["phase"]}
            names = ("o1", "xp", "xm", "yp", "ym", "o2")
            axis_record["imagePaths"] = {n: str((task / audit["runDirectory"] / Path(s["capture"]["rawPath"]).name).resolve()) for n, s in zip(names, group)}
            try:
                axis_record.update(status="PASS", result=measure_pixel_axes(
                    {n: read_image(task / audit["runDirectory"] / Path(s["capture"]["rawPath"]).name) for n, s in zip(names, group)},
                    {n: s["captureStageUm"] for n, s in zip(names, group)}, shot["capture"]["pixelSizeUm"], cfg))
            except ValueError as exc:
                axis_record.update(status="REJECTED", reason=str(exc), retryable=getattr(exc, "retryable", True),
                                   diagnostics=getattr(exc, "diagnostics", {}))
            result["axes"].append(axis_record)
    if args.historical:
        references = {r["sequence"]: r for r in read_json(args.historical / "matches.json")}
        for shot in read_json(args.historical / "manifest.json")["shots"]:
            replay(args.historical / Path(shot["capture"]["rawPath"]).name, shot["sequence"],
                   shot["capture"]["pixelSizeUm"], references[shot["sequence"]], "historical_final")
    write_json_atomic(output, result)
    print(f"offline_audit={output}", flush=True)


if __name__ == "__main__":
    main()
