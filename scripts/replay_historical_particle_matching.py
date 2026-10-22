"""Offline mask-vs-photo replay. Historical success is NOT an identity label.

No microscope connections, no report/task mutation. Deterministic sampling groups
repeat captures by source fingerprint and sequence, avoiding train/test leakage.
"""
import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def no_network(event, args):
    if event in {"socket.connect", "socket.bind", "socket.sendto", "socket.getaddrinfo", "socket.gethostbyname"}:
        raise RuntimeError("Historical replay forbids network access")


sys.addaudithook(no_network)

import cv2
import numpy as np
from backend.cleanliness_core.optical_particle_analysis import load_optical_export
from backend.cleanliness_core.cleanliness_detection import build_sem_anchor_candidates, extract_optical_shape
from backend.cleanliness_core.cleanliness_task import _decision_for_candidate

KNOWN_TASK = "e89dead0-ec77-477f-af60-dd36948e213b"
KNOWN_WRONG = {10, 19, 177}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=Path.home()/"Documents/ZEM Cleanliness/tasks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=80)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or out.is_relative_to(args.task_root.resolve()):
        parser.error("Use a new output directory outside the task root")
    out.mkdir(parents=True)
    sources, inventory, cases, seen = {}, [], [], set()
    for path in sorted(args.task_root.glob("*/task.json")):
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        info = {"task":path.parent.name, "taskSha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "createdAt":manifest.get("createdAt"), "records":len(manifest.get("particles") or []),
                "usableCaptures":0}
        inventory.append(info)
        if not info["records"]:
            continue
        source = Path(manifest["input"]["path"])
        try:
            if str(source) not in sources:
                sources[str(source)] = load_optical_export(source)
            export = sources[str(source)]
            if export.sha256 != manifest["input"]["sha256"]:
                raise ValueError("Source hash changed")
            pixel = float(manifest["baselineImaging"]["pixelSizeUm"])
        except Exception as exc:
            info["excluded"] = str(exc)
            continue
        info["sourceSha256"] = export.sha256
        for record in manifest["particles"]:
            p = export.by_sequence[record["sequence"]]
            paths = [a.get("rawPath") for a in record.get("captureAttempts", [])]
            paths += [(record.get("images") or {}).get("semRawImage")]
            for rel in dict.fromkeys(x for x in paths if x):
                image = path.parent/rel
                if not image.is_file():
                    continue
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                key = (export.sha256, p.sequence, digest)
                if key in seen:
                    continue
                seen.add(key)
                info["usableCaptures"] += 1
                cases.append(dict(task=path.parent.name, rank=record["rank"], sequence=p.sequence,
                                  particle=p, source=export.sha256, image=image, imageSha256=digest,
                                  pixel=pixel, previous=record.get("match"),
                                  knownWrong=path.parent.name==KNOWN_TASK and p.sequence in KNOWN_WRONG))
    (out/"inventory.json").write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding="utf-8")
    chosen = []
    for source in sorted({c["source"] for c in cases}):
        group = [c for c in cases if c["source"]==source]
        groups = sorted({c["sequence"] for c in group}, key=lambda s:hashlib.sha256(f"{source}:{s}".encode()).hexdigest())
        # One capture per physical identity for bounded replay; keep all known wrong captures.
        for seq in groups[:args.per_source]:
            chosen.append(next(c for c in group if c["sequence"]==seq))
    chosen += [c for c in cases if c["knownWrong"] and not any(x is c for x in chosen)]
    print(f"inventoried={len(cases)} unique captures; replay={len(chosen)}",flush=True)
    results = []
    for index,c in enumerate(chosen):
        image = cv2.imdecode(np.frombuffer(c["image"].read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED)
        row = {k:v for k,v in c.items() if k not in {"particle","image"}}
        row["image"] = str(c["image"])
        for label,p in [("photo",replace(c["particle"],mask_data_base64="")),("mask",c["particle"])]:
            try:
                shape = extract_optical_shape(p)
                candidates = build_sem_anchor_candidates(p,image,pixel_size_um=c["pixel"],stage_center_um=(0,0))
                if not candidates:
                    row[label] = {"status":"NO_CANDIDATE"}
                    continue
                first = candidates[0]
                decision = _decision_for_candidate(first,len(candidates),reason_prefix="offline replay")
                row[label] = {"status":decision.status,"shapeSource":shape.source if shape else None,
                              "center":list(first.pixel_xy),"shapeScore":first.shape_score,
                              "lengthRatio":first.candidate.length_ratio,"widthRatio":first.candidate.width_ratio,
                              "areaRatio":first.candidate.area_ratio,
                              "candidates":[{"center":list(x.pixel_xy),"score":x.combined_score} for x in candidates]}
            except Exception as exc:
                row[label] = {"status":"ERROR","error":str(exc)}
        if "center" in row["photo"] and "center" in row["mask"]:
            row["centerChangePx"] = float(np.linalg.norm(np.subtract(row["photo"]["center"],row["mask"]["center"])))
        results.append(row)
        with (out/"replay.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(row,ensure_ascii=False)+"\n")
        if index%20==0:
            print(f"replayed={index+1}/{len(chosen)}",flush=True)
    summary = {"segmentation":"classic-only controlled comparison, not exact historical configuration",
               "uniqueCaptures":len(cases),"tasks":len(inventory),"replayed":len(results),
               "sources":len({c['source'] for c in cases}),
               "notice":"Unlabelled replay measures changes, not accuracy. Known wrong labels apply only to historical selected centers.",
               "bySource":[]}
    for source in sorted({r["source"] for r in results}):
        rows = [r for r in results if r["source"]==source]
        summary["bySource"].append(dict(source=source,count=len(rows),
            photoAccepted=sum(r["photo"]["status"]=="MATCHED_HIGH" for r in rows),
            maskAccepted=sum(r["mask"]["status"]=="MATCHED_HIGH" for r in rows),
            changedOver5Px=sum(r.get("centerChangePx",0)>5 for r in rows),
            errors=sum(r["mask"]["status"]=="ERROR" for r in rows)))
    summary["knownWrong"]=[r for r in results if r["knownWrong"]]
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!='knownWrong'},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
