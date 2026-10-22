"""Re-run the production pre-EDS identity gate on a saved offline replay manifest."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.replay_historical_particle_matching import no_network  # installs offline audit hook
import cv2
import numpy as np
from loguru import logger
from backend.cleanliness_core.optical_particle_analysis import load_optical_export
from backend.cleanliness_core.cleanliness_detection import build_sem_anchor_candidates, CandidateSegmentationConfig, CandidateGate
from backend.cleanliness_core.cleanliness_task import _decision_for_candidate, _verify_candidate_mask_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay",type=Path,required=True)
    parser.add_argument("--task-root",type=Path,default=Path.home()/"Documents/ZEM Cleanliness/tasks")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--saved-settings",action="store_true",help="Replay local segmentation and gates from each saved settings snapshot")
    args=parser.parse_args()
    logger.remove()
    if args.output.exists() or args.output.resolve().is_relative_to(args.task_root.resolve()):
        parser.error("Use a fresh output outside the task root")
    rows=[json.loads(line) for line in args.replay.read_text(encoding="utf-8").splitlines()]
    exports={}
    results=[]
    for row in rows:
        manifest=json.loads((args.task_root/row['task']/"task.json").read_text(encoding="utf-8-sig"))
        source=manifest['input']['path']
        if source not in exports:
            exports[source]=load_optical_export(Path(source))
        export=exports[source]
        assert export.sha256==row['source']
        p=export.by_sequence[row['sequence']]
        if not p.mask_data_base64:
            continue
        result={k:row[k] for k in ('task','rank','sequence','knownWrong','image','imageSha256')}
        try:
            segmentation=None
            gate=None
            if args.saved_settings:
                settings_path=args.task_root/row['task']/manifest['settingsSnapshot']
                settings=json.loads(settings_path.read_text(encoding='utf-8-sig'))['calibration']
                segmentation=CandidateSegmentationConfig.from_dict(settings.get('candidateSegmentation'),settings_path=settings_path)
                if segmentation.backend=='sam3':
                    raise ValueError('Remote segmentation excluded from offline replay')
                gate=CandidateGate.from_dict(settings.get('candidateGate'))
                result['segmentation']=segmentation.backend
            image=cv2.imdecode(np.frombuffer(Path(row['image']).read_bytes(),np.uint8),-1)
            import hashlib
            assert hashlib.sha256(Path(row['image']).read_bytes()).hexdigest()==row['imageSha256']
            candidates=build_sem_anchor_candidates(p,image,pixel_size_um=row['pixel'],stage_center_um=(0,0),segmentation=segmentation,gate=gate)
            decision=_decision_for_candidate(candidates[0],len(candidates),reason_prefix="offline",gate=gate)
            decision,audit=_verify_candidate_mask_identity(image,p,row['pixel'],decision)
            result.update(status=decision.status,center=list(candidates[0].pixel_xy),verification=audit)
        except Exception as exc:
            result.update(status="REJECTED_INPUT_OR_CANDIDATE",error=str(exc))
        results.append(result)
        if len(results)%20==0:print(f"verified={len(results)}",flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    from collections import Counter
    print(json.dumps(dict(Counter(r['status'] for r in results)),ensure_ascii=False))
    print(json.dumps([r for r in results if r['knownWrong']],ensure_ascii=False))


if __name__=="__main__":main()
