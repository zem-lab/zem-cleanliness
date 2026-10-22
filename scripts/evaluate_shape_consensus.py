"""Record final-pipeline decisions for real A/B fixes and preserved competitors."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.cleanliness_core.cleanliness_detection import CandidateSegmentationConfig,build_sem_anchor_candidates
from backend.cleanliness_core.cleanliness_task import _select_mask_guided_candidate
from backend.cleanliness_core.optical_particle_analysis import load_optical_export
logger.remove()

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--perturbations',type=Path)
args=parser.parse_args()
if args.output.exists():parser.error('Use a new output directory')
args.output.mkdir(parents=True)
model=ROOT/'resources/models/cleanliness/particle_yolo11n_seg_reann_20260811_best.onnx'
seg=CandidateSegmentationConfig(backend='yolo',yolo_model_path=model)
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'modelSha256':hashfile(model),'deviceOperations':False,'historicalRecordsOverwritten':False,
        'codeSha256':{name:hashfile(ROOT/'backend/cleanliness_core'/name) for name in ['cleanliness_task.py','shape_consensus.py','shape_comparison.py','particle_refinement.py']},'cases':[]}
def save_case(name,case,particle,pixel,path,expected,reference,kind):
    raw=path.read_bytes()
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED)
    candidates=build_sem_anchor_candidates(particle,image,pixel_size_um=pixel,stage_center_um=(0,0),segmentation=seg)
    chosen,decision,audit=_select_mask_guided_candidate(image,particle,pixel,candidates,stage_center_um=(0,0),segmentation=seg)
    center=list(chosen.pixel_xy) if chosen is not None else None
    distance=float(np.linalg.norm(np.asarray(center)-reference)) if center and reference else None
    assert decision.matched is expected,(case,audit)
    if expected:
        assert distance is not None and distance<=12,(case,center,reference)
        assert audit['accepted'] and audit['maskIoU']>=.5
    else:
        assert not audit['accepted']
    report['cases'].append({'corpus':name,'rank':case.get('reportRank',case.get('rank')),
        'sequence':particle.sequence,'attempt':case.get('attempt'),'particleId':particle.particle_id,
        'sourceImage':str(path.resolve()),'imageSha256':hashlib.sha256(raw).hexdigest(),
        'referenceKind':kind,'expectedAccepted':expected,'referenceCenterPx':reference,
        'accepted':decision.matched,'selectedCenterPx':center,'referenceDistancePx':distance,
        'decision':decision.to_dict(),'maskVerification':audit})
    (args.output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'done':len(report['cases']),'corpus':name,'rank':case.get('reportRank',case.get('rank')),
                      'accepted':decision.matched,'referenceDistancePx':distance}),flush=True)

for name in ['human_confirmed_20260918','template_identity_cases']:
    folder=ROOT/'backend/tests/fixtures'/name
    spec=json.loads((folder/'cases.json').read_text(encoding='utf-8'))
    assert hashfile(folder/'optical.json')==spec.get('opticalSubsetSha256',spec.get('opticalSha256'))
    export=load_optical_export(folder/'optical.json')
    for case in spec['cases']:
        if name=='template_identity_cases' and case['rank']!=118:continue
        path=folder/case['image'];assert hashfile(path)==case['imageSha256']
        new=name=='human_confirmed_20260918'
        reference=case['correctTarget']['referenceCenterPx'] if new else None
        save_case(name,case,export.by_sequence[case['sequence']],spec['pixelSizeUm'],path,new,reference,
                  case.get('reviewStatus','compatibility_rejection_baseline'))
if args.perturbations:
    folder=ROOT/'backend/tests/fixtures/human_confirmed_20260918'
    spec=json.loads((folder/'cases.json').read_text(encoding='utf-8'))
    export=load_optical_export(folder/'optical.json');byid={c['caseId']:c for c in spec['cases']}
    variants=json.loads((args.perturbations/'results.json').read_text(encoding='utf-8'))
    for v in variants:
        case=byid[v['caseId']];path=args.perturbations/v['image']
        assert hashfile(path)==v['imageSha256']
        save_case('derived_'+v['condition'],case,export.by_sequence[case['sequence']],spec['pixelSizeUm'],path,True,
                  v['correctTargetPx'],'derived_robustness_check')
report['summary']={'cases':len(report['cases']),'accepted':sum(c['accepted'] for c in report['cases']),
                   'preservedRejections':sum(not c['accepted'] for c in report['cases'])}
(args.output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report['summary']),flush=True)
