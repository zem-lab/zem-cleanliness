"""Compare old unmasked captures with immutable committed code, not new output.

Writes a separate version review; never changes imported historical expectations.
"""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));logger.remove()
from backend.cleanliness_core import cleanliness_task as working_task
from backend.cleanliness_core.cleanliness_detection import CandidateSegmentationConfig,build_sem_anchor_candidates,yolo_seg_candidates
from backend.cleanliness_core.optical_particle_analysis import load_optical_export


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


folder=ROOT/'backend/tests/fixtures/matching_expansion_20260918'
destination=folder/'legacy_version_review.json'
if destination.exists():raise RuntimeError('Review already exists; do not overwrite audited baselines')
corpus=json.loads((folder/'cases.json').read_text(encoding='utf-8'))
commit=git('rev-parse','HEAD').decode().strip()
sources={name:git('show',f'{commit}:backend/cleanliness_core/{name}') for name in [
    'cleanliness_task.py','cleanliness_detection.py','optical_particle_analysis.py']}
# The candidate generation and optical decoding used for the committed replay
# must be byte-equivalent after line-ending normalization to committed code.
for name in ['cleanliness_detection.py','optical_particle_analysis.py']:
    current=(ROOT/'backend/cleanliness_core'/name).read_text(encoding='utf-8').replace('\r\n','\n')
    assert current==sources[name].decode('utf-8').replace('\r\n','\n'),name
tree=ast.parse(sources['cleanliness_task.py'].decode('utf-8'))
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'_decision_for_candidate','_select_mask_guided_candidate'}]
assert len(nodes)==2
namespace=dict(vars(working_task))
exec(compile(ast.Module(body=nodes,type_ignores=[]),f'committed:{commit}:cleanliness_task.py','exec'),namespace)
committed_selector=namespace['_select_mask_guided_candidate']
spec=next(s for s in corpus['sources'] if s['id']=='legacy_without_mask')
export=load_optical_export(folder/spec['optical']);pixel=spec['pixelSizeUm']
seg=CandidateSegmentationConfig(backend='yolo',yolo_model_path=ROOT/corpus['model'])
review={'schemaVersion':1,'referenceCommit':commit,
    'referenceCodeSha256':{name:hashlib.sha256(raw).hexdigest() for name,raw in sources.items()},
    'importedHistoricalManifestSha256':hashlib.sha256((folder/'cases.json').read_bytes()).hexdigest(),
    'modelSha256':corpus['modelSha256'],
    'labelMeaning':'Versioned compatibility only. Neither old nor committed selection establishes human identity truth.',
    'reason':'These no-mask frames predate the current workflow. Six disagreements already reproduce with immutable committed HEAD, before the local shape-consensus changes. Candidate-generation and optical-decoding sources equal that committed version. Freeze its behavior separately; retain every imported historical result.',
    'identityReviewStatus':'unconfirmed; historical and current coordinates are not human target labels',
    'cases':[]}
for case in corpus['cases']:
    if case['source']!='legacy_without_mask':continue
    particle=export.by_sequence[case['sequence']];assert not particle.mask_data_base64
    raw=(folder/case['image']).read_bytes();assert hashlib.sha256(raw).hexdigest()==case['imageSha256']
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED)
    candidates=build_sem_anchor_candidates(particle,image,pixel_size_um=pixel,stage_center_um=(0,0),segmentation=seg)
    committed=committed_selector(image,particle,pixel,candidates,stage_center_um=(0,0),segmentation=seg)
    current=working_task._select_mask_guided_candidate(image,particle,pixel,candidates,stage_center_um=(0,0),segmentation=seg)
    chosen,decision,audit=committed
    assert audit is current[2] is None
    assert decision.to_dict()==current[1].to_dict()
    assert chosen.pixel_xy==current[0].pixel_xy
    historical_center=(case['historicalMatch'].get('best')or{}).get('centerPx')
    distance=float(np.linalg.norm(np.subtract(chosen.pixel_xy,historical_center))) if historical_center else None
    predicted_yolo=yolo_seg_candidates(image,expected_length_px=particle.length_um/pixel,
        expected_width_px=particle.width_um/pixel,expected_area_px=particle.area_um2/pixel**2,model_path=seg.yolo_model_path)
    closest_yolo=min((float(np.linalg.norm(np.subtract(c.center,historical_center))) for c in predicted_yolo),default=None) if historical_center else None
    changed=decision.matched!=case['expectedAccepted'] or bool(case['expectedAccepted'] and distance>case['centerTolerancePx'])
    record={'caseId':case['caseId'],'rank':case['rank'],'sequence':case['sequence'],
        'historicalExpected':{'accepted':case['expectedAccepted'],'status':case['expectedDecisionStatus'],'centerPx':historical_center},
        'committedExpected':{'accepted':decision.matched,'status':decision.status,'centerPx':list(chosen.pixel_xy),'centerTolerancePx':3.0},
        'committedDecision':decision.to_dict(),'workingTreeReproducesCommitted':True,
        'differenceAlreadyPresentInCommittedCode':changed,'historicalToCommittedDistancePx':distance,
        'nearestYoloToHistoricalCenterPx':closest_yolo}
    review['cases'].append(record)
    print(json.dumps({k:v for k,v in record.items() if k!='committedDecision'}),flush=True)
assert len(review['cases'])==8
review['summary']={'reviewed':8,'historicalDifferences':sum(c['differenceAlreadyPresentInCommittedCode'] for c in review['cases']),
                   'committedAccepted':sum(c['committedExpected']['accepted'] for c in review['cases'])}
destination.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(review['summary']),flush=True)
