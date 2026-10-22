"""Derived imaging perturbations of the saved A/B cases; no new identity labels."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.cleanliness_core.cleanliness_detection import yolo_seg_candidates
from backend.cleanliness_core.optical_particle_analysis import load_optical_export
from backend.cleanliness_core.particle_refinement import optical_mask
from backend.cleanliness_core.shape_comparison import compare_shapes

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():parser.error('Use a new output directory')
args.output.mkdir(parents=True)
folder=ROOT/'backend/tests/fixtures/human_confirmed_20260918'
spec=json.loads((folder/'cases.json').read_text(encoding='utf-8'))
assert hashlib.sha256((folder/'optical.json').read_bytes()).hexdigest()==spec['opticalSubsetSha256']
export=load_optical_export(folder/'optical.json')
model=ROOT/spec['model']['path']
assert hashlib.sha256(model.read_bytes()).hexdigest()==spec['model']['sha256']
pixel=spec['pixelSizeUm']
code_sha=hashlib.sha256((ROOT/'backend/cleanliness_core/shape_comparison.py').read_bytes()).hexdigest()
results=[]
for case in spec['cases']:
    raw=(folder/case['image']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==case['imageSha256']
    source=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED)
    p=export.by_sequence[case['sequence']]
    mask,optical_pixel,_=optical_mask(p)
    noise=np.random.default_rng(20260918).normal(0,2,source.shape)
    variants={
        'contrast_0.8_offset_10':(np.rint(source.astype(float)*.8+10).clip(0,255).astype(np.uint8),(0,0)),
        'gaussian_noise_sigma_2_seed_20260918':(np.rint(source.astype(float)+noise).clip(0,255).astype(np.uint8),(0,0)),
        'gaussian_blur_sigma_0.5':(cv2.GaussianBlur(source,(0,0),.5),(0,0)),
        'translate_x1_yminus1':(cv2.warpAffine(source,np.array([[1,0,1],[0,1,-1]],float),source.shape[::-1],borderMode=cv2.BORDER_REFLECT_101),(1,-1)),
    }
    for name,(image,delta) in variants.items():
        candidates=yolo_seg_candidates(image,expected_length_px=p.length_um/pixel,
            expected_width_px=p.width_um/pixel,expected_area_px=p.area_um2/pixel**2,model_path=model)
        result=compare_shapes(mask,optical_pixel,pixel,[c.contour for c in candidates],image.shape)
        correct=np.asarray(case['correctTarget']['referenceCenterPx'])+delta
        wrong=np.asarray(case['wrongTarget']['referenceCenterPx'])+delta
        first=result['candidates'][0] if result['candidates'] else None
        agrees=bool(first and np.linalg.norm(np.subtract(first['centerPx'],correct))<=spec['associationTolerancePx'])
        filename=f"rank{case['reportRank']}_attempt{case['attempt']}_{name}.png"
        encoded=cv2.imencode('.png',image)[1].tobytes()
        (args.output/filename).write_bytes(encoded)
        row={'caseId':case['caseId'],'sourceImageSha256':case['imageSha256'],'condition':name,
             'shapeCodeSha256':code_sha,'modelSha256':spec['model']['sha256'],
             'image':filename,'imageSha256':hashlib.sha256(encoded).hexdigest(),
             'referenceKind':'derived_robustness_check_not_new_human_annotation',
             'correctTargetPx':correct.tolist(),'wrongTargetPx':wrong.tolist(),
             'ranksAFirst':agrees,'allowsEds':result['allowsEds'],
             'scoreMargin':result['scoreMargin'],'shapeComparison':result}
        results.append(row)
        (args.output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'done':len(results),'caseId':case['caseId'],'condition':name,'ranksAFirst':agrees,'margin':result['scoreMargin']}),flush=True)
print(json.dumps({'frames':len(results),'ranksAFirst':sum(r['ranksAFirst'] for r in results)}),flush=True)
