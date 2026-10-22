"""Run the offline shape experiment on all versioned real-image corpora."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.cleanliness_core.cleanliness_detection import yolo_seg_candidates
from backend.cleanliness_core.optical_particle_analysis import load_optical_export
from backend.cleanliness_core.particle_refinement import optical_mask
from backend.cleanliness_core.shape_comparison import compare_shapes


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory; evidence must not be overwritten')
    args.output.mkdir(parents=True)
    model = ROOT/'resources/models/cleanliness/particle_yolo11n_seg_reann_20260811_best.onnx'
    model_sha = sha(model)
    started = time.monotonic()
    report = {'method':'offline_physical_binary_iou_v1', 'allowsEds':False,
              'modelSha256':model_sha, 'productionAlgorithmChanged':False,
              'codeSha256':sha(ROOT/'backend/cleanliness_core/shape_comparison.py'),
              'labelPolicy':'Historical acceptance is a compatibility baseline; only the specified latest frames have direct human identity labels.',
              'cases':[]}
    corpora=[('human_confirmed_20260918','cases.json'), ('template_identity_cases','cases.json'),
             ('mask_matching_sample2','baseline.json')]
    for name, filename in corpora:
        folder=ROOT/'backend/tests/fixtures'/name
        spec=json.loads((folder/filename).read_text(encoding='utf-8'))
        expected_optical=spec.get('opticalSubsetSha256',spec.get('opticalSha256'))
        assert sha(folder/'optical.json')==expected_optical
        expected_model=spec.get('modelSha256')
        if isinstance(spec.get('model'),dict):
            expected_model=spec['model'].get('sha256')
        if expected_model:
            assert expected_model==model_sha
        export=load_optical_export(folder/'optical.json')
        pixel=spec['pixelSizeUm']
        for case in spec['cases']:
            image_path=folder/case['image']
            assert sha(image_path)==case['imageSha256']
            particle=export.by_sequence[case['sequence']]
            if 'particleId' in case:
                assert particle.particle_id==case['particleId']
            image=cv2.imdecode(np.fromfile(str(image_path),np.uint8),cv2.IMREAD_UNCHANGED)
            # Call YOLO directly: failures cannot silently fall back to another detector.
            candidates=yolo_seg_candidates(image, expected_length_px=particle.length_um/pixel,
                expected_width_px=particle.width_um/pixel, expected_area_px=particle.area_um2/pixel**2,
                model_path=model)
            mask,optical_pixel,_=optical_mask(particle)
            result=compare_shapes(mask,optical_pixel,pixel,[c.contour for c in candidates],image.shape)
            first=result['candidates'][0] if result['candidates'] else None
            is_new=name=='human_confirmed_20260918'
            original_accepted=case.get('accepted',case.get('expectedAccepted',False))
            reference=case['correctTarget']['referenceCenterPx'] if is_new else case.get('newCenterPx',case.get('expectedCenterPx'))
            distance=float(np.linalg.norm(np.subtract(first['centerPx'],reference))) if first and reference else None
            same_object=distance is not None and distance<=max(8.,particle.width_um/pixel*.25)
            row={'corpus':name,'sourceTaskId':spec['sourceTaskId'],
                 'rank':case.get('rank',case.get('reportRank')),'sequence':case['sequence'],
                 'attempt':case.get('attempt'), 'image':str(image_path.relative_to(ROOT)),
                 'imageSha256':case['imageSha256'],'particleId':particle.particle_id,
                 'referenceKind':case.get('reviewStatus','compatibility_baseline'),
                 'referenceCenterPx':reference,'historicalAccepted':original_accepted,
                 'sameObjectAsReference':same_object,'referenceDistancePx':distance,
                 'yoloCount':len(candidates),'shapeComparison':result}
            if is_new:
                wrong=case['wrongTarget']['referenceCenterPx']
                row['wrongTargetDistancePx']=float(np.linalg.norm(np.subtract(first['centerPx'],wrong))) if first else None
            if case.get('knownWrongCenterPx') and first:
                row['knownWrongCenterDistancePx']=float(np.linalg.norm(np.subtract(first['centerPx'],case['knownWrongCenterPx'])))
            report['cases'].append(row)
            (args.output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({'done':len(report['cases']),'corpus':name,'rank':row['rank'],'attempt':row['attempt'],
                              'sameObject':same_object,'bestIoU':first['shapeIoU'] if first else None,
                              'margin':result['scoreMargin'],'unavailable':len(result['unavailableCandidates'])}),flush=True)
    report['elapsedSeconds']=time.monotonic()-started
    report['summary']={'frames':len(report['cases']),
        'newCaseFramesRankA':sum(c['sameObjectAsReference'] for c in report['cases'] if c['corpus']=='human_confirmed_20260918'),
        'previouslyAcceptedFrames':sum(c['historicalAccepted'] for c in report['cases']),
        'previouslyAcceptedSameObject':sum(c['historicalAccepted'] and c['sameObjectAsReference'] for c in report['cases']),
        'previouslyRejectedFrames':sum(not c['historicalAccepted'] for c in report['cases'] if c['corpus']!='human_confirmed_20260918')}
    (args.output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':
    main()
