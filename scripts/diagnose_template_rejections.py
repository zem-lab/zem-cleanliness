"""Instrument unchanged template matching on saved frames; no hardware or network.

Captures runner-up location/warp that production logs currently omit. Geometry
overlap is diagnostic evidence, not an identity ground-truth label.
"""
import argparse
import base64
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from loguru import logger
from backend.cleanliness_core import particle_refinement as refinement
from backend.cleanliness_core.cleanliness_detection import yolo_seg_candidates, build_sem_anchor_candidates
from backend.cleanliness_core.optical_particle_analysis import load_optical_export


def warped(mask, optical_px, pixel, peak, shape):
    h,w=mask.shape
    scale=optical_px/pixel*peak['scaleFactor']
    n=int(math.ceil(math.hypot(h,w)*scale))+20
    warp=cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),peak['angleDeg'],scale)
    warp[:,2]+=np.array([(n-w)/2,(n-h)/2])+np.asarray(peak['centerPixel'])-(n-1)/2
    return cv2.warpAffine(np.fliplr(mask).astype(np.float32) if peak['mirrored'] else mask.astype(np.float32),warp,(shape[1],shape[0]))>.5


def overlap(a,b):
    intersection=int(np.logical_and(a,b).sum())
    return dict(iou=intersection/max(1,int(np.logical_or(a,b).sum())),
                intersectionOverSmaller=intersection/max(1,min(int(a.sum()),int(b.sum()))))


def candidate_mask(c,shape):
    a=np.zeros(shape,np.uint8)
    cv2.fillPoly(a,[np.asarray(c.contour,np.int32)],1)
    return a.astype(bool)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();logger.remove()
    if args.output.exists():parser.error('Use a fresh diagnostic output')
    args.output.mkdir(parents=True)
    def deny_network(event,args):
        if event in {'socket.connect','socket.sendto','socket.bind'}:raise RuntimeError('Offline diagnostic')
    sys.addaudithook(deny_network)
    code=inspect.getsource(refinement.match_particle)
    assert 'return {**best, "status": status' in code
    instrumented=code.replace('return {**best, "status": status','return {"diagnosticPeaks": peaks, **best, "status": status')
    namespace=dict(refinement.__dict__)
    exec(compile(instrumented,'<diagnostic-copy-of-match_particle>','exec'),namespace)
    matcher=namespace['match_particle']
    (args.output/'matcher_snapshot.py').write_text(code,encoding='utf-8')
    m=json.loads((args.task/'task.json').read_text(encoding='utf-8-sig'))
    export=load_optical_export(Path(m['input']['path']))
    assert export.sha256==m['input']['sha256']
    pixel=m['baselineImaging']['pixelSizeUm']
    rows=[]
    for r in m['particles']:
        if r['rank'] not in {140,135,122,118,111,90,88,84,73,42,38,8}:continue
        p=export.by_sequence[r['sequence']]
        mask,optical_px,_=refinement.optical_mask(p)
        for attempt in r['captureAttempts']:
            path=args.task/attempt['rawPath']
            raw=path.read_bytes();image=cv2.imdecode(np.frombuffer(raw,np.uint8),-1)
            result=matcher(image,p,pixel,refinement.RefinementSettings())
            peaks=result.pop('diagnosticPeaks')
            saved=attempt['maskVerification']
            assert abs(result['score']-saved['score'])<1e-5
            assert abs(result['margin']-saved['margin'])<1e-5
            radius=max(8,p.width_um/pixel*.6)
            runner=max((x for x in peaks if np.linalg.norm(np.subtract(x['centerPixel'],result['centerPixel']))>radius),key=lambda x:x['score'])
            a=warped(mask,optical_px,pixel,result,image.shape)
            b=warped(mask,optical_px,pixel,runner,image.shape)
            yolo=yolo_seg_candidates(image,expected_length_px=p.length_um/pixel,expected_width_px=p.width_um/pixel,
                expected_area_px=p.area_um2/pixel**2,
                model_path=Path('resources/models/cleanliness/particle_yolo11n_seg_reann_20260811_best.onnx').resolve())
            classic=build_sem_anchor_candidates(p,image,pixel_size_um=pixel,stage_center_um=(0,0),maximum_candidates=64)
            overlaps=[]
            for label,pool in [('yolo',yolo),('classic',[c.candidate for c in classic])]:
                for index,c in enumerate(pool):
                    cm=candidate_mask(c,image.shape)
                    overlaps.append(dict(method=label,index=index,center=list(c.center),box=list(c.box),
                        primary=overlap(a,cm),runner=overlap(b,cm),lengthRatio=c.length_ratio,widthRatio=c.width_ratio))
            row=dict(rank=r['rank'],sequence=r['sequence'],attempt=attempt['attempt'],image=str(path),imageSha256=hashlib.sha256(raw).hexdigest(),
                     maskShape=list(mask.shape),lengthUm=p.length_um,widthUm=p.width_um,radiusPx=radius,
                     primary=result,runner=runner,centerDistancePx=float(np.linalg.norm(np.subtract(result['centerPixel'],runner['centerPixel']))),
                     templateOverlap=overlap(a,b),yoloCount=len(yolo),classicCount=len(classic),candidateOverlaps=overlaps)
            rows.append(row)
            for label,arr in [('primary',a),('runner',b)]:
                cv2.imwrite(str(args.output/f"rank{r['rank']}_attempt{attempt['attempt']}_{label}.png"),arr.astype(np.uint8)*255)
            print(json.dumps({k:row[k] for k in ['rank','attempt','radiusPx','centerDistancePx','templateOverlap','yoloCount']}),flush=True)
    (args.output/'diagnostics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
