"""Create explicitly derived controls; never relabel originals or auto-baseline."""
import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.cleanliness_core.optical_particle_analysis import load_optical_export

SOURCE=ROOT/'backend/tests/fixtures/human_confirmed_20260918'
OUT=ROOT/'backend/tests/fixtures/real_matching_controls_20260918'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    if OUT.exists():raise RuntimeError('Do not overwrite derived controls or reviewed expectations')
    spec=json.loads((SOURCE/'cases.json').read_text(encoding='utf-8'))
    optical=load_optical_export(SOURCE/'optical.json')
    assert optical.sha256==spec['opticalSubsetSha256']
    (OUT/'images').mkdir(parents=True)
    corpus={'schemaVersion':1,'sourceCorpus':'human_confirmed_20260918',
        'sourceCorpusSha256':sha((SOURCE/'cases.json').read_bytes()),
        'sourceOpticalSha256':spec['opticalSubsetSha256'],'model':spec['model'],
        'pixelSizeUm':spec['pixelSizeUm'],
        'labelMeaning':'Deterministically derived controls based on human A-correct/B-wrong identity feedback. These are not raw acquisitions or new human annotations.',
        'expectedResultsPolicy':'Expected positives preserve target A; background/fragment/duplicate/target-erased controls must reject. Expectations follow the transformations and human identity labels, never the current algorithm output.',
        'cases':[]}
    for case in spec['cases']:
        if case['reviewStatus']!='human_confirmed':continue
        p=optical.by_sequence[case['sequence']]
        raw=(SOURCE/case['image']).read_bytes();assert sha(raw)==case['imageSha256']
        image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED)
        assert image.dtype==np.uint8 and image.shape==(512,512)
        r=math.ceil(p.length_um/spec['pixelSizeUm']*.7)+6
        x,y=np.rint(case['correctTarget']['referenceCenterPx']).astype(int)
        bx,by=np.rint(case['wrongTarget']['referenceCenterPx']).astype(int)
        region=(int(x-r),int(y-r),int(x+r),int(y+r))
        left,top,right,bottom=region
        donor=image[top:bottom,left-120:right-120].copy()
        assert donor.shape==(2*r,2*r)
        erased=image.copy();erased[top:bottom,left:right]=donor
        assert np.array_equal(erased[by-10:by+10,bx-10:bx+10],image[by-10:by+10,bx-10:bx+10])
        duplicate=image.copy();duplicate[by-r:by+r,bx-r:bx+r]=image[top:bottom,left:right]
        background=np.tile(donor,(math.ceil(image.shape[0]/len(donor)),math.ceil(image.shape[1]/len(donor))))[:image.shape[0],:image.shape[1]]
        fragment=background.copy()
        half=max(1,round(p.width_um/spec['pixelSizeUm']*.1))
        fragment[top:bottom,x-half:x+half]=image[top:bottom,x-half:x+half]
        noise=np.random.default_rng(20260918).normal(0,2,image.shape)
        variants={
            'contrast':(np.rint(image.astype(float)*.8+10).clip(0,255).astype(np.uint8),True,[0,0],{'factor':.8,'offset':10}),
            'noise':(np.rint(image.astype(float)+noise).clip(0,255).astype(np.uint8),True,[0,0],{'sigma':2,'seed':20260918}),
            'blur':(cv2.GaussianBlur(image,(0,0),.5),True,[0,0],{'sigma':.5}),
            'translation':(cv2.warpAffine(image,np.array([[1,0,1],[0,1,-1]],float),image.shape[::-1],borderMode=cv2.BORDER_REFLECT_101),True,[1,-1],{'dx':1,'dy':-1,'border':'REFLECT_101'}),
            'target_erased_competitor_remains':(erased,False,[0,0],{'replaceXyxy':region,'donorOffsetPx':[-120,0]}),
            'identical_competitor':(duplicate,False,[0,0],{'copyFromXyxy':region,'copyToCenterPx':[int(bx),int(by)]}),
            'background_only':(background,False,[0,0],{'tiledDonorXyxy':[left-120,top,right-120,bottom]}),
            'incomplete_target':(fragment,False,[0,0],{'tiledDonorXyxy':[left-120,top,right-120,bottom],'retainedTargetStripeXyxy':[int(x-half),top,int(x+half),bottom]}),
        }
        for condition,(variant,expected,delta,parameters) in variants.items():
            path=f"images/rank{case['reportRank']}_{condition}.png"
            encoded=cv2.imencode('.png',variant)[1].tobytes();(OUT/path).write_bytes(encoded)
            center=(np.asarray(case['correctTarget']['referenceCenterPx'])+delta).tolist()
            corpus['cases'].append({'caseId':f"rank{case['reportRank']}-{condition}",'sourceCaseId':case['caseId'],
                'rank':case['reportRank'],'sequence':case['sequence'],'particleId':case['particleId'],
                'sourceImage':case['image'],'sourceImageSha256':case['imageSha256'],
                'image':path,'imageSha256':sha(encoded),'condition':condition,'derivation':parameters,
                'labelKind':'derived_control_not_raw_acquisition','expectedAccepted':expected,
                'expectedTargetCenterPx':center if expected else None,
                'knownWrongCenterPx':case['wrongTarget']['referenceCenterPx'] if condition=='target_erased_competitor_remains' else None})
    (OUT/'cases.json').write_text(json.dumps(corpus,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'controls':len(corpus['cases']),'expectedAccepted':sum(c['expectedAccepted'] for c in corpus['cases'])}))


if __name__=='__main__':main()
