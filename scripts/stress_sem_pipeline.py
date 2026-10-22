"""Accelerated complete SEM workflow, strictly simulated devices."""
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.cleanliness_core.sem_direct import build_plan,run_task,fingerprint
from backend.cleanliness_core.atomic_json import write_json_atomic
from backend.cleanliness_core.sem_task_store import load_sem_task
from backend.cleanliness_core.sem_progress import read_progress
from scripts.stress_sem_internal import resources


def main():
    out=ROOT/'artifacts/sem-internal-performance'/time.strftime('pipeline-%Y%m%d-%H%M%S');out.mkdir(parents=True)
    model=out/'fake-model.onnx';model.write_bytes(b'SIMULATED ONLY')
    (out/'navigation.png').write_bytes(b'SIMULATED ONLY')
    plan=build_plan({'roi':{'shape':'square','x0':0,'y0':0,'x1':10240,'y1':10240},'fieldUm':512,'overlap':0,'imageSize':512,'minLengthUm':5},[0,0,10240,10240])
    m={'taskId':'SIMULATED-PIPELINE','mode':'sem_direct','status':'DRAFT','input':{},'particles':[],
       'request':{'microscopeHost':'never-connect','microscopePort':0,'captureLevel':1,'edsEnabled':True,'restoreStartPosition':False,'gunOffAfterTask':False},
       'navigation':{'boundsUm':[0,0,10240,10240],'fieldUm':512,'magnification':100},
       'plan':plan,'model':{'path':str(model),'sha256':fingerprint(model)}}
    write_json_atomic(out/'task.json',m)
    class Scope:
        position=(256,256)
        def connect(self):pass
        def get_position(self):return self.position
        def get_hv_status(self):return 'Open'
        def move_xy(self,x,y):self.position=(x,y);return {'after':{'x':x,'y':y},'positionErrorUm':0}
        def capture(self,path):
            path.parent.mkdir(exist_ok=True);im=np.zeros((512,512),np.uint8);cv2.imwrite(str(path),im)
            return im,{'pixelSizeUm':1}
        def restore_video(self):pass
    class Analyzer:
        calls=0
        def analyze(self,p,path):
            self.calls+=1
            if self.calls%200==0:
                view=read_progress(out)
                assert next(x for x in view['task']['particles'] if x['sequence']==p['sequence'])['status']=='ACQUIRING'
                print(json.dumps({'edsCalls':self.calls,**resources()}),flush=True)
            return {'elements':[{'symbol':'Fe','value':98}], 'mock':True,'jobId':f'SIMULATED-{self.calls}'}
        def close(self):pass
    shapes=[{'points':[[50+i*35,100],[65+i*35,100],[65+i*35,115],[50+i*35,115]],'confidence':1} for i in range(10)]
    analyzer=Analyzer();started=time.perf_counter();report_calls=[]
    # PDF contents/layout have separate real renderer tests; stress the scan engine here.
    with patch('backend.cleanliness_core.sem_direct.write_reports',lambda *a:report_calls.append(1)):
        result=run_task(out,microscope=Scope(),analyzer=analyzer,segmenter=lambda *a:shapes)
    assert analyzer.calls==4000 and len(report_calls)==1
    assert result['status']=='COMPLETED'
    recovered=load_sem_task(out)
    assert all(p['status']=='SUCCEEDED' for p in recovered['particles'])
    assert len({p['sequence'] for p in recovered['particles']})==4000
    assert all(t['status']=='COMPLETED' for t in recovered['plan']['tiles'])
    summary={'tiles':400,'successfulParticles':4000,'seconds':time.perf_counter()-started,
             'singleFinalReport':True,'durableReplayPassed':True,**resources()}
    (out/'result.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
