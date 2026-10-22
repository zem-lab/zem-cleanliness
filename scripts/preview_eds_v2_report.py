"""Render a clearly marked offline report using the user-provided EDS log.

SEM/navigation images are synthetic. No microscope or EDS connection is made.
"""
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from google.protobuf.json_format import ParseDict

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.cleanliness_core.energy_spectrum import process_energy_spectrum_particle
from backend.cleanliness_core.eds.generated.v2 import job_events_pb2 as pb
from backend.cleanliness_core.sem_direct import build_plan, evaluate_shapes
from backend.cleanliness_core.sem_direct_report import build_sem_reports
from backend.cleanliness_core.cleanliness_report import write_cleanliness_html,write_cleanliness_pdf

def main():
    out=ROOT/'output/pdf/eds-v2-report-preview'
    out.mkdir(parents=True,exist_ok=True)
    (out/'tiles').mkdir(exist_ok=True)
    im=np.full((512,512),40,np.uint8)
    cv2.circle(im,(250,250),35,210,-1)
    cv2.putText(im,'OFFLINE SAMPLE - SYNTHETIC SEM',(12,40),cv2.FONT_HERSHEY_SIMPLEX,.6,255,1)
    cv2.imwrite(str(out/'tiles/00001.png'),im)
    cv2.imwrite(str(out/'navigation.png'),im)
    data=json.loads((ROOT/'backend/tests/fixtures/eds_v2_20260929/completed.json').read_text(encoding='utf-8'))
    final=ParseDict(data,pb.JobCompleted())
    class Client:
        target='offline-log-sample'
        def open_session(self): return 'offline-session'
        def start_spectrum_job(self,session): return 'offline-job'
        def watch_job(self,*args): yield pb.JobEvent(seq=14,completed=final)
        def close_session(self,*args): pass
    with patch('backend.cleanliness_core.energy_spectrum.start_scan_points'):
        spectrum=process_energy_spectrum_particle(Client(),{'label':'1'},str(out/'tiles/00001.png'),'unused',0,target_point=(250,250))
    # The fake client's elapsed call time is not a measured hardware time.
    spectrum.pop('clientCallSeconds',None)
    spectrum['mock']=True
    options={'roi':{'shape':'square','x0':0,'y0':0,'x1':512,'y1':512},'fieldUm':512,'imageSize':512,'overlap':.1,'minLengthUm':5}
    plan=build_plan(options,[0,0,512,512])
    tile=plan['tiles'][0]
    tile.update(status='COMPLETED',raw='tiles/00001.png',capture={'pixelSizeUm':1},actualStageUm=[256,256])
    points=cv2.ellipse2Poly((250,250),(35,35),0,0,360,10).tolist()
    particle=evaluate_shapes([{'points':points,'confidence':1}],(512,512),1,[256,256],plan,1)[0]
    particle.update(sequence=1,tileId=1,status='SUCCEEDED',spectrum=spectrum)
    manifest={'taskId':'OFFLINE-EDS-V2-REPORT-SAMPLE','mode':'sem_direct','status':'COMPLETED',
        'input':{'workpieceName':'EDS 新协议离线排版样例（图像为模拟，非真机检测）'},
        'plan':plan,'particles':[particle],'request':{'showReportCoordinates':False},
        'navigation':{'path':'navigation.png','boundsUm':[0,0,512,512],'imageShape':[512,512]},
        'model':{'backend':'sam3','scoreThreshold':.1},'cleanup':{}}
    build_sem_reports(out,manifest)
    (out/'sample-task.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    optical={'taskId':'OFFLINE-OPTICAL-EDS-V2-SAMPLE','status':'COMPLETED','input':{'particleCount':1},
             'selection':{'count':1,'filter':{}},'calibration':{'result':{'equation':{'stageX':'x','stageY':'y'},'fit':{},'validation':{}},'anchors':[],'validation':[]},
             'particles':[{'sequence':1,'particleId':'OFFLINE SAMPLE','status':'SUCCEEDED','label':'离线排版样例',
                           'lengthUm':70,'widthUm':70,'images':{},'energySpectrum':copy.deepcopy(spectrum)}]}
    write_cleanliness_html(out,optical)
    write_cleanliness_pdf(out,optical)
    print(out)

if __name__=='__main__': main()
