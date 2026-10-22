import base64
import json
import cv2
import numpy as np
import pytest

from backend.cleanliness_core import sem_segmentation as sam
from backend.cleanliness_core.sem_direct import run_task
from backend.cleanliness_core.atomic_json import write_json_atomic
from backend.tests.test_sem_direct import task, Microscope, Analyzer, options


def detection(x=100, y=100, size=30, fragmented=False, shape=(512,512)):
    m=np.zeros(shape,np.uint8)
    m[y:y+size,x:x+size]=255
    if fragmented: m[300:310,300:310]=255
    return {'score':.7,'text':'white particle','mask_png_b64':base64.b64encode(cv2.imencode('.png',m)[1]).decode()}


@pytest.mark.parametrize('detections,count', [
    ([detection()],1), ([detection(),detection()],1),
    ([detection(),detection(160)],2), ([detection(),detection(115)],0),
    ([],0), ([detection(size=1)],0), ([detection(fragmented=True)],0), ([detection(0)],0),
])
def test_remote_gates_and_eds_boundary(task, monkeypatch, detections, count):
    directory,manifest=task
    manifest['model']=sam.sam3_config({})
    write_json_atomic(directory/'task.json',manifest)
    calls=[]
    def request(url,payload=None):
        calls.append((url,payload))
        return {'ok':True,'model':'sam3'} if payload is None else {'detections':detections}
    monkeypatch.setattr(sam,'request_json',request)
    analyzer=Analyzer()
    result=run_task(directory,microscope=Microscope(),analyzer=analyzer)
    assert result['status']=='COMPLETED'
    assert len(analyzer.calls)==count
    assert calls[-1][1]['texts']==sam.DEFAULT_TEXTS
    assert calls[-1][1]['score_threshold']==.1
    assert (directory/'tiles/00001.sam3.json').is_file()


@pytest.mark.parametrize('response', [{}, {'detections':[detection(shape=(32,32))]}, {'detections':[detection()]*1000}])
def test_invalid_remote_response_never_eds(task,monkeypatch,response):
    directory,manifest=task
    manifest['model']=sam.sam3_config({})
    write_json_atomic(directory/'task.json',manifest)
    monkeypatch.setattr(sam,'request_json',lambda url,payload=None: {'ok':True,'model':'sam3'} if payload is None else response)
    analyzer=Analyzer()
    with pytest.raises((ValueError,RuntimeError)):
        run_task(directory,microscope=Microscope(),analyzer=analyzer)
    assert not analyzer.calls


def test_unavailable_service_before_capture(task,monkeypatch):
    directory,manifest=task
    manifest['model']=sam.sam3_config({})
    write_json_atomic(directory/'task.json',manifest)
    def fail(*args): raise RuntimeError('service offline')
    monkeypatch.setattr(sam,'request_json',fail)
    scope,analyzer=Microscope(),Analyzer()
    with pytest.raises(RuntimeError,match='service offline'): run_task(directory,microscope=scope,analyzer=analyzer)
    assert scope.captures==0 and not analyzer.calls


def test_input_hash_changes_with_frame_and_config(monkeypatch):
    payloads=[]
    monkeypatch.setattr(sam,'request_json',lambda url,payload: payloads.append(payload) or {'detections':[]})
    a=np.zeros((64,64),np.uint8); b=a.copy(); b[10:20,10:20]=255
    for image,conf in [(a,.1),(b,.1),(b,.2)]: sam.segment_sam3(image,sam.sam3_config({'scoreThreshold':conf}))
    assert len({p['image_id'] for p in payloads})==3


@pytest.mark.parametrize('changes',[{'url':'file:///tmp/x'},{'texts':[]},{'scoreThreshold':float('nan')},{'scoreThreshold':0}])
def test_invalid_configuration(changes):
    with pytest.raises(ValueError): sam.sam3_config(changes)


def test_default_server_and_existing_task_address_preserved():
    assert sam.sam3_config({})['url'] == 'http://192.168.0.42:8765'
    assert sam.sam3_config({'url':'http://192.168.0.207:8765'})['url'] == 'http://192.168.0.207:8765'


def test_save_switch_and_started_task_locked(task,monkeypatch):
    from backend import server
    directory,manifest=task
    monkeypatch.setattr(server,'task_manifest',lambda task_id: (directory/'task.json',json.loads((directory/'task.json').read_text())))
    monkeypatch.setattr(server.JOBS,'active_for_task',lambda *args: None)
    monkeypatch.setitem(server.MODEL_FILES,'particle',directory/'model.onnx')
    body={**options(),'segmentation':{'backend':'sam3'}}
    result=server.save_sem_plan('test',body)
    assert result['model']['backend']=='sam3' and 'path' not in result['model']
    assert server.save_sem_plan('test',options())['model']['backend']=='sam3'
    result=server.save_sem_plan('test',{**body,'segmentation':{'backend':'yolo'}})
    assert result['model']['sha256']==manifest['model']['sha256']
    result['status']='STOPPED'; write_json_atomic(directory/'task.json',result)
    with pytest.raises(ValueError): server.save_sem_plan('test',body)
