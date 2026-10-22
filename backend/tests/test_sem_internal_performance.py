import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.cleanliness_core.sem_segmentation import decode_detections,sam3_config,segment_sam3
from backend.cleanliness_core.sem_task_store import SemTaskStore,load_sem_task
from backend.cleanliness_core.sem_progress import read_progress,refresh_inactive_header
from backend.cleanliness_core.sem_spatial import ContourIndex
from backend.cleanliness_core.sem_direct import contours_overlap,evaluate_shapes
from backend.tests.test_sem_direct import task,box

FIXTURE=Path(__file__).parent/'fixtures/sem_performance_20260929'


def reference():
    spec=importlib.util.spec_from_file_location('reference',FIXTURE/'reference_decoder.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.decode_detections


def encoded(mask,score=.8):
    return {'score':score,'text':'particle','mask_png_b64':base64.b64encode(cv2.imencode('.png',mask)[1]).decode()}


def test_real_frame_contours_scores_and_duplicate_indices_are_exactly_preserved():
    provenance=json.loads((FIXTURE/'provenance.json').read_text())
    assert hashlib.sha256((FIXTURE/'field1.png').read_bytes()).hexdigest()==provenance['sourceSha256']
    detections=json.loads((FIXTURE/'response.json').read_text())['detections']
    actual=decode_detections(detections,(1024,1024),sam3_config({'scoreThreshold':.01}))
    assert actual==json.loads((FIXTURE/'expected-shapes.json').read_text())
    case=json.loads((FIXTURE/'gate-case.json').read_text())
    assert evaluate_shapes(actual,**case)==json.loads((FIXTURE/'expected-gates.json').read_text(encoding='utf-8'))


def test_mask_optimization_matches_baseline_for_fragmented_empty_and_competing_masks():
    rng=np.random.default_rng(742)
    masks=[]
    for i in range(45):
        m=np.zeros((96,96),np.uint8)
        x,y=rng.integers(0,75,2);w,h=rng.integers(1,21,2)
        m[y:y+h,x:x+w]=255
        if i%7==0: m[90:93,90:93]=255
        if i%9==0: m[:]=0
        masks.append(encoded(m,float(rng.uniform(.2,1))))
        if i%3==0: masks.append(encoded(m,.6))
    config=sam3_config({})
    assert decode_detections(masks,(96,96),config)==reference()(masks,(96,96),config)


def test_grayscale_png_preserves_service_rgb_pixels(monkeypatch):
    from backend.cleanliness_core import sem_segmentation as sam
    from backend.cleanliness_core.optical_particle_analysis import normalize_uint8
    rng=np.random.default_rng(31);source=rng.integers(0,65535,(512,512),dtype=np.uint16)
    bodies=[]
    monkeypatch.setattr(sam,'request_json',lambda url,payload:bodies.append(payload) or {'detections':[]})
    segment_sam3(source,sam3_config({}))
    decoded=cv2.imdecode(np.frombuffer(base64.b64decode(bodies[0]['image_png_b64']),np.uint8),cv2.IMREAD_COLOR)
    assert np.array_equal(decoded,cv2.cvtColor(normalize_uint8(source),cv2.COLOR_GRAY2BGR))


def test_oversized_mask_rejected_before_decoder_allocation(monkeypatch):
    import struct
    png=b'\x89PNG\r\n\x1a\n'+struct.pack('>I',13)+b'IHDR'+struct.pack('>II',100000,100000)
    monkeypatch.setattr(cv2,'imdecode',lambda *a:pytest.fail('must check PNG size before decoding'))
    with pytest.raises(ValueError,match='尺寸'):
        decode_detections([{'score':.9,'mask_png_b64':base64.b64encode(png).decode()}],(1024,1024),sam3_config({}))


def test_oversized_remote_response_is_not_parsed_or_truncated(monkeypatch):
    from backend.cleanliness_core import sem_segmentation as sam
    from types import SimpleNamespace
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):assert n==65;return b'x'*n
    monkeypatch.setattr(sam,'MAX_RESPONSE_BYTES',64)
    monkeypatch.setattr(sam.urllib.request,'build_opener',lambda *a:SimpleNamespace(open=lambda *a,**k:Response()))
    with pytest.raises(RuntimeError,match='超过'):
        sam.request_json('http://unused/segment_text',{})


def test_spatial_filter_never_excludes_an_intersecting_contour():
    rng=np.random.default_rng(17);items=[];index=ContourIndex(64)
    for i in range(180):
        x,y=rng.integers(-300,300,2); side=int(rng.integers(1,100))
        contour=box(int(x),int(y),side)['points'];items.append(contour);index.add(contour,i)
    for contour in items[::3]+[box(0,0,64)['points'],box(-64,-64,64)['points']]:
        exact={i for i,old in enumerate(items) if contours_overlap(contour,old)}
        assert exact <= set(index.candidates(contour))


def test_progress_delta_unchanged_reset_counts_and_recent_limit(task):
    directory,m=task
    m['particles']=[dict(box(),sequence=i+1,tileId=1,status='PENDING') for i in range(130)]
    store=SemTaskStore(directory,m)
    try:
        initial=read_progress(directory)
        assert initial['reset'] and len(initial['task']['particles'])==100
        assert initial['task']['stats']['particleCount']==130
        assert 'contour' not in initial['task']['particles'][0]
        assert read_progress(directory,initial['cursor'])['unchanged']
        p=m['particles'][0];p['status']='SUCCEEDED'
        store.save(m,m['plan']['tiles'][0],dirty_particles=[p])
        delta=read_progress(directory,initial['cursor'])
        assert not delta['reset'] and len(delta['tiles'])==1
        assert delta['task']['stats']['measured']==1  # Older than last 100 rows.
        assert read_progress(directory,'old-epoch:999')['reset']
        assert read_progress(directory,store.epoch+':999999')['reset']
        assert load_sem_task(directory)['particles']==m['particles']
    finally: store.close()


def test_dirty_save_never_serializes_unchanged_particles(task,monkeypatch):
    directory,m=task
    m['particles']=[{'contour':box()['points'],'sequence':i+1,'tileId':1,'status':'PENDING'} for i in range(200)]
    store=SemTaskStore(directory,m)
    original=store.encode;seen=[]
    def encode(value):
        if isinstance(value,dict) and 'sequence' in value and 'contour' in value:seen.append(value['sequence'])
        return original(value)
    monkeypatch.setattr(store,'encode',encode)
    m['particles'][3]['status']='ACQUIRING'
    store.save(m,m['plan']['tiles'][0],dirty_particles=[m['particles'][3]])
    assert seen==[4]
    assert load_sem_task(directory)['particles'][3]['status']=='ACQUIRING'
    store.close()


def test_ui_and_durable_state_roll_back_together(task):
    import sqlite3
    directory,m=task;store=SemTaskStore(directory,m)
    before=read_progress(directory)
    store.connection.execute("CREATE TRIGGER fail_ui BEFORE INSERT ON ui_header BEGIN SELECT RAISE(ABORT,'UI write failure'); END")
    m['particles']=[dict(box(),sequence=1,tileId=1,status='ACQUIRING')]
    with pytest.raises(sqlite3.DatabaseError):store.save(m,m['plan']['tiles'][0],dirty_particles=m['particles'])
    assert read_progress(directory)==before and not load_sem_task(directory)['particles']
    store.close()


def test_progress_http_does_not_read_full_manifest(task,monkeypatch):
    import threading,urllib.request
    from http.server import ThreadingHTTPServer
    from backend import server
    directory,m=task;store=SemTaskStore(directory,m)
    monkeypatch.setattr(server,'TASK_ROOT',directory.parent)
    monkeypatch.setattr(server,'log_event',lambda *a,**k:None)
    monkeypatch.setattr(server,'task_manifest',lambda *a:pytest.fail('progress must not read full task'))
    http=ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
    thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        url=f'http://127.0.0.1:{http.server_port}/tasks/{directory.name}/sem/progress'
        with opener.open(url,timeout=5) as response: data=json.load(response)
        assert data['task']['taskId']==m['taskId'] and data['reset']
        with opener.open(url+'?cursor='+data['cursor'],timeout=5) as response: same=json.load(response)
        assert same['unchanged']
    finally:
        http.shutdown();http.server_close();thread.join(timeout=5);store.close()


def test_eds_audit_removes_png_before_json_conversion_and_keeps_final_image(tmp_path,monkeypatch):
    from google.protobuf import json_format
    from backend.cleanliness_core import energy_spectrum
    from backend.cleanliness_core.eds.generated.v2 import job_events_pb2 as pb
    from backend.tests.test_energy_spectrum import _FakeEdsClient
    raw=tmp_path/'tiles/00001.png';raw.parent.mkdir();raw.write_bytes(b'frame')
    png=b'large-final-spectrum'*10000
    event=pb.JobEvent(completed=pb.JobCompleted(spectrum_image=pb.SpectrumImage(png=png),quantify_result_map={'Fe':99}))
    original=json_format.MessageToDict;seen=[]
    def convert(message,**kwargs):
        seen.append(message.completed.spectrum_image.png)
        return original(message,**kwargs)
    monkeypatch.setattr(json_format,'MessageToDict',convert)
    monkeypatch.setattr(energy_spectrum,'start_scan_points',lambda *a:None)
    result=energy_spectrum.process_energy_spectrum_particle(_FakeEdsClient([event]),{'label':'1'},str(raw),'unused',0,target_point=(10,10))
    assert seen==[b''] and event.completed.spectrum_image.png==png
    assert (tmp_path/result['spectrumImage']['path']).read_bytes()==png
    log=next((tmp_path/'eds_events').glob('*.jsonl')).read_text()
    assert '[omitted from event log]' in log and len(log)<1000


def test_completed_job_history_is_bounded_without_evicting_active_jobs(monkeypatch):
    from backend import server
    monkeypatch.setattr(server,'log_event',lambda *a,**k:None)
    jobs=server.JobManager()
    active=jobs.create('active','sem_report')
    last=None
    for i in range(1000):
        last=jobs.create(f'task-{i}','sem_report')
        jobs.update(last['id'],status='completed')
    assert len(jobs._jobs)==65 and len(jobs._cancel)==65
    assert jobs.get(active['id'])['status']=='queued'
    assert jobs.get(last['id'])['status']=='completed'


def test_manual_report_header_refresh_does_not_change_recovery_sequence(task):
    directory,m=task;store=SemTaskStore(directory,m)
    store.save(m,dirty_particles=[]);store.checkpoint(m);store.close()
    before=read_progress(directory)
    m.update(reportState='READY',reports={'pdf':'sem_report.pdf'})
    refresh_inactive_header(directory,m)
    view=read_progress(directory,before['cursor'])
    assert not view['reset'] and view['tiles']==[]
    assert view['task']['reports']['pdf']=='sem_report.pdf'
    assert load_sem_task(directory)['incrementalState']==m['incrementalState']


def test_completed_large_job_result_is_immutable_compressed_and_wire_compatible(monkeypatch):
    from backend import server
    logs=[]
    monkeypatch.setattr(server,'log_event',lambda *a,**k:logs.append(k))
    jobs=server.JobManager();job=jobs.create('task','analysis')
    result={'taskId':'task','status':'COMPLETED','particles':[{'sequence':i,'contour':[[1,2]]*500} for i in range(100)]}
    expected=copy.deepcopy(result)
    jobs.update(job['id'],status='completed',result=result)
    assert 'result' not in jobs._jobs[job['id']]
    assert len(jobs._jobs[job['id']]['_packed_result'])<len(json.dumps(result))/10
    result['particles'][0]['sequence']=-1
    assert jobs.get(job['id'])['result']==expected
    assert '_packed_result' not in jobs.get(job['id'])
    assert 'particles' not in logs[-1]['values']['result']


def test_legacy_list_cache_keeps_only_metadata_and_invalidates_on_file_change(tmp_path):
    from backend import server
    from backend.cleanliness_core.atomic_json import write_json_atomic
    path=tmp_path/'task.json'
    m={'mode':'sem_direct','status':'DRAFT','input':{},'particles':[{'contour':[[1,2]]*1000}]}
    write_json_atomic(path,m)
    stat=path.stat();first=server._cached_task_list_summary(str(path),stat.st_mtime_ns,stat.st_size)
    assert 'particles' not in first and first['status']=='DRAFT'
    m['status']='COMPLETED';write_json_atomic(path,m)
    stat=path.stat();second=server._cached_task_list_summary(str(path),stat.st_mtime_ns,stat.st_size)
    assert second['status']=='COMPLETED'


def test_cancel_racing_completed_job_eviction_does_not_crash(monkeypatch):
    import threading
    from backend import server
    monkeypatch.setattr(server,'log_event',lambda *a,**k:None)
    jobs=server.JobManager();ids=[];errors=[];stop=threading.Event()
    def cancel():
        try:
            while not stop.is_set():
                for job_id in ids[-10:]:
                    jobs.cancel(job_id);jobs.cancelled(job_id)
                stop.wait(.0005)
        except BaseException as e:errors.append(e)
    thread=threading.Thread(target=cancel);thread.start()
    try:
        for i in range(1000):
            job=jobs.create(str(i),'sem_report');ids.append(job['id']);jobs.update(job['id'],status='completed')
    finally:stop.set();thread.join(timeout=5)
    assert not errors and not thread.is_alive()


def test_empty_timing_history_and_compact_checkpoint_round_trip(task):
    directory,m=task;m['timingRuns']=[]
    store=SemTaskStore(directory,m)
    store.save(m,dirty_particles=[]);store.checkpoint(m);store.close()
    assert load_sem_task(directory)==m
    assert '\n  ' not in (directory/'task.json').read_text()
