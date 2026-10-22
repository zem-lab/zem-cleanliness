"""Persistent-state/UI stress with concurrent readers; no device commands."""
import argparse
import collections
import ctypes
import gc
import json
import sys
import threading
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.cleanliness_core.sem_task_store import SemTaskStore,load_sem_task
from backend.cleanliness_core.sem_progress import read_progress


def resources():
    import ctypes.wintypes as w
    class Memory(ctypes.Structure):
        _fields_=[('cb',w.DWORD),('PageFaultCount',w.DWORD),('PeakWorkingSetSize',ctypes.c_size_t),
                  ('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),
                  ('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),
                  ('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
    kernel=ctypes.windll.kernel32;psapi=ctypes.windll.psapi
    kernel.GetCurrentProcess.restype=w.HANDLE
    psapi.GetProcessMemoryInfo.argtypes=[w.HANDLE,ctypes.POINTER(Memory),w.DWORD]
    psapi.GetProcessMemoryInfo.restype=w.BOOL
    kernel.GetProcessHandleCount.argtypes=[w.HANDLE,ctypes.POINTER(w.DWORD)]
    kernel.GetProcessHandleCount.restype=w.BOOL
    handle=kernel.GetCurrentProcess();m=Memory();m.cb=ctypes.sizeof(m)
    if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(m),m.cb):raise ctypes.WinError()
    count=w.DWORD()
    if not kernel.GetProcessHandleCount(handle,ctypes.byref(count)):raise ctypes.WinError()
    return {'workingSetMB':m.WorkingSetSize/1e6,'handles':count.value}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--updates',type=int,default=20000)
    args=parser.parse_args()
    out=ROOT/'artifacts/sem-internal-performance'/time.strftime('stress-%Y%m%d-%H%M%S');out.mkdir(parents=True)
    tiles=[{'id':i+1,'x':(i%32)*512,'y':(i//32)*512,'status':'PENDING'} for i in range(1024)]
    particles=[{'sequence':i+1,'tileId':i//20+1,'status':'PENDING','lengthUm':10,
                'contour':[[10,10],[20,10],[20,20],[10,20]],'stageContourUm':[[i,0],[i+1,0],[i+1,1],[i,1]],
                'safePoint':[15,15]} for i in range(20000)]
    m={'taskId':'LOCAL-STRESS-NO-HARDWARE','mode':'sem_direct','status':'RUNNING','input':{},'request':{},
       'navigation':{},'plan':{'tiles':tiles,'tileCount':len(tiles)},'particles':particles,
       'timingRuns':[{'events':[],'stages':{}}]}
    store=SemTaskStore(out,m)
    stop=threading.Event();errors=[];reads=[0];max_payload=[0]
    def reader():
        cursor=''
        try:
            while not stop.is_set():
                view=read_progress(out,cursor)
                assert view
                if not view.get('unchanged'):
                    assert len(view['task']['particles'])<=100
                    assert view['task']['stats']['particleCount']==20000
                    max_payload[0]=max(max_payload[0],len(json.dumps(view).encode()))
                cursor=view['cursor'];reads[0]+=1
                stop.wait(.005)
        except BaseException as e:errors.append(repr(e));stop.set()
    threads=[threading.Thread(target=reader) for _ in range(2)]
    for t in threads:t.start()
    samples=[];start=time.perf_counter()
    try:
        for i in range(args.updates):
            if errors: raise AssertionError(errors)
            p=particles[(i//2)%len(particles)];tile=tiles[p['tileId']-1]
            p['status']='ACQUIRING' if i%2==0 else 'SUCCEEDED'
            if i%2:
                p['spectrum']={'elements':[{'symbol':'Fe','value':98},{'symbol':'C','value':2}],
                               'result_text':'synthetic stress data','jobId':f'stress-{i}'}
            m['timingRuns'][0]['events'].append({'stage':'eds','sequence':p['sequence'],'seconds':0})
            store.save(m,tile,dirty_particles=[p])
            if (i+1)%1000==0:
                store.checkpoint(m)
                gc.collect();sample={'updates':i+1,'elapsedSeconds':time.perf_counter()-start,**resources()}
                samples.append(sample);print(json.dumps(sample),flush=True)
        recovered=load_sem_task(out)
        assert recovered==m
        counts=collections.Counter(p['status'] for p in recovered['particles'])
        ui=read_progress(out)
        assert ui['task']['stats']['measured']==counts['SUCCEEDED']
        store.checkpoint(m)
    finally:
        stop.set()
        for t in threads:t.join(timeout=10)
        store.close()
    assert not errors and all(not t.is_alive() for t in threads)
    # Repeat idle reads after GC to check that DB handles/connections do not leak.
    gc.collect();idle_before=resources()
    cursor=read_progress(out)['cursor']
    for _ in range(3000):assert read_progress(out,cursor)['unchanged']
    gc.collect();idle_after=resources()
    assert idle_after['handles']<=idle_before['handles']+4
    assert idle_after['workingSetMB']<=idle_before['workingSetMB']+32
    summary={'updates':args.updates,'particles':20000,'tiles':1024,'concurrentReaders':2,'readCount':reads[0],
             'maxResponseBytes':max_payload[0],'seconds':time.perf_counter()-start,'samples':samples,
             'idleBefore':idle_before,'idleAfter':idle_after,'replayExact':True,'errors':errors}
    (out/'result.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
