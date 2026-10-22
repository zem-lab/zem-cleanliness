"""Compare serialization/durable writes on a COPY of the real SEM manifest."""
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.cleanliness_core.atomic_json import write_json_atomic
from backend.cleanliness_core.sem_task_store import SemTaskStore,load_sem_task

source=Path.home()/'Documents/ZEM Cleanliness/tasks/b7efbb36-c863-4c0b-9d20-0a003c570140/task.json'
raw=source.read_bytes()
original=json.loads(raw)
out=ROOT/'artifacts/sem-persistence-benchmark'/time.strftime('%Y%m%d-%H%M%S')
out.mkdir(parents=True)
results=[]
for mode in ['full','incremental']:
    directory=out/mode; directory.mkdir()
    m=copy.deepcopy(original)
    tile=m['plan']['tiles'][281]
    particle=next(p for p in m['particles'] if p['tileId']==tile['id'])
    started=time.perf_counter()
    store=SemTaskStore(directory,m) if mode=='incremental' else None
    for i in range(40):
        particle['benchmarkTick']=i
        m['updatedAt']=str(i)
        if store: store.save(m,tile)
        else: write_json_atomic(directory/'task.json',m)
    if store:
        replay=load_sem_task(directory)
        assert replay==m
        store.checkpoint(m)
        store.close()
    seconds=time.perf_counter()-started
    results.append({'mode':mode,'seconds':seconds,'updates':40,'diskBytes':sum(p.stat().st_size for p in directory.iterdir()),
                    'note':'Incremental includes initial/final full snapshots and one replay validation'})
report={'sourceSha256':hashlib.sha256(raw).hexdigest(),'sourceBytes':len(raw),'results':results,
        'speedup':results[0]['seconds']/results[1]['seconds'],'scope':'Disk-write simulation only; no microscope or EDS commands'}
(out/'result.json').write_text(json.dumps(report,indent=2))
assert source.read_bytes()==raw
print(json.dumps(report,indent=2))
print(out)
