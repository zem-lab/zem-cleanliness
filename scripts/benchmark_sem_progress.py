"""Measure progress transport on a copy of the completed real task data."""
import copy
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.cleanliness_core.sem_task_store import SemTaskStore
from backend.cleanliness_core.sem_progress import read_progress

def main():
    source=Path.home()/'Documents/ZEM Cleanliness/tasks/b7efbb36-c863-4c0b-9d20-0a003c570140/task.json'
    raw=source.read_bytes();m=json.loads(raw)
    out=ROOT/'artifacts/sem-internal-performance'/time.strftime('progress-%Y%m%d-%H%M%S');out.mkdir(parents=True)
    start=time.perf_counter()
    for _ in range(20):old=json.dumps(json.loads(raw),ensure_ascii=False).encode()
    before=(time.perf_counter()-start)/20
    store=SemTaskStore(out,copy.deepcopy(m))
    first=read_progress(out)
    store.save(m,m['plan']['tiles'][282],dirty_particles=[])
    start=time.perf_counter()
    for _ in range(20):delta=json.dumps(read_progress(out,first['cursor']),ensure_ascii=False).encode()
    after=(time.perf_counter()-start)/20
    stable=json.dumps(read_progress(out,read_progress(out)['cursor'])).encode()
    store.close()
    report={'fullBytes':len(old),'initialBytes':len(json.dumps(first,ensure_ascii=False).encode()),
            'changedBytes':len(delta),'unchangedBytes':len(stable),'beforeMeanSeconds':before,
            'afterMeanSeconds':after,'changedPayloadReduction':1-len(delta)/len(old),
            'scope':'20 local calls, real task copy; before excludes disk reads; no device access'}
    (out/'result.json').write_text(json.dumps(report,indent=2))
    assert source.read_bytes()==raw
    print(json.dumps(report,indent=2));print(out)

if __name__=='__main__':main()
