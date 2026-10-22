"""Compare field scanning vs explicit dirty records with real saved contours."""
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.cleanliness_core.sem_task_store import SemTaskStore,load_sem_task

def main():
    # Frozen parent of this optimization, not whatever HEAD is after committing.
    code=subprocess.check_output(['git','show','fa0bf7c:backend/cleanliness_core/sem_task_store.py'],cwd=ROOT).decode('utf-8')
    scope={'__name__':'backend.cleanliness_core._reference_store','__package__':'backend.cleanliness_core'}
    exec(code,scope)
    shapes=json.loads((ROOT/'backend/tests/fixtures/sem_performance_20260929/expected-shapes.json').read_text())
    original={'taskId':'OFFLINE-DIRTY-SAVE-BENCHMARK','input':{},'status':'RUNNING',
              'plan':{'tiles':[{'id':1,'x':0,'y':0,'status':'RUNNING'}]},
              'particles':[dict(contour=s['points'],sequence=i+1,tileId=1,status='PENDING') for i,s in enumerate(shapes)]}
    out=ROOT/'artifacts/sem-internal-performance'/time.strftime('dirty-save-%Y%m%d-%H%M%S');out.mkdir(parents=True)
    result={}
    for name,cls in [('before',scope['SemTaskStore']),('after',SemTaskStore)]:
        directory=out/name;directory.mkdir();m=copy.deepcopy(original);store=cls(directory,m)
        started=time.perf_counter()
        for i in range(200):
            p=m['particles'][(i//2)%len(shapes)];p['status']='ACQUIRING' if i%2==0 else 'SUCCEEDED'
            m['input']['selectedCount']=sum(p['status']=='SUCCEEDED' for p in m['particles'])
            store.save(m,m['plan']['tiles'][0],**({'dirty_particles':[p]} if name=='after' else {}))
        result[name]={'seconds':time.perf_counter()-started,'cacheBytes':sum(len(v) for v in store.particles.values())}
        assert load_sem_task(directory)['particles']==m['particles']
        store.close()
    result.update(particles=len(shapes),updates=200,identicalParticleState=True)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));print(out)

if __name__=='__main__':main()
