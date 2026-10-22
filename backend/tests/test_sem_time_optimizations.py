import copy
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.cleanliness_core import sem_direct, sem_task_store, optical_particle_task
from backend.cleanliness_core.atomic_json import write_json_atomic
from backend.tests.test_sem_direct import task, box, Microscope, Analyzer


def test_incremental_replay_and_checkpoints_are_idempotent(task):
    directory,m=task
    m['timingRuns']=[{'events':[],'stages':{}}]
    store=sem_task_store.SemTaskStore(directory,m)
    tile=m['plan']['tiles'][0]
    m['particles']=[dict(box(),sequence=1,tileId=1,status='ACQUIRING')]
    store.save(m,tile)
    assert sem_task_store.load_sem_task(directory)['particles'][0]['status']=='ACQUIRING'
    m['particles'][0].update(status='SUCCEEDED',spectrum={'elements':[{'symbol':'Fe','value':99}]})
    m['timingRuns'][0]['events'].append({'stage':'eds','seconds':7})
    store.save(m,tile)
    recovered=sem_task_store.load_sem_task(directory)
    assert recovered==m
    store.checkpoint(m)
    assert sem_task_store.load_sem_task(directory)==m
    store.close()
    m['timingRuns'].append({'events':[],'stages':{}})
    store=sem_task_store.SemTaskStore(directory,m)
    m['timingRuns'][-1]['events'].append({'stage':'connect','seconds':1})
    store.save(m,tile)
    assert sem_task_store.load_sem_task(directory)==m
    store.close()


def test_committed_acquisition_survives_process_loss_and_never_repeats_eds(task):
    directory,m=task
    store=sem_task_store.SemTaskStore(directory,m)
    m['status']='RUNNING'
    tile=m['plan']['tiles'][0]; tile['status']='RUNNING'
    m['particles']=[dict(box(),sequence=1,tileId=1,status='ACQUIRING')]
    store.save(m,tile)
    store.close()  # Simulate losing the process before any final checkpoint.
    assert not json.loads((directory/'task.json').read_text())['particles']
    analyzer=Analyzer(); scope=Microscope()
    result=sem_direct.run_task(directory,microscope=scope,analyzer=analyzer,segmenter=lambda *a:[box()])
    assert not analyzer.calls and scope.captures==0
    assert result['particles'][0]['status']=='REJECTED'
    assert result['plan']['tiles'][0]['status']=='INCOMPLETE'


def test_failed_transaction_does_not_advance_sequence(task):
    directory,m=task
    store=sem_task_store.SemTaskStore(directory,m)
    store.connection.execute("CREATE TRIGGER fail_commit BEFORE INSERT ON changes BEGIN SELECT RAISE(ABORT,'disk failure'); END")
    m['particles']=[dict(box(),sequence=1,tileId=1,status='ACQUIRING')]
    with pytest.raises(sqlite3.DatabaseError): store.save(m,m['plan']['tiles'][0])
    assert store.seq==0
    assert not sem_task_store.load_sem_task(directory)['particles']
    store.close()


def test_actual_process_exit_retains_committed_wal(task):
    directory,m=task
    script = '''
import os,sys
from backend.cleanliness_core.sem_task_store import load_sem_task,SemTaskStore
m=load_sem_task(sys.argv[1]); s=SemTaskStore(sys.argv[1],m)
m['particles']=[{'sequence':1,'tileId':1,'status':'ACQUIRING'}]
s.save(m,m['plan']['tiles'][0])
os._exit(17)
'''
    result=subprocess.run([sys.executable,'-c',script,str(directory)],capture_output=True,
                          cwd=Path(__file__).resolve().parents[2])
    assert result.returncode==17, result.stderr.decode(errors='replace')
    assert sem_task_store.load_sem_task(directory)['particles'][0]['status']=='ACQUIRING'


def test_eds_starts_only_after_acquiring_is_durable(task):
    directory,m=task
    class CheckedAnalyzer(Analyzer):
        def analyze(self,particle,path):
            state=sem_task_store.load_sem_task(directory)
            saved=next(p for p in state['particles'] if p['sequence']==particle['sequence'])
            assert saved['status']=='ACQUIRING'
            assert saved['edsAttempts'][-1]['status']=='RUNNING'
            return super().analyze(particle,path)
    sem_direct.run_task(directory,microscope=Microscope(),analyzer=CheckedAnalyzer(),segmenter=lambda *a:[box()])


def test_database_and_sequence_damage_fail_closed(task):
    directory,m=task
    store=sem_task_store.SemTaskStore(directory,m)
    store.save(m); store.save(m)
    store.connection.execute('DELETE FROM changes WHERE seq=1'); store.connection.commit()
    with pytest.raises(ValueError,match='序号'): sem_task_store.load_sem_task(directory)
    store.close()
    (directory/sem_task_store.DB_NAME).unlink()
    with pytest.raises(ValueError,match='缺失'): sem_task_store.load_sem_task(directory)


def test_only_changed_particle_is_in_transaction(task):
    directory,m=task
    m['particles']=[dict(box(),sequence=i+1,tileId=1,status='PENDING') for i in range(30)]
    store=sem_task_store.SemTaskStore(directory,m)
    m['particles'][10]['status']='ACQUIRING'
    store.save(m,m['plan']['tiles'][0])
    patch=json.loads(store.connection.execute('SELECT payload FROM changes').fetchone()[0])
    assert [p['sequence'] for p in patch['particles']]==[11]
    assert 'plan' not in patch['header'] and 'particles' not in patch['header']
    store.close()


@pytest.mark.parametrize('fail',[False,True])
def test_interruption_skips_full_report_but_keeps_data_and_timing(task,monkeypatch,fail):
    directory,m=task
    reports=[]
    monkeypatch.setattr(sem_direct,'write_reports',lambda *a:reports.append(a))
    scope=Microscope()
    if fail:
        def capture(path): raise RuntimeError('capture failed')
        scope.capture=capture
        with pytest.raises(RuntimeError): sem_direct.run_task(directory,microscope=scope,analyzer=Analyzer())
    else:
        sem_direct.run_task(directory,microscope=scope,analyzer=Analyzer(),should_stop=lambda:True)
    saved=sem_task_store.load_sem_task(directory)
    assert not reports and saved['reportState']=='DEFERRED'
    assert (directory/'timing_report.html').is_file()
    assert saved['status'] in {'FAILED','STOPPED'}


def test_complete_reports_once_and_manual_report_no_hardware(task,monkeypatch):
    from backend import server
    directory,m=task
    reports=[]
    monkeypatch.setattr(sem_direct,'write_reports',lambda *a:reports.append(a))
    result=sem_direct.run_task(directory,microscope=Microscope(),analyzer=Analyzer(),segmenter=lambda *a:[box()])
    assert len(reports)==1 and result['reportState']=='READY'
    monkeypatch.setattr(server,'TASK_ROOT',directory.parent)
    job=server.JOBS.create(directory.name,'sem_report')
    server._run_job_impl(job['id'],directory.name,'sem_report')
    assert len(reports)==2 and server.JOBS.get(job['id'])['status']=='completed'
    saved=sem_task_store.load_sem_task(directory)
    assert len(saved['manualReportTimings'])==1


def test_no_per_particle_full_manifest_write(task,monkeypatch):
    directory,m=task
    writes=[]
    original=sem_task_store.write_json_atomic
    monkeypatch.setattr(sem_task_store,'write_json_atomic',lambda path,value,**kwargs:(writes.append(path),original(path,value,**kwargs))[-1])
    shapes=[box(30+i*50,30,20) for i in range(8)]
    analyzer=Analyzer()
    sem_direct.run_task(directory,microscope=Microscope(),analyzer=analyzer,segmenter=lambda *a:shapes)
    assert len(analyzer.calls)==8
    assert len(writes)==3  # initial + before report + final, independent of particles.


class Clock:
    def __init__(self): self.now=0; self.sleeps=[]
    def monotonic(self): return self.now
    def sleep(self,seconds): self.sleeps.append(seconds); self.now+=seconds


def stage_scope(monkeypatch,statuses,positions):
    from zemrpc.zem.conn.protocol import EnumStageStatus
    clock=Clock()
    monkeypatch.setattr(optical_particle_task.time,'sleep',clock.sleep)
    monkeypatch.setattr(optical_particle_task.time,'monotonic',clock.monotonic)
    states=iter(statuses); points=iter(positions); moves=[]
    scope=optical_particle_task.ZemMicroscope('unused',1,{'responsiveStagePolling':True})
    def read():
        x,y=next(points); return SimpleNamespace(x=x,y=y)
    scope._api=SimpleNamespace(stage_xy_goto=lambda xy:moves.append(xy),get_stage_pos=read,
                               device=SimpleNamespace(get_stage_status=lambda:next(states,EnumStageStatus.Idle)))
    return scope,clock,moves


def test_responsive_polling_preserves_start_settle_and_two_position_reads(monkeypatch):
    from zemrpc.zem.conn.protocol import EnumStageStatus
    moving=next(s for s in EnumStageStatus if s!=EnumStageStatus.Idle)
    scope,clock,moves=stage_scope(monkeypatch,[moving,moving,EnumStageStatus.Idle],[(0,0),(100,0),(100,0)])
    result=scope.move_xy(100,0)
    assert result['attempts'][0]['withinTolerance']
    assert clock.sleeps==[.5,.2,.2,.3,.3]
    assert moves==[(100,0)]


def test_polling_does_not_accept_idle_with_off_target_coordinates(monkeypatch):
    from zemrpc.zem.conn.protocol import EnumStageStatus
    from backend.cleanliness_core.cleanliness_task import _move_stage_checked
    scope,clock,moves=stage_scope(monkeypatch,[EnumStageStatus.Idle]*3,[(0,0)]+[(80,0)]*6)
    with pytest.raises(RuntimeError,match='误差'): _move_stage_checked(scope,(100,0))
    assert len(moves)==3


def test_polling_timeout_and_bad_readback(monkeypatch):
    from zemrpc.zem.conn.protocol import EnumStageStatus
    scope,clock,moves=stage_scope(monkeypatch,[],[(0,0)])
    scope.api.device.get_stage_status=lambda:None
    with pytest.raises(TimeoutError): scope.move_xy(100,0)
    assert len(moves)==1  # Unknown motion must not be replayed blindly.
    scope,clock,moves=stage_scope(monkeypatch,[],[(float('nan'),0)])
    with pytest.raises(ValueError,match='坐标'): scope.move_xy(100,0)
    assert not moves
