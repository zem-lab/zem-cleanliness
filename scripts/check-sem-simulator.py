"""External ZEM_SIM acceptance. Close the desktop backend first; gun stays off.

This script deliberately stops further writes after an unconfirmed operation.
It does not acknowledge quarantine, open the gun, or use a production ACL file.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / '.tmp/mcp-simulator-acceptance' / time.strftime('%Y%m%d-%H%M%S')
from backend.cleanliness_core.device_control import BackendOwnerLock, DeviceCoordinator
from backend.sem_control import SemControl, RealMicroscope
from backend.sem_rpc_catalog import WRITES, READS

SIM = 'http://192.168.0.207:8848'
RPC_HOST, RPC_PORT = '192.168.0.44', 61557
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
REPORT = {'scope':'external simulator only, gun closed', 'rpc':f'{RPC_HOST}:{RPC_PORT}',
          'simulator':SIM, 'cases':[], 'reads':[], 'skipped':{}}


def http(path, body=None):
    request = urllib.request.Request(SIM + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type':'application/json'})
    with HTTP.open(request, timeout=15) as response: return json.load(response)


def save():
    (OUT/'results.json').write_text(json.dumps(REPORT, ensure_ascii=False, indent=2), encoding='utf-8')


def ensure_no_pending(history_root):
    for prior_state in history_root.glob('*/control/control-state.json'):
        state = json.loads(prior_state.read_text('utf-8'))
        pending = state.get('unresolved', [])
        if pending or state.get('active_owner') or state.get('active_interrupt'):
            raise RuntimeError(f'Prior simulator test needs operator verification: {prior_state}, {state}')


def run():
    # Independent health/state guard: never run this against a physical backend.
    REPORT['health'] = http('/api/health')
    REPORT['before'] = http('/api/state')
    if REPORT['health'].get('version') != '0.9.0' or not REPORT['before'].get('ok'):
        raise RuntimeError('Unexpected simulator; no actions sent')
    device = RealMicroscope(RPC_HOST, RPC_PORT)
    # Save the actual photo properties too; the similarly named UDP getters do
    # not necessarily refer to the same settings. Discovery is read-only.
    import base64
    discovered = device.rpc_call('ZemRpc', ['simulator-test-baseline'])
    discovery = json.loads(base64.b64decode(discovered[0],validate=True).decode('utf-8'))
    endpoint = urlsplit(discovery['result']['url'])
    if endpoint.scheme != 'http' or not endpoint.port or endpoint.path != '/api/v1/jsonrpc':
        raise RuntimeError('Unexpected read-only status endpoint')
    status_request = urllib.request.Request(f'http://{RPC_HOST}:{endpoint.port}/api/v1/jsonrpc',
        data=json.dumps({'jsonrpc':'2.0','id':'simulator-test-baseline','method':'Status','params':[]}).encode(),
        headers={'Content-Type':'application/json'})
    with HTTP.open(status_request,timeout=10) as response: REPORT['upper_before'] = json.load(response)
    if not isinstance((REPORT['upper_before'].get('result') or {}).get('imaging'),dict):
        raise RuntimeError('Cannot preserve actual photo properties; no actions sent')
    save()
    control = SemControl(OUT/'control', device, DeviceCoordinator(OUT/'control/control-state.json'),
        allow_rpc=list(WRITES) + ['SetBeamShift'],
        bounds=[-50,-50,50,50], max_move_um=30, poll_interval=.12, rpc_timeout=8, move_timeout=8)
    def read(command): return control.call('sem_rpc_read', {'command':command}, 'test')['values']
    def finish(job):
        while True:
            result = control.call('sem_wait_job', {'job_id':job['job_id'],'timeout_seconds':20}, 'test')
            if not result['wait_timed_out']: return result
    def execute(command, values, request):
        assert read('GetHvStatus') == [0], 'Gun must remain closed'
        control.rpc_limits[command] = [[float(v),float(v)] for v in values]
        result = finish(control.call('sem_rpc_call', {'command':command,'values':values,'request_id':request}, 'test'))
        return result
    def require(result):
        if result['status'] != 'completed':
            raise RuntimeError('Stopped writes after non-completed operation: ' + json.dumps(
                {k:result.get(k) for k in ['job_id','status','error']},ensure_ascii=False))
    def parameter(command, target, *, restore_command=None, restore_values=None, state_key=None, no_change=False):
        getter = WRITES[command]['verify']
        before = read(getter)
        row = {'command':command, 'getter':getter, 'before':before, 'target':target,
               'verification_scope':'unchanged-value request only' if no_change else 'RPC setting/readback roundtrip',
               'sim_before':http('/api/state') if state_key else None}
        REPORT['cases'].append(row); save()
        row['set'] = execute(command,target,command+'-set'); save(); require(row['set'])
        if state_key:
            state = http('/api/state')['state']; row['sim_after_set'] = state
            observed = state
            for key in state_key: observed = observed[key]
            row['independent_value'] = observed
            expected = round(target[0] / 100 * 65535) if command == 'SetOB' else target[0]
            if not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=.001):
                raise RuntimeError(f'Independent simulator state mismatch: {command}, {observed}, {expected}')
        if command == 'SetMag' and before[0] != int(before[0]):
            # The wire command accepts integers, while the simulator stores doubles.
            # Restore the independently saved exact value through the simulator API;
            # do not pretend an integer RPC can restore a fractional magnification.
            exact = REPORT['before']['state']['optics']['magnification']
            row['restore_api'] = http('/api/optics/mag', {'magnification':exact})
            for _ in range(20):
                if math.isclose(read(getter)[0],exact,rel_tol=1e-8,abs_tol=1e-6): break
                time.sleep(.1)
            else: raise RuntimeError('Fractional magnification restoration unconfirmed')
            row['restore_note'] = 'Exact fractional simulator value restored via independent API, not an RPC roundtrip'
        else:
            values = restore_values if restore_values is not None else [int(v) if isinstance(v,bool) else v for v in before]
            row['restore'] = execute(restore_command or command,values,command+'-restore'); save(); require(row['restore'])
        row['after_restore'] = read(getter)
        row['passed'] = all(math.isclose(a,b,rel_tol=1e-6,abs_tol=.001) for a,b in zip(row['after_restore'],before))
        assert row['passed'], row
        print(json.dumps({'command':command,'passed':True,'scope':row['verification_scope']},ensure_ascii=False),flush=True)
        save()
    try:
        REPORT['snapshot_before'] = control.call('sem_get_snapshot',{})
        assert REPORT['snapshot_before']['gun_state'] == 0
        for command in READS:
            try: REPORT['reads'].append({'command':command,'values':read(command),'passed':True})
            except Exception as error: REPORT['reads'].append({'command':command,'passed':False,'error':str(error),'code':getattr(error,'code',None)})
        save()
        # Real, independently observable simulator controls first.
        ob = int(read('GetOB')[0])
        for command,target,key in [
            ('SetRotation',[read('GetRotation')[0]+1],None),
            ('SetHV',[read('GetSetHV')[0]+.1],['optics','landing_kV']),
            ('SetWD',[read('GetWD')[0]+.01],['stage','z_mm']),
            ('SetOBRaw',[ob+1],['optics','ob']),
            ('SetOB',[(ob+2)*100/65535],['optics','ob']),
        ]:
            parameter(command,target,state_key=key,restore_command='SetOBRaw' if command=='SetOB' else None,
                      restore_values=[ob] if command=='SetOB' else None)
        for command,target in [
            ('SetBeamSz',[1]),('SetAvMode',[1]),('SetAvFrames',[2]),
            ('SetSEGain',[100]),('SetSECV',[10]),('SetEDChannel',[1]),('SetAst',[10,20]),
            ('SetImgSz',[3]),('SetWindowPos',[16,16]),('SetStigX',[100]),('SetStigY',[100]),
            ('SetCompressA',[100]),('SetCompressB',[100]),
            ('SetBseCfg',[1,2048,1,2047,1,2047,1,2047]),('SetBseState',[0]),
            ('SetCustomRotation',[1]),
        ]: parameter(command,target)
        # These write only model fields or control a separate accessory. Check
        # current-value wire/readback without asserting an effect was exercised.
        for command in ['SetUpCenterX','SetUpCenterY','SetDownCenterX','SetDownCenterY','SetEDSEnable','VideoPlay']:
            original = read(WRITES[command]['verify'])
            parameter(command,[int(v) if isinstance(v,bool) else v for v in original],no_change=True)
        parameter('SetMag',[int(round(read('GetMag')[0]))+1],state_key=['optics','magnification'])
        # Exercise the bounded XY route and restore within the simulator's
        # explicitly reported repeatability; never label this exact restoration.
        before = list(read('GetStagePos'))
        row = {'command':'SetStagePos','before':before,'target':[before[0]+10,before[1]+10]}
        REPORT['cases'].append(row); save()
        for phase,values in [('set',row['target']),('restore',before)]:
            row[phase] = finish(control.call('sem_rpc_call',{'command':'SetStagePos','values':values,'request_id':'xy-'+phase},'test'))
            save(); require(row[phase])
            row['sim_'+phase] = http('/api/state')
        row['after_restore'] = list(read('GetStagePos'))
        row['restore_error_um'] = math.dist(before,row['after_restore'])
        row['passed'] = row['restore_error_um'] <= 2
        assert row['passed']; save()
        # Positive navigation capture is allowed while closed; new cache creation
        # is recorded separately from read-only retrieval of an existing cache.
        row = {'command':'SnapNav'}; REPORT['cases'].append(row); save()
        row['result'] = execute('SnapNav',[],'nav-create'); save()
        if row['result']['status'] == 'completed':
            nav = control.call('sem_rpc_read',{'command':'GetNavImg','request_id':'nav-after-create'},'test')
            row['navigation'] = finish(nav); require(row['navigation']); row['passed'] = True
        else:
            row['passed'] = False
            if control.coordinator.unresolved: require(row['result'])
        # This is deliberately last: accepted quarantines further normal writes.
        old = REPORT['before']['state']['optics']
        row = {'command':'SetBeamShift','target':[old['beam_shift_x_um']+1,old['beam_shift_y_um']-1]}
        REPORT['cases'].append(row); save()
        try:
            row['result'] = execute('SetBeamShift',row['target'],'beam-shift')
            row['sim_after_set'] = http('/api/state')
            optics = row['sim_after_set']['state']['optics']
            row['independent_effect_verified'] = all(math.isclose(a,b,abs_tol=1e-6) for a,b in zip(
                row['target'],[optics['beam_shift_x_um'],optics['beam_shift_y_um']]))
            row['passed'] = row['result']['status']=='accepted' and row['independent_effect_verified']
        finally:
            row['restore_api'] = http('/api/optics/beam_shift',{'x_um':old['beam_shift_x_um'],'y_um':old['beam_shift_y_um']})
            row['restore_note'] = 'Independent simulator API cleanup; accepted job is retained, not acknowledged or marked completed'
            save()
        REPORT['skipped'] = {
            'SetAvLenth':'Writes photo gear while GetAvLenth reads video gear; no valid UDP completion readback',
            'SetHV_ON':'Keep gun closed; AllowOpenGun false',
            'SetHV_OFF':'No gun-state transition prepared; keep current closed state',
            'AutoFocus':'Keep gun closed: positive imaging precondition unavailable',
            'Snap':'Keep gun closed: positive imaging precondition unavailable',
            'RequestOpenVac':'No modeled vacuum/exchange completion; not treating UI-only state as physical acceptance',
            'RequestCloseVac':'No modeled vacuum/exchange completion; no exchange started',
            'other_acknowledgement_actions':'No independent completion observation; no fabricated acceptance',
        }
    except Exception as error:
        REPORT['stopped_reason'] = str(error)
        print(json.dumps({'stopped':str(error),'evidence':str(OUT)},ensure_ascii=False),flush=True)
        raise
    finally:
        REPORT['after'] = http('/api/state')
        REPORT['snapshot_after'] = control.call('sem_get_snapshot',{})
        REPORT['gun_stayed_closed'] = REPORT['snapshot_after'].get('gun_state') == 0
        REPORT['unresolved_jobs'] = control.coordinator.unresolved
        save(); control.close()
        print(json.dumps({'evidence':str(OUT),'cases':len(REPORT['cases']),
            'passed':sum(bool(r.get('passed')) for r in REPORT['cases']),
            'reads_passed':sum(r['passed'] for r in REPORT['reads']),
            'gun_closed':REPORT['gun_stayed_closed'],'unresolved':REPORT['unresolved_jobs']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    # A fresh output directory must never hide a pending earlier external test.
    ensure_no_pending(OUT.parent)
    OUT.mkdir(parents=True, exist_ok=False)
    with BackendOwnerLock(): run()
