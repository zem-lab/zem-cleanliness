import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.cleanliness_core.preflight import run_preflight, ZemPreflightReader

ROOT = Path(__file__).resolve().parents[2]


class Reader:
    def __init__(self, **changes):
        self.values = dict(StatusCheck='Idle', GetHvStatus=1, GetStgStatus=1,
                           GetStagePos=(100.,200.), GetHV=(15000.,10.), GetSetHV=15000.,
                           IsChangingSamp=0, AllowOpenGun=True, GetMag=120.,
                           GetScanInfo={'fieldSizeUm':1058.3,'rotationDeg':0.})
        self.values.update(changes); self.calls=[]
    def read(self, command):
        self.calls.append(command)
        value=self.values[command]
        if isinstance(value,Exception): raise value
        if callable(value): return value()
        return value


@pytest.fixture
def inputs(tmp_path):
    source = ROOT/'backend/tests/fixtures/sample_cleanliness_report/optical.json'
    config=SimpleNamespace(settings_path=ROOT/'resources/conf/cleanliness_detection.json',
        output_dir=tmp_path, experience_calibration_path=ROOT/'resources/conf/optical_particle_calibration.json',
        microscope_host='unused.invalid',microscope_port=61557,eds_enabled=True)
    manifest={'input':{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},
              'selection':{'sequences':[1,2]},'calibration':{'result':{'matrix':[[1,0],[0,1]],'translation':[0,0]}}}
    return config,manifest


def check(inputs, reader=None, **kwargs):
    config,manifest=inputs
    return run_preflight(config,manifest,'analysis',reader=reader or Reader(),
        sleep=lambda _:None,disk_usage=lambda _:SimpleNamespace(free=20*1024**3),
        eds_check=kwargs.pop('eds_check',lambda:{'status':'READY'}),**kwargs)


def test_healthy_checks_are_read_only_and_limits_are_not_claimed_normal(inputs):
    before=copy.deepcopy(inputs[1]); reader=Reader()
    result=check(inputs,reader)
    assert result['passed'] and result['deviceActionsPerformed'] is False
    assert inputs[1]==before
    assert set(reader.calls) <= set(ZemPreflightReader.READS)
    assert reader.calls.count('GetHvStatus')==3 and reader.calls.count('GetStagePos')==2
    assert next(i for i in result['items'] if i['id']=='limits')['status']=='WARN'


@pytest.mark.parametrize('changes,failed',[
    ({'StatusCheck':'Busy'},'idle'),
    ({'GetStgStatus':0},'stage'),({'GetStagePos':(float('nan'),2)},'stage'),
    ({'IsChangingSamp':1},'sample'),({'IsChangingSamp':-1},'sample'),
    ({'GetHV':(0,0)},'voltage'),({'GetSetHV':20000},'voltage'),
    ({'GetHvStatus':TimeoutError('offline')},'gun'),
])
def test_unsafe_or_unknown_essential_state_blocks_start(inputs,changes,failed):
    result=check(inputs,Reader(**changes))
    assert not result['passed']
    assert next(i for i in result['items'] if i['id']==failed)['status']=='FAIL'


@pytest.mark.parametrize('status', [0, 2])
def test_closed_or_switching_gun_does_not_block_start(inputs, status):
    result = check(inputs, Reader(GetHvStatus=status))
    assert result['passed']
    assert next(item for item in result['items'] if item['id'] == 'gun')['status'] == 'WARN'


def test_closed_gun_zero_voltage_does_not_block_start(inputs):
    result = check(inputs, Reader(GetHvStatus=0, GetHV=(0, 0)))
    assert result['passed']
    assert next(item for item in result['items'] if item['id'] == 'gun')['status'] == 'WARN'
    assert next(item for item in result['items'] if item['id'] == 'voltage')['status'] == 'WARN'


def test_stage_drift_and_voltage_baseline_are_checked(inputs):
    positions=iter([(0,0),(2,0)])
    assert not check(inputs,Reader(GetStagePos=lambda:next(positions)))['passed']
    inputs[1]['baselineImaging']={'kv':10}
    assert not check(inputs)['passed']


def test_state_change_during_check_is_rechecked_before_start(inputs):
    values=iter([1,1,0])
    result=check(inputs,Reader(GetStgStatus=lambda:next(values)))
    assert not result['passed']
    assert next(i for i in result['items'] if i['id']=='final_state')['status']=='FAIL'


def test_known_scan_rotation_mismatch_blocks_before_capture(inputs):
    result=check(inputs,Reader(GetScanInfo={'fieldSizeUm':1058.3,'rotationDeg':30}))
    assert not result['passed']
    assert next(i for i in result['items'] if i['id']=='rotation')['status']=='FAIL'
    assert check(inputs,Reader(GetScanInfo={'fieldSizeUm':1058.3,'rotationDeg':360}))['passed']


def test_unknown_optional_states_are_explicit_warnings(inputs):
    result=check(inputs,Reader(IsChangingSamp=TimeoutError('unsupported'),AllowOpenGun=False,GetScanInfo=None))
    assert result['passed']
    assert all(next(i for i in result['items'] if i['id']==key)['status']=='WARN' for key in ['sample','interlock','imaging'])


def test_local_failure_prevents_all_device_queries(inputs):
    inputs[1]['input']['sha256']='changed'
    reader=Reader();result=check(inputs,reader)
    assert not result['passed'] and not reader.calls


def test_outside_roi_is_rejected_before_device_queries(inputs):
    inputs[1]['calibration']['result']['translation']=[1000000,1000000]
    reader=Reader();result=check(inputs,reader)
    assert not result['passed'] and not reader.calls


def test_recalibration_is_not_blocked_by_the_old_transform(inputs):
    inputs[1]['calibration']['result']['translation']=[1000000,1000000]
    result=run_preflight(*inputs,'calibration',reader=Reader(),sleep=lambda _:None)
    assert result['passed']
    assert next(i for i in result['items'] if i['id']=='targets')['status']=='SKIP'


def test_missing_model_and_low_disk_space_block_start(inputs):
    reader=Reader()
    result=check(inputs,reader,model_paths=[Path('missing-model.onnx')])
    assert not result['passed'] and not reader.calls
    config,manifest=inputs
    result=run_preflight(config,manifest,'analysis',reader=reader,sleep=lambda _:None,
        disk_usage=lambda _:SimpleNamespace(free=1))
    assert not result['passed'] and not reader.calls


def test_eds_required_only_for_enabled_analysis(inputs):
    bad=Mock(side_effect=RuntimeError('EDS unavailable'))
    assert not check(inputs,eds_check=bad)['passed']
    inputs[0].eds_enabled=False
    bad.reset_mock()
    assert check(inputs,eds_check=bad)['passed'];bad.assert_not_called()
    inputs[0].eds_enabled=True
    result=run_preflight(*inputs,'ab_refine',reader=Reader(),sleep=lambda _:None,eds_check=bad)
    assert result['passed'];bad.assert_not_called()


def test_disabled_and_cancelled_checks_do_not_touch_device(inputs):
    reader=Reader()
    assert run_preflight(None,None,'analysis',enabled=False,reader=reader)['passed']
    assert not check(inputs,reader,should_stop=lambda:True)['passed']
    assert not reader.calls


def test_reader_rejects_actuator_commands_before_any_io():
    with pytest.raises(ValueError):ZemPreflightReader('unused.invalid',61557).read('SetHV_ON')


def test_set_voltage_wire_kv_is_normalized_to_volts(monkeypatch):
    from zemrpc.zem.conn.protocol import append_crc
    body=append_crc('$GetSetHV:15,')
    monkeypatch.setattr('backend.cleanliness_core.microscope_safety._exchange_modern_command',lambda *a,**k:body.decode())
    assert ZemPreflightReader('unused.invalid',61557).read('GetSetHV')==15000


@pytest.mark.parametrize('token,expected',[('0',0),('1',1),('-1',-1),('false',0),('true',1)])
def test_sample_change_parser_does_not_treat_numeric_states_as_false(monkeypatch,token,expected):
    from zemrpc.zem.conn.protocol import append_crc
    # Use a real SDK-format reply to guard its legacy bool-only parser bug.
    body=append_crc(f'$IsChangingSamp:{token},')
    monkeypatch.setattr('backend.cleanliness_core.microscope_safety._exchange_modern_command',lambda *a,**k:body.decode())
    assert ZemPreflightReader('unused.invalid',61557).read('IsChangingSamp')==expected
