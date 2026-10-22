from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import threading
import time

import pytest
from jsonschema import Draft202012Validator

from backend.cleanliness_core.device_control import DeviceBusy, DeviceCoordinator
from backend.sem_control import RealMicroscope, SemControl, SemError, SimulatedMicroscope
from backend.sem_rpc_catalog import READS, WRITES, UNAVAILABLE, catalog
from backend.sem_rpc_transport import exchange, parse_reply
from backend.sem_tools import TOOLS


@pytest.fixture
def control(tmp_path):
    service = SemControl(tmp_path, SimulatedMicroscope(), DeviceCoordinator(), allow_rpc=WRITES,
                         poll_interval=.001, rpc_timeout=.025)
    yield service
    service.close()


def submit(service, command, values=(), request='rpc-1'):
    return service.call('sem_rpc_execute', {'request_id': request, 'command': command, 'values': list(values)}, 'test')


def finish(service, job):
    return service.call('sem_wait_job', {'job_id': job['job_id'], 'timeout_seconds': 2}, 'test')


# Independent examples transcribed from the upper-computer wire contract.
CASES = [
    ('SetMag', [5000], 'GetMag', [5000]), ('SetHV', [5], 'GetSetHV', [5]),
    ('SetBeamSz', [2], 'GetBeamSz', [2]), ('SetAvMode', [1], 'GetAvMode', [1]),
    ('SetAvFrames', [2], 'GetAvFrames', [2]),
    ('SetSEGain', [20], 'GetSEGain', [20]), ('SetSECV', [100], 'GetSECV', [100]),
    ('SetOB', [50], 'GetOB', [32768]), ('SetOBRaw', [14000], 'GetOB', [14000]),
    ('SetRotation', [20], 'GetRotation', [20]), ('SetEDChannel', [0], 'GetEDChannel', [0]),
    ('SetWD', [8.2], 'GetWD', [8.2]), ('SetAst', [100, 200], 'GetAst', [100, 200]),
    ('SetImgSz', [2], 'GetImgSz', [2]), ('VideoPlay', [1], 'IsVideoAlive', [1]),
    ('SetWindowPos', [100, 200], 'GetWindowPos', [100, 200]),
    ('SetHV_ON', [], 'GetHvStatus', [1]), ('SetHV_OFF', [], 'GetHvStatus', [0]),
    ('RequestOpenVac', [], 'CheckRequestOpenVac', [0]), ('RequestCloseVac', [], 'CheckRequestCloseVac', [0]),
    ('SnapNav', [], 'CheckSnapNav', [0]), ('AutoFocus', [], 'CheckAutoFocus', [0]),
    ('AutoFocus', [14000, 2000], 'CheckAutoFocus', [0]),
    ('SetStigX', [-10], 'GetStigX', [-10]), ('SetStigY', [10], 'GetStigY', [10]),
    ('SetCompressA', [20], 'GetCompressA', [20]), ('SetCompressB', [30], 'GetCompressB', [30]),
    ('SetBseCfg', [0, 10, 1, 20, 2, 30, 3, 40], 'GetBseCfg', [0, 10, 1, 20, 2, 30, 3, 40]),
    ('SetBseState', [1], 'GetBseState', [True]), ('SetEDSEnable', [0], 'GetEDSEnable', [0]),
    ('SetUpCenterX', [10], 'GetUpCenterX', [10]), ('SetUpCenterY', [20], 'GetUpCenterY', [20]),
    ('SetDownCenterX', [30], 'GetDownCenterX', [30]), ('SetDownCenterY', [40], 'GetDownCenterY', [40]),
    ('SetCustomRotation', [45], 'GetCustomRotation', [45]),
]


@pytest.mark.parametrize('command,values,getter,readback', CASES)
def test_typed_write_confirms_independent_readback_once(control, monkeypatch, command, values, getter, readback):
    calls = []
    def status():
        return {'device_status': 'Idle', 'stage_state': 1, 'gun_state': 0 if command.startswith('Request') or command == 'SnapNav' else 1,
                'sample_changing': 0}
    def read(name):
        calls.append(('read', name))
        if name == 'AllowOpenGun': return [True]
        if name == 'IsChangingSamp': return [int(command == 'RequestOpenVac')]
        assert name == getter
        return readback
    monkeypatch.setattr(control.device, 'status', status)
    monkeypatch.setattr(control.device, 'rpc_read', read)
    monkeypatch.setattr(control.device, 'rpc_write', lambda cmd, args: calls.append(('write', cmd, args)) or [])
    result = finish(control, submit(control, command, values))
    assert result['status'] == 'completed', result
    assert result['result']['rpc']['readback'] == readback
    assert result['result']['rpc']['verified']
    assert [call for call in calls if call[0] == 'write'] == [('write', command, values)]
    assert sum(call == ('read', getter) for call in calls) >= 2
    assert submit(control, command, values)['job_id'] == result['job_id']
    assert len([call for call in calls if call[0] == 'write']) == 1


@pytest.mark.parametrize('command', sorted(READS))
def test_all_catalog_reads_are_readonly_and_keep_source(control, command):
    result = control.call('sem_rpc_read', {'command': command})
    assert result['command'] == command and result['simulated']
    assert result['unit'] == READS[command][1]
    assert control.device.commands == (['StatusCheck', command] if command != 'StatusCheck' else [command])


@pytest.mark.parametrize('command,values', [('SetMag', [0]), ('SetMag', [1.5]), ('SetWD', [float('nan')]),
    ('SetOBRaw', [65536]), ('SetCompressA', [-1]), ('SetAst', [1]), ('SetAst', [1, 2, 3]),
    ('SetEDChannel', [2]), ('SetHV_ON', [1]), ('SetHV', ['5,127.0.0.1:1']), ('SetMag', [True]),
    ('AutoFocus', [14000]), ('AutoFocus', [14000, 0]), ('AutoFocus', [14000, 2000, 1]),
    ('AutoFocus', [-1, 1000]), ('AutoFocus', [14000, 65536])])
def test_invalid_parameter_never_reaches_device(control, command, values):
    with pytest.raises(SemError) as error: submit(control, command, values)
    assert error.value.code == 'INVALID_ARGUMENT'
    assert not control.device.commands


def test_snapshot_serializes_concurrent_reads_without_self_busy(control, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = control.device.status
    def status():
        entered.set()
        assert release.wait(2)
        return original()
    monkeypatch.setattr(control.device, 'status', status)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(control.call, 'sem_get_status', {})
        assert entered.wait(1)
        second = pool.submit(control.call, 'sem_get_position', {})
        assert not second.done()
        release.set()
        assert first.result()['state_text']['gun_state'] == '开启'
        assert second.result()['position_um'] == (0, 0)
    assert control.coordinator.owner is None


@pytest.mark.parametrize('owner', ['desktop:analysis', 'mcp:physical-step'])
def test_read_queue_never_intrudes_into_device_job(control, owner):
    with control.coordinator.operation(owner):
        with pytest.raises(DeviceBusy): control.call('sem_get_snapshot', {})
    assert not control.device.commands


def test_bounded_queue_timeout_does_not_release_another_query(control):
    with control.coordinator.query('first'):
        with pytest.raises(DeviceBusy):
            with control.coordinator.query('second', timeout=.001): pytest.fail('must not acquire')
        assert control.coordinator.owner == 'mcp-read:first'
    assert control.coordinator.owner is None


def test_unknown_status_not_reported_as_healthy(control, monkeypatch):
    monkeypatch.setattr(control.device, 'status', lambda: {'device_status': 'Idle', 'gun_state': 123,
                                                        'stage_state': None, 'sample_changing': -1})
    result = control.call('sem_get_snapshot', {})
    assert set(result['state_text'].values()) == {'未知'}
    assert result['gun_state'] == 123 and '不代表' in result['diagnostic_limits']
    assert result['timing_ms']['device'] >= 0


def test_busy_snapshot_does_not_query_position(control, monkeypatch):
    monkeypatch.setattr(control.device, 'status', lambda: {'device_status': 'Busy'})
    assert 'position_um' not in control.call('sem_get_snapshot', {})
    assert not control.device.commands


def test_write_default_disabled_and_real_limits_required(tmp_path):
    device = SimulatedMicroscope()
    device.simulated = False
    device.host, device.port = 'not-contacted', 61557
    service = SemControl(tmp_path, device, DeviceCoordinator(), allow_rpc=['SetMag'])
    try:
        with pytest.raises(SemError, match='范围'): submit(service, 'SetMag', [5000])
        service.rpc_limits = {'SetMag': [[100, 1000]]}
        with pytest.raises(SemError) as error: submit(service, 'SetMag', [5000])
        assert error.value.code == 'OUT_OF_RANGE'
        with pytest.raises(SemError) as error: submit(service, 'SetHV_ON')
        assert error.value.code == 'OPERATION_DISABLED' and not device.commands
    finally: service.close()


@pytest.mark.parametrize('status,command', [
    ({'gun_state': 1}, 'RequestOpenVac'), ({'sample_changing': 1}, 'SetHV_ON'),
    ({'stage_state': None}, 'SetMag')])
def test_interlocks_prevent_dispatch(control, monkeypatch, status, command):
    monkeypatch.setattr(control.device, 'status', lambda: {'device_status': 'Idle', 'stage_state': 1,
        'sample_changing': 0, 'gun_state': 1, **status})
    values = [5000] if command == 'SetMag' else []
    result = finish(control, submit(control, command, values))
    assert result['status'] == 'failed'
    assert command not in control.device.commands


def test_autofocus_allows_closed_gun(control, monkeypatch):
    monkeypatch.setattr(control.device, 'status', lambda: {
        'device_status': 'Idle', 'stage_state': 1, 'sample_changing': 0, 'gun_state': 0})
    result = finish(control, submit(control, 'AutoFocus'))
    assert result['status'] == 'completed'
    assert control.device.commands.count('AutoFocus') == 1


def test_gun_on_requires_explicit_true_interlock(control):
    control.device.rpc_values['AllowOpenGun'] = [None]
    result = finish(control, submit(control, 'SetHV_ON'))
    assert result['status'] == 'failed' and 'SetHV_ON' not in control.device.commands


@pytest.mark.parametrize('mode', ['mismatch', 'lost_response', 'verify_exception'])
def test_unknown_write_quarantines_and_never_replays(control, monkeypatch, mode):
    calls = []
    def write(cmd, values):
        calls.append(cmd)
        if mode == 'lost_response': raise TimeoutError('lost')
        return []
    def read(cmd):
        if mode == 'verify_exception': raise OSError('offline')
        return [-1]
    monkeypatch.setattr(control.device, 'rpc_write', write)
    monkeypatch.setattr(control.device, 'rpc_read', read)
    result = finish(control, submit(control, 'SetMag', [5000]))
    assert result['status'] == 'unknown' and result['job_id'] in control.coordinator.unresolved
    assert submit(control, 'SetMag', [5000])['status'] == 'unknown'
    with pytest.raises(DeviceBusy): submit(control, 'SetMag', [5000], request='new')
    assert calls == ['SetMag']


def test_wait_timeout_does_not_cancel_or_release_active_operation(control, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = control.device.rpc_write
    def write(*args):
        entered.set()
        assert release.wait(2)
        return original(*args)
    monkeypatch.setattr(control.device, 'rpc_write', write)
    job = submit(control, 'SetMag', [5000])
    try:
        assert entered.wait(1)
        waiting = control.call('sem_wait_job', {'job_id': job['job_id'], 'timeout_seconds': .001}, 'test')
        assert waiting['wait_timed_out'] and not waiting['cancel_requested']
        assert control.coordinator.owner == 'mcp:' + job['job_id']
        with pytest.raises(SemError) as error:
            control.call('sem_wait_job', {'job_id': job['job_id']}, 'another')
        assert error.value.code == 'NOT_FOUND'
    finally: release.set()
    assert finish(control, job)['status'] == 'completed'


def test_catalog_and_schema_are_valid_without_hardware(control):
    source = json.loads((Path(__file__).parent / 'fixtures/sem_rpc_hisi_inventory.json').read_text(encoding='utf-8'))
    assert {entry['command'] for entry in catalog()} == set(source['commands'])
    assert len(catalog()) == 115  # 110 model registrations + 5 ExternServer core commands.
    assert len({entry['command'] for entry in catalog()}) == 115
    assert set(command for command, *_ in CASES) == {c for c,s in WRITES.items() if s['verify'] is not None}
    for tool in TOOLS: Draft202012Validator.check_schema(tool['inputSchema'])
    unavailable = control.call('sem_list_rpc', {'kind': 'unavailable'})['commands']
    assert {entry['command'] for entry in unavailable} == set(UNAVAILABLE)
    assert not control.device.commands


@pytest.mark.parametrize('sample', [0, 1])
def test_navigation_capture_allows_device_specific_exchange_guard(control, sample):
    control.device.rpc_values.update(GetHvStatus=[0], IsChangingSamp=[sample])
    result = finish(control, submit(control, 'SnapNav'))
    assert result['status'] == 'completed', result
    assert control.device.commands.count('SnapNav') == 1


@pytest.mark.parametrize('state', [{'GetHvStatus': [1]}, {'IsChangingSamp': [-1]}])
def test_navigation_capture_rejects_open_gun_or_unknown_exchange(control, state):
    control.device.rpc_values['GetHvStatus'] = [0]
    control.device.rpc_values.update(state)
    result = finish(control, submit(control, 'SnapNav'))
    assert result['status'] == 'failed'
    assert 'SnapNav' not in control.device.commands


@pytest.mark.parametrize('text,expected', [('$GetMag:5000,', [5000]), ('$GetBseState:True,', [True]),
    ('$SetRotation:Acknowledge,Error Code:0,', ['Acknowledge', 'Error Code:0']),
    ('$GetModel:ZEM_SIM,', ['ZEM_SIM']), ('$GetStagePos:1.2,-3.4,', [1.2, -3.4])])
def test_reply_parser_preserves_types(text, expected):
    from zemrpc.zem.conn.protocol import append_crc
    assert parse_reply(text[1:].split(':')[0], append_crc(text)) == expected


@pytest.mark.parametrize('payload,code', [('$GetMag:Error Code:5,', 'UNSUPPORTED'),
    ('$GetMag:Error Code:3,', 'DEVICE_BUSY'), ('$GetMag:NaN,', 'INVALID_RESPONSE')])
def test_reply_errors_never_look_successful(payload, code):
    from zemrpc.zem.conn.protocol import append_crc
    with pytest.raises(SemError) as error: parse_reply('GetMag', append_crc(payload))
    assert error.value.code == code


def test_mismatched_or_bad_crc_replies_rejected():
    from zemrpc.zem.conn.protocol import append_crc
    for reply in (append_crc('$GetMag:5,'), b'$GetWD:8.2,0000%'):
        with pytest.raises(SemError): parse_reply('GetWD', reply)


def test_real_rpc_transport_uses_one_framed_request_without_sdk_api(monkeypatch):
    from zemrpc.zem.conn.protocol import append_crc
    from backend.cleanliness_core import microscope_safety
    monkeypatch.setattr(microscope_safety, '_route_local_ip', lambda *args: '127.0.0.1')
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
        server.bind(('127.0.0.1', 0))
        requests = []
        def reply():
            data, addr = server.recvfrom(4096)
            requests.append(data)
            server.sendto(append_crc('$SetMag:Acknowledge,Error Code:0,'), addr)
        worker = threading.Thread(target=reply)
        worker.start()
        assert exchange('127.0.0.1', server.getsockname()[1], 'SetMag', [5000]) == ['Acknowledge', 'Error Code:0']
        worker.join(1)
        assert len(requests) == 1 and requests[0].startswith(b'$SetMag:5000,127.0.0.1:')
        assert append_crc(requests[0][:-5]) == requests[0]
