"""Full RPC routes and wire contracts, including failures and durable ACKs."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import socket
import threading

import numpy as np
import pytest

from backend.cleanliness_core.device_control import DeviceBusy, DeviceCoordinator
from backend.sem_admin import acknowledge
from backend.sem_control import SemControl, SemError, SimulatedMicroscope
from backend.sem_rpc_full import SPECS, WRITE_COMMANDS, validate, validate_paths, wire_values
from backend.sem_rpc_transport import exchange, image_exchange
from backend.tests.test_sem_rpc_expansion import CASES


# Examples transcribed from ExternModel and ExternServer, not generated from SPECS.
EXTRA = {
    'SetAvLenth': [3],
    'SetBrightness': [40], 'SetContrast': [60], 'SetBeamShift': [1.5, -2],
    'OpenVAC': [], 'ReleaseVAC': [],
    'SetStageXPos': [.001], 'SetStageYPos': [-.001], 'SetStageZPos': [0],
    'SetStageRPos': [0], 'SetStageTPos': [0], 'SetStgTarget': [0, 0, .1, -.1, 100],
    'SetAutoSnapTask': [r'C:\MCP\任务.json'],
    'SaveImage': [r'C:\MCP\图像', 'sample.001', 'png', 1, 'BSE'],
    'StartScanPoints': [0, 0, 10, 20], 'StopScanPoints': [], 'SnapCancel': [],
    'SetIe': [1], 'ZemRpc': ['discovery-1'], 'SetStagePos': [1, 2],
    'GetNavImgBuf': [], 'Snap': [0, 1, 1, 3], 'GetImg': [0], 'GetNavImg': [],
}
INVENTORY = json.loads((Path(__file__).parent / 'fixtures/sem_rpc_hisi_inventory.json').read_text('utf-8'))['commands']
WRITE_CASES = {c: (v, g, r) for c, v, g, r in CASES}


@pytest.fixture
def control(tmp_path):
    service = SemControl(tmp_path, SimulatedMicroscope(), DeviceCoordinator(tmp_path / 'control-state.json'),
        allow_rpc=WRITE_COMMANDS, allow_capture=True, bounds=[-100, -100, 100, 100], max_move_um=100,
        rpc_path_roots=[r'C:\MCP'], poll_interval=.001, rpc_timeout=.03)
    yield service
    service.close()


def call(service, command, values, request='full-1', caller='test'):
    return service.call('sem_rpc_call', {'command': command, 'values': values, 'request_id': request}, caller)


def finish(service, job):
    return service.call('sem_wait_job', {'job_id': job['job_id'], 'timeout_seconds': 2}, 'test')


@pytest.mark.parametrize('command', INVENTORY)
def test_every_audited_rpc_has_executable_typed_route(control, monkeypatch, command):
    assert set(INVENTORY) == set(SPECS)
    if command in WRITE_CASES:
        values, getter, expected = WRITE_CASES[command]
        monkeypatch.setattr(control.device, 'status', lambda: {'device_status': 'Idle', 'stage_state': 1,
            'sample_changing': 0, 'gun_state': 0 if command.startswith('Request') or command == 'SnapNav' else 1})
        def read(name):
            if name == 'AllowOpenGun': return [True]
            if name == 'IsChangingSamp': return [int(command == 'RequestOpenVac')]
            assert name == getter
            return expected
        monkeypatch.setattr(control.device, 'rpc_read', read)
        record = finish(control, call(control, command, values))
        assert record['status'] == 'completed', record
        assert record['result']['rpc']['verified']
        assert record['result']['rpc']['readback'] == expected
    elif command in EXTRA:
        if command == 'SetIe':
            with pytest.raises(SemError) as error: call(control, command, EXTRA[command])
            assert error.value.code == 'UNSUPPORTED' and not control.device.commands
            return
        if command in {'OpenVAC', 'ReleaseVAC'}: control.device.rpc_values['GetHvStatus'] = [0]
        result = call(control, command, EXTRA[command])
        if command in {'ZemRpc', 'GetNavImgBuf'}:
            assert result['completion_confirmed']
            assert command in control.device.commands
            return
        record = finish(control, result)
        if command in {'Snap', 'GetImg', 'GetNavImg', 'SetStagePos'}:
            assert record['status'] == 'completed', record
            if command == 'SetStagePos': assert control.device.position == (1, 2)
            else:
                image = record['result']['image']
                assert image['rpc_command'] == command and image['sha256'] and image['position_verified']
                assert image['width'] == (128 if command == 'Snap' else 512)
        else:
            assert record['status'] == 'accepted', record
            assert record['result']['rpc']['completion_confirmed'] is False
            assert record['result']['rpc']['verified'] is False
            assert 'completed_at' not in record['result']
            assert record['job_id'] in control.coordinator.unresolved
    else:
        result = call(control, command, [])
        assert result['command'] == command and result['simulated']
        assert command in control.device.commands


@pytest.mark.parametrize('command,values', list(EXTRA.items()))
def test_new_wire_arguments_are_encoded_and_sent_once(monkeypatch, command, values):
    from zemrpc.zem.conn.protocol import append_crc
    from backend.cleanliness_core import microscope_safety
    monkeypatch.setattr(microscope_safety, '_route_local_ip', lambda *args: '127.0.0.1')
    expected = list(values)
    if command in {'SaveImage', 'SetAutoSnapTask'}:
        expected[0] = base64.b64encode(values[0].encode('utf-8')).decode('ascii')
    if command == 'ZemRpc':
        expected = [base64.b64encode(json.dumps({'jsonrpc': '2.0', 'id': values[0]}).encode()).decode()]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
        server.bind(('127.0.0.1', 0)); server.settimeout(2)
        requests = []
        def reply():
            data, address = server.recvfrom(16384)
            requests.append(data)
            server.sendto(append_crc(f'${command}:Acknowledge,Error Code:0,'), address)
        worker = threading.Thread(target=reply); worker.start()
        try: assert exchange('127.0.0.1', server.getsockname()[1], command, values) == ['Acknowledge', 'Error Code:0']
        finally: worker.join(3)
    assert len(requests) == 1
    assert requests[0].startswith(('$' + command + ':' + ','.join(map(str, expected)) + ',127.0.0.1:').encode())
    assert append_crc(requests[0][:-5]) == requests[0]


@pytest.mark.parametrize('command,values', [
    ('SetBrightness', [-1]), ('SetContrast', [101]), ('SetBeamShift', [float('nan'), 0]),
    ('SetBeamShift', ['1,127.0.0.1:1', 0]), ('SetStageZPos', [True]),
    ('StartScanPoints', [1, 2, 3]), ('StartScanPoints', []), ('StartScanPoints', [1.5, 2]),
    ('Snap', [0, 8, 1, 0]), ('Snap', [0, 1, 0, 0]), ('Snap', [0, 1, 1, 4]),
    ('SaveImage', [r'C:\MCP', '../escape', 'png', 0, 'ALL']),
    ('SaveImage', [r'C:\MCP', '..', 'png', 0, 'ALL']),
    ('ZemRpc', [{'method': 'SetHV_ON'}]), ('StopScanPoints', [1]),
])
def test_new_invalid_arguments_do_not_dispatch(control, command, values):
    with pytest.raises(SemError) as error: call(control, command, values)
    assert error.value.code == 'INVALID_ARGUMENT' and not control.device.commands


@pytest.mark.parametrize('path', [r'C:\OTHER\out', r'C:\MCP-other', r'C:\MCP\..\out',
    r'C:\MCP\sub\.. \out', r'C:\MCP\x:stream', r'\\server\MCP\out',
    r'\\?\C:\MCP\out', r'C:MCP\out', 'relative.json'])
def test_remote_path_escape_rejected_before_dispatch(control, path):
    with pytest.raises(SemError): call(control, 'SetAutoSnapTask', [path])
    assert not control.device.commands


def test_encoded_paths_preserve_unicode_delimiters_and_percent():
    path = r'C:\MCP\图片,100%\task.json'
    validate_paths('SetAutoSnapTask', [path], [r'C:\MCP'])
    assert base64.b64decode(wire_values('SetAutoSnapTask', [path])[0]).decode('utf-8') == path


def test_accepted_is_durable_blocks_new_writes_and_never_replays(control):
    result = finish(control, call(control, 'SetBrightness', [40]))
    assert result['status'] == 'accepted'
    assert call(control, 'SetBrightness', [40])['job_id'] == result['job_id']
    with pytest.raises(DeviceBusy): call(control, 'SetContrast', [50], 'new')
    assert control.device.commands.count('SetBrightness') == 1
    assert control.call('sem_cancel_job', {'job_id': result['job_id']}, 'test')['status'] == 'accepted'
    assert call(control, 'GetWD', [])['values'] == [8]
    restarted = SemControl(control.root, SimulatedMicroscope(), DeviceCoordinator(control.root / 'control-state.json'),
                           allow_rpc=WRITE_COMMANDS)
    try:
        assert call(restarted, 'SetBrightness', [40])['status'] == 'accepted'
        with pytest.raises(DeviceBusy): call(restarted, 'SetContrast', [50], 'new')
        assert not restarted.device.commands
    finally: restarted.close()


def test_operator_acknowledgement_preserves_accepted_evidence(control):
    result = finish(control, call(control, 'SetBrightness', [40]))
    acknowledge(control.root, result['job_id'], 'simulator state independently checked', simulated=True)
    state = json.loads((control.root / 'control-state.json').read_text('utf-8'))
    record = control.call('sem_get_job', {'job_id': result['job_id']}, 'test')
    assert not state['unresolved']
    assert record['operator_acknowledgement']['previous_status'] == 'accepted'
    assert record['status'] == 'failed' and record['result']['rpc']['completion_confirmed'] is False
    assert control.device.commands.count('SetBrightness') == 1


@pytest.mark.parametrize('mode,status', [('reject', 'failed'), ('timeout', 'unknown'), ('malformed', 'unknown')])
def test_ack_failure_is_not_success_and_not_retried(control, monkeypatch, mode, status):
    calls = []
    def rpc(command, values):
        calls.append(command)
        if mode == 'reject': raise SemError('RPC_REJECTED', 'upper rejected')
        if mode == 'timeout': raise TimeoutError('lost reply')
        return ['unexpected']
    monkeypatch.setattr(control.device, 'rpc_call', rpc)
    record = finish(control, call(control, 'SetContrast', [50]))
    assert record['status'] == status and record['error']
    assert call(control, 'SetContrast', [50])['job_id'] == record['job_id']
    assert calls == ['SetContrast']
    assert (record['job_id'] in control.coordinator.unresolved) == (status == 'unknown')


def test_explicit_stop_can_overlap_job_without_releasing_original_owner(control, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = control.device.rpc_write
    def write(*args):
        entered.set()
        assert release.wait(3)
        return original(*args)
    monkeypatch.setattr(control.device, 'rpc_write', write)
    job = call(control, 'SetMag', [5000])
    try:
        assert entered.wait(1)
        stop = finish(control, call(control, 'SnapCancel', [], 'stop'))
        assert stop['status'] == 'accepted' and not stop['result']['rpc']['completion_confirmed']
        assert control.coordinator.owner == 'mcp:' + job['job_id']
        with pytest.raises(DeviceBusy): call(control, 'GetWD', [])
    finally: release.set()
    assert finish(control, job)['status'] == 'completed'
    assert control.coordinator.unresolved == [stop['job_id']]


def test_interrupt_owner_and_crash_quarantine_are_independent(tmp_path):
    path = tmp_path / 'state.json'
    coordinator = DeviceCoordinator(path)
    coordinator.acquire('desktop:analysis')
    coordinator.acquire_interrupt('mcp:stop')
    assert coordinator.owner == 'desktop:analysis'
    restored = DeviceCoordinator(path)
    assert set(restored.unresolved) == {'desktop:analysis', 'stop'}
    coordinator.release('desktop:analysis')
    assert coordinator.owner == 'mcp:stop'
    with pytest.raises(DeviceBusy): coordinator.acquire('new')
    with pytest.raises(DeviceBusy):
        with coordinator.query('new'): pytest.fail('query must not interrupt stop')
    coordinator.release('mcp:stop')
    with coordinator.query('short'):
        with pytest.raises(DeviceBusy): coordinator.acquire_interrupt('stop2')
    assert coordinator.owner is None


def test_new_permissions_limits_and_catalog_match_effective_controls(control):
    control.allow_rpc = frozenset()
    for command in ['SetBrightness', 'OpenVAC', 'StopScanPoints', 'Snap']:
        with pytest.raises(SemError) as error: call(control, command, EXTRA[command])
        assert error.value.code == 'OPERATION_DISABLED'
    control.allow_rpc = WRITE_COMMANDS
    control.allow_capture = False
    with pytest.raises(SemError): call(control, 'Snap', EXTRA['Snap'])
    entries = {x['command']: x for x in control.call('sem_list_rpc', {})['commands']}
    assert not entries['Snap']['enabled'] and entries['SetStagePos']['enabled']
    control.device.simulated = False
    with pytest.raises(SemError, match='范围'): call(control, 'SetBrightness', [40])
    control.rpc_limits = {'SetBrightness': [[0, 20]]}
    with pytest.raises(SemError) as error: call(control, 'SetBrightness', [40])
    assert error.value.code == 'OUT_OF_RANGE' and not control.device.commands


@pytest.mark.parametrize('command', ['Snap', 'StartScanPoints', 'SetAutoSnapTask'])
def test_new_imaging_commands_allow_closed_gun(control, command):
    control.device.rpc_values['GetHvStatus'] = [0]
    record = finish(control, call(control, command, EXTRA[command]))
    assert record['status'] in {'completed', 'accepted'}
    assert control.device.commands.count(command) == 1


@pytest.mark.parametrize('command,depth,shape,dtype', [
    ('Snap', 7, (2, 3), '<u2'), ('GetImg', 8, (2, 3), 'u1'),
    ('GetNavImg', 24, (2, 3, 3), 'u1'), ('GetNavImg', 32, (2, 3, 4), 'u1')])
def test_actual_udp_tcp_images_preserve_pixels_and_depth(monkeypatch, command, depth, shape, dtype):
    from zemrpc.zem.conn.protocol import append_crc
    from backend.cleanliness_core import microscope_safety
    monkeypatch.setattr(microscope_safety, '_route_local_ip', lambda *args: '127.0.0.1')
    original = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    with socket.socket() as tcp, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        tcp.bind(('127.0.0.1', 0)); tcp.listen(1); tcp.settimeout(3)
        udp.bind(('127.0.0.1', 0)); udp.settimeout(3)
        requests = []
        def serve():
            frame, address = udp.recvfrom(16384); requests.append(frame)
            udp.sendto(append_crc(f'${command}:ZEM20 Image service:127.0.0.1:{tcp.getsockname()[1]},{depth},3,2,'), address)
            with tcp.accept()[0] as client:
                raw = original.tobytes(); client.sendall(raw[:3]); client.sendall(raw[3:])
        worker = threading.Thread(target=serve); worker.start()
        try: received, meta = image_exchange('127.0.0.1', udp.getsockname()[1], command, EXTRA[command], timeout=2)
        finally: worker.join(4)
    expected = original if len(shape) == 2 else original[:, :, [2, 1, 0]]
    np.testing.assert_array_equal(received, expected)
    assert received.dtype == np.dtype(dtype) and meta['raw_transfer_bytes'] == original.nbytes
    assert len(requests) == 1


@pytest.mark.parametrize('reply', [
    ['ZEM20 Image service:192.0.2.1:1234', 8, 2, 2],
    ['ZEM20 Image service:127.0.0.1:0', 8, 2, 2],
    ['ZEM20 Image service:127.0.0.1:1234', 16, 2, 2],
    ['ZEM20 Image service:127.0.0.1:1234', 8, 8193, 2],
    ['ZEM20 Image service:127.0.0.1:1234', 32, 8192, 8192], []])
def test_invalid_image_reply_never_opens_tcp(monkeypatch, reply):
    from backend import sem_rpc_transport
    monkeypatch.setattr(sem_rpc_transport, 'exchange', lambda *a, **k: reply)
    monkeypatch.setattr(socket, 'create_connection', lambda *a, **k: pytest.fail('must reject before connect'))
    with pytest.raises(SemError) as error: image_exchange('127.0.0.1', 1, 'GetNavImg', [])
    assert error.value.code == 'INVALID_RESPONSE'


def test_truncated_image_not_accepted(monkeypatch):
    from backend import sem_rpc_transport
    with socket.socket() as server:
        server.bind(('127.0.0.1', 0)); server.listen(1); server.settimeout(3)
        monkeypatch.setattr(sem_rpc_transport, 'exchange', lambda *a, **k:
            [f'ZEM20 Image service:127.0.0.1:{server.getsockname()[1]}', 8, 2, 2])
        def serve():
            with server.accept()[0] as client: client.sendall(b'\x01\x02')
        worker = threading.Thread(target=serve); worker.start()
        try:
            with pytest.raises(SemError) as error: image_exchange('127.0.0.1', 1, 'GetNavImg', [], timeout=2)
            assert error.value.code == 'INVALID_IMAGE'
        finally: worker.join(4)


@pytest.mark.parametrize('wrong_id', [False, True])
def test_zemrpc_decodes_discovery_only_and_rejects_wrong_id(control, monkeypatch, wrong_id):
    control.device.simulated = False
    payload = {'jsonrpc': '2.0', 'id': 'wrong' if wrong_id else 'discovery-1',
               'result': {'name': 'ZEM_SIM', 'url': 'http://192.0.2.1:8848'}}
    monkeypatch.setattr(control.device, 'rpc_call', lambda *a:
        [base64.b64encode(json.dumps(payload).encode()).decode()])
    if wrong_id:
        with pytest.raises(SemError) as error: call(control, 'ZemRpc', ['discovery-1'])
        assert error.value.code == 'INVALID_RESPONSE'
    else: assert call(control, 'ZemRpc', ['discovery-1'])['values'] == payload


def test_all_agent_prompts_explain_full_rpc_and_acknowledgement():
    from backend.chat_runner import build_prompt, build_claude_prompt, build_system_prompt
    for prompt in [build_prompt('question', '{}'), build_claude_prompt('question', '{}'), build_system_prompt('{}')]:
        assert all(term in prompt for term in ['sem_rpc_call', 'accepted', 'completion_confirmed=false', 'SetIe', 'ZemRpc'])


@pytest.mark.parametrize('command,values', [('ZemRpc', ['read-only']), ('GetNavImgBuf', []), ('GetImg', [0]), ('GetNavImg', [])])
def test_readonly_entry_exposes_new_reads_without_action_annotations(control, command, values):
    from backend.sem_tools import TOOL_MAP
    assert TOOL_MAP['sem_rpc_read']['annotations']['readOnlyHint']
    result = control.call('sem_rpc_read', {'command': command, 'values': values, 'request_id': 'query'}, 'test')
    if command in {'GetImg', 'GetNavImg'}:
        result = finish(control, result)
        assert result['status'] == 'completed' and result['result']['image']['sha256']
    else: assert result['completion_confirmed']
    assert not control.coordinator.unresolved


@pytest.mark.parametrize('command', sorted(WRITE_COMMANDS | {'SetIe'}))
def test_readonly_entry_rejects_every_action_before_device_access(control, command):
    with pytest.raises(SemError) as error:
        control.call('sem_rpc_read', {'command': command, 'values': EXTRA.get(command, []), 'request_id':'query'}, 'test')
    assert error.value.code == 'INVALID_ARGUMENT' and not control.device.commands


def test_interrupted_readonly_image_does_not_quarantine_hardware(control):
    record = finish(control, call(control, 'GetImg', [0]))
    assert record['writes_device'] is False
    record.update(status='running')  # Reproduce a process lost while reading existing pixels.
    control._save(record)
    restarted = SemControl(control.root, SimulatedMicroscope(), DeviceCoordinator(control.root/'control-state.json'))
    try:
        assert restarted.call('sem_get_job', {'job_id':record['job_id']}, 'test')['status'] == 'unknown'
        assert not restarted.coordinator.unresolved
        assert not restarted.device.commands
    finally: restarted.close()


@pytest.mark.parametrize('video_gear', [3, 5])
@pytest.mark.parametrize('entry', ['sem_rpc_call','sem_rpc_execute'])
def test_snap_average_never_verified_by_different_video_property(control, video_gear, entry):
    # Reproduce the external upper computer: writing photo gear leaves video gear
    # unchanged. Even an accidental equal value must not count as confirmation.
    control.device.rpc_values['GetAvLenth'] = [video_gear]
    job = control.call(entry, {'command':'SetAvLenth','values':[3],'request_id':'separate-photo-video'}, 'test')
    result = finish(control, job)
    assert result['status'] == 'accepted'
    assert result['result']['rpc']['completion_confirmed'] is False
    assert result['result']['rpc']['verified'] is False
    assert 'completed_at' not in result['result']
    assert 'GetAvLenth' not in control.device.commands
    assert control.device.rpc_values['GetAvLenth'] == [video_gear]
    assert result['job_id'] in control.coordinator.unresolved
    with pytest.raises(DeviceBusy): call(control, 'SetMag', [5000], 'blocked-after-photo')


@pytest.mark.parametrize('state', [{'unresolved':['external-unknown']}, {'active_owner':'mcp:interrupted'},
                                  {'active_interrupt':'mcp:stop-interrupted'}])
def test_external_diagnostic_cannot_hide_pending_job_in_new_run_directory(tmp_path, monkeypatch, state):
    import runpy
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/check-sem-simulator.py'))
    path = tmp_path/'previous-run/control/control-state.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(state),encoding='utf-8')
    monkeypatch.setattr(socket,'create_connection',lambda *a,**k: pytest.fail('must not connect'))
    with pytest.raises(RuntimeError, match='operator verification'):
        namespace['ensure_no_pending'](tmp_path)
    assert json.loads(path.read_text('utf-8')) == state


def test_external_diagnostic_allows_clean_history_without_erasing_records(tmp_path):
    import runpy
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/check-sem-simulator.py'))
    path = tmp_path/'old-run/control/control-state.json'
    path.parent.mkdir(parents=True)
    state = {'active_owner':None,'active_interrupt':None,'unresolved':[], 'acknowledgements':[{'reason':'previous check'}]}
    path.write_text(json.dumps(state),encoding='utf-8')
    namespace['ensure_no_pending'](tmp_path)
    assert json.loads(path.read_text('utf-8')) == state
