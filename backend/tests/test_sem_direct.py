"""Focused prototype regressions; every hardware/EDS dependency is a fake."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.cleanliness_core.sem_direct import build_plan, evaluate_shapes, fingerprint, run_task
from backend.cleanliness_core.preflight import run_preflight
from backend.cleanliness_core.atomic_json import write_json_atomic


def options(side=512):
    return {'roi': {'shape': 'square', 'x0': 0, 'y0': 0, 'x1': side, 'y1': side},
            'fieldUm': 512, 'imageSize': 512, 'overlap': .1, 'minLengthUm': 5}


def box(x=100, y=100, side=30):
    return {'points': [[x, y], [x+side, y], [x+side, y+side], [x, y+side]], 'confidence': .9}


@pytest.mark.parametrize('change', [
    {'fieldUm': float('nan')}, {'fieldUm': 0}, {'overlap': .5}, {'imageSize': 513},
    {'roi': {'shape': 'square', 'x0': -10, 'y0': 0, 'x1': 502, 'y1': 512}},
])
def test_bad_coordinates_and_parameters_rejected(change):
    with pytest.raises(ValueError):
        build_plan({**options(), **change}, [0, 0, 2000, 2000])


def test_overlap_particle_has_one_owner():
    plan = build_plan(options(900), [0, 0, 1000, 1000])
    accepted = []
    for tile in plan['tiles']:
        # One real particle at stage 440,440, seen in all four overlapping fields.
        x, y = (440-tile['x'])/plan['pixelXAxisSign']+256, 440-tile['y']+256
        records = evaluate_shapes([box(x-5, y-5, 10)], (512, 512), 1, [tile['x'], tile['y']], plan, tile['id'])
        accepted.extend(p for p in records if p['status'] == 'PENDING')
    assert len(accepted) == 1


class Microscope:
    def __init__(self, bad_coordinate=False):
        self.position = (256, 256)
        self.captures = 0
        self.bad_coordinate = bad_coordinate
        self.closed = False
    def connect(self): pass
    def get_position(self): return self.position
    def get_hv_status(self): return 'Open'
    def move_xy(self, x, y):
        self.position = (x, y)
        return {'after': {'x': x+10 if self.bad_coordinate else x, 'y': y}, 'positionErrorUm': 10 if self.bad_coordinate else 0}
    def capture(self, path):
        self.captures += 1
        path.parent.mkdir(exist_ok=True)
        image = np.zeros((512, 512), np.uint16)
        cv2.imwrite(str(path), image)
        return image, {'pixelSizeUm': 1}
    def restore_video(self): pass
    def gun_off(self): self.closed = True; return 'Close'


class Analyzer:
    def __init__(self): self.calls = []
    def analyze(self, particle, path):
        assert Path(path).is_file()
        assert cv2.pointPolygonTest(np.asarray(particle['contour'], np.float32), tuple(map(float, particle['safePoint'])), False) > 0
        self.calls.append(particle['sequence'])
        return {'elements': [{'symbol': 'Fe', 'value': 99.0}]}
    def close(self): pass


@pytest.fixture
def task(tmp_path, monkeypatch):
    from backend.cleanliness_core import sem_direct
    from backend.cleanliness_core.sem_direct_report import build_sem_reports
    monkeypatch.setattr(sem_direct, 'write_reports', lambda d,m: build_sem_reports(d,m,make_pdf=False))
    model = tmp_path/'model.onnx'
    model.write_bytes(b'fake-model-never-executed')
    (tmp_path/'navigation.png').write_bytes(b'fake-navigation')
    manifest = {'taskId': 'test', 'mode': 'sem_direct', 'status': 'DRAFT', 'input': {'workpieceName': 'Test'},
                'plan': build_plan(options(), [0, 0, 2000, 2000]), 'particles': [],
                'navigation': {'boundsUm': [0, 0, 2000, 2000], 'fieldUm': 512, 'magnification': 100, 'sha256': fingerprint(tmp_path/'navigation.png')},
                'request': {'microscopeHost': 'unused', 'microscopePort': 1, 'captureLevel': 1, 'edsEnabled': True, 'restoreStartPosition': True, 'gunOffAfterTask': True},
                'model': {'path': str(model), 'sha256': fingerprint(model)}}
    write_json_atomic(tmp_path/'task.json', manifest)
    return tmp_path, manifest


@pytest.mark.parametrize('shapes,expected', [
    ([box()], 1), ([box(), box(150)], 2), ([], 0),
    ([box(), box(110)], 0),  # competing particles: both rejected
    ([{'points': [[3, 4], [5, 6]]}], 0),  # insufficient contour
    ([box(-5)], 0), ([box(0)], 0),  # wrong coordinates / truncated edge
    ([{**box(), 'fragmented': True}], 0),
])
def test_segmentation_gate_controls_real_eds_boundary(task, shapes, expected):
    directory, _ = task
    scope, analyzer = Microscope(), Analyzer()
    result = run_task(directory, microscope=scope, analyzer=analyzer, segmenter=lambda *a: shapes)
    assert len(analyzer.calls) == expected
    assert result['status'] == 'COMPLETED'
    assert scope.closed
    assert result['cleanup']['gunOffVerified']
    assert (directory/'sem_report.html').is_file()
    assert (directory/'tiles/00001.segmentation.json').is_file()


def test_closed_gun_does_not_block_analysis_and_does_not_fire_gun(task):
    directory, _ = task
    scope, analyzer = Microscope(), Analyzer()
    scope.get_hv_status = lambda: 'Close'
    fired = []
    scope.gun_on = lambda: fired.append('on') or 'Open'
    result = run_task(directory, microscope=scope, analyzer=analyzer, segmenter=lambda *a: [box()])
    assert result['status'] == 'COMPLETED'
    assert scope.captures == 1
    assert analyzer.calls == [1]
    assert fired == []


def test_wrong_stage_coordinate_never_captures_or_acquires(task):
    directory, _ = task
    scope, analyzer = Microscope(True), Analyzer()
    with pytest.raises(RuntimeError, match='回读误差'):
        run_task(directory, microscope=scope, analyzer=analyzer, segmenter=lambda *a: [box()])
    assert not analyzer.calls and not scope.captures
    assert scope.closed
    assert json.loads((directory/'task.json').read_text('utf-8'))['status'] == 'FAILED'


def test_segmentation_failure_is_not_empty_success(task):
    directory, _ = task
    def broken(*args): raise RuntimeError('segmentation unavailable')
    analyzer = Analyzer()
    with pytest.raises(RuntimeError, match='segmentation unavailable'):
        run_task(directory, microscope=Microscope(), analyzer=analyzer, segmenter=broken)
    assert not analyzer.calls
    assert json.loads((directory/'task.json').read_text('utf-8'))['status'] == 'FAILED'


def test_partial_frame_resume_never_replays_unknown_eds(task):
    directory, manifest = task
    manifest['status'] = 'RUNNING'
    manifest['particles'] = [{'sequence': 1, 'tileId': 1, 'status': 'ACQUIRING'}]
    write_json_atomic(directory/'task.json', manifest)
    scope, analyzer = Microscope(), Analyzer()
    result = run_task(directory, microscope=scope, analyzer=analyzer, segmenter=lambda *a: [box()])
    assert not analyzer.calls and not scope.captures
    assert result['particles'][0]['status'] == 'REJECTED'
    assert result['status'] == 'COMPLETED_WITH_FAILURES'


def test_pause_resume_retains_completed_tiles(task):
    directory, manifest = task
    manifest['plan'] = build_plan(options(900), [0, 0, 2000, 2000])
    write_json_atomic(directory/'task.json', manifest)
    scope, analyzer = Microscope(), Analyzer()
    result = run_task(directory, microscope=scope, analyzer=analyzer, segmenter=lambda *a: [box()], should_stop=lambda: scope.captures == 1)
    assert result['status'] == 'STOPPED' and scope.captures == 1
    completed = copy.deepcopy(result['particles'])
    scope2, analyzer2 = Microscope(), Analyzer()
    result = run_task(directory, microscope=scope2, analyzer=analyzer2, segmenter=lambda *a: [box()])
    assert scope2.captures == 3
    assert result['particles'][:len(completed)] == completed
    assert set(analyzer.calls).isdisjoint(analyzer2.calls)


def test_sem_preflight_requires_no_optical_file_and_blocks_rotation(task):
    from backend.tests.test_preflight import Reader
    directory, manifest = task
    config = SimpleNamespace(output_dir=directory, microscope_host='unused', microscope_port=1, eds_enabled=True)
    for rotation, expected in [(0, True), (10, False)]:
        result = run_preflight(config, manifest, 'sem_analysis', reader=Reader(GetScanInfo={'fieldSizeUm': 512, 'rotationDeg': rotation}),
                               eds_check=lambda: {'status': 'READY'}, sleep=lambda *a: None)
        assert result['passed'] is expected
        assert next(i for i in result['items'] if i['id'] == 'input')['status'] == 'PASS'


def test_global_hardware_job_exclusion():
    from backend.server import JobManager
    manager = JobManager()
    manager.create('optical', 'analysis')
    with pytest.raises(ValueError): manager.create('sem', 'sem_analysis')
    with pytest.raises(ValueError): manager.create('sem', 'sem_navigation')


@pytest.mark.parametrize('missing_navigation', [False, True])
def test_navigation_prepares_transport_before_first_device_query(tmp_path, monkeypatch, missing_navigation):
    """Regression for actual GetStagePos timeout despite a device reply.

    Without the transport preparation the fresh navigation entry point failed;
    importing/using the old optical workflow first happened to hide the bug.
    """
    from backend import server
    from backend.cleanliness_core import microscope_discovery, optical_particle_task
    import importlib
    api_module = importlib.import_module('zemrpc.zem.zem_api')
    prepared = []
    device = SimpleNamespace(m=SimpleNamespace(mag=119, scan_info=SimpleNamespace(scan_um=1067.2268907563025, rot_degree=0)))
    def validate(api, address):
        assert prepared == ['timeouts', 'device']
        prepared.append('status')
        return 'Idle'
    def sync():
        assert prepared == ['timeouts', 'device', 'status', 'interface']
        prepared.append('sync')
    fake = SimpleNamespace(device=device, sync_stage_and_scan_field=sync,
                           get_nav_img=lambda: None if missing_navigation else SimpleNamespace(img=np.zeros((1080, 1920, 3), np.uint8), bounds=(-48241.5789047247, -25873.16054560485, 44478.67332441751, 26281.981333287647)))
    monkeypatch.setattr(api_module, 'zem_api', fake)
    monkeypatch.setattr(optical_particle_task, '_configure_zemrpc_timeouts', lambda: prepared.append('timeouts'))
    monkeypatch.setattr(optical_particle_task, '_configure_zemrpc_device', lambda d: prepared.append('device'))
    monkeypatch.setattr(optical_particle_task, '_cache_zemrpc_interface', lambda *a: prepared.append('interface'))
    monkeypatch.setattr(microscope_discovery, 'validate_zem_status', validate)
    monkeypatch.setattr(server, 'TASK_ROOT', tmp_path)
    monkeypatch.setattr(server, 'JOBS', server.JobManager())
    if missing_navigation:
        with pytest.raises(ValueError, match='没有可用导航图'):
            server.create_sem_navigation()
        assert not list(tmp_path.glob('*/task.json'))
        assert all(j['status'] == 'failed' for j in server.JOBS._jobs.values())
    else:
        manifest = server.create_sem_navigation()
        assert manifest['navigation']['imageShape'] == [1080, 1920]
        assert manifest['status'] == 'DRAFT' and manifest['particles'] == []
        assert (tmp_path/manifest['taskId']/'navigation.png').is_file()
        assert all(j['status'] == 'completed' for j in server.JOBS._jobs.values())


def test_failed_gun_preflight_never_enters_runner(task, monkeypatch):
    from backend import server
    from backend.cleanliness_core import preflight, sem_direct
    directory, manifest = task
    manifest['request'].update(preflightEnabled=True, edsHost='unused', edsPort=1)
    write_json_atomic(directory/'task.json', manifest)
    monkeypatch.setattr(server, 'task_manifest', lambda _: (directory/'task.json', manifest))
    monkeypatch.setattr(server, 'JOBS', server.JobManager())
    report = {'enabled': True, 'passed': False, 'items': [{'id': 'gun', 'label': '电子枪', 'status': 'FAIL', 'detail': '电子枪关闭'}]}
    monkeypatch.setattr(preflight, 'run_preflight', lambda *a, **k: report)
    def forbidden(*a, **k): pytest.fail('Rejected preflight must never move, capture or acquire EDS')
    monkeypatch.setattr(sem_direct, 'run_task', forbidden)
    job = server.JOBS.create('test', 'sem_analysis')
    server._run_job_impl(job['id'], 'test', 'sem_analysis')
    assert server.JOBS.get(job['id'])['status'] == 'failed'
    assert json.loads((directory/'task.json').read_text('utf-8'))['status'] == 'DRAFT'


@pytest.mark.parametrize('failure', [False, True])
def test_iteration_setting_never_switches_gun_off(task, failure):
    directory, manifest = task
    manifest['request']['gunOffAfterTask'] = False
    write_json_atomic(directory/'task.json', manifest)
    scope = Microscope()
    def infer(*args):
        if failure: raise RuntimeError('segmentation failure')
        return [box()]
    if failure:
        with pytest.raises(RuntimeError):
            run_task(directory, microscope=scope, analyzer=Analyzer(), segmenter=infer)
    else:
        run_task(directory, microscope=scope, analyzer=Analyzer(), segmenter=infer)
    assert not scope.closed
    saved = json.loads((directory/'task.json').read_text('utf-8'))
    assert saved['cleanup']['gunOffRequested'] is False


def test_formal_sem_report_preserves_source_and_excludes_optical(task):
    from backend.cleanliness_core.sem_direct_report import build_sem_reports
    directory, _ = task
    result = run_task(directory, microscope=Microscope(), analyzer=Analyzer(), segmenter=lambda *a: [box(), box(0)])
    result['particles'][0]['spectrum']['elements'] = [{'symbol':'C','value':90}, {'symbol':'Fe','value':8}, {'symbol':'Ti','value':2}]
    before = copy.deepcopy(result['particles'])
    build_sem_reports(directory, result, make_pdf=False)
    document = (directory/'sem_report.html').read_text('utf-8')
    assert '光镜' not in document and 'OPTICAL' not in document
    assert '80.000' in document and '20.000' in document
    # Original four pages plus acquisition/composition and classification detail.
    assert document.count('<section class="sheet">') == 6
    assert '视野边缘轮廓截断' in document
    assert (directory/'report_assets/p00001_context.jpg').is_file()
    assert (directory/'report_assets/p00001_detail.png').is_file()
    assert result['particles'] == before
    result['request']['showReportCoordinates'] = False
    build_sem_reports(directory, result, make_pdf=False)
    assert '颗粒估计台坐标' not in (directory/'sem_report.html').read_text('utf-8')


def test_report_context_colors_and_current_priority_preserve_source():
    from backend.cleanliness_core.sem_direct_report import (
        annotate_field, FIELD_CURRENT_COLOR, FIELD_SUCCESS_COLOR, FIELD_DETECTED_COLOR,
    )
    gray = np.full((256,256), 90, np.uint8)
    records = [{'sequence':i+1,'status':status,'contour':box(x,80)['points']}
               for i,(x,status) in enumerate([(30,'SUCCEEDED'),(110,'SUCCEEDED'),(190,'REJECTED')])]
    before = copy.deepcopy(records)
    image = annotate_field(gray, records, 1)
    assert tuple(image[95,30]) == FIELD_CURRENT_COLOR
    assert tuple(image[95,110]) == FIELD_SUCCESS_COLOR
    assert tuple(image[95,190]) == FIELD_DETECTED_COLOR
    assert np.all(gray == 90) and records == before
    overview = annotate_field(gray, records, None)
    assert tuple(overview[95,30]) == FIELD_SUCCESS_COLOR
    assert tuple(overview[95,190]) == FIELD_DETECTED_COLOR


def test_report_groups_details_by_capture_and_never_marks_other_frames(task):
    from backend.cleanliness_core.sem_direct_report import build_sem_reports, FIELD_SUCCESS_COLOR
    directory, _ = task
    manifest = run_task(directory, microscope=Microscope(), analyzer=Analyzer(), segmenter=lambda *a:[box()])
    other = copy.deepcopy(manifest['particles'][0])
    other.update(sequence=2,tileId=2,contour=box(350,350)['points'])
    manifest['particles'].insert(0, other)  # Deliberately interleaved input order.
    tile = copy.deepcopy(manifest['plan']['tiles'][0])
    tile['id'] = 2
    manifest['plan']['tiles'].append(tile)
    build_sem_reports(directory,manifest,make_pdf=False)
    document = (directory/'sem_report.html').read_text('utf-8')
    assert document.index('视野 01 · SEM 原视野') < document.index('视野 01 / 颗粒') < document.index('视野 02 · SEM 原视野') < document.index('视野 02 / 颗粒')
    image = cv2.imread(str(directory/'report_assets/p00001_context.jpg'))
    assert not np.any(np.all(image[340:390,340:390]==FIELD_SUCCESS_COLOR,axis=2))


def test_unvisited_large_plan_does_not_make_thousands_of_report_pages(task):
    from backend.cleanliness_core.sem_direct_report import build_sem_reports
    directory, manifest = task
    manifest['plan']['tiles'] = [{'id':i,'status':'PENDING'} for i in range(1,1025)]
    manifest['plan']['tileCount'] = 1024
    build_sem_reports(directory,manifest,make_pdf=False)
    document = (directory/'sem_report.html').read_text('utf-8')
    assert document.count('<section class="sheet">') == 1
    assert 'ROI:' in document


def test_capture_failure_timing_is_saved_without_eds_or_gun_off(task):
    directory, manifest = task
    manifest['request']['gunOffAfterTask'] = False
    write_json_atomic(directory/'task.json',manifest)
    scope, analyzer = Microscope(),Analyzer()
    def failed_capture(path): raise TimeoutError('capture timeout')
    scope.capture = failed_capture
    with pytest.raises(TimeoutError):
        run_task(directory,microscope=scope,analyzer=analyzer)
    saved=json.loads((directory/'task.json').read_text('utf-8'))
    event=next(e for e in saved['timingRuns'][-1]['events'] if e['stage']=='capture')
    assert event['outcome']=='FAILED' and event['seconds']>=0
    assert not analyzer.calls and not scope.closed
    assert (directory/'timing_report.html').is_file()


def test_timing_report_does_not_double_count_device_eds(task):
    from backend.cleanliness_core.sem_timing_report import write_timing_report
    directory, manifest = task
    manifest['timingRuns']=[{'wallSeconds':100,'stages':{'move':10,'capture':20,'eds':50},'events':[],'persistenceSeconds':3}]
    manifest['preflightTimings']=[{'seconds':2}]
    manifest['particles']=[{'sequence':1,'tileId':1,'status':'SUCCEEDED','timing':{'eds':50},'spectrum':{'elapsedSeconds':40}}]
    summary=write_timing_report(directory,manifest)
    assert summary['totalMeasuredSeconds']==102
    assert summary['stages']['other']==17
    assert sum(summary['stages'].values())==102
    assert summary['edsDeviceSeconds']==40


class RetryAnalyzer(Analyzer):
    def __init__(self, failures=1, terminal=True, on_failure=lambda:None):
        super().__init__()
        self.failures,self.terminal,self.on_failure=failures,terminal,on_failure
    def analyze(self,particle,path):
        from backend.cleanliness_core.eds.client_v2 import EdsV2Error
        result=super().analyze(particle,path)
        if len(self.calls)<=self.failures:
            self.on_failure()
            raise EdsV2Error('Extender-Local module has not been enabled.', terminal=self.terminal,
                             code=-2146233088,session_id='failed-session',job_id=f'failed-job-{len(self.calls)}')
        return result


def test_terminal_eds_failure_retries_same_particle_and_preserves_error(task,monkeypatch):
    from backend.cleanliness_core import sem_direct
    monkeypatch.setattr(sem_direct.time,'sleep',lambda _:None)
    directory,_=task
    analyzer=RetryAnalyzer()
    result=run_task(directory,microscope=Microscope(),analyzer=analyzer,segmenter=lambda *a:[box()])
    assert analyzer.calls==[1,1]
    particle=result['particles'][0]
    assert particle['status']=='SUCCEEDED'
    assert [a['status'] for a in particle['edsAttempts']]==['FAILED','SUCCEEDED']
    assert particle['edsAttempts'][0]['error']['jobId']=='failed-job-1'


@pytest.mark.parametrize('terminal,expected_calls',[(True,3),(False,1)])
def test_exhausted_or_unknown_eds_failure_stops_with_failed_status(task,monkeypatch,terminal,expected_calls):
    from backend.cleanliness_core import sem_direct
    from backend.cleanliness_core.eds.client_v2 import EdsV2Error
    monkeypatch.setattr(sem_direct.time,'sleep',lambda _:None)
    directory,_=task
    analyzer=RetryAnalyzer(failures=5,terminal=terminal)
    with pytest.raises(EdsV2Error):
        run_task(directory,microscope=Microscope(),analyzer=analyzer,segmenter=lambda *a:[box(),box(160)])
    assert len(analyzer.calls)==expected_calls and set(analyzer.calls)=={1}
    saved=json.loads((directory/'task.json').read_text(encoding='utf-8'))
    assert saved['particles'][0]['status']=='FAILED'
    assert saved['particles'][0]['failureStage']=='eds'
    assert saved['plan']['tiles'][0]['status']=='FAILED'


def test_stage_drift_between_retries_blocks_second_acquisition(task,monkeypatch):
    from backend.cleanliness_core import sem_direct
    monkeypatch.setattr(sem_direct.time,'sleep',lambda _:None)
    directory,_=task
    scope=Microscope()
    def drift(): scope.position=(260,256)
    analyzer=RetryAnalyzer(on_failure=drift)
    with pytest.raises(ValueError,match='样品台位置已改变'):
        run_task(directory,microscope=scope,analyzer=analyzer,segmenter=lambda *a:[box()])
    assert analyzer.calls==[1]


def test_recovery_requires_unique_contour_and_exact_safe_point():
    from backend.cleanliness_core.sem_direct import verify_recovery_frame
    image=np.random.default_rng(4).integers(0,255,(256,256),dtype=np.uint8)
    old={'sequence':1,'status':'FAILED','contour':box(80,80)['points'],'safePoint':[95,95]}
    fresh={'status':'PENDING','contour':old['contour']}
    assert verify_recovery_frame(image,image,[old],[fresh])[1]['contourIoU']==1
    for candidates in [[],[fresh,fresh],[{'status':'REJECTED','contour':old['contour']}],[{'status':'PENDING','contour':box(150,80)['points']}]]:
        with pytest.raises(ValueError): verify_recovery_frame(image,image,[old],candidates)
    with pytest.raises(ValueError): verify_recovery_frame(image,np.zeros_like(image),[old],[fresh])
    with pytest.raises(ValueError): verify_recovery_frame(image,image,[{**old,'safePoint':[180,180]}],[fresh])
    with pytest.raises(ValueError): verify_recovery_frame(image,image,[{**old,'contour':[[1,2]]}],[fresh])


def test_known_failed_frame_is_recaptured_and_only_failed_particle_retried(task):
    from backend.cleanliness_core.eds.client_v2 import EdsV2Error
    directory,manifest=task
    manifest['request']['edsRetryCount']=0
    write_json_atomic(directory/'task.json',manifest)
    class PatternScope(Microscope):
        def capture(self,path):
            path.parent.mkdir(exist_ok=True)
            image=np.random.default_rng(3).integers(0,60000,(512,512),dtype=np.uint16)
            cv2.imwrite(str(path),image)
            self.captures+=1
            return image,{'pixelSizeUm':1}
    class FailSecond(Analyzer):
        def analyze(self,p,path):
            result=super().analyze(p,path)
            if p['sequence']==2: raise EdsV2Error('Get EMSA failed',terminal=True)
            return result
    with pytest.raises(EdsV2Error):
        run_task(directory,microscope=PatternScope(),analyzer=FailSecond(),segmenter=lambda *a:[box(),box(200)])
    before=json.loads((directory/'task.json').read_text(encoding='utf-8'))
    analyzer=Analyzer()
    result=run_task(directory,microscope=PatternScope(),analyzer=analyzer,segmenter=lambda *a:[box(),box(200)])
    assert result['status']=='COMPLETED' and analyzer.calls==[2]
    assert result['particles'][0]==before['particles'][0]
    assert result['particles'][1]['recoveryEvidence'][0]['contourIoU']==1
    assert result['plan']['tiles'][0]['sha256']==before['plan']['tiles'][0]['sha256']


def test_zem20f_positive_stage_x_moves_feature_right_but_world_point_stays_fixed():
    plan=build_plan(options(),[0,0,1000,1000])
    before=evaluate_shapes([box(100,100)],(512,512),1,[256,256],plan,1)[0]
    after=evaluate_shapes([box(250,100)],(512,512),1,[406,256],plan,1)[0]
    assert before['stageUm']==after['stageUm']==[397,115]


def test_large_report_prints_bounded_parts_and_merges_all_pages(task,monkeypatch):
    from backend.cleanliness_core import sem_direct_report
    from pypdf import PdfReader,PdfWriter
    directory,_=task
    manifest=run_task(directory,microscope=Microscope(),analyzer=Analyzer(),segmenter=lambda *a:[box()])
    original=manifest['particles'][0]
    manifest['particles']=[{**copy.deepcopy(original),'sequence':i} for i in range(1,62)]
    calls=[]
    def print_stub(html_path,pdf_path):
        count=html_path.read_text(encoding='utf-8').count('<section class="sheet">')
        calls.append(count)
        writer=PdfWriter()
        for _ in range(count):writer.add_blank_page(width=842,height=595)
        with pdf_path.open('wb') as stream:writer.write(stream)
        writer.close()
        return pdf_path
    monkeypatch.setattr(sem_direct_report,'html_to_pdf_via_edge',print_stub)
    sem_direct_report.build_sem_reports(directory,manifest)
    # 61 particles now each have morphology + acquisition/composition +
    # classification pages. Cover and field overview remain two pages.
    assert calls==[40,40,40,40,25]
    assert len(PdfReader(directory/'sem_report.pdf').pages)==185


def test_recovery_redetects_never_acquired_candidates_without_replaying_success(task):
    directory,manifest=task
    class PatternScope(Microscope):
        def capture(self,path):
            path.parent.mkdir(exist_ok=True)
            image=np.random.default_rng(13).integers(0,60000,(512,512),dtype=np.uint16)
            cv2.imwrite(str(path),image)
            return image,{'pixelSizeUm':1}
    scope=PatternScope()
    result=run_task(directory,microscope=scope,analyzer=Analyzer(),segmenter=lambda *a:[box(),box(200)])
    result['status']='FAILED';result['plan']['tiles'][0]['status']='FAILED'
    pending=result['particles'][1]
    pending.pop('spectrum');pending.pop('edsAttempts');pending['status']='PENDING'
    write_json_atomic(directory/'task.json',result)
    analyzer=Analyzer()
    recovered=run_task(directory,microscope=PatternScope(),analyzer=analyzer,segmenter=lambda *a:[box(),box(201,101,32)])
    assert analyzer.calls==[3]
    assert recovered['particles'][0]['status']=='SUCCEEDED'
    assert recovered['particles'][1]['status']=='REJECTED'
    assert recovered['particles'][2]['sourceCapture'].endswith('.verify-1.png')
    assert recovered['particles'][2]['status']=='SUCCEEDED'
