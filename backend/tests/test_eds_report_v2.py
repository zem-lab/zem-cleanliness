import copy
import csv
import json
from pathlib import Path

from google.protobuf.json_format import ParseDict

from backend.cleanliness_core import energy_spectrum, eds_report_details
from backend.cleanliness_core.eds.generated.v2 import job_events_pb2 as pb
from backend.tests.test_energy_spectrum import _FakeEdsClient, _run
from backend.tests.test_cleanliness_report import _minimal_manifest, _particle_with_elements
from backend.tests.test_sem_direct import task, Microscope, Analyzer, box


def sample():
    data=json.loads((Path(__file__).parent/'fixtures/eds_v2_20260929/completed.json').read_text(encoding='utf-8'))
    return ParseDict(data,pb.JobCompleted())


def test_new_final_result_preserves_raw_sigma_classifications_and_clock(tmp_path,monkeypatch):
    final=sample()
    final.element_quantifications[0].ClearField('weight_percent_sigma')
    final.element_quantifications[1].weight_percent_sigma=0
    events=[pb.JobEvent(snapshot=pb.SpectrumSnapshot(quantify_result_map={'Si':100},spectrum_image=pb.SpectrumImage(png=b'preview'))),
            pb.JobEvent(completed=final)]
    result=_run(tmp_path,_FakeEdsClient(events),monkeypatch)
    assert result['materialClassification']['materialClass']=='Si/Si-C/Si-N'
    assert result['vdaClassification']['materialClass']=='brass'
    assert len(result['materialClassification']['warnings'])==16
    assert result['materialClassification']['boundaryNotes']
    assert result['vdaAnalysisComposition']=={'Cu':68.73,'Zn':31.27,'O':0}
    assert 'C' not in result['vdaAnalysisComposition']
    assert 'sigma' not in result['elementQuantifications'][0]
    assert result['elementQuantifications'][1]['sigma']==0
    assert result['realTimeSeconds']==2.405 and result['liveTimeSeconds']==2.143
    assert result['elapsedSeconds']==2.85 and result['clientCallSeconds']>=0
    assert result['targetReached'] is True and result['targetCounts']==50000
    assert result['quantMethod']=='default' and result['startedAt']=='2026-09-29 14:34:04'
    assert 'classificationReport' in result
    assert 'spectrumImage' not in result  # Preview must never become a final image.
    assert all(e['symbol']!='Si' for e in result['elements'])


def test_maximum_duration_completed_retains_result_without_claiming_target_or_retry(tmp_path,monkeypatch):
    final=sample(); final.stop_reason=pb.SPECTRUM_STOP_REASON_MAXIMUM_DURATION; final.accumulated_counts=40000
    client=_FakeEdsClient([pb.JobEvent(completed=final)])
    result=_run(tmp_path,client,monkeypatch)
    assert result['targetReached'] is False and result['elements']
    assert '计数未达标' in eds_report_details.stop_label(result)
    assert client.closed_session


def test_legacy_fields_stay_missing_and_html_escapes_server_strings(tmp_path,monkeypatch):
    final=pb.JobCompleted(quantify_result_map={'Fe':99.12345})
    result=_run(tmp_path,_FakeEdsClient([pb.JobEvent(completed=final)]),monkeypatch)
    assert result['elements'][0]['value']==99.12345
    assert 'realTimeSeconds' not in result and 'targetCounts' not in result and 'vdaClassification' not in result
    result['materialClassification']={'warnings':['<script>alert(1)</script>']}
    body=''.join(body for _,body in eds_report_details.detail_sections(result))
    assert '未提供' in body and '<script>' not in body and '&lt;script&gt;' in body


def test_optical_html_csv_use_service_compositions_and_do_not_change_source(tmp_path,monkeypatch):
    from backend.cleanliness_core.cleanliness_report import write_cleanliness_html,write_cleanliness_csv
    result=_run(tmp_path,_FakeEdsClient([pb.JobEvent(completed=sample())]),monkeypatch)
    record=_particle_with_elements([]); record['energySpectrum']=result
    manifest=_minimal_manifest([record]); before=copy.deepcopy(manifest)
    html_path=write_cleanliness_html(tmp_path,manifest)
    document=html_path.read_text(encoding='utf-8')
    for text in ['Oxford','VDA','brass','68.73','31.27','标准差','实际时间','活时间','临界元素','catch-all']:
        assert text in document
    assert manifest==before
    path=write_cleanliness_csv(tmp_path,manifest)
    row=next(csv.DictReader(path.open(encoding='utf-8-sig')))
    assert json.loads(row['edsVdaComposition'])=={'Cu':68.73,'Zn':31.27,'O':0}
    assert json.loads(row['edsOxfordClassification'])['materialClass']=='Si/Si-C/Si-N'
    assert row['edsClassificationReport']==result['classificationReport']
    assert row['edsTargetCounts']=='50000'


def test_time_report_keeps_nested_time_metrics_separate():
    from backend.cleanliness_core.eds_report_details import csv_values
    result={'elapsedSeconds':2.85,'realTimeSeconds':2.405,'liveTimeSeconds':2.143,'clientCallSeconds':3.3}
    row=csv_values(result)
    assert row['edsRealTimeSeconds']==2.405 and row['edsServerSeconds']==2.85


def test_hide_only_oxford_missing_sigma_warnings_without_mutating_raw_data():
    warning='Wt% Sigma is unavailable for Z=26; UseError uses the nominal weight percent.'
    s={'materialClassification':{'warnings':[warning,warning[:-1], 'Matched a catch-all class with no child criteria.',
        'Detector calibration failed.']},'vdaClassification':{'warnings':[warning]}}
    before=copy.deepcopy(s)
    body=''.join(body for _,body in eds_report_details.detail_sections(s))
    assert 'Oxford 警告：Wt%' not in body
    assert 'VDA 警告：Wt%' in body
    assert 'catch-all' in body and 'Detector calibration failed.' in body
    assert s==before
    assert warning in eds_report_details.csv_values(s)['edsOxfordClassification']


def test_sem_report_csv_and_timing_keep_three_compositions_and_unmet_target(task,monkeypatch):
    from backend.cleanliness_core.sem_direct import run_task
    from backend.cleanliness_core.sem_timing_report import write_timing_report
    directory,_=task
    final=sample(); final.stop_reason=pb.SPECTRUM_STOP_REASON_MAXIMUM_DURATION; final.accumulated_counts=40000
    spectrum=_run(directory,_FakeEdsClient([pb.JobEvent(completed=final)]),monkeypatch)
    class SampleAnalyzer(Analyzer):
        def analyze(self,*args): return copy.deepcopy(spectrum)
    result=run_task(directory,microscope=Microscope(),analyzer=SampleAnalyzer(),segmenter=lambda *a:[box()])
    doc=(directory/'sem_report.html').read_text(encoding='utf-8')
    assert '计数未达标' in doc and 'VDA wt%' in doc and 'Oxford wt%' in doc
    assert 'catch-all' in doc and '临界元素' in doc
    row=next(csv.DictReader((directory/'sem_results.csv').open(encoding='utf-8-sig')))
    assert json.loads(row['edsVdaComposition'])==spectrum['vdaAnalysisComposition']
    assert row['edsTargetReached']=='False'
    timing=write_timing_report(directory,result)
    assert timing['belowTargetCount']==1
    assert timing['acquisitionTimes']['realTimeSeconds']=={'reportedCount':1,'seconds':2.405}
    assert timing['acquisitionTimes']['liveTimeSeconds']=={'reportedCount':1,'seconds':2.143}
    assert timing['edsReportedSeconds']==2.85
