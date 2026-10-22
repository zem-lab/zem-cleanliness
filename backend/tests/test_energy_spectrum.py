from pathlib import Path

import pytest

from backend.cleanliness_core import energy_spectrum
from backend.cleanliness_core.eds.client_v2 import EdsV2Error
from backend.cleanliness_core.eds.generated.v2 import job_events_pb2


def test_explicit_failed_event_preserves_job_identity_and_terminal_state(tmp_path,monkeypatch):
    event=job_events_pb2.JobEvent(failed=job_events_pb2.JobFailed(error_code=-2146233088,error_message='Extender-Local module has not been enabled.'))
    client=_FakeEdsClient([event])
    with pytest.raises(EdsV2Error) as caught:
        _run(tmp_path,client,monkeypatch)
    assert caught.value.terminal is True
    assert caught.value.job_id=='job-1' and caught.value.session_id=='session-1'
    assert caught.value.code==-2146233088 and client.closed_session


class _FakeEdsClient:
    target = "fake:6200"

    def __init__(self, events):
        self.events = events
        self.started = False
        self.closed_session = False

    def open_session(self):
        return "session-1"

    def start_spectrum_job(self, session_id):
        assert session_id == "session-1"
        self.started = True
        return "job-1"

    def watch_job(self, session_id, job_id):
        assert (session_id, job_id) == ("session-1", "job-1")
        yield from self.events

    def close_session(self, session_id):
        assert session_id == "session-1"
        self.closed_session = True


def _run(tmp_path: Path, client: _FakeEdsClient, monkeypatch):
    monkeypatch.setattr(energy_spectrum, "start_scan_points", lambda *args, **kwargs: None)
    sem_path = tmp_path / "sem" / "0001.png"
    return energy_spectrum.process_energy_spectrum_particle(
        client,
        {"label": "1", "points": [[0, 0], [2, 0], [2, 2]]},
        str(sem_path),
        "microscope",
        5050,
        target_point=(10, 20),
    )


def test_process_particle_saves_spectrum_and_hv_classification(tmp_path: Path, monkeypatch):
    png = b"\x89PNG\r\n\x1a\nreport-spectrum"
    snapshot = job_events_pb2.JobEvent(
        snapshot=job_events_pb2.SpectrumSnapshot(
            spectrum_image=job_events_pb2.SpectrumImage(
                png=png,
                media_type="image/png",
                width=800,
                height=450,
            )
        )
    )
    completed = job_events_pb2.JobEvent(
        completed=job_events_pb2.JobCompleted(
            result_text="Fe=88",
            quantify_result_map={"Fe": 88.0, "Cr": 12.0},
            material_classification=job_events_pb2.MaterialClassification(
                material_class="Low Alloy Steel",
                parent_class="Ferrous Metals(HV<400)",
                hardness_label="HV<400",
                hardness_band="Below400",
                confidence=1.0,
            ),
            elapsed_seconds=10.5,
            accumulated_counts=382438,
            spectrum_image=job_events_pb2.SpectrumImage(png=png,media_type='image/png',width=800,height=450),
        )
    )
    client = _FakeEdsClient([snapshot, completed])

    result = _run(tmp_path, client, monkeypatch)

    assert client.started is True
    assert client.closed_session is True
    assert result["elements"] == [
        {"symbol": "Fe", "value": 88.0},
        {"symbol": "Cr", "value": 12.0},
    ]
    assert result["materialClassification"]["hardnessLabel"] == "HV<400"
    assert result["spectrumImage"]["path"] == "spectra/0001.png"
    assert (tmp_path / "spectra" / "0001.png").read_bytes() == png


def test_accepts_early_completed_notification_without_waiting_ten_seconds(
    tmp_path: Path, monkeypatch
):
    client = _FakeEdsClient(
        [
            job_events_pb2.JobEvent(
                completed=job_events_pb2.JobCompleted(
                    result_text="Al=91",
                    quantify_result_map={"Al": 91.2, "O": 8.8},
                    elapsed_seconds=1.2,
                    accumulated_counts=42000,
                )
            )
        ]
    )

    result = _run(tmp_path, client, monkeypatch)

    assert result["elapsedSeconds"] == pytest.approx(1.2)
    assert result["elements"] == [
        {"symbol": "Al", "value": 91.2},
        {"symbol": "O", "value": 8.8},
    ]
    assert client.closed_session is True


def test_snapshot_preview_quantify_is_not_final_result(tmp_path: Path, monkeypatch):
    client = _FakeEdsClient(
        [
            job_events_pb2.JobEvent(
                snapshot=job_events_pb2.SpectrumSnapshot(
                    quantify_result_map={"Si": 100.0},
                    quantification_text="preview Si",
                    elapsed_seconds=0.6,
                )
            ),
            job_events_pb2.JobEvent(
                completed=job_events_pb2.JobCompleted(
                    result_text="Fe=88",
                    quantify_result_map={"Fe": 88.0, "Cr": 12.0},
                    elapsed_seconds=2.4,
                    accumulated_counts=90000,
                )
            ),
        ]
    )

    result = _run(tmp_path, client, monkeypatch)

    assert result["elements"] == [
        {"symbol": "Fe", "value": 88.0},
        {"symbol": "Cr", "value": 12.0},
    ]
    assert all(item["symbol"] != "Si" for item in result["elements"])


def test_failed_job_does_not_return_elements(tmp_path: Path, monkeypatch):
    client = _FakeEdsClient(
        [
            job_events_pb2.JobEvent(
                failed=job_events_pb2.JobFailed(
                    error_code=7,
                    error_message="detector timeout",
                )
            )
        ]
    )

    with pytest.raises(EdsV2Error, match="detector timeout"):
        _run(tmp_path, client, monkeypatch)
    assert client.closed_session is True


def test_cancelled_job_does_not_return_elements(tmp_path: Path, monkeypatch):
    client = _FakeEdsClient(
        [
            job_events_pb2.JobEvent(
                cancelled=job_events_pb2.JobCancelled(reason="operator stop")
            )
        ]
    )

    with pytest.raises(EdsV2Error, match="operator stop"):
        _run(tmp_path, client, monkeypatch)
    assert client.closed_session is True


def test_stream_without_completed_does_not_pass_empty_elements(tmp_path: Path, monkeypatch):
    client = _FakeEdsClient(
        [
            job_events_pb2.JobEvent(
                snapshot=job_events_pb2.SpectrumSnapshot(
                    quantify_result_map={"C": 55.0},
                    elapsed_seconds=0.4,
                )
            )
        ]
    )

    with pytest.raises(EdsV2Error, match="without a completed result"):
        _run(tmp_path, client, monkeypatch)
    assert client.closed_session is True


def test_center_region_eds_uses_explicit_point_without_invented_contour(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    from backend.cleanliness_core.cleanliness_task import RealSpectrumAnalyzer
    path=tmp_path/'sem.png';Image.new('L',(512,512),180).save(path)
    client=_FakeEdsClient([job_events_pb2.JobEvent(completed=job_events_pb2.JobCompleted(quantify_result_map={'Fe':98.0}))])
    moves=[]
    monkeypatch.setattr(energy_spectrum,'start_scan_points',lambda host,port,points:moves.append(points))
    def forbidden(*a,**k):raise AssertionError('A contour center must not be invented for center-region EDS')
    monkeypatch.setattr(energy_spectrum,'particle_center_pixel',forbidden)
    analyzer=RealSpectrumAnalyzer.__new__(RealSpectrumAnalyzer)
    analyzer.client=client;analyzer.microscope_host='fake';analyzer.microscope_port=61557
    result=analyzer.analyze(SimpleNamespace(sequence=3),None,path,(256,256))
    assert moves==[[(256,256)]]
    assert result['target']=={'mode':'large_particle_center_region','x':256,'y':256}
    assert client.started and client.closed_session


@pytest.mark.parametrize('point',[None,(-1,256),(512,256),(256,512),(256.5,256)])
def test_invalid_center_point_stops_before_beam_or_eds(tmp_path,monkeypatch,point):
    from PIL import Image
    path=tmp_path/'sem.png';Image.new('L',(512,512),180).save(path)
    client=_FakeEdsClient([]);moves=[]
    monkeypatch.setattr(energy_spectrum,'start_scan_points',lambda *a:moves.append(a))
    with pytest.raises(ValueError):
        energy_spectrum.process_energy_spectrum_particle(client,{'label':'3','points':[]},str(path),'fake',61557,
            target_point=point,target_mode='large_particle_center_region')
    assert not moves and not client.started
    assert client.closed_session is False
