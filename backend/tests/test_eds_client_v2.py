from backend.cleanliness_core.eds.client_v2 import (
    EdsV2Client,
    EdsV2Error,
    build_spectrum_acquisition_request,
)
from backend.cleanliness_core.eds.generated.v2 import jobs_pb2


def test_spectrum_request_uses_automatic_preset_not_fixed_realtime():
    req = build_spectrum_acquisition_request("session-1", max_duration_seconds=60.0)

    assert req.WhichOneof("stop_condition") == "preset"
    assert req.preset == jobs_pb2.SPECTRUM_ACQUISITION_PRESET_AUTOMATIC
    assert req.HasField("preset")
    assert not req.HasField("real_time_seconds")
    assert not req.HasField("live_time_seconds")
    assert not req.HasField("target_counts")
    assert req.max_duration_seconds == 60.0

    parsed = jobs_pb2.StartSpectrumAcquisitionJobRequest.FromString(req.SerializeToString())
    assert parsed.WhichOneof("stop_condition") == "preset"
    assert parsed.preset == jobs_pb2.SPECTRUM_ACQUISITION_PRESET_AUTOMATIC
    assert not parsed.HasField("real_time_seconds")


def test_start_spectrum_job_sends_automatic_preset_and_timeout_cap():
    captured = {}

    class _Stub:
        def StartSpectrumAcquisitionJob(self, req, timeout):
            captured["req"] = req
            captured["timeout"] = timeout
            return jobs_pb2.StartSpectrumAcquisitionJobResponse(
                success=True,
                job_id="job-auto",
            )

    client = EdsV2Client("192.168.0.15", 6200, timeout=60.0)
    client._channel = object()
    client._job_stub = _Stub()

    assert client.start_spectrum_job("session-9") == "job-auto"
    req = captured["req"]
    assert captured["timeout"] == 60.0
    assert req.session_id == "session-9"
    assert req.WhichOneof("stop_condition") == "preset"
    assert req.preset == jobs_pb2.SPECTRUM_ACQUISITION_PRESET_AUTOMATIC
    assert not req.HasField("real_time_seconds")
    assert req.max_duration_seconds == 60.0
    assert req.image.include_png is True


def test_start_spectrum_job_rejects_unsuccessful_response():
    class _Stub:
        def StartSpectrumAcquisitionJob(self, req, timeout):
            return jobs_pb2.StartSpectrumAcquisitionJobResponse(
                success=False,
                message="detector offline",
            )

    client = EdsV2Client("127.0.0.1", 6200, timeout=30.0)
    client._channel = object()
    client._job_stub = _Stub()

    try:
        client.start_spectrum_job("session-err")
    except EdsV2Error as exc:
        assert "detector offline" in str(exc)
    else:
        raise AssertionError("expected EdsV2Error")
