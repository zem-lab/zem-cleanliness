from types import SimpleNamespace
import grpc
import pytest
from backend.cleanliness_core.eds.client_v2 import EdsV2Client,EdsV2Error
from backend.cleanliness_core.eds.generated.v2 import job_events_pb2


class Drop(grpc.RpcError):
    def __init__(self,code):self.status=code
    def code(self):return self.status


def test_stream_reconnect_resumes_same_job_without_duplicate_events(monkeypatch):
    client=EdsV2Client(timeout=1)
    calls=[]
    event1=job_events_pb2.JobEvent(seq=1,progress=job_events_pb2.JobProgress(percent=30))
    done=job_events_pb2.JobEvent(seq=2,completed=job_events_pb2.JobCompleted(result_text='Fe=100'))
    def watch(request,timeout):
        calls.append((request.session_id,request.job_id,request.last_seq))
        if len(calls)==1:
            yield event1
            raise Drop(grpc.StatusCode.UNAVAILABLE)
        yield event1  # server replay is deduplicated
        yield done
    def channel():client._job_stub=SimpleNamespace(WatchJob=watch)
    monkeypatch.setattr(client,'_ensure_channel',channel)
    monkeypatch.setattr(client,'close',lambda:None)
    monkeypatch.setattr('backend.cleanliness_core.eds.client_v2.time.sleep',lambda _:None)
    events=list(client.watch_job('session-a','job-a'))
    assert [e.seq for e in events]==[1,2]
    assert calls==[('session-a','job-a',0),('session-a','job-a',1)]


@pytest.mark.parametrize('code,count',[(grpc.StatusCode.UNAVAILABLE,3),(grpc.StatusCode.PERMISSION_DENIED,1)])
def test_stream_retry_is_bounded_and_keeps_unknown_job_identity(monkeypatch,code,count):
    client=EdsV2Client(timeout=1)
    calls=[]
    def watch(request,timeout):
        calls.append(request.job_id)
        raise Drop(code)
    def channel():client._job_stub=SimpleNamespace(WatchJob=watch)
    monkeypatch.setattr(client,'_ensure_channel',channel)
    monkeypatch.setattr(client,'close',lambda:None)
    monkeypatch.setattr('backend.cleanliness_core.eds.client_v2.time.sleep',lambda _:None)
    with pytest.raises(EdsV2Error) as caught:list(client.watch_job('s','j'))
    assert calls==['j']*count and caught.value.terminal is False
    assert caught.value.session_id=='s' and caught.value.job_id=='j'
