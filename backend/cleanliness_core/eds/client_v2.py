from __future__ import annotations

import logging
import time
from typing import Iterator

import grpc

from backend.cleanliness_core.eds.generated.v2 import (
    jobs_pb2,
    jobs_pb2_grpc,
    job_events_pb2,
    session_pb2,
    session_pb2_grpc,
)

logger = logging.getLogger(__name__)


def validate_target_counts(value: int | None) -> int | None:
    """None preserves automatic acquisition; explicit counts must be exact."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2**53 - 1:
        raise ValueError("EDS 目标计数必须是正整数，或为空以使用自动模式")
    return value


class EdsV2Error(RuntimeError):
    """EDS V2 gRPC 操作异常。"""

    def __init__(self, message, *, terminal=False, code=None, session_id=None, job_id=None):
        super().__init__(message)
        self.terminal = bool(terminal)
        self.code = code
        self.session_id = session_id
        self.job_id = job_id

    def to_dict(self):
        return {'message':str(self),'terminal':self.terminal,'code':self.code,
                'sessionId':self.session_id,'jobId':self.job_id}


def build_spectrum_acquisition_request(
    session_id: str,
    *,
    progress_interval_ms: int = 500,
    image_width: int = 800,
    image_height: int = 450,
    quant_method: str = "",
    max_duration_seconds: float | None = None,
    target_counts: int | None = None,
) -> jobs_pb2.StartSpectrumAcquisitionJobRequest:
    """Build a spectrum job that stops when EDS decides the spectrum is enough.

    ``SPECTRUM_ACQUISITION_PRESET_AUTOMATIC`` is enum value 0. Assign it onto
    the oneof after construction so proto3 still serializes the stop condition.
    """

    target_counts = validate_target_counts(target_counts)
    req = jobs_pb2.StartSpectrumAcquisitionJobRequest(
        session_id=session_id,
        progress_interval_ms=max(100, int(progress_interval_ms)),
        quant_method=quant_method,
        image=jobs_pb2.SpectrumImageOptions(
            include_png=True,
            width=max(1, int(image_width)),
            height=max(1, int(image_height)),
        ),
    )
    if target_counts is None:
        req.preset = jobs_pb2.SPECTRUM_ACQUISITION_PRESET_AUTOMATIC
    else:
        req.target_counts = target_counts
    if max_duration_seconds is not None and float(max_duration_seconds) > 0:
        req.max_duration_seconds = float(max_duration_seconds)
    return req


class EdsV2Client:
    """EDS V2 gRPC 同步客户端。

    每个颗粒独立执行 OpenSession -> 分析 -> CloseSession 流程。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 6200, timeout: float = 30.0,
                 *, target_counts: int | None = None):
        self.host = host or "127.0.0.1"
        self.port = int(port)
        self.timeout = float(timeout)
        self.target_counts = validate_target_counts(target_counts)
        self._channel: grpc.Channel | None = None
        self._session_stub: session_pb2_grpc.SessionServiceStub | None = None
        self._job_stub: jobs_pb2_grpc.JobServiceStub | None = None

    @property
    def target(self) -> str:
        return f"{self.host}:{self.port}"

    def _ensure_channel(self):
        if self._channel is None:
            self._channel = grpc.insecure_channel(self.target)
            self._session_stub = session_pb2_grpc.SessionServiceStub(self._channel)
            self._job_stub = jobs_pb2_grpc.JobServiceStub(self._channel)

    def close(self):
        if self._channel is not None:
            self._channel.close()
        self._channel = None
        self._session_stub = None
        self._job_stub = None

    def open_session(self, client_name: str = "ZEM Cleanliness", client_version: str = "1.0") -> str:
        """打开会话，返回 session_id。"""
        self._ensure_channel()
        assert self._session_stub is not None
        req = session_pb2.OpenSessionRequest(client_name=client_name, client_version=client_version)
        try:
            resp = self._session_stub.OpenSession(req, timeout=self.timeout)
        except grpc.RpcError as exc:
            raise EdsV2Error(f"OpenSession 失败: {exc}") from exc
        if not resp.success:
            raise EdsV2Error(f"OpenSession 失败: {resp.message}")
        logger.info(f"EDS V2 OpenSession 成功, session_id={resp.session_id}")
        return resp.session_id

    def close_session(self, session_id: str) -> None:
        """关闭指定会话。"""
        self._ensure_channel()
        assert self._session_stub is not None
        req = session_pb2.CloseSessionRequest(session_id=session_id)
        try:
            resp = self._session_stub.CloseSession(req, timeout=self.timeout)
        except grpc.RpcError as exc:
            logger.warning(f"CloseSession 异常: {exc}")
            return
        if not resp.success:
            logger.warning(f"CloseSession 返回失败: {resp.message}")

    def start_point_list_job(
        self,
        session_id: str,
        mode: str = "",
    ) -> str:
        """启动 PointListMeasurement 任务，返回 job_id。

        束位需提前通过电镜 StartScanPoints 设置好。
        """
        self._ensure_channel()
        assert self._job_stub is not None
        req = jobs_pb2.StartPointListMeasurementJobRequest(
            session_id=session_id,
            mode=mode,
        )
        try:
            resp = self._job_stub.StartPointListMeasurementJob(req, timeout=self.timeout)
        except grpc.RpcError as exc:
            raise EdsV2Error(f"StartPointListMeasurementJob 失败: {exc}") from exc
        if not resp.success:
            raise EdsV2Error(f"StartPointListMeasurementJob 失败: {resp.message}")
        logger.info(f"EDS V2 StartPointListMeasurementJob 成功, job_id={resp.job_id}")
        return resp.job_id

    def start_spectrum_job(
        self,
        session_id: str,
        *,
        progress_interval_ms: int = 500,
        image_width: int = 800,
        image_height: int = 450,
        quant_method: str = "",
    ) -> str:
        """按本任务的指定计数或 AUTOMATIC 模式采谱，结果经 WatchJob 推送。"""
        self._ensure_channel()
        assert self._job_stub is not None
        req = build_spectrum_acquisition_request(
            session_id,
            progress_interval_ms=progress_interval_ms,
            image_width=image_width,
            image_height=image_height,
            quant_method=quant_method,
            max_duration_seconds=self.timeout,
            target_counts=self.target_counts,
        )
        try:
            resp = self._job_stub.StartSpectrumAcquisitionJob(req, timeout=self.timeout)
        except grpc.RpcError as exc:
            raise EdsV2Error(f"StartSpectrumAcquisitionJob 失败: {exc}") from exc
        if not resp.success:
            raise EdsV2Error(f"StartSpectrumAcquisitionJob 失败: {resp.message}")
        logger.info(
            "EDS V2 StartSpectrumAcquisitionJob 成功, job_id=%s, stop_condition=%s, target_counts=%s, max_duration=%.1fs",
            resp.job_id,
            req.WhichOneof("stop_condition"),
            self.target_counts,
            self.timeout,
        )
        return resp.job_id

    def watch_job(self, session_id: str, job_id: str) -> Iterator[job_events_pb2.JobEvent]:
        """流式接收 Job 事件，直到任务结束。"""
        self._ensure_channel()
        assert self._job_stub is not None
        last_seq=0
        for attempt in range(3):
            self._ensure_channel()
            req = jobs_pb2.WatchJobRequest(session_id=session_id, job_id=job_id, last_seq=last_seq)
            try:
                stream = self._job_stub.WatchJob(req, timeout=self.timeout)
                for event in stream:
                    if event.seq and event.seq<=last_seq:
                        continue
                    last_seq=max(last_seq,int(event.seq))
                    yield event
                    if event.WhichOneof('payload') in {'completed','failed','cancelled'}:
                        return
                # A cleanly ended stream without a terminal event is incomplete.
                raise EdsV2Error('WatchJob ended without a terminal event',session_id=session_id,job_id=job_id)
            except grpc.RpcError as exc:
                if attempt>=2 or exc.code() not in {grpc.StatusCode.UNAVAILABLE,grpc.StatusCode.DEADLINE_EXCEEDED,grpc.StatusCode.CANCELLED}:
                    raise EdsV2Error(f"WatchJob 异常: {exc}",session_id=session_id,job_id=job_id) from exc
                logger.warning('WatchJob connection interrupted; reconnecting SAME job=%s after seq=%s (attempt %s/2)',job_id,last_seq,attempt+1)
                self.close()
                time.sleep(.5)
