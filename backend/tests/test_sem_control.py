from __future__ import annotations

import base64
from functools import wraps
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.cleanliness_core.device_control import BackendOwnerLock, DeviceBusy, DeviceCoordinator
from backend.sem_control import RealMicroscope, SemControl, SemError, SimulatedMicroscope
from backend.cleanliness_core.optical_particle_task import _configure_zemrpc_device, zemrpc_single_attempt


@pytest.fixture
def control(tmp_path):
    instance = SemControl(tmp_path, SimulatedMicroscope(), DeviceCoordinator(), allow_capture=True,
                          bounds=[-100, -100, 100, 100], max_move_um=100, poll_interval=.001, move_timeout=.03)
    yield instance
    instance.close()


def wait(control, job, caller="test"):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        record = control.call("sem_get_job", {"job_id": job["job_id"]}, caller)
        if record["status"] in control.TERMINAL:
            return record
        time.sleep(.005)
    raise AssertionError("模拟作业未结束")


def action(control, operation="sem_move_and_capture", caller="test", **kwargs):
    arguments = {"request_id": kwargs.pop("request_id", "request-1"), **kwargs}
    if operation in control.MOVES:
        arguments = {"x_um": 10., "y_um": 20., **arguments}
    return control.call(operation, arguments, caller)


def test_discovery_has_no_hardware_side_effect(control):
    result = control.call("sem_get_capabilities", {})
    assert result["mode"] == "simulated"
    assert result["position_tolerance_um"] == 2
    assert "EDS" in result["unsupported"]
    assert not control.device.commands


def test_readonly_queries_do_not_move_capture_or_prepare_imaging(control):
    assert control.call("sem_get_position", {})["position_um"] == (0, 0)
    assert control.call("sem_get_status", {})["gun_state"] == 1
    assert control.device.commands == ["position", "status"]


def test_move_capture_has_confirmed_coordinates_and_lossless_source(control):
    job = wait(control, action(control))
    assert job["status"] == "completed", job
    assert job["result"]["movement"]["position_error_um"] == 0
    image = job["result"]["image"]
    assert image["position_um"] == [10, 20]
    assert image["source"] == "simulated" and image["dtype"] == "uint16"
    assert control.device.commands.count("move") == 1
    assert control.device.commands.count("capture") == 1
    for variant, dtype in [("original", np.uint16), ("preview", np.uint8)]:
        result = control.call("sem_get_image", {"image_id": image["image_id"], "variant": variant}, "test")
        data = np.frombuffer(base64.b64decode(result["image"]["data"]), np.uint8)
        assert cv2.imdecode(data, cv2.IMREAD_UNCHANGED).dtype == dtype


def test_same_request_is_not_executed_twice_even_after_bridge_reconnect(control):
    first = wait(control, action(control))
    second = action(control)
    assert first["job_id"] == second["job_id"]
    assert control.device.commands.count("move") == control.device.commands.count("capture") == 1
    with pytest.raises(SemError, match="不同参数"):
        action(control, x_um=11)


@pytest.mark.parametrize("arguments", [
    {"x_um": float("nan")}, {"x_um": float("inf")}, {"x_um": True},
    {"size": 513}, {"pixel_avg": 8}, {"pixel_avg": True},
    {"request_id": "../unsafe"}, {"unit": "mm"}, {"extra": 1},
])
def test_invalid_parameters_never_reach_device(control, arguments):
    with pytest.raises(SemError):
        action(control, **arguments)
    assert not control.device.commands


def test_range_and_distance_are_both_checked(control):
    with pytest.raises(SemError) as error:
        action(control, x_um=101)
    assert error.value.code == "OUT_OF_RANGE"
    result = wait(control, action(control, x_um=90, y_um=90))
    assert result["error"]["code"] == "OUT_OF_RANGE"
    assert "move" not in control.device.commands


def test_current_position_outside_safe_region_blocks_movement(control):
    control.device.position = (200., 0.)
    result = wait(control, action(control, "sem_move_xy", x_um=100, y_um=0))
    assert result["error"]["code"] == "OUT_OF_RANGE" and "move" not in control.device.commands


def test_wrong_image_resolution_is_preserved_but_not_accepted(control, monkeypatch):
    monkeypatch.setattr(control.device, "capture", lambda *args: (np.zeros((32, 32), np.uint16), {}))
    result = wait(control, action(control, "sem_capture"))
    assert result["status"] == "failed" and result["error"]["code"] == "INVALID_IMAGE"
    assert result["result"]["image"]["width"] == 32
    assert (control.root / "images" / result["job_id"] / "original.png").is_file()


def test_position_drifts_after_arrival_blocks_capture_and_followup(control, monkeypatch):
    confirm = control._confirm_position
    def confirm_then_drift(target):
        arrived = confirm(target)
        control.device.position = (40., 50.)
        return arrived
    monkeypatch.setattr(control, "_confirm_position", confirm_then_drift)
    result = wait(control, action(control))
    assert result["status"] == "unknown" and result["error"]["code"] == "POSITION_UNCONFIRMED"
    assert result["result"]["movement"]["after_um"] == [10., 20.]
    assert control.device.commands.count("move") == 1 and "capture" not in control.device.commands
    assert action(control)["job_id"] == result["job_id"]
    with pytest.raises(DeviceBusy):
        action(control, request_id="follow-up")


def test_position_changes_during_capture_preserves_evidence_and_blocks_followup(control, monkeypatch):
    original = control.device.capture
    def capture(*args):
        result = original(*args)
        control.device.position = (5, 5)
        return result
    monkeypatch.setattr(control.device, "capture", capture)
    result = wait(control, action(control, "sem_capture"))
    assert result["status"] == "unknown"
    metadata = result["result"]["image"]
    assert metadata["position_um"] is None and metadata["position_verified"] is False
    assert metadata["position_before_um"] == [0, 0] and metadata["position_after_um"] == [5, 5]
    assert (control.root / "images" / result["job_id"] / "original.png").is_file()
    with pytest.raises(DeviceBusy):
        action(control, "sem_capture", request_id="follow-up")


def test_real_capture_reads_metadata_and_preserves_unprocessed_pixels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from addict import Dict
    from zemrpc.zem.models import MetaItem, ZemImage
    device = RealMicroscope("not-used.invalid", 61557)
    calls = []
    frame = ZemImage(img=np.full((512, 512), 12345, np.uint16),
                     meta=Dict({MetaItem.PixelSize: 1., "bounds": [0., 0., 512., 512.]}))
    def snap(**kwargs):
        assert kwargs["raw"] is True and kwargs["save"] is False
        calls.append("snap")
        return frame
    api = SimpleNamespace(device=SimpleNamespace(
        get_model=lambda: calls.append("model"), get_hv=lambda: calls.append("voltage"),
        get_beam_size=lambda: calls.append("beam"), get_protocol_ng=lambda: calls.append("protocol")),
        sync_stage_and_scan_field=lambda: calls.append("scan"), snap=snap)
    monkeypatch.setattr(device, "_api", lambda: api)
    image, metadata = device.capture(512, 7, tmp_path)
    assert image.dtype == np.uint16 and np.all(image == 12345)
    assert metadata["raw_sha256"]
    saved = cv2.imread(str(tmp_path / metadata["raw_file"]), cv2.IMREAD_UNCHANGED)
    assert saved.dtype == np.uint16 and np.array_equal(saved, frame.img)
    assert (tmp_path / metadata["raw_file"]).read_bytes().startswith(b"\x89PNG")
    assert calls == ["model", "voltage", "beam", "protocol", "scan", "snap"]


def test_capture_keeps_storage_paths_when_sdk_changes_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    vendor_home = tmp_path / "vendor-home"
    vendor_home.mkdir()
    device = SimulatedMicroscope()
    original = device.capture
    def capture(*args):
        monkeypatch.chdir(vendor_home)
        return original(*args)
    monkeypatch.setattr(device, "capture", capture)
    service = SemControl(Path("mcp"), device, DeviceCoordinator(Path("mcp/control-state.json")), allow_capture=True)
    try:
        result = wait(service, action(service, "sem_capture"))
        assert result["status"] == "completed"
        assert (tmp_path / "mcp" / "images" / result["job_id"] / "original.png").is_file()
        assert (tmp_path / "mcp" / "control-state.json").is_file()
        assert not (vendor_home / "mcp").exists()
    finally:
        service.close()


def test_real_operations_are_disabled_without_site_configuration(tmp_path):
    service = SemControl(tmp_path, SimulatedMicroscope(), DeviceCoordinator())
    try:
        with pytest.raises(SemError) as error:
            action(service)
        assert error.value.code == "OPERATION_DISABLED"
        assert not service.device.commands
    finally:
        service.close()


@pytest.mark.parametrize("changes", [
    {"device_status": "Busy"}, {"stage_state": 0}, {"sample_changing": 1},
    {"sample_changing": None},
])
def test_unsafe_or_unknown_state_blocks_actuation(control, monkeypatch, changes):
    status = {**control.device.status(), **changes}
    monkeypatch.setattr(control.device, "status", lambda: status)
    result = wait(control, action(control))
    assert result["status"] == "failed"
    assert "move" not in control.device.commands and "capture" not in control.device.commands


def test_capture_allows_closed_or_unknown_gun(control, monkeypatch):
    status = {**control.device.status(), "gun_state": 0}
    monkeypatch.setattr(control.device, "status", lambda: status)
    result = wait(control, action(control, "sem_capture"))
    assert result["status"] == "completed"
    assert control.device.commands.count("capture") == 1


def test_unconfirmed_movement_does_not_capture_or_retry(control, monkeypatch):
    monkeypatch.setattr(control.device, "move", lambda *args: control.device.commands.append("move"))
    result = wait(control, action(control))
    assert result["status"] == "unknown"
    assert result["error"]["code"] == "POSITION_UNCONFIRMED"
    assert control.device.commands.count("move") == 1 and "capture" not in control.device.commands
    with pytest.raises(DeviceBusy, match="人工核验"):
        action(control, request_id="another")
    assert control.call("sem_get_status", {})["unresolved_jobs"] == [result["job_id"]]


def test_lost_move_response_keeps_unknown_and_never_replays(control, monkeypatch):
    def lost(*_):
        control.device.commands.append("move")
        raise TimeoutError("response lost")
    monkeypatch.setattr(control.device, "move", lost)
    first = wait(control, action(control))
    assert first["status"] == "unknown"
    assert action(control)["job_id"] == first["job_id"]
    assert control.device.commands.count("move") == 1


def test_cancellation_waits_for_physical_step_and_does_not_capture(control, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = control.device.move
    def move(*target):
        entered.set()
        assert release.wait(3)
        original(*target)
    monkeypatch.setattr(control.device, "move", move)
    job = action(control)
    assert entered.wait(2)
    cancelled = control.call("sem_cancel_job", {"job_id": job["job_id"]}, "test")
    assert cancelled["status"] == "cancelling" and control.coordinator.owner
    with pytest.raises(DeviceBusy):
        control.coordinator.acquire("desktop")
    release.set()
    result = wait(control, job)
    assert result["status"] == "cancelled" and "capture" not in control.device.commands
    assert control.coordinator.owner is None


def test_desktop_and_mcp_exclude_each_other(control):
    from backend.server import JobManager
    desktop = JobManager(control.coordinator)
    job = desktop.create("task", "analysis")
    with pytest.raises(DeviceBusy):
        action(control)
    desktop.cancel(job["id"])
    with pytest.raises(DeviceBusy):
        action(control)
    desktop.update(job["id"], status="completed")
    assert wait(control, action(control))["status"] == "completed"


def test_image_and_job_access_are_scoped_to_caller(control):
    result = wait(control, action(control))
    with pytest.raises(SemError):
        control.call("sem_cancel_job", {"job_id": result["job_id"]}, "other")
    with pytest.raises(SemError):
        control.call("sem_get_image", {"image_id": result["result"]["image"]["image_id"]}, "other")
    with pytest.raises(SemError):
        control.call("sem_get_image", {"image_id": "../../Windows"}, "test")


def test_corrupted_saved_image_is_not_returned_as_original(control):
    result = wait(control, action(control))
    image_id = result["result"]["image"]["image_id"]
    (control.root / "images" / image_id / "original.png").write_bytes(b"corrupted")
    with pytest.raises(SemError) as error:
        control.call("sem_get_image", {"image_id": image_id, "variant": "original"}, "test")
    assert error.value.code == "IMAGE_CHANGED"


def test_persisted_inflight_job_is_unknown_after_restart(tmp_path):
    first = SemControl(tmp_path, SimulatedMicroscope(), DeviceCoordinator())
    record = {"job_id": "a" * 32, "operation": "sem_move_xy", "status": "running"}
    first._db.execute("INSERT INTO jobs VALUES (?,?,?,?,?)", (record["job_id"], "test", "interrupted", "{}", json.dumps(record)))
    first._db.commit()
    first.close()
    recovered = SemControl(tmp_path, SimulatedMicroscope(), DeviceCoordinator())
    try:
        assert recovered.call("sem_get_job", {"job_id": record["job_id"]}, "test")["status"] == "unknown"
        assert recovered.coordinator.unresolved == [record["job_id"]]
        assert not recovered.device.commands
    finally:
        recovered.close()


def test_navigation_does_not_require_gun_and_is_not_a_sem_capture(control, monkeypatch):
    status = {**control.device.status(), "gun_state": 0}
    monkeypatch.setattr(control.device, "status", lambda: status)
    result = wait(control, action(control, "sem_get_navigation"))
    assert result["status"] == "completed"
    assert "capture" not in control.device.commands and "move" not in control.device.commands
    assert result["result"]["image"]["bounds_um"] == [-256, -256, 256, 256]


def test_sdk_retries_disabled_only_inside_mcp_scope():
    class Device:
        calls = 0
        def _prepare_comm(self): pass
        def once(self, request):
            self.calls += 1
            raise TimeoutError("lost")
        @wraps(once)
        def pubsub_send(self, request):
            for _ in range(3):
                try: return self.once(request)
                except TimeoutError: pass
            raise TimeoutError("lost")
    device = Device()
    _configure_zemrpc_device(device)
    with pytest.raises(TimeoutError), zemrpc_single_attempt():
        device.pubsub_send(SimpleNamespace(cmd=SimpleNamespace(name="SetStagePos")))
    assert device.calls == 1
    with pytest.raises(TimeoutError):
        device.pubsub_send(SimpleNamespace(cmd=SimpleNamespace(name="SetStagePos")))
    assert device.calls == 4


def test_backend_owner_lock_is_cross_process(tmp_path):
    path = tmp_path / "owner.lock"
    source = ("import sys; from pathlib import Path; "
              "from backend.cleanliness_core.device_control import BackendOwnerLock; "
              "lease=BackendOwnerLock(Path(sys.argv[1])); lease.__enter__()")
    with BackendOwnerLock(path):
        result = subprocess.run([sys.executable, "-c", source, str(path)], capture_output=True, timeout=5,
                                cwd=Path(__file__).resolve().parents[2])
        assert result.returncode != 0 and b"DeviceBusy" in result.stderr
    with BackendOwnerLock(path):
        pass


def test_abandoned_desktop_owner_blocks_new_writes_after_restart(tmp_path):
    journal = tmp_path / "control-state.json"
    first = DeviceCoordinator(journal)
    first.acquire("desktop:interrupted")
    restarted = DeviceCoordinator(journal)
    assert restarted.unresolved == ["desktop:interrupted"]
    with pytest.raises(DeviceBusy):
        restarted.acquire("mcp:new")
    with restarted.operation("status", writes=False):
        pass
    assert DeviceCoordinator(journal).unresolved == ["desktop:interrupted"]


def test_operator_acknowledgement_is_offline_audited_and_does_not_replay(tmp_path):
    from backend.sem_admin import acknowledge
    service = SemControl(tmp_path, SimulatedMicroscope(), DeviceCoordinator(tmp_path / "control-state.json"))
    record = {"job_id": "b" * 32, "operation": "sem_move_xy", "status": "unknown"}
    service._db.execute("INSERT INTO jobs VALUES (?,?,?,?,?)", (record["job_id"], "test", "uncertain", "{}", json.dumps(record)))
    service._db.commit()
    service.coordinator.quarantine(record["job_id"])
    service.close()
    with BackendOwnerLock(tmp_path / "simulated-backend.lock"), pytest.raises(DeviceBusy):
        acknowledge(tmp_path, record["job_id"], "测试：现场核验完成", simulated=True)
    acknowledge(tmp_path, record["job_id"], "测试：现场核验完成", simulated=True)
    coordinator = DeviceCoordinator(tmp_path / "control-state.json")
    assert not coordinator.unresolved
    restarted = SemControl(tmp_path, SimulatedMicroscope(), coordinator)
    try:
        result = restarted.call("sem_get_job", {"job_id": record["job_id"]}, "test")
        assert result["status"] == "failed" and result["operator_acknowledgement"]["previous_status"] == "unknown"
        assert not restarted.device.commands
        with coordinator.operation("new-desktop-job"):
            pass
        saved = json.loads((tmp_path / "control-state.json").read_text(encoding="utf-8"))
        assert saved["acknowledgements"][0]["operation_id"] == record["job_id"]
    finally:
        restarted.close()
