"""One operation owner for desktop jobs and MCP, independent of the SDK."""
from __future__ import annotations

import os
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path


class DeviceBusy(ValueError):
    pass


class DeviceCoordinator:
    def __init__(self, state_path: Path | None = None):
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._owner: str | None = None
        self._interrupt_owner: str | None = None
        self._unresolved: set[str] = set()
        self._state_path = Path(state_path).resolve() if state_path is not None else None
        self._writes = False
        self._history = []
        if self._state_path is not None and self._state_path.is_file():
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
            self._unresolved.update(state.get("unresolved", []))
            self._history = state.get("acknowledgements", [])
            owner = state.get("active_owner")
            if owner:
                self._unresolved.add(owner.removeprefix("mcp:"))
            if state.get('active_interrupt'):
                self._unresolved.add(state['active_interrupt'].removeprefix('mcp:'))

    def _persist(self, active_owner=None):
        if self._state_path is not None:
            from .atomic_json import write_json_atomic
            write_json_atomic(self._state_path, {"active_owner": active_owner,
                                                "active_interrupt": self._interrupt_owner,
                                                "unresolved": sorted(self._unresolved),
                                                "acknowledgements": self._history})

    @property
    def owner(self) -> str | None:
        with self._lock:
            return self._owner or self._interrupt_owner

    @property
    def unresolved(self) -> list[str]:
        with self._lock:
            return sorted(self._unresolved)

    def quarantine(self, job_id: str) -> None:
        with self._lock:
            self._unresolved.add(job_id)
            self._persist(self._owner if self._writes else None)

    def acquire(self, owner: str, *, writes: bool = True) -> None:
        with self._lock:
            if self._owner is not None or self._interrupt_owner is not None:
                raise DeviceBusy(f"设备已有操作正在运行：{self._owner}，请等待结束")
            if writes and self._unresolved:
                raise DeviceBusy("设备有执行结果待人工核验的作业，不能继续下发动作")
            if writes:
                self._persist(owner)
            self._writes = writes
            self._owner = owner

    def acquire_interrupt(self, owner: str) -> None:
        """Explicit stop requests may overlap an operation, never take its lock."""
        with self._changed:
            if self._interrupt_owner is not None or (self._owner or '').startswith('mcp-read:'):
                raise DeviceBusy('另一条停止命令或短查询正在执行，请稍后核对原请求')
            self._interrupt_owner = owner
            try:
                self._persist(self._owner if self._writes else None)
            except BaseException:
                self._interrupt_owner = None
                raise

    def release(self, owner: str) -> None:
        # Jobs may be submitted on the HTTP thread and completed on a worker.
        with self._lock:
            if self._interrupt_owner == owner:
                self._interrupt_owner = None
                self._persist(self._owner if self._writes else None)
                self._changed.notify_all()
                return
            if self._owner == owner:
                if self._writes:
                    self._persist()
                self._owner = None
                self._writes = False
                self._changed.notify_all()

    @contextmanager
    def query(self, owner: str, *, timeout: float = 2.):
        """Queue only short MCP reads; never wait through desktop/device jobs."""
        deadline = time.monotonic() + timeout
        owner = "mcp-read:" + owner
        with self._changed:
            while self._owner is not None or self._interrupt_owner is not None:
                if self._interrupt_owner is not None:
                    raise DeviceBusy('停止请求尚未返回，未插入查询')
                if not self._owner.startswith("mcp-read:"):
                    raise DeviceBusy(f"设备已有操作正在运行：{self._owner}，请等待结束")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DeviceBusy("只读查询排队超时，未向设备发送查询")
                self._changed.wait(remaining)
            self._owner, self._writes = owner, False
        try:
            yield
        finally:
            self.release(owner)

    @contextmanager
    def operation(self, owner: str, *, writes: bool = True):
        self.acquire(owner, writes=writes)
        try:
            yield
        finally:
            self.release(owner)


class BackendOwnerLock:
    """Hold an OS file lock for the backend lifetime; crashes release the lock.

    First version manages one microscope per logged-in user. All new desktop
    backends use the same lock, even with different task directories. A bridge
    never takes this lock or connects to hardware itself.
    """

    def __init__(self, path: Path | None = None):
        profile = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local" / "state")))
        self.path = Path(path).resolve() if path is not None else profile / "Zeptools" / "zem-cleanliness" / "device-owner.lock"
        self._handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            if self.path.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            raise DeviceBusy("已有电镜后端正在运行，请连接现有服务，不能重复接管设备") from error
        self._handle = handle
        return self

    def __exit__(self, *_):
        if self._handle is not None:
            self._handle.close()
            self._handle = None
