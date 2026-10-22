from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)

MAX_REPLACE_ATTEMPTS = 10
INITIAL_RETRY_DELAY_SECONDS = 0.05
MAX_RETRY_DELAY_SECONDS = 0.5
TRANSIENT_WINDOWS_REPLACE_ERRORS = frozenset({5, 32})


def _temporary_path(path: Path) -> Path:
    return path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )


def _is_transient_windows_lock(error: PermissionError) -> bool:
    return getattr(error, "winerror", None) in TRANSIENT_WINDOWS_REPLACE_ERRORS


def _retry_delay(attempt: int) -> float:
    return min(
        INITIAL_RETRY_DELAY_SECONDS * (2**attempt),
        MAX_RETRY_DELAY_SECONDS,
    )


def write_json_atomic(path: Path, payload: dict[str, Any], *, indent: int | None = 2) -> None:
    """Durably write JSON and atomically replace *path*.

    Windows temporarily rejects ``os.replace`` while another process has the
    destination open without delete sharing. Retry only those transient sharing
    errors. A complete temporary file is intentionally kept after a final
    replace failure so that the newest state remains recoverable.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    serialized = json.dumps(payload, ensure_ascii=False, indent=indent,
                            separators=(',', ':') if indent is None else None)

    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    for attempt in range(MAX_REPLACE_ATTEMPTS):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            final_attempt = attempt == MAX_REPLACE_ATTEMPTS - 1
            if not _is_transient_windows_lock(error) or final_attempt:
                if hasattr(error, "add_note"):
                    error.add_note(f"最新 JSON 已保留在恢复文件：{temporary}")
                raise

            delay = _retry_delay(attempt)
            LOGGER.warning(
                "JSON 原子替换遇到 Windows 文件占用，%.0f ms 后重试（%d/%d）：%s -> %s",
                delay * 1000,
                attempt + 1,
                MAX_REPLACE_ATTEMPTS,
                temporary,
                path,
            )
            time.sleep(delay)
