"""Offline operator acknowledgement. Deliberately absent from MCP tool discovery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from backend.cleanliness_core.atomic_json import write_json_atomic
from backend.cleanliness_core.device_control import BackendOwnerLock
from backend.sem_control import stamp


def acknowledge(root: Path, operation_id: str, reason: str, *, simulated=False):
    if not reason.strip():
        raise ValueError("必须填写人工检查记录；此操作不会验证设备或恢复任务")
    owner_path = root / "simulated-backend.lock" if simulated else None
    with BackendOwnerLock(owner_path):
        state_path = root / "control-state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        unresolved = set(state.get("unresolved", []))
        if state.get("active_owner"):
            unresolved.add(state["active_owner"].removeprefix("mcp:"))
        if state.get('active_interrupt'):
            unresolved.add(state['active_interrupt'].removeprefix('mcp:'))
        path = root / "operations.sqlite3"
        database = sqlite3.connect(f"{path.resolve().as_uri()}?mode=rw", uri=True)
        try:
            row = database.execute("SELECT record FROM jobs WHERE id=?", (operation_id,)).fetchone()
            if row:
                record = json.loads(row[0])
                if (record["status"] not in {"unknown", "accepted", "running", "queued", "cancelling"}
                        and not (record.get("error") or {}).get("code") == "OPERATOR_ACKNOWLEDGED_UNKNOWN"):
                    raise ValueError("该作业没有待核验的执行结果")
                record.setdefault("operator_acknowledgement", {"at": stamp(), "reason": reason,
                    "previous_status": record["status"], "previous_error": record.get("error")})
                record.update(status="failed", error={"code": "OPERATOR_ACKNOWLEDGED_UNKNOWN",
                    "message": "人工已检查设备并允许新操作；原作业未判定为成功，不自动重发"})
                database.execute("UPDATE jobs SET record=? WHERE id=?", (json.dumps(record, ensure_ascii=False), operation_id))
                database.commit()
            elif operation_id not in unresolved:
                raise ValueError("不存在对应的待核验作业或桌面操作")
            unresolved.discard(operation_id)
            history = list(state.get("acknowledgements", []))
            history.append({"operation_id": operation_id, "at": stamp(), "reason": reason})
            write_json_atomic(state_path, {"active_owner": None, "unresolved": sorted(unresolved),
                                           "acknowledgements": history})
        finally:
            database.close()


def main():
    parser = argparse.ArgumentParser(description="关闭软件并人工核验设备后，解除未知或仅受理操作拦截；不执行设备命令")
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--acknowledge-operation", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--simulated", action="store_true")
    args = parser.parse_args()
    acknowledge(args.state_dir, args.acknowledge_operation, args.reason, simulated=args.simulated)
    print("已记录人工核验，原操作不标记成功且不会重发；重新启动软件后再查询设备状态。")


if __name__ == "__main__":
    main()
