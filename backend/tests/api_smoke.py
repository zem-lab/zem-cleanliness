from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "backend" / "tests" / "fixtures" / "sample_cleanliness_report"
PORT = 18765
BASE = f"http://127.0.0.1:{PORT}"


def request(method: str, route: str, body: dict | None = None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        BASE + route,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        environment = {
            **os.environ,
            "ZEM_TASK_ROOT": str(Path(temporary) / "tasks"),
            "PYTHONIOENCODING": "utf-8",
        }
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "backend" / "server.py"), "--port", str(PORT)],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            for _ in range(50):
                try:
                    assert request("GET", "/health")["ok"] is True
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.1)
            else:
                raise AssertionError("backend did not become healthy")

            models = request("GET", "/models")["models"]
            assert len(models) == 3 and all(row["ready"] for row in models)
            print("models=ok")

            source = str(FIXTURE_ROOT / "optical.json")
            preview = request("POST", "/optical/preview", {"inputPath": source, "filters": {"minLengthUm": 70}})
            assert preview["particleCount"] == 5 and preview["selectedCount"] == 3
            print("preview=ok")

            task = request("POST", "/tasks", {"inputPath": source, "filters": {"minLengthUm": 70}, "selectedSequences": [row["sequence"] for row in preview["particles"]], "mode": "multi_particle"})
            folder_id = task["folderId"]
            assert task["status"] == "DRAFT"
            assert request("GET", f"/tasks/{folder_id}")["folderId"] == folder_id
            assert any(row["id"] == folder_id for row in request("GET", "/tasks")["tasks"])
            print("task_persistence=ok")

            saved = request("POST", "/settings", {"microscopeHost": "127.0.0.1", "microscopePort": 61557, "edsHost": "127.0.0.1", "edsPort": 6200})
            assert saved["gunOffAfterTask"] is True
            assert saved["showReportCoordinates"] is True
            assert saved["captureLevel"] == 7
            assert saved["candidateScoreMargin"] == 0.07
            assert saved["templateMatchMinScore"] == 0.65
            assert saved["contourOverlapMin"] == 0.5
            assert saved["maxCaptureAttempts"] == 2
            assert request("GET", "/settings")["microscopeHost"] == "127.0.0.1"
            print("settings=ok")

            unlocked = request("POST", "/engineer/unlock", {"password": "zeptools"})
            engineer = request("POST", "/engineer/settings", {"token": unlocked["token"]})
            assert len(engineer["definitions"]) == 15
            assert "task.gunOffAfterTask" not in engineer["values"]
            print("engineer_settings=ok")

            devices = request("GET", "/devices")["devices"]
            assert all(row["status"] == "offline" for row in devices)
            print("device_truthfulness=ok")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
