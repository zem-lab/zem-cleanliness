"""Dependency smoke check uses a subprocess so its network guard stays isolated."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_driver_runtime_check_exercises_lazy_imports_without_server(tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "ZEM_TASK_ROOT": str(tmp_path / "tasks"), "ZEM_LOG_DIR": str(tmp_path / "logs")}
    result = subprocess.run([sys.executable, str(root / "backend/server.py"), "--check-runtime"],
                            env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary["runtimeCheck"] == "passed"
    assert summary["driverModels"] == "SnapArgs"
    assert summary["largeParticleCenterCheck"] == "passed"
    assert summary["mcpCheck"] == "passed"
    assert summary["networkActions"] == 0
