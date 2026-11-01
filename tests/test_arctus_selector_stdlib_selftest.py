"""Fail-closed Arctus V6 selector smoke runner stays CPU, offline, stdlib."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_portable_selector_smoke_run_has_valid_results_and_no_weights():
    root=Path(__file__).resolve().parents[1]
    cmd=[sys.executable, str(root/"frontier350"/"offline_selector_selftest.py")]
    response=subprocess.run(
        cmd,cwd=root,check=True,capture_output=True,text=True,timeout=15)
    report=json.loads(response.stdout)
    assert report["state"]=="PASS_OFFLINE"
    assert report["tests_passed"]>=5
    assert report["model_loaded"] is False
    assert report["training_started"] is False
    assert report["network_requested"] is False
    assert report["paid_gpu_used"] is False
