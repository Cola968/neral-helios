import json
from pathlib import Path

import pytest

from gpu_control_bridge import ALLOWED_ACTIONS, inspect_b2, load_request


def write_request(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "request.json"
    p.write_text(json.dumps(payload), encoding="utf8")
    return p


def test_allowed_actions_are_read_only_surface():
    assert ALLOWED_ACTIONS == {"gpu_status", "inspect_b2", "benchmark_b2", "run_tests"}


def test_load_request_accepts_exact_v1_contract(tmp_path):
    p = write_request(tmp_path, {"version": 1, "request_id": "status-001", "action": "gpu_status"})
    assert load_request(p) == {"version": 1, "request_id": "status-001", "action": "gpu_status"}


@pytest.mark.parametrize("payload", [
    {"version": 2, "request_id": "x", "action": "gpu_status"},
    {"version": 1, "request_id": "../escape", "action": "gpu_status"},
    {"version": 1, "request_id": "x", "action": "shell"},
    {"version": 1, "request_id": "x", "action": "gpu_status", "command": "whoami"},
])
def test_load_request_fail_closes(tmp_path, payload):
    with pytest.raises(ValueError):
        load_request(write_request(tmp_path, payload))


def test_inspect_b2_reports_summary_and_hashes_without_loading_weights(tmp_path):
    run = tmp_path / "runs" / "neral50m_b2_100m"
    run.mkdir(parents=True)
    summary = {
        "stop": "target_reached",
        "complete": True,
        "cumulative_tokens": 100_000_000,
        "target_cumulative_tokens": 100_000_000,
        "train_block_replay": False,
        "instruction_tuning_performed": False,
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf8")
    (run / "best_validation.pt").write_bytes(b"fake-private-checkpoint")
    result = inspect_b2(tmp_path)
    assert result["summary_exists"] is True
    assert result["summary"]["complete"] is True
    assert result["best_validation.pt"]["exists"] is True
    assert len(result["best_validation.pt"]["sha256"]) == 64
    assert result["weights_modified"] is False
