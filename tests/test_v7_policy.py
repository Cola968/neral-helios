from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frontier350"))

from v7_policy import fast2_gate, gpu_guard_reason, recovery_targets, semantic_resume_contract


def test_resume_contract_ignores_only_execution_profile():
    base = {
        "compile_mode": "max-autotune",
        "fast_replay_only": True,
        "fused_replay_main": True,
        "lr": 2.6e-5,
        "data_sha256": {"main": {"x": "abc"}},
    }
    safe = {**base, "compile_mode": "default", "fused_replay_main": False}
    assert semantic_resume_contract(base) == semantic_resume_contract(safe)
    changed = {**safe, "lr": 1.0e-5}
    assert semantic_resume_contract(base) != semantic_resume_contract(changed)


def test_gpu_guard_has_headroom_hysteresis_inputs():
    assert gpu_guard_reason(reserved_mib=8000, free_mib=2000, temp_c=78) is None
    assert gpu_guard_reason(reserved_mib=9300, free_mib=2000, temp_c=78) == "reserved_vram_ceiling"
    assert gpu_guard_reason(reserved_mib=8000, free_mib=1000, temp_c=78) == "free_vram_floor"
    assert gpu_guard_reason(reserved_mib=8000, free_mib=2000, temp_c=82) == "gpu_temperature_ceiling"


def test_fast2_gate_requires_speed_fidelity_and_memory():
    good = {
        "fast2_ratio": 1.10,
        "loss_abs_diff": 1e-5,
        "fast2": {"peak_reserved_gib": 8.5},
    }
    assert fast2_gate(good)[0]
    bad = {**good, "fast2": {"peak_reserved_gib": 9.2}}
    allowed, failures = fast2_gate(bad)
    assert not allowed and "peak_reserved_vram" in failures


def test_recovery_targets_exclude_endpoint():
    assert recovery_targets(5_000_000, 1_000_000) == [1_000_000, 2_000_000, 3_000_000, 4_000_000]
