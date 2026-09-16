"""CPU-only regression checks for V6 ten-update telemetry and 5M soak windows."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from apply_v8_2_metric_cadence_fix import patch_gate, patch_soak


def test_gate_cadence_patch_is_idempotent_and_syntax_valid() -> None:
    source = (HERE / "v8_2_throughput_gate.py").read_text(encoding="utf-8")
    patched = patch_gate(source)
    assert patch_gate(patched) == patched
    assert "args.updates <= 180" in patched
    assert "len(train) < 8" in patched
    assert "int(r.get('step', 0)) > args.warmup_updates" in patched
    assert "row['metric_steps'] != plain['metric_steps']" in patched
    assert "len(train) != args.updates" not in patched
    compile(patched, "gate-cadence", "exec")


def test_5m_soak_uses_10_update_windows() -> None:
    source = (HERE / "v8_2_thermal_soak.py").read_text(encoding="utf-8")
    patched = patch_soak(source)
    assert patch_soak(patched) == patched
    assert "int(r.get('step', 0)) > 30" in patched
    assert "len(stable_rows) < 12" in patched
    assert "full[10:]" not in patched
    compile(patched, "soak-cadence", "exec")


def test_existing_source_contract_remains_unmodified() -> None:
    source = (HERE / "v8_2_throughput_gate.py").read_text(encoding="utf-8")
    patched = patch_gate(source)
    assert "MAX_LOSS_DIFF = 2e-4" in patched
    assert "read_only_ok = anchor_sha_after == anchor_sha_before" in patched
    assert "'step', -1" in patched
    # 100 updates -> 10 telemetry windows; after 20 warmup, 8 remain.
    assert len([i for i in range(10, 101, 10) if i > 20]) == 8
    # 5M tokens -> 203 full updates -> 20 full windows; 17 beyond step 30.
    full_windows = [i for i in range(10, 204, 10)]
    assert len([i for i in full_windows if i > 30]) == 17
