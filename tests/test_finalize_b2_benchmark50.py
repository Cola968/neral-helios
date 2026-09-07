import json
from pathlib import Path

import pytest

from finalize_b2_benchmark50 import MIN_B2_TOKENS, resolve_state, sha256_file, validate_b2_summary


def good_summary():
    return {
        "complete": True,
        "stop": "target_reached",
        "target_cumulative_tokens": MIN_B2_TOKENS,
        "cumulative_tokens": MIN_B2_TOKENS,
        "train_block_replay": False,
        "instruction_tuning_performed": False,
    }


def test_completed_100m_summary_passes():
    validate_b2_summary(good_summary())


@pytest.mark.parametrize("field,value", [
    ("complete", False),
    ("stop", "forgetting_abort"),
    ("train_block_replay", True),
    ("instruction_tuning_performed", True),
])
def test_fail_closed_stage_contract(field, value):
    row = good_summary(); row[field] = value
    with pytest.raises(ValueError):
        validate_b2_summary(row)


def test_under_100m_is_rejected():
    row = good_summary(); row["cumulative_tokens"] = MIN_B2_TOKENS - 1
    with pytest.raises(ValueError):
        validate_b2_summary(row)


def test_resolve_state_verifies_best_checkpoint_hash(tmp_path: Path):
    assets = tmp_path / "assets"
    b2 = assets / "runs" / "neral50m_b2_100m"; b2.mkdir(parents=True)
    b1 = assets / "runs" / "neral50m_stage_b_full"; b1.mkdir(parents=True)
    base = assets / "export" / "pilot_02"; base.mkdir(parents=True)
    (base / "neral20m.pt").write_bytes(b"20m")
    (b1 / "best_validation.pt").write_bytes(b"b1")
    (b2 / "best_validation.pt").write_bytes(b"b2")
    summary = good_summary() | {"best_sha256": sha256_file(b2 / "best_validation.pt")}
    (b2 / "summary.json").write_text(json.dumps(summary), encoding="utf8")
    state = resolve_state(assets)
    assert state["baseline20"].name == "neral20m.pt"
    assert state["b1"].name == "best_validation.pt"
    assert state["b2"].name == "best_validation.pt"


def test_resolve_state_rejects_checkpoint_hash_mismatch(tmp_path: Path):
    assets = tmp_path / "assets"
    b2 = assets / "runs" / "neral50m_b2_100m"; b2.mkdir(parents=True)
    b1 = assets / "runs" / "neral50m_stage_b_full"; b1.mkdir(parents=True)
    base = assets / "export" / "pilot_02"; base.mkdir(parents=True)
    (base / "neral20m.pt").write_bytes(b"20m")
    (b1 / "best_validation.pt").write_bytes(b"b1")
    (b2 / "best_validation.pt").write_bytes(b"b2")
    (b2 / "summary.json").write_text(json.dumps(good_summary() | {"best_sha256": "0" * 64}), encoding="utf8")
    with pytest.raises(ValueError, match="SHA-256"):
        resolve_state(assets)
