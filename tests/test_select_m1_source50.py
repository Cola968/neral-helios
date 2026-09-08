import hashlib
import json
from pathlib import Path

import pytest

from select_m1_source50 import choose


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf8")


def write_ckpt(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def base_assets(tmp_path: Path, ablation_id: int = 123) -> Path:
    assets = tmp_path / "assets"
    i1 = assets / "runs/neral50m_i1_instruct_d3/best_instruct.pt"
    i1_sha = write_ckpt(i1, b"i1")
    write_json(assets / "runs/neral50m_i1_instruct_d3/summary.json", {"complete": True, "best_sha256": i1_sha})
    write_json(assets / f"reports/stage_ablation_{ablation_id}/STAGE_ABLATION.json", {
        "diagnosis": {"rawmath_exact_regressed": True, "rawmath_nll_regressed": True}
    })
    return assets


def test_falls_back_to_i1_without_verified_e2(tmp_path):
    assets = base_assets(tmp_path)
    result = choose(assets, 123)
    assert result["selected"]["source_label"] == "i1_instruct"
    assert result["e2"]["eligible"] is False
    assert result["c3_auto_promotion"] is False


def test_selects_newest_verified_e2(tmp_path):
    assets = base_assets(tmp_path)
    e2 = assets / "runs/neral50m_i2_weakness_e2/best_repair.pt"
    e2_sha = write_ckpt(e2, b"e2-best")
    write_json(assets / "runs/neral50m_i2_weakness_e2/summary.json", {
        "complete": True, "effective_weight_candidate": True, "best_sha256": e2_sha,
    })
    write_json(assets / "reports/runner_50m_i2_200/I2_RESULT.json", {
        "stage": "I2 weakness repair E2 tri-loss", "weakness_gate_pass": True,
        "effective_weight_candidate": True, "candidate_sha": e2_sha,
    })
    result = choose(assets, 123)
    assert result["selected"]["source_label"] == "i2_e2"
    assert result["selected"]["sha256"] == e2_sha
    assert result["e2"]["eligible"] is True


def test_failed_e2_gate_cannot_override_i1(tmp_path):
    assets = base_assets(tmp_path)
    e2_sha = write_ckpt(assets / "runs/neral50m_i2_weakness_e2/best_repair.pt", b"e2")
    write_json(assets / "runs/neral50m_i2_weakness_e2/summary.json", {
        "complete": True, "effective_weight_candidate": True, "best_sha256": e2_sha,
    })
    write_json(assets / "reports/runner_50m_i2_201/I2_RESULT.json", {
        "stage": "I2 weakness repair E2 tri-loss", "weakness_gate_pass": False,
        "effective_weight_candidate": True, "candidate_sha": e2_sha,
    })
    result = choose(assets, 123)
    assert result["selected"]["source_label"] == "i1_instruct"
    assert result["e2"]["eligible"] is False


def test_hash_mismatch_blocks_e2(tmp_path):
    assets = base_assets(tmp_path)
    actual = write_ckpt(assets / "runs/neral50m_i2_weakness_e2/best_repair.pt", b"actual")
    write_json(assets / "runs/neral50m_i2_weakness_e2/summary.json", {
        "complete": True, "effective_weight_candidate": True, "best_sha256": actual,
    })
    write_json(assets / "reports/runner_50m_i2_202/I2_RESULT.json", {
        "stage": "I2 weakness repair E2 tri-loss", "weakness_gate_pass": True,
        "effective_weight_candidate": True, "candidate_sha": "0" * 64,
    })
    result = choose(assets, 123)
    assert result["selected"]["source_label"] == "i1_instruct"
    assert result["e2"]["reason"] == "e2_summary_or_hash_verification_failed"


def test_ablation_is_required(tmp_path):
    assets = tmp_path / "assets"
    with pytest.raises(FileNotFoundError):
        choose(assets, 999)
