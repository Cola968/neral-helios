"""Arctus350 V6 checkpoint admission: RED tests for invalid benchmark values.

CPU-only, synthetic reports. Never rents GPU, loads model weights or promotes PRs.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from frontier350.select_bilingual_recovery_v6 import main


SHA = "a" * 64
CHECKPOINT_SHA = "b" * 64
SUITE = "arctus-bilingual-holdout-v1"


def fixtures(tmp_path, *, baseline_overall=45.0, holdout_overall=47.0,
             holdout_directions=None, validation_loss=1.25,
             include_loss=True, sha=SHA):
    baseline = {
        "suite": SUITE, "cases_sha256": SHA,
        "cases": 400,
        "overall_accuracy_pct": baseline_overall,
        "directions": {"de_to_en": {"accuracy_pct": 40.0, "correct":80, "total":200},
                       "en_to_de": {"accuracy_pct": 50.0, "correct":100, "total":200}},
    }
    holdout = {
        "suite": SUITE, "cases_sha256": sha,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "cases":400,
        "overall_accuracy_pct": holdout_overall,
        "directions": holdout_directions or {
            "de_to_en": {"accuracy_pct": 44.0, "correct":88, "total":200},
            "en_to_de": {"accuracy_pct": 50.0, "correct":100, "total":200}},
    }
    candidate = {
        "name": "candidate-A",
        "overall_accuracy_pct": 60.0,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "categories": {"bilingual": 55.0},
    }
    if include_loss:
        candidate["validation_loss"] = validation_loss
    gate = {"selection_allowed": True, "selected": "candidate-A",
            "candidates": [candidate]}
    for name, value in (("baseline", baseline), ("holdout", holdout), ("gate", gate)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value), encoding="utf8")
    return tmp_path / "result.json"


def choose(tmp_path, out):
    arguments = [
        "selector", "--baseline-holdout", str(tmp_path / "baseline.json"),
        "--candidate", "candidate-A", str(tmp_path / "gate.json"),
        str(tmp_path / "holdout.json"), str(tmp_path / "checkpoint-out"),
        "--out", str(out),
    ]
    with patch.object(sys, "argv", arguments):
        main()
    return json.loads(out.read_text(encoding="utf8"))


def test_valid_numeric_candidate_is_selected_without_gpu(tmp_path):
    out=fixtures(tmp_path)
    report=choose(tmp_path,out)
    assert report["selected"]=="candidate-A"
    assert report["selection_allowed"] is True
    assert report["candidates"][0]["pass"] is True


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 105.0])
def test_invalid_holdout_accuracy_never_passes(tmp_path,bad):
    out=fixtures(tmp_path,holdout_overall=bad)
    report=choose(tmp_path,out)
    assert report["selected"] is None
    assert report["selection_allowed"] is False
    assert report["candidates"][0]["pass"] is False
    assert math.isfinite(report["candidates"][0]["holdout_overall_accuracy_pct"])
    assert "invalid_holdout_metric" in report["candidates"][0]["failures"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -2.0])
def test_invalid_validation_loss_disqualifies_candidate(tmp_path,bad):
    out=fixtures(tmp_path,validation_loss=bad)
    report=choose(tmp_path,out)
    assert report["selected"] is None
    assert "invalid_validation_loss" in report["candidates"][0]["failures"]


def test_missing_validation_loss_does_not_silently_default_to_1e9(tmp_path):
    out=fixtures(tmp_path,include_loss=False)
    report=choose(tmp_path,out)
    assert report["selected"] is None
    assert "invalid_validation_loss" in report["candidates"][0]["failures"]


def test_nan_direction_cannot_bypass_nonregression(tmp_path):
    out=fixtures(tmp_path,holdout_directions={
        "de_to_en":{"accuracy_pct":float("nan")},
        "en_to_de":{"accuracy_pct":51.0},
    })
    report=choose(tmp_path,out)
    assert report["selected"] is None
    assert "invalid_holdout_metric" in report["candidates"][0]["failures"]


@pytest.mark.parametrize("baseline_overall", [float("nan"),float("inf"),-1.0,101.0])
def test_invalid_baseline_fails_closed_without_creating_decision(tmp_path,baseline_overall):
    out=fixtures(tmp_path,baseline_overall=baseline_overall)
    with pytest.raises(RuntimeError,match="baseline"):
        choose(tmp_path,out)
    assert not out.exists()


def test_missing_holdout_identity_on_both_sides_fails_closed(tmp_path):
    out=fixtures(tmp_path)
    for name in ("baseline","holdout"):
        path=tmp_path/f"{name}.json"
        data=json.loads(path.read_text())
        data.pop("cases_sha256")
        path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError,match="baseline"):
        choose(tmp_path,out)
    assert not out.exists()


def test_missing_baseline_direction_set_is_invalid(tmp_path):
    out=fixtures(tmp_path)
    path=tmp_path/"baseline.json"
    report=json.loads(path.read_text())
    report["directions"]={}
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError,match="baseline"):
        choose(tmp_path,out)
    assert not out.exists()



def test_bilingual_gpu_evaluator_uses_existing_safe_checkpoint_loader():
    """Future GPU runs must not deserialize pickle with weights_only=False."""
    source=(Path(__file__).resolve().parents[1] /
            "frontier350" / "eval_bilingual_holdout_v1.py").read_text(encoding="utf8")
    assert "from secure_checkpoint350 import load_checkpoint" in source
    assert "ck = load_checkpoint(args.checkpoint)" in source
    assert "weights_only=False" not in source



def test_forged_holdout_percentage_without_matching_correct_counts_rejected(tmp_path):
    out=fixtures(tmp_path,holdout_overall=90.0)
    result=choose(tmp_path,out)
    assert result["selected"] is None
    assert result["selection_allowed"] is False
    assert "invalid_holdout_metric" in result["candidates"][0]["failures"]


def test_corrupt_direction_counts_rejected_even_with_finite_percentages(tmp_path):
    out=fixtures(tmp_path)
    path=tmp_path/"holdout.json"
    report=json.loads(path.read_text())
    report["directions"]["de_to_en"]["correct"]=201
    path.write_text(json.dumps(report))
    result=choose(tmp_path,out)
    assert result["selected"] is None
    assert "invalid_holdout_metric" in result["candidates"][0]["failures"]


def test_missing_case_counts_on_candidate_fail_closed(tmp_path):
    out=fixtures(tmp_path)
    path=tmp_path/"holdout.json"
    report=json.loads(path.read_text())
    report.pop("cases")
    path.write_text(json.dumps(report))
    result=choose(tmp_path,out)
    assert result["selected"] is None
    assert "invalid_holdout_metric" in result["candidates"][0]["failures"]


def test_wrong_total_holdout_cases_or_unbalanced_directions_rejected(tmp_path):
    out=fixtures(tmp_path)
    path=tmp_path/"holdout.json"
    report=json.loads(path.read_text())
    report["directions"]["de_to_en"]["total"]=201
    path.write_text(json.dumps(report))
    result=choose(tmp_path,out)
    assert result["selected"] is None
    assert "invalid_holdout_metric" in result["candidates"][0]["failures"]


def test_baseline_must_reconcile_case_counts(tmp_path):
    out=fixtures(tmp_path)
    path=tmp_path/"baseline.json"
    report=json.loads(path.read_text())
    report["directions"]["de_to_en"]["correct"]=199
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError,match="baseline"):
        choose(tmp_path,out)
    assert not out.exists()


@pytest.mark.parametrize("name", [
    "eval_bilingual_holdout_v1.py",
    "eval_recovery_dev_v2.py",
    "eval_validation_loss350.py",
])
def test_all_v6_evaluators_use_allowlisted_checkpoint_deserialization(name):
    source=(Path(__file__).resolve().parents[1] / "frontier350" / name).read_text(encoding="utf8")
    assert "from secure_checkpoint350 import load_checkpoint" in source
    assert "ck = load_checkpoint(args.checkpoint)" in source
    assert "weights_only=False" not in source




def test_holdout_must_measure_same_weights_as_dev_and_validation_gate(tmp_path):
    out=fixtures(tmp_path)
    path=tmp_path/"holdout.json"
    report=json.loads(path.read_text())
    report["checkpoint_sha256"]="c"*64
    path.write_text(json.dumps(report))
    result=choose(tmp_path,out)
    assert result["selected"] is None
    assert "checkpoint_identity_mismatch" in result["candidates"][0]["failures"]


def test_missing_dev_or_holdout_checkpoint_digest_rejected(tmp_path):
    for filename in ("gate.json","holdout.json"):
        root=tmp_path/filename.replace(".json","")
        root.mkdir()
        out=fixtures(root)
        path=root/filename
        payload=json.loads(path.read_text())
        if filename=="gate.json":
            payload["candidates"][0].pop("checkpoint_sha256")
        else:
            payload.pop("checkpoint_sha256")
        path.write_text(json.dumps(payload))
        result=choose(root,out)
        assert result["selected"] is None
        assert "checkpoint_identity_mismatch" in result["candidates"][0]["failures"]


def test_lr_selector_publishes_verified_checkpoint_digest_for_holdout_binding():
    source=(Path(__file__).resolve().parents[1] /
            "frontier350"/"select_recovery_lr_pilot.py").read_text(encoding="utf8")
    assert '"checkpoint_sha256": dev["checkpoint_sha256"]' in source


@pytest.mark.parametrize("selected", [None, "candidate-B", "baseline"])
def test_gate_selection_must_be_this_named_candidate(tmp_path, selected):
    out=fixtures(tmp_path)
    file=tmp_path/"gate.json"
    gate=json.loads(file.read_text(encoding="utf8"))
    gate["selected"]=selected
    file.write_text(json.dumps(gate),encoding="utf8")
    result=choose(tmp_path,out)
    assert result["selected"] is None
    assert "gate_candidate_identity_mismatch" in result["candidates"][0]["failures"]


def test_gate_cannot_relabel_other_candidate_or_bundle_multiple_candidates(tmp_path):
    for mutation in ("rename","two"):
        root=tmp_path/mutation
        root.mkdir()
        out=fixtures(root)
        file=root/"gate.json"
        gate=json.loads(file.read_text(encoding="utf8"))
        if mutation=="rename":
            gate["candidates"][0]["name"]="candidate-B"
        else:
            gate["candidates"].append(dict(gate["candidates"][0],name="candidate-B"))
        file.write_text(json.dumps(gate),encoding="utf8")
        result=choose(root,out)
        assert result["selected"] is None
        assert "gate_candidate_identity_mismatch" in result["candidates"][0]["failures"]

