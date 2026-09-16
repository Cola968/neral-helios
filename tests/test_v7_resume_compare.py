"""Adversarial CPU fixtures only; no Neral checkpoint or GPU workload."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frontier350"))

from nextgen_v1.compare_exact_resume import compare_states, main
from v7_recovery import capture_exact_recovery


def _fixture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    torch.manual_seed(21)
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(2, 2)).square().mean().backward()
    opt.step()
    return capture_exact_recovery(
        named=list(model.named_parameters()), optimizer=opt,
        rng=random.Random(22), contract={"lr": 0.01, "data_sha256": {"toy": "b" * 64}},
        anchor_sha256="a" * 64, source_processed_tokens=100,
        done=16, step=1, chain="d" * 64, counts={"main": 16},
    )


def test_exact_identical_copy_is_a_match(monkeypatch):
    state = _fixture(monkeypatch)
    assert compare_states(state, deepcopy(state)) == (True, "all serialized fields equal")


def test_optimizer_tensor_and_rng_changes_fail(monkeypatch):
    state = _fixture(monkeypatch)
    changed = deepcopy(state)
    key = next(iter(changed["optimizer"]["state"]))
    changed["optimizer"]["state"][key]["exp_avg"][0, 0] += 0.01
    equal, reason = compare_states(state, changed)
    assert not equal and "optimizer" in reason and "tensor values" in reason
    changed = deepcopy(state)
    changed["torch_rng"][0] ^= 1
    equal, reason = compare_states(state, changed)
    assert not equal and "torch_rng" in reason
    changed = deepcopy(state)
    changed["sampler_rng"] = random.Random(99).getstate()
    equal, reason = compare_states(state, changed)
    assert not equal and "sampler_rng" in reason


def test_token_chain_and_inconsistent_counters_fail(monkeypatch):
    state = _fixture(monkeypatch)
    changed = deepcopy(state)
    changed["batch_chain"] = "e" * 64
    equal, reason = compare_states(state, changed)
    assert not equal and "batch_chain" in reason
    changed = deepcopy(state)
    changed["source_tokens"]["main"] = 15
    equal, reason = compare_states(state, changed)
    assert not equal and "per-source token totals" in reason
    changed = deepcopy(state)
    changed.pop("numpy_rng")
    equal, reason = compare_states(state, changed)
    assert not equal and "missing numpy_rng" in reason


def test_local_cli_requires_pinned_distinct_files_and_never_promotes(
    monkeypatch, tmp_path, capsys,
):
    state = _fixture(monkeypatch)
    a, b = tmp_path / "uninterrupted.pt", tmp_path / "resumed.pt"
    torch.save(state, a)
    torch.save(deepcopy(state), b)
    sha_a = hashlib.sha256(a.read_bytes()).hexdigest()
    sha_b = hashlib.sha256(b.read_bytes()).hexdigest()
    args = ["compare_exact_resume.py", "--uninterrupted", str(a), "--resumed", str(b),
            "--uninterrupted-sha256", sha_a, "--resumed-sha256", sha_b,
            "--anchor-sha256", "a" * 64, "--acknowledge-trusted-local-pickle"]
    monkeypatch.setattr(sys, "argv", args)
    assert main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["exact_recovery_state_equal"] is True
    assert report["eligible_for_production"] is False
    assert report["real_gpu_training_verified"] is False
    monkeypatch.setattr(sys, "argv", args[:-5] + ["--uninterrupted-sha256", "f" * 64,
                                        "--resumed-sha256", sha_b, "--anchor-sha256", "a" * 64,
                                        "--acknowledge-trusted-local-pickle"])
    assert main() == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "rejected" and "independently pinned" in report["reason"]
    assert a.read_bytes() and b.read_bytes()
