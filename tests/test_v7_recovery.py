from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frontier350"))

from v7_recovery import capture_exact_recovery, restore_exact_recovery


def _named(model: torch.nn.Linear):
    return [("weight", model.weight), ("bias", model.bias)]


def _seed_optimizer(model: torch.nn.Linear, optimizer: torch.optim.Optimizer) -> None:
    x = torch.tensor([[1.0, -2.0, 0.5]])
    loss = model(x).square().sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_compact_exact_recovery_roundtrip_cpu(monkeypatch):
    # Production captures CUDA RNG too. CPU CI substitutes an empty CUDA RNG vector while
    # still exercising trainable masters, AdamW state, sampler/Python/Torch RNG and counters.
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _state: None)

    torch.manual_seed(71)
    random.seed(72)
    source = torch.nn.Linear(3, 2)
    source_opt = torch.optim.AdamW(source.parameters(), lr=2.6e-5)
    _seed_optimizer(source, source_opt)
    sampler = random.Random(73)
    sampler.random()

    contract = {
        "compile_mode": "max-autotune-no-cudagraphs",
        "fused_replay_main": True,
        "fast_replay_only": True,
        "lr": 2.6e-5,
        "data_sha256": {"main": {"shard": "abc"}},
    }
    state = capture_exact_recovery(
        named=_named(source),
        optimizer=source_opt,
        rng=sampler,
        contract=contract,
        anchor_sha256="a" * 64,
        source_processed_tokens=1_500_000_000,
        done=1_024_000,
        step=42,
        chain="chain-42",
        counts={"main": 900_000, "code": 124_000},
    )

    target = torch.nn.Linear(3, 2)
    target_opt = torch.optim.AdamW(target.parameters(), lr=2.6e-5)
    target_sampler = random.Random(999)
    done, step, chain, counts = restore_exact_recovery(
        state,
        named=_named(target),
        optimizer=target_opt,
        rng=target_sampler,
    )

    assert done == 1_024_000
    assert step == 42
    assert chain == "chain-42"
    assert counts == {"main": 900_000, "code": 124_000}
    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)
    assert target_opt.state_dict()["state"].keys() == source_opt.state_dict()["state"].keys()
    for key, source_row in source_opt.state_dict()["state"].items():
        target_row = target_opt.state_dict()["state"][key]
        for field, source_value in source_row.items():
            target_value = target_row[field]
            if torch.is_tensor(source_value):
                assert torch.equal(target_value, source_value)
            else:
                assert target_value == source_value
    assert target_sampler.getstate() == state["sampler_rng"]
