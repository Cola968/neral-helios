"""Offline functional A/B test for the REAL V7 recovery serializer/restorer.

This exercises a small CPU Linear model, not the Neral350M CUDA trainer or a real
checkpoint. A passing test establishes deterministic continuation for this fixture
only. It cannot authorize an actual checkpoint, execution-profile switch or release.
"""
from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frontier350"))

from v7_recovery import (capture_exact_recovery, load_exact_recovery,
                         restore_exact_recovery, save_exact_recovery)

ANCHOR = "a" * 64
SOURCE_TOKENS = 1_010_003_584  # Fixture counter; NOT a checkpoint identity.
TOKENS_PER_STEP = 24
STEPS = 8
SPLIT = 3
CONTRACT = {
    "anchor_sha256": ANCHOR,
    "data_sha256": {"main": {"synthetic-fixture": "b" * 64}},
    "trainable_names": ["weight", "bias"],
    "lr": 0.004,
    "new_tokens": STEPS * TOKENS_PER_STEP,
    "tokens_per_update": TOKENS_PER_STEP,
    "compile_mode": "default",
    "fast_replay_only": False,
    "fused_replay_main": False,
}


def _create():
    torch.manual_seed(100)
    np.random.seed(101)
    random.seed(102)
    sampler = random.Random(103)
    model = torch.nn.Linear(3, 2).double()
    opt = torch.optim.AdamW(model.parameters(), lr=CONTRACT["lr"],
                            betas=(0.9, 0.95), weight_decay=0.01, foreach=False)
    return model, opt, sampler, 0, 0, "0" * 64, {"main": 0, "replay": 0}


def _named(model):
    return list(model.named_parameters())


def _advance(model, opt, sampler, done, step, chain, counts, number):
    counts = dict(counts)
    for _ in range(number):
        domain = "main" if sampler.randrange(3) else "replay"
        # All four RNG sources influence training or the input chain.
        x = torch.randn(4, 3, dtype=torch.float64)
        y = torch.randn(4, 2, dtype=torch.float64)
        x = x + float(np.random.uniform(-0.1, 0.1))
        y = y + random.uniform(-0.1, 0.1)
        x = x + sampler.uniform(-0.1, 0.1)
        chain = hashlib.sha256(
            bytes.fromhex(chain) + domain.encode("ascii")
            + TOKENS_PER_STEP.to_bytes(8, "little")
            + x.numpy().tobytes() + y.numpy().tobytes()
        ).hexdigest()
        opt.zero_grad(set_to_none=True)
        (model(x) - y).square().mean().backward()
        opt.step()
        done += TOKENS_PER_STEP
        step += 1
        counts[domain] += TOKENS_PER_STEP
    return done, step, chain, counts


def _capture(model, opt, sampler, done, step, chain, counts):
    return capture_exact_recovery(
        named=_named(model), optimizer=opt, rng=sampler, contract=CONTRACT,
        anchor_sha256=ANCHOR, source_processed_tokens=SOURCE_TOKENS,
        done=done, step=step, chain=chain, counts=counts,
    )


def _identical(left, right):
    if torch.is_tensor(left):
        assert torch.is_tensor(right)
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _identical(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _identical(a, b)
    else:
        assert left == right


def test_uninterrupted_and_serialized_resume_match_exactly(tmp_path, monkeypatch):
    # No CUDA is available in CI. The production module's CUDA RNG interface is
    # mocked ONLY for this CPU fixture; actual GPU RNG fidelity remains untested.
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _: None)

    reference = _create()
    m, opt, sampler, done, step, chain, counts = reference
    done, step, chain, counts = _advance(m, opt, sampler, done, step, chain, counts, STEPS)
    straight = _capture(m, opt, sampler, done, step, chain, counts)

    m, opt, sampler, done, step, chain, counts = _create()
    done, step, chain, counts = _advance(m, opt, sampler, done, step, chain, counts, SPLIT)
    recovery = tmp_path / "isolated" / "v7_recovery_72.pt"
    saved = save_exact_recovery(
        recovery, named=_named(m), optimizer=opt, rng=sampler,
        contract=CONTRACT, anchor_sha256=ANCHOR,
        source_processed_tokens=SOURCE_TOKENS,
        done=done, step=step, chain=chain, counts=counts,
    )
    assert recovery.is_file() and not recovery.with_suffix(".pt.tmp").exists()
    before_digest = hashlib.sha256(recovery.read_bytes()).hexdigest()

    # A fallback execution profile may change ONLY execution-only contract keys.
    other_profile = {**CONTRACT, "compile_mode": "max-autotune-no-cudagraphs",
                     "fast_replay_only": True, "fused_replay_main": True}
    loaded = load_exact_recovery(recovery, expected_anchor_sha256=ANCHOR,
                                 contract=other_profile)
    _identical(saved, loaded)
    with pytest.raises(ValueError, match="anchor differs"):
        load_exact_recovery(recovery, expected_anchor_sha256="f" * 64,
                            contract=other_profile)
    with pytest.raises(ValueError, match="semantic contract differs"):
        load_exact_recovery(recovery, expected_anchor_sha256=ANCHOR,
                            contract={**other_profile, "data_sha256": {"main": {"wrong": "c" * 64}}})
    assert hashlib.sha256(recovery.read_bytes()).hexdigest() == before_digest

    # Scramble everything; restoration must not depend on the current RNG state.
    torch.manual_seed(999)
    np.random.seed(998)
    random.seed(997)
    restored = torch.nn.Linear(3, 2).double()
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=CONTRACT["lr"],
                                    betas=(0.9, 0.95), weight_decay=0.01, foreach=False)
    restored_sampler = random.Random(996)
    done, step, chain, counts = restore_exact_recovery(
        loaded, named=_named(restored), optimizer=restored_opt, rng=restored_sampler)
    assert (done, step) == (SPLIT * TOKENS_PER_STEP, SPLIT)
    done, step, chain, counts = _advance(
        restored, restored_opt, restored_sampler, done, step, chain, counts, STEPS - SPLIT)
    resumed = _capture(restored, restored_opt, restored_sampler, done, step, chain, counts)
    _identical(straight, resumed)  # weights, AdamW, all RNGs, hashes and counters.
    assert resumed["processed_tokens"] == SOURCE_TOKENS + STEPS * TOKENS_PER_STEP
    assert sum(resumed["source_tokens"].values()) == resumed["new_tokens"]
    assert hashlib.sha256(recovery.read_bytes()).hexdigest() == before_digest


def test_restore_rejects_different_trainable_parameter_names(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    m, opt, sampler, done, step, chain, counts = _create()
    done, step, chain, counts = _advance(m, opt, sampler, done, step, chain, counts, 1)
    state = _capture(m, opt, sampler, done, step, chain, counts)
    different = torch.nn.Linear(3, 2).double()
    target_opt = torch.optim.AdamW(different.parameters(), lr=CONTRACT["lr"])
    with pytest.raises(ValueError, match="trainable parameter set differs"):
        restore_exact_recovery(
            state, named=[("other.weight", different.weight), ("bias", different.bias)],
            optimizer=target_opt, rng=random.Random(1))
