from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from helios1b.checkpointing import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
)
from helios1b.config import HeliosConfig
from helios1b.data import DeterministicTokenBatcher
from helios1b.learning_experiment import build_learnability_stream, write_stream
from helios1b.train_proxy import ProxyTrainingConfig, run_training


def test_rng_state_roundtrip():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    state = capture_rng_state()
    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )

    restore_rng_state(state)
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_batcher_cursor_resume_is_exact(tmp_path: Path):
    path = tmp_path / "tokens.bin"
    np.arange(2048, dtype="<u2").tofile(path)

    a = DeterministicTokenBatcher(
        path,
        seq_len=16,
        batch_size=4,
        seed=77,
        dtype="uint16",
    )
    _ = a.next_batch(torch.device("cpu"))
    state = a.state_dict()
    expected_x, expected_y = a.next_batch(torch.device("cpu"))

    b = DeterministicTokenBatcher(
        path,
        seq_len=16,
        batch_size=4,
        seed=77,
        dtype="uint16",
    )
    b.load_state_dict(state)
    actual_x, actual_y = b.next_batch(torch.device("cpu"))

    assert torch.equal(actual_x, expected_x)
    assert torch.equal(actual_y, expected_y)


def _write_training_data(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    train = root / "train.bin"
    val = root / "val.bin"
    write_stream(train, build_learnability_stream(700, 120))
    write_stream(val, build_learnability_stream(701, 60))
    return train, val


def _cfg(train: Path, val: Path, out: Path, **changes):
    base = ProxyTrainingConfig(
        train_bin=str(train),
        val_bin=str(val),
        out_dir=str(out),
        total_steps=4,
        seq_len=24,
        batch_size=4,
        grad_accum=1,
        eval_every=2,
        eval_batches=2,
        save_every=1,
        seed=702,
        peak_lr=8e-4,
        min_lr=8e-5,
        warmup_steps=1,
        weight_decay=0.05,
        grad_clip=1.0,
        device="cpu",
    )
    return replace(base, **changes)


def _model_cfg():
    return HeliosConfig(
        vocab_size=128,
        d_model=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        ffn_hidden=128,
        max_seq_len=32,
        z_loss_coeff=1e-4,
    )


def test_split_resume_matches_uninterrupted_training(tmp_path: Path):
    train, val = _write_training_data(tmp_path / "data")
    model_cfg = _model_cfg()

    full_dir = tmp_path / "full"
    split_dir = tmp_path / "split"

    full = run_training(_cfg(train, val, full_dir), model_cfg)
    assert full["completed_step"] == 4

    first = run_training(
        _cfg(train, val, split_dir, stop_after=2),
        model_cfg,
    )
    assert first["completed_step"] == 2

    resumed = run_training(
        _cfg(
            train,
            val,
            split_dir,
            stop_after=2,
            resume=str(split_dir / "latest.pt"),
        ),
        model_cfg,
    )
    assert resumed["completed_step"] == 4

    full_ckpt = load_checkpoint(full_dir / "latest.pt")
    split_ckpt = load_checkpoint(split_dir / "latest.pt")

    assert full_ckpt["completed_step"] == split_ckpt["completed_step"] == 4
    assert full_ckpt["tokens_seen"] == split_ckpt["tokens_seen"]
    assert full_ckpt["scheduler"] == split_ckpt["scheduler"]
    assert full_ckpt["batcher"] == split_ckpt["batcher"]

    for name, tensor in full_ckpt["model"].items():
        assert torch.equal(tensor, split_ckpt["model"][name]), name
