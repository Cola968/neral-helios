"""Read-only Arctus350 metadata admission; no real weights, downloads or GPU."""
from pathlib import Path

import pytest
from frontier350.chat350_contract import EXPECTED_CONFIG, EXPECTED_TOKENIZER_SHA256
from frontier350.readonly_checkpoint_preflight import inspect_metadata, inspect_file
from frontier350.vast4090_guard import sha256_file


def valid_meta():
    return {
        "config": EXPECTED_CONFIG,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "model": {"test": 1},
        "optimizer": {"test": 2},
        "torch_rng": b"synthetic",
        "numpy_rng": "synthetic",
        "python_rng": "synthetic",
        "step": 200,
        "processed_tokens": 2_073_161_216,
        "target_tokens": 3_000_000_000,
    }


def test_complete_resume_is_eligible_without_touching_gpu():
    report=inspect_metadata(valid_meta())
    assert report["has_optimizer"] is True
    assert report["rng_complete"] is True
    assert report["schedule_has_remaining_tokens"] is True
    assert report["resume_checkpoint_metadata_compatible"] is True
    assert report["training_started"] is False
    assert report["corpus_verified"] is False


def test_incomplete_legacy_export_fails_closed():
    payload=valid_meta()
    del payload["optimizer"]
    payload["target_tokens"]=payload["processed_tokens"]
    payload["numpy_rng"]=None
    report=inspect_metadata(payload)
    assert report["resume_checkpoint_metadata_compatible"] is False
    assert report["has_optimizer"] is False
    assert report["rng_complete"] is False
    assert report["schedule_has_remaining_tokens"] is False
    assert report["blocking_reasons"]


def test_hash_guard_rejects_wrong_weights_before_loader_runs(tmp_path):
    f=tmp_path/"resume.pt"
    f.write_bytes(b"fake checkpoint")
    calls=[]
    def loader(p):
        calls.append(p)
        return valid_meta()
    with pytest.raises(ValueError,match="SHA256"):
        inspect_file(f,"a"*64,checkpoint_loader=loader)
    assert calls == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["resume.pt"]


def test_read_only_file_probe_has_no_side_effects(tmp_path):
    f=tmp_path/"resume.pt"
    f.write_bytes(b"synthetic mock file, not real model")
    report=inspect_file(f,sha256_file(f),checkpoint_loader=lambda _:valid_meta())
    assert report["checkpoint_sha256_verified"] is True
    assert report["resume_checkpoint_metadata_compatible"] is True
    assert report["private_checkpoint_uploaded"] is False
    assert sorted(p.name for p in tmp_path.iterdir()) == ["resume.pt"]
