"""Offline contracts for transferring only a bounded audited Arctus BPE pilot.

No real checkpoints, networks, private data, model downloads, GPU or Vast host.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from frontier350.chat350_contract import EXPECTED_CONFIG, EXPECTED_TOKENIZER_SHA256
from frontier350.vast4090_guard import validate_data_manifest, sha256_file
from frontier350.vast4090_transfer import build_bundle, verify_bundle


def corpus(tmp_path):
    root = tmp_path / "legacy-bpe"
    for part, sizes in (("train", (4096,4096,4096)), ("valid", (4096,))):
        (root / part).mkdir(parents=True)
        for i, n in enumerate(sizes):
            (root / part / f"{part}_{i:05d}.bin").write_bytes(
                bytes([1, 0]) * n)
    manifest = {
        "schema": 2,
        "architecture": "frontier350-v1",
        "corpus_role": "foundation-pretraining",
        "benchmark_content_included": False,
        "tokenizer": {"tokenizer_sha256": EXPECTED_TOKENIZER_SHA256},
        "train_files": [], "valid_files": [],
    }
    for part in ("train", "valid"):
        for path in sorted((root / part).glob("*.bin")):
            manifest[f"{part}_files"].append({
                "path": path.name,
                "tokens": path.stat().st_size // 2,
                "sha256": sha256_file(path),
            })
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def fake_checkpoint():
    return {
        "config": EXPECTED_CONFIG,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "checkpoint_schema": 2,
        "model": {"w": "test"},
        "optimizer": {"state": "test"},
        "step": 100002,
        "processed_tokens": 2_073_161_216,
        "target_tokens": 10_000_000_000,
        "torch_rng": "test", "python_rng": "test", "numpy_rng": "test",
    }


def files(tmp_path):
    ck = tmp_path / "original-resume.pt"
    ck.write_bytes(b"synthetic checkpoint fixture, no private weights")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"test":"only"}')
    return ck, tokenizer


def pack(tmp_path, **overrides):
    root = corpus(tmp_path)
    ck, tok = files(tmp_path)
    args = dict(
        checkpoint=ck, expected_checkpoint_sha256=sha256_file(ck),
        tokenizer=tok, corpus_manifest=root / "manifest.json",
        output=tmp_path / "transfer",
        min_train_tokens=7000, min_valid_tokens=3000,
        checkpoint_loader=lambda path: fake_checkpoint(),
        test_tokenizer_sha256=sha256_file(tok),
    )
    args.update(overrides)
    return build_bundle(**args)


def test_small_transfer_is_hash_verified_and_compatible_with_existing_trainer(tmp_path):
    report = pack(tmp_path)
    out = tmp_path / "transfer"
    assert report["files_packed"] == 6  # checkpoint, tokenizer, manifest, 2 train and 1 valid
    assert report["training_started"] is False
    assert report["gpu_rented"] is False
    assert report["uploaded"] is False
    assert report["selected_train_tokens"] == 8192
    assert report["selected_valid_tokens"] == 4096
    assert sorted(x.name for x in (out/"inputs"/"data"/"train").glob("*.bin")) == [
        "train_00000.bin", "train_00001.bin"]
    assert not (out/"inputs"/"data"/"train"/"train_00002.bin").exists()
    summary = validate_data_manifest(out/"inputs"/"data"/"manifest.json")
    assert summary["train_shards"] == 2
    assert summary["valid_shards"] == 1
    assert verify_bundle(out, test_tokenizer_sha256=sha256_file(out/"inputs"/"tokenizer.json"))["verified"] is True


def test_package_does_not_overwrite_existing_output_or_source(tmp_path):
    root = corpus(tmp_path)
    ck, tok = files(tmp_path)
    original_ck = ck.read_bytes()
    original_sha = sha256_file(root/"train"/"train_00000.bin")
    pack_args = dict(checkpoint=ck,expected_checkpoint_sha256=sha256_file(ck),
        tokenizer=tok, corpus_manifest=root/"manifest.json",
        output=tmp_path/"transfer", min_train_tokens=4096,min_valid_tokens=4096,
        checkpoint_loader=lambda path: fake_checkpoint(),
        test_tokenizer_sha256=sha256_file(tok))
    build_bundle(**pack_args)
    with pytest.raises(FileExistsError):
        build_bundle(**pack_args)
    assert ck.read_bytes() == original_ck
    assert sha256_file(root/"train"/"train_00000.bin") == original_sha


def test_tampering_and_extraneous_files_fail_verification(tmp_path):
    pack(tmp_path)
    out = tmp_path/"transfer"
    victim = out/"inputs"/"data"/"train"/"train_00000.bin"
    data = bytearray(victim.read_bytes())
    data[0] ^= 1  # same file size, only the cryptographic content check detects it
    victim.write_bytes(data)
    with pytest.raises(ValueError,match="SHA256"):
        verify_bundle(out, test_tokenizer_sha256=sha256_file(out/"inputs"/"tokenizer.json"))


def test_corrupt_original_checkpoint_digest_fails_before_output(tmp_path):
    ck, tok = files(tmp_path)
    root = corpus(tmp_path)
    with pytest.raises(ValueError,match="SHA256"):
        build_bundle(checkpoint=ck,expected_checkpoint_sha256="f"*64,
            tokenizer=tok,corpus_manifest=root/"manifest.json",
            output=tmp_path/"transfer",min_train_tokens=4000,min_valid_tokens=4000,
            checkpoint_loader=lambda p:fake_checkpoint(),
            test_tokenizer_sha256=sha256_file(tok))
    assert not (tmp_path/"transfer").exists()


def test_out_of_schedule_or_incomplete_checkpoint_fails_before_copy(tmp_path):
    ck, tok = files(tmp_path)
    root = corpus(tmp_path)
    def expired(_):
        return {**fake_checkpoint(),"target_tokens":2_000_000_000}
    with pytest.raises(ValueError, match="schedule"):
        build_bundle(checkpoint=ck,expected_checkpoint_sha256=sha256_file(ck),
            tokenizer=tok, corpus_manifest=root/"manifest.json",
            output=tmp_path/"transfer", min_train_tokens=4096,min_valid_tokens=4096,
            checkpoint_loader=expired,test_tokenizer_sha256=sha256_file(tok))
    assert not (tmp_path/"transfer").exists()


def test_symlink_or_bad_source_manifest_rejected(tmp_path):
    root = corpus(tmp_path)
    ck, tok = files(tmp_path)
    (root/"train"/"train_00000.bin").unlink()
    (root/"train"/"train_00000.bin").symlink_to(root/"train"/"train_00001.bin")
    with pytest.raises(ValueError):
        build_bundle(checkpoint=ck,expected_checkpoint_sha256=sha256_file(ck),
            tokenizer=tok, corpus_manifest=root/"manifest.json",
            output=tmp_path/"transfer", min_train_tokens=4000,min_valid_tokens=4000,
            checkpoint_loader=lambda p:fake_checkpoint(),test_tokenizer_sha256=sha256_file(tok))


def test_invalid_bundle_receipt_file_path_is_rejected(tmp_path):
    pack(tmp_path)
    out=tmp_path/"transfer"
    path=out/"transfer-receipt.json"
    data=json.loads(path.read_text())
    data["files"][0]["path"]="../secret.env"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="path"):
        verify_bundle(out,test_tokenizer_sha256=sha256_file(out/"inputs"/"tokenizer.json"))


def test_no_false_claim_that_2b_has_been_trained_again(tmp_path):
    report=pack(tmp_path)
    assert report["trained_tokens"]==0
    assert report["training_started"] is False
    assert report["checkpoint_processed_tokens"]==2_073_161_216
    assert report["continue_without_schedule_reset"] is True
    assert report["private_weights_in_bundle"] is True
