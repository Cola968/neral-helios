"""Read-only Arctus350 corpus inventory checks. Synthetic fixtures only.

A metadata pass never certifies training readiness or dataset provenance. Do
not hash multi-GB shards or deserialize private checkpoints during this probe.
"""
import json
from pathlib import Path

import pytest

from frontier350.readonly_corpus_inventory import inspect_corpus, inspect_candidates
from frontier350.chat350_contract import EXPECTED_TOKENIZER_SHA256


def corpus(tmp_path, *, token_sha=EXPECTED_TOKENIZER_SHA256):
    root=tmp_path/"private-corpus"
    for name in ("train","valid"):
        folder=root/name
        folder.mkdir(parents=True)
        (folder/f"{name}_00000.bin").write_bytes(b"\x01\x00"*2048)
    manifest={
        "schema":2,"architecture":"frontier350-v1",
        "tokenizer":{"tokenizer_sha256":token_sha},
        "benchmark_content_included":False,
        "train_files":[{"path":"train_00000.bin","sha256":"a"*64,"tokens":2048}],
        "valid_files":[{"path":"valid_00000.bin","sha256":"b"*64,"tokens":2048}]
    }
    (root/"manifest.json").write_text(json.dumps(manifest),encoding="utf8")
    return root


def test_metadata_probe_never_claims_provenance_or_sha_verified(tmp_path):
    root=corpus(tmp_path)
    result=inspect_corpus(root)
    assert result=={
        "present":True,"format_supported":True,
        "metadata_compatible":True,
        "train_shards":1,"valid_shards":1,
        "train_tokens_declared":2048,"valid_tokens_declared":2048,
        "sha256_verified":False,"corpus_provenance_verified":False,
        "training_ready":False,"issue":None,
    }
    assert (root/"train"/"train_00000.bin").read_bytes()==b"\x01\x00"*2048


def test_no_private_paths_or_raw_text_in_report(tmp_path):
    root=corpus(tmp_path)
    result=inspect_candidates([root,tmp_path/"missing",root])
    assert len(result["candidates"])==3
    assert result["training_started"] is False
    assert result["private_data_uploaded"] is False
    report=json.dumps(result)
    assert "private-corpus" not in report
    assert str(tmp_path) not in report
    assert "train_00000.bin" not in report
    assert result["candidates"][1]["present"] is False


def test_mismatch_tokenizer_and_bad_schema_fail_closed(tmp_path):
    root=corpus(tmp_path,token_sha="0"*64)
    assert inspect_corpus(root)["metadata_compatible"] is False
    assert inspect_corpus(root)["issue"]=="identity_mismatch"
    manifest=root/"manifest.json"
    data=json.loads(manifest.read_text())
    data.update({"schema":1,"benchmark_content_included":True})
    manifest.write_text(json.dumps(data))
    assert inspect_corpus(root)["metadata_compatible"] is False


@pytest.mark.parametrize("rel",["../secrets.bin","X:/malicious.bin","mixed/name.bin","a\\b.bin"])
def test_manifest_shard_path_cannot_escape_root(tmp_path,rel):
    root=corpus(tmp_path)
    manifest=root/"manifest.json"
    data=json.loads(manifest.read_text())
    data["train_files"][0]["path"]=rel
    manifest.write_text(json.dumps(data))
    result=inspect_corpus(root)
    assert result["metadata_compatible"] is False
    assert result["training_ready"] is False


def test_missing_or_truncated_shard_cannot_pass_metadata_check(tmp_path):
    root=corpus(tmp_path)
    file=root/"train"/"train_00000.bin"
    file.write_bytes(b"\x00\x00"*1024)
    out=inspect_corpus(root)
    assert out["metadata_compatible"] is False
    assert out["issue"]=="shard_size_mismatch"


def test_symlinked_manifests_and_shards_rejected(tmp_path):
    root=corpus(tmp_path)
    destination=tmp_path/"external.bin"
    destination.write_bytes(b"\x01\x00"*2048)
    shard=root/"train"/"train_00000.bin"
    shard.unlink()
    try:
        shard.symlink_to(destination)
    except (OSError,NotImplementedError):
        pytest.skip("Symlinks unavailable on this OS")
    out=inspect_corpus(root)
    assert out["metadata_compatible"] is False


def test_oversized_and_invalid_manifest_fails_without_raw_data(tmp_path):
    root=corpus(tmp_path)
    manifest=root/"manifest.json"
    manifest.write_text("{broken",encoding="utf8")
    assert inspect_corpus(root)["metadata_compatible"] is False
    manifest.write_bytes(b"x"*(2*1024*1024+1))
    assert inspect_corpus(root)["metadata_compatible"] is False


def test_empty_or_extra_shards_not_ready(tmp_path):
    root=corpus(tmp_path)
    (root/"train"/"extra.bin").write_bytes(b"\x00\x00"*2048)
    report=inspect_corpus(root)
    assert report["metadata_compatible"] is False
    assert report["issue"]=="inventory_mismatch"


def test_cross_platform_candidate_results_are_bounded(tmp_path):
    root=corpus(tmp_path)
    with pytest.raises(ValueError):
        inspect_candidates([root]*6)
