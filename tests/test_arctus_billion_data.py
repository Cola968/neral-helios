"""Offline RED→GREEN tests for a future billion-token Arctus *data* pipeline.

No external datasets, downloads, GPU, checkpoints or model training in these tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arctus_billion_data import plan, build_from_streams


def source(name="HuggingFaceFW/fineweb-2"):
    return dict(id="fineweb2-de-general", repo=name, config="deu_Latn",
                revision="a" * 40, language="de", domain="general",
                fraction=1.0, license="ODC-By-1.0")


class ToyTokenizer:
    def encode(self, text):
        return SimpleNamespace(ids=list(range(max(1, len(text.split())))))


def rows(n=32):
    for i in range(n):
        yield {"id": f"row-{i:04d}", "text": (
            f"Ein wissenschaftlicher Fachabsatz Nummer {i} behandelt die Bewegungen "
            "von Himmelskörpern und erläutert die zugrunde liegenden physikalischen "
            "Prinzipien. Wissenschaftliche Erklärungen nutzen überprüfbare Aussagen "
            "und nachvollziehbare Beispiele in ganzen Sätzen. "
        ) * 3, "url": f"https://example.org/docs/{i}"}


def test_plan_is_offline_and_does_not_claim_trained_tokens(tmp_path):
    report = plan(target_train_tokens=1_000_000_000)
    assert report["target_train_tokens"] == 1_000_000_000
    assert report["actual_staged_tokens"] == 0
    assert report["trained_tokens"] == 0
    assert sum(s["target_train_tokens"] for s in report["sources"]) == 1_000_000_000
    assert not list(tmp_path.iterdir())


def test_streamed_shards_are_hashed_and_not_reported_as_training(tmp_path):
    out = tmp_path / "arctus-bulk"
    specs = [source()]
    manifest = build_from_streams(
        sources=specs, output=out, target_train_tokens=350,
        tokenizer=ToyTokenizer(), stream_factory=lambda spec: rows(),
        contamination_phrases=(), max_shard_bytes=1800)
    assert manifest["actual_train_tokens"] >= 350
    assert manifest["trained_tokens"] == 0
    assert manifest["training_started"] is False
    assert manifest["tokenizer_kind"] == "NERAL-16K-BPE"
    assert manifest["sources"][0]["revision"] == "a" * 40
    assert len(manifest["shards"]) >= 2
    assert all(s["sha256"] and (out / s["file"]).is_file() for s in manifest["shards"])
    assert (out / "manifest.json").is_file()
    from arctus_billion_data import sha256_file
    assert all(sha256_file(out / shard["file"]) == shard["sha256"] for shard in manifest["shards"])
    assert sum(s["tokens"] for s in manifest["shards"] if s["split"] == "train") == manifest["actual_train_tokens"]


def test_rejected_duplicates_pii_and_benchmark_text(tmp_path):
    base = (
        "Dies ist ein ausführlicher Lehrtext mit geprüften Sätzen über mathematische "
        "und physikalische Zusammenhänge. " * 6
    )
    def stream(spec):
        yield {"id": "one", "text": base, "url": "https://example.org/one"}
        yield {"id": "duplicate", "text": base, "url": "https://example.org/duplicate"}
        yield {"id": "pii", "text": base + " Kontakt: test@example.org"}
        yield {"id": "contaminated", "text": base + " FROZEN-CASE-NEVER-TRAIN"}
        for i, row in enumerate(rows(18)):
            yield row
    report = build_from_streams(
        sources=[source()], output=tmp_path / "out", target_train_tokens=250,
        tokenizer=ToyTokenizer(), stream_factory=stream,
        contamination_phrases=("frozen-case-never-train",), max_shard_bytes=2000)
    rejected = report["rejected"]
    assert rejected.get("duplicate", 0) >= 1
    assert rejected.get("contact_or_secret_pattern", 0) >= 1
    assert rejected.get("frozen_benchmark_contamination", 0) >= 1


@pytest.mark.parametrize("repo,revision", [
    ("openai/gsm8k", "a"*40),
    ("HuggingFaceTB/finemath", "a"*40),
    ("HuggingFaceFW/fineweb-2", "v2.1.0"),
    ("unknown/unknown", "a"*40),
])
def test_unapproved_or_unpinned_source_fails_closed(tmp_path,repo,revision):
    s = source(repo)
    s["revision"] = revision
    with pytest.raises(ValueError):
        build_from_streams(
            sources=[s], output=tmp_path / "out", target_train_tokens=10,
            tokenizer=ToyTokenizer(), stream_factory=lambda spec: rows(),
            contamination_phrases=())


def test_incomplete_target_does_not_publish_a_manifest(tmp_path):
    out = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="quota"):
        build_from_streams(
            sources=[source()], output=out, target_train_tokens=100_000,
            tokenizer=ToyTokenizer(), stream_factory=lambda spec: rows(2),
            contamination_phrases=())
    assert not (out / "manifest.json").exists()


def test_finalized_output_cannot_be_overwritten(tmp_path):
    out = tmp_path / "out"
    build_from_streams(sources=[source()], output=out,target_train_tokens=30,
                       tokenizer=ToyTokenizer(),stream_factory=lambda spec: rows(),
                       contamination_phrases=())
    with pytest.raises(FileExistsError):
        build_from_streams(sources=[source()], output=out,target_train_tokens=30,
                           tokenizer=ToyTokenizer(),stream_factory=lambda spec: rows(),
                           contamination_phrases=())


def test_historical_b2_b3_document_hashes_are_excluded(tmp_path):
    from arctus_billion_data import sha256_file
    from prepare_pretrain50_b2 import clean_text, normalize_for_dedup
    import hashlib

    first = next(rows(1))
    cleaned, reason = clean_text(first["text"], "general")
    assert reason is None
    digest = hashlib.sha256(normalize_for_dedup(cleaned).encode("utf8")).hexdigest()
    old = tmp_path / "historic-documents.jsonl"
    old.write_text(json.dumps({"source_id": "legacy50", "id": "old-1",
                               "normalized_sha256": digest}) + "\n", encoding="utf8")
    out = tmp_path / "new"
    manifest = build_from_streams(
        sources=[source()], output=out, target_train_tokens=200,
        tokenizer=ToyTokenizer(), stream_factory=lambda spec: rows(30),
        historical_indexes=[old], contamination_phrases=())
    assert manifest["rejected"].get("historical_overlap", 0) == 1
    assert manifest["historical_exclusion_indexes"][0]["sha256"] == sha256_file(old)
    assert not any(digest in (out / s["file"]).read_text(encoding="utf8")
                   for s in manifest["shards"])


def test_invalid_historical_index_does_not_create_final_manifest(tmp_path):
    old = tmp_path / "bad.jsonl"
    old.write_text('{"normalized_sha256": "nope"}\n', encoding="utf8")
    with pytest.raises(ValueError):
        build_from_streams(sources=[source()], output=tmp_path / "out",
                           target_train_tokens=10, tokenizer=ToyTokenizer(),
                           stream_factory=lambda spec: rows(),
                           historical_indexes=[old], contamination_phrases=())
    assert not (tmp_path / "out" / "manifest.json").exists()
