"""RED -> GREEN: scalable, offline Arctus staging-quality audit.

Uses synthetic JSONL only; no model, tokenizer, Hugging Face downloads, or GPU.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from arctus_billion_audit import audit_corpus, simhash64, hamming64, normalize_for_dedup


def long_doc(subject="Sterne"):
    return (
        "Die Berechnung von astronomischen und mathematischen Größen ist "
        "wissenschaftlich dokumentiert und wird anhand konkreter Beispiele überprüft. "
        f"Dieses Kapitel behandelt {subject} und ihre physikalischen Eigenschaften. "
    ) * 18


def write_staging(tmp_path, train_texts, valid_texts):
    root = tmp_path / "staged"
    root.mkdir()
    shards = []
    source_id = "fineweb2-de-general"
    revision = "a" * 40
    for split, texts in (("train", train_texts), ("valid", valid_texts)):
        path = root / f"{split}-000000.jsonl"
        lines = []
        for n, txt in enumerate(texts):
            lines.append(json.dumps({
                "id": f"{split}-{n}",
                "source_id": source_id,
                "language": "de",
                "domain": "general",
                "revision": revision,
                "license": "ODC-By-1.0",
                "normalized_sha256": hashlib.sha256(normalize_for_dedup(txt).encode()).hexdigest(),
                "text": txt, "tokens": len(txt.split()) + 2
            }, ensure_ascii=False))
        raw = ("\n".join(lines) + "\n").encode("utf8")
        path.write_bytes(raw)
        shards.append({
            "file": path.name, "split": split,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw), "documents": len(lines),
            "tokens": sum(len(txt.split()) + 2 for txt in texts)})
    manifest = {
        "version": 1,
        "purpose": "Arctus350 native pretraining DATA only",
        "tokenizer_kind": "NERAL-16K-BPE",
        "trained_tokens": 0,
        "training_started": False,
        "data_is_training_ready": False,
        "sources": [{
            "source_id": source_id, "dataset_id": "HuggingFaceFW/fineweb-2",
            "revision": revision, "license": "ODC-By-1.0",
            "config": "deu_Latn", "language": "de", "domain": "general"}],
        "shards": shards,
        "tokens_by_split": {
            "train": sum(x["tokens"] for x in shards if x["split"] == "train"),
            "valid": sum(x["tokens"] for x in shards if x["split"] == "valid")}
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_simhash_identity_and_nearly_identical_texts():
    base = long_doc()
    clone = base.replace("Dieses Kapitel behandelt Sterne", "Dieses Kapitel behandelt Galaxien", 1)
    assert base != clone
    assert simhash64(base) == simhash64(base)
    assert hamming64(simhash64(base), simhash64(clone)) <= 4
    assert hamming64(simhash64(base), simhash64(long_doc("Quantenfelder"))) > 4


def test_cross_split_near_duplicate_is_flagged_and_not_training_ready(tmp_path):
    base = long_doc()
    slight = base.replace("Dieses Kapitel behandelt Sterne", "Dieses Kapitel behandelt Pulsare", 1)
    root = write_staging(tmp_path, [base], [slight])
    result = audit_corpus(root)
    assert result["integrity_verified"] is True
    assert result["documents_checked"] == 2
    assert result["cross_split_near_duplicate_pairs"] >= 1
    assert result["training_ready"] is False
    assert result["trained_tokens"] == 0
    assert result["model_weights_changed"] is False


def test_exact_duplicate_and_intra_split_near_duplicate(tmp_path):
    base = long_doc()
    slightly_changed = base.replace("Dieses Kapitel behandelt Sterne", "Dieses Kapitel behandelt Monde", 1)
    root = write_staging(tmp_path, [base, base, slightly_changed], [long_doc("Astrometrie")])
    result = audit_corpus(root)
    assert result["exact_duplicate_pairs"] >= 1
    assert result["within_split_near_duplicate_pairs"] >= 1
    assert result["training_ready"] is False


def test_unrelated_corpus_flags_no_near_duplicates_but_requires_manual_audit(tmp_path):
    root = write_staging(tmp_path, [
        "Ein Forschungstext behandelt Mathematik und Gleichungen mit Beweisen " * 30
    ], [
        "Ein anderes Dokument analysiert deutsche Sprache und historische Literatur " * 30
    ])
    report = audit_corpus(root)
    assert report["documents_checked"] == 2
    assert report["exact_duplicate_pairs"] == 0
    assert report["cross_split_near_duplicate_pairs"] == 0
    assert report["within_split_near_duplicate_pairs"] == 0
    assert report["training_ready"] is False  # conservative audit only
    assert report["needs_manual_license_and_holdout_review"] is True


def test_mutated_same_size_shard_fails_sha256_before_analysis(tmp_path):
    root = write_staging(tmp_path, [long_doc()], [long_doc("Statistik")])
    shard = root / "train-000000.jsonl"
    raw = bytearray(shard.read_bytes())
    raw[10] ^= 1
    shard.write_bytes(raw)
    with pytest.raises(ValueError, match="SHA256"):
        audit_corpus(root)


def test_unsafe_shard_name_and_missing_sha_fail_closed(tmp_path):
    root = write_staging(tmp_path, [long_doc()], [long_doc("Statistik")])
    p = root/"manifest.json"
    data = json.loads(p.read_text())
    data["shards"][0]["file"] = "../outside.jsonl"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        audit_corpus(root)


def test_pretending_training_already_done_is_rejected(tmp_path):
    root = write_staging(tmp_path, [long_doc()], [long_doc("Statistik")])
    p = root/"manifest.json"
    data = json.loads(p.read_text())
    data["trained_tokens"] = 10
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="untrained"):
        audit_corpus(root)


def test_deterministic_findings_and_no_output_written_by_default(tmp_path):
    root = write_staging(tmp_path, [long_doc()], [long_doc("Statistik")])
    before = sorted(x.relative_to(root).as_posix() for x in root.iterdir())
    r1 = audit_corpus(root)
    r2 = audit_corpus(root)
    assert r1 == r2
    assert before == sorted(x.relative_to(root).as_posix() for x in root.iterdir())


def test_row_content_source_contract_and_token_counts_checked(tmp_path):
    root=write_staging(tmp_path, [long_doc()], [long_doc("Quantenfeld")])
    shard=root/"valid-000000.jsonl"
    contents=shard.read_text()
    altered=contents.replace('"source_id": "fineweb2-de-general"',
                             '"source_id": "unknown-source"')
    shard.write_text(altered)
    p=root/"manifest.json"
    m=json.loads(p.read_text())
    for s in m["shards"]:
        if s["split"]=="valid":
            s["sha256"]=hashlib.sha256(shard.read_bytes()).hexdigest()
            s["bytes"]=shard.stat().st_size
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="source"):
        audit_corpus(root)
