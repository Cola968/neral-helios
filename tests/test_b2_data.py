import hashlib
import json
from pathlib import Path

import pytest

from prepare_pretrain50_b2 import (
    benchmark_contamination,
    clean_text,
    normalize_for_dedup,
    select_train,
    split_for,
    validate_manifest,
)


def _write_source(root: Path,name: str,rows):
    p=root/name
    p.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf8")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_clean_text_rejects_secrets_and_keeps_prose():
    good=("This is a complete educational paragraph with enough alphabetic content to be useful for a small language model. "*4)
    text,reason=clean_text(good)
    assert reason is None and text
    bad=good+" contact name@example.com for access"
    assert clean_text(bad)[0] is None
    assert normalize_for_dedup("  Straße   TEST ")=="strasse test"


def test_split_is_stable_and_document_grouped():
    a=split_for("source-a","doc-17")
    assert a==split_for("source-a","doc-17")
    assert a in {"train","val","test","holdout"}


def test_manifest_requires_hashes_and_exact_mix(tmp_path):
    sha_a=_write_source(tmp_path,"a.jsonl",[{"id":"1","text":"x"}])
    sha_b=_write_source(tmp_path,"b.jsonl",[{"id":"2","text":"y"}])
    base={"license":"TEST","license_url":"https://example.invalid/license","provenance_url":"https://example.invalid/source"}
    manifest={"version":1,"sources":[
        base|{"id":"a","file":"a.jsonl","sha256":sha_a,"language":"de","domain":"general","target_fraction":.4},
        base|{"id":"b","file":"b.jsonl","sha256":sha_b,"language":"en","domain":"education","target_fraction":.6},
    ]}
    sources=validate_manifest(manifest,tmp_path)
    assert [s["id"] for s in sources]==["a","b"]
    broken=json.loads(json.dumps(manifest));broken["sources"][0]["sha256"]="0"*64
    with pytest.raises(ValueError,match="SHA-256 mismatch"):
        validate_manifest(broken,tmp_path)


def test_frozen_benchmark_contamination_includes_v3_and_v4():
    root=Path(__file__).resolve().parents[1]
    phrases,files=benchmark_contamination(root)
    # Long literals are extracted statically with ast.literal_eval, never by importing the eval files.
    assert normalize_for_dedup("Berlin ist die Hauptstadt von") in phrases
    assert normalize_for_dedup("A black hole has gravity so strong that") in phrases
    assert "benchmark_v3.py" in files and "benchmark_v4.py" in files
    assert all(len(v)==64 for v in files.values())


def test_train_selection_enforces_source_quotas():
    sources=[{"id":"a","target_fraction":.4},{"id":"b","target_fraction":.6}]
    rows=[]
    for sid in ("a","b"):
        for i in range(20):
            rows.append({"source_id":sid,"id":str(i),"split":"train","tokens":100,"selection_hash":f"{i:064x}"})
    selected,meta=select_train(rows,sources,1000)
    assert meta["a"]["selected_tokens"]>=400
    assert meta["b"]["selected_tokens"]>=600
    assert sum(r["tokens"] for r in selected)>=1000


def test_train_selection_fails_instead_of_silent_replay():
    sources=[{"id":"a","target_fraction":1.0}]
    rows=[{"source_id":"a","id":"1","split":"train","tokens":100,"selection_hash":"0"*64}]
    with pytest.raises(ValueError,match="only 100 clean train tokens"):
        select_train(rows,sources,1000)
