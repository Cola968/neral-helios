import hashlib
import json
from pathlib import Path

import pytest

from prepare_sft50 import clean_field, normalize, pair_from_row, select_train, split_for, validate_manifest


def _write(root: Path,name: str,rows):
    path=root/name
    path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base():
    return {"license":"TEST","license_url":"https://example.invalid/license","provenance_url":"https://example.invalid/source"}


def test_clean_fields_and_pair_formats():
    assert normalize("  Straße   TEST ")=="strasse test"
    assert clean_field("Explain gravity simply.","prompt")[1] is None
    assert clean_field("A useful answer.","answer")[1] is None
    assert clean_field("mail me at name@example.com","answer")[0] is None
    p,a=pair_from_row({"messages":[{"role":"user","content":"Question here"},{"role":"assistant","content":"Answer here"}]})
    assert (p,a)==("Question here","Answer here")


def test_split_is_stable():
    value=split_for("source","row-17")
    assert value==split_for("source","row-17")
    assert value in {"train","val","test","holdout"}


def test_manifest_requires_exact_hash_and_mix(tmp_path):
    sha_a=_write(tmp_path,"a.jsonl",[{"id":"1","prompt":"Question enough","answer":"Answer"}])
    sha_b=_write(tmp_path,"b.jsonl",[{"id":"2","prompt":"Another question","answer":"Another answer"}])
    base=_base();manifest={"version":1,"sources":[
        base|{"id":"a","file":"a.jsonl","sha256":sha_a,"language":"de","domain":"chat","target_fraction":.4},
        base|{"id":"b","file":"b.jsonl","sha256":sha_b,"language":"en","domain":"science","target_fraction":.6},
    ]}
    sources=validate_manifest(manifest,tmp_path)
    assert [s["id"] for s in sources]==["a","b"]
    broken=json.loads(json.dumps(manifest));broken["sources"][1]["sha256"]="0"*64
    with pytest.raises(ValueError,match="SHA-256 mismatch"):
        validate_manifest(broken,tmp_path)


def test_train_selection_obeys_declared_source_mix():
    sources=[{"id":"a","target_fraction":.25},{"id":"b","target_fraction":.75}]
    rows=[]
    for sid in ("a","b"):
        for i in range(100):
            rows.append({"source_id":sid,"id":str(i),"split":"train","selection_hash":f"{i:064x}"})
    chosen,meta=select_train(rows,sources,40)
    assert len(chosen)==40
    assert meta["a"]["selected_examples"]==10
    assert meta["b"]["selected_examples"]==30


def test_train_selection_fails_instead_of_silent_replay():
    sources=[{"id":"a","target_fraction":1.0}]
    rows=[{"source_id":"a","id":"1","split":"train","selection_hash":"0"*64}]
    with pytest.raises(ValueError,match="only 1 clean train examples"):
        select_train(rows,sources,2)
