import collections
import hashlib
import json
from pathlib import Path

import pytest

from fetch_oasst1_sft50 import choose_pairs
from prepare_sft50 import (
    audit_prompt_group_splits,
    auto_train_target,
    clean_field,
    load_rows,
    normalize,
    pair_from_row,
    prompt_group_hash,
    select_train,
    split_for,
    split_for_prompt,
    validate_manifest,
)


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


def test_legacy_split_is_stable():
    value=split_for("source","row-17")
    assert value==split_for("source","row-17")
    assert value in {"train","val","test","holdout"}


def test_prompt_group_split_ignores_source_row_and_formatting_variants():
    a="  Warum ist der Himmel blau?  "
    b="WARUM   IST DER HIMMEL BLAU?"
    assert prompt_group_hash(a)==prompt_group_hash(b)
    assert split_for_prompt(a)==split_for_prompt(b)


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


def test_same_normalized_question_with_different_answers_stays_in_one_split(tmp_path):
    rows_a=[{"id":"a1","prompt":"Warum ist Wasser nass?","answer":"Eine kurze Antwort über Wasser."}]
    rows_b=[{"id":"b1","prompt":"  WARUM   IST WASSER NASS? ","answer":"Eine andere, ebenfalls vollständige Antwort."}]
    sha_a=_write(tmp_path,"a.jsonl",rows_a);sha_b=_write(tmp_path,"b.jsonl",rows_b);base=_base()
    manifest={"version":1,"sources":[
        base|{"id":"a","file":"a.jsonl","sha256":sha_a,"language":"de","domain":"chat","target_fraction":.5},
        base|{"id":"b","file":"b.jsonl","sha256":sha_b,"language":"de","domain":"chat","target_fraction":.5},
    ]}
    sources=validate_manifest(manifest,tmp_path)
    loaded=load_rows(sources,[],collections.Counter(),max_seq=512)
    assert len(loaded)==2
    assert loaded[0]["prompt_group_sha256"]==loaded[1]["prompt_group_sha256"]
    assert loaded[0]["split"]==loaded[1]["split"]
    audit=audit_prompt_group_splits(loaded)
    assert audit["cross_split_prompt_group_overlap"]==0


def test_overlap_audit_rejects_manually_cross_split_group():
    group=prompt_group_hash("This question is sufficiently long.")
    rows=[
        {"prompt_group_sha256":group,"split":"train"},
        {"prompt_group_sha256":group,"split":"val"},
    ]
    with pytest.raises(ValueError,match="prompt leakage"):
        audit_prompt_group_splits(rows)


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


def test_auto_target_is_limited_by_scarcest_source_after_filtering():
    sources=[{"id":"de","target_fraction":.25},{"id":"en","target_fraction":.75}]
    rows=[]
    rows += [{"source_id":"de","split":"train"} for _ in range(1200)]
    rows += [{"source_id":"en","split":"train"} for _ in range(5000)]
    target=auto_train_target(rows,sources,recommended=10000,cap=24000)
    assert 4700 <= target <= 4800
    chosen=[]
    for sid,count in (("de",1200),("en",5000)):
        for i in range(count):chosen.append({"source_id":sid,"id":str(i),"split":"train","selection_hash":f"{i:064x}"})
    selected,_=select_train(chosen,sources,target)
    assert len(selected)==target


def test_oasst_pair_selection_requires_positive_review_and_best_rank():
    prompt={"message_id":"p","parent_id":None,"text":"Explain gravity clearly please.","role":"prompter","lang":"en","review_result":True,"deleted":False,"synthetic":False}
    good1={"message_id":"a1","parent_id":"p","text":"Gravity attracts masses.","role":"assistant","lang":"en","review_result":True,"deleted":False,"synthetic":False,"rank":1}
    good0={"message_id":"a0","parent_id":"p","text":"Gravity is an interaction between masses.","role":"assistant","lang":"en","review_result":True,"deleted":False,"synthetic":False,"rank":0}
    rejected={"message_id":"bad","parent_id":"p","text":"Bad answer", "role":"assistant","lang":"en","review_result":False,"deleted":False,"synthetic":False,"rank":-1}
    pairs=choose_pairs([prompt,good1,good0,rejected],10,10)
    assert len(pairs["en"])==1
    assert pairs["en"][0]["id"]=="a0"
