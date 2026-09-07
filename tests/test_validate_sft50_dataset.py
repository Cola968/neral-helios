import json
from pathlib import Path

import pytest

from prepare_sft50 import prompt_group_hash, sha256_file
from validate_sft50_dataset import SPLITS, validate


def _row(i,split,lang="de"):
    prompt=f"Question number {i} for split {split} is long enough."
    answer="A complete short answer."
    group=prompt_group_hash(prompt)
    import hashlib
    pair=hashlib.sha256((prompt.lower()+"\0"+answer.lower()).encode()).hexdigest()
    return {"id":f"{split}-{i}","source_id":"test","language":lang,"domain":"chat","prompt":prompt,"answer":answer,"prompt_group_sha256":group,"pair_sha256":pair,"prompt_bytes":len(prompt.encode()),"answer_bytes":len(answer.encode())}


def _dataset(root: Path):
    summaries={}
    for j,split in enumerate(SPLITS):
        rows=[_row(j,split,"de" if j%2==0 else "en")]
        path=root/f"{split}.jsonl"
        path.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf8")
        summaries[split]={"examples":1,"sha256":sha256_file(path),"prompt_groups":1}
    manifest={"version":1,"split_contract_version":2,"splits":summaries}
    (root/"manifest.json").write_text(json.dumps(manifest),encoding="utf8")


def test_valid_dataset_passes_and_reports_composition(tmp_path):
    _dataset(tmp_path)
    audit=validate(tmp_path)
    assert audit["valid"] is True
    assert audit["cross_split_prompt_group_overlap"]==0
    assert audit["composition"]["train"]["rows"]==1


def test_old_split_contract_is_rejected(tmp_path):
    _dataset(tmp_path)
    p=tmp_path/"manifest.json";m=json.loads(p.read_text());m["split_contract_version"]=1;p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match="predates"):
        validate(tmp_path)


def test_prompt_group_hash_tampering_is_rejected(tmp_path):
    _dataset(tmp_path)
    p=tmp_path/"val.jsonl";row=json.loads(p.read_text());row["prompt_group_sha256"]="0"*64;p.write_text(json.dumps(row)+"\n")
    m=json.loads((tmp_path/"manifest.json").read_text());m["splits"]["val"]["sha256"]=sha256_file(p);(tmp_path/"manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError,match="prompt_group_sha256 mismatch"):
        validate(tmp_path)


def test_cross_split_duplicate_prompt_is_rejected_even_with_different_answers(tmp_path):
    _dataset(tmp_path)
    train=json.loads((tmp_path/"train.jsonl").read_text())
    val=json.loads((tmp_path/"val.jsonl").read_text())
    val["prompt"]=train["prompt"];val["prompt_group_sha256"]=train["prompt_group_sha256"]
    # Keep a distinct pair hash to prove the prompt-group rule is stronger than pair dedup.
    import hashlib
    val["pair_sha256"]=hashlib.sha256(b"different-pair").hexdigest()
    p=tmp_path/"val.jsonl";p.write_text(json.dumps(val)+"\n")
    m=json.loads((tmp_path/"manifest.json").read_text());m["splits"]["val"]["sha256"]=sha256_file(p);(tmp_path/"manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError,match="normalized prompt overlap"):
        validate(tmp_path)
