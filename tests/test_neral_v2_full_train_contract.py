"""No training without same-artifact source digest and independent Docker proof."""
import json
import pytest
from pathlib import Path
from scripts.neral_v2_full_train import load_splits

@pytest.fixture()
def curriculum(tmp_path):
    import os
    d=Path(os.environ.get("NERAL_FULL_DATASET_DIR",
       "/tmp/neral-v2-full-1100-verified-source-20261116-v2"))
    if not d.exists():pytest.skip("Generated training artifacts unavailable in local workspace")
    manifest=json.loads((d/"manifest.json").read_text())
    proof={"status":"python_docker_all_pass","docker_verified":False,
           "python_cases_verified":216,"one_step_ast_repairs_verified":42,
           "train_sha256":manifest["split_sha256"]["train"],
           "val_sha256":manifest["split_sha256"]["validation"]}
    f=tmp_path/"proof.json";f.write_text(json.dumps(proof))
    return d,f,proof

def test_reject_cpu_only_claim_not_real_docker(curriculum):
    d,f,_=curriculum
    with pytest.raises(RuntimeError,match="Docker"):
        load_splits(d/"helios_terra_full.train.jsonl",
                    d/"helios_terra_full.val.jsonl",d/"manifest.json",f,"helios")

def test_reject_wrong_provenance_digest_even_with_docker_flag(curriculum):
    d,f,proof=curriculum
    proof["docker_verified"]=True
    proof["train_sha256"]="0"*64
    f.write_text(json.dumps(proof))
    with pytest.raises(RuntimeError,match="Docker"):
        load_splits(d/"helios_terra_full.train.jsonl",
                    d/"helios_terra_full.val.jsonl",d/"manifest.json",f,"helios")

def test_verified_split_shapes_only_with_matching_docker_receipt(curriculum):
    d,f,proof=curriculum
    proof["docker_verified"]=True
    f.write_text(json.dumps(proof))
    a,b,_=load_splits(d/"helios_terra_full.train.jsonl",
                      d/"helios_terra_full.val.jsonl",d/"manifest.json",f,"helios")
    assert (len(a),len(b))==(1100,140)
    c,e,_=load_splits(d/"luna_v1.train.jsonl",d/"luna_v1.val.jsonl",
                      d/"manifest.json",f,"luna")
    assert (len(c),len(e))==(480,75)
    assert all(x.get("distillation",{}).get("think") is False for x in c+e)
