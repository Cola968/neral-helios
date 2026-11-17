"""Regression coverage for crash-safe DEV24 checkpoints and routing budgets."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.neral_v2_eval_checkpoint import prepare, load_completed, save_completed, PROTOCOL
from neral_cpu_foundation.router import choose_route, escalate


def cases():
    return [{"id":f"case-{i}","question_sha256":f"{i:064x}",
             "category":("mathematics","logic","programming","mixed")[i%4],
             "family":"known_public_dev","question":"NO TRAINING","gold_integer":i} for i in range(24)]


def outcome(c):
    return {"id":c["id"],"question_sha256":c["question_sha256"],
            "category":c["category"],"family":c["family"],
            "base_strict_correct":True,"neral_strict_correct":True,
            "independent_verified":True,"base_seconds":1.0,
            "neral_seconds":0.4,"proposal_calls":1,
            "fallback_calls":0,"docker_calls":1}


def test_resume_preserves_case_and_refuses_rewrite(tmp_path):
    cs=cases()
    root=tmp_path/"check"
    folder=prepare(root,label="v1",casepack_sha="a"*64,adapter_sha="b"*64,
                   max_new_tokens=256)
    save_completed(folder,index=0,case=cs[0],result=outcome(cs[0]))
    assert load_completed(folder,cs)=={0:outcome(cs[0])}
    with pytest.raises(FileExistsError):
        save_completed(folder,index=0,case=cs[0],result=outcome(cs[0]))
    assert prepare(root,label="v1",casepack_sha="a"*64,adapter_sha="b"*64,
                   max_new_tokens=256)==folder


def test_refuse_pin_drift_and_tampered_case(tmp_path):
    cs=cases()
    folder=prepare(tmp_path,label="v2full",casepack_sha="a"*64,
                   adapter_sha="b"*64,max_new_tokens=256)
    with pytest.raises(RuntimeError,match="drift"):
        prepare(tmp_path,label="v2full",casepack_sha="a"*64,
                adapter_sha="c"*64,max_new_tokens=256)
    save_completed(folder,index=7,case=cs[7],result=outcome(cs[7]))
    fake=json.loads((folder/"cases"/"07.json").read_text())
    fake["case_id"]="foreign"
    (folder/"cases"/"07.json").write_text(json.dumps(fake))
    with pytest.raises(RuntimeError,match="casepack"):
        load_completed(folder,cs)


def test_refuse_raw_or_gold_in_persisted_case(tmp_path):
    cs=cases()
    folder=prepare(tmp_path,label="v1",casepack_sha="a"*64,
                   adapter_sha="b"*64,max_new_tokens=256)
    with pytest.raises(RuntimeError,match="raw"):
        save_completed(folder,index=0,case=cs[0],
                       result={**outcome(cs[0]),"gold_integer":42})


def test_reject_wrong_token_budget(tmp_path):
    with pytest.raises(ValueError,match="approved"):
        prepare(tmp_path,label="v1",casepack_sha="a"*64,
                adapter_sha="b"*64,max_new_tokens=512)
