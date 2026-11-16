"""Fail-closed final metrics validation without materializing 40/24 test cases."""
import json
import pytest
from pathlib import Path

from scripts.neral_v2_final_eval40 import validate_frozen,pairwise
from scripts.neral_v2_dialogue_rag_sample import casepack,score


def test_40_refuses_unfrozen_adapter_before_opening_test(tmp_path):
    fake=tmp_path/"adapter";fake.mkdir()
    v1=tmp_path/"v1";v1.mkdir()
    with pytest.raises(FileNotFoundError):
        validate_frozen(fake,v1,tmp_path/"out.json")


def test_40_checks_full_rank16_and_adapter_training_receipt(tmp_path):
    full=tmp_path/"full";full.mkdir()
    v1=tmp_path/"v1";v1.mkdir()
    (full/"manifest.json").write_text(json.dumps({
      "status":"frozen_adapter_ready_for_heldout_comparison",
      "rank":8,"family":"helios","train_rows":1100,
      "val_rows":140,"test_rows_opened":0,"epochs":2,
      "selected_epoch":2,"max_steps":0
    }))
    with pytest.raises(RuntimeError,match="Full 9B rank16"):
        validate_frozen(full,v1,tmp_path/"scores.json")


def test_regression_counts_are_per_identical_ordered_cases():
    a={"correct_bitset":"1"+"0"*39,"correct":1}
    b={"correct_bitset":"0"+"1"*39,"correct":39}
    assert pairwise(a,b)=={"gained":39,"regressed":1,"net":38}


def test_dialogue_rag_casebook_is_new_and_proxy_scores():
    rows=casepack()
    assert len(rows)==20 and {r["kind"] for r in rows}=={
      "rag_present","rag_missing","simple_calc","clarification"}
    yes=next(x for x in rows if x["kind"]=="rag_present")
    missing=next(x for x in rows if x["kind"]=="rag_missing")
    assert score(yes,"Das Prüfintervall beträgt "+yes["must"][0]+" Tage. "+yes["must"][1])
    assert score(missing,"Nicht in den vorliegenden Quellen enthalten.")
    assert not score(missing,"Die Farbe ist schwarz.")
