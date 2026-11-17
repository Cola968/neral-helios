"""Safety and source provenance for the balanced 1100+140 curriculum."""
import ast
import json
import pytest
from pathlib import Path
from scripts.neral_v2_mix_build import (
    TRAIN_COUNTS, VAL_COUNTS, SEED, protected, split_validation_and_training,
    distill, core11, grounding, normalize, digest, compact
)

ROOT=Path(__file__).resolve().parents[1]


def test_protected_32_40_24_source_gates_are_pinned():
    prompts,tasks=protected(ROOT)
    assert len(prompts)==96
    assert len(tasks)==72


def test_one_step_ast_repair_has_one_syntax_change_and_untrusted_docker_status():
    rows=core11(55,SEED+101)
    repairs=[r for r in rows if r["evidence"].get("ast_1step_repair")]
    assert repairs
    for row in repairs:
        task=row["source_task"]
        assert task["original_broken"]
        with pytest.raises(SyntaxError):ast.parse(task["original_broken"])
        ast.parse(task["code"])
        assert task["code"]!=task["original_broken"]
        assert row["evidence"]["docker_verified"] is False


def test_sqlite_synthetic_grounding_and_honest_missing_source():
    rows=grounding(21,SEED+505)
    assert len(rows)==21
    assert all(r["evidence"].get("synthetic") is True for r in rows)
    assert any("Nicht in den vorliegenden Quellen enthalten" in r["messages"][-1]["content"] for r in rows)
    assert all("[testdoc-" in r["messages"][1]["content"] or "[Profil:" in r["messages"][2]["content"]
               or "synthetisch" in r["messages"][1]["content"].lower() for r in rows)


def test_full_curriculum_exact_balance_and_protected_split(tmp_path):
    train,val=split_validation_and_training(ROOT)
    assert len(train)==1100 and len(val)==140
    for group,n in TRAIN_COUNTS.items():
        assert sum(x["family"]==group for x in train)==n
    for group,n in VAL_COUNTS.items():
        assert sum(x["family"]==group for x in val)==n
    protected_prompts,protected_tasks=protected(ROOT)
    train_ids={x["id"] for x in train}
    val_ids={x["id"] for x in val}
    assert len(train_ids)==1100 and len(val_ids)==140
    assert not(train_ids & val_ids)
    assert not((train_ids|val_ids) & protected_prompts)
    train_tasks={digest(compact(x["source_task"])) for x in train if "source_task" in x}
    val_tasks={digest(compact(x["source_task"])) for x in val if "source_task" in x}
    assert not(train_tasks & val_tasks)
    assert not((train_tasks|val_tasks)&protected_tasks)
    assert len(distill(train))==480
    assert all(x["evidence"].get("method")!="synthetic_style" or
               x["evidence"].get("requires_human_review") is True for x in train)
