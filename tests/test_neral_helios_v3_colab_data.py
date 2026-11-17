"""CPU-only v3 generator contracts; no holdout content printed or copied."""
import json
import ast
from scripts.neral_helios_v3_colab_data import (
    generate, recipe, fingerprint, TRAIN, VAL, _oracle, sha,
)

def test_expected_train_validation_sizes_and_fingerprints():
    train=generate("train")
    val=generate("validation")
    assert len(train)==480
    assert len(val)==60
    assert len({x["id"] for x in train+val})==540
    assert sum(x["family"]=="tool_trace_python" for x in train)==100
    assert sum(x["family"].startswith("rag_style") for x in val)==12
    assert all(x["id"]==fingerprint(x["messages"][1]["content"]) for x in train+val)
    assert all([m["role"] for m in x["messages"]]==["system","user","assistant"] for x in train+val)

def test_trace_proposals_are_wrong_and_repairs_are_self_consistent():
    for tool in ("z3","sympy","networkx","python"):
        for i in (0,19,73):
            row=generate("train")[["z3","sympy","networkx","python"].index(tool)*100+i]
            task=row["source_task"]
            target=json.loads(row["messages"][-1]["content"])
            assert task["tool"]==tool
            assert target["initial_rejected"]["arguments"]!=target["corrected_call"]["arguments"]
            assert target["verified_expected"]==task["result"]
            assert target["final_answer"]==task["result"]

def test_no_sandbox_bypass_or_hidden_eval_case_dependencies():
    import inspect
    from scripts import neral_helios_v3_colab_data as prep
    source=inspect.getsource(prep)
    assert "from scripts.neral_v2_mix_build import protected" in source
    assert "protected_answers_used_for_training" in source
    assert "actual_runtime_tool_call" in source
    assert "model.generate(" not in source
