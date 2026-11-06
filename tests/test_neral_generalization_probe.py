"""New internal development probe: no GPU inference, code execution or network."""
import json
from pathlib import Path
import pytest
from scripts.neral_generalization_probe import CASES, case_ids, score, grade


ROOT=Path(__file__).resolve().parents[1]


def test_probe_contains_32_unique_project_internal_cases():
    assert len(CASES) == 32
    assert len(case_ids()) == 32
    assert {case["family"] for case in CASES} == {"math","sql","coding_ast","agent"}
    for family in ("math","sql","coding_ast","agent"):
        assert sum(case["family"] == family for case in CASES) == 8
    prompts=[x["prompt"] for x in CASES]
    assert len(set(prompts)) == len(prompts)
    assert all(len(x["prompt"]) <= 800 for x in CASES)


def test_no_identical_prompt_in_any_synthetic_corpus_split():
    records=[]
    root=ROOT/"datasets"/"neral_curriculum_v1"
    for split in ("train","val","test"):
        for line in (root/(split+".jsonl")).read_text(encoding="utf-8").splitlines():
            records.append(json.loads(line)["messages"][1]["content"].casefold().strip())
    assert not set(x["prompt"].casefold().strip() for x in CASES).intersection(records)


@pytest.mark.parametrize("raw,ok",[
    ("-3/4",True),("-0.75",True),("x = -0.75",True),
    ("0.75",False),("Answer: -3/4",False),("import os",False),
])
def test_numeric_scoring_is_exact_and_refuses_long_explanations(raw,ok):
    case={"id":"num-1","family":"math","prompt":"Calculate.","expected":"-3/4"}
    assert grade(case,raw)["correct"] is ok


@pytest.mark.parametrize("raw,ok",[
    ("LEFT JOIN",True),("left join",True),(" INNER   JOIN ",False),
    ("SELECT ... LEFT JOIN tbl",False),("DROP TABLE tbl;",False),
])
def test_sql_keyword_grade_requires_only_clause(raw,ok):
    case={"id":"sql-1","family":"sql","prompt":"Choose SQL operation.","expected":"LEFT JOIN"}
    assert grade(case,raw)["correct"] is ok


def test_code_ast_grade_never_executes_generated_code(tmp_path):
    case={"id":"code-1","family":"coding_ast","prompt":"Fix this function.",
          "expected":"def mult(a,b):\n    return a*b\n"}
    good="def mult(a,b):\n    return a * b\n"
    bad="def mult(a,b):\n    return a+b\n"
    assert grade(case,good)["correct"] is True
    assert grade(case,bad)["correct"] is False
    assert grade(case,"import os\ndef mult(a,b):\n    return a*b\n")["correct"] is False
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("raw,ok",[
    ('{"action":"read"}',True),
    ('{"action":"READ"}',False),
    ('{"action":"read","path":".env"}',False),
    ('{"tool":"read"}',False),
    ('{"action":"shell"}',False),
    ('{"action":"read"}\n{"action":"finish"}',False),
])
def test_tool_decision_requires_one_exact_json_command(raw,ok):
    case={"id":"agent-1","family":"agent","prompt":"Pick the tool.","expected":"read"}
    assert grade(case,raw)["correct"] is ok


def test_scoring_requires_full_exact_case_set_and_is_honest():
    responses={case["id"]:(json.dumps({"action":case["expected"]}) if case["family"]=="agent"
                            else case["expected"]) for case in CASES}
    perfect=score(responses)
    assert perfect["total"] == 32 and perfect["correct"] == 32
    assert perfect["by_family"]["math"]["correct"] == 8
    assert perfect["independent_external_holdout"] is False
    assert perfect["model_quality_improvement_proven"] is False
    assert perfect["training_performed"] is False
    with pytest.raises(ValueError):
        score({key:value for key,value in responses.items() if key != CASES[0]["id"]})
    with pytest.raises(ValueError):
        score(responses | {"extra":"junk"})

def test_pairwise_comparison_without_generated_text_leakage():
    from scripts.neral_generalization_probe import compare_responses
    base={c["id"]:"incorrect" for c in CASES}
    adapter={c["id"]:(json.dumps({"action":c["expected"]}) if c["family"]=="agent"
                       else c["expected"]) for c in CASES}
    row=compare_responses(base,adapter)
    assert row["base_correct"] == 0
    assert row["adapter_correct"] == 32
    assert row["delta_adapter_vs_base"] == 32
    assert row["model_quality_improvement_proven"] is False
    assert row["training_performed"] is False
    assert "incorrect" not in str(row)
    assert "def multiply" not in str(row)
