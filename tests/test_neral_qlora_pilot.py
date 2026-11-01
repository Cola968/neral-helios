"""Pure offline contract tests; never loads CUDA or installs dependencies."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest


@pytest.fixture
def pilot():
    path = Path(__file__).resolve().parents[1] / "scripts" / "neral_qlora_pilot.py"
    spec = importlib.util.spec_from_file_location("neral_qlora_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def example():
    return {"messages": [
        {"role": "system", "content": "Du bist NERAL. Antworte präzise."},
        {"role": "user", "content": "Was ist SOLVE?"},
        {"role": "assistant", "content": "SOLVE ist ein deterministischer Rechenmodus."},
    ]}


def test_valid_dataset_maps_to_completion_only(pilot, tmp_path):
    target = tmp_path / "good.jsonl"
    target.write_text(json.dumps(example(), ensure_ascii=False), encoding="utf-8")
    rows = pilot.read_examples(target)
    assert [m["role"] for m in rows[0]["prompt"]] == ["system", "user"]
    assert [m["role"] for m in rows[0]["completion"]] == ["assistant"]


@pytest.mark.parametrize("mutation", [
    lambda row: row["messages"][2].update(content="<think>private reasoning</think>"),
    lambda row: row["messages"][2].update(content="ghp_" + "x" * 30),
    lambda row: row["messages"][1].update(role="assistant"),
    lambda row: row.update(extra="unexpected"),
])
def test_invalid_training_examples_are_rejected(pilot, tmp_path, mutation):
    row = example(); mutation(row)
    target = tmp_path / "bad.jsonl"
    target.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError):
        pilot.read_examples(target)


def test_default_is_no_gpu_no_training(pilot, tmp_path, monkeypatch, capsys):
    target = tmp_path / "sample.jsonl"
    target.write_text(json.dumps(example()), encoding="utf-8")
    monkeypatch.setattr(pilot, "run_training", lambda *_: pytest.fail("GPU path called"))
    pilot.main(["--dataset", str(target)])
    assert "PLAN ONLY" in capsys.readouterr().out
    assert "torch" not in sys.modules or not pilot.__dict__.get("torch")


def test_training_output_rejects_repository(pilot):
    repo = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError):
        pilot.ensure_output_dir(repo / "weights")
