"""Offline RED/GREEN tests for an explicitly gated NERAL-4B adaptation pipeline."""
import json
from pathlib import Path

import pytest

from neral4b_training import (
    TrainingGateError, dataset_contract, prepare_dataset,
    build_training_plan, authorize_training,
)

REV = "179dd21fc55192153d94adb0e0ce8f69e222bf75"


def sample(i, *, source="OpenAssistant/oasst2", split="train"):
    return {
        "id": f"{split}-{i}", "prompt": f"Bitte erkläre, warum ein Flugzeug {i} steigt.",
        "answer": f"Das Flugzeug {i} steigt, wenn es entsprechend gesteuert wird.",
        "source": source,
        "license": "Apache-2.0",
        "revision": REV,
        "reviewed": True,
        "language": "de",
        "domain": "general",
    }


def write(root, train, val):
    root.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        (root / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
    return root


def test_valid_contract_and_independent_fingerprints(tmp_path):
    data = write(tmp_path / "dataset", [sample(1), sample(2)], [sample(3, split="val")])
    contract = dataset_contract(data)
    assert contract["train_examples"] == 2
    assert contract["validation_examples"] == 1
    assert len(contract["contract_sha256"]) == 64
    prepared = prepare_dataset(data, tmp_path / "prepared")
    assert prepared["model"] == "Qwen/Qwen3.5-4B"
    row = json.loads((tmp_path / "prepared/train.jsonl").read_text().splitlines()[0])
    assert row["prompt"][0]["role"] == "system"
    assert row["prompt"][-1]["role"] == "user"
    assert row["completion"] == [{"role": "assistant", "content": sample(1)["answer"]}]
    assert "Qwen3.5" in row["prompt"][0]["content"]
    assert prepared["contract_sha256"] == contract["contract_sha256"]


def test_data_order_does_not_change_contract_or_converted_artifacts(tmp_path):
    first = write(tmp_path / "a", [sample(2), sample(1)], [sample(3, split="val")])
    second = write(tmp_path / "b", [sample(1), sample(2)], [sample(3, split="val")])
    assert dataset_contract(first)["contract_sha256"] == dataset_contract(second)["contract_sha256"]
    prepare_dataset(first, tmp_path / "out1")
    prepare_dataset(second, tmp_path / "out2")
    assert (tmp_path / "out1/train.jsonl").read_bytes() == (tmp_path / "out2/train.jsonl").read_bytes()


@pytest.mark.parametrize("mutate", [
    lambda x: x.update({"reviewed": False}),
    lambda x: x.update({"source": "random/web"}),
    lambda x: x.update({"license": "unknown"}),
    lambda x: x.update({"revision": "main"}),
    lambda x: x.update({"answer": " "}),
    lambda x: x.update({"role": "system"}),
    lambda x: x.update({"id": "../../break"}),
])
def test_unapproved_or_malformed_examples_fail_closed(tmp_path, mutate):
    row = sample(1)
    mutate(row)
    data = write(tmp_path / "data", [row], [sample(2, split="val")])
    with pytest.raises(ValueError):
        dataset_contract(data)


def test_split_contamination_rejected_even_if_answer_differs(tmp_path):
    a = sample(1)
    b = sample(8, split="val")
    b["prompt"] = a["prompt"].upper()
    data = write(tmp_path / "data", [a], [b])
    with pytest.raises(ValueError, match="overlap"):
        dataset_contract(data)


def test_duplicate_ids_and_prompts_rejected(tmp_path):
    data = write(tmp_path / "data", [sample(1), sample(1)], [sample(2, split="val")])
    with pytest.raises(ValueError, match="duplicate"):
        dataset_contract(data)


def test_training_needs_volume_and_two_part_approval(tmp_path, monkeypatch):
    data = write(tmp_path / "data", [sample(1)], [sample(2, split="val")])
    plan = build_training_plan(data)
    assert plan["training_eligible"] is False
    monkeypatch.setenv("NERAL_4B_TRAIN_APPROVED", "YES")
    with pytest.raises(TrainingGateError, match="dataset"):
        authorize_training(plan, execute=True)
    assert authorize_training(plan, execute=False) is False


def test_mature_dataset_still_requires_explicit_execution_and_environment(tmp_path, monkeypatch):
    data = write(
        tmp_path / "data",
        [sample(i) for i in range(100)],
        [sample(i, split="val") for i in range(100, 120)],
    )
    plan = build_training_plan(data)
    assert plan["training_eligible"] is True
    assert authorize_training(plan, execute=False) is False
    monkeypatch.delenv("NERAL_4B_TRAIN_APPROVED", raising=False)
    with pytest.raises(TrainingGateError, match="approval"):
        authorize_training(plan, execute=True)
    monkeypatch.setenv("NERAL_4B_TRAIN_APPROVED", "YES")
    assert authorize_training(plan, execute=True) is True


def test_strict_limits_and_no_symlink_tricks(tmp_path):
    data = write(tmp_path / "data", [sample(1)], [sample(2, split="val")])
    obj = sample(3)
    obj["answer"] = "A" * 7000
    with (data / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj) + "\n")
    with pytest.raises(ValueError):
        dataset_contract(data)
