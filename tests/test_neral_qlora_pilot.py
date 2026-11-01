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


def test_review_rejects_changed_dataset(pilot, tmp_path):
    root = Path(__file__).resolve().parents[1]
    review = root / 'datasets/neral_qlora_pilot.review.json'
    dataset = root / 'datasets/neral_qlora_pilot.jsonl'
    pilot.verify_review(dataset, review)
    changed = tmp_path / 'changed.jsonl'
    changed.write_bytes(dataset.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='reviewed corpus'):
        pilot.verify_review(changed, review)


class BoundaryTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        if len(messages) == 2:
            return [1, 2, 3]
        return [1, 2, 3, 4, 5]


def test_tokenization_masks_prompt_and_keeps_whole_completion(pilot):
    result = pilot.tokenize_examples([{'prompt': [{}, {}], 'completion': [{}]}], BoundaryTokenizer(), 5)
    assert result == [{'input_ids': [1, 2, 3, 4, 5], 'completion_mask': [0, 0, 0, 1, 1]}]


def test_tokenization_rejects_truncation(pilot):
    with pytest.raises(ValueError, match='no silent truncation'):
        pilot.tokenize_examples([{'prompt': [{}, {}], 'completion': [{}]}], BoundaryTokenizer(), 4)


def test_tokenization_rejects_wrong_template_boundary(pilot):
    class BadTokenizer(BoundaryTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return [7, 8] if len(messages) == 2 else [8, 7, 1]
    with pytest.raises(ValueError, match='boundary'):
        pilot.tokenize_examples([{'prompt': [{}, {}], 'completion': [{}]}], BadTokenizer(), 5)


def test_training_output_rejects_nonempty_and_file(pilot, tmp_path):
    existing = tmp_path / 'occupied'
    existing.write_text('retain me')
    with pytest.raises(ValueError):
        pilot.ensure_output_dir(existing)
    with pytest.raises(ValueError):
        pilot.ensure_output_dir(tmp_path)


def test_failed_gpu_stage_records_error_and_memory_without_training(pilot, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import model_catalog
    root = Path(__file__).resolve().parents[1]
    base = tmp_path / 'base'
    base.mkdir()
    output = tmp_path / 'new-adapter'
    monkeypatch.setattr(model_catalog, 'validate_hf_snapshot', lambda *a, **k: {
        'folder': base, 'base_model': pilot.MODEL_ID, 'revision': 'a'*40, 'sha256': 'b'*64})
    cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
        get_device_name=lambda _: 'mock device, not real GPU evidence',
        mem_get_info=lambda _: (100, 200), reset_peak_memory_stats=lambda _: None,
        max_memory_allocated=lambda _: 70, max_memory_reserved=lambda _: 90)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda='mock')))
    tokenizer = BoundaryTokenizer()
    tokenizer.pad_token = 'PAD'
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        Qwen2TokenizerFast=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer), TrainerCallback=object))
    monkeypatch.setitem(sys.modules, 'neral_training_preflight', SimpleNamespace(check_stack=lambda **k: {'mock': True}))
    monkeypatch.setattr(pilot, 'version', lambda _: 'mock')
    def fail(*args):
        raise RuntimeError('simulated load failure; no training executed')
    monkeypatch.setattr(pilot, '_train', fail)
    args = SimpleNamespace(model_dir=str(base), output_dir=str(output),
        dataset=str(root/'datasets/neral_qlora_pilot.jsonl'),
        review_manifest=str(root/'datasets/neral_qlora_pilot.review.json'), steps=3, max_length=512)
    with pytest.raises(RuntimeError, match='simulated'):
        pilot.run_training(args, [{'prompt': [{}, {}], 'completion': [{}]}])
    report = json.loads((output/'training-report.json').read_text())
    assert report['status'] == 'failed' and report['error_type'] == 'RuntimeError'
    assert report['peak_allocated_bytes'] == 70 and report['peak_reserved_bytes'] == 90
    assert not (output/'neral-adapter.json').exists()
