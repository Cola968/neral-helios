"""Offline candidate checks: integrity, provenance, version identity and rollback."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import neral_adapter_candidate as adapter
import qwen_local_runtime as runtime


@pytest.fixture
def bundle(tmp_path):
    folder = tmp_path / '0.1.0-pilot.1'
    folder.mkdir()
    config = {'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': 16, 'lora_alpha': 16,
              'lora_dropout': 0, 'bias': 'none', 'target_modules': sorted(adapter.TARGETS)}
    (folder / 'adapter_config.json').write_text(json.dumps(config))
    # Dummy bytes validate integrity only; no test pretends these are real weights.
    (folder / 'adapter_model.safetensors').write_bytes(b'offline fixture, not loadable')
    base = {'revision': 'a'*40, 'sha256': 'b'*64}
    adapter.seal_adapter(folder, base, 'c'*64, '0.1.0-pilot.1')
    return folder, base


def test_valid_bundle_and_immutable_seal(bundle):
    folder, base = bundle
    assert adapter.validate_adapter(folder, base)['product'] == 'NERAL'
    with pytest.raises(FileExistsError):
        adapter.seal_adapter(folder, base, 'c'*64, '0.1.0-pilot.1')


def test_wrong_base_and_tamper_rejected(bundle):
    folder, base = bundle
    with pytest.raises(ValueError, match='base'):
        adapter.validate_adapter(folder, {**base, 'revision': 'd'*40})
    (folder / 'adapter_model.safetensors').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='integrity'):
        adapter.validate_adapter(folder, base)


@pytest.mark.parametrize('extra', ['evil.py', 'adapter_model.bin'])
def test_extra_code_or_pickle_rejected(bundle, extra):
    folder, base = bundle
    (folder / extra).write_text('not allowed')
    with pytest.raises(ValueError, match='Unexpected'):
        adapter.validate_adapter(folder, base)


def test_symlink_rejected(bundle, tmp_path):
    folder, base = bundle
    link = tmp_path / 'linked'
    link.symlink_to(folder, target_is_directory=True)
    with pytest.raises(ValueError, match='directory'):
        adapter.validate_adapter(link, base)


def test_unsupported_config_rejected_even_when_rehashed(bundle):
    folder, base = bundle
    config = folder / 'adapter_config.json'
    row = json.loads(config.read_text())
    row['modules_to_save'] = ['lm_head']
    config.write_text(json.dumps(row))
    manifest = folder / 'neral-adapter.json'
    row = json.loads(manifest.read_text())
    row['files']['adapter_config.json'] = adapter.sha256(config)
    manifest.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='configuration'):
        adapter.validate_adapter(folder, base)


def fake_runtime():
    class Model:
        def eval(self): pass
        def requires_grad_(self, value): assert value is False
    result = SimpleNamespace(identity='qwen:base', model=Model(), closed=False)
    def close():
        result.closed = True
        result.model = None
    result.close = close
    return result


def test_unvalidated_candidate_requires_explicit_opt_in(bundle, monkeypatch):
    folder, base = bundle
    monkeypatch.setattr(runtime, 'validate_hf_snapshot', lambda *a, **k: base)
    monkeypatch.setattr(runtime, 'load', lambda *a: pytest.fail('must reject before loading GPU'))
    with pytest.raises(ValueError, match='opt-in'):
        adapter.load_candidate(folder, folder)


def test_failed_peft_load_discards_mutated_model_and_reloads_base(bundle, monkeypatch):
    folder, base = bundle
    first, fallback = fake_runtime(), fake_runtime()
    loads = iter([first, fallback])
    monkeypatch.setattr(runtime, 'validate_hf_snapshot', lambda *a, **k: base)
    monkeypatch.setattr(runtime, 'load', lambda *a: next(loads))
    def fail(model, *a, **k):
        model.partial_mutation = True
        raise RuntimeError('private local details')
    monkeypatch.setitem(sys.modules, 'peft', SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=fail)))
    result = adapter.load_candidate(folder, folder, allow_unvalidated=True)
    assert first.closed and result is fallback
    assert result.adapter_status == {'active': False, 'fallback': 'fresh-base', 'error_type': 'RuntimeError'}
    assert result.identity == 'qwen:base'


def test_success_changes_context_identity(bundle, monkeypatch):
    folder, base = bundle
    candidate = fake_runtime()
    monkeypatch.setattr(runtime, 'validate_hf_snapshot', lambda *a, **k: base)
    monkeypatch.setattr(runtime, 'load', lambda *a: candidate)
    monkeypatch.setitem(sys.modules, 'peft', SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=lambda model, *a, **k: model)))
    result = adapter.load_candidate(folder, folder, allow_unvalidated=True)
    assert ':neral-lora:0.1.0-pilot.1:' in result.identity
    assert result.adapter_status['active'] is True
