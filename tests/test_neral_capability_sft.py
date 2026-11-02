"""Inert validation entrypoint; no model/training fixtures."""
import builtins
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]


def api():
    path=ROOT/'scripts/validate_neral_capability_sft.py'
    assert path.is_file(),'Missing dedicated capability-SFT validation entrypoint'
    spec=importlib.util.spec_from_file_location('capability_validator',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_default_validation_never_imports_model_stack(monkeypatch):
    original=builtins.__import__
    def guarded(name,*a,**k):
        if name.split('.')[0] in {'torch','transformers','peft','unsloth','bitsandbytes'}:
            pytest.fail('Default validation imported the model stack')
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',guarded)
    result=api().validate_data(ROOT)
    assert (result['new_sft_examples'],result['total_sft_examples'])==(60,86)
    assert result['stage']=='DATA_VALIDATION_ONLY' and result['training_executed'] is False
    assert result['tokenizer_provenance'] is None and result['model_weights_loaded'] is False


def test_invalid_tokenizer_snapshot_is_rejected_before_transformers(monkeypatch,tmp_path):
    original=builtins.__import__
    def guarded(name,*a,**k):
        if name.split('.')[0] in {'transformers','torch'}:pytest.fail('Unverified tokenizer reached model stack')
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises((ValueError,OSError)):
        api().validate_data(ROOT,tokenizer_dir=tmp_path)


@pytest.mark.parametrize('flag',['--train','--epochs','--confirm-training'])
def test_training_options_are_not_supported(flag):
    with pytest.raises(SystemExit) as e:api().main([flag])
    assert e.value.code==2
