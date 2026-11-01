"""Offline validation of the explicitly gated GPU smoke; never imports torch."""
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def smoke():
    path=Path(__file__).resolve().parents[1]/'scripts'/'qwen25_gpu_smoke.py'
    spec=importlib.util.spec_from_file_location('qwen_smoke_contract',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_default_plan_does_not_initialize_gpu(smoke,monkeypatch,capsys):
    monkeypatch.setattr(sys,'argv',['smoke','--repo-root','.'])
    monkeypatch.setattr(smoke,'execute',lambda *_:pytest.fail('GPU invoked without opt-in'))
    smoke.main()
    assert 'Plan only' in capsys.readouterr().out


def test_code_answer_is_checked_without_execution(smoke):
    assert smoke.validate_add_function('```python\ndef add(a, b):\n    return a + b\n```')
    for text in ['def add(a,b):\n return a-b','def other(a,b):\n return a+b',
                 'raise RuntimeError("must never execute")\ndef add(a,b):\n return a+b']:
        with pytest.raises(ValueError):smoke.validate_add_function(text)


def test_false_provenance_and_truncation_cannot_pass(smoke):
    identity='qwen2.5-coder:'+'a'*64
    result=dict(source='external-model',model_family='qwen2.5-coder',
                base_model='Qwen/Qwen2.5-Coder-7B-Instruct',runtime_id=identity,
                model_generated=True,finish_reason='eos',incomplete=False,cancelled=False)
    smoke.verify_result(result,identity)
    for change in [dict(source='native-model'),dict(runtime_id='stale'),
                   dict(finish_reason='length'),dict(incomplete=True),dict(cancelled=True)]:
        with pytest.raises(ValueError):smoke.verify_result({**result,**change},identity)


def test_device_meter_units_and_scope(smoke,monkeypatch):
    meter=smoke.DeviceMeter()
    def sample(*args,**kwargs):meter.stop.set();return '2048\n'
    monkeypatch.setattr(smoke.subprocess,'check_output',sample)
    meter._sample()
    # The helper's thread was not started in this offline test.
    monkeypatch.setattr(meter.thread,'join',lambda *_:None)
    result=meter.finish()
    assert result['whole_device_sampled_peak_bytes']==2*1024**3
    assert result['whole_device_sample_count']==1
    assert 'other applications' in result['sampling_note']
