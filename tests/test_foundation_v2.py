import json

import pytest

from architectures100_variants import EXPECTED, report
from hf_curriculum50 import route
from hf_source_policy50 import assert_training_allowed
from neral100_checkpoint import init_audit, experimental_50m_migration_plan, make_training_contract
from public_eval_registry50 import validate_registry, assert_never_train
from training_observability import TrainingObserver


def test_100m_contracts_and_init():
    assert report('a1')['parameters'] == EXPECTED['a1'] == 99_783_168
    assert report('a2')['parameters'] == EXPECTED['a2'] == 100_470_912
    assert report('a2')['kv_cache_bf16_4096_bytes'] == 48 * 1024 * 1024
    audit = init_audit('a2')
    assert audit['finite'] and audit['tied_embeddings']


def test_100m_training_contract_is_explicit():
    contract = make_training_contract(
        'a2', seq=2048, microbatch=1, grad_accum=16,
        lr=2e-4, min_lr=2e-5, warmup_fraction=.05,
    )
    assert contract['optimizer'] == 'AdamW'
    assert contract['precision'] == 'bf16-autocast'
    assert contract['foreign_weights'] is False
    with pytest.raises(ValueError):
        make_training_contract('a2', seq=2048, microbatch=1, grad_accum=16, lr=1e-5, min_lr=2e-5, warmup_fraction=.05)


def test_migration_stays_experimental():
    plan = experimental_50m_migration_plan({'d_model': 624}, 'a1')
    assert plan['status'] == 'experimental_only' and plan['production_pretraining'] == 'from_scratch'


def test_curriculum_routes_german_math():
    routed = route(
        ' Die Gleichung und die Ableitung werden Schritt für Schritt hergeleitet. ',
        'HuggingFaceFW/fineweb-2', 'de',
    )
    assert routed['scores']['german'] > 0 and routed['scores']['math'] > 0


def test_public_eval_registry_fail_closed():
    validate_registry()
    for dataset in ('openai/gsm8k', 'google-research-datasets/mbpp', 'deepmind/code_contests'):
        assert assert_never_train(dataset)
        with pytest.raises(ValueError):
            assert_training_allowed(dataset)


def test_observer_always_writes_local_metrics(tmp_path, monkeypatch):
    monkeypatch.delenv('NERAL_TRACKIO_ENABLED', raising=False)
    observer = TrainingObserver(tmp_path, 'test-run', {'variant': 'a2'})
    observer.log(1, {'train_loss': 2.0, 'tokens_per_second': 1234.0})
    observer.alert('test', 'local only', 'INFO', 1)
    observer.finish()
    rows = [json.loads(x) for x in (tmp_path / 'telemetry.jsonl').read_text(encoding='utf8').splitlines()]
    assert any(r.get('event') == 'metrics' and r.get('train_loss') == 2.0 for r in rows)
    assert any(r.get('event') == 'alert' for r in rows)
