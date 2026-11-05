"""Synthetic contract fixtures only; these are never model quality evidence."""
import copy
import hashlib
import importlib
import json
import subprocess
import sys

import pytest


def inputs():
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    manifest = {'schema': 1, 'benchmark_version': 'external-review-v1',
                'external_unseen_assertion': True, 'independent_reviewer_assertion': True,
                'tasks': [{'id': f'case-{i}', 'semantic_group': f'group-{i}',
                           'prompt_sha256': digest(f'prompt-{i}'),
                           'rubric_sha256': digest(f'rubric-{i}')} for i in range(100)]}
    settings = {'base_revision': 'a'*40, 'quantization': 'nf4', 'seed': 42,
                'decoding': {'temperature': 0, 'do_sample': False},
                'system_prompt': 'Shared instructions', 'budgets': {'tokens': 256, 'seconds': 30},
                'tools': [], 'adapter_sha256': None, 'controller_memory': False}
    results = [{'id': task['id'], 'status': 'fail', 'complete': True,
                'unsupported_claims': 0, 'latency_ms': 100, 'tool_calls': 0,
                'input_tokens': 10, 'output_tokens': 20,
                'resources': {'peak_memory_bytes': 1024}, 'failure_causes': ['reasoning']}
               for task in manifest['tasks']]
    runs = {name: {'settings': copy.deepcopy(settings), 'results': copy.deepcopy(results)}
            for name in 'ABCD'}
    for name in 'BCD':
        runs[name]['settings']['tools'] = ['solve.v1']
    for name in 'CD':
        runs[name]['settings']['adapter_sha256'] = 'b'*64
    runs['D']['settings']['controller_memory'] = True
    return manifest, runs


def evaluate(manifest, runs):
    # A missing implementation is a clear feature assertion, not collection failure.
    try:
        module = importlib.import_module('neral_control.evaluation')
    except ModuleNotFoundError:
        pytest.fail('Missing four-condition capability evaluation implementation')
    return module.evaluate(manifest, runs)


def full(row):
    row.update(status='full', failure_causes=[])


def test_exact_twenty_percent_excludes_already_solved_base_cases():
    manifest, runs = inputs()
    for i in range(20):
        full(runs['A']['results'][i])
    for i in range(36):
        full(runs['D']['results'][i])
    report = evaluate(manifest, runs)
    assert report['target']['numerator'] == 16
    assert report['target']['denominator'] == 80
    assert report['target']['achieved'] is True
    assert report['target']['rate'] == .2
    assert report['benchmark_version'] == manifest['benchmark_version']
    assert report['release_authorized'] is False
    assert report['independent_quality_proven'] is False


def test_condition_attribution_regressions_resources_and_remedies():
    manifest, runs = inputs()
    full(runs['A']['results'][0])
    full(runs['B']['results'][1])
    full(runs['C']['results'][2])
    full(runs['D']['results'][3])
    report = evaluate(manifest, runs)
    assert report['comparisons']['B-A']['gained'] == ['case-1']
    assert report['comparisons']['B-A']['regressed'] == ['case-0']
    assert report['comparisons']['C-B']['attribution'] == 'adapter'
    assert report['comparisons']['D-C']['attribution'] == 'orchestration_and_memory'
    assert report['conditions']['D']['full'] == 1
    assert report['conditions']['D']['fail'] == 99
    assert report['conditions']['D']['metrics']['latency_ms']['total'] == 10000
    assert report['conditions']['D']['metrics']['resources']['peak_memory_bytes']['max'] == 1024
    assert report['conditions']['D']['failure_causes']['reasoning']['count'] == 99
    assert report['conditions']['D']['failure_causes']['reasoning']['remedy']


def test_zero_denominator_undefined_and_no_unearned_full_credit():
    manifest, runs = inputs()
    for row in runs['A']['results']:
        full(row)
    full(runs['D']['results'][0])
    runs['D']['results'][0]['complete'] = False
    full(runs['D']['results'][1])
    runs['D']['results'][1]['unsupported_claims'] = 1
    report = evaluate(manifest, runs)
    assert report['target'] == {'numerator': 0, 'denominator': 0, 'rate': None, 'achieved': False}
    assert report['conditions']['D']['full'] == 0
    assert report['conditions']['D']['partial'] == 2
    assert report['conditions']['D']['unsupported_claims'] == 1


@pytest.mark.parametrize('field,value', [
    ('base_revision', 'c'*40), ('quantization', 'fp16'), ('seed', 2),
    ('decoding', {'do_sample': True}), ('system_prompt', 'Unequal'),
    ('budgets', {'tokens': 512}), ('tools', ['different']),
])
def test_unfair_abc_settings_rejected(field, value):
    manifest, runs = inputs()
    runs['C']['settings'][field] = value
    with pytest.raises(ValueError):
        evaluate(manifest, runs)


@pytest.mark.parametrize('mutation', [
    'few', 'group', 'prompt', 'id', 'version', 'unseen', 'reviewer', 'digest',
    'missing', 'duplicate', 'unknown', 'bool_score', 'incomplete_bool', 'nan',
    'negative', 'cause', 'base_adapter', 'adapter_missing', 'adapter_mismatch',
    'a_tools', 'b_memory', 'condition',
])
def test_invalid_or_untraceable_import_rejected(mutation):
    manifest, runs = inputs()
    row = runs['D']['results'][0]
    if mutation == 'few': manifest['tasks'].pop()
    elif mutation in ('group', 'prompt', 'id'):
        key = {'group': 'semantic_group', 'prompt': 'prompt_sha256', 'id': 'id'}[mutation]
        manifest['tasks'][1][key] = manifest['tasks'][0][key]
    elif mutation == 'version': manifest['benchmark_version'] = ''
    elif mutation == 'unseen': manifest['external_unseen_assertion'] = False
    elif mutation == 'reviewer': manifest['independent_reviewer_assertion'] = False
    elif mutation == 'digest': manifest['tasks'][0]['rubric_sha256'] = 'invalid'
    elif mutation == 'missing': runs['D']['results'].pop()
    elif mutation == 'duplicate': runs['D']['results'][1] = copy.deepcopy(row)
    elif mutation == 'unknown': row['id'] = 'not-a-case'
    elif mutation == 'bool_score': row['status'] = True
    elif mutation == 'incomplete_bool': row['complete'] = 1
    elif mutation == 'nan': row['latency_ms'] = float('nan')
    elif mutation == 'negative': row['output_tokens'] = -1
    elif mutation == 'cause': row['failure_causes'] = ['mystery']
    elif mutation == 'base_adapter': runs['A']['settings']['adapter_sha256'] = 'b'*64
    elif mutation == 'adapter_missing': runs['C']['settings']['adapter_sha256'] = None
    elif mutation == 'adapter_mismatch': runs['D']['settings']['adapter_sha256'] = 'c'*64
    elif mutation == 'a_tools': runs['A']['settings']['tools'] = ['solve.v1']
    elif mutation == 'b_memory': runs['B']['settings']['controller_memory'] = True
    elif mutation == 'condition': del runs['B']
    with pytest.raises(ValueError):
        evaluate(manifest, runs)


def test_json_cli_imports_without_models_or_writing_outputs(tmp_path):
    manifest, runs = inputs()
    mf, rf = tmp_path/'manifest.json', tmp_path/'runs.json'
    mf.write_text(json.dumps(manifest))
    rf.write_text(json.dumps(runs))
    result = subprocess.run([sys.executable, '-m', 'neral_control.evaluation',
                             '--manifest', str(mf), '--runs', str(rf)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['target']['denominator'] == 100
    assert sorted(p.name for p in tmp_path.iterdir()) == ['manifest.json', 'runs.json']


@pytest.mark.parametrize('condition,field,value', [
    ('D', 'base_revision', 'c'*40), ('D', 'decoding', {'temperature': 1}),
    ('D', 'budgets', {'tokens': 999}), ('D', 'tools', ['other.v1']),
    ('D', 'controller_memory', False), ('C', 'controller_memory', True),
])
def test_d_cannot_hide_unequal_core_settings(condition, field, value):
    manifest, runs = inputs()
    runs[condition]['settings'][field] = value
    with pytest.raises(ValueError):
        evaluate(manifest, runs)


def test_threshold_below_twenty_percent_is_not_rounded_up():
    manifest, runs = inputs()
    for row in runs['D']['results'][:19]:
        full(row)
    report = evaluate(manifest, runs)
    assert report['target'] == {'numerator': 19, 'denominator': 100, 'rate': .19, 'achieved': False}
    assert report['assertions']['independently_verified'] is False


def test_partial_score_and_partial_regression_are_reported():
    manifest, runs = inputs()
    runs['A']['results'][0]['status'] = 'partial'
    runs['B']['results'][1]['status'] = 'partial'
    report = evaluate(manifest, runs)
    assert report['conditions']['A']['partial'] == 1
    assert report['comparisons']['B-A']['regressed'] == ['case-0']
    assert report['comparisons']['B-A']['gained'] == ['case-1']


def test_unmeasured_resources_are_not_reported_as_zero():
    manifest, runs = inputs()
    for row in runs['D']['results'][1:]:
        row['resources'] = {}
    report = evaluate(manifest, runs)
    resource = report['conditions']['D']['metrics']['resources']['peak_memory_bytes']
    assert resource['measured_cases'] == 1
    assert resource['mean'] == 1024


def test_cli_rejects_duplicate_json_keys_without_revealing_input(tmp_path):
    mf, rf = tmp_path/'manifest.json', tmp_path/'runs.json'
    mf.write_text('{"secret":"private-value","secret":"private-value"}')
    rf.write_text('{}')
    result = subprocess.run([sys.executable, '-m', 'neral_control.evaluation',
                             '--manifest', str(mf), '--runs', str(rf)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert not result.stdout
    assert 'private-value' not in result.stderr
