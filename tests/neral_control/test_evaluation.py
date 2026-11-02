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
    settings = {'base_model_id':'Qwen/Qwen2.5-Coder-7B-Instruct',
                'base_revision': 'a'*40, 'quantization': 'nf4', 'seed': 42,
                'decoding': {'temperature': 0, 'do_sample': False},
                'system_prompt': 'Shared instructions', 'budgets': {'tokens': 256, 'seconds': 30},
                'tools': [], 'adapter_sha256': None, 'controller_memory': False,
                'lora_adapter_id':None, 'lora_weights_version':None, 'adapter_hash':None,
                'adapter_integrity_verified':False, 'controller_active':False,
                'context_control_active':False}
    results = [{'id': task['id'], 'status': 'fail', 'complete': True,
                'unsupported_claims': 0, 'latency_ms': 100, 'tool_calls': 0,
                'input_tokens': 10, 'output_tokens': 20,
                'resources': {'peak_memory_bytes': 1024}, 'failure_causes': ['reasoning'],
                'completion_status':'complete','used_tools':[], 'enabled_tools':[],
                'tool_provenance':[]}
               for task in manifest['tasks']]
    runs = {name: {'settings': copy.deepcopy(settings), 'results': copy.deepcopy(results)}
            for name in 'ABCD'}
    for name in 'BCD':
        runs[name]['settings']['tools'] = ['solve.v1']
        for row in runs[name]['results']:row['enabled_tools']=['solve.v1']
    for name in 'CD':
        runs[name]['settings']['adapter_sha256'] = 'b'*64
        runs[name]['settings'].update(lora_adapter_id='neral-fixture',lora_weights_version='1.0.0',
                                     adapter_hash='b'*64,adapter_integrity_verified=True)
    runs['D']['settings']['controller_memory'] = True
    runs['D']['settings'].update(controller_active=True,context_control_active=True)
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


@pytest.mark.parametrize('condition,field,value', [
    ('B','lora_adapter_id','hidden'), ('B','lora_weights_version','1.0.0'),
    ('B','adapter_integrity_verified',True), ('C','lora_adapter_id',None),
    ('C','lora_weights_version',None), ('C','adapter_hash',None),
    ('C','adapter_integrity_verified',False), ('D','controller_active',False),
    ('D','context_control_active',False)])
def test_explicit_adapter_and_controller_contracts(condition, field, value):
    manifest, runs = inputs()
    runs[condition]['settings'][field] = value
    with pytest.raises(ValueError):
        evaluate(manifest, runs)


def test_legacy_hash_only_adapter_is_rejected():
    manifest, runs = inputs()
    # A hash string alone never establishes adapter activation or integrity.
    for run in runs.values():
        for key in ('lora_adapter_id','lora_weights_version','adapter_hash',
                    'adapter_integrity_verified','controller_active','context_control_active'):
            del run['settings'][key]
    with pytest.raises(ValueError):
        evaluate(manifest, runs)


@pytest.mark.parametrize('field,value', [
    ('used_tools',['solve.v1']), ('enabled_tools',['solve.v1']),
    ('tool_provenance',[{'tool_id':'solve.v1','call_index':0,'status':'ok'}])])
def test_zero_call_baseline_cannot_hide_tool_metadata(field, value):
    manifest, runs = inputs()
    runs['A']['results'][0][field] = value
    with pytest.raises(ValueError):evaluate(manifest,runs)


def test_incomplete_status_cannot_receive_full_credit():
    manifest, runs = inputs()
    full(runs['D']['results'][0])
    runs['D']['results'][0]['completion_status']='timed_out'
    assert evaluate(manifest,runs)['conditions']['D']['full']==0


def test_reported_tool_usage_must_match_enabled_tools_and_count():
    manifest,runs=inputs()
    row=runs['B']['results'][0]
    row.update(tool_calls=1,used_tools=['solve.v1'],
               tool_provenance=[{'tool_id':'solve.v1','call_index':0,'status':'ok'}])
    assert evaluate(manifest,runs)['conditions']['B']['metrics']['tool_calls']['total']==1
    row['used_tools']=[]
    with pytest.raises(ValueError,match='provenance'):evaluate(manifest,runs)


def test_missing_token_measurements_stay_unknown():
    manifest,runs=inputs()
    for row in runs['D']['results']:row.update(input_tokens=None,output_tokens=None)
    metrics=evaluate(manifest,runs)['conditions']['D']['metrics']
    assert metrics['output_tokens']=={'total':None,'mean':None,'max':None,'measured_cases':0}


@pytest.mark.parametrize('field,value', [
    ('lora_adapter_id','other'), ('lora_weights_version','2.0.0')])
def test_cd_adapter_identity_must_match_beyond_hash(field,value):
    manifest,runs=inputs()
    runs['D']['settings'][field]=value
    with pytest.raises(ValueError,match='identities'):evaluate(manifest,runs)


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


@pytest.mark.parametrize('count', [1, 2, 100])
def test_baseline_results_cannot_claim_any_tool_execution(count):
    manifest, runs = inputs()
    runs['A']['results'][0]['tool_calls'] = count
    with pytest.raises(ValueError, match='A.*tool'):
        evaluate(manifest, runs)


@pytest.mark.parametrize('tool', ['calculator', 'solve.v0', 'solve.v1.extra', 'solve.v1\n'])
def test_toolset_requires_explicit_positive_version(tool):
    manifest, runs = inputs()
    for condition in 'BCD':
        runs[condition]['settings']['tools'] = [tool]
    with pytest.raises(ValueError):
        evaluate(manifest, runs)


def schema2_inputs():
    manifest, runs = inputs()
    manifest.update(schema=2, split='dev', external_unseen_assertion=False,
                    independent_reviewer_assertion=False)
    for index, task in enumerate(manifest['tasks']):
        task['semantic_family'] = task.pop('semantic_group')
        task.update(category='coding' if index < 50 else 'logic',
                    task_version='1.0.0',
                    source={'kind':'original', 'reference':'fixture-author/task-'+str(index),
                            'license':'CC0-1.0'},
                    review={'status':'reviewed', 'reviewer':'fixture-reviewer',
                            'method':'hand-checked deterministic rubric'})
    return manifest, runs


def test_schema2_reuses_categories_and_reports_coverage_without_independence_claim():
    manifest, runs = schema2_inputs()
    # A semantic family may contain several tasks inside a single partition.
    manifest['tasks'][1]['semantic_family'] = manifest['tasks'][0]['semantic_family']
    report = evaluate(manifest, runs)
    assert report['manifest_schema'] == 2
    assert report['split'] == 'dev'
    assert report['category_counts'] == {'coding':50, 'logic':50}
    assert report['assertions']['external_unseen'] is False
    assert report['assertions']['independent_reviewer'] is False
    assert report['independent_quality_proven'] is False


@pytest.mark.parametrize('field,value', [
    ('id','case-1'), ('prompt_sha256','bad'), ('category',''),
    ('semantic_family',''), ('task_version',''),
    ('source',{'kind':'original', 'reference':'', 'license':'CC0-1.0'}),
    ('review',{'status':'draft', 'reviewer':'fixture', 'method':'unchecked'}),
])
def test_schema2_requires_unique_tasks_and_reviewed_provenance(field, value):
    manifest, runs = schema2_inputs()
    manifest['tasks'][0][field] = value
    with pytest.raises(ValueError):
        evaluate(manifest, runs)


def test_schema2_final_requires_external_unseen_and_independent_assertions():
    manifest, runs = schema2_inputs()
    manifest['split'] = 'final'
    with pytest.raises(ValueError):
        evaluate(manifest, runs)
    manifest.update(external_unseen_assertion=True, independent_reviewer_assertion=True)
    assert evaluate(manifest, runs)['split'] == 'final'


def test_versioned_migration_requires_review_metadata_and_preserves_commitments():
    module = importlib.import_module('neral_control.evaluation')
    assert callable(getattr(module, 'migrate_manifest_v1', None)), 'Explicit schema migration is missing'
    old, _ = inputs()
    original = copy.deepcopy(old)
    modern, _ = schema2_inputs()
    metadata = {task['id']: {k:task[k] for k in ('category','task_version','source','review')}
                for task in modern['tasks']}
    result = module.migrate_manifest_v1(old, benchmark_version='dev-migrated-v2',
                                       split='dev', task_metadata=metadata)
    assert old == original
    assert result['schema'] == 2 and result['benchmark_version'] == 'dev-migrated-v2'
    assert result['tasks'][0]['semantic_family'] == 'group-0'
    assert result['tasks'][0]['prompt_sha256'] == old['tasks'][0]['prompt_sha256']
    assert result['tasks'][0]['rubric_sha256'] == old['tasks'][0]['rubric_sha256']
    with pytest.raises(ValueError):
        module.migrate_manifest_v1(old, benchmark_version='dev-migrated-v2',
                                  split='dev', task_metadata={})


def test_declared_semantic_family_or_prompt_cannot_cross_partitions():
    module = importlib.import_module('neral_control.evaluation')
    assert callable(getattr(module, 'validate_partition_separation', None)), 'Partition gate is missing'
    dev, _ = schema2_inputs()
    final = copy.deepcopy(dev)
    final.update(split='final', benchmark_version='final-v1',
                 external_unseen_assertion=True, independent_reviewer_assertion=True)
    with pytest.raises(ValueError, match='partition'):
        module.validate_partition_separation([dev, final])
    for i, task in enumerate(final['tasks']):
        task['semantic_family'] = 'independent-'+str(i)
    # Distinct family labels cannot hide an identical committed prompt.
    with pytest.raises(ValueError, match='partition'):
        module.validate_partition_separation([dev, final])
    for i, task in enumerate(final['tasks']):
        task['prompt_sha256'] = hashlib.sha256(('new prompt '+str(i)).encode()).hexdigest()
    result = module.validate_partition_separation([dev, final])
    assert result['checked_partitions'] == ['dev', 'final']
    assert result['semantic_leakage_proven_absent'] is False


@pytest.mark.parametrize('value', [[], {}, ['dev'], {'split':'dev'}, 2, None])
def test_malformed_partition_types_fail_with_controlled_validation(value):
    manifest, runs = schema2_inputs()
    manifest['split'] = value
    with pytest.raises(ValueError):
        evaluate(manifest, runs)


@pytest.mark.parametrize('value', [[], {}, ['reviewed'], {'status':'reviewed'}])
def test_malformed_review_status_types_fail_with_controlled_validation(value):
    manifest, runs = schema2_inputs()
    manifest['tasks'][0]['review']['status'] = value
    with pytest.raises(ValueError):
        evaluate(manifest, runs)
