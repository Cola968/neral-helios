"""Import already scored outputs for a four-condition external capability review.

No model, network, training, benchmark creation or dataset writes occur here.
Python API: ``evaluate(manifest, runs)``. CLI accepts two existing JSON files
and prints the report: ``python -m neral_control.evaluation --manifest FILE
--runs FILE``. Supply actual independently reviewed outputs, not synthetic
fixtures. Input assertions cannot establish that outputs were really generated,
that reviewers were independent, or that tasks were previously unseen.

Legacy manifest contract: {schema: 1, benchmark_version: nonempty string,
external_unseen_assertion: true, independent_reviewer_assertion: true,
tasks: [{id, semantic_group, prompt_sha256, rubric_sha256}, ...]}.
At least 100 unique IDs, semantic groups and prompt digests are required.
Digests identify externally frozen tasks/rubrics; no task text is synthesized.

Schema 2 adds split (train/dev/final) and replaces semantic_group with reusable
category and semantic_family. Every task also needs task_version, source exactly
{kind, reference, license}, and review exactly {status: "reviewed", reviewer,
method}. IDs and prompt digests stay unique; categories/families may repeat inside
a partition. DEV can declare false unseen/independent assertions; final requires
both true. The assertions are reported honestly and never verified by this code.
Training partitions are accepted for separation checks, not benchmark scoring.
Explicit migrate_manifest_v1() requires caller-supplied reviewed metadata and a
new benchmark version; it preserves task IDs and prompt/rubric commitments.
validate_partition_separation() checks declared families and prompt hashes across
partitions. It cannot establish semantic uniqueness or review independence.

Runs is exactly {A: run, B: run, C: run, D: run}. Each run contains settings
and results. Settings keys: base_revision (40 hex), quantization (string), seed
(integer), decoding (nonempty JSON object), system_prompt (string), budgets
(nonempty JSON object of positive numeric per-call limits), tools (unique
versioned tool IDs), adapter_sha256 (null or 64 hex), controller_memory (bool).
Comparison contract 2 additionally requires base_model_id, lora_adapter_id,
lora_weights_version, adapter_hash, adapter_integrity_verified, controller_active
and context_control_active. Hash-only legacy settings are rejected, not upgraded
with invented integrity assertions. validate_condition_settings() checks a single
available condition without fabricating missing C/D runs.
All four share base/revision, decoding, quantization, seed, prompt and per-call
budgets. A has no tools/adapter/memory; B has tools only; C adds the adapter;
D uses the same tools/adapter and enables controller/memory. These comparisons
attribute B-A to tools, C-B to the adapter, D-C to orchestration plus memory.
They do not identify memory and orchestration effects separately.

Every result has exactly: id, status ('full'|'partial'|'fail'), complete (bool),
unsupported_claims (nonnegative integer), latency_ms (finite nonnegative number),
tool_calls/input_tokens/output_tokens (nonnegative integers), resources (object
of named finite nonnegative measurements with units encoded in field names),
failure_causes (unique list from REMEDIES), completion_status, used_tools,
enabled_tools, and tool_provenance. Each tool provenance row is exactly
{tool_id, call_index, status}; zero-based indices and counts must agree.
Token counts can be null when unmeasured; aggregate measured_cases exposes gaps.
Every task must appear exactly once
in every condition; failed attempts must be included. Incomplete or unsupported
answers cannot receive full credit. Full credit is a reviewer assertion, never
an automatically verified correctness claim. Resources are reported only when
measured; absent resource fields are not replaced with invented zeros.

The 20% target is exactly count(D full AND A not full)/count(A not full),
with absolute numerator/denominator and benchmark version. Zero denominator
is undefined, not achieved. The target is descriptive, not release permission,
statistical significance or independent quality proof. Regressions are retained.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


REMEDIES = {
    'knowledge': 'Review missing facts and licensed training or retrieval evidence.',
    'reasoning': 'Inspect failed steps and add independently verified reasoning cases.',
    'tool': 'Check tool selection, argument schema, permissions and tool results.',
    'context': 'Inspect context selection, truncation and memory relevance.',
    'format': 'Check output schema and instruction adherence.',
    'code': 'Review static findings and independently authorized code tests.',
    'incomplete': 'Inspect cancellation, output limits and completion handling.',
    'hallucination': 'Require source support and abstention for unsupported claims.',
    'runtime': 'Inspect runtime errors, resource limits and provider availability.',
}
CORE = {'base_model_id', 'base_revision', 'quantization', 'seed', 'decoding', 'system_prompt', 'budgets'}
SETTINGS = CORE | {'tools', 'adapter_sha256', 'controller_memory', 'lora_adapter_id',
                   'lora_weights_version','adapter_hash','adapter_integrity_verified',
                   'controller_active','context_control_active'}
RESULT = {'id', 'status', 'complete', 'unsupported_claims', 'latency_ms',
          'tool_calls', 'input_tokens', 'output_tokens', 'resources', 'failure_causes',
          'completion_status','used_tools','enabled_tools','tool_provenance'}
HEX64 = re.compile(r'[a-f0-9]{64}\Z')


def _text(value, limit=8192):
    return isinstance(value, str) and 0 < len(value.strip()) <= limit


def _number(value, *, positive=False, integer=False):
    return (type(value) is int if integer else type(value) in (int, float)) and (
        math.isfinite(value) and (value > 0 if positive else value >= 0))


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError('Expected finite JSON metadata.') from exc


def _provenance(task):
    source, review = task['source'], task['review']
    return (isinstance(source, dict) and set(source) == {'kind', 'reference', 'license'}
            and all(_text(source[key], 2000) for key in source)
            and isinstance(review, dict) and set(review) == {'status', 'reviewer', 'method'}
            and review['status'] == 'reviewed'
            and _text(review['reviewer'], 200) and _text(review['method'], 2000))


def _manifest(manifest):
    common = {'schema', 'benchmark_version', 'external_unseen_assertion',
              'independent_reviewer_assertion', 'tasks'}
    if not isinstance(manifest, dict) or type(manifest.get('schema')) is not int:
        raise ValueError('Invalid manifest envelope.')
    schema = manifest['schema']
    if schema not in (1, 2) or set(manifest) != common | ({'split'} if schema == 2 else set()):
        raise ValueError('Unsupported manifest schema or fields.')
    if (not _text(manifest['benchmark_version'], 200)
            or type(manifest['external_unseen_assertion']) is not bool
            or type(manifest['independent_reviewer_assertion']) is not bool
            or not isinstance(manifest['tasks'], list)
            or not 100 <= len(manifest['tasks']) <= 10000):
        raise ValueError('Benchmark requires version, review assertions and 100..10000 tasks.')
    split = manifest.get('split', 'final')
    if not isinstance(split, str) or split not in {'train', 'dev', 'final'}:
        raise ValueError('Unknown benchmark partition.')
    if (schema == 1 or split == 'final') and not (
            manifest['external_unseen_assertion'] and manifest['independent_reviewer_assertion']):
        raise ValueError('Final review requires unseen and independent reviewer assertions.')
    # Categories and families are deliberately reusable inside one partition.
    # Prompt hashes and IDs remain unique. Family metadata is not semantic proof.
    seen = {key: set() for key in ('id', 'prompt_sha256')}
    if schema == 1:
        seen['semantic_group'] = set()
    legacy_fields = {'id', 'semantic_group', 'prompt_sha256', 'rubric_sha256'}
    modern_fields = {'id', 'category', 'semantic_family', 'prompt_sha256',
                     'rubric_sha256', 'task_version', 'source', 'review'}
    for task in manifest['tasks']:
        if (not isinstance(task, dict) or set(task) != (legacy_fields if schema == 1 else modern_fields)
                or not _text(task['id'], 200)
                or any(not isinstance(task[key], str) or not HEX64.fullmatch(task[key])
                       for key in ('prompt_sha256', 'rubric_sha256'))):
            raise ValueError('Invalid frozen benchmark task.')
        if schema == 1:
            if not _text(task['semantic_group'], 200):
                raise ValueError('Invalid legacy semantic group.')
        elif (any(not _text(task[key], 200) for key in ('category', 'semantic_family', 'task_version'))
              or not _provenance(task)):
            raise ValueError('Schema 2 tasks require category, family, version and reviewed provenance.')
        for key, values in seen.items():
            if task[key] in values:
                raise ValueError('Duplicate task, semantic group or prompt digest.')
            values.add(task[key])
    _canonical(manifest)
    return seen['id']


def migrate_manifest_v1(manifest, *, benchmark_version, split, task_metadata):
    """Pure explicit migration. Never invent category, authorship or review facts.

    ``task_metadata`` maps every legacy task ID to exactly category, task_version,
    source and review. Semantic groups become families; commitments are preserved.
    The caller supplies a distinct benchmark version and partition deliberately.
    """
    expected = _manifest(manifest)
    if (manifest['schema'] != 1 or benchmark_version == manifest['benchmark_version']
            or not isinstance(task_metadata, dict) or set(task_metadata) != expected):
        raise ValueError('Migration requires schema 1, a new version and metadata for every task.')
    metadata_fields = {'category', 'task_version', 'source', 'review'}
    tasks = []
    for old in manifest['tasks']:
        extra = task_metadata[old['id']]
        if not isinstance(extra, dict) or set(extra) != metadata_fields:
            raise ValueError('Migration needs explicitly reviewed task metadata.')
        tasks.append({'id': old['id'], 'semantic_family': old['semantic_group'],
                      'prompt_sha256': old['prompt_sha256'], 'rubric_sha256': old['rubric_sha256'],
                      **deepcopy(extra)})
    result = {**deepcopy(manifest), 'schema': 2, 'benchmark_version': benchmark_version,
              'split': split, 'tasks': tasks}
    _manifest(result)
    return result


def validate_partition_separation(manifests):
    """Reject declared family/prompt overlap across train/dev/final partitions.

    This metadata gate does not see sealed prompt contents and cannot prove that
    two independently named families are not paraphrases or semantically related.
    """
    if not isinstance(manifests, (list, tuple)) or not 1 <= len(manifests) <= 100:
        raise ValueError('Expected bounded partition manifests.')
    seen_families, seen_prompts, partitions = {}, {}, set()
    for manifest in manifests:
        _manifest(manifest)
        if manifest['schema'] != 2:
            raise ValueError('Partition checking requires migrated schema 2.')
        split = manifest['split']
        partitions.add(split)
        for task in manifest['tasks']:
            for mapping, key in ((seen_families, 'semantic_family'), (seen_prompts, 'prompt_sha256')):
                previous = mapping.setdefault(task[key], split)
                if previous != split:
                    raise ValueError('Declared family or prompt crosses a partition boundary.')
    return {'checked_partitions': sorted(partitions),
            'semantic_leakage_proven_absent': False}


def validate_condition_settings(name, setting):
    """Validate one explicit condition; execution/artifact assertions need external proof."""
    if name not in ('A','B','C','D') or not isinstance(setting, dict) or set(setting) != SETTINGS:
        raise ValueError('Missing or unknown comparison setting.')
    if (not _text(setting['base_model_id'], 300)
            or not isinstance(setting['base_revision'], str)
            or not re.fullmatch('[a-f0-9]{40}', setting['base_revision'])
            or not _text(setting['quantization'], 200)
            or type(setting['seed']) is not int or not _text(setting['system_prompt'])
            or not isinstance(setting['decoding'], dict) or not setting['decoding']
            or not isinstance(setting['budgets'], dict) or not setting['budgets']
            or any(not _text(k,100) or not _number(v,positive=True) for k,v in setting['budgets'].items())
            or not isinstance(setting['tools'],list)
            or any(not isinstance(t,str) or not re.fullmatch(r'[a-z][a-z0-9_.-]{0,150}\.v[1-9][0-9]*(?:\.[0-9]+){0,2}',t) for t in setting['tools'])
            or len(set(setting['tools'])) != len(setting['tools'])
            or any(type(setting[k]) is not bool for k in ('controller_memory','controller_active',
                'context_control_active','adapter_integrity_verified'))):
        raise ValueError('Invalid run settings or budgets.')
    _canonical(setting)
    if name in 'AB':
        if (any(setting[k] is not None for k in ('adapter_sha256','adapter_hash','lora_adapter_id','lora_weights_version'))
                or setting['adapter_integrity_verified']):
            raise ValueError('Base conditions cannot include an adapter.')
    elif (not _text(setting['lora_adapter_id'],300)
            or not isinstance(setting['lora_weights_version'],str)
            or not re.fullmatch(r'\d+\.\d+\.\d+(?:-pilot\.\d+)?',setting['lora_weights_version'])
            or not isinstance(setting['adapter_hash'],str) or not HEX64.fullmatch(setting['adapter_hash'])
            or setting['adapter_sha256'] != setting['adapter_hash']
            or setting['adapter_integrity_verified'] is not True):
        raise ValueError('unverified_or_missing_adapter')
    if any(setting[k] != (name=='D') for k in ('controller_memory','controller_active','context_control_active')):
        raise ValueError('Only D enables the reasoning controller and context control.')
    if (name=='A' and setting['tools']) or (name!='A' and not setting['tools']):
        raise ValueError('A has no tools; B/C/D require tools.')


def _settings(runs):
    if not isinstance(runs, dict) or set(runs) != set('ABCD'):
        raise ValueError('Exactly four conditions A/B/C/D are required.')
    for name, run in runs.items():
        if not isinstance(run, dict) or set(run) != {'settings', 'results'}:
            raise ValueError('Invalid condition envelope.')
        setting = run['settings']
        validate_condition_settings(name, setting)
    base = runs['A']['settings']
    for name in 'BCD':
        if any(_canonical(runs[name]['settings'][key]) != _canonical(base[key]) for key in CORE):
            raise ValueError('Unequal base, decoding, seed, prompt or per-call budget.')
    if base['tools'] or not runs['B']['settings']['tools']:
        raise ValueError('A has no tools; B must supply the declared shared toolset.')
    if any(set(runs[name]['settings']['tools']) != set(runs['B']['settings']['tools']) for name in 'CD'):
        raise ValueError('B/C/D must have the identical toolset.')
    if any(runs['C']['settings'][k] != runs['D']['settings'][k]
           for k in ('adapter_sha256','adapter_hash','lora_adapter_id','lora_weights_version')):
        raise ValueError('C/D adapter identities differ.')


def _results(rows, expected, setting):
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError('Every attempted task must be reported exactly once.')
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != RESULT:
            raise ValueError('Invalid scored result fields.')
        identity = row['id']
        if not isinstance(identity, str) or identity not in expected or identity in indexed:
            raise ValueError('Unknown or duplicate scored task.')
        if (not isinstance(row['status'], str) or row['status'] not in {'full', 'partial', 'fail'}
                or not isinstance(row['completion_status'],str)
                or row['completion_status'] not in {'complete','incomplete','cancelled','timed_out','failed'}
                or type(row['complete']) is not bool
                or any(not _number(row[key], integer=True) for key in (
                    'unsupported_claims', 'tool_calls'))
                or any(row[key] is not None and not _number(row[key],integer=True)
                       for key in ('input_tokens','output_tokens'))
                or not _number(row['latency_ms'])
                or not isinstance(row['resources'], dict)
                or any(not _text(k, 100) or not _number(v) for k, v in row['resources'].items())
                or not isinstance(row['failure_causes'], list)
                or any(not isinstance(cause, str) or cause not in REMEDIES for cause in row['failure_causes'])
                or len(set(row['failure_causes'])) != len(row['failure_causes'])):
            raise ValueError('Invalid score, completion, measurement or failure cause.')
        if not setting['tools'] and any(row[k] for k in
                ('tool_calls','used_tools','enabled_tools','tool_provenance')):
            raise ValueError('A baseline cannot report tool execution or tool provenance.')
        for key in ('used_tools','enabled_tools'):
            if (not isinstance(row[key],list) or any(type(t) is not str for t in row[key])
                    or len(row[key]) != len(set(row[key]))):
                raise ValueError('Invalid tool evidence.')
        if set(row['enabled_tools']) != set(setting['tools']):
            raise ValueError('Enabled tools differ from the declared condition.')
        provenance = row['tool_provenance']
        if not isinstance(provenance,list) or len(provenance) != row['tool_calls']:
            raise ValueError('Tool call count and provenance differ.')
        for i, call in enumerate(provenance):
            if (not isinstance(call,dict) or set(call) != {'tool_id','call_index','status'}
                    or type(call['tool_id']) is not str or call['tool_id'] not in setting['tools']
                    or type(call['call_index']) is not int or call['call_index'] != i
                    or type(call['status']) is not str
                    or call['status'] not in {'ok','error','denied','cancelled','timeout'}):
                raise ValueError('Invalid tool provenance.')
        if set(row['used_tools']) != {call['tool_id'] for call in provenance}:
            raise ValueError('Used tools and provenance differ.')
        effective = row['status']
        causes = set(row['failure_causes'])
        completed = row['complete'] and row['completion_status']=='complete'
        if not completed:
            causes.add('incomplete')
        if row['unsupported_claims']:
            causes.add('hallucination')
        if effective == 'full' and (not completed or row['unsupported_claims']):
            effective = 'partial'
        if effective == 'full' and causes:
            raise ValueError('Full scores cannot include failure causes.')
        if effective != 'full' and not causes:
            raise ValueError('Failed/partial scores require an explicit failure diagnosis.')
        indexed[identity] = {**row, 'effective_status': effective, 'failure_causes': sorted(causes)}
    return indexed


def _measurement(values):
    if not values:
        return {'total':None,'mean':None,'max':None,'measured_cases':0}
    return {'total': sum(values), 'mean': sum(values)/len(values), 'max': max(values), 'measured_cases': len(values)}


def _summary(rows):
    counts = Counter(row['effective_status'] for row in rows.values())
    causes = Counter(cause for row in rows.values() for cause in row['failure_causes'])
    metrics = {key: _measurement([row[key] for row in rows.values() if row[key] is not None])
               for key in ('latency_ms', 'tool_calls', 'input_tokens', 'output_tokens')}
    resource_keys = sorted({key for row in rows.values() for key in row['resources']})
    metrics['resources'] = {key: _measurement([row['resources'][key] for row in rows.values()
                                               if key in row['resources']]) for key in resource_keys}
    return {**{status: counts[status] for status in ('full', 'partial', 'fail')},
            'unsupported_claims': sum(row['unsupported_claims'] for row in rows.values()),
            'metrics': metrics,
            'failure_causes': {cause: {'count': causes[cause], 'remedy': remedy}
                               for cause, remedy in REMEDIES.items()}}


def evaluate(manifest, runs):
    """Validate an externally supplied complete review and return an advisory report."""
    expected = _manifest(manifest)
    _settings(runs)
    if manifest.get('split') == 'train':
        raise ValueError('Training partition cannot be scored as a benchmark.')
    rows = {name: _results(run['results'], expected, run['settings']) for name, run in runs.items()}
    if any(row['tool_calls'] != 0 for row in rows['A'].values()):
        raise ValueError('A baseline results must report zero tool calls.')
    rank = {'fail': 0, 'partial': 1, 'full': 2}
    comparisons = {}
    for left, right, attribution in [('A', 'B', 'tools'), ('B', 'C', 'adapter'),
                                     ('C', 'D', 'orchestration_and_memory')]:
        comparisons[f'{right}-{left}'] = {
            'attribution': attribution,
            'gained': sorted(identity for identity in expected if rank[rows[right][identity]['effective_status']] > rank[rows[left][identity]['effective_status']]),
            'regressed': sorted(identity for identity in expected if rank[rows[right][identity]['effective_status']] < rank[rows[left][identity]['effective_status']]),
            'full_delta': sum(row['effective_status'] == 'full' for row in rows[right].values()) - sum(row['effective_status'] == 'full' for row in rows[left].values()),
        }
    unsolved = {identity for identity in expected if rows['A'][identity]['effective_status'] != 'full'}
    numerator = sum(rows['D'][identity]['effective_status'] == 'full' for identity in unsolved)
    denominator = len(unsolved)
    return {
        'schema': 2, 'comparison_contract_version':2, 'manifest_schema': manifest['schema'],
        'base_model_id':runs['A']['settings']['base_model_id'],
        'adapter_hash':runs['D']['settings']['adapter_hash'],
        'split': manifest.get('split', 'legacy_final'),
        'category_counts': dict(sorted(Counter(task['category'] for task in manifest['tasks']).items()))
            if manifest['schema'] == 2 else None,
        'benchmark_version': manifest['benchmark_version'], 'cases': len(expected),
        'manifest_sha256': hashlib.sha256(_canonical(manifest).encode()).hexdigest(),
        'runs_sha256': hashlib.sha256(_canonical(runs).encode()).hexdigest(),
        'assertions': {'external_unseen': manifest['external_unseen_assertion'],
                       'independent_reviewer': manifest['independent_reviewer_assertion'],
                       'independently_verified': False},
        'target': {'numerator': numerator, 'denominator': denominator,
                   'rate': numerator/denominator if denominator else None,
                   'achieved': bool(denominator and 5*numerator >= denominator)},
        'conditions': {name: _summary(result) for name, result in rows.items()},
        'comparisons': comparisons,
        'regressions_D_vs_A': sorted(identity for identity in expected if rank[rows['D'][identity]['effective_status']] < rank[rows['A'][identity]['effective_status']]),
        'per_case': [{'id': identity, 'conditions': {name: rows[name][identity] for name in 'ABCD'}}
                     for identity in sorted(expected)],
        'release_authorized': False, 'independent_quality_proven': False,
        'limitations': ['Imported scores and provenance assertions are not independent verification of real model execution or unseen tasks.',
                       'Task digests identify supplied task/rubric commitments; semantic uniqueness requires independent review.',
                       'Per-call budgets do not bind total run budget, runtime, tokenizer, hardware or memory snapshots.',
                       'No statistical or broad quality claim; review regressions before any separate release decision.'],
    }


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key.')
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--runs', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        def read(path):
            if path.stat().st_size > 32*1024*1024:
                raise ValueError('Input JSON exceeds 32 MiB.')
            return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_keys,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON.')))
        report = evaluate(read(args.manifest), read(args.runs))
    except (ValueError, OSError, RecursionError) as exc:
        parser.exit(2, f'Invalid evaluation input: {type(exc).__name__}\n')
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
