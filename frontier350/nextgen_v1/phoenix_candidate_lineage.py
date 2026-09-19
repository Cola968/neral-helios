"""Fail-closed metadata checks for a proposed own-weight pilot evaluation.

These checks authenticate neither a human approval nor actual training: metadata
is self-reported inside a separately SHA-pinned checkpoint. No Torch imports.
"""
from __future__ import annotations

import math
from pathlib import Path
import re

SOURCE_SHA = 'f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7'
TOKENIZER_SHA = '4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd'
ORIGINAL_TOKENS = 2_073_161_216
MAX_NEW_TOKENS = 262_144
SAMPLER = 'uniform-valid-windows-v1'
SHA = re.compile(r'[0-9a-f]{64}\Z')
LINEAGE_FIELDS = {'kind', 'source_sha256', 'historical_processed_tokens',
                  'new_run_initial_tokens', 'new_run_initial_step', 'approval_sha256',
                  'holdout_sha256', 'source_path', 'output_path', 'new_target_tokens',
                  'run_contract'}
RUN_FIELDS = {'seq', 'microbatch', 'grad_accum', 'seed', 'lr', 'schedule',
              'weight_decay', 'warmup_frac', 'gradient_checkpointing',
              'initialization', 'source_sha256', 'compile_mode'}


def validate_candidate(state: dict, *, checkpoint: Path, holdout_sha: str) -> None:
    """Check pilot origin, matching holdout, update counts and metadata consistency."""
    if not isinstance(holdout_sha, str) or not SHA.fullmatch(holdout_sha):
        raise ValueError('actual frozen holdout SHA required')
    if not isinstance(state, dict) or type(state.get('checkpoint_schema')) is not int or state['checkpoint_schema'] != 2:
        raise ValueError('candidate missing native checkpoint schema')
    lineage = state.get('warmstart_lineage')
    if not isinstance(lineage, dict) or set(lineage) != LINEAGE_FIELDS:
        raise ValueError('candidate lineage missing required fields')
    if (lineage['kind'] != 'explicit_weight_only_new_optimizer_v1' or
            lineage['source_sha256'] != SOURCE_SHA or
            type(lineage['historical_processed_tokens']) is not int or
            lineage['historical_processed_tokens'] != ORIGINAL_TOKENS or
            type(lineage['new_run_initial_tokens']) is not int or
            lineage['new_run_initial_tokens'] != 0 or
            type(lineage['new_run_initial_step']) is not int or
            lineage['new_run_initial_step'] != 0 or
            not isinstance(lineage['approval_sha256'], str) or
            not SHA.fullmatch(lineage['approval_sha256']) or
            lineage['holdout_sha256'] != holdout_sha):
        raise ValueError('candidate source, approval metadata or holdout lineage mismatch')
    source, output = lineage['source_path'], lineage['output_path']
    if (not isinstance(source, str) or not isinstance(output, str) or
            not Path(source).is_absolute() or not Path(output).is_absolute() or
            Path(source).resolve() == Path(output).resolve() or
            checkpoint.name != 'resume.pt' or checkpoint.parent.resolve() != Path(output).resolve()):
        raise ValueError('candidate checkpoint outside its own separate pilot output')
    run = lineage['run_contract']
    if not isinstance(run, dict) or set(run) != RUN_FIELDS or state.get('run_contract') != run:
        raise ValueError('candidate optimizer/run contract inconsistent with lineage')
    if (run['source_sha256'] != SOURCE_SHA or
            run['initialization'] != 'weight_only_new_optimizer_v1' or
            run['compile_mode'] not in ('none', 'default') or
            run['schedule'] != 'cosine' or run['gradient_checkpointing'] is not True or
            any(type(run[k]) is not int or run[k] <= 0 for k in ('seq', 'microbatch', 'grad_accum')) or
            type(run['seed']) is not int or run['seq'] > 512 or
            run['microbatch'] > 2 or run['grad_accum'] > 16 or
            run['seq'] * run['microbatch'] * run['grad_accum'] > 4096 or
            any(type(run[k]) not in (int, float) or not math.isfinite(run[k])
                for k in ('lr', 'weight_decay', 'warmup_frac')) or
            not 0 < run['lr'] <= 0.0001 or
            not 0 <= run['weight_decay'] <= 0.5 or
            not 0 <= run['warmup_frac'] <= 1):
        raise ValueError('candidate run parameters outside approved bounded family')
    target, processed, step = (state.get('target_tokens'),
                               state.get('processed_tokens'), state.get('step'))
    update = run['seq'] * run['microbatch'] * run['grad_accum']
    if (type(lineage['new_target_tokens']) is not int or
            type(target) is not int or target != lineage['new_target_tokens'] or
            not 0 < target <= MAX_NEW_TOKENS or
            type(processed) is not int or not 0 < processed <= target or
            type(step) is not int or step != (processed + update - 1) // update or
            (processed % update != 0 and processed != target)):
        raise ValueError('candidate new-token budget/update lineage inconsistent')
    if (state.get('tokenizer_sha256') != TOKENIZER_SHA or
            state.get('sampler_version') != SAMPLER or
            any(not isinstance(state.get(k), str) or not SHA.fullmatch(state[k])
                for k in ('train_identity', 'validation_identity')) or
            not isinstance(state.get('model'), dict) or not state['model'] or
            not isinstance(state.get('optimizer'), dict) or
            not isinstance(state['optimizer'].get('state'), dict) or
            not state['optimizer']['state']):
        raise ValueError('candidate missing tokenizer/sampler/data or optimizer-state metadata')
