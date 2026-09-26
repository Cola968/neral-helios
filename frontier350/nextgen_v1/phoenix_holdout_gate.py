"""Mandatory exact-overlap gate for the optional audited Phoenix pilot launcher.

Never grants approval. Reads the existing operator-approved frozen holdout and
shards locally and refuses literal token matches before delegating to training.
A negative exact-token scan is NOT proof of semantic holdout independence.
"""
from __future__ import annotations

from pathlib import Path


def require_zero_exact_overlap(args, plan: dict) -> dict:
    """Fail closed on changed holdout, missing tokenizer or any exact hit.

    The source pilot has ALREADY validated explicit execute/trust, source files,
    manifest SHA, human approval declarations, corpus identities and disk. Its
    complete preflight is repeated immediately before any GPU access.
    """
    from helios350_preview import trusted_asset
    from phoenix_blind_ab_audit import load, validate_holdout
    from phoenix_candidate_lineage import TOKENIZER_SHA
    from phoenix_holdout_overlap_audit import audit
    from phoenix_shard_distribution_audit import VOCAB_SIZE
    from tokenizers import Tokenizer

    if not isinstance(plan, dict) or not isinstance(plan.get('lineage'), dict):
        raise ValueError('validated original pilot lineage required')
    pinned = plan['lineage'].get('holdout_sha256')
    frozen = Path(args.frozen_holdout)
    holdout, actual = load(frozen)
    if actual != pinned:
        raise ValueError('frozen holdout bytes changed since operator approval')
    cases = validate_holdout(holdout)
    root = Path(args.assets_root)
    tok_path = trusted_asset(root, root / 'frontier350_assets' / 'tokenizer.json',
                             TOKENIZER_SHA)
    tokenizer = Tokenizer.from_file(str(tok_path))
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise ValueError('approved own tokenizer vocabulary mismatch')
    result = audit(cases, tokenizer, Path(args.train_data), Path(args.valid_data), args.seq)
    training = result['train_case_ids_with_exact_hits']
    validation = result['validation_case_ids_with_exact_hits']
    if training or validation:
        # No private prompt, case ID, local path, token or match position in errors/logs.
        raise ValueError('frozen holdout exact-token overlap: pilot refused; review privately')
    if (result.get('holdout_independence_verified') is not False or
            result.get('training_approved') is not False or
            result.get('full_corpus_token_ids_valid') is not True or
            result.get('case_count') != len(cases) or
            result.get('pattern_count') != 2 * len(cases)):
        raise ValueError('overlap diagnostic supplied incomplete or unexpected evidence')
    return {'case_count': len(cases), 'train_exact_hit_cases': 0,
            'validation_exact_hit_cases': 0,
            'exact_overlap_gate_passed': True,
            'holdout_independence_verified': False, 'training_approved': False}
