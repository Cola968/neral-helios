"""Opt-in, CPU-only REAL Neral 350M free-answer report producer.

Run separately on pinned unchanged original and explicitly SHA-pinned own pilot
checkpoint. Does not create a holdout, authenticate human independence, train,
grade answers, or claim a quality gain. Actual answers stay private on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from helios350_preview import CONTINUITY_SHA, TOKENIZER_SHA, sha256_file, trusted_asset
from helios_quality_diagnostic import as_prompt, output_target
from phoenix_blind_ab_audit import SHA, digest, load, validate_holdout
from phoenix_continuity_cpu_probe import EXPECTED_TOKENS, require_pinned_checkpoint


def report_from_inference(holdout: dict, holdout_sha: str, *, checkpoint_sha: str,
                          tokenizer, bos: int, eos: int, budget: int, infer) -> dict:
    """Ingest genuine inference callback; never substitute fixture answers in CLI."""
    cases = validate_holdout(holdout)
    if (not isinstance(checkpoint_sha, str) or not SHA.fullmatch(checkpoint_sha) or
            not isinstance(holdout_sha, str) or not SHA.fullmatch(holdout_sha) or
            type(budget) is not int or not 1 <= budget <= 64 or
            type(bos) is not int or type(eos) is not int or bos == eos):
        raise ValueError('invalid pinned inference configuration')
    rows = []
    for case in cases:
        prompt = case['prompt']
        if len(prompt) > 1024:
            raise ValueError('prompt exceeds preview limit')
        ids = as_prompt(tokenizer, bos, prompt)
        if len(ids) + budget > 512:
            raise ValueError('model context budget exceeded')
        input_sha = digest(json.dumps(
            ids, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))
        answer, used, finish = infer(prompt, list(ids), budget)
        if (not isinstance(answer, str) or len(answer) > 20_000 or
                type(used) is not int or not 0 <= used <= budget or
                finish not in ('eos', 'max_tokens') or
                (finish == 'max_tokens' and used != budget) or
                (finish == 'eos' and used == 0)):
            raise ValueError('inference returned malformed answer/stop metadata')
        rows.append({'id': case['id'],
                     'prompt_sha256': digest(prompt.encode('utf-8')),
                     'input_ids_sha256': input_sha,
                     'answer': answer, 'generated_tokens': used,
                     'finish_reason': finish})
    return {'schema': 1, 'checkpoint_sha256': checkpoint_sha,
            'tokenizer_sha256': TOKENIZER_SHA, 'holdout_sha256': holdout_sha,
            'max_new_tokens': budget,
            'decode': {'strategy': 'greedy_argmax', 'temperature': 0,
                       'prompt_template': 'User: {prompt}\nAssistant:',
                       'bos_id': bos, 'eos_id': eos, 'skip_special_tokens': True,
                       'device': 'cpu', 'dtype': 'float32'},
            'cases': rows}


def greedy_infer(model, tokenizer, eos: int, torch, ids: list[int], budget: int):
    """Count EOS as a generated step, unlike older UI's non-EOS token count."""
    emitted = []
    with torch.inference_mode():
        for step in range(1, budget + 1):
            logits, _ = model(torch.tensor([ids], dtype=torch.long, device='cpu'))
            token = int(logits[0, -1].float().argmax().item())
            if token == eos:
                return tokenizer.decode(emitted, skip_special_tokens=True), step, 'eos'
            emitted.append(token)
            ids.append(token)
    return tokenizer.decode(emitted, skip_special_tokens=True), budget, 'max_tokens'


def load_candidate(checkpoint: Path, expected_sha: str, assets_root: Path,
                   *, allow_legacy_pickle: bool):
    """Hash and identity checks before any potentially unsafe Torch deserialization."""
    if not checkpoint.is_absolute() or checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError('candidate checkpoint must be explicit regular absolute file')
    if not isinstance(expected_sha, str) or not SHA.fullmatch(expected_sha) or expected_sha == CONTINUITY_SHA:
        raise ValueError('distinct full candidate SHA-256 required')
    checkpoint = checkpoint.resolve(strict=True)
    if sha256_file(checkpoint) != expected_sha:
        raise ValueError('candidate file SHA-256 mismatch; load refused')
    root = assets_root.resolve(strict=True)
    tok_file = trusted_asset(root, root / 'frontier350_assets' / 'tokenizer.json', TOKENIZER_SHA)
    import torch
    from tokenizers import Tokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350, Neral350M
    try:
        state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    except (ValueError, RuntimeError, TypeError, __import__('pickle').UnpicklingError):
        if not allow_legacy_pickle:
            raise ValueError('restricted checkpoint decoding failed; explicit trusted legacy opt-in required') from None
        # ONLY the operator's own exact SHA-pinned checkpoint, never untrusted input.
        state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if not isinstance(state, dict) or state.get('config') != FRONTIER_350.to_dict() or state.get('tokenizer_sha256') != TOKENIZER_SHA:
        raise ValueError('candidate architecture/tokenizer mismatch')
    lineage = state.get('warmstart_lineage')
    if (not isinstance(lineage, dict) or lineage.get('kind') != 'explicit_weight_only_new_optimizer_v1'
            or lineage.get('source_sha256') != CONTINUITY_SHA or
            type(state.get('processed_tokens')) is not int or
            not 0 < state['processed_tokens'] <= 262_144 or
            not isinstance(state.get('model'), dict)):
        raise ValueError('candidate is not an approved-family bounded own-pilot checkpoint')
    tokenizer = Tokenizer.from_file(str(tok_file))
    if tokenizer.get_vocab_size() != FRONTIER_350.vocab_size:
        raise ValueError('candidate tokenizer vocabulary mismatch')
    bos, eos = tokenizer.token_to_id('<|bos|>'), tokenizer.token_to_id('<|eos|>')
    if bos is None or eos is None or bos == eos:
        raise ValueError('candidate tokenizer BOS/EOS mismatch')
    model = Neral350M()
    model.load_state_dict(state['model'], strict=True)
    del state
    return model.to(device='cpu', dtype=torch.float32).eval(), tokenizer, int(bos), int(eos), torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--trusted-own-checkpoint', action='store_true')
    p.add_argument('--role', choices=('original', 'candidate'), required=True)
    p.add_argument('--assets-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--candidate-sha256', default='')
    p.add_argument('--allow-trusted-legacy-pickle', action='store_true')
    p.add_argument('--holdout', type=Path, required=True)
    p.add_argument('--private-dir', type=Path, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--max-new-tokens', type=int, default=64)
    args = p.parse_args()
    if not args.execute or not args.trusted_own_checkpoint or os.getenv('CUDA_VISIBLE_DEVICES') != '-1':
        p.error('explicit own-file trust, execute, CUDA_VISIBLE_DEVICES=-1 required')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', args.tag):
        p.error('invalid output tag')
    if args.role == 'original' and (args.candidate_sha256 or args.allow_trusted_legacy_pickle):
        p.error('candidate-only options forbidden for original')
    dest = output_target(args.private_dir, f'helios_quality_ab_{args.role}_{args.tag}.json')
    holdout, holdout_sha = load(args.holdout)
    validate_holdout(holdout)  # Before model load.
    if type(args.max_new_tokens) is not int or not 1 <= args.max_new_tokens <= 64:
        p.error('bounded generation cap 1..64 required')
    if args.role == 'original':
        source = require_pinned_checkpoint(args.checkpoint)
        import helios_guarded_preview as preview
        model, tokenizer, bos, eos, processed, torch = preview.load_continuity_cpu(
            args.assets_root, source)
        if processed != EXPECTED_TOKENS:
            raise ValueError('original processed-token identity mismatch')
        sha = CONTINUITY_SHA
    else:
        model, tokenizer, bos, eos, torch = load_candidate(
            args.checkpoint, args.candidate_sha256, args.assets_root,
            allow_legacy_pickle=args.allow_trusted_legacy_pickle)
        sha = args.candidate_sha256
    if next(model.parameters()).device.type != 'cpu':
        raise RuntimeError('inference device must remain CPU')
    report = report_from_inference(
        holdout, holdout_sha, checkpoint_sha=sha, tokenizer=tokenizer,
        bos=bos, eos=eos, budget=args.max_new_tokens,
        infer=lambda prompt, ids, budget: greedy_infer(model, tokenizer, eos, torch, ids, budget))
    # Reports contain the entire private holdout answer text: never log or upload.
    with dest.open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write('\n')
    print(json.dumps({'event': 'private_real_cpu_inference_report_created',
                      'role': args.role, 'cases': len(report['cases']),
                      'holdout_sha256': holdout_sha, 'checkpoint_sha256': sha,
                      'training_started': False, 'quality_gain_proven': False}, sort_keys=True))


if __name__ == '__main__':
    main()
