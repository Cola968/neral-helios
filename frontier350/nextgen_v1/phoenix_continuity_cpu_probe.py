"""Opt-in, read-only 2.073B-token own-checkpoint CPU diagnostic; no training.

The source checkpoint lives on D: outside the private tokenizer assets root.
Do not copy it to C:, weaken a SHA gate, or publish any generated text.
Eight PUBLIC development prompts are NOT an independent quality benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from helios350_preview import (CONTINUITY_SHA, TOKENIZER_SHA, sha256_file,
                               trusted_asset)
from phoenix_real_smoke import smoke

EXPECTED_TOKENS = 2_073_161_216


def require_pinned_checkpoint(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('explicit real, non-symlinked absolute local checkpoint required')
    path = path.resolve(strict=True)
    if sha256_file(path) != CONTINUITY_SHA:
        raise ValueError('2.073B own checkpoint SHA mismatch; no pickle/model load')
    return path


def check_metadata(state: dict, config: dict) -> None:
    if not isinstance(state, dict):
        raise ValueError('checkpoint must be a dictionary')
    if (state.get('config') != config or
            state.get('tokenizer_sha256') != TOKENIZER_SHA or
            type(state.get('processed_tokens')) is not int or
            state['processed_tokens'] != EXPECTED_TOKENS or
            not isinstance(state.get('model'), dict)):
        raise ValueError('architecture, tokenizer, tokens or weights mismatch')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--trusted-local-checkpoint', action='store_true')
    parser.add_argument('--assets-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--token-budget', type=int, choices=(1, 32, 64), default=32)
    args = parser.parse_args()
    if not args.execute or not args.trusted_local_checkpoint:
        parser.error('explicit --execute and trust in your own SHA-pinned local checkpoint required')
    checkpoint = require_pinned_checkpoint(args.checkpoint)  # Before importing Torch.
    root = args.assets_root.resolve(strict=True)
    tok_path = trusted_asset(root, root / 'frontier350_assets' / 'tokenizer.json', TOKENIZER_SHA)
    import torch
    from tokenizers import Tokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350, Neral350M
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    # This exact trusted local pickle is SHA-pinned and opt-in. Weights-only is
    # attempted first; historical optimizer/RNG objects may need legacy load.
    try:
        state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
        mode = 'restricted'
    except Exception as exc:
        if not isinstance(exc, (ValueError, RuntimeError, TypeError, __import__('pickle').UnpicklingError)):
            raise
        print('Restricted loader did not support legacy checkpoint fields; falling back ONLY for the exact SHA-pinned explicitly trusted local file.', flush=True)
        state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        mode = 'pinned_trusted_legacy'
    check_metadata(state, FRONTIER_350.to_dict())
    tok = Tokenizer.from_file(str(tok_path))
    if tok.get_vocab_size() != FRONTIER_350.vocab_size:
        raise ValueError('wrong vocabulary')
    bos, eos = tok.token_to_id('<|bos|>'), tok.token_to_id('<|eos|>')
    if bos is None or eos is None:
        raise ValueError('BOS/EOS absent')
    model = Neral350M()
    model.load_state_dict(state['model'], strict=True)
    del state
    model = model.to(device=torch.device('cpu'), dtype=torch.float32).eval()
    print(json.dumps({'event':'2b_checkpoint_verified_loaded','tokens':EXPECTED_TOKENS,
                      'sha256':CONTINUITY_SHA,'device':'cpu','loader':mode,
                      'training_started':False,'checkpoint_modified':False}), flush=True)
    result = smoke(model, tok, int(bos), int(eos), torch, args.token_budget)
    result.update({'checkpoint_sha256':CONTINUITY_SHA,'processed_tokens':EXPECTED_TOKENS,
                   'device':'cpu','training_started':False,'checkpoint_modified':False,
                   'independent_quality_benchmark':False})
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
