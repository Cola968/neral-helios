"""Explicit, local-only CPU inference for the locked native Arctus350 model.

No hub APIs, discovery of training runs, writes to assets, or implicit loading.
Reference: PR #140 at 105502d9be58517f87d8cfff5d9a5b0e286c22b3.
"""
from __future__ import annotations

import hashlib
import os
import re
import secrets
import time
from pathlib import Path

import numpy as np
import torch

from arctus350_core.chat350_contract import (
    EXPECTED_CONFIG, EXPECTED_PARAMS, EXPECTED_SPECIALS,
    EXPECTED_TOKENIZER_SHA256 as TOKENIZER_SHA256,
)
from arctus350_core.chat_prompt350 import prompt_ids
from arctus350_core.model350 import Neral350M


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checked_checkpoint(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r'[a-fA-F0-9]{64}', expected):
        raise ValueError('Checkpoint-SHA-256 muss ausdrücklich konfiguriert sein.')
    handle = Path(path).open('rb')
    try:
        if not 0 < os.fstat(handle.fileno()).st_size <= 6*1024**3:
            raise ValueError('Checkpoint-Dateigröße außerhalb des erlaubten Bereichs.')
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(chunk)
        if not secrets.compare_digest(digest.hexdigest(), expected.lower()):
            raise ValueError('Checkpoint-SHA-256 stimmt nicht überein.')
        handle.seek(0)
        return handle
    except BaseException:
        handle.close()
        raise


def validate_metadata(payload):
    if not isinstance(payload, dict) or payload.get('config') != EXPECTED_CONFIG:
        raise ValueError('Arctus350-Modellkonfiguration ist inkompatibel.')
    if payload.get('tokenizer_sha256') != TOKENIZER_SHA256:
        raise ValueError('Checkpoint und Tokenizer sind inkompatibel.')
    if not isinstance(payload.get('model'), dict):
        raise ValueError('Checkpoint enthält keine Modellgewichte.')


def _load_restricted(handle):
    # Same narrow NumPy RNG allowance as frontier350/secure_checkpoint350.py.
    # Use the already hashed open file, never an unrestricted pickle fallback.
    try:
        from numpy._core.multiarray import _reconstruct, scalar
    except ImportError:
        from numpy.core.multiarray import _reconstruct, scalar
    allowed = [(_reconstruct, 'numpy._core.multiarray._reconstruct'),
               (_reconstruct, 'numpy.core.multiarray._reconstruct'),
               (scalar, 'numpy._core.multiarray.scalar'),
               (scalar, 'numpy.core.multiarray.scalar'), np.ndarray, np.dtype]
    allowed += [type(np.dtype(t)) for t in ('uint32','int64','float64','bool')]
    with torch.serialization.safe_globals(allowed):
        return torch.load(handle, map_location='cpu', weights_only=True)


def load_runtime(checkpoint, expected_sha256, tokenizer_path):
    with Path(tokenizer_path).open('rb') as handle:
        raw_tokenizer = handle.read(16*1024**2+1)
    if len(raw_tokenizer)>16*1024**2 or hashlib.sha256(raw_tokenizer).hexdigest() != TOKENIZER_SHA256:
        raise ValueError('Tokenizer-SHA-256 stimmt nicht mit Arctus350 überein.')
    # Optional dependency is imported only on an explicit model-load request.
    from tokenizers import Tokenizer
    tok = Tokenizer.from_str(raw_tokenizer.decode('utf-8'))
    if tok.get_vocab_size(with_added_tokens=True) != 16384 or any(
        tok.token_to_id(token) != tid for token, tid in EXPECTED_SPECIALS.items()
    ):
        raise ValueError('Tokenizer-Vokabular oder Spezialtokens inkompatibel.')
    with checked_checkpoint(checkpoint, expected_sha256) as handle:
        payload = _load_restricted(handle)
    validate_metadata(payload)
    state = payload['model']
    # Validate against a meta model before allocating a full random 350M model.
    with torch.device('meta'):
        candidate = Neral350M()
    template = candidate.state_dict()
    if state.keys() != template.keys():
        raise ValueError('Checkpoint-Modellschlüssel sind inkompatibel.')
    for key, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.shape != template[key].shape or not value.is_floating_point()
                or not torch.isfinite(value).all()):
            raise ValueError('Checkpoint-Tensoren sind inkompatibel oder ungültig.')
    if not torch.equal(state['tok_emb.weight'], state['lm_head.weight']):
        raise ValueError('Inkonsistente gebundene Gewichte.')
    if len({v.dtype for v in state.values()}) != 1:
        raise ValueError('Gemischte Gewichtstypen werden nicht unterstützt.')
    candidate.load_state_dict(state, strict=True, assign=True)
    candidate.lm_head.weight = candidate.tok_emb.weight
    candidate.eval().requires_grad_(False)
    if candidate.parameter_report()['total'] != EXPECTED_PARAMS:
        raise ValueError('Arctus350-Parameterzahl stimmt nicht.')
    return ArctusRuntime(candidate, tok, expected_sha256.lower())


class ArctusRuntime:
    model_family = 'arctus350'
    eos_id = 1
    vocab_size = 16384

    def __init__(self, model, tokenizer, checkpoint_sha256):
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_sha256 = checkpoint_sha256

    def encode_history(self, history, prompt):
        turns = [(row['role'], row['content']) for row in history]
        return prompt_ids(self.tokenizer, turns + [('user', prompt)])

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @torch.inference_mode()
    def generate(self, ids, max_new, temperature, top_k, event, on_tokens=None):
        if not ids or len(ids)+max_new > self.model.cfg.max_seq_len:
            raise ValueError('Kontext und Antwort überschreiten das Arctus350-Modelllimit.')
        if type(top_k) is not int or not 0 <= top_k <= self.model.cfg.vocab_size:
            raise ValueError('Ungültiges Top-k.')
        output = []
        deadline = time.monotonic()+120
        for _ in range(max_new):
            if time.monotonic() > deadline:
                event.set()
            if event.is_set():
                break
            x = torch.tensor([ids+output], dtype=torch.long, device='cpu')
            logits, _ = self.model(x)
            z = logits[0,-1].float().clone()
            z[[0,2,3]] = -float('inf')
            if temperature == 0:
                nxt = int(z.argmax())
            else:
                z /= temperature
                if top_k:
                    z[z < torch.topk(z, top_k).values[-1]] = -float('inf')
                nxt = int(torch.multinomial(torch.softmax(z, -1), 1))
            output.append(nxt)
            if on_tokens and (len(output)%4 == 0 or nxt == self.eos_id):
                on_tokens(self.decode(output), len(output))
            if nxt == self.eos_id:
                break
        return output
