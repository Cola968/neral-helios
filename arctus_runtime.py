"""Opt-in local Arctus350 adapter. No downloads, training, or external inference."""
from __future__ import annotations

import hashlib
import os
import re
import secrets
import time
from pathlib import Path

import torch

from frontier350.chat350_contract import (
    EXPECTED_CONFIG, EXPECTED_PARAMS, EXPECTED_SPECIALS, EXPECTED_TOKENIZER_SHA256,
)
from frontier350.chat_prompt350 import prompt_ids

MODEL_ID = 'arctus350'
FAMILY = 'neral-arctus350'
LABEL = 'Arctus350 · experimentell'
# Conservative app budget; the native architecture itself supports 8192 tokens.
APP_CONTEXT = 1024


class ArctusError(ValueError):
    """Only these controlled messages may be returned to the browser."""


def enabled():
    return os.environ.get('HELIOS_ENABLE_ARCTUS350') == '1'


def _digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def load():
    if not enabled():
        raise ArctusError('Arctus350 ist nicht aktiviert.')
    checkpoint = os.environ.get('HELIOS_ARCTUS350_CHECKPOINT', '')
    tokenizer_path = os.environ.get('HELIOS_ARCTUS350_TOKENIZER', '')
    expected = os.environ.get('HELIOS_ARCTUS350_SHA256', '').lower()
    if not checkpoint or not tokenizer_path:
        raise ArctusError('Arctus350: Lokalen Checkpoint und 16K-BPE-Tokenizer konfigurieren.')
    if not re.fullmatch('[a-f0-9]{64}', expected):
        raise ArctusError('Arctus350: Eine lokal verifizierte Checkpoint-SHA-256 ist erforderlich.')
    try:
        path = Path(tokenizer_path)
        if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            raise ArctusError('Arctus350: 16K-BPE-Tokenizer fehlt oder ist zu groß.')
        raw = path.read_bytes()
        if not secrets.compare_digest(hashlib.sha256(raw).hexdigest(), EXPECTED_TOKENIZER_SHA256):
            raise ArctusError('Arctus350: Der Tokenizer entspricht nicht der festgelegten SHA-256.')
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ArctusError('Arctus350: Optionale Abhängigkeit tokenizers ist nicht installiert.') from exc
        tok = Tokenizer.from_str(raw.decode('utf-8'))
        if tok.get_vocab_size() != 16384 or any(tok.token_to_id(k) != v for k, v in EXPECTED_SPECIALS.items()):
            raise ArctusError('Arctus350: Tokenizer-Vokabular oder Kontrolltoken sind unpassend.')
        path = Path(checkpoint)
        if not path.is_file() or not 0 < path.stat().st_size <= 6 * 1024**3:
            raise ArctusError('Arctus350: Lokaler Checkpoint fehlt oder überschreitet 6 GiB.')
        from frontier350.secure_checkpoint350 import load_checkpoint
        # Verify and deserialize the same open file; never unrestricted pickle.
        with path.open('rb') as stream:
            actual = _digest(stream)
            if not secrets.compare_digest(actual, expected):
                raise ArctusError('Arctus350: Checkpoint-Prüfsumme stimmt nicht überein.')
            stream.seek(0)
            payload = load_checkpoint(stream)
        candidate = _model_from_payload(payload)
        return ArctusRuntime(candidate, tok, actual)
    except ArctusError:
        raise
    except Exception as exc:
        raise ArctusError('Arctus350 konnte nicht sicher geladen werden. Architektur, Gewichte und lokale Dateien prüfen.') from exc


def _model_from_payload(payload):
    from frontier350.model350 import Neral350M
    if payload.get('config') != EXPECTED_CONFIG or payload.get('tokenizer_sha256') != EXPECTED_TOKENIZER_SHA256:
        raise ArctusError('Arctus350: Checkpoint-Architektur oder Tokenizer-Zuordnung ist unpassend.')
    state = payload['model']
    if not state or not all(isinstance(v, torch.Tensor) and v.is_floating_point() and torch.isfinite(v).all() for v in state.values()):
        raise ArctusError('Arctus350: Ungültige Modellgewichte.')
    if not torch.equal(state.get('tok_emb.weight'), state.get('lm_head.weight')):
        raise ArctusError('Arctus350: Inkonsistente gebundene Gewichte.')
    # The exact existing architecture; meta construction avoids a second random 350M allocation.
    with torch.device('meta'):
        candidate = Neral350M()
    candidate.load_state_dict(state, strict=True, assign=True)
    candidate.lm_head.weight = candidate.tok_emb.weight
    candidate.to(device='cpu', dtype=torch.float32)
    candidate.eval()
    candidate.requires_grad_(False)
    if candidate.parameter_report()['total'] != EXPECTED_PARAMS:
        raise ArctusError('Arctus350: Parameterzahl stimmt nicht überein.')
    return candidate


class ArctusRuntime:
    def __init__(self, model, tokenizer, sha256):
        self.model, self.tokenizer, self.sha256 = model, tokenizer, sha256
        self.identity = FAMILY + ':' + sha256

    def encode(self, history, prompt, max_new):
        for message in history:
            if message['role'] == 'assistant' and (
                message.get('runtime_id') != self.identity or
                message.get('model_family') != FAMILY or
                message.get('checkpoint') != MODEL_ID or
                message.get('source') != 'native-model' or
                message.get('imported') is not False or message.get('interrupted') is not False
            ):
                raise ArctusError('Arctus350: Verlauf gehört nicht vollständig zu diesem Checkpoint. Neuen Chat beginnen.')
        turns = [(m['role'], m['content']) for m in history] + [('user', prompt)]
        ids = prompt_ids(self.tokenizer, turns)
        if len(ids) + max_new > APP_CONTEXT:
            raise ArctusError('Arctus350: Kontext und Antwort überschreiten das App-Limit von 1024 BPE-Token.')
        return ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def generate(self, ids, max_new, temperature, top_k, event, on_tokens=None):
        # Same native forward/sampling path as chat_cli350; no byte-token substitution.
        x = torch.tensor([ids], dtype=torch.long, device='cpu')
        output = []
        deadline = time.monotonic() + 120
        for _ in range(max_new):
            if time.monotonic() > deadline:
                event.set()
            if event.is_set():
                break
            logits, _ = self.model(x)
            z = logits[:, -1, :].float().clone()
            for token in (0, 2, 3):
                z[:, token] = -float('inf')
            if temperature == 0:
                next_id = z.argmax(-1, keepdim=True)
            else:
                z /= temperature
                if top_k:
                    threshold = torch.topk(z, top_k).values[:, -1, None]
                    z = z.masked_fill(z < threshold, -float('inf'))
                next_id = torch.multinomial(torch.softmax(z, dim=-1), 1)
            token = int(next_id.item())
            if token == EXPECTED_SPECIALS['<|eos|>']:
                break
            output.append(token)
            x = torch.cat([x, next_id], dim=1)
            if on_tokens and len(output) % 4 == 0:
                on_tokens(self.decode(output), len(output))
        return output
