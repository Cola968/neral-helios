"""Local experimental multi-turn Helios UI for SHA-verified own 350M weights.

Opt-in, read-only inference. No training, cloud, local chat persistence or quality claim.
"""
from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading

import helios350_preview as base
import helios_guarded_preview as guarded
from helios_chat_core import generate_turn

MAX_BODY_BYTES = 20_480


def parse_chat_request(raw: bytes):
    if not 0 < len(raw) <= MAX_BODY_BYTES:
        raise ValueError('chat request size exceeds 20480 bytes')
    try:
        data = json.loads(raw.decode('utf-8'))
    except (ValueError, UnicodeError) as exc:
        raise ValueError('valid UTF-8 JSON required') from exc
    if not isinstance(data, dict) or set(data) != {'messages', 'max_new_tokens'}:
        raise ValueError('expected messages and max_new_tokens')
    return data['messages'], data['max_new_tokens']


def make_chat_handler(model, tokenizer, bos, eos, torch, processed, kind, port):
    parent = base.make_handler(model, tokenizer, bos, eos, torch, processed, kind, 'cpu', port)
    mutex = threading.Lock()
    ui = Path(__file__).with_name('helios_chat_ui.html').read_bytes()

    class ChatHandler(parent):
        def do_GET(self):
            if self.path == '/':
                if not self._allowed():
                    return self._json(403, {'error': 'localhost access only'})
                return self._send(200, ui, 'text/html; charset=utf-8')
            return super().do_GET()

        def do_POST(self):
            if not self._allowed():
                return self._json(403, {'error': 'localhost access only'})
            if self.path != '/chat':
                return self._json(404, {'error': 'not found'})
            if self.headers.get('Content-Type', '').split(';', 1)[0].strip().lower() != 'application/json':
                return self._json(415, {'error': 'application/json required'})
            if self.headers.get('Transfer-Encoding'):
                return self._json(400, {'error': 'chunked requests not supported'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                return self._json(400, {'error': 'invalid request size'})
            if not 1 <= size <= MAX_BODY_BYTES:
                return self._json(413, {'error': 'request exceeds size limit'})
            try:
                messages, budget = parse_chat_request(self.rfile.read(size))
                def next_token(ids, emitted):
                    logits, _ = model(torch.tensor([ids], device='cpu', dtype=torch.long))
                    scores = logits[0, -1].float().clone()
                    for token in set(emitted[-32:]):
                        if token != eos:
                            scores[token] = guarded.penalized_value(float(scores[token].item()))
                    return int(scores.argmax().item())
                with mutex, torch.inference_mode():
                    result = generate_turn(tokenizer, bos, eos, messages, budget, next_token)
                return self._json(200, {**result, 'model': kind,
                    'decoder': 'experimental_chat_repetition_guard',
                    'quality_verified': False, 'history_stored_on_server': False})
            except ValueError as exc:
                return self._json(400, {'error': str(exc)})
            except Exception:
                return self._json(500, {'error': 'local model inference failed; see local environment'})
    return ChatHandler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--trusted-local-checkpoint', action='store_true')
    parser.add_argument('--assets-root', type=Path, default=os.getenv('NERAL_ASSETS_ROOT'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model', choices=('safe', 'continuity-2b'), default='safe')
    parser.add_argument('--port', type=int, default=9683)
    args = parser.parse_args(argv)
    if not args.execute or not args.trusted_local_checkpoint or args.assets_root is None:
        parser.error('explicit --execute, --trusted-local-checkpoint and assets root required')
    if not 1024 <= args.port <= 65535:
        parser.error('port must be 1024..65535')
    if args.model == 'safe':
        model, tok, bos, eos, processed, torch = base.load_verified_model(
            args.assets_root, args.checkpoint, 'safe', 'cpu')
    else:
        model, tok, bos, eos, processed, torch = guarded.load_continuity_cpu(
            args.assets_root, args.checkpoint)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_chat_handler(
        model, tok, bos, eos, torch, processed, args.model, args.port))
    print(json.dumps({'event': 'helios_local_chat_ready', 'url': f'http://127.0.0.1:{args.port}/',
                      'model': args.model, 'processed_tokens': processed,
                      'training_started': False, 'quality_verified': False}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
