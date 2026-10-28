"""Explicit read-only probe against app.py with user-supplied local assets.

Run from the repository root. JSON goes to stdout. No asset writes or downloads.
The error test injects a forward exception; it does not corrupt any weights.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rows(response):
    return [json.loads(line) for line in response.data.splitlines() if line.strip()]


def probe(args):
    # Resolve before the temporary state directory is selected. Never discover
    # checkpoints, contact a hub, or infer a model digest from the tokenizer SHA.
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    tokenizer = Path(args.tokenizer).resolve(strict=True)
    from arctus350_runtime import sha256_file, TOKENIZER_SHA256
    if sha256_file(checkpoint) != args.sha256.lower():
        raise ValueError('checkpoint_sha256_mismatch')
    if sha256_file(tokenizer) != TOKENIZER_SHA256:
        raise ValueError('tokenizer_sha256_mismatch')
    with tempfile.TemporaryDirectory(prefix='neral-arctus-probe-') as state:
        os.environ.update(HELIOS_STATE_ROOT=state, HELIOS_ASSETS_ROOT=state,
                          HELIOS_PREPARATION_MODE='0', HELIOS_CHECKPOINT='arctus350',
                          HELIOS_ARCTUS350_CHECKPOINT=str(checkpoint),
                          HELIOS_ARCTUS350_SHA256=args.sha256,
                          HELIOS_ARCTUS350_TOKENIZER=str(tokenizer))
        import app
        client = app.app.test_client(); client.get('/')
        assert client.get('/api/info').json['checkpoint_loaded'] is False
        loaded = client.post('/api/model', json={'checkpoint':'arctus350'})
        if loaded.status_code != 200:
            raise RuntimeError('explicit_model_load_failed_see_server_log')
        assert loaded.json['model_family'] == 'arctus350'
        assert loaded.json['checkpoint_sha256'] == args.sha256.lower()
        payload = dict(prompt='Antworte kurz: Hallo.', max_new_tokens=args.max_new,
                       temperature=0, top_k=40, stream=True)
        complete = rows(client.post('/api/generate', json=payload))
        assert complete[0]['type'] == 'start' and complete[-1]['type'] == 'done'
        result = complete[-1]['result']
        assert result['model_generated'] and result['source'] == 'native-model'
        assert result['model_family'] == 'arctus350'
        assert result['checkpoint_sha256'] == args.sha256.lower()
        assert not app._generation_lock.locked() and not app._jobs
        forward = app.model.forward
        started = threading.Event()
        def observed(*a, **kw):
            started.set()
            return forward(*a, **kw)
        app.model.forward = observed
        response = None
        try:
            response = client.post('/api/generate', json={**payload,'request_id':'probe-cancel-0001'}, buffered=False)
            assert started.wait(30), 'forward_not_started'
            cancellation = client.post('/api/cancel', json={'request_id':'probe-cancel-0001'}).json
            cancelled = rows(response)[-1]
            assert cancelled['type'] == 'done'
            cancel_ok = cancellation['cancelled'] and cancelled['result']['cancelled']
        finally:
            if response is not None:response.close()
            app.model.forward = forward
        assert not app._generation_lock.locked() and not app._jobs
        def injected_failure(*a, **kw):raise RuntimeError('synthetic-private-detail')
        app.model.forward = injected_failure
        try:
            failed = client.post('/api/generate', json=payload)
            assert rows(failed)[-1]['type'] == 'error'
            assert b'synthetic-private-detail' not in failed.data
            assert not app._generation_lock.locked() and not app._jobs
        finally:
            app.model.forward = forward
        assert sha256_file(checkpoint) == args.sha256.lower(), 'checkpoint_changed'
        assert sha256_file(tokenizer) == TOKENIZER_SHA256, 'tokenizer_changed'
        return dict(status='passed' if cancel_ok else 'incomplete',
                    checkpoint_sha256=args.sha256.lower(),tokenizer_sha256=TOKENIZER_SHA256,
                    explicit_load='passed',streaming='passed',
                    generated_tokens=result['generated_tokens'],seconds=result['seconds'],
                    cancellation='passed' if cancel_ok else 'not_measured_request_completed_before_cancel',
                    injected_forward_error='passed',assets_unchanged=True,
                    device='cpu',quality_evaluation=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sha256', required=True, help='Checkpoint digest, not tokenizer digest')
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--max-new', type=int, default=8, choices=range(2,33))
    args = parser.parse_args()
    try:
        result = probe(args)
    except Exception as error:
        # Local logs may contain details; the portable receipt has no paths/text.
        print(json.dumps({'status':'failed','error_type':type(error).__name__}))
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'passed' else 2


if __name__ == '__main__':sys.exit(main())
