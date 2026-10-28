"""Optional adapter tests: synthetic fixtures, never a real 350M checkpoint claim."""
import hashlib
import json
import threading

import pytest
import torch
import app


@pytest.fixture
def client(monkeypatch, tmp_path):
    app._hits.clear()
    monkeypatch.setattr(app, 'PREPARATION_MODE', False)
    monkeypatch.setattr(app, 'checkpoint_loaded', False)
    monkeypatch.setenv('HELIOS_ARCTUS350_CHECKPOINT', str(tmp_path/'missing.pt'))
    monkeypatch.setenv('HELIOS_ARCTUS350_SHA256', '0'*64)
    monkeypatch.setenv('HELIOS_ARCTUS350_TOKENIZER', str(tmp_path/'tokenizer.json'))
    c = app.app.test_client()
    c.get('/')
    return c


def test_optional_model_is_offered_but_never_loaded_by_startup(client, monkeypatch):
    monkeypatch.setenv('HELIOS_CHECKPOINT', 'arctus350')
    app.initialize_runtime()
    info = client.get('/api/info').json
    assert 'arctus350' in info['checkpoints']
    assert not info['checkpoint_loaded']
    assert info['native_arctus350_integrated'] is True


def test_explicit_load_failure_preserves_old_runtime(client, monkeypatch):
    monkeypatch.setattr(app, 'checkpoint_loaded', True)
    original = app.model
    result = client.post('/api/model', json={'checkpoint':'arctus350'})
    assert result.status_code == 400
    assert app.model is original and app.checkpoint_loaded
    assert not app._generation_lock.locked()


def test_unconfigured_arctus_is_not_offered(client, monkeypatch):
    monkeypatch.delenv('HELIOS_ARCTUS350_CHECKPOINT')
    assert 'arctus350' not in client.get('/api/info').json['checkpoints']


def test_tokenizer_identity_rejected_before_checkpoint_load(tmp_path):
    from arctus350_runtime import load_runtime, TOKENIZER_SHA256
    assert TOKENIZER_SHA256 == '4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd'
    tok = tmp_path/'tok.json'; tok.write_text('{}')
    with pytest.raises(ValueError, match='Tokenizer'):
        load_runtime(tmp_path/'missing.pt', '0'*64, tok)


def test_checkpoint_hash_is_required_before_deserialization(tmp_path):
    from arctus350_runtime import checked_checkpoint
    checkpoint = tmp_path/'model.pt'; checkpoint.write_bytes(b'not a pickle')
    with pytest.raises(ValueError, match='SHA-256'):
        checked_checkpoint(checkpoint, '0'*64)


def test_wrong_config_or_tokenizer_metadata_is_rejected():
    from arctus350_runtime import validate_metadata
    from arctus350_core.chat350_contract import EXPECTED_CONFIG
    for config, digest in [({}, '0'*64), (EXPECTED_CONFIG, '0'*64)]:
        with pytest.raises(ValueError):
            validate_metadata({'config':config,'tokenizer_sha256':digest,'model':{}})


def test_segmented_prompt_and_special_mask_cancellation():
    from arctus350_runtime import ArctusRuntime
    from arctus350_core.model350 import Neral350M, Neral350Config
    class Tok:
        def token_to_id(self, value):
            return {'<|bos|>':0,'<|eos|>':1,'<|sep|>':2,'<|pad|>':3}[value]
        def encode(self, text):
            from types import SimpleNamespace
            # Segment count exposes accidentally encoding a whole transcript.
            return SimpleNamespace(ids=[4]*len(text.split()))
        def decode(self, ids, **kw):return ' '.join(map(str, ids))
    cfg = Neral350Config(vocab_size=8,d_model=8,n_layers=1,n_heads=2,n_kv_heads=1,ffn_hidden=16,max_seq_len=128)
    m = Neral350M(cfg).eval()
    rt = ArctusRuntime(m, Tok(), 'a'*64)
    assert rt.encode_history([{'role':'user','content':'Hi'},{'role':'assistant','content':'Hello'}], 'Next') == [0]+[4]*7
    event = threading.Event(); event.set()
    assert rt.generate([0,4], 4, 0, 0, event) == []
    before = {k:v.clone() for k,v in m.state_dict().items()}
    tokens = rt.generate([0,4], 4, 0, 0, threading.Event())
    assert all(t not in (0,2,3) for t in tokens)
    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())


@pytest.fixture
def tiny_runtime(client, monkeypatch):
    from arctus350_runtime import ArctusRuntime
    from arctus350_core.model350 import Neral350M, Neral350Config
    class Tok:
        def token_to_id(self, value):return {'<|bos|>':0,'<|eos|>':1,'<|sep|>':2,'<|pad|>':3}[value]
        def encode(self, text):
            from types import SimpleNamespace
            return SimpleNamespace(ids=[4]*len(text.split()))
        def decode(self, ids, **kw):return ' '.join(str(t) for t in ids if t not in (0,1,2,3))
    cfg = Neral350Config(vocab_size=8,d_model=8,n_layers=1,n_heads=2,n_kv_heads=1,ffn_hidden=16,max_seq_len=128)
    m = Neral350M(cfg).eval()
    rt = ArctusRuntime(m, Tok(), 'a'*64)
    monkeypatch.setattr(app, 'model', m)
    monkeypatch.setattr(app, 'cfg', cfg)
    monkeypatch.setattr(app, 'report', {'trainable':m.parameter_report()['total']})
    monkeypatch.setattr(app, 'tokenizer', rt)
    monkeypatch.setattr(app, 'active_checkpoint', 'arctus350')
    monkeypatch.setattr(app, 'checkpoint_sha256', 'a'*64)
    monkeypatch.setattr(app, 'checkpoint_loaded', True)
    return rt


def request_payload(**extra):
    return dict(prompt='Hallo', max_new_tokens=8, top_k=0, **extra)


def test_real_flask_stream_uses_adapter_identity(client, tiny_runtime):
    response = client.post('/api/generate', json=request_payload(stream=True))
    rows = [json.loads(line) for line in response.data.splitlines()]
    assert rows[0]['type'] == 'start'
    assert rows[-1]['type'] == 'done'
    result = rows[-1]['result']
    assert result['source'] == 'native-model' and result['model_family'] == 'arctus350'
    assert result['active_checkpoint'] == 'arctus350'
    assert result['checkpoint_sha256'] == 'a'*64
    assert not app._generation_lock.locked()
    assert not app._jobs


def test_real_flask_cancel_is_observed_between_forward_passes(client, tiny_runtime, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    forward = tiny_runtime.model.forward
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return forward(*args, **kwargs)
    monkeypatch.setattr(tiny_runtime.model, 'forward', blocked)
    response = client.post('/api/generate', json=request_payload(stream=True,request_id='c'*16), buffered=False)
    try:
        assert entered.wait(5)
        assert client.post('/api/cancel', json={'request_id':'c'*16}).json['cancelled'] is True
    finally:
        release.set()
    rows = [json.loads(line) for line in response.data.splitlines()]
    assert rows[-1]['type'] == 'done'
    assert rows[-1]['result']['cancelled'] is True
    assert rows[-1]['result']['generated_tokens'] <= 1
    assert not app._generation_lock.locked() and not app._jobs


def test_real_flask_forward_error_is_sanitized_and_unlocks(client, tiny_runtime, monkeypatch):
    def broken(*args, **kwargs):raise RuntimeError('private file /secret/checkpoint')
    monkeypatch.setattr(tiny_runtime.model, 'forward', broken)
    response = client.post('/api/generate', json=request_payload(stream=True))
    rows = [json.loads(line) for line in response.data.splitlines()]
    assert rows[-1]['type'] == 'error'
    assert b'/secret' not in response.data
    assert not app._generation_lock.locked() and not app._jobs
    response = client.post('/api/generate', json=request_payload())
    assert response.status_code == 500 and '/secret' not in response.json['error']
    assert not app._generation_lock.locked() and not app._jobs


def test_context_failure_does_not_leave_model_busy(client, tiny_runtime):
    response = client.post('/api/generate', json={'prompt':'Hallo', 'max_new_tokens':128})
    assert response.status_code == 413
    assert not app._generation_lock.locked() and not app._jobs


def test_tool_answer_never_claims_arctus_origin(client, tiny_runtime):
    response = client.post('/api/generate', json={'prompt':'7+8','stream':True})
    assert response.json['text'] == '15'
    assert response.json['source'] == 'tool:calculator'
    assert response.json['model_generated'] is False
    assert 'checkpoint_sha256' not in response.json


def test_explicit_load_installs_adapter_metadata(client, tiny_runtime, monkeypatch):
    import arctus350_runtime
    monkeypatch.setattr(app, 'checkpoint_loaded', False)
    monkeypatch.setattr(app, 'active_checkpoint', 'original')
    def local_fixture(checkpoint, digest, tokenizer):
        # Replaces only file loading: the installed adapter/model remain real.
        assert str(checkpoint).endswith('missing.pt')
        assert digest == '0'*64
        assert str(tokenizer).endswith('tokenizer.json')
        return tiny_runtime
    monkeypatch.setattr(arctus350_runtime, 'load_runtime', local_fixture)
    response = client.post('/api/model', json={'checkpoint':'arctus350'})
    assert response.status_code == 200
    assert response.json['checkpoint_loaded'] and response.json['model_family'] == 'arctus350'
    assert response.json['tokenizer'] == 'Arctus350-BPE-16384'
    assert response.json['checkpoint_meta']['tokenizer_sha256'] == '4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd'


def test_launcher_recognizes_arctus350_readiness(monkeypatch):
    import io
    import launcher
    class Opener:
        def open(self, url, **kw):
            return io.BytesIO(json.dumps({'name':'Neral / Helios','parameters':349825152}).encode())
    monkeypatch.setattr(launcher.urllib.request, 'build_opener', lambda *a: Opener())
    assert launcher.status() is True
