"""CPU contract/integration tests; synthetic models are never quality measurements."""
import io
import threading
from types import SimpleNamespace

import pytest
import torch

import app as web
import arctus_runtime as ar
from frontier350.chat_prompt350 import prompt_ids
from frontier350.model350 import Neral350Config, Neral350M
from frontier350.secure_checkpoint350 import load_checkpoint

SHA = 'a' * 64
IDENTITY = ar.FAMILY + ':' + SHA


class TokenizerStub:
    def token_to_id(self, token):return ar.EXPECTED_SPECIALS[token]
    def encode(self, text):return SimpleNamespace(ids=[4] * len(text))
    def decode(self, ids, **kwargs):return 'synthetic answer' if ids else ''


class LogitsStub(torch.nn.Module):
    def __init__(self):
        super().__init__();self.calls=0
    def forward(self, x):
        self.calls+=1
        z=torch.zeros((1,x.shape[1],8))
        z[:,-1,0]=100  # BOS must be suppressed, even if it has the largest logit.
        z[:,-1,5 if self.calls==1 else 1]=90
        return z,None


@pytest.fixture
def client(monkeypatch,tmp_path):
    for name,value in [('arctus',None),('PREPARATION_MODE',False),('checkpoint_loaded',False),('checkpoint_sha256',None),('checkpoint_meta',{}),('active_checkpoint','original'),('runtime_error',None)]:
        monkeypatch.setattr(web,name,value)
    # Record the byte runtime too, so every test restores the original global state.
    monkeypatch.setattr(web,'model',web.model)
    monkeypatch.setattr(web,'cfg',web.cfg)
    monkeypatch.setattr(web,'report',web.report)
    monkeypatch.setattr(web,'chat_store',web.ChatStore(tmp_path/'chats.sqlite3'))
    monkeypatch.setattr(web,'ASSETS',tmp_path)
    monkeypatch.delenv('HELIOS_ENABLE_ARCTUS350',raising=False)
    web._hits.clear()
    c=web.app.test_client();c.get('/',base_url='http://localhost')
    return c


def post(c,path,data):return c.post(path,json=data,base_url='http://localhost')


def activate(monkeypatch):
    runtime=ar.ArctusRuntime(LogitsStub(),TokenizerStub(),SHA)
    monkeypatch.setattr(ar,'load',lambda:runtime)
    monkeypatch.setenv('HELIOS_ENABLE_ARCTUS350','1')
    return runtime


def test_disabled_by_default_and_preserves_byte_runtime(client):
    info=client.get('/api/info',base_url='http://localhost').json
    assert info['model_family']=='neral-byte' and not info['external_inference']
    assert 'arctus350' not in info['checkpoints']
    original=web.model
    r=post(client,'/api/model',{'checkpoint':'arctus350'})
    assert r.status_code==400 and 'nicht aktiviert' in r.json['error']
    assert web.model is original and web.arctus is None


def test_missing_files_and_failed_switch_preserve_active_runtime(client,monkeypatch):
    monkeypatch.setenv('HELIOS_ENABLE_ARCTUS350','1')
    monkeypatch.delenv('HELIOS_ARCTUS350_CHECKPOINT',raising=False)
    monkeypatch.delenv('HELIOS_ARCTUS350_TOKENIZER',raising=False)
    monkeypatch.setattr(web,'checkpoint_loaded',True)
    original=web.model
    r=post(client,'/api/model',{'checkpoint':'arctus350'})
    assert r.status_code==400 and 'konfigurieren' in r.json['error']
    assert web.model is original and web.checkpoint_loaded
    assert not web._generation_lock.locked()


def test_preparation_mode_never_loads(client,monkeypatch):
    monkeypatch.setattr(web,'PREPARATION_MODE',True)
    monkeypatch.setattr(ar,'load',lambda:pytest.fail('loader called in preparation mode'))
    assert post(client,'/api/model',{'checkpoint':'arctus350'}).status_code==400


def test_switch_generate_provenance_and_unload(client,monkeypatch):
    native=activate(monkeypatch)
    info=post(client,'/api/model',{'checkpoint':'arctus350'}).json
    assert info['model_family']==ar.FAMILY and info['vocab_size']==16384
    assert info['runtime_id']==IDENTITY and info['tokenizer']=='NERAL-16K-BPE'
    r=post(client,'/api/generate',{'prompt':'Hallo','runtime_id':IDENTITY,'max_new_tokens':4})
    assert r.status_code==200 and r.json['text']=='synthetic answer'
    assert r.json['source']=='native-model' and r.json['model_family']==ar.FAMILY
    assert r.json['runtime_id']==IDENTITY and r.json['token_unit']=='BPE-Token'
    assert native.model.calls==2 and web.model is None
    r=post(client,'/api/model',{'checkpoint':'none'})
    assert r.status_code==200 and not r.json['checkpoint_loaded']
    assert web.arctus is None and web.model is None
    assert post(client,'/api/generate',{'prompt':'Hallo'}).status_code==503
    assert post(client,'/api/generate',{'prompt':'7+8'}).json['text']=='15'


def test_stale_tab_rejected_and_lock_released(client,monkeypatch):
    native=activate(monkeypatch);web.load_runtime('arctus350')
    for identity in [None,'neral-byte:'+SHA,ar.FAMILY+':'+'b'*64]:
        r=post(client,'/api/generate',{'prompt':'Hallo','runtime_id':identity})
        assert r.status_code==409
        assert native.model.calls==0 and not web._generation_lock.locked()


def test_busy_generation_cannot_switch_or_unload(client,monkeypatch):
    activate(monkeypatch);web.load_runtime('arctus350')
    with web._generation_lock:
        assert post(client,'/api/model',{'checkpoint':'none'}).status_code==409
        assert post(client,'/api/model',{'checkpoint':'original'}).status_code==409
        assert post(client,'/api/generate',{'prompt':'Hallo','runtime_id':IDENTITY}).status_code==429
    assert web.arctus is not None


def assistant(**overrides):
    return dict(role='assistant',content='Antwort',source='native-model',checkpoint='arctus350',model_family=ar.FAMILY,runtime_id=IDENTITY,imported=False,interrupted=False,**overrides)


@pytest.mark.parametrize('field,value',[('runtime_id','different'),('model_family','neral-byte'),('checkpoint','original'),('source','tool:calculator'),('imported',True),('interrupted',True)])
def test_foreign_or_incomplete_context_rejected(client,monkeypatch,field,value):
    native=activate(monkeypatch);web.load_runtime('arctus350')
    row=assistant();row[field]=value
    r=post(client,'/api/generate',{'prompt':'Weiter','runtime_id':IDENTITY,'history':[{'role':'user','content':'Frage'},row]})
    assert r.status_code==400 and native.model.calls==0
    assert not web._generation_lock.locked()


def test_native_history_format_and_context_bound(monkeypatch):
    native=ar.ArctusRuntime(LogitsStub(),TokenizerStub(),SHA)
    history=[{'role':'user','content':'Frage'},assistant()]
    assert native.encode(history,'Weiter',4)==prompt_ids(native.tokenizer,[('user','Frage'),('assistant','Antwort'),('user','Weiter')])
    with pytest.raises(ar.ArctusError,match='1024 BPE'):
        native.encode([], 'x'*1024, 4)


def test_cancelled_native_generation_does_not_call_model():
    native=ar.ArctusRuntime(LogitsStub(),TokenizerStub(),SHA)
    stop=threading.Event();stop.set()
    assert native.generate([0,4],8,0,0,stop)==[]
    assert native.model.calls==0


def test_stream_uses_native_runtime_and_releases_lock(client,monkeypatch):
    activate(monkeypatch);web.load_runtime('arctus350')
    r=post(client,'/api/generate',{'prompt':'Hallo','runtime_id':IDENTITY,'stream':True,'max_new_tokens':4})
    import json
    records=[json.loads(line) for line in r.data.splitlines()]
    assert records[-1]['type']=='done'
    assert records[-1]['result']['model_family']==ar.FAMILY
    assert not web._generation_lock.locked()


def test_chat_sync_preserves_checkpoint_identity(client):
    chat={'title':'Test','messages':[{'role':'user','content':'Frage'},assistant()]}
    r=post(client,'/api/chats/abcdefghijklmnop',{'revision':0,'chat':chat})
    assert r.status_code==200
    row=client.get('/api/chats',base_url='http://localhost').json['chats'][0]['chat']['messages'][1]
    assert row['runtime_id']==IDENTITY and row['model_family']==ar.FAMILY


def test_wrong_tokenizer_digest_rejected_before_dependency_or_checkpoint_load(monkeypatch,tmp_path):
    tok=tmp_path/'tokenizer.json';tok.write_text('{}')
    for key,value in {'HELIOS_ENABLE_ARCTUS350':'1','HELIOS_ARCTUS350_CHECKPOINT':str(tmp_path/'missing.pt'),'HELIOS_ARCTUS350_TOKENIZER':str(tok),'HELIOS_ARCTUS350_SHA256':SHA}.items():monkeypatch.setenv(key,value)
    with pytest.raises(ar.ArctusError,match='Tokenizer entspricht nicht'):ar.load()


def test_restricted_checkpoint_loader_accepts_verified_open_stream():
    raw=io.BytesIO();torch.save({'model':{'a':torch.ones(1)}},raw);raw.seek(0)
    assert torch.equal(load_checkpoint(raw)['model']['a'],torch.ones(1))


def test_exact_native_architecture_with_tiny_synthetic_weights(monkeypatch):
    # Real Neral350M layers and forward, reduced dimensions for CPU regression only.
    cfg=Neral350Config(vocab_size=16,d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,ffn_hidden=32,max_seq_len=32)
    tiny=Neral350M(cfg)
    import frontier350.model350 as source
    monkeypatch.setattr(source,'Neral350M',lambda:Neral350M(cfg))
    monkeypatch.setattr(ar,'EXPECTED_PARAMS',tiny.parameter_report()['total'])
    payload={'config':dict(ar.EXPECTED_CONFIG),'tokenizer_sha256':ar.EXPECTED_TOKENIZER_SHA256,'model':tiny.state_dict()}
    restored=ar._model_from_payload(payload)
    assert restored.lm_head.weight is restored.tok_emb.weight
    assert not restored.training and not any(p.requires_grad for p in restored.parameters())
    assert restored(torch.tensor([[0,4,5]]))[0].shape==(1,3,16)
    payload['config']={}
    with pytest.raises(ar.ArctusError,match='Architektur'):ar._model_from_payload(payload)
