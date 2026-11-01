"""Offline contracts: no weights, CUDA initialization or optional packages needed."""
import hashlib
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from model_catalog import catalog


def snapshot(tmp_path):
    root = tmp_path / 'installed_models'
    folder = root / 'hf-qwen25-coder-7b'
    folder.mkdir(parents=True)
    config = dict(model_type='qwen2', architectures=['Qwen2ForCausalLM'],
                  hidden_size=3584, intermediate_size=18944, num_hidden_layers=28,
                  num_attention_heads=28, num_key_value_heads=4, vocab_size=152064,
                  max_position_embeddings=32768, eos_token_id=151645,
                  bos_token_id=151643, tie_word_embeddings=False)
    files = {'config.json': json.dumps(config).encode(),
             'tokenizer_config.json': b'{"tokenizer_class":"Qwen2TokenizerFast"}',
             'tokenizer.json': b'{}', 'model.safetensors': b'offline fixture, never loaded'}
    for name, raw in files.items():
        (folder / name).write_bytes(raw)
    manifest = dict(base_model='Qwen/Qwen2.5-Coder-7B-Instruct', revision='a'*40,
                    files={name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()})
    (folder / 'helios-manifest.json').write_text(json.dumps(manifest))
    (root / 'catalog.json').write_text(json.dumps([
        dict(id=folder.name, label='Qwen2.5 Coder 7B · extern', backend='transformers-nf4')]))
    return folder


def test_explicit_external_snapshot_is_listed_separately(tmp_path):
    snapshot(tmp_path)
    row = catalog(tmp_path)['hf-qwen25-coder-7b']
    assert row['backend'] == 'transformers-nf4'
    assert 'paths' not in row


def test_hash_tampering_rejected_before_loading(tmp_path):
    from model_catalog import validate_hf_snapshot
    folder = snapshot(tmp_path)
    (folder / 'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='integrity'):
        validate_hf_snapshot(folder, verify_hashes=True)


@pytest.mark.parametrize('name,raw', [
    ('model.bin', b'pickle'), ('custom.py', b'code'),
    ('adapter_config.json', b'{}'),
])
def test_unmanifested_and_executable_files_rejected(tmp_path, name, raw):
    folder = snapshot(tmp_path)
    (folder / name).write_bytes(raw)
    with pytest.raises(ValueError):
        catalog(tmp_path)


def test_config_spoof_and_remote_code_rejected(tmp_path):
    folder = snapshot(tmp_path)
    config = json.loads((folder / 'config.json').read_text())
    config['auto_map'] = {'AutoModelForCausalLM': 'evil.Model'}
    (folder / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError):
        catalog(tmp_path)


def test_duplicate_catalog_ids_rejected(tmp_path):
    folder = snapshot(tmp_path)
    path = folder.parent / 'catalog.json'
    rows = json.loads(path.read_text())
    path.write_text(json.dumps(rows + rows))
    with pytest.raises(ValueError):
        catalog(tmp_path)


def test_manifest_path_traversal_and_missing_shards_rejected(tmp_path):
    folder=snapshot(tmp_path)
    path=folder/'helios-manifest.json'
    manifest=json.loads(path.read_text())
    manifest['files']['../private.safetensors']='a'*64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):catalog(tmp_path)


def test_token_budget_and_chat_identity():
    from qwen_local_runtime import Runtime, LocalQwenError, CHAT_TEMPLATE
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            assert kwargs['chat_template']==CHAT_TEMPLATE
            assert messages[0]['role']=='system'
            return [1]*2000
    r=Runtime(None,Tokenizer(),'a'*64)
    with pytest.raises(LocalQwenError,match='2048'):r.encode([],'Hello',49)
    assert len(r.encode([],'Hello',48))==2000
    with pytest.raises(LocalQwenError,match='anderen Modell'):
        r.encode([dict(role='assistant',content='foreign')],'Hello',1)


@pytest.fixture
def hf_api(monkeypatch):
    import torch
    captured={}
    class Tokenizer:
        @classmethod
        def from_pretrained(cls,*args,**kwargs):captured['tokenizer']=kwargs;return cls()
        def decode(self,ids,**kwargs):return 'Hallo Welt ' * len(ids)
    class Model:
        is_loaded_in_4bit=True
        @classmethod
        def from_pretrained(cls,*args,**kwargs):captured['model']=kwargs;return cls()
        def eval(self):pass
        def requires_grad_(self,value):assert value is False
        def generate(self,**kwargs):
            captured['generation']=kwargs
            inputs=kwargs['input_ids']
            kwargs['streamer'].put(inputs)
            ids=[]
            for token in [42,43,44,45,151645]:
                if kwargs['stopping_criteria'][0](inputs,None):break
                ids.append(token)
                kwargs['streamer'].put(torch.tensor([token]))
            return torch.cat([inputs,torch.tensor([ids])],dim=1)
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        Qwen2ForCausalLM=Model,Qwen2TokenizerFast=Tokenizer,
        BitsAndBytesConfig=lambda **kw:SimpleNamespace(**kw),
        GenerationConfig=lambda **kw:SimpleNamespace(**kw),
        StoppingCriteria=object,StoppingCriteriaList=list))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:True)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda _: (10*1024**3,12*1024**3))
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda value,index:captured.update(fraction=value))
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    tensor=torch.tensor
    monkeypatch.setattr(torch,'tensor',lambda *a,**kw:tensor(*a,**{**kw,'device':'cpu'}))
    return captured


def test_local_only_nf4_loader_and_real_generation_contract(tmp_path,hf_api):
    from qwen_local_runtime import load
    r=load(snapshot(tmp_path))
    kw=hf_api['model'];quant=kw['quantization_config']
    assert kw['local_files_only'] and kw['use_safetensors']
    assert kw['trust_remote_code'] is False and kw['token'] is False
    assert kw['device_map']=={'':0} and 'offload_folder' not in kw
    assert quant.load_in_4bit and quant.bnb_4bit_quant_type=='nf4'
    assert quant.bnb_4bit_use_double_quant
    assert hf_api['fraction']==.625
    progress=[]
    result=r.generate([1,2],8,0,40,threading.Event(),lambda *v:progress.append(v))
    assert result==[42,43,44,45,151645] and r.finish_reason=='eos'
    assert progress[-1][1]==5 and not hasattr(hf_api['generation']['generation_config'],'custom_generate')
    assert hf_api['generation']['generation_config'].num_beams==1


def test_stream_stop_and_pre_cancel_never_run_model(tmp_path,hf_api):
    from qwen_local_runtime import load
    r=load(snapshot(tmp_path));event=threading.Event()
    ids=r.generate([1],10,0,40,event,lambda *_:event.set())
    assert len(ids)==4 and r.finish_reason=='cancelled'
    hf_api.pop('generation')
    assert r.generate([1],10,0,40,event)==[]
    assert 'generation' not in hf_api


@pytest.fixture
def web_runtime(monkeypatch,tmp_path):
    import app
    from qwen_local_runtime import Runtime
    class Tokenizer:
        def apply_chat_template(self,*a,**kw):return [1,2]
        def decode(self,ids,**kw):return 'Hallo Welt.'
    class Backend(Runtime):
        def generate(self,ids,max_new,temperature,top_k,event,on_tokens=None):
            if on_tokens:on_tokens('Eine sichere Antwort. '*20,4)
            self.finish_reason='length'
            return [42]*4
        def close(self):self.closed=True
    backend=Backend(None,Tokenizer(),'a'*64)
    app._hits.clear()
    for name,value in dict(qwen25=backend,arctus=None,QWEN25_ENABLED=True,
                           PREPARATION_MODE=False,checkpoint_loaded=True,
                           active_checkpoint='hf-qwen25-coder-7b').items():
        monkeypatch.setattr(app,name,value)
    monkeypatch.setattr(app.torch.cuda,'get_device_name',lambda *_:'offline fake RTX')
    client=app.app.test_client();client.get('/')
    return app,client,backend


def test_flask_ndjson_provenance_incomplete_and_lock(web_runtime):
    app,client,backend=web_runtime
    response=client.post('/api/generate',json=dict(prompt='Schreibe eine Funktion.',stream=True,
                                                   runtime_id=backend.identity))
    events=[json.loads(line) for line in response.data.splitlines()]
    assert events[0]['type']=='start' and events[-1]['type']=='done'
    result=events[-1]['result']
    assert result['source']=='external-model' and result['model_family']=='qwen2.5-coder'
    assert result['base_model']=='Qwen/Qwen2.5-Coder-7B-Instruct'
    assert result['incomplete'] and result['finish_reason']=='length'
    assert result['token_unit']=='BPE-Token'
    assert not app._generation_lock.locked() and not app._jobs


def test_flask_stale_identity_foreign_history_and_guards(web_runtime):
    app,client,backend=web_runtime
    assert client.post('/api/generate',json=dict(prompt='Hallo',runtime_id='stale')).status_code==409
    history=[dict(role='user',content='Hi'),dict(role='assistant',content='Hello')]
    assert client.post('/api/generate',json=dict(prompt='Hallo',runtime_id=backend.identity,history=history)).status_code==400
    assert not app._generation_lock.locked()
    assert client.post('/api/generate',json=dict(prompt='Hallo',runtime_id=backend.identity),
                       headers={'Origin':'https://evil.invalid'}).status_code==403


def test_tools_and_explicit_unload(web_runtime):
    app,client,backend=web_runtime
    for mode,prompt in [('solve','2+3'),('code','x = 1')]:
        result=client.post('/api/generate',json=dict(prompt=prompt,mode=mode)).json
        assert result['model_generated'] is False
    assert client.post('/api/generate',json=dict(prompt='Hallo',mode='assistant')).status_code==409
    result=client.post('/api/model',json=dict(checkpoint='none'))
    assert result.status_code==200 and app.qwen25 is None and backend.closed


def test_failed_external_load_preserves_active_runtime(monkeypatch,tmp_path):
    import app
    folder=snapshot(tmp_path)
    monkeypatch.setattr(app,'ASSETS',tmp_path)
    monkeypatch.setattr(app,'qwen25',None)
    monkeypatch.setattr(app,'QWEN25_ENABLED',True)
    monkeypatch.setattr(app,'PREPARATION_MODE',False)
    original=app.model
    monkeypatch.setattr(app.qwen_local_runtime,'load',lambda *_: (_ for _ in ()).throw(RuntimeError('OOM')))
    with pytest.raises(RuntimeError):app.load_runtime(folder.name)
    assert app.model is original and app.qwen25 is None


def test_external_disabled_does_not_load_or_list(monkeypatch,tmp_path):
    import app
    folder=snapshot(tmp_path)
    monkeypatch.setattr(app,'ASSETS',tmp_path)
    monkeypatch.setattr(app,'QWEN25_ENABLED',False)
    assert folder.name not in app.info_payload()['checkpoints']
    assert folder.name not in app.checkpoint_paths()
