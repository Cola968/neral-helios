from types import SimpleNamespace
import pytest
import app as runtime


@pytest.fixture
def client(monkeypatch, tmp_path):
    runtime._hits.clear()
    monkeypatch.setattr(runtime,'checkpoint_loaded',False)
    monkeypatch.setattr(runtime,'runtime_error','Modell fehlt; Werkzeuge bereit.')
    monkeypatch.setattr(runtime,'project_store',runtime.ProjectStore(tmp_path/'projects.sqlite3'))
    client=runtime.app.test_client()
    client.get('/',base_url='http://localhost')
    return client


def test_cpu_default_and_truthful_status(client):
    info=client.get('/api/info',base_url='http://localhost').get_json()
    assert info['device']=='cpu'
    assert info['runtime_status']=='unavailable' and not info['checkpoint_loaded']
    assert info['inference_only'] is True
    assert info['tools']==['solve','code_check']


@pytest.mark.parametrize('mode,prompt,expected',[('solve','2+3','solve-tool'),('code','x = 1','code-tool')])
def test_tools_without_checkpoint(client,mode,prompt,expected):
    r=client.post('/api/generate',json={'mode':mode,'prompt':prompt},base_url='http://localhost')
    assert r.status_code==200
    assert r.json['mode']==expected and r.json['model_generated'] is False


def test_no_random_model_or_external_fallback(client,monkeypatch):
    def forbidden(*args):raise AssertionError('unloaded model invoked')
    monkeypatch.setattr(runtime,'generate_tokens',forbidden)
    assert client.post('/api/generate',json={'prompt':'Hallo Neral'},base_url='http://localhost').status_code==503


def test_startup_failure_leaves_tools_available(client,monkeypatch):
    class Assets:
        def __truediv__(self,other):return self
        def is_file(self):return True
    monkeypatch.setattr(runtime,'ASSETS',Assets())
    monkeypatch.setattr(runtime,'checkpoint_paths',lambda:{'pilot_02':(Assets(),Assets())})
    def broken(name):raise ValueError('secret local path')
    monkeypatch.setattr(runtime,'load_runtime',broken)
    runtime.initialize_runtime()
    assert 'secret local path' not in runtime.runtime_error
    assert client.get('/',base_url='http://localhost').status_code==200


def test_empty_integrity_file_rejected_before_deserialization(client,monkeypatch):
    class File:
        def is_file(self):return True
        def stat(self):return SimpleNamespace(st_size=3)
        def read_bytes(self):return b'own'
        def read_text(self,**kwargs):return ''
    monkeypatch.setattr(runtime,'checkpoint_paths',lambda:{'original':(File(),File())})
    with pytest.raises(ValueError,match='Integrität'):runtime.load_runtime('original')


@pytest.mark.parametrize('value',['../runs/latest.pt','C:/private/model.pt',[],None])
def test_client_cannot_select_arbitrary_paths(client,value):
    r=client.post('/api/model',json={'checkpoint':value},base_url='http://localhost')
    assert r.status_code==400
    assert not runtime.checkpoint_loaded


def test_failed_switch_preserves_runtime_and_unlocks(client,monkeypatch):
    monkeypatch.setattr(runtime,'checkpoint_loaded',True)
    original=runtime.model
    def broken(name):raise OSError('C:/private/hidden')
    monkeypatch.setattr(runtime,'load_runtime',broken)
    r=client.post('/api/model',json={'checkpoint':'pilot_02'},base_url='http://localhost')
    assert r.status_code==400 and 'C:/private' not in r.json['error']
    assert runtime.model is original and runtime.checkpoint_loaded
    assert not runtime._generation_lock.locked()


@pytest.mark.parametrize('headers',[{'Origin':'https://evil.invalid'},{'Host':'evil.invalid'},{'Sec-Fetch-Site':'cross-site'}])
def test_security_guards_in_tools_only_mode(client,headers):
    assert client.post('/api/generate',json={'mode':'solve','prompt':'2+3'},headers=headers,base_url='http://localhost').status_code==403


def test_ui_reconnect_and_csp(client):
    r=client.get('/',base_url='http://localhost')
    assert b'id="reconnect"' in r.data
    assert "script-src 'self'" in r.headers['Content-Security-Policy']
    assert r.headers['X-Frame-Options']=='DENY'


def test_frontier_runtime_selection_and_chat_proxy(client,monkeypatch):
    row={'label':'Arctus 350M · Phoenix','runtime_family':runtime.FRONTIER350_RUNTIME,
         'status':'candidate','parameters':349_825_152,'context_tokens':512,'runtime_port':9683,
         'model_sha256':'a'*64,'tokenizer_sha256':'b'*64,'app_loadable':False}
    monkeypatch.setattr(runtime,'imported_models',lambda:{'neral-arctus-350':row})
    health={'status':'experimental-ready','runtime_family':runtime.FRONTIER350_RUNTIME,
            'checkpoint_sha256':'a'*64,'tokenizer_sha256':'b'*64,'device':'cuda',
            'gpu':'RTX 4070','max_new_tokens':64,'processed_tokens':2_073_161_216}
    monkeypatch.setattr(runtime,'probe_frontier',lambda r:health)
    monkeypatch.setattr(runtime,'frontier_chat',lambda r,messages,budget:(
        {'answer':'Arctus antwortet.','generated_tokens':5,'finish_reason':'eos'},health))
    monkeypatch.setattr(runtime,'checkpoint_loaded',False)
    monkeypatch.setattr(runtime,'active_runtime_family',runtime.LEGACY_RUNTIME)
    monkeypatch.setattr(runtime,'active_frontier_health',None)
    runtime.select_frontier_runtime('neral-arctus-350')
    info=runtime.info_payload()
    assert info['active_runtime_family']==runtime.FRONTIER350_RUNTIME
    assert info['parameters']==349_825_152 and info['tokenizer']=='Neral BPE-16K'
    assert info['device']=='cuda' and info['cancel_supported'] is False
    response=client.post('/api/generate',json={'prompt':'Hallo','temperature':0},base_url='http://localhost')
    assert response.status_code==200
    assert response.json['text']=='Arctus antwortet.'
    assert response.json['runtime_family']==runtime.FRONTIER350_RUNTIME
    assert response.json['active_checkpoint']=='neral-arctus-350'
