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


def test_assistant_is_opt_in_and_never_silently_replaces_own_model(client,monkeypatch):
    monkeypatch.setattr(runtime,'ASSISTANT_ENABLED',False,raising=False)
    called=[]
    monkeypatch.setattr(runtime,'assistant_turn',lambda *a,**kw:called.append(True),raising=False)
    r=client.post('/api/generate',json={'mode':'assistant','prompt':'Hallo'},base_url='http://localhost')
    assert r.status_code==503
    assert called==[]
    assert client.get('/api/info',base_url='http://localhost').json['assistant']['enabled'] is False


def test_explicit_assistant_uses_local_provider_and_reports_origin(client,monkeypatch):
    monkeypatch.setattr(runtime,'ASSISTANT_ENABLED',True,raising=False)
    monkeypatch.setattr(runtime,'ASSISTANT_MODEL','qwen3.5:9b',raising=False)
    monkeypatch.setattr(runtime,'ASSISTANT_THINK',False,raising=False)
    observed=[]
    def stub(prompt,history,**kwargs):
        observed.append((prompt,history,kwargs))
        return {'text':'Hallo aus NERAL.', 'mode':'neral-assistant',
                'model_generated':True,'base_model':'qwen3.5:9b',
                'active_checkpoint':'Qwen3.5 (Ollama local); not Arctus',
                'generated_tokens':9,'tokens_per_second':13.2,'seconds':0.68,
                'tools_used':[]}
    monkeypatch.setattr(runtime,'assistant_turn',stub,raising=False)
    r=client.post('/api/generate',json={'mode':'assistant','prompt':'Hallo'},base_url='http://localhost')
    assert r.status_code==200
    assert r.json['base_model']=='qwen3.5:9b'
    assert r.json['mode']=='neral-assistant'
    assert observed[0][0]=='Hallo'
    assert observed[0][2]['model']=='qwen3.5:9b'
    assert client.get('/api/info',base_url='http://localhost').json['assistant']['enabled']


def test_assistant_disallows_access_during_preparation_and_when_busy(client,monkeypatch):
    monkeypatch.setattr(runtime,'ASSISTANT_ENABLED',True,raising=False)
    monkeypatch.setattr(runtime,'PREPARATION_MODE',True)
    monkeypatch.setattr(runtime,'assistant_turn',lambda *a,**k:pytest.fail('must not infer'),raising=False)
    r=client.post('/api/generate',json={'mode':'assistant','prompt':'Hallo'},base_url='http://localhost')
    assert r.status_code==503
    monkeypatch.setattr(runtime,'PREPARATION_MODE',False)
    assert runtime._generation_lock.acquire(blocking=False)
    try:
        r=client.post('/api/generate',json={'mode':'assistant','prompt':'Hallo'},base_url='http://localhost')
        assert r.status_code==429
    finally:
        runtime._generation_lock.release()


def test_missing_local_assistant_returns_service_unavailable(client,monkeypatch):
    from neral_assistant import AssistantUnavailable
    monkeypatch.setattr(runtime,'ASSISTANT_ENABLED',True,raising=False)
    monkeypatch.setattr(runtime,'assistant_turn',lambda *a,**k:(_ for _ in ()).throw(AssistantUnavailable('model not installed')),raising=False)
    r=client.post('/api/generate',json={'mode':'assistant','prompt':'Hallo'},base_url='http://localhost')
    assert r.status_code==503
    assert 'model not installed' not in r.get_data(as_text=True)
