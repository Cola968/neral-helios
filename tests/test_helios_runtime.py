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

# Every listed data/action endpoint must fail before resource validation if
# no app session is present. This cannot test cross-user isolation: NERAL has
# no separate user identities yet.
@pytest.mark.parametrize("method,path",[
    ("GET", "/api/projects"),
    ("POST", "/api/projects"),
    ("GET", "/api/projects/unknown"),
    ("POST", "/api/projects/unknown/preview"),
    ("GET", "/api/chats"),
    ("POST", "/api/chats/1234567890123456"),
    ("POST", "/api/generate"),
    ("POST", "/api/cancel"),
    ("POST", "/api/model"),
    ("GET", "/api/info"),
    ("GET", "/api/metrics"),
    ("GET", "/api/health"),
    ("GET", "/api/agent/capabilities"),
    ("POST", "/api/agent/start"),
    ("POST", "/api/agent/check"),
    ("GET", "/api/agent/1234567890abcdef12345678"),
    ("POST", "/api/agent/1234567890abcdef12345678/next"),
    ("POST", "/api/agent/1234567890abcdef12345678/approve"),
    ("POST", "/api/agent/1234567890abcdef12345678/undo"),
])
def test_all_user_data_routes_reject_missing_session(method,path):
    unauthenticated = runtime.app.test_client()
    response = unauthenticated.open(path, method=method, json={} if method == "POST" else None,
                                    base_url="http://localhost")
    assert response.status_code == 401
    assert response.json["code"] == "session_expired"


def test_no_file_upload_surface_or_multipart_project_import(client):
    from io import BytesIO
    response = client.post("/api/projects", data={
        "name": "Untrusted",
        "file": (BytesIO(b"not-a-safe-file"), "sample.php"),
    }, content_type="multipart/form-data", base_url="http://localhost")
    assert response.status_code == 415
    response = client.post("/api/upload", data=b"file", base_url="http://localhost")
    assert response.status_code == 404
