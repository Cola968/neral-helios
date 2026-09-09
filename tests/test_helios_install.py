import hashlib,json,threading,http.client
import pytest
import app
from phone_gateway import create_server

def test_install_routes_and_no_cached_api():
    app._hits.clear();client=app.app.test_client()
    response=client.get('/manifest.webmanifest',base_url='http://localhost')
    manifest=json.loads(response.data.decode('utf-8-sig'))
    assert manifest['display']=='standalone' and manifest['start_url']=='/'
    assert {i['sizes'] for i in manifest['icons']} >= {'192x192','512x512'}
    script=client.get('/sw.js',base_url='http://localhost')
    assert script.status_code==200 and b"!SHELL.includes(u.pathname)" in script.data
    assert b"/api/" not in script.data
    assert "script-src 'self'" in script.headers['Content-Security-Policy']

def test_explicit_missing_default_never_falls_back(monkeypatch):
    monkeypatch.setenv('HELIOS_CHECKPOINT','missing')
    monkeypatch.setattr(app,'PREPARATION_MODE',False)
    monkeypatch.setattr(app,'load_runtime',lambda name:pytest.fail('Must not fall back'))
    app.initialize_runtime()
    assert 'ausgewählte' in app.runtime_error

@pytest.fixture
def gateway():
    config={'host':'127.0.0.1','port':0,'pair_hash':hashlib.sha256(b'test-only-code').hexdigest()}
    server=create_server(config,backend_port=1);config['port']=server.server_address[1]
    # Handler closes over the fixed authority; use ephemeral port captured at construction.
    server.server_close()
    server=create_server(config,backend_port=1)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield config
    server.shutdown();server.server_close();thread.join()

def call(config,path='/',method='GET',headers=None,body=None):
    c=http.client.HTTPConnection('127.0.0.1',config['port'],timeout=3)
    c.request(method,path,body=body,headers=headers or {});r=c.getresponse();result=(r.status,dict(r.getheaders()),r.read());c.close();return result

def test_gateway_blocks_unpaired_and_cross_origin(gateway):
    assert call(gateway,'/api/info')[0]==401
    assert call(gateway,headers={'Host':'evil.invalid'})[0]==403
    assert call(gateway,headers={'Origin':'https://evil.invalid'})[0]==403
    assert call(gateway,headers={'Sec-Fetch-Site':'cross-site'})[0]==403
    assert call(gateway,'/pair','POST',body='code=wrong')[0]==403
    assert call(gateway,'/api/generate','POST',headers={'Content-Length':'999999'})[0]==413

def test_pair_cookie_and_unavailable_backend(gateway):
    status,headers,_=call(gateway,'/pair','POST',body='code=test-only-code')
    assert status==200
    cookie=headers['Set-Cookie'];assert all(flag in cookie for flag in ['Secure','HttpOnly','SameSite=Strict'])
    assert call(gateway,'/api/info',headers={'Cookie':cookie.split(';')[0]})[0]==502

def test_pair_rate_limit(gateway):
    for _ in range(5):assert call(gateway,'/pair','POST',body='code=wrong')[0]==403
    assert call(gateway,'/pair','POST',body='code=wrong')[0]==429
