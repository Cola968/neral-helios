"""App-only tests: fake token production, no model inference or GPU access."""
import json, os, threading, time
import pytest
import app
from chat_store import ChatStore, Conflict
from stream_runtime import response_stream, safe_prefix
from model_catalog import catalog
from import_neral import import_checkpoint
import storage_codec

@pytest.fixture
def client(monkeypatch,tmp_path):
    monkeypatch.setattr(app,'chat_store',ChatStore(tmp_path/'chats.db'))
    app._hits.clear()
    client=app.app.test_client()
    client.get('/')
    return client

def chat(title='Draft'):
    return dict(title=title,messages=[dict(role='user',content='Hello')],draft='unsent',mode='chat')

def test_session_required(client):
    client.delete_cookie('neral_local')
    assert client.get('/api/chats').status_code==401
    assert client.post('/api/generate',json={'prompt':'2+2'}).status_code==401
    client.get('/')
    assert client.get('/api/health').json['device']=='cpu'

def test_sync_conflict_delete_restart(client,tmp_path):
    key='12345678-1234-1234-1234-123456789abc'
    assert client.post('/api/chats/'+key,json={'revision':0,'chat':chat()}).json['revision']==1
    assert client.post('/api/chats/'+key,json={'revision':0,'chat':chat('stale')}).status_code==409
    assert client.get('/api/chats').json['chats'][0]['chat']['title']=='Draft'
    assert client.post('/api/chats/'+key,json={'revision':1,'deleted':True}).json['revision']==2
    assert ChatStore(app.chat_store.path).all()[0]['deleted']

@pytest.mark.parametrize('payload',[
    {'revision':True,'chat':chat()}, {'revision':0,'chat':{**chat(),'temporary':True}},
    {'revision':0,'chat':{**chat(),'messages':[{'role':'system','content':'bad'}]}},
    {'revision':0,'chat':{**chat(),'draft':'a'*8193}}, {'revision':0,'deleted':'yes'}])
def test_invalid_sync(client,payload):
    assert client.post('/api/chats/'+'a'*20,json=payload).status_code==400

def test_storage_protection_roundtrip():
    value={'secret':'private chat ä'}
    encoded=storage_codec.dumps(value)
    assert storage_codec.loads(encoded)==value
    assert storage_codec.loads(json.dumps(value))==value
    if os.name=='nt':
        assert encoded.startswith('dpapi1:') and 'private chat' not in encoded
        with pytest.raises(Exception):storage_codec.loads('dpapi1:AAAA')

def test_stream_progress_before_completion_and_release(client,monkeypatch):
    release=threading.Event()
    monkeypatch.setattr(app,'PREPARATION_MODE',False)
    monkeypatch.setattr(app,'checkpoint_loaded',True)
    def fake(ids,count,temp,k,event,on_tokens=None):
        on_tokens('Readable words ' * 20,20)
        assert release.wait(3)
        return app.tokenizer.encode('Finished.')
    monkeypatch.setattr(app,'generate_tokens',fake)
    response=client.post('/api/generate',json={'prompt':'Hello','stream':True},buffered=False)
    iterator=iter(response.response)
    assert json.loads(next(iterator))['type']=='start'
    progress=json.loads(next(iterator))
    assert progress['type']=='progress' and progress['text'] and app._generation_lock.locked()
    release.set()
    result=[json.loads(line) for line in iterator]
    assert result[-1]['result']['text']=='Finished.'
    response.close()
    assert not app._generation_lock.locked()

def test_stream_disconnect_cancels_worker():
    stop=threading.Event();finished=threading.Event()
    def run(callback):
        assert stop.wait(3)
        return {'cancelled':True}
    stream=response_stream(run,stop,finished.set)
    next(stream);stream.close()
    assert finished.wait(3)

def test_stream_failure_hides_exception():
    def run(callback):raise RuntimeError('private filesystem and prompt')
    output=''.join(response_stream(run,threading.Event(),lambda:None))
    assert '"error"' in output and 'private filesystem' not in output

@pytest.mark.parametrize('secret',[
    'sk-'+'a'*45, 'Bearer '+'b'*80,
    '-----BEGIN PRIVATE KEY-----\nPRIVATEVALUE\n-----END PRIVATE KEY-----',
    'github_pat_'+'x'*50])
def test_partial_secret_never_streamed(secret):
    raw='Safe introductory words. '*8+secret+' trailing words '*15
    for end in range(1,len(raw)+1):
        prefix=safe_prefix(raw[:end])
        assert secret not in prefix
        assert 'PRIVATEVALUE' not in prefix
        assert 'a'*20 not in prefix and 'b'*20 not in prefix and 'x'*20 not in prefix

def test_catalog_only_complete_explicit_imports(tmp_path,monkeypatch):
    root=tmp_path/'installed_models';root.mkdir()
    (root/'catalog.json').write_text(json.dumps([{'id':'neral-future','label':'Future Neral'}]))
    assert catalog(tmp_path)=={}
    target=root/'neral-future';target.mkdir();(target/'model.pt').write_bytes(b'fixture')
    (target/'model.sha256').write_text('0'*64)
    monkeypatch.setattr(app,'ASSETS',tmp_path)
    assert 'neral-future' in app.checkpoint_paths()
    assert app.info_payload()['model_labels']['neral-future']=='Future Neral'
    (root/'catalog.json').write_text(json.dumps([{'id':'../runs','label':'bad'}]))
    with pytest.raises(ValueError):catalog(tmp_path)

def test_import_hash_failure_preserves_source(tmp_path):
    source=tmp_path/'source.pt';source.write_bytes(b'not weights')
    with pytest.raises(ValueError,match='integrity'):
        import_checkpoint(source,'0'*64,'neral-future','Future',tmp_path/'assets')
    assert source.read_bytes()==b'not weights'
    assert not (tmp_path/'assets').exists()

def test_generation_rate_does_not_block_cancel(client):
    for _ in range(app.RATE_LIMIT):client.post('/api/generate',json={'prompt':'2+2'})
    assert client.post('/api/generate',json={'prompt':'2+2'}).status_code==429
    assert client.post('/api/cancel',json={'request_id':'a'*32}).status_code==200

def test_gateway_streams_before_backend_completes(tmp_path):
    import hashlib, http.client
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    from phone_gateway import create_server
    release=threading.Event();token_file=tmp_path/'token';token_file.write_text('backend-test-token')
    class Backend(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            assert self.headers['X-Helios-Gateway']=='backend-test-token'
            assert not self.headers.get('Cookie')
            self.send_response(200);self.send_header('Content-Type','application/x-ndjson');self.end_headers()
            self.wfile.write(b'{"type":"progress"}\n');self.wfile.flush()
            release.wait(3);self.wfile.write(b'{"type":"done"}\n')
    backend=ThreadingHTTPServer(('127.0.0.1',0),Backend)
    bt=threading.Thread(target=backend.serve_forever,daemon=True);bt.start()
    config={'host':'127.0.0.1','port':0,'pair_hash':hashlib.sha256(b'fixture').hexdigest(),'backend_token_file':str(token_file)}
    gateway=create_server(config,backend.server_port);config['port']=gateway.server_port;gateway.server_close()
    gateway=create_server(config,backend.server_port)
    gt=threading.Thread(target=gateway.serve_forever,daemon=True);gt.start()
    conn=http.client.HTTPConnection('127.0.0.1',gateway.server_port,timeout=3)
    try:
        conn.request('POST','/pair',body='code=fixture');response=conn.getresponse();cookie=response.getheader('Set-Cookie').split(';')[0];response.read();conn.close()
        conn=http.client.HTTPConnection('127.0.0.1',gateway.server_port,timeout=3)
        conn.request('POST','/api/generate',headers={'Cookie':cookie})
        response=conn.getresponse()
        assert response.readline()==b'{"type":"progress"}\n' and not release.is_set()
        release.set();assert response.readline()==b'{"type":"done"}\n'
    finally:
        release.set();conn.close();gateway.shutdown();gateway.server_close();backend.shutdown();backend.server_close();bt.join();gt.join()
