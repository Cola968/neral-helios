"""Regression contracts for Ollama tag replacement; all model replies simulated."""
import io
import urllib.request

import pytest
import app as web
import neral_assistant as provider

MODEL='qwen3.5:9b'
SOURCE='assistant:ollama-qwen'
OLD='a'*64
NEW='b'*64


def identity(digest=OLD,availability='ready'):
    return {'availability':availability,'model_digest':digest if availability=='ready' else None}


def previous(digest=OLD):
    return {'role':'assistant','content':'Frühere Antwort','source':SOURCE,'base_model':MODEL,
            'model_digest':digest,'imported':False,'interrupted':False}


@pytest.fixture
def client(monkeypatch,tmp_path):
    web._hits.clear()
    for key,value in [('ASSISTANT_ENABLED',True),('ASSISTANT_MODEL',MODEL),('ASSISTANT_THINK',False),
                      ('PREPARATION_MODE',False),('arctus',None),('checkpoint_loaded',False),('ASSETS',tmp_path)]:
        monkeypatch.setattr(web,key,value)
    monkeypatch.setattr(web,'local_ollama_identity',lambda _:identity(),raising=False)
    monkeypatch.setattr(web,'local_ollama_status',lambda _:'ready')
    monkeypatch.setattr(web,'chat_store',web.ChatStore(tmp_path/'chats.sqlite3'))
    monkeypatch.setattr(provider,'local_ollama_request',lambda _:{'model':MODEL,'done':True,'done_reason':'stop',
        'message':{'role':'assistant','content':'Synthetische Antwort'}})
    c=web.app.test_client();c.get('/',base_url='http://localhost');return c


def post(c,path,data):return c.post(path,json=data,base_url='http://localhost')
def ask(c,**extra):return post(c,'/api/generate',{'mode':'assistant','prompt':'Hallo','base_model':MODEL,**extra})


def test_status_and_new_response_expose_digest(client):
    status=client.get('/api/assistant/status',base_url='http://localhost').json
    assert status['availability']=='ready' and status['model_digest']==OLD
    result=ask(client)
    assert result.status_code==200 and result.json['model_digest']==OLD
    assert result.json['source']==SOURCE


def test_stale_client_identity_rejected_before_model_generation(client,monkeypatch):
    monkeypatch.setattr(web,'local_ollama_identity',lambda _:identity(NEW),raising=False)
    invoked=[]
    monkeypatch.setattr(provider,'local_ollama_request',lambda _:invoked.append(True))
    result=ask(client,model_digest=OLD)
    assert result.status_code==409 and result.json['code']=='assistant_model_changed'
    assert invoked==[] and not web._generation_lock.locked()


@pytest.mark.parametrize('digest',[NEW,None,'a'*63,'malformed'])
def test_old_or_unknown_history_not_sent_to_replaced_model(client,monkeypatch,digest):
    invoked=[]
    monkeypatch.setattr(provider,'local_ollama_request',lambda _:invoked.append(True))
    history=[{'role':'user','content':'Frage','imported':False},previous(digest)]
    result=ask(client,model_digest=OLD,history=history)
    assert result.status_code in (400,409) and invoked==[]


def test_same_digest_history_is_allowed(client,monkeypatch):
    captured=[]
    def fake(payload):
        captured.append(payload)
        return {'model':MODEL,'done':True,'done_reason':'stop','message':{'role':'assistant','content':'OK'}}
    monkeypatch.setattr(provider,'local_ollama_request',fake)
    result=ask(client,model_digest=OLD,history=[{'role':'user','content':'Frage','imported':False},previous()])
    assert result.status_code==200 and result.json['model_digest']==OLD
    assert [m['role'] for m in captured[0]['messages'][-3:]]==['user','assistant','user']


def test_tag_changed_during_generation_does_not_return_or_store_answer(client,monkeypatch):
    states=iter([identity(OLD),identity(NEW)])
    monkeypatch.setattr(web,'local_ollama_identity',lambda _:next(states),raising=False)
    result=ask(client,model_digest=OLD)
    assert result.status_code==409 and result.json['code']=='assistant_model_changed'
    assert 'Synthetische Antwort' not in str(result.json)
    assert not web._generation_lock.locked()


def test_model_disappears_after_generation_does_not_return_answer(client,monkeypatch):
    states=iter([identity(OLD),identity(availability='offline')])
    monkeypatch.setattr(web,'local_ollama_identity',lambda _:next(states),raising=False)
    result=ask(client)
    assert result.status_code==503 and 'Synthetische Antwort' not in str(result.json)
    assert not web._generation_lock.locked()


def test_digest_survives_chat_sync(client):
    chat={'title':'Qwen','mode':'assistant','messages':[{'role':'user','mode':'assistant','content':'Frage'},
          {'mode':'neral-assistant',**previous()}]}
    assert post(client,'/api/chats/abcdefghijklmnop',{'revision':0,'chat':chat}).status_code==200
    message=client.get('/api/chats',base_url='http://localhost').json['chats'][0]['chat']['messages'][1]
    assert message['model_digest']==OLD


@pytest.mark.parametrize('models,expected',[
    ([{'name':MODEL,'digest':'a'*64}],identity()),
    ([{'name':MODEL,'digest':'A'*64}],identity()),
    ([],identity(availability='missing')),
    ([{'name':MODEL}],identity(availability='invalid')),
    ([{'name':MODEL,'digest':'bad'}],identity(availability='invalid')),
    ([{'name':MODEL,'digest':'a'*64},{'name':MODEL,'digest':'b'*64}],identity(availability='invalid')),
])
def test_ollama_tags_digest_is_bounded_validated_and_never_uses_proxy(monkeypatch,models,expected):
    import json
    urls=[]
    class Opener:
        def open(self,url,timeout):
            urls.append((url,timeout));return io.BytesIO(json.dumps({'models':models}).encode())
    def make_opener(*handlers):
        proxy=next(h for h in handlers if isinstance(h,urllib.request.ProxyHandler))
        redirect=next(h for h in handlers if isinstance(h,provider._NoRedirect))
        assert proxy.proxies=={} and redirect.redirect_request(None,None,None,None,None) is None
        return Opener()
    monkeypatch.setattr(urllib.request,'build_opener',make_opener)
    assert provider.local_ollama_identity(MODEL)==expected
    assert urls==[('http://127.0.0.1:11434/api/tags',2)]
