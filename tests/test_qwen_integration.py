"""Real Flask handlers with simulated Ollama; no model downloads or GPU."""
import pytest
import app as web
import neral_assistant as assistant

MODEL='qwen3.5:9b'
SOURCE='assistant:ollama-qwen'


def reply(text='Antwort aus Testmodell', **extra):
    return {'model':MODEL,'done':True,'done_reason':'stop',
            'message':{'role':'assistant','content':text},**extra}


def turn():
    return {'role':'assistant','content':'Frühere Antwort',
            'source':SOURCE,'base_model':MODEL,'imported':False,'interrupted':False}


@pytest.fixture
def client(monkeypatch,tmp_path):
    web._hits.clear()
    monkeypatch.setattr(web,'ASSISTANT_ENABLED',True,raising=False)
    monkeypatch.setattr(web,'ASSISTANT_MODEL',MODEL,raising=False)
    monkeypatch.setattr(web,'ASSISTANT_THINK',False,raising=False)
    monkeypatch.setattr(web,'PREPARATION_MODE',False)
    monkeypatch.setattr(web,'arctus',None)
    monkeypatch.setattr(web,'checkpoint_loaded',False)
    monkeypatch.setattr(web,'ASSETS',tmp_path)
    monkeypatch.setattr(web,'chat_store',web.ChatStore(tmp_path/'chats.sqlite3'))
    c=web.app.test_client();c.get('/',base_url='http://localhost')
    return c


def post(c,path,data,headers=None):
    return c.post(path,json=data,base_url='http://localhost',headers=headers or {})


def generate(c,headers=None,**extra):
    return post(c,'/api/generate',{'mode':'assistant','prompt':'Hallo NERAL',
                                   'base_model':MODEL,**extra},headers=headers)


def test_configuration_reports_opt_in_without_claiming_server_is_online(client):
    info=client.get('/api/info',base_url='http://localhost').json
    assert info['assistant']['enabled'] is True
    assert info['assistant']['base_model']==MODEL
    assert info['assistant']['availability']=='unknown'
    assert info['external_inference'] is False


def test_simulated_ollama_response_has_truthful_identity(client,monkeypatch):
    captured=[]
    monkeypatch.setattr(assistant,'local_ollama_request',lambda payload:(captured.append(payload) or reply()))
    r=generate(client)
    assert r.status_code==200
    assert r.json['source']==SOURCE and r.json['mode']=='neral-assistant'
    assert r.json['base_model']==MODEL and r.json['model_generated'] is True
    assert r.json.get('active_checkpoint') is None
    assert captured[0]['stream'] is False and captured[0]['model']==MODEL
    assert captured[0]['messages'][-1]=={'role':'user','content':'Hallo NERAL'}
    assert not web._generation_lock.locked()


def test_optional_4b_uses_its_own_identity_and_rejects_9b_history(client,monkeypatch):
    model='qwen3.5:4b'
    monkeypatch.setattr(web,'ASSISTANT_MODEL',model,raising=False)
    seen=[]
    def fake(payload):
        seen.append(payload)
        return {'model':model,'done':True,'done_reason':'stop',
                'message':{'role':'assistant','content':'4B Testantwort'}}
    monkeypatch.setattr(assistant,'local_ollama_request',fake)
    r=generate(client,base_model=model)
    assert r.status_code==200 and r.json['base_model']==model and seen[0]['model']==model
    r=generate(client,base_model=model,history=[{'role':'user','content':'Frage','imported':False},turn()])
    assert r.status_code==400 and len(seen)==1


def test_disabled_or_preparation_mode_never_calls_ollama(client,monkeypatch):
    monkeypatch.setattr(assistant,'local_ollama_request',lambda _:pytest.fail('unexpected request'))
    monkeypatch.setattr(web,'ASSISTANT_ENABLED',False,raising=False)
    assert generate(client).status_code==503
    monkeypatch.setattr(web,'ASSISTANT_ENABLED',True,raising=False)
    monkeypatch.setattr(web,'PREPARATION_MODE',True)
    assert generate(client).status_code==503


def test_missing_ollama_is_503_and_previous_model_is_untouched(client,monkeypatch):
    def offline(_):raise assistant.AssistantUnavailable('unavailable')
    monkeypatch.setattr(assistant,'local_ollama_request',offline)
    native=object();monkeypatch.setattr(web,'arctus',native)
    r=generate(client)
    assert r.status_code==503 and 'Ollama' in r.json['error']
    assert web.arctus is native and not web._generation_lock.locked()


def test_real_native_model_remains_loaded_after_qwen_turn(client,monkeypatch):
    native=object();monkeypatch.setattr(web,'arctus',native)
    monkeypatch.setattr(assistant,'local_ollama_request',lambda _:reply())
    assert generate(client).status_code==200 and web.arctus is native


def test_stale_model_and_streaming_are_rejected(client,monkeypatch):
    monkeypatch.setattr(assistant,'local_ollama_request',lambda _:pytest.fail('unexpected request'))
    assert generate(client,base_model='qwen3.5:4b').status_code==409
    assert generate(client,stream=True).status_code==400


def test_history_is_forwarded_only_after_provenance_check(client,monkeypatch):
    captured=[]
    monkeypatch.setattr(assistant,'local_ollama_request',lambda payload:(captured.append(payload) or reply()))
    r=generate(client,history=[{'role':'user','content':'Frage','imported':False},turn()])
    assert r.status_code==200
    assert captured[0]['messages'][-3:]==[{'role':'user','content':'Frage'},
        {'role':'assistant','content':'Frühere Antwort'},{'role':'user','content':'Hallo NERAL'}]


@pytest.mark.parametrize('field,value',[
    ('source','native-model'),('source','tool:calculator'),('base_model','qwen3.5:4b'),
    ('imported',True),('interrupted',True),('source',None),('base_model',None)])
def test_foreign_imported_or_incomplete_history_is_rejected(client,monkeypatch,field,value):
    monkeypatch.setattr(assistant,'local_ollama_request',lambda _:pytest.fail('foreign context sent'))
    row=turn();row[field]=value
    r=generate(client,history=[{'role':'user','content':'Frage','imported':False},row])
    assert r.status_code==400


def test_imported_user_message_cannot_enter_context(client,monkeypatch):
    monkeypatch.setattr(assistant,'local_ollama_request',lambda _:pytest.fail('import sent'))
    assert generate(client,history=[{'role':'user','content':'Frage','imported':True},turn()]).status_code==400


def test_shared_lock_prevents_switch_and_unload_while_qwen_is_busy(client):
    with web._generation_lock:
        assert generate(client).status_code==429
        assert post(client,'/api/model',{'checkpoint':'none'}).status_code==409


def test_auth_origin_and_prompt_guard_stay_ahead_of_provider(client,monkeypatch):
    monkeypatch.setattr(assistant,'local_ollama_request',lambda _:pytest.fail('guard bypassed'))
    assert generate(client,headers={'Origin':'https://evil.invalid'}).status_code==403
    anonymous=web.app.test_client()
    assert generate(anonymous).status_code==401


def test_qwen_identity_survives_sqlite_sync(client):
    chat={'title':'Qwen','mode':'assistant','messages':[{'role':'user','mode':'assistant','content':'Frage'},
        {'role':'assistant','mode':'neral-assistant',**{k:v for k,v in turn().items() if k!='role'}}]}
    assert post(client,'/api/chats/abcdefghijklmnop',{'revision':0,'chat':chat}).status_code==200
    saved=client.get('/api/chats',base_url='http://localhost').json['chats'][0]['chat']
    assert saved['mode']=='assistant' and saved['messages'][1]['base_model']==MODEL
    assert saved['messages'][1]['source']==SOURCE


def test_output_limited_by_ollama_is_excluded_from_future_context(client,monkeypatch):
    partial=reply();partial['done_reason']='length'
    monkeypatch.setattr(assistant,'local_ollama_request',lambda _:partial)
    r=generate(client)
    assert r.status_code==200 and r.json['incomplete'] is True


def test_status_reports_installed_not_loaded_and_never_probes_when_disabled(client,monkeypatch):
    monkeypatch.setattr(web,'local_ollama_status',lambda model:'ready' if model==MODEL else 'missing')
    r=client.get('/api/assistant/status',base_url='http://localhost')
    assert r.status_code==200 and r.json=={'availability':'ready','base_model':MODEL}
    monkeypatch.setattr(web,'ASSISTANT_ENABLED',False,raising=False)
    monkeypatch.setattr(web,'local_ollama_status',lambda _:pytest.fail('disabled mode probed'))
    assert client.get('/api/assistant/status',base_url='http://localhost').status_code==503


def test_byte_runtime_rejects_qwen_context(client,monkeypatch):
    monkeypatch.setattr(web,'checkpoint_loaded',True)
    r=post(client,'/api/generate',{'mode':'chat','prompt':'Weiter','history':[
        {'role':'user','content':'Frage','imported':False},turn()]})
    assert r.status_code==409 and not web._generation_lock.locked()


def test_byte_runtime_rejects_explicit_other_byte_checkpoint(client,monkeypatch):
    monkeypatch.setattr(web,'checkpoint_loaded',True)
    monkeypatch.setattr(web,'active_checkpoint','original')
    row={'role':'assistant','content':'Antwort','source':'native-model',
         'model_family':'neral-byte','checkpoint':'neral50m','imported':False,'interrupted':False}
    r=post(client,'/api/generate',{'mode':'chat','prompt':'Weiter',
        'history':[{'role':'user','content':'Frage','imported':False},row]})
    assert r.status_code==409 and not web._generation_lock.locked()


def test_arctus_runtime_rejects_qwen_context(client,monkeypatch):
    import arctus_runtime as ar
    native=ar.ArctusRuntime(object(),object(),'b'*64)
    monkeypatch.setattr(web,'arctus',native);monkeypatch.setattr(web,'checkpoint_loaded',True)
    r=post(client,'/api/generate',{'mode':'chat','prompt':'Weiter','runtime_id':native.identity,
        'history':[{'role':'user','content':'Frage','imported':False},turn()]})
    assert r.status_code==400 and not web._generation_lock.locked()


def test_qwen_calculator_use_is_explicit(client,monkeypatch):
    calls=[]
    def fake(payload):
        calls.append(payload)
        if len(calls)==1:
            return {'model':MODEL,'done':True,'message':{'role':'assistant','content':'',
                    'tool_calls':[{'function':{'name':'calculator','arguments':{'expression':'12*14'}}}]}}
        return reply('12 × 14 = 168')
    monkeypatch.setattr(assistant,'local_ollama_request',fake)
    r=generate(client)
    assert r.status_code==200 and r.json['tool_assisted'] is True
    assert r.json['tools_used']==['calculator'] and r.json['source']==SOURCE
    assert calls[-1]['messages'][-1]['role']=='tool'


def test_status_uses_fixed_loopback_without_proxy_or_redirect(monkeypatch):
    import io, urllib.request
    urls=[]
    class FakeOpener:
        def open(self,url,timeout):
            urls.append((url,timeout))
            return io.BytesIO(b'{"models":[{"name":"qwen3.5:9b"}]}')
    def make_opener(*handlers):
        proxy=next(h for h in handlers if isinstance(h,urllib.request.ProxyHandler))
        redirect=next(h for h in handlers if isinstance(h,assistant._NoRedirect))
        assert proxy.proxies=={} and redirect.redirect_request(None,None,None,None,None) is None
        return FakeOpener()
    monkeypatch.setattr(urllib.request,'build_opener',make_opener)
    assert assistant.local_ollama_status(MODEL)=='ready'
    assert urls==[('http://127.0.0.1:11434/api/tags',2)]
    assert assistant.local_ollama_status('arbitrary-model')=='invalid'
