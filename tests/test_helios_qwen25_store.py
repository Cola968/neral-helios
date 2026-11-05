from chat_store import ChatStore


def test_external_base_model_survives_sync(tmp_path):
    store=ChatStore(tmp_path/'chats.db')
    base='Qwen/Qwen2.5-Coder-7B-Instruct'
    chat=dict(title='Qwen',messages=[dict(role='assistant',content='Hello',mode='trained-generation',
             source='external-model',model_family='qwen2.5-coder',runtime_id='qwen2.5-coder:'+'a'*64,
             checkpoint='hf-qwen25-coder-7b',base_model=base,imported=False,interrupted=False)])
    store.put('12345678-1234-1234-1234-123456789abc',0,chat)
    saved=store.all()[0]['chat']['messages'][0]
    assert saved['base_model']==base
    assert saved['source']=='external-model'
