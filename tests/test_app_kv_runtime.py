import threading

import torch

import app
from config import NeralConfig
from model import Neral20M


def tiny_model():
    cfg=NeralConfig(vocab_size=260,d_model=32,n_layers=2,n_heads=2,ffn_hidden=64,max_seq_len=64,dropout=0.0)
    torch.manual_seed(991)
    return Neral20M(cfg).eval()


def test_app_kv_generation_matches_model_greedy(monkeypatch):
    model=tiny_model();monkeypatch.setattr(app,'model',model);monkeypatch.setattr(app,'device',torch.device('cpu'))
    ids=app.tokenizer.encode('User: Hallo\nAssistant:',bos=True)
    prompt=torch.tensor([ids],dtype=torch.long)
    with torch.inference_mode():
        expected=model.generate(prompt.clone(),max_new_tokens=10,temperature=0,eos_id=app.tokenizer.eos_id,use_cache=True)
    expected_ids=expected[0,len(ids):].tolist()
    actual=app.generate_tokens(ids,10,0.0,40,threading.Event())
    assert actual==expected_ids


def test_app_generation_cancels_before_prefill(monkeypatch):
    model=tiny_model();monkeypatch.setattr(app,'model',model);monkeypatch.setattr(app,'device',torch.device('cpu'))
    event=threading.Event();event.set()
    ids=app.tokenizer.encode('User: Hallo\nAssistant:',bos=True)
    assert app.generate_tokens(ids,10,0.0,40,event)==[]


def test_info_declares_parameter_free_kv_cache():
    info=app.info_payload()
    assert info['inference']=={'backend':'kv_cache','prefill_once':True,'cache_in_checkpoint':False}
