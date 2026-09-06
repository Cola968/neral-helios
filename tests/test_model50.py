import pytest,torch
from architectures import FIFTY,expected_parameters
from model import Neral20M
from model50 import Neral50M

def test_fifty_exact_parameters_and_padding_equivalence():
    torch.set_num_threads(4);torch.manual_seed(51);original=Neral20M(FIFTY);padded=Neral50M(FIFTY);padded.load_state_dict(original.state_dict());original.eval();padded.eval()
    assert sum(p.numel() for p in padded.parameters() if p.requires_grad)==50_012_976==expected_parameters(FIFTY)
    x=torch.randint(0,260,(1,32))
    with torch.inference_mode():assert torch.allclose(original(x)[0],padded(x)[0],atol=1e-5,rtol=1e-5)

def test_missing_checkpoint_fails_closed_but_tools_work(monkeypatch,tmp_path):
    import app
    monkeypatch.setattr(app,'checkpoint_loaded',False);app._hits.clear();c=app.app.test_client();kw={'base_url':'http://localhost'}
    assert c.post('/api/generate',json={'prompt':'Hallo'},**kw).status_code==503
    r=c.post('/api/generate',json={'prompt':'9 + 2'},**kw)
    assert r.status_code==200 and r.json['solve']['result']=='11' and not r.json['model_generated']
    assert c.post('/api/generate',json={'prompt':'Erkläre warum 9 + 2 = 11'},**kw).status_code==503

def test_generation_diagnostics_detect_utf8_and_loops():
    from free_generation50 import diagnostics
    assert not diagnostics([199], '\ufffd')['valid_utf8']
    d=diagnostics([53]*20,'one one one one one one');assert d['longest_identical_token_run']==20 and d['repeated_word_ngrams']['3']>.5

def test_corpus_filters_and_split_metadata():
    from prepare_pretrain50 import clean
    assert clean('x\ufffd')[0] is None
    assert clean('Contact me at name@example.com.')[0] is None

def test_corpus_split_boundaries_when_assets_present():
    import json
    from pathlib import Path
    p=Path('pretrain50_data/documents.jsonl')
    if not p.exists():pytest.skip('Local corpus deliberately excluded from source repository')
    rows=[json.loads(l) for l in p.read_text(encoding='utf8').splitlines()];seen=set()
    for r in rows:
        assert r['id'] not in seen;seen.add(r['id']);assert r['end']-r['start']==r['bytes']+2
