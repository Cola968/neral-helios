from pathlib import Path
import pytest
import app

@pytest.fixture
def client(monkeypatch):
    app._hits.clear();monkeypatch.setattr(app,'checkpoint_loaded',False)
    client=app.app.test_client()
    client.get('/',base_url='http://localhost')
    return client

@pytest.mark.parametrize('history',[[{'role':'assistant','content':'x'}],[{'role':'assistant','content':'x'},{'role':'user','content':'y'}],[{'role':'user','content':'x'},{'role':'user','content':'y'}]])
def test_orphan_or_wrong_order_history_rejected(client,history):
    assert client.post('/api/generate',json={'prompt':'Hi','history':history},base_url='http://localhost').status_code==400

def test_complete_pair_reaches_runtime_check(client):
    r=client.post('/api/generate',json={'prompt':'Hi','history':[{'role':'user','content':'Frage'},{'role':'assistant','content':'Antwort'}]},base_url='http://localhost')
    assert r.status_code==503

@pytest.mark.parametrize('prompt,result',[('3 × 7','21'),('12 ÷ 4','3'),('5 − 8','-3')])
def test_solve_accepts_typographic_operators(client,prompt,result):
    r=client.post('/api/generate',json={'prompt':prompt,'mode':'solve'},base_url='http://localhost')
    assert r.status_code==200 and r.json['solve']['result']==result
    assert r.json['model_generated'] is False

def test_i1_catalog_requires_export_and_hash_without_reading_runs(monkeypatch,tmp_path):
    monkeypatch.setattr(app,'ASSETS',tmp_path)
    assert 'neral50m_i1' not in app.checkpoint_paths()
    i1=tmp_path/'export/neral50m_i1/neral50m.pt'
    monkeypatch.setattr(Path,'is_file',lambda p:p in {i1,i1.with_suffix('.sha256')})
    paths=app.checkpoint_paths()
    assert paths['neral50m_i1']==(i1,i1.with_suffix('.sha256'))
    assert all('runs' not in p.parts for pair in paths.values() for p in pair)

def test_runtime_stays_cpu_and_reports_pending_i1(client):
    info=client.get('/api/info',base_url='http://localhost').json
    assert info['device']=='cpu' and info['inference_only']
    assert isinstance(info['i1_export_available'],bool)

def test_preparation_mode_never_loads_weights(client,monkeypatch):
    monkeypatch.setattr(app,'PREPARATION_MODE',True)
    def forbidden():raise AssertionError('Checkpoint catalog should not be read at startup')
    monkeypatch.setattr(app,'checkpoint_paths',forbidden)
    app.initialize_runtime()
    assert 'Vorbereitungsmodus' in app.runtime_error
    with pytest.raises(ValueError,match='Vorbereitungsmodus'):
        app.load_runtime('pilot_02')

def test_preparation_blocks_inference_even_with_loaded_flag(client,monkeypatch):
    monkeypatch.setattr(app,'PREPARATION_MODE',True)
    monkeypatch.setattr(app,'checkpoint_loaded',True)
    monkeypatch.setattr(app,'generate_tokens',lambda *a,**k:pytest.fail('Inference forbidden'))
    r=client.post('/api/generate',json={'prompt':'Hallo'},base_url='http://localhost')
    assert r.status_code==503 and 'Vorbereitungsmodus' in r.json['error']
    r=client.post('/api/generate',json={'prompt':'3 × 7','mode':'solve'},base_url='http://localhost')
    assert r.status_code==200 and r.json['model_generated'] is False
