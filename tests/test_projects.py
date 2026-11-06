import json
import pytest
from project_store import ProjectStore,ripple
from solve import solve

@pytest.fixture
def graph(tmp_path):
    store=ProjectStore(tmp_path/'projects.sqlite3');p=store.create('Budgetprojekt');pid=p['id']
    for r in [dict(id='budget',title='Budget',kind='fact',value='1200'),dict(id='reserve',title='Reserve',kind='formula',value='budget * 0.1'),dict(id='available',title='Verfügbar',kind='formula',value='budget - reserve',live=True),dict(id='answer',title='Plan',kind='answer',value='Verfügbar: {{available}} EUR',live=True),dict(id='task',title='Kaufen',kind='task',value='Angebot prüfen',depends_on=['available'])]:store.add(pid,r)
    return store,pid

def test_cascade_preview_apply_undo_and_versions(graph):
    s,p=graph;initial=s.get(p);preview=s.preview(p,'budget','1600')
    assert s.get(p)==initial
    assert len(preview['paths'])==4
    applied=s.apply(p,preview['id']);assert applied['nodes']['available']['result']=='1440';assert applied['nodes']['task']['stale']
    assert applied['nodes']['answer']['versions'][-1]['previous']=='Verfügbar: 1080 EUR'
    undo=s.undo(p,applied['last_transaction']);assert undo['nodes']['budget']['value']=='1200';assert not undo['nodes']['task']['stale']
    assert [v['text'] for v in undo['nodes']['answer']['versions']]==['Verfügbar: 1080 EUR','Verfügbar: 1440 EUR','Verfügbar: 1080 EUR']
    with pytest.raises(ValueError):s.undo(p,applied['last_transaction'])

def test_conflicting_preview_and_double_apply(graph):
    s,p=graph;a=s.preview(p,'budget','1300');b=s.preview(p,'budget','1500');s.apply(p,a['id'])
    for x in [a,b]:
        with pytest.raises(ValueError):s.apply(p,x['id'])

def test_persistence_and_live_no_fake_changes(graph):
    s,p=graph;before=s.get(p);s.refresh(p);again=ProjectStore(s.path).get(p)
    assert again['nodes']['answer']['versions']==before['nodes']['answer']['versions']
    assert again['nodes']['answer']['last_checked']>=before['nodes']['answer']['last_checked']

def test_ripple_transitive_and_no_mutation(graph):
    s,p=graph;n=s.get(p)['nodes'];assert next(r for r in ripple(n,'budget') if r['id']=='answer')['path']==['budget','available','answer'];assert ripple(n,'answer')==[]

@pytest.mark.parametrize('bad',[dict(id='invalid-key'),dict(kind='shell'),dict(value=''),dict(depends_on=['missing']),dict(live='yes'),dict(kind='formula',value='__import__("os")'),dict(kind='formula',value='budget / 0'),dict(kind='answer',value='{{missing}}')])
def test_invalid_nodes_atomic(graph,bad):
    s,p=graph;before=s.get(p)
    with pytest.raises(ValueError):s.add(p,dict(id='new',title='Neu',kind='fact',value='2')|bad)
    assert s.get(p)==before

def test_invalid_change_does_not_poison_graph(graph):
    s,p=graph;before=s.get(p)
    with pytest.raises(ValueError):s.preview(p,'budget','nicht numerisch')
    assert s.get(p)==before

def test_cross_project_preview_and_undo_blocked(graph):
    s,p=graph;q=s.create('Anderes Projekt')['id'];preview=s.preview(p,'budget','1300')
    with pytest.raises(ValueError):s.apply(q,preview['id'])
    applied=s.apply(p,preview['id'])
    with pytest.raises(ValueError):s.undo(q,applied['last_transaction'])

def test_review_resolves_task_without_overwriting_later_changes(graph):
    s,p=graph;v=s.preview(p,'budget','1400');a=s.apply(p,v['id']);s.review(p,'task')
    assert not s.get(p)['nodes']['task']['stale']
    with pytest.raises(ValueError):s.undo(p,a['last_transaction'])

@pytest.mark.parametrize('expr,result',[('2*x + 3 = 11','x = 4'),('2x=4','x = 2'),('x/3 = 1/2','x = 3/2'),('5*x-3 = 2*x + 9','x = 4'),('x=x','Unendlich viele Lösungen'),('x=x+1','Keine Lösung'),('0.1*x = 0.3','x = 3'),('-x=2','x = -2')])
def test_linear_solver(expr,result):assert solve(expr)['result']==result

@pytest.mark.parametrize('expr',['x*x=9','1/x=2','x/0=1','2*x=__import__(1)','x**2=4','x=1=2','x='+'9'*170])
def test_linear_solver_rejects(expr):
    with pytest.raises(ValueError):solve(expr)

def test_api_projects_confirmation_and_security(tmp_path,monkeypatch):
    import app
    monkeypatch.setattr(app,'project_store',ProjectStore(tmp_path/'api.sqlite3'))
    app._hits.clear();client=app.app.test_client();kw={'base_url':'http://localhost'};client.get('/',**kw)
    p=client.post('/api/projects',json={'name':'Test'},**kw).get_json();path='/api/projects/'+p['id']
    assert client.post(path+'/nodes',json={'id':'budget','title':'Budget','kind':'fact','value':'1200'},**kw).status_code==200
    preview=client.post(path+'/preview',json={'node':'budget','value':'1600'},**kw).get_json()
    assert client.post(path+'/apply',json={'preview':preview['id']},**kw).status_code==400
    assert client.post(path+'/apply',json={'preview':preview['id'],'confirm':True},**kw).get_json()['nodes']['budget']['result']=='1600'
    assert client.post(path+'/preview',json={'node':[],'value':'1'},**kw).status_code==400
    assert client.post(path+'/refresh',json={},headers={'Origin':'https://evil.invalid'},**kw).status_code==403

def test_all_linear_integer_solutions():
    # 756 independently constructed equations, including negative coefficients.
    for a in [-7,-3,-1,1,2,7]:
        for x in range(-10,11):
            for b in [-9,-1,0,1,4,11]:assert solve(f'{a}*x + ({b}) = {a*x+b}')['result']==f'x = {x}'
