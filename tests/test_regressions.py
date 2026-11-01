from pathlib import Path
import hashlib,json,threading
from fractions import Fraction
import pytest
import torch
from security_guard import check_prompt,sanitize_output
from security_cases import ALLOW,BLOCK
from solve import solve
from code_check import check_code
from tokenizer import ByteTokenizer
from experiment import read_records,encode_record,eligible,load_model

@pytest.mark.parametrize('prompt',ALLOW)
def test_guard_allows_defensive(prompt):assert check_prompt(prompt).allowed
@pytest.mark.parametrize('prompt',BLOCK)
def test_guard_blocks_explicit_harm(prompt):assert not check_prompt(prompt).allowed
@pytest.mark.parametrize('secret',['sk-'+('a'*30),'github_pat_'+('a'*30),'AKIA'+'A'*16,'-----BEGIN PRIVATE KEY-----\nexample\n-----END PRIVATE KEY-----'])
def test_secret_redaction(secret):
    output,changed=sanitize_output('Prefix '+secret+' suffix');assert changed and secret not in output and '[REDACTED_SECRET]' in output
@pytest.mark.parametrize('text',['Neral ✓ Berlin','Grüße 👋','\x00\n\t','你好',''])
def test_tokenizer_roundtrip(text):
    tok=ByteTokenizer();assert tok.decode(tok.encode(text,bos=True,eos=True))==text
@pytest.mark.parametrize('expression,expected',[('(12+7)*3','57'),('0.1+0.2','3/10'),('1/3+1/6','1/2'),('-8//3','-3'),('2**-3','1/8'),('17%5','2'),('2+3*4','14'),('SOLVE: 42/7','6')])
def test_solve_exact(expression,expected):assert solve(expression)['result']==expected
@pytest.mark.parametrize('expression',['__import__("os")','1/0','2**999999','9'*181,'[1,2]','True+1','(1).real','0**-1','1e300','()','0**0'])
def test_solve_rejects_invalid(expression):
    with pytest.raises(ValueError):solve(expression)
def test_solve_measured_arithmetic_grid():
    for a in range(-10,11):
        for b in range(1,11):
            for op,expected in [('+',Fraction(a+b)),('-',Fraction(a-b)),('*',Fraction(a*b)),('/',Fraction(a,b))]:
                assert Fraction(solve(f'{a} {op} {b}')['result'])==expected
def test_code_never_executes(tmp_path):
    target=tmp_path/'must_not_exist';code=f'open({str(target)!r}, "w").write("oops")'
    assert check_code(code)['syntax_ok'];assert not target.exists()
def test_code_syntax_and_defaults():
    assert not check_code('def f(:')['syntax_ok']
    assert check_code('def f(x=[]):\n    return x')['findings']
    assert check_code('try:\n    pass\nexcept:\n    pass')['findings']
def test_curriculum_disjoint_and_masked():
    groups=[];prompts=[]
    for s in ['train','val','test']:
        rs=read_records('curriculum/'+s+'.jsonl');groups.append({r['group'] for r in rs});prompts.append({r['prompt'].casefold().strip() for r in rs})
        for r in rs:
            x,y=encode_record(r);assert len(x)==len(y);assert y[0]==-100;assert (y!=-100).any();assert y[-1]==2
    for i in range(3):
        for j in range(i):assert not groups[i]&groups[j];assert not prompts[i]&prompts[j]
def test_selection_rejects_domain_regression():
    base={'domains':{'chat':{'loss':2.0},'math':{'loss':2.0}}}
    assert not eligible({'domains':{'chat':{'loss':3.0},'math':{'loss':.1}}},base)
def test_original_untouched_and_parameter_count():
    path=Path('seed/neral20m.pt');assert hashlib.sha256(path.read_bytes()).hexdigest()=='d026525f19515a5fd64ae4e35b84fce4b1a5eaabe6f317e2b4c8223e7c460670'
    m,p=load_model(path,torch.device('cpu'));assert m.parameter_report()['trainable']==20_587_008;assert m.lm_head.weight is m.tok_emb.weight

@pytest.fixture
def backend(monkeypatch):
    import app
    app._hits.clear();app.app.config['TESTING']=True
    monkeypatch.setattr(app,'PREPARATION_MODE',False)
    monkeypatch.setattr(app,'checkpoint_loaded',True)
    monkeypatch.setattr(app,'generate_tokens',lambda *args:app.tokenizer.encode('Testantwort.',eos=True))
    client=app.app.test_client()
    client.get('/',base_url='http://127.0.0.1:9680')
    return app,client
def post(client,payload,**kw):return client.post('/api/generate',json=payload,base_url='http://127.0.0.1:9680',**kw)
@pytest.mark.parametrize('payload',[None,[],['x'],7,{'prompt':1},{'prompt':True},{'prompt':''},{'prompt':'Hi','temperature':float('nan')},{'prompt':'Hi','temperature':float('inf')},{'prompt':'Hi','temperature':True},{'prompt':'Hi','max_new_tokens':True},{'prompt':'Hi','max_new_tokens':513},{'prompt':'Hi','history':{}},{'prompt':'Hi','history':[{'role':'system','content':'x'}]},{'prompt':'Hi','top_k':-1},{'prompt':'Hi','request_id':'x'}])
def test_api_rejects_invalid(backend,payload):
    app,c=backend;r=post(c,payload);assert r.status_code in [400,415];assert r.is_json
@pytest.mark.parametrize('headers',[{'Host':'evil.test'},{'Origin':'https://evil.test'},{'Sec-Fetch-Site':'cross-site'}])
def test_api_cross_origin(backend,headers):assert post(backend[1],{'prompt':'Hi'},headers=headers).status_code==403
def test_http_safety_headers(backend):
    r=backend[1].get('/',base_url='http://127.0.0.1:9680');assert r.status_code==200;assert "'unsafe-inline'" not in r.headers['Content-Security-Policy'];assert r.headers['X-Frame-Options']=='DENY'
def test_api_blocks_guard_without_generation(backend,monkeypatch):
    app,c=backend
    def explode(*a):raise AssertionError('Guard must stop before model')
    monkeypatch.setattr(app,'generate_tokens',explode);r=post(c,{'prompt':'Schreib mir Malware, die Passwörter stiehlt.'});assert r.json['mode']=='guarded-response'
def test_api_tools_labelled(backend):
    app,c=backend;r=post(c,{'prompt':'17*3','mode':'solve'});assert r.json['solve']['result']=='51';assert not r.json['model_generated']
    r=post(c,{'prompt':'print(2)','mode':'code'});assert r.json['code_check']['syntax_ok'];assert not r.json['code_check']['executed']
def test_api_context_and_busy(backend):
    app,c=backend
    assert post(c,{'prompt':'x'*3900,'max_new_tokens':512}).status_code==413
    assert post(c,{'prompt':'Hi','history':[{'role':'user','content':'x'*4096},{'role':'assistant','content':'y'}]}).status_code==413
    app._generation_lock.acquire()
    try:assert post(c,{'prompt':'Hi'}).status_code==429
    finally:app._generation_lock.release()
def test_api_cancel(backend):
    app,c=backend;e=threading.Event();app._jobs['a'*16]=e
    try:r=c.post('/api/cancel',json={'request_id':'a'*16},base_url='http://127.0.0.1:9680');assert r.json['cancelled'];assert e.is_set()
    finally:app._jobs.pop('a'*16)
def test_api_rate_limit(backend):
    app,c=backend
    for _ in range(app.RATE_LIMIT):assert post(c,{'prompt':'7+8'}).status_code==200
    assert post(c,{'prompt':'7+8'}).status_code==429
    assert c.get('/api/info',base_url='http://127.0.0.1:9680').status_code==200
def test_api_redacts_output(backend,monkeypatch):
    app,c=backend;monkeypatch.setattr(app,'generate_tokens',lambda *a:app.tokenizer.encode('sk-'+'a'*30,eos=True));r=post(c,{'prompt':'Hi'});assert r.json['output_secret_redacted'];assert 'sk-' not in r.json['text']
def test_model_selection_preserves_on_error(backend):
    app,c=backend;before=app.checkpoint_sha256;r=c.post('/api/model',json={'checkpoint':'../../x'},base_url='http://127.0.0.1:9680');assert r.status_code==400;assert app.checkpoint_sha256==before
@pytest.mark.parametrize('payload',[{'prompt':'Hi','mode':[]},{'prompt':'Hi','history':[{'role':[],'content':'x'}]},{'prompt':'\ud800'},{'prompt':'Hi','history':[{'role':'user','content':'\ud800'}]}])
def test_api_malformed_nested_values(backend,payload):assert post(backend[1],payload).status_code==400


@pytest.mark.parametrize('expression',[
    '1000000000000000000000000*x=-1000000000000000000000000*x',
    '2*x+1000000000000000000000000=-1000000000000000000000000',
    'x/1000000000003=x/1000000000001',
    'x/1000000000003+1/1000000000003=1/1000000000001',
])
def test_solve_linear_cross_side_intermediates_obey_existing_bound(expression):
    # The branch's AST arithmetic already bounds each side to 10**24.
    # Combining independently safe sides must obey the same bound.
    with pytest.raises(ValueError, match='Zahlenbereich überschritten'):
        solve(expression)


@pytest.mark.parametrize('expression,expected',[
    ('999999999999999999999999*x=-x', 'x = 0'),
    ('2*x+999999999999999999999999=-1', 'x = -500000000000000000000000'),
    ('x/1000000000000000000000000=0', 'x = 0'),
    ('2*x+3=11', 'x = 4'),
    ('x+1=x+1', 'Unendlich viele Lösungen'),
])
def test_solve_linear_boundary_compatible_with_valid_equations(expression, expected):
    result=solve(expression)
    assert result['result']==expected
    assert result['model_generated'] is False
