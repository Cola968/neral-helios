"""Flask-client integration using the unchanged Helios security guard."""
import ast
from collections import defaultdict, deque
from pathlib import Path
import secrets
import threading
import time
from types import SimpleNamespace

from flask import Flask, jsonify, request
import pytest

from neral_control.helios import install
from neral_control.providers import ProviderReply


@pytest.fixture
def helios():
    root = Path(__file__).resolve().parents[2]
    app = Flask('control-test', template_folder=str(root/'templates'), static_folder=str(root/'static'))
    h = SimpleNamespace(app=app, SESSION_TOKEN='test-session', _generation_lock=threading.Lock(),
        _job_lock=threading.Lock(), _jobs={}, qwen25=None, ASSISTANT_ENABLED=False,
        PREPARATION_MODE=True, ASSISTANT_MODEL='qwen3.5:4b', MAX_PROMPT_BYTES=4096)
    # Compile and execute the original guard without importing/loading torch or weights.
    tree = ast.parse((root/'app.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='local_and_rate_guard')
    env = dict(app=app, request=request, jsonify=jsonify, secrets=secrets, time=time,
        ALLOWED_HOSTS={'localhost'}, ALLOWED_ORIGINS={'http://localhost'}, SESSION_TOKEN=h.SESSION_TOKEN,
        _hit_lock=threading.Lock(), _hits=defaultdict(deque), RATE_WINDOW_S=60, RATE_LIMIT=30)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(root/'app.py'), 'exec'), env)
    h.allowed_hosts = env['ALLOWED_HOSTS']
    h.body = lambda: (request.get_json(silent=True), None)
    return h


def client(h):
    c = h.app.test_client()
    c.set_cookie('neral_local', h.SESSION_TOKEN)
    return c


def test_tools_only_real_controller_and_registry(helios):
    install(helios)
    c = client(helios)
    caps = c.get('/api/neral/capabilities').get_json()
    assert caps['model_inference_enabled'] is False
    assert caps['profiles'] == ['fast', 'think', 'deep']
    assert {t['name'] for t in caps['tools'] if t['available']} == {'calculator', 'code_check'}
    math = c.post('/api/neral/task', json={'prompt':'2 + 3', 'request_id':'a'*16}).get_json()
    assert math['status']=='success' and '5' in math['text']
    code = c.post('/api/neral/task', json={'prompt':'CODE_CHECK:\nprint(1)'}).get_json()
    assert code['status']=='success' and code['tools_used']==['code_check']
    assert not helios._jobs and not helios._generation_lock.locked()
    assert c.get('/neral/control').status_code==200


def test_original_security_guard_applies(helios):
    install(helios)
    c = client(helios)
    assert c.get('/api/neral/capabilities', headers={'Host':'evil.test'}).status_code==403
    assert c.post('/api/neral/task', json={'prompt':'2+3'}, headers={'Origin':'https://evil.test'}).status_code==403
    assert helios.app.test_client().get('/api/neral/capabilities').status_code==401


def test_lock_validation_and_no_model_inference(helios):
    install(helios)
    c = client(helios)
    helios._generation_lock.acquire()
    try:
        assert c.post('/api/neral/task', json={'prompt':'2+3'}).status_code==409
    finally:
        helios._generation_lock.release()
    assert c.post('/api/neral/task', json={'prompt':'hi','provider':'local-qwen'}).status_code==503
    assert c.post('/api/neral/task', json={'prompt':'2+3','request_id':'bad/id'}).status_code==400
    assert c.post('/api/neral/task', json={'prompt':'2+3','permissions':['network']}).status_code==400
    assert c.post('/api/neral/task', json={'prompt':'2+3','stream':'yes'}).status_code==400
    assert not helios._jobs and not helios._generation_lock.locked()


class FakeProvider:
    def __init__(self, runtime):
        self.identity = dict(runtime_id=runtime.identity, source='external-model', base_model='fake-qwen')
    def complete(self, messages, tools, options, event, on_progress=None):
        return ProviderReply(content='A verified answer.', finish_reason='eos')


def test_explicit_fake_provider_stale_identity_and_cleanup(helios, monkeypatch):
    monkeypatch.setattr('neral_control.helios.LocalQwenProvider', FakeProvider)
    helios.qwen25 = SimpleNamespace(identity='loaded-qwen-v1')
    install(helios, allow_model_inference=True)
    c = client(helios)
    payload = dict(prompt='Explain trees.', provider='local-qwen', runtime_id='old')
    assert c.post('/api/neral/task', json=payload).status_code==409
    assert not helios._jobs and not helios._generation_lock.locked()
    payload['runtime_id']='loaded-qwen-v1'
    result = c.post('/api/neral/task', json=payload).get_json()
    assert result['status']=='success' and result['text']=='A verified answer.'
    assert not helios._jobs and not helios._generation_lock.locked()


def test_cancel_uses_original_jobs_and_exception_cleanup(helios):
    def factory(registry, provider):
        class Controller:
            def run(self, *args, **kwargs):
                assert helios._generation_lock.locked()
                assert kwargs['event'] in helios._jobs.values()
                raise RuntimeError('test failure')
        return Controller()
    install(helios, controller_factory=factory)
    c = client(helios)
    e = threading.Event()
    helios._jobs['b'*16]=e
    assert c.post('/api/neral/cancel', json={'request_id':'b'*16}).get_json()['cancelled']
    assert e.is_set()
    helios._jobs.clear()
    assert c.post('/api/neral/cancel', json={'request_id':'bad'}).status_code==400
    assert c.post('/api/neral/task', json={'prompt':'2+3'}).status_code==500
    assert not helios._jobs and not helios._generation_lock.locked()


def test_stream_finishes_and_cleans_up(helios):
    install(helios)
    response=client(helios).post('/api/neral/task', json={'prompt':'3*7','stream':True})
    assert response.status_code==200
    assert b'"type": "done"' in response.data
    assert b'21' in response.data
    assert not helios._jobs and not helios._generation_lock.locked()


def test_panel_bootstraps_original_session(helios):
    install(helios)
    c=helios.app.test_client()
    assert c.get('/neral/control').status_code==200
    assert c.get('/api/neral/capabilities').status_code==200


def test_disabled_platform_project_reader_is_reported(helios, monkeypatch, tmp_path):
    monkeypatch.delattr('neral_control.helios.os.O_NOFOLLOW')
    install(helios, project_root=tmp_path)
    caps=client(helios).get('/api/neral/capabilities').get_json()
    assert caps['project_read_unavailable_reason']
    assert not next(t for t in caps['tools'] if t['name']=='project_files')['available']


def test_browser_real_tools_and_mobile_layout(monkeypatch, tmp_path):
    import shutil
    import subprocess
    from werkzeug.serving import make_server
    if not shutil.which('node') or not shutil.which('chromium'):
        pytest.skip('Browser runtime unavailable')
    pytest.importorskip('torch')
    import importlib.util
    import sys
    root=Path(__file__).resolve().parents[2]
    assets=tmp_path/'assets'
    assets.mkdir()
    for key,value in {'HELIOS_PREPARATION_MODE':'1','HELIOS_DEVICE':'cpu',
        'HELIOS_STATE_ROOT':str(tmp_path/'state'),'HELIOS_ASSETS_ROOT':str(assets)}.items():
        monkeypatch.setenv(key,value)
    spec=importlib.util.spec_from_file_location('_neral_control_browser_actual',root/'app.py')
    helios=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,spec.name,helios)
    spec.loader.exec_module(helios)
    from neral_control.controller import ReasoningController
    class CancellationFixture:
        # Explicit cooperative stop fixture; arithmetic/code still use real controller.
        def __init__(self,registry,provider):
            self.delegate=ReasoningController(registry,provider)
        def run(self,prompt,**kwargs):
            if prompt=='__cancel_fixture__':
                stopped=kwargs['event'].wait(timeout=5)
                return dict(status='cancelled' if stopped else 'timeout',text='Cooperative cancellation fixture',tools_used=[])
            return self.delegate.run(prompt,**kwargs)
    install(helios,controller_factory=CancellationFixture)
    server=make_server('127.0.0.1',0,helios.app,threaded=True)
    port=server.server_port
    helios.ALLOWED_HOSTS.add(f'localhost:{port}')
    helios.ALLOWED_ORIGINS.add(f'http://localhost:{port}')
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    script=r"""
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:'/usr/bin/chromium',args:['--no-sandbox']});
  try {
    const page=await browser.newPage(); const errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(process.argv[1]+'/neral/control');
    await page.waitForFunction(()=>document.getElementById('capabilityStatus').textContent.includes('Bereit'));
    assert.equal(await page.locator('#toolOptions input:enabled').count(),2);
    assert.equal(await page.locator('#toolOptions input:disabled').count(),2);
    assert.equal(await page.locator('#taskProvider option').count(),1);
    await page.locator('#taskPrompt').fill('125 * 48');
    await page.locator('#taskSubmit').focus();
    await page.keyboard.press('Enter');
    await page.waitForFunction(()=>document.getElementById('taskStage').textContent==='Erfolgreich');
    assert.match(await page.locator('#taskResult').textContent(),/6000/);
    await page.locator('#taskProfile').selectOption('deep');
    // Preserve an actual newline for the explicit static checker contract.
    await page.locator('#taskPrompt').fill(['CODE_CHECK:','print(1)'].join(String.fromCharCode(10)));
    await page.locator('#taskSubmit').click();
    await page.waitForFunction(()=>document.getElementById('taskStage').textContent==='Erfolgreich');
    assert.match(await page.locator('#taskMeta').textContent(),/code_check/);
    for (const width of [360,390,768,1440]) {
      await page.setViewportSize({width,height:900});
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true,`overflow at ${width}`);
    }
    await page.locator('#taskProfile').selectOption('think');
    assert.equal(await page.locator('#taskProfile').inputValue(),'think');
    await page.route('**/api/neral/task',route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({error:'Test unavailable response'})}));
    await page.locator('#taskPrompt').fill('A non-math task');
    await page.locator('#taskSubmit').click();
    await page.waitForFunction(()=>document.getElementById('taskStage').textContent==='Test unavailable response');
    assert.equal(await page.locator('#taskStage').getAttribute('class'),'small muted control-error');
    await page.unroute('**/api/neral/task');
    // Explicit cooperative controller fixture, no model inference. Real cancel endpoint.
    // Delay only the cancel acknowledgment until the real stream has completed.
    await page.route('**/api/neral/cancel',async route=>{
      const response=await route.fetch();
      await page.waitForFunction(()=>document.getElementById('taskStage').textContent==='Abgebrochen');
      await route.fulfill({response});
    });
    await page.locator('#taskPrompt').fill('__cancel_fixture__');
    await Promise.all([page.waitForResponse(r=>r.url().endsWith('/api/neral/task')),page.locator('#taskSubmit').click()]);
    await Promise.all([page.waitForResponse(r=>r.url().endsWith('/api/neral/cancel')),page.locator('#taskStop').click()]);
    await page.waitForTimeout(100);
    assert.equal(await page.locator('#taskStage').textContent(),'Abgebrochen');
    assert.equal(await page.locator('#taskResult').textContent(),'Cooperative cancellation fixture');
    assert.equal(await page.locator('#taskStop').isDisabled(),true);
    assert.deepEqual(errors,[]);
    console.log('Chromium actual Helios: real calculator/code checker, profiles, session bootstrap, keyboard submit, error handling, cooperative cancellation fixture, 360/390/768/1440 layout passed');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
"""
    try:
        result=subprocess.run(['node','-e',script,f'http://localhost:{port}'],text=True,capture_output=True,timeout=40)
        assert result.returncode==0,result.stdout+result.stderr
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def test_launcher_import_and_help_are_inert(monkeypatch):
    import builtins
    import importlib.util
    path=Path(__file__).resolve().parents[2]/'scripts/neral_control_helios.py'
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name=='app':
            raise AssertionError('Help must not import the Helios runtime')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    spec=importlib.util.spec_from_file_location('_neral_control_launcher_test',path)
    launcher=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    with pytest.raises(SystemExit) as exited:
        launcher.main(['--help'])
    assert exited.value.code==0


@pytest.mark.parametrize('flags,enabled,language', [([],False,None),(['--enable-web-search','--search-language','en'],True,'en')])
def test_launcher_search_is_explicit_without_startup_queries(monkeypatch,flags,enabled,language):
    import importlib.util
    import sys
    path=Path(__file__).resolve().parents[2]/'scripts/neral_control_helios.py'
    calls=[]
    monkeypatch.setenv('HELIOS_PREPARATION_MODE','0')
    fake=SimpleNamespace(app=SimpleNamespace(run=lambda **kwargs:calls.append(('serve',kwargs))),APP_PORT=9680)
    monkeypatch.setitem(sys.modules,'app',fake)
    monkeypatch.setattr('neral_control.helios.install',lambda module,**kwargs:calls.append(('install',kwargs)))
    def no_search(*args,**kwargs):
        raise AssertionError('No query is allowed on startup')
    monkeypatch.setattr('neral_control.search.WikipediaSearch.search',no_search)
    spec=importlib.util.spec_from_file_location('_neral_control_launcher_config_test',path)
    launcher=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    launcher.main(flags)
    settings=calls[0][1]
    assert settings['allow_model_inference'] is False
    provider=settings['search_provider']
    assert (provider is not None)==enabled
    if provider is not None:
        assert provider.host==language+'.wikipedia.org' and provider.enabled
    assert calls[1][0]=='serve'


def test_launcher_tools_only_forces_preparation_before_importing_legacy_app(monkeypatch):
    import builtins,importlib.util,os,sys
    fake=SimpleNamespace(app=SimpleNamespace(run=lambda **kwargs:None),APP_PORT=9680)
    monkeypatch.setitem(sys.modules,'app',fake)
    monkeypatch.setattr('neral_control.helios.install',lambda *a,**k:None)
    monkeypatch.setenv('HELIOS_PREPARATION_MODE','0')
    real_import=builtins.__import__
    def checked(name,*args,**kwargs):
        if name=='app':
            preparation=os.environ.get('HELIOS_PREPARATION_MODE')
            assert preparation=='1'
        return real_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',checked)
    path=Path(__file__).resolve().parents[2]/'scripts/neral_control_helios.py'
    spec=importlib.util.spec_from_file_location('_tools_only_start',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.main([])


def test_project_capability_uses_portable_reader_support_not_posix_only(helios,monkeypatch,tmp_path):
    class PortableReader:
        @staticmethod
        def supported():return True
        def __init__(self,root):self.root=root
        def read(self,path):return {'path':path,'text':'safe source'}
    monkeypatch.setattr('neral_control.helios.ProjectReader',PortableReader)
    monkeypatch.delattr('neral_control.helios.os.O_NOFOLLOW',raising=False)
    install(helios,project_root=tmp_path)
    caps=client(helios).get('/api/neral/capabilities').get_json()
    assert next(t for t in caps['tools'] if t['name']=='project_files')['available'] is True
