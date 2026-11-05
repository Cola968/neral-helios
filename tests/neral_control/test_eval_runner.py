"""CPU contract fixtures, never real model-quality evidence."""
import importlib.util
from pathlib import Path
import pytest


def api():
    path=Path(__file__).resolve().parents[2]/'scripts/run_neral_abcd_eval.py'
    assert path.is_file(), 'Missing executable A/B/C/D runner'
    spec=importlib.util.spec_from_file_location('runner_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fixture(module,monkeypatch):
    manifest,_=comparison_inputs()
    tasks=[{'id':f'case-{i}','prompt':f'prompt-{i}',
            'prompt_hash':manifest['tasks'][i]['prompt_sha256'],
            'rubric_hash':manifest['tasks'][i]['rubric_sha256']} for i in range(100)]
    monkeypatch.setattr(module,'load_corpus',lambda root:(tasks,manifest,{'leakage':{'flagged_pairs':[]}}))
    return tasks,manifest


def comparison_inputs():
    spec=importlib.util.spec_from_file_location('comparison_fixtures',Path(__file__).with_name('test_evaluation.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.inputs()


def test_collect_requires_confirmation_before_any_work(monkeypatch):
    module=api()
    monkeypatch.setattr(module,'plan',lambda *a,**k:pytest.fail('preflight ran before confirmation'))
    with pytest.raises(PermissionError):module.collect('.',confirm_execution=False)


def test_mode_flag_selects_cpu_plan(monkeypatch,capsys):
    import json
    module=api()
    monkeypatch.setattr(module,'plan',lambda *a,**kw:{'model_started':False,'mode':'plan'})
    module.main(['--mode','plan'])
    assert json.loads(capsys.readouterr().out)=={'model_started':False,'mode':'plan'}


def test_conflicting_mode_forms_are_rejected(monkeypatch):
    module=api()
    monkeypatch.setattr(module,'collect',lambda *a,**k:pytest.fail('conflicting CLI reached inference'))
    with pytest.raises(SystemExit) as e:module.main(['plan','--mode','collect','--confirm-execution'])
    assert e.value.code==2


def test_abbreviated_confirmation_does_not_authorize_execution(monkeypatch):
    module=api()
    monkeypatch.setattr(module,'collect',lambda *a,**k:pytest.fail('abbreviated confirmation reached execution'))
    with pytest.raises(SystemExit) as e:module.main(['--mode','collect','--conf'])
    assert e.value.code==2


def test_plan_cpu_skips_unverified_adapter_without_loader(monkeypatch):
    module=api();fixture(module,monkeypatch)
    result=module.plan('.')
    assert result['conditions']['A']['settings']['tools']==[]
    assert result['conditions']['B']['settings']['lora_adapter_id'] is None
    assert result['conditions']['C']['status']=='SKIPPED_NO_VERIFIED_ADAPTER'
    assert result['conditions']['D']['status']=='SKIPPED_NO_VERIFIED_ADAPTER'
    assert result['model_started'] is False


def test_missing_adapter_never_loads_or_fabricates_outputs(monkeypatch):
    module=api();fixture(module,monkeypatch)
    result=module.collect('.',confirm_execution=True,conditions=('C','D'),
        backend_factory=lambda *a,**k:pytest.fail('missing adapter reached loader'))
    assert all(v['results']==[] for v in result['conditions'].values())
    assert result['conditions']['C']['status']=='SKIPPED_NO_VERIFIED_ADAPTER'


def test_collect_records_real_backend_contract_and_closes_it(monkeypatch):
    module=api();fixture(module,monkeypatch);closed=[]
    class Backend:
        execution_kind='contract_fixture'
        def run(self,task,name,setting,limits):
            assert not setting['tools'] if name=='A' else setting['tools']
            return dict(text='answer',completion_status='complete',tool_provenance=[],
                model_calls=1,input_tokens=None,output_tokens=None,resources={})
        def close(self):closed.append(True)
    result=module.collect('.',confirm_execution=True,conditions=('A',),
                          backend_factory=lambda *a,**k:Backend())
    assert len(result['conditions']['A']['results'])==100 and closed==[True]
    assert result['conditions']['A']['results'][0]['input_tokens'] is None
    assert result['execution_kind']=='contract_fixture'


def test_scoring_binds_review_to_exact_output_and_preserves_skips(monkeypatch,tmp_path):
    module=api();fixture(module,monkeypatch)
    class Backend:
        execution_kind='contract_fixture'
        def run(self,*args):return dict(text='answer',completion_status='complete',
            tool_provenance=[],model_calls=1,input_tokens=3,output_tokens=2,resources={})
        def close(self):pass
    collected=module.collect('.',confirm_execution=True,conditions=('A','C','D'),
                             backend_factory=lambda *a,**k:Backend())
    reviews={'A':[{'id':r['id'],'status':'fail','unsupported_claims':0,
        'failure_causes':['reasoning'],'reviewer':'contract-fixture',
        'prompt_hash':r['prompt_hash'],'rubric_hash':r['rubric_hash'],
        'answer_sha256':r['answer_sha256']} for r in collected['conditions']['A']['results']]}
    report=module.score(collected,reviews)
    assert report['target']['rate'] is None and report['target']['denominator']==100
    assert report['conditions']['D']['status']=='SKIPPED_NO_VERIFIED_ADAPTER'
    assert report['model_quality_gain_measured'] is False
    assert 'SKIPPED_NO_VERIFIED_ADAPTER' in module.markdown(report)
    import json
    source=tmp_path/'collection.json';review_file=tmp_path/'reviews.json'
    source.write_text(json.dumps(collected));review_file.write_text(json.dumps(reviews))
    output=tmp_path/'report.json';markdown=tmp_path/'report.md'
    module.main(['--mode','report','--collection',str(source),'--reviews',str(review_file),
                 '--output',str(output),'--markdown-output',str(markdown)])
    assert json.loads(output.read_text())['target']['denominator']==100
    assert 'SKIPPED_NO_VERIFIED_ADAPTER' in markdown.read_text(encoding='utf8')
    reviews['A'][0]['answer_sha256']='f'*64
    with pytest.raises(ValueError):module.score(collected,reviews)


def test_failed_loader_is_preserved_without_synthetic_case_outputs(monkeypatch):
    module=api();fixture(module,monkeypatch)
    def failed(*a,**kw):raise RuntimeError('private exception text must not escape')
    result=module.collect('.',confirm_execution=True,conditions=('A',),backend_factory=failed)
    assert result['conditions']['A']['status']=='FAILED_RUNTIME_UNAVAILABLE'
    assert result['conditions']['A']['results']==[]
    assert 'private exception' not in str(result)


def test_abcd_target_reports_twenty_percent_and_regression_separately(monkeypatch):
    import copy
    module=api();fixture(module,monkeypatch)
    prepared=module.plan('.');_,settings=comparison_inputs()
    for name in 'ABCD':
        prepared['conditions'][name].update(status='PLANNED_REQUIRES_EXPLICIT_EXECUTION',
                                           settings=settings[name]['settings'])
    monkeypatch.setattr(module,'plan',lambda *a,**kw:copy.deepcopy(prepared))
    class Backend:
        execution_kind='contract_fixture'
        def run(self,*a):return dict(text='fixture',completion_status='complete',
            tool_provenance=[],model_calls=1,input_tokens=2,output_tokens=1,resources={})
        def close(self):pass
    collected=module.collect('.',confirm_execution=True,backend_factory=lambda *a,**kw:Backend())
    reviews={}
    for name in 'ABCD':
        reviews[name]=[]
        for i,row in enumerate(collected['conditions'][name]['results']):
            full=(name=='A' and i<20) or (name=='D' and 1<=i<36)
            reviews[name].append(dict(id=row['id'],status='full' if full else 'fail',
                unsupported_claims=0,failure_causes=[] if full else ['reasoning'],reviewer='fixture',
                prompt_hash=row['prompt_hash'],rubric_hash=row['rubric_hash'],answer_sha256=row['answer_sha256']))
    report=module.score(collected,reviews)
    assert report['target']==dict(numerator=16,denominator=80,rate=.2,percent=20.,achieved=True)
    assert report['regressions_D_vs_A']==['case-0']
    assert report['model_quality_gain_measured'] is False


def test_output_exists_refuses_before_collection(monkeypatch,tmp_path):
    module=api();path=tmp_path/'existing.json';path.write_text('preserve')
    monkeypatch.setattr(module,'collect',lambda *a,**kw:pytest.fail('model execution attempted'))
    with pytest.raises(SystemExit) as e:
        module.main(['collect','--confirm-execution','--output',str(path)])
    assert e.value.code==2 and path.read_text()=='preserve'


def test_a_is_plain_generation_without_tools_or_controller(monkeypatch):
    import sys
    from types import SimpleNamespace
    module=api();captured=[]
    class Stop:
        def __init__(self,*a):pass
        def is_set(self):return False
    monkeypatch.setattr('neral_control.providers._GenerationStop',Stop,raising=False)
    monkeypatch.setattr('neral_control.registry.default_registry',lambda *a:pytest.fail('A constructed tool registry'))
    monkeypatch.setattr('neral_control.controller.ReasoningController',lambda *a:pytest.fail('A constructed controller'))
    cuda=SimpleNamespace(reset_peak_memory_stats=lambda *_:None,
        max_memory_allocated=lambda *_:0,max_memory_reserved=lambda *_:0)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(manual_seed=lambda *_:None,cuda=cuda))
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):captured.extend(messages);return [1]
    backend=module._LocalBackend.__new__(module._LocalBackend)
    backend.runtime=SimpleNamespace(tokenizer=Tokenizer(),finish_reason='eos',
        generate=lambda *a:[1,2],decode=lambda *_:'plain answer')
    backend.provider=SimpleNamespace();backend.identity={}
    result=backend.run({'prompt':'hello'},'A',{'system_prompt':'shared','seed':42},
                       {'seconds':20,'model_calls':1})
    assert captured==[{'role':'system','content':'shared'},{'role':'user','content':'hello'}]
    assert result['model_calls']==1 and result['tool_provenance']==[]
