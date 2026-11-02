import json
import threading
import time
from unittest.mock import patch

from neral_control.registry import ToolContext, ToolRegistry, ToolSpec, default_registry


def context(*permissions, **kw):
    return ToolContext(frozenset(permissions), **kw)


def test_manifest_optional_availability_and_provider_schema():
    registry = default_registry()
    manifest = {x['name']: x for x in registry.describe()}
    assert set(manifest) == {'calculator', 'code_check', 'project_files', 'web_search'}
    assert not manifest['web_search']['available']
    assert registry.names() == ('calculator', 'code_check')
    assert [s['function']['name'] for s in registry.schemas(['code_check'])] == ['code_check']
    json.dumps(registry.describe(), allow_nan=False)


def test_builtin_reuses_existing_bounded_helpers_and_denies_without_permission():
    registry = default_registry()
    assert registry.execute('calculator', {'expression': '17*23'}, context())['status'] == 'denied'
    result = registry.execute('calculator', {'expression': '17*23'}, context('math'))
    assert result['status'] == 'ok'
    assert result['output']['result'] == '391'
    with patch('neral_assistant._code_check', return_value='{"executed":false,"finding_count":0}') as helper:
        result = default_registry().execute('code_check', {'code': 'pass'}, context('code_check'))
        assert result['status'] == 'ok'
        helper.assert_called_once_with({'function': {'name': 'code_check', 'arguments': {'code': 'pass'}}})
    assert result['provenance']['executed'] is False


def test_invalid_inputs_unknown_tool_and_missing_optional_never_execute():
    registry = default_registry()
    for args in ({'expression':'2+2', 'shell':'x'}, {'expression': True}, {'expression': 'x'*121}):
        assert registry.execute('calculator', args, context('math'))['status'] == 'error'
    assert registry.execute('shell', {}, context('math'))['status'] == 'unavailable'
    assert registry.execute('web_search', {'query':'hello'}, context('network'))['status'] == 'unavailable'


def test_cancellation_deadline_and_cooperative_postcheck():
    registry = default_registry()
    event = threading.Event(); event.set()
    assert registry.execute('calculator', {'expression':'1'}, context('math', cancel_event=event))['status'] == 'cancelled'
    assert registry.execute('calculator', {'expression':'1'}, context('math', deadline=time.monotonic()-1))['status'] == 'timeout'
    class Search:
        def search(self, query, limit=5, *, cancel_event, deadline):
            cancel_event.set()
            return {'sources': []}
    result = default_registry(search_provider=Search()).execute('web_search', {'query':'hi'}, context('network'))
    assert result['status'] == 'cancelled'
    assert result['output'] is None


def test_optional_handlers_and_no_exception_secrets():
    class Reader:
        def read(self, path): return {'path':path, 'content':'hello'}
    registry = default_registry(project_reader=Reader())
    assert registry.execute('project_files', {'relative_path':'readme.md'}, context('project_read'))['output']['content'] == 'hello'
    class Broken:
        def search(self, *args, **kwargs): raise RuntimeError('private-super-secret')
    result = default_registry(search_provider=Broken()).execute('web_search', {'query':'hi'}, context('network'))
    assert result['status'] == 'error'
    assert 'private-super-secret' not in json.dumps(result)


def test_registry_rejects_unlisted_names_duplicate_and_invalid_schema_output():
    registry = ToolRegistry()
    schema = {'type':'object','properties':{'count':{'type':'integer','minimum':0,'maximum':2}},'required':['count'],'additionalProperties':False}
    spec = ToolSpec('calculator', '1', schema, schema, 'math', 1, lambda args, ctx: {'count': True}, 'test')
    registry.register(spec)
    assert registry.execute('calculator', {'count':1}, context('math'))['status'] == 'error'
    import pytest
    with pytest.raises(ValueError): registry.register(spec)
    with pytest.raises(ValueError): registry.register(ToolSpec('shell','1',schema,schema,'shell',1,lambda a,c:a,'no'))


def test_provider_unavailability_is_not_success_and_redacts_output():
    class Search:
        def search(self,*args,**kwargs): return {'available':False,'error':'provider unavailable'}
    result=default_registry(search_provider=Search()).execute('web_search',{'query':'hi'},context('network'))
    assert result['status']=='unavailable'
    assert result['output'] is None


def test_output_size_and_nonfinite_numbers_fail_closed():
    schema={'type':'object','additionalProperties':True}
    for output in ({'n':float('nan')},{'text':'x'*70000}):
        registry=ToolRegistry()
        registry.register(ToolSpec('calculator','1',schema,schema,'math',1,lambda a,c:output,'test'))
        assert registry.execute('calculator',{},context('math'))['status']=='error'


def test_schema_support_includes_nested_arrays_and_booleans():
    schema={'type':'object','properties':{'items':{'type':'array','maxItems':2,'items':{'type':'boolean'}}},'required':['items'],'additionalProperties':False}
    registry=ToolRegistry()
    registry.register(ToolSpec('calculator','1',schema,schema,'math',1,lambda a,c:a,'test'))
    assert registry.execute('calculator',{'items':[True,False]},context('math'))['status']=='ok'
    for items in ([1],[True,False,True]):
        assert registry.execute('calculator',{'items':items},context('math'))['status']=='error'


def test_calculation_router_exact_whole_request_and_context_controls():
    registry=default_registry()
    result=registry.route_calculation('Berechne 17 mal 23',context('math'))
    assert result['status']=='ok'
    assert '391' in result['output']['text']
    assert result['tool_id']=='calculator'
    assert result['provenance']['source']=='tool:calculator'
    assert registry.route_calculation('Berechne 2+2 und lies meine Dateien',context('math')) is None
    assert registry.route_calculation('hello there',context('math')) is None
    assert registry.route_calculation('Calculate 2+2',context())['status']=='denied'
    event=threading.Event();event.set()
    assert registry.route_calculation('Calculate 2+2',context('math',cancel_event=event))['status']=='cancelled'
    assert registry.route_calculation('Calculate 2+2',context('math',deadline=time.monotonic()-1))['status']=='timeout'
