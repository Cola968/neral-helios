import threading
import pytest
from neral_control.providers import ProviderReply
from neral_control.registry import default_registry

class Provider:
 identity={'source':'external-model','base_model':'Qwen/test','model_family':'qwen-test','runtime_id':'r1','lora_weights_version':None,'active_adapter_id':None}
 def __init__(self,*replies):self.replies=list(replies);self.calls=[]
 def complete(self,messages,tools,options,event,on_progress=None):
  self.calls.append((messages.copy(),tools,options.copy()))
  value=self.replies.pop(0)
  if isinstance(value,Exception):raise value
  return value

def run(provider=None,**kw):
 from neral_control.controller import ReasoningController
 return ReasoningController(default_registry(),provider).run('Prüfe und antworte.',**kw)

def call(name='calculator',**arguments):return ProviderReply(tool_calls=[{'function':{'name':name,'arguments':arguments}}],generated_tokens=8,input_tokens=12)

def test_fast_deterministic_math_never_generates_and_reports_system_gain():
 from neral_control.controller import ReasoningController
 provider=Provider()
 out=ReasoningController(default_registry(),provider).run('Berechne 17 mal 23',permissions=frozenset({'math'}))
 assert out['text']=='391' and out['source']=='tool:calculator'
 assert out['metrics']['model_calls']==0 and out['metrics']['tool_calls']==1
 assert provider.calls==[] and out['verification']['learned_capability_proven'] is False

def test_explicit_static_code_check_is_not_execution():
 from neral_control.controller import ReasoningController
 out=ReasoningController(default_registry(),None).run('CODE_CHECK:\ndef broken(:\n pass',allowed_tools=('code_check',),permissions=frozenset({'code_check'}))
 assert out['status']=='success' and out['tool_results'][0]['output']['syntax_ok'] is False
 assert out['tool_results'][0]['output']['executed'] is False
 assert out['metrics']['model_calls']==0

@pytest.mark.parametrize('profile,expected',[('fast',1),('think',2),('deep',4)])
def test_profile_limits_distinct_from_native_thinking(profile,expected):
 p=Provider(ProviderReply(content='Fertig',generated_tokens=3,input_tokens=4))
 out=run(p,profile=profile)
 assert out['status']=='success' and len(p.calls)==1
 assert out['limits']['model_calls']==expected
 assert p.calls[0][2]['profile']=='standard' and p.calls[0][2]['native_think'] is False


def test_think_executes_existing_calculator_then_final_answer():
 p=Provider(call(expression='17*23'),ProviderReply(content='Das Ergebnis ist 391.',generated_tokens=4,input_tokens=18))
 out=run(p,profile='think',permissions=frozenset({'math'}))
 assert out['status']=='success' and out['tools_used']==['calculator']
 assert out['metrics']['model_calls']==2 and out['metrics']['tool_calls']==1
 assert '391' in p.calls[1][0][-1]['content']
 assert out['metrics']['input_tokens']==30 and out['metrics']['output_tokens']==12


def test_deep_two_distinct_tools_and_final_preserves_provenance():
 p=Provider(call(expression='2+2'),call('code_check',code='pass'),ProviderReply(content='4; statisch geprüft.'))
 out=run(p,profile='deep',permissions=frozenset({'math','code_check'}))
 assert out['tools_used']==['calculator','code_check'] and out['metrics']['model_calls']==3
 assert out['base_model']=='Qwen/test' and out['runtime_id']=='r1'

@pytest.mark.parametrize('reply',[
 ProviderReply(tool_calls='bad'),
 ProviderReply(content=42,tool_calls=[{'function':{'name':'calculator','arguments':{'expression':'1'}}}]),
 ProviderReply(tool_calls=[{'function':{'name':'calculator','arguments':{'expression':'1'}}}],incomplete=True),
 ProviderReply(tool_calls=[{'function':{'name':'shell','arguments':{}}}]),
 ProviderReply(tool_calls=[{'function':{'name':'calculator','arguments':{'expression':'1'}}}]*2),
])
def test_invalid_or_incomplete_tool_reply_never_executes(reply):
 out=run(Provider(reply),profile='deep',permissions=frozenset({'math'}))
 assert out['status']=='error' and out['metrics']['tool_calls']==0


def test_repeat_tool_is_denied_before_second_execution():
 p=Provider(call(expression='2+2'),call(expression='2+2'))
 out=run(p,profile='deep',permissions=frozenset({'math'}))
 assert out['status']=='error' and out['metrics']['tool_calls']==1 and len(p.calls)==2


def test_tools_disabled_are_not_advertised_or_executed():
 p=Provider(call(expression='2+2'))
 out=run(p,allowed_tools=(),permissions=frozenset({'math'}))
 assert out['status']=='error' and p.calls[0][1]==[] and out['metrics']['tool_calls']==0


def test_precancel_and_stale_context_stop_without_generation():
 p=Provider();event=threading.Event();event.set()
 out=run(p,event=event)
 assert out['status']=='cancelled' and p.calls==[]
 with pytest.raises(ValueError):run(p,runtime_id='other-runtime')


def test_length_answer_preserved_as_incomplete_not_success():
 out=run(Provider(ProviderReply(content='Halbe Antwort',incomplete=True,finish_reason='length')))
 assert out['status']=='incomplete' and out['incomplete'] is True


def test_provider_error_is_counted_without_private_details():
 out=run(Provider(RuntimeError('private secret path')))
 assert out['status']=='error' and out['metrics']['failed_steps']
 assert 'private secret path' not in str(out)


def test_no_model_is_honestly_unavailable():
 out=run()
 assert out['status']=='unavailable' and out['metrics']['model_calls']==0


def test_context_budget_rejected_before_generation():
 p=Provider(ProviderReply(content='never'))
 p.count_messages=lambda messages,tools:3000
 p.identity['model_family']='qwen2.5-coder'
 out=run(p)
 assert out['status']=='error' and p.calls==[] and out['metrics']['model_calls']==0


def test_unperformed_test_claim_is_flagged_not_quality_proof():
 out=run(Provider(ProviderReply(content='Alle Tests bestanden.')))
 assert out['verification']['unsupported_claims']==['test_execution_unavailable']
 assert out['verification']['correctness_verified'] is False


@pytest.mark.parametrize('kind',['local','ollama'])
def test_same_controller_uses_both_real_provider_adapters_and_same_tool(kind):
 from neral_control.controller import ReasoningController
 from neral_control.providers import LocalQwenProvider,OllamaProvider
 if kind=='local':
  class Runtime:
   identity='qwen2.5-coder:'+'a'*64
   finish_reason='eos'
   def __init__(self):
    self.outputs=iter(['{"tool_call":{"name":"calculator","arguments":{"expression":"17*23"}}}','{"answer":"391"}'])
    self.tokenizer=type('Tokenizer',(),{'apply_chat_template':lambda self,*a,**k:[1]*10})()
   def generate(self,*args):self.text=next(self.outputs);return [1,2,3]
   def decode(self,ids):return self.text
  provider=LocalQwenProvider(Runtime())
 else:
  sent=[]
  def transport(payload):
   sent.append(payload)
   msg={'role':'assistant','content':'391'}
   if len(sent)==1:msg={'role':'assistant','content':'','tool_calls':[{'id':'safe-id','type':'function','index':0,'function':{'name':'calculator','arguments':{'expression':'17*23'}}}]}
   return {'model':'qwen3.5:4b','done':True,'done_reason':'stop','message':msg,'eval_count':3}
  provider=OllamaProvider('qwen3.5:4b',transport=transport)
 out=ReasoningController(default_registry(),provider).run('Rechne den Teil des Problems.',profile='think',permissions=frozenset({'math'}))
 assert out['status']=='success' and out['text']=='391' and out['tools_used']==['calculator']
 assert out['metrics']['model_calls']==2 and out['active_adapter_id'] is None
 assert out['model_family']==('qwen2.5-coder' if kind=='local' else 'qwen3.5')


def test_cancel_after_generation_keeps_actual_usage_and_never_executes_tool():
 event=threading.Event()
 class Cancels(Provider):
  def complete(self,*args):
   event.set()
   return ProviderReply(tool_calls=[{'function':{'name':'calculator','arguments':{'expression':'2+2'}}}],generated_tokens=3,input_tokens=10,cancelled=True)
 out=run(Cancels(),event=event,permissions=frozenset({'math'}))
 assert out['status']=='cancelled' and out['metrics']['tool_calls']==0
 assert out['metrics']['output_tokens']==3 and out['metrics']['input_tokens']==10


def test_failed_provider_attempt_does_not_claim_a_generated_answer():
 out=run(Provider(RuntimeError('transport unavailable')))
 assert out['metrics']['model_calls']==1
 assert out['model_generated'] is False


def test_new_profile_contracts_and_structured_result():
 from neral_control.controller import ControllerRunResult
 for profile,models,tools in [('FAST',1,1),('THINK',2,2),('DEEP',4,4)]:
  out=run(Provider(ProviderReply(content='Antwort',generated_tokens=3,input_tokens=5)),profile=profile)
  assert isinstance(out,ControllerRunResult)
  assert out.profile==profile.lower() and out['limits']['model_calls']==models and out['limits']['tool_calls']==tools
  assert out.model_calls==1 and out.tool_calls==0
  assert out.tokens_used=={'input':5,'output':3,'total':8}
  assert out.cancelled is False and out.timed_out is False
  assert 'lora_adapter_id' in out.provenance and out.provenance['lora_adapter_id'] is None


def test_think_can_execute_two_distinct_tools_before_final_second_call():
 calls=[{'function':{'name':'calculator','arguments':{'expression':'2+2'}}},
        {'function':{'name':'code_check','arguments':{'code':'pass'}}}]
 p=Provider(ProviderReply(tool_calls=calls),ProviderReply(content='4 und statisch geprüft.'))
 out=run(p,profile='think',permissions=frozenset({'math','code_check'}))
 assert out['status']=='success' and out['model_calls']==2 and out['tool_calls']==2


def test_deep_allows_different_arguments_same_tool_but_not_identical_canonical_call():
 p=Provider(call(expression='2+2'),call(expression='3+3'),ProviderReply(content='4 und 6'))
 out=run(p,profile='deep',permissions=frozenset({'math'}))
 assert out['status']=='success' and out['tool_calls']==2
 p=Provider(call(expression='2+2'),call(expression='2+2'))
 out=run(p,profile='deep',permissions=frozenset({'math'}))
 assert out['status']=='error' and out['tool_calls']==1


def test_deep_four_tool_budget_and_four_model_calls_are_real_caps():
 calls=[{'function':{'name':'calculator','arguments':{'expression':f'{i}+1'}}} for i in range(4)]
 p=Provider(ProviderReply(tool_calls=calls[:2]),ProviderReply(tool_calls=calls[2:3]),ProviderReply(tool_calls=calls[3:]),ProviderReply(content='Fertig'))
 out=run(p,profile='deep',permissions=frozenset({'math'}))
 assert out['status']=='success' and out['model_calls']==4 and out['tool_calls']==4


def test_deep_allows_only_one_changed_argument_correction_after_verified_failure():
 p=Provider(call('code_check',code='def broken(:'),call('code_check',code='def still_broken(:'),call('code_check',code='pass'))
 out=run(p,profile='deep',permissions=frozenset({'code_check'}))
 assert out['status']=='error' and out['metrics']['correction_steps']==1
 assert out['tool_calls']==2


def test_complete_batch_validation_precedes_first_tool_execution():
 p=Provider(ProviderReply(tool_calls=[{'function':{'name':'calculator','arguments':{'expression':'2+2'}}},
                                    {'function':{'name':'code_check','arguments':{'code':'pass','shell':'no'}}}]))
 out=run(p,profile='think',permissions=frozenset({'math','code_check'}))
 assert out['status']=='error' and out['tool_calls']==0


def test_explicit_context_manager_is_connected_to_exact_provider_counter():
 p=Provider(ProviderReply(content='Aus Quelle beantwortet.'))
 p.identity=p.identity.copy();p.identity['model_family']='qwen2.5-coder'
 p.count_messages=lambda messages,tools:sum(len(m['content'])//8+1 for m in messages)
 from neral_control.controller import ReasoningController
 out=ReasoningController(default_registry(),p).run('Was sagt die Quelle?',project_records=[{'path':'a.py','text':'value=42','sha256':'a'*64}],permissions=frozenset())
 assert out['status']=='success' and out['context_provenance'][0]['path']=='a.py'
 assert 'untrusted' in p.calls[0][0][-1]['content']


def test_deadline_after_model_is_reported_separately_from_cancel():
 import time
 class Clock:
  now=time.monotonic()
  def __call__(self):return self.now
 clock=Clock()
 class Slow(Provider):
  def complete(self,*args):
   clock.now+=100
   return ProviderReply(content='late',generated_tokens=2)
 from neral_control.controller import ReasoningController
 out=ReasoningController(default_registry(),Slow(),clock=clock).run('Beantworte diese Frage')
 assert out['timed_out'] is True and out['cancelled'] is False
 assert out['tokens_used']['output']==2


def test_missing_output_token_report_remains_unknown_with_hard_call_limits():
    p=Provider(ProviderReply(content='answer',output_tokens_reported=False))
    result=run(p)
    assert result.tokens_used['output'] is None and result.tokens_used['total'] is None
    assert result.model_calls==1
