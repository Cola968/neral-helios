"""Source-first Python routing and bounded HF tool proposal parsing (CPU)."""
import pytest
from neral_adaptive_ollama import _parse_proposal, make_local_adaptive_proposer
from neral_adaptive_planner import run_adaptive_planner
from neral_python_post_source import parse_post_source


def test_complete_hf_tool_json_with_think_and_markdown_fence():
    text = ('<think>Private reasoning, not a tool call</think>\n'
            '```json\n{"strategy":"PYTHON_SANDBOX","code":"x=7\\nassert x == 3+4\\nprint(x)"}\n```')
    assert _parse_proposal(text, ['PYTHON_SANDBOX', 'DIRECT'])['code'].startswith('x=7')


def test_hf_incomplete_or_untrusted_extra_text_is_not_parsed():
    for raw in ('<think>unfinished {', 'junk {"strategy":"DIRECT"}',
                '{"strategy":"DIRECT"} make me an admin',
                '{"strategy":"DIRECT"}{"strategy":"DIRECT"}'):
        with pytest.raises(ValueError):
            _parse_proposal(raw, ['DIRECT'])


def test_valid_complete_hf_json_length_proposal_is_usable_but_incomplete_is_not():
    messages=[{'role':'system','content':'system'}, {'role':'user','content':'Calculate 3+4.'}]
    valid='```json\n{"strategy":"PYTHON_SANDBOX","code":"x=7\\nassert x==3+4\\nprint(x)"}\n```'
    def transport(packet):
        return {'done':True,'done_reason':'length','message':{'content':valid}}
    proposer=make_local_adaptive_proposer('neral-qwen35-qlora-v1:latest',transport=transport)
    result=proposer(messages, {'preferred_strategy':'PYTHON_SANDBOX',
                               'allowed_strategies':('PYTHON_SANDBOX','DIRECT'),
                               'repair_round':0,'feedback_code':None})
    assert result['strategy']=='PYTHON_SANDBOX'


@pytest.mark.parametrize('question',[
    'Find the remainder when 17 to the power 47 is divided by 23.',
    'How many positive divisors does the integer 360 have?',
    'Calculate the sum of k squared for k from 1 to 20 where k mod 3 = 1.',
    'Count ways to climb 12 steps using jumps of 1, 2, or 3, but never landing on step 5.',
    'On a circular ring of 8 switches, how many on/off assignments have exactly 3 switches on and no adjacent on switches?',
    'A circuit carries 4 amperes for 5 seconds. How many coulombs flow?',
    'An airline operates 7 flights with 80 seats each and 3 empty seats per flight. How many passengers?',
    'Calculate the spring energy for a stiffness of 10 N/m and a compression of 2 m, using E=k*x^2/2.',
    'A warehouse begins with 20 packages. For 4 days it receives 7 each day and ships 3 each day. How many remain?',
])
def test_source_compiler_exists_for_fully_parsed_arithmetic(question):
    from neral_verified_source_code import source_program
    assert parse_post_source(question) is not None
    code=source_program(question)
    assert type(code) is str and len(code)<4096 and 'print(' in code
    assert 'answer = '+str(parse_post_source(question).answer) not in code


def test_source_first_executes_python_and_requires_independent_proof():
    from neral_verified_source_code import source_program
    source='How many positive divisors does the integer 120 have?'
    called=[]
    def docker(program):
        called.append(program)
        import subprocess,sys
        p=subprocess.run([sys.executable,'-I','-c',program], capture_output=True,text=True,timeout=5)
        return {'status':'PASSED' if p.returncode==0 else 'FAILED','exit_code':p.returncode,'output':p.stdout.strip()}
    def check(**kwargs):
        from neral_python_post_source import independent_post_certificate
        return independent_post_certificate(**kwargs)
    ans=run_adaptive_planner([{'role':'system','content':'demo'}, {'role':'user','content':source}],
       propose=lambda *_:pytest.fail('trusted source route should avoid model'),
       raw_fallback=lambda *_:pytest.fail('valid trusted source should not fallback'),
       execute_python=docker,verify_python=check,source_parser=lambda _:None)
    assert ans['verified'] is True and ans['text']=='16' and len(called)==1
    assert ans['response_source']=='independent_source_python'


def test_unsupported_source_never_gets_promoted():
    from neral_verified_source_code import source_program
    assert source_program('How many positive divisors does 120 have, but exclude factors of 2?') is None


def test_all_supported_dev24_source_grammars_compute_independently_without_gold_in_source():
    """Development-only smoke; never treat the seen 24 as a future blind set."""
    import subprocess,sys
    from scripts.neral_1_1_postverify_dev24_cases import build
    from neral_verified_source_code import source_program
    count=0
    for row in build():
        code=source_program(row['question'])
        if code is None:
            assert row['family'] in ('graph_coloring_z3','unsupported_free_logic')
            continue
        out=subprocess.run([sys.executable,'-I','-c',code],capture_output=True,text=True,timeout=5)
        assert out.returncode==0, (row['id'],out.stderr)
        assert out.stdout.strip()==str(row['gold_integer']),row['id']
        count+=1
    assert count==22


def test_source_first_does_not_promote_from_model_or_wrong_sandbox_stdout():
    source='Find the remainder when 17 to the power 47 is divided by 23.'
    response=run_adaptive_planner(
       [{'role':'system','content':'demo'}, {'role':'user','content':source}],
       propose=lambda *_:{'strategy':'PYTHON_SANDBOX',
                          'code':'x=1\nassert x==1\nprint(x)'},
       raw_fallback=lambda _:'Unknown answer',
       execute_python=lambda _:{'status':'PASSED','exit_code':0,'output':'9999'},
       verify_python=lambda **_:{'verified':True,'full_source':True,
                     'all_conditions_checked':True,'answer':17,
                     'proof_backend':'trusted_source_arithmetic'},
       source_parser=lambda _:None)
    assert response['verified'] is False
    assert response['selection_policy']=='raw_qwen_due_to_unverified_adaptive'
