"""Explicit Docker-backed CPU integration: source-verified Python and Z3.

Runs only trusted DEV24 source examples. No GPU/model simulation is claimed;
model proposal is deliberately forbidden for already certified source tasks.
"""
from __future__ import annotations
import json
from scripts.neral_1_1_postverify_dev24_cases import build
from neral_adaptive_planner import run_adaptive_planner,run_python_in_docker
from neral_python_source_router import verify_python_full_source


def main():
    cases=build()
    families=('modular_exponent','forbidden_rooks','original_python_program','electric_charge','graph_coloring_z3')
    tested=[]
    for family in families:
        case=next(c for c in cases if c['family']==family)
        docker_calls=[]
        def docker(code):
            docker_calls.append('called')
            return run_python_in_docker(code)
        def no_model(*_):
            raise RuntimeError('Pre-certified full source must not rely on an HF completion')
        answer=run_adaptive_planner(
            [{'role':'system','content':'NERAL tool integration smoke'},
             {'role':'user','content':case['question']}],
            no_model,no_model,docker,verify_python_full_source)
        if (answer.get('verified') is not True
                or answer.get('safe_to_promote') is not True
                or answer.get('text')!=str(case['gold_integer'])
                or len(docker_calls)!=(0 if family=='graph_coloring_z3' else 1)):
            raise RuntimeError('Docker/Z3 integration failure: '+family+' status='+str(answer.get('selection_policy')))
        tested.append({'family':family,'route':answer.get('planner_route'),
                       'docker_calls':len(docker_calls),'backend':answer.get('proof_backend')})
    print('NERAL_HF_SOURCE_CPU_DOCKER_Z3_PASS '+json.dumps(tested,sort_keys=True),flush=True)

if __name__=='__main__': main()
