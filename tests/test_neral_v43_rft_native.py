"""Separate required native CPU gate (not mocked; missing dependencies fail)."""
import unittest
from scripts import neral_v43_rft_cpu as r

class NativeGate(unittest.TestCase):
    def test_native_math_and_logic(self):
        import sympy,z3
        for case in r.catalog('8'*64,12):
            if case['kind'] in ('sympy','z3'):
                gold=str(case['gold']);wrong=str(case['gold']+1) if case['kind']=='sympy' else ('Nein' if gold=='Ja' else 'Ja')
                self.assertTrue(r.verify_candidate(case,gold)[0])
                ok,evidence=r.verify_candidate(case,wrong)
                self.assertFalse(ok);self.assertTrue(evidence)
    def test_native_pytest_hypothesis_rejection(self):
        import pytest,hypothesis
        case=next(x for x in r.catalog('9'*64,3) if x['kind']=='code')
        threshold,rate1,rate2,base=case['oracle_params']
        gold=f'def solve(x):\n    n=max(x,0)\n    return {base}+min(n,{threshold})*{rate1}+max(n-{threshold},0)*{rate2}'
        wrong='def solve(x):\n    return 0'
        self.assertTrue(r.verify_candidate(case,gold)[0])
        ok,evidence=r.verify_candidate(case,wrong)
        self.assertFalse(ok);self.assertTrue(evidence)
        key=r.digest(r.repair_prompt(case,wrong,evidence))
        result=r.select(case,[{'draft':wrong,'repair':gold,'repair_input_sha256':key}])
        self.assertTrue(result['meta']['repair'])
if __name__=='__main__':unittest.main()
