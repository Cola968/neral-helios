import unittest
from fractions import Fraction
from scripts.neral_math_v3_balanced_pack import (
    new_equations, build_plan, proof_equation, SIZE, CORE, LEGACY, ORIGINAL_EQ, NEW_EQ, OTHER,
)

class BalancedPackTests(unittest.TestCase):
    def test_reproducible_equations_all_independently_verified(self):
        a=new_equations(); b=new_equations()
        self.assertEqual(a,b)
        self.assertEqual(len(a),100)
        self.assertTrue(all(proof_equation(x) for x in a))
        self.assertEqual(len({x['prompt'] for x in a}),100)
        self.assertTrue(any(x['mode']=='explain' for x in a))
        self.assertTrue(any(x['mode']=='numeric' for x in a))
    def test_injected_answer_rejected(self):
        example=dict(new_equations()[0]); example['expected']='999999'
        self.assertFalse(proof_equation(example))
    def test_stratified_balance(self):
        self.assertEqual(sum([CORE,LEGACY,ORIGINAL_EQ,NEW_EQ,OTHER]),SIZE)
        self.assertEqual((CORE,LEGACY,ORIGINAL_EQ,NEW_EQ,OTHER,SIZE),(86,40,70,100,88,384))
    def test_planned_counts_and_no_promotion(self):
        p=build_plan()
        self.assertEqual(p['examples'],384)
        self.assertFalse(p['training_performed'])
        self.assertFalse(p['release_approved'])
        self.assertEqual(p['core_percent'],round(100*86/384,2))

if __name__ == '__main__':unittest.main()
