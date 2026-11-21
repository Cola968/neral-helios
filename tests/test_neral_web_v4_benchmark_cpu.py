import json
import unittest
from scripts.neral_web_v4_benchmark import CASES,prompt,check_response
from neral_evidence_agent import _comparison_entities

class WebV4BenchmarkContracts(unittest.TestCase):
    def test_20_new_questions_parse_to_exact_product_names(self):
        self.assertEqual(len(CASES),20)
        self.assertEqual(len({prompt(c) for c in CASES}),20)
        for case in CASES:
            with self.subTest(case=case):
                self.assertEqual(_comparison_entities(prompt(case)),case[:2])
                self.assertLessEqual(len(prompt(case)[:160]),160)

    def test_grade_requires_both_newly_read_sources_and_correct_engine(self):
        case=CASES[0]
        body={'tool_verified':True,'error_code':None,
              'sources':[{'url':'https://mozilla.org/a','page_read':True},
                         {'url':'https://chromium.org/b','page_read':True}],
              'text':'Chrome: Blink\nFirefox: Gecko\nHinweis: Desktop; iOS kann abweichen.\n[web1] [web2]'}
        self.assertTrue(check_response(case,body))
        self.assertFalse(check_response(case,{**body,'text':body['text'].replace('Gecko','WebKit')}))
        self.assertFalse(check_response(case,{**body,'sources':[body['sources'][0]]}))
        self.assertFalse(check_response(case,{**body,'sources':[{**body['sources'][0],'page_read':False},body['sources'][1]]}))
        self.assertFalse(check_response(case,{**body,'tool_verified':False}))

if __name__=='__main__':unittest.main()
