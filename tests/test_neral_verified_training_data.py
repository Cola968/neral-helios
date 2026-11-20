import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from neral_verified_training import verifier_ok, append_verified_example, import_benchmark

class VerifiedTrainingContracts(unittest.TestCase):
    def test_fail_closed_on_unverified_or_proof_free_model_output(self):
        good={'agent_checked':True,'tool_verified':True,'verified':True,
              'error_code':None,'text':'Ergebnis: 8','tool_used':'python_sandbox'}
        self.assertTrue(verifier_ok(good))
        for patch in ({'verified':False},{'tool_verified':False},
                      {'agent_checked':False},{'error_code':'sandbox_error'}):
            self.assertFalse(verifier_ok({**good,**patch}))
        self.assertFalse(verifier_ok({'tool_verified':True,'agent_checked':True,
                                       'verified':False,'text':'8','error_code':None}))

    def test_literal_web_proof_requires_read_https_sources(self):
        base={'agent_checked':True,'tool_verified':True,'verified':False,'error_code':None,
              'text':'A: X\nB: Y','verification_status':'literal_property_verified_model_not_accepted',
              'sources':[{'page_read':True,'url':'https://a.example/x'},
                         {'page_read':True,'url':'https://b.example/y'}]}
        self.assertTrue(verifier_ok(base))
        self.assertFalse(verifier_ok({**base,'sources':[base['sources'][0]]}))
        self.assertFalse(verifier_ok({**base,'sources':[base['sources'][0],
                  {'url':'https://b.example/y','page_read':False}]}))

    def test_deduplicates_single_instruction_and_keeps_user_data_local(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data'/'verified_training_dataset.jsonl'
            result={'agent_checked':True,'tool_verified':True,'verified':True,
                    'error_code':None,'text':'Ergebnis: 8','tool_used':'python_sandbox'}
            self.assertTrue(append_verified_example('4+4?',result,path))
            self.assertFalse(append_verified_example('4+4?',result,path))
            self.assertFalse(append_verified_example('4+4?',{**result,'text':'wrong'},path))
            self.assertEqual(len(path.read_text().splitlines()),1)
            saved=json.loads(path.read_text())
            self.assertEqual(saved['instruction'],'4+4?')
            self.assertEqual(saved['response'],'Ergebnis: 8')
            self.assertTrue(saved['meta']['verifier_ok'])
            self.assertFalse(append_verified_example('api_key=sk-proj-SECRET',result,path))
            self.assertEqual(len(path.read_text().splitlines()),1)

    def test_benchmark_grader_pass_is_not_claimed_as_live_verifier(self):
        import hashlib
        records=[]
        for i,pass_ok in enumerate((True,True,False)):
            question='Frage '+str(i)
            records.append({'case':{'id':'c'+str(i),'category':'knowledge',
                        'question':question,
                        'question_sha256':hashlib.sha256(question.encode()).hexdigest()},
                        'observed':{'text':'Antwort '+str(i),'tool_verified':False,'verified':False},
                        'grader':{'strict_e2e_pass':pass_ok,'answer_correct':pass_ok},
                        'http_status':200})
        details='\n'.join(json.dumps(r) for r in records)+'\n'
        summary={'cases':3,'detail_count':3,'strict_e2e_pass':2,
                 'details_sha256':hashlib.sha256(details.encode()).hexdigest(),
                 'source_sha':'abc','status':'COMPLETE'}
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'seed.jsonl'
            count=import_benchmark(details.encode(),summary,target,
                                   expected_source='abc',expected_pass=2)
            self.assertEqual(count,2)
            rows=[json.loads(x) for x in target.read_text().splitlines()]
            self.assertEqual(len(rows),2)
            self.assertFalse(rows[0]['meta']['verifier_ok'])
            self.assertEqual(rows[0]['meta']['verification_kind'],'benchmark_strict_e2e')
            with self.assertRaises(ValueError):
                import_benchmark(details.encode()+b'\n',summary,target,
                                  expected_source='abc',expected_pass=2)

if __name__=='__main__':unittest.main()
