import hashlib,json,random,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from scripts import neral_v43_rft_cpu as r

class Tests(unittest.TestCase):
    def test_catalog_reproducible_and_unique(self):
        a=r.catalog('4'*64, 40)
        self.assertEqual(a,r.catalog('4'*64,40))
        self.assertEqual(len({x['id'] for x in a}),40)
        self.assertEqual(set(x['kind'] for x in a),{'sympy','z3','code'})
    def test_reference_math(self):
        for x in r.catalog('4'*64,40):
            if x['kind']=='sympy':
                self.assertTrue(r.math_ok(x,str(x['gold'])))
                self.assertFalse(r.math_ok(x,str(x['gold']+1)))
    def test_exact_replay(self):
        def row(i,kind,cat):return {'instruction':str(i),'response':'ok','meta':{'kind':kind,'category':cat,'task_hash':r.digest(str(i))}}
        rows=[row(i,'rft','math') for i in range(20)]
        replay=[row(i+20,'replay',['everyday','knowledge','math','code'][i%4]) for i in range(16)]
        out=r.mix(rows,replay,20,set(),seed=1)
        self.assertEqual(sum(x['meta']['kind']=='replay' for x in out),6)
        self.assertEqual(len(out),20)
    def test_denylist_rejects_task_and_repair(self):
        row={'instruction':'repair','response':'ok','meta':{'kind':'rft','category':'math','task_hash':r.digest('original')}}
        with self.assertRaises(ValueError):r.validate_rows([row],{r.digest('original')})
    def test_denylist_rejects_direct_prompt(self):
        row={'instruction':'held out','response':'ok','meta':{'kind':'rft','category':'math','task_hash':r.digest('other')}}
        with self.assertRaises(ValueError):r.validate_rows([row],{r.digest('held out')})
    def test_masks_only_final_answer(self):
        class Tok:
            def apply_chat_template(self,turns,**kw):
                prefix=[1,2,3]; return prefix if kw['add_generation_prompt'] else prefix+[4,5]
        row={'instruction':'repair prompt with evidence','response':'fixed'}
        out=r.mask(row,Tok(),16)
        self.assertEqual(out['labels'],[-100,-100,-100,4,5])
    def test_overlength_fails_not_truncates(self):
        class Tok:
            def apply_chat_template(self,turns,**kw):return [1] if kw['add_generation_prompt'] else [1,2,3]
        with self.assertRaises(ValueError):r.mask({'instruction':'x','response':'y'},Tok(),2)

    def test_mask_excludes_trailing_template_tokens_and_rejects_control_tokens(self):
        class Tok:
            eos_token_id=9
            all_special_tokens=["<END>"]
            def apply_chat_template(self,turns,**kw):
                return {"input_ids":[1,2] if kw["add_generation_prompt"] else [1,2,3,9,4]}
        self.assertEqual(r.mask({"instruction":"x","response":"y"},Tok(),7)["labels"],[-100,-100,3,9,-100])
        with self.assertRaises(ValueError):
            r.mask({"instruction":"x","response":"y<END>"},Tok(),7)
    def test_bad_hash_rejected(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'x';p.write_text('abc');self.assertRaises(ValueError,r.read_pinned,p,'0'*64)
    def test_repair_requires_failed_independent_check(self):
        task={'id':'1','prompt':'question','kind':'sympy','gold':5,'expression':'2+3'}
        def verify(t,answer):return (answer=='5','independent check')
        key=r.digest(r.repair_prompt(task,'4','independent check'))
        out=r.select(task,[{'draft':'4','repair':'5','repair_input_sha256':key}],verify)
        self.assertEqual(out['response'],'5');self.assertIn('4',out['instruction']);self.assertEqual(out['meta']['repair'],True)
        self.assertIsNone(r.select(task,[{'draft':'4','repair':'6','repair_input_sha256':key}],verify))
    def test_nonfinite_or_missing_train_targets_fail(self):
        with self.assertRaises(ValueError):r.plan(19,1024)
        with self.assertRaises(ValueError):r.plan(20,2048)
        self.assertEqual(r.plan(20,1024)['replay_rows'],6)
if __name__=='__main__':unittest.main()

class SchemaTests(unittest.TestCase):
    def test_logic_terms_are_declarative(self):
        def check(t):
            self.assertIsInstance(t,dict)
            if 'args' in t:
                for x in t['args']:check(x)
            else:self.assertTrue(set(t) in ({'const'},{'var'}))
        for t in r.catalog('5'*64,40):
            if t['kind']=='z3':
                for term in t['spec']['premises']+[t['spec']['claim']]:check(term)
    def test_catalog_code_hidden_cases_are_independent(self):
        for t in r.catalog('6'*64,40):
            if t['kind']=='code':
                self.assertGreater(len(t['hidden']),4)
                self.assertTrue(t['spec']['properties'])
    def test_protected_output_denied(self):
        with self.assertRaises(ValueError):r.output_path('Reference50/new')
    def test_insufficient_replay_fails(self):
        with self.assertRaises(ValueError):r.mix([],[],20,set(),1)

class FeedbackTests(unittest.TestCase):
    def test_feedback_never_uses_candidate_claimed_proof(self):
        t={'id':'1','prompt':'question','kind':'sympy','gold':5,'expression':'2+3'}
        verify=lambda t,a:(a=='5','counterexample')
        f=r.feedback([t],[{'id':'1','draft':'4','verified':True}],verify)[0]
        self.assertIsNone(r.select(t,[{'draft':'4','repair':'5','repair_input_sha256':'0'*64}],verify))
        self.assertIsNotNone(r.select(t,[{'draft':'4','repair':'5','repair_input_sha256':f['repair_input_sha256']}],verify))
    def test_unsupported_check_never_creates_repair(self):
        t={'id':'1','prompt':'question'}
        self.assertEqual(r.feedback([t],[{'id':'1','draft':'4'}],lambda *a:(False,'')),[])
