import unittest
from collections import Counter
from scripts.neral_v44_rft_quality import catalog_v2, verify_v2, audit_quality, prompt_shape, choose_quality_pilot, audit_pinned_quality
from scripts.neral_v43_rft_cpu import select, digest, repair_prompt

class QualityTests(unittest.TestCase):
    def test_catalog_has_nine_families_and_is_reproducible(self):
        tasks = catalog_v2('7'*64, 600)
        self.assertEqual(tasks, catalog_v2('7'*64, 600))
        self.assertEqual(len({digest(t['prompt']) for t in tasks}), 600)
        self.assertEqual(Counter(t['kind'] for t in tasks), {'sympy':200,'z3':200,'code':200})
        for kind in ('sympy','z3','code'):
            self.assertEqual(len({t['family'] for t in tasks if t['kind']==kind}),3)

    def test_math_system_and_extremum_checked_independently(self):
        tasks = [t for t in catalog_v2('8'*64, 90) if t['kind']=='sympy']
        for t in tasks:
            self.assertEqual(verify_v2(t,str(t['gold'])),(True,''))
            ok,evidence = verify_v2(t,str(t['gold']+1))
            self.assertFalse(ok)
            self.assertTrue(evidence)
            self.assertEqual(verify_v2(t,'unparseable'),(False,''))
            wrong=dict(t,gold=t['gold']+1)
            self.assertEqual(verify_v2(wrong,str(wrong['gold'])),(False,''))

    def test_logic_and_code_use_existing_verifiers(self):
        tasks=catalog_v2('9'*64,9)
        for task in tasks:
            if task['kind']=='z3':
                self.assertTrue(verify_v2(task,task['gold'])[0])
            if task['kind']=='code':
                p=task['params']
                if task['family']=='code.tariff':
                    a,b,c,d=p
                    answer=f'def solve(x):\n    return {d}+min(max(0,x),{a})*{b}+max(0,x-{a})*{c}'
                elif task['family']=='code.triangular':
                    n=p[0]
                    answer=f'def solve(x):\n    y=min(max(0,x),{n})\n    return y*(y+1)//2'
                else:
                    answer=f'def solve(x):\n    return max(0,x)//{p[0]}'
                self.assertTrue(verify_v2(task,answer)[0],task['family'])

    def test_parameter_change_is_not_independence(self):
        self.assertEqual(prompt_shape('Berechne 21 + 34.'),prompt_shape('berechne 7+12.'))
        self.assertNotEqual(prompt_shape('Berechne 21 + 34.'),prompt_shape('Berechne 21 * 34.'))

    def fixture(self):
        tasks=catalog_v2('a'*64,600)
        # Simulate model output; this fixture does not claim to be GPU training data.
        def verifier(task,text):
            return (True,'') if text=='correct' else (False,'counterexample') if text=='wrong' else (False,'')
        candidates=[]; selected=[]
        for kind,quota in [('sympy',50),('z3',45),('code',45)]:
            pool=[t for t in tasks if t['kind']==kind][:quota]
            for i,t in enumerate(pool):
                c={'id':t['id'],'draft':'wrong' if i<8 else 'correct'}
                if i<8:
                    c.update(repair='correct',repair_input_sha256=digest(repair_prompt(t,'wrong','counterexample')))
                candidates.append(c)
                selected.append(select(t,[c],verify=verifier))
        for cat in ('everyday','knowledge','math','code'):
            for i in range(15):
                prompt=f'replay {cat} token{i} item'+('a'*(i+1))
                selected.append({'instruction':prompt,'response':'fixture','meta':{'kind':'replay','category':cat,'task_hash':digest(prompt),'reviewed':True,'source':'independent_replay','family':f'replay.{cat}'}})
        holdout=[{'prompt':f'holdout {cat} token{i} item'+('b'*(i+1)),'category':cat,'family':f'holdout.{cat}','reviewed':True,'source':'independent_holdout'} for cat in ('everyday','knowledge','math','code') for i in range(15)]
        return tasks,candidates,selected,holdout,verifier

    def test_audit_rechecks_real_repair_chain_and_family_split(self):
        t,c,s,h,v=self.fixture()
        result=audit_quality(s,h,t,c,verify=v)
        self.assertEqual(result['repair_rows'],24)
        self.assertEqual(result['holdout_rows'],60)
        self.assertEqual(len(result['rft_families']),9)
        s[0]['response']='forged'
        with self.assertRaises(ValueError):audit_quality(s,h,t,c,verify=v)

    def test_audit_rejects_family_overlap_even_with_different_prompt(self):
        t,c,s,h,v=self.fixture()
        h[0]['family']=t[0]['family']
        with self.assertRaises(ValueError):audit_quality(s,h,t,c,verify=v)

    def test_audit_rejects_numeric_template_overlap_even_if_label_changed(self):
        t,c,s,h,v=self.fixture()
        h[0]['prompt']=t[0]['prompt'].replace('Berechne','berechne').replace(' ', '  ')
        with self.assertRaises(ValueError):audit_quality(s,h,t,c,verify=v)

    def test_audit_rejects_forged_repair_receipt(self):
        t,c,s,h,v=self.fixture()
        c[0]['repair_input_sha256']='0'*64
        with self.assertRaises(ValueError):audit_quality(s,h,t,c,verify=v)

    def test_audit_rejects_unreviewed_holdout(self):
        t,c,s,h,v=self.fixture()
        h[0]['reviewed']=False
        with self.assertRaises(ValueError):audit_quality(s,h,t,c,verify=v)

    def test_balanced_selection_prefers_repairs_within_each_family(self):
        t,c,s,h,v=self.fixture()
        rft=[x for x in s if x['meta']['kind']=='rft']
        replay=[x for x in s if x['meta']['kind']=='replay']
        selected=choose_quality_pilot(rft,replay,t,seed=17)
        self.assertEqual(len(selected),200)
        self.assertEqual(selected,choose_quality_pilot(rft[::-1],replay[::-1],t,seed=17))
        self.assertEqual(audit_quality(selected,h,t,c,verify=v)['repair_rows'],24)
        with self.assertRaises(ValueError):choose_quality_pilot(rft[:-1],replay,t,seed=17)

    def test_real_sympy_repair_replayed_not_only_claimed(self):
        task=next(t for t in catalog_v2('b'*64,9) if t['family']=='sympy.linear_system')
        draft=str(task['gold']+1);correct=str(task['gold'])
        ok,evidence=verify_v2(task,draft)
        self.assertFalse(ok)
        key=digest(repair_prompt(task,draft,evidence))
        self.assertTrue(select(task,[dict(draft=draft,repair=correct,repair_input_sha256=key)],verify=verify_v2)['meta']['repair'])
        self.assertIsNone(select(task,[dict(draft=draft,repair=correct,repair_input_sha256='0'*64)],verify=verify_v2))
        self.assertFalse(select(task,[dict(draft=correct,repair=correct,repair_input_sha256=key)],verify=verify_v2)['meta']['repair'])

    def test_pinned_quality_recomputes_sources_and_refuses_tampering(self):
        import hashlib,json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        t,c,s,h,v=self.fixture()
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            def seal(name,rows):
                p=root/name
                p.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf8')
                return str(p),hashlib.sha256(p.read_bytes()).hexdigest()
            tp,ts=seal('catalog.jsonl',t);cp,cs=seal('candidates.jsonl',c)
            m=dict(catalog_version='v2',catalog_seed='a'*64,quality_catalog_path=tp,catalog_sha256=ts,quality_candidates_path=cp,candidates_sha256=cs,quality_report=audit_quality(s,h,t,c,verify=v))
            self.assertEqual(audit_pinned_quality(m,s,h,set(),verify=v)['repair_rows'],24)
            Path(cp).write_text('tampered',encoding='utf8')
            with self.assertRaises(ValueError):audit_pinned_quality(m,s,h,set(),verify=v)

if __name__=='__main__':unittest.main()
