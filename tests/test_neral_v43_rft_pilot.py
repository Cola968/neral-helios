import hashlib
import unittest
from collections import Counter
from scripts.neral_v43_rft_pilot import choose_pilot, audit_pilot, holdout_hashes


def h(text):
    return hashlib.sha256(' '.join(text.split()).encode()).hexdigest()


def sample(kind, category, i, proof=None, repair=False):
    prompt = f'{kind}:{category}:{proof}:{i}'
    meta = {'kind': kind, 'category': category, 'task_hash': h(prompt)}
    if kind == 'rft':
        meta.update(proof=proof, repair=repair, source='fresh_catalog')
    else:
        meta.update(reviewed=True, source='independent_replay')
    return {'instruction': prompt, 'response': 'synthetic fixture', 'meta': meta}


def pools():
    rft = [sample('rft', 'code' if proof == 'code' else 'math', i, proof, i < 10)
           for proof in ('sympy', 'z3', 'code') for i in range(80)]
    replay = [sample('replay', cat, i) for cat in ('everyday', 'knowledge', 'math', 'code') for i in range(30)]
    return rft, replay


class PilotTests(unittest.TestCase):
    def test_balanced_pilot(self):
        rft, replay = pools()
        items = choose_pilot(rft, replay, 7)
        self.assertEqual(len(items), 200)
        self.assertEqual(Counter(x['meta']['category'] for x in items if x['meta']['kind']=='replay'),
                         dict.fromkeys(('everyday','knowledge','math','code'),15))
        self.assertEqual(Counter(x['meta']['proof'] for x in items if x['meta']['kind']=='rft'),
                         {'sympy':50,'z3':45,'code':45})
        self.assertGreaterEqual(sum(x['meta'].get('repair',False) for x in items), 20)
        self.assertEqual(items, choose_pilot(rft,replay,7))

    def test_input_order_independent(self):
        rft, replay = pools()
        self.assertEqual(choose_pilot(rft,replay,7),choose_pilot(rft[::-1],replay[::-1],7))

    def test_replay_shortfall_fails_instead_of_filling_from_other_categories(self):
        rft,replay = pools()
        replay = [x for x in replay if x['meta']['category']!='knowledge']+replay[30:44]
        with self.assertRaises(ValueError): choose_pilot(rft,replay,7)

    def test_insufficient_repairs_rejected(self):
        rft,replay = pools()
        for x in rft: x['meta']['repair']=False
        with self.assertRaises(ValueError): choose_pilot(rft,replay,7)

    def test_proof_category_mismatch_rejected(self):
        rft,replay = pools(); rft[0]['meta']['category']='code'
        with self.assertRaises(ValueError): choose_pilot(rft,replay,7)

    def test_duplicate_task_refused(self):
        rft,replay = pools()
        with self.assertRaises(ValueError): choose_pilot(rft+[rft[0]],replay,7)

    def test_training_audit_rejects_swapped_replay(self):
        rft,replay=pools(); items=choose_pilot(rft,replay,7)
        i=next(i for i,x in enumerate(items) if x['meta']['kind']=='replay' and x['meta']['category']=='knowledge')
        used={x['meta']['task_hash'] for x in items}
        items[i]=next(x for x in replay if x['meta']['category']=='everyday' and x['meta']['task_hash'] not in used)
        with self.assertRaises(ValueError): audit_pilot(items)

    def test_holdout_coverage_and_overlap(self):
        items=[{'prompt':f'holdout {cat} {i}','category':cat} for cat in ('everyday','knowledge','math','code') for i in range(15)]
        hashes=holdout_hashes(items,set())
        self.assertEqual(len(hashes),60)
        with self.assertRaises(ValueError): holdout_hashes(items,{h(items[0]['prompt'])})
        with self.assertRaises(ValueError): holdout_hashes(items[:-1],set())
        with self.assertRaises(ValueError): holdout_hashes(items+[items[0]],set())

class IntegrationTests(unittest.TestCase):
    def test_builder_uses_pilot_policy(self):
        from scripts import neral_v43_rft_cpu as cpu
        rft,replay=pools()
        self.assertEqual(cpu.mix(rft,replay,200,set(),7), choose_pilot(rft,replay,7))

    def test_bundle_requires_pinned_holdout_and_rechecks_it(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from scripts import neral_v43_rft_train as train
        rft,replay=pools(); selected=choose_pilot(rft,replay,7)
        holdout=[{'prompt':f'holdout {cat} {i}','category':cat} for cat in ('everyday','knowledge','math','code') for i in range(15)]
        with TemporaryDirectory() as tmp, patch.object(train,'protected_hashes',return_value=set()):
            root=Path(tmp); protected=root/'protected.json'; protected.write_text('{}')
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            data=root/'train.jsonl'; data.write_text(''.join(json.dumps(x)+'\n' for x in selected))
            hp=root/'holdout.jsonl'; hp.write_text(''.join(json.dumps(x)+'\n' for x in holdout))
            manifest=root/'manifest.json'
            m={'train_sha256':sha(data),'protected_manifest_sha256':sha(protected)}
            manifest.write_text(json.dumps(m))
            with self.assertRaises(ValueError):train.load_bundle(root,sha(manifest),protected,sha(protected),1024)
            m.update(development_holdout_path=str(hp),development_holdout_sha256=sha(hp))
            manifest.write_text(json.dumps(m))
            self.assertEqual(len(train.load_bundle(root,sha(manifest),protected,sha(protected),1024)[0]),200)
            hp.write_text('changed')
            with self.assertRaises(ValueError):train.load_bundle(root,sha(manifest),protected,sha(protected),1024)

if __name__=='__main__': unittest.main()
