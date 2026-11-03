"""TDD security and capability-quality tests: no torch, model, or external requests."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_candidate_quality_gate import (
    GateError, compare_scored_runs, read_json, sha256_file,
)

H = 'a' * 64
B = 'b' * 64
C = 'c' * 64
MODEL = 'Qwen/Qwen2.5-Coder-7B-Instruct'


def build_data(*, n=100, baseline_full=20, candidate_full=45, regress=0, split='final', independent=True):
    tasks = []
    classes = ['coding', 'math', 'reasoning', 'instruction', 'reliability']
    for i in range(n):
        tasks.append({'id': f'case_{i:03}', 'category': classes[i % 5],
                      'prompt_sha256': hashlib.sha256(f'p{i}'.encode()).hexdigest(),
                      'rubric_sha256': hashlib.sha256(f'r{i}'.encode()).hexdigest()})
    manifest = {'schema': 1, 'benchmark_version': 'sealed-1', 'split': split,
                'independent_review_assertion': independent, 'tasks': tasks}
    sha = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    def run(role, full_count):
        rows = []
        for i,t in enumerate(tasks):
            full = i < full_count if role == 'baseline' else i < candidate_full and i >= regress
            # Candidate's first regress cases were previously solved.
            rows.append({'id': t['id'], 'prompt_sha256': t['prompt_sha256'],
                         'rubric_sha256': t['rubric_sha256'],
                         'answer_sha256': hashlib.sha256(f'{role}-{i}'.encode()).hexdigest(),
                         'status': 'full' if full else 'fail', 'complete': full,
                         'unsupported_claims': 0, 'reviewer': 'independent-reviewer'})
        return {'schema': 1, 'role': role, 'base_model_id': MODEL,
                'base_revision': 'd'*40, 'base_snapshot_sha256': 'e'*64,
                'tokenizer_sha256': H, 'prompt_template_sha256': B,
                'decoding_sha256': C, 'tools_sha256': H,
                'benchmark_sha256': sha,
                'adapter_sha256': None if role == 'baseline' else 'f'*64,
                'cases': rows}
    return manifest, run('baseline', baseline_full), run('candidate', candidate_full)


class CandidateQualityGateTests(unittest.TestCase):
    def test_disallows_nonfinal_and_nonindependent_benchmark(self):
        for kwargs in [{'split':'dev'}, {'independent':False}]:
            m,b,c=build_data(**kwargs)
            with self.assertRaises(GateError): compare_scored_runs(m,b,c,'f'*64)

    def test_positive_paired_gain_only_recommends_manual_review(self):
        m,b,c=build_data()
        v=compare_scored_runs(m,b,c,'f'*64)
        self.assertEqual(v['decision'],'MANUAL_REVIEW_REQUIRED')
        self.assertEqual(v['baseline_full'],20)
        self.assertEqual(v['candidate_full'],45)
        self.assertEqual(v['newly_solved'],25)
        self.assertEqual(v['regressions'],0)
        self.assertFalse(v['weights_modified'])
        self.assertFalse(v['deployment_authorized'])
        self.assertLessEqual(v['one_sided_exact_p'],0.05)

    def test_equal_or_worse_rejected(self):
        for count in [0,10,20]:
            m,b,c=build_data(candidate_full=count)
            self.assertEqual(compare_scored_runs(m,b,c,'f'*64)['decision'],'REJECT')

    def test_small_gain_is_not_significant(self):
        m,b,c=build_data(candidate_full=23)
        self.assertEqual(compare_scored_runs(m,b,c,'f'*64)['decision'],'REJECT')

    def test_hidden_regression_in_category_is_rejected(self):
        m,b,c=build_data(candidate_full=60,regress=2)
        # Remove gains in the first two categories but keep the overall gain large.
        v=compare_scored_runs(m,b,c,'f'*64)
        self.assertEqual(v['decision'],'REJECT')
        self.assertEqual(v['regressions'],2)

    def test_abstentions_do_not_pass_for_full(self):
        m,b,c=build_data()
        c['cases'][24]['complete']=False
        with self.assertRaises(GateError):compare_scored_runs(m,b,c,'f'*64)

    def test_missing_or_duplicate_cases_refused(self):
        m,b,c=build_data()
        c['cases'].pop()
        with self.assertRaises(GateError):compare_scored_runs(m,b,c,'f'*64)
        m,b,c=build_data();c['cases'][-1]=c['cases'][0]
        with self.assertRaises(GateError):compare_scored_runs(m,b,c,'f'*64)

    def test_changed_base_or_decoding_refused(self):
        for field,value in [('base_revision','0'*40),('decoding_sha256','0'*64),
                            ('tools_sha256','0'*64),('tokenizer_sha256','0'*64)]:
            m,b,c=build_data();c[field]=value
            with self.assertRaises(GateError):compare_scored_runs(m,b,c,'f'*64)

    def test_adapter_hash_mismatch_and_baseline_adapter_refused(self):
        m,b,c=build_data()
        with self.assertRaises(GateError):compare_scored_runs(m,b,c,'0'*64)
        b['adapter_sha256']='f'*64
        with self.assertRaises(GateError):compare_scored_runs(m,b,c,'f'*64)

    def test_unreviewed_or_unbound_answers_refused(self):
        for key,value in [('reviewer',''), ('rubric_sha256','a'*64), ('answer_sha256','bad'),
                          ('prompt_sha256','b'*64)]:
            m,b,c=build_data();c['cases'][0][key]=value
            with self.assertRaises(GateError):compare_scored_runs(m,b,c,'f'*64)

    def test_claimed_unknown_status_refused(self):
        m,b,c=build_data();c['cases'][0]['status']='perfect'
        with self.assertRaises(GateError):compare_scored_runs(m,b,c,'f'*64)

    def test_extra_unsupported_claims_rejects_even_quality_gain(self):
        m,b,c=build_data();c['cases'][60]['unsupported_claims']=1
        self.assertEqual(compare_scored_runs(m,b,c,'f'*64)['decision'],'REJECT')

    def test_exact_input_reader_rejects_duplicate_keys_and_nan(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.json'
            for payload in ['{"a":1,"a":2}','{"a":NaN}']:
                p.write_text(payload)
                with self.assertRaises(GateError):read_json(p)

    def test_adapter_hash_reads_real_bytes_and_rejects_links(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'candidate.safetensors';p.write_bytes(b'original-sealed-adapter')
            self.assertEqual(sha256_file(p),hashlib.sha256(p.read_bytes()).hexdigest())
            link=Path(d)/'copy.safetensors';link.symlink_to(p)
            with self.assertRaises(GateError):sha256_file(link)

    def test_invalid_or_too_small_inventory_refused(self):
        m,b,c=build_data(n=25)
        with self.assertRaises(GateError):compare_scored_runs(m,b,c,'f'*64)

    def test_never_promotes_with_fake_favorable_review_source(self):
        m,b,c=build_data();c['cases'][4]['reviewer']='model-itself'
        with self.assertRaises(GateError):compare_scored_runs(m,b,c,'f'*64)

if __name__ == '__main__': unittest.main()
