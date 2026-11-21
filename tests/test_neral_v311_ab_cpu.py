import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_v311_ab_holdout import (HOLDOUT_SHA256, check_frozen_holdout,
    score_answer, compare_reports, web_evidence_for)
from scripts.neral_v310_freeze_holdout100 import generate
from scripts.neral_qlora_verified_v39 import load_verified_dataset

TRAIN=Path(__file__).resolve().parents[1]/'datasets'/'neral_v39_seed97.jsonl'

class ABEvaluationContracts(unittest.TestCase):
    def test_pinned_digest_is_literal_and_mismatch_fails_closed(self):
        self.assertEqual(HOLDOUT_SHA256,'6b2e7d2412c950deb8fda3f8ad843a7adab2a3e0f428c580b5d4d909234f0e04')
        with tempfile.TemporaryDirectory() as tmp:
            h=Path(tmp)/'private.jsonl'
            train=load_verified_dataset(TRAIN)
            gold=generate('a1'*32,{row['meta']['instruction_sha256'] for row in train})
            h.write_text(''.join(json.dumps(x)+'\n' for x in gold),encoding='utf8')
            with self.assertRaisesRegex(ValueError,'SHA256'):
                check_frozen_holdout(h,TRAIN)

    def test_numeric_and_knowledge_scoring(self):
        self.assertTrue(score_answer({'category':'mathematics','kind':'integer','gold':42},'42'))
        self.assertFalse(score_answer({'category':'mathematics','kind':'integer','gold':42},'43'))
        self.assertTrue(score_answer({'category':'knowledge','kind':'word','gold':'Sauerstoff'},'Sauerstoff'))
        self.assertFalse(score_answer({'category':'knowledge','kind':'word','gold':'Sauerstoff'},'Kohlenstoff'))

    def test_web_sources_scored_with_browser_platform_scope_and_noncanonical_kernel_names(self):
        case={'category':'web','kind':'web','entities':['Brave','Vivaldi'],
              'gold':['Blink','Blink']}
        ctx=web_evidence_for(case)
        self.assertIn('Desktop',ctx)
        self.assertIn('https://',ctx)
        self.assertTrue(score_answer(case,'Brave und Vivaldi verwenden auf Desktop beide Blink [1][2]. Auf iOS WebKit.'))
        self.assertFalse(score_answer(case,'Brave nutzt Blink, Vivaldi nutzt Gecko [1][2].'))
        bsd={'category':'web','kind':'web','entities':['OpenBSD','DragonFly BSD'],
             'gold':['OpenBSD-Kernel','DragonFly-BSD-Kernel']}
        self.assertTrue(score_answer(bsd,'OpenBSD besitzt seinen eigenen BSD-Kernel [1], und DragonFly BSD hat einen eigenständig entwickelten BSD-Kernel [2].'))

    def test_promotion_gate_requires_total_gain_and_no_category_regression(self):
        categories=['mathematics','logic','programming','knowledge','web']
        base={c:12 for c in categories}
        cand={c:13 for c in categories}
        good=compare_reports(base,cand)
        self.assertTrue(good['a_b_gate_pass'])
        self.assertFalse(good['live_adapter_promoted'])
        with self.assertRaises(ValueError):compare_reports(base,{**cand,'logic':11})
        with self.assertRaises(ValueError):compare_reports(base,base)
        with self.assertRaises(ValueError):compare_reports(base,{**cand,'web':21})

if __name__=='__main__':unittest.main()
