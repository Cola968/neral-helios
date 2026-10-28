"""Test-first contracts: freeze 99% of native weights, protect skills, balance domains."""
import unittest
from datetime import datetime
from frontier350.arctus350_frozen_ffn_replay import (
    choose_trainable_names, normalized_task_scales, round_robin_replay,
    safe_candidate_gate, admission,
)

class FrozenFFNReplayContracts(unittest.TestCase):
    def test_exactly_three_last_feedforward_matrices_trainable(self):
        params=["tok_emb.weight","blocks.0.ffn.up.weight","blocks.23.attn.o.weight",
                "blocks.23.ffn.gate.weight","blocks.23.ffn.up.weight","blocks.23.ffn.down.weight",
                "blocks.23.n2.weight","lm_head.weight"]
        self.assertEqual(choose_trainable_names(params),
             ["blocks.23.ffn.gate.weight","blocks.23.ffn.up.weight","blocks.23.ffn.down.weight"])
        with self.assertRaises(ValueError):
            choose_trainable_names([p for p in params if p!='blocks.23.ffn.down.weight'])
        with self.assertRaises(ValueError):
            choose_trainable_names(params+["blocks.23.ffn.gate.weight"])

    def test_gradient_scales_clamp_math_dominance(self):
        scales=normalized_task_scales({"math":1.622599,"format":.700385,"human_chat":.325768})
        self.assertEqual(set(scales),{"math","format","human_chat"})
        strengths={k:scales[k]*v for k,v in {"math":1.622599,"format":.700385,"human_chat":.325768}.items()}
        self.assertLessEqual(max(strengths.values())/min(strengths.values()),1.02)
        self.assertLess(scales["math"],scales["format"])
        self.assertLessEqual(max(scales.values()),1.0)
        with self.assertRaises(ValueError):
            normalized_task_scales({"math":0,"format":1,"human_chat":1})
        with self.assertRaises(ValueError):
            normalized_task_scales({"math":1,"format":1})

    def test_replay_no_duplicate_ids_and_interleaves_all_domains(self):
        cohorts={"math":[("m1",1),("m2",2)],
                 "format":[("f1",1),("f2",2)],
                 "human_chat":[("h1",1),("h2",2)]}
        s=round_robin_replay(cohorts)
        self.assertEqual([x[0] for x in s],["math","format","human_chat"]*2)
        self.assertEqual(len(s),6)
        with self.assertRaises(ValueError):
            round_robin_replay({**cohorts,"format":[("m1",12)]})

    def test_guard_rejects_any_format_eos_chat_repetition_or_dev_loss(self):
        good=dict(
            new_math_before=0,new_math_after=2,
            prior_word_before=32,prior_word_after=32,
            prior_math_before=5,prior_math_after=5,
            prior_eos_before=76,prior_eos_after=76,
            prior_rep_before=0.,prior_rep_after=0.,
            human_nll_before=3.,human_nll_after=3.02,
            dev_exact_before=50.,dev_exact_after=50.,
            dev_eos_before=1.,dev_eos_after=1.,
            dev_rep_before=.019,dev_rep_after=.018)
        self.assertTrue(safe_candidate_gate(**good))
        for one in ({"new_math_after":1},{"prior_word_after":31},
                    {"prior_math_after":4},{"prior_eos_after":75},
                    {"prior_rep_after":.001},{"human_nll_after":3.11},
                    {"dev_exact_after":0.},{"dev_eos_after":.917},
                    {"dev_rep_after":.03}):
            self.assertFalse(safe_candidate_gate(**{**good,**one}))

    def test_hard_cutoff_before_thursday_and_gpu_pressure(self):
        self.assertTrue(admission(True,9500,9400,58,"2026-10-28T19:00:00+01:00",900))
        self.assertFalse(admission(True,7500,9400,58,"2026-10-28T19:00:00+01:00",900))
        self.assertFalse(admission(True,9500,9400,80,"2026-10-28T19:00:00+01:00",900))
        self.assertFalse(admission(True,9500,9400,58,"2026-10-28T20:35:00+01:00",900))
        self.assertFalse(admission(True,9500,9400,58,"2026-10-29T00:00:00+01:00",900))
        self.assertFalse(admission(False,9500,9400,58,"2026-10-28T19:00:00+01:00",900))
        with self.assertRaises(ValueError):
            admission(True,9500,9400,float("nan"),"2026-10-28T19:00:00+01:00",900)

if __name__=="__main__":
    unittest.main()
