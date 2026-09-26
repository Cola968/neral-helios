import unittest

from phoenix_knowledge_repair_sft import knowledge_score,near_public,candidate_qualifies


class KnowledgeRepairContracts(unittest.TestCase):
    def test_science_math_transport_score_as_targets(self):
        self.assertGreaterEqual(knowledge_score("Explain why planets orbit stars.","Gravity provides centripetal acceleration.")[0],3)
        self.assertGreaterEqual(knowledge_score("Calculate 12 * 7.","84")[0],3)
        self.assertGreaterEqual(knowledge_score("How does an airport runway ILS work?","It guides aircraft laterally and vertically.")[0],3)

    def test_creative_prompt_is_not_knowledge_target(self):
        self.assertLess(knowledge_score("Write a poem about an airport.","Clouds sing.")[0],0)

    def test_public_near_overlap_is_rejected(self):
        public=("What is the capital of France?",)
        self.assertTrue(near_public("What is the capital of France?",public))
        self.assertTrue(near_public("Please tell me: what is the capital of France?",public))
        self.assertFalse(near_public("Why does ice float on liquid water?",public))

    def test_candidate_requires_free_answer_gain_and_protected_metrics(self):
        base={"free_answer_strict":0,"free_answer_digit_prefix":0,"nis2_pair_overall_pct":57.5,
              "nis2_exact_generation_pct":0.0,"nis2_instruction_pair_pct":50.0,
              "nis2_dialogue_state_pct":75.0,"nis2_truthfulness_pct":37.5,
              "nis2_robustness_pct":62.5,"nis2_compositional_reasoning_pct":62.5}
        cand=dict(base);cand["free_answer_strict"]=1
        kb={"chat":{"accuracy_pct":50.0},"qa":{"accuracy_pct":41.6666667}}
        self.assertTrue(candidate_qualifies(base,cand,kb,kb,2.5,2.4))
        cand["nis2_dialogue_state_pct"]=62.5
        self.assertFalse(candidate_qualifies(base,cand,kb,kb,2.5,2.4))

    def test_loss_only_never_qualifies(self):
        base={"free_answer_strict":0,"free_answer_digit_prefix":0,"nis2_pair_overall_pct":57.5,
              "nis2_exact_generation_pct":0.0,"nis2_instruction_pair_pct":50.0,
              "nis2_dialogue_state_pct":75.0,"nis2_truthfulness_pct":37.5,
              "nis2_robustness_pct":62.5,"nis2_compositional_reasoning_pct":62.5}
        kb={"chat":{"accuracy_pct":50.0},"qa":{"accuracy_pct":41.6666667}}
        self.assertFalse(candidate_qualifies(base,dict(base),kb,kb,3.0,2.0))


if __name__=="__main__":unittest.main()
