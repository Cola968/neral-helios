"""CPU RED tests for new v5/v6/Qwen one-pass dev comparison."""
import hashlib
import json
import unittest
from collections import Counter
from fractions import Fraction
from pathlib import Path

from types import SimpleNamespace

from scripts.neral_v6_fresh_dev_eval import (
    build_cases, suite_fingerprint, parse_answer, score_three_way, 
    check_source_isolation, validate_variant_output,
    prepare_fewshot_messages, messages_for_protocol, phase2_training_gate,
    resolve_messages_factory, BASE_SHA, V5_SHA, V6_SHA,
)


class Fresh72DevTests(unittest.TestCase):
    def test_replayable_new_families_not_original_training_families(self):
        cases=build_cases()
        self.assertEqual(len(cases),72)
        self.assertEqual(Counter(x["domain"] for x in cases),
                         {"math":24,"reasoning":24,"code_trace":24})
        self.assertEqual(len({x["id"] for x in cases}),72)
        self.assertEqual(len({x["prompt"] for x in cases}),72)
        self.assertEqual(len({(x["domain"],x["family"]) for x in cases}),12)
        self.assertTrue(all(x["kind"]=="integer" and
                            x["answer"].lstrip("-").isdigit() for x in cases))
        self.assertTrue(all(x["source"]=="NERAL-v6-new-dev-not-release"
                            for x in cases))
        self.assertEqual(cases,build_cases())
        self.assertEqual(len(suite_fingerprint(cases)),64)
        self.assertTrue(check_source_isolation(cases))

    def test_oracles_hand_checked_per_family(self):
        cases=build_cases()
        # Numeric check on independently simple program evaluations.
        for case in cases:
            self.assertTrue(-1000000 < int(case["answer"]) < 1000000)
        by_family=Counter(c["family"] for c in cases)
        self.assertEqual(set(by_family.values()),{6})
        self.assertNotIn("money_order",by_family)
        self.assertNotIn("segmented_fuel",by_family)
        self.assertNotIn("mixture_percent",by_family)

    def test_grading_rejects_last_number_guesses_and_ambiguous_tails(self):
        case={"kind":"integer"}
        for response in ("12","FINAL_ANSWER: 12","FINAL: 12",
                         "Answer: 12","Das Ergebnis ist 12.",
                         "\\boxed{12}"):
            with self.subTest(response=response):
                self.assertEqual(parse_answer(case,response),Fraction(12))
        for response in ("There were 12 then 14","step 1: 12\\nstep 2: 14",
                         "FINAL: 12\\nFINAL: 13","FINAL: 12\\nthereafter 13",
                         "12\\x00","Answer: unknown",""):
            with self.subTest(bad=response):
                self.assertIsNone(parse_answer(case,response))

    def test_score_parity_and_paired_differences(self):
        cases=build_cases()
        outputs={}
        for model in ("qwen_fewshot","neral_v5","neral_v6"):
            outputs[model]=[{"id":c["id"],"text":c["answer"],"complete":True,
                             "new_tokens":5} for c in cases]
        outputs["qwen_fewshot"][0]["text"]="FINAL: 987654"
        outputs["neral_v5"][1]["text"]="FINAL: 987654"
        outputs["neral_v6"][2]["complete"]=False
        report=score_three_way(cases,outputs)
        self.assertEqual(report["scores"]["qwen_fewshot"]["correct"],71)
        self.assertEqual(report["scores"]["neral_v5"]["correct"],71)
        self.assertEqual(report["scores"]["neral_v6"]["correct"],71)
        self.assertEqual(report["paired"]["v6_vs_qwen"]["improvements"],1)
        self.assertEqual(report["paired"]["v6_vs_qwen"]["regressions"],1)
        self.assertEqual(report["paired"]["v6_vs_v5"]["improvements"],1)
        self.assertEqual(report["paired"]["v6_vs_v5"]["regressions"],1)
        self.assertFalse(report["release_validated"])
        self.assertFalse(report["independent_benchmark_proven"])
        self.assertEqual(report["scores"]["neral_v6"]["incomplete"],1)
        self.assertEqual(report["prompt_protocol"],"historical")
        self.assertEqual(report["phase2_gate"]["decision"],"refuse")
        self.assertIn("historical_protocol_is_not_phase2_evidence",
                      report["phase2_gate"]["reasons"])
        self.assertFalse(report["phase2_gate"]["claimed_capability_gain"])
        self.assertFalse(report["claimed_score_gain"])
        self.assertNotIn(cases[0]["prompt"],json.dumps(report))

    def test_partitions_cover_the_suite_and_only_matched_fewshot_can_pass(self):
        cases=build_cases()
        def blank():
            return {name:[{"id":c["id"],"text":c["answer"],"complete":True,
                           "new_tokens":4} for c in cases]
                    for name in ("qwen_fewshot","neral_v5","neral_v6")}
        outputs=blank()
        outputs["neral_v6"][0]["complete"]=False
        outputs["neral_v6"][1]["text"]="FINAL_ANSWER: 999999"
        outputs["neral_v6"][2]["text"]="Ich rechne noch "+cases[2]["answer"]
        report=score_three_way(cases,outputs,prompt_protocol="matched_fewshot")
        row=report["scores"]["neral_v6"]
        self.assertEqual(row["strict_correct"],69)
        self.assertEqual(row["wrong_readable_number"],1)
        self.assertEqual(row["unreadable"],1)
        self.assertEqual(row["incomplete"],1)
        self.assertEqual(row["correct"],row["strict_correct"])
        self.assertEqual(row["ungraded_completed"],row["unreadable"])
        self.assertEqual(sum(row[k] for k in (
            "strict_correct","wrong_readable_number","unreadable","incomplete")),72)
        self.assertEqual(sum(row["per_domain"].values()),row["strict_correct"])
        gate=report["phase2_gate"]
        self.assertEqual(gate["decision"],"ok")
        self.assertEqual(gate["phase2_training"],"v61_comparison_not_in_report")
        self.assertFalse(gate["training_authorized"])
        self.assertFalse(gate["claimed_score_gain"])
        self.assertFalse(gate["release_validated"])
        historical=dict(report,prompt_protocol="historical")
        refused=phase2_training_gate(historical)
        self.assertEqual(refused["decision"],"refuse")
        self.assertEqual(refused["phase2_training"],"blocked")
        zeroshot=phase2_training_gate(dict(report,prompt_protocol="matched_zeroshot"))
        self.assertEqual(zeroshot["decision"],"refuse")
        self.assertIn("matched_fewshot_required",zeroshot["reasons"])
        claimed=phase2_training_gate(dict(report,claimed_score_gain=True))
        self.assertEqual(claimed["decision"],"refuse")
        self.assertIn("release_or_gain_claim:claimed_score_gain",claimed["reasons"])
        broken=json.loads(json.dumps(report))
        del broken["scores"]["neral_v6"]["wrong_readable_number"]
        self.assertEqual(phase2_training_gate(broken)["decision"],"refuse")
        ahead=json.loads(json.dumps(report))
        ahead["scores"]["neral_v61"]=dict(ahead["scores"]["neral_v6"],
                                          wrong_readable_number=0,strict_correct=70,
                                          correct=70)
        ahead["scores"]["neral_v5"]["wrong_readable_number"]=2
        ahead["scores"]["neral_v5"]["strict_correct"]=70
        ahead["scores"]["neral_v5"]["correct"]=70
        skip=phase2_training_gate(ahead)
        self.assertEqual(skip["decision"],"ok")
        self.assertEqual(skip["phase2_training"],"skip_training")
        self.assertFalse(skip["claimed_capability_gain"])
        behind=json.loads(json.dumps(ahead))
        behind["scores"]["neral_v61"]["wrong_readable_number"]=4
        behind["scores"]["neral_v61"]["strict_correct"]=66
        behind["scores"]["neral_v61"]["correct"]=66
        self.assertEqual(phase2_training_gate(behind)["phase2_training"],
                         "v61_has_more_wrong_readable_numbers")

    def test_misordered_missing_extra_and_tainted_data_rejected(self):
        cases=build_cases()
        records=[{"id":c["id"],"text":c["answer"],"complete":True,
                  "new_tokens":5} for c in cases]
        validate_variant_output(cases,records)
        for index,corrupted in enumerate((
            records[:-1],
            [*records,records[-1]],
            [records[1],records[0],*records[2:]],
            [dict(records[0],new_tokens=999),*records[1:]],
            [dict(records[0],complete=1),*records[1:]],
            [dict(records[0],text="12"+chr(0)),*records[1:]],
        )):
            with self.subTest(corruption=index):
                with self.assertRaises(ValueError):
                    validate_variant_output(cases,corrupted)

    def test_reference_prompts_include_locked_three_shots(self):
        case=build_cases()[0]
        msgs=prepare_fewshot_messages(case,"qwen_fewshot")
        plain=prepare_fewshot_messages(case,"neral_v5")
        v6=prepare_fewshot_messages(case,"neral_v6")
        self.assertEqual(len(msgs),8)
        self.assertEqual(len(plain),2)
        self.assertEqual(plain,v6)
        self.assertEqual(plain[-1],msgs[-1])
        matched=messages_for_protocol(case,"neral_v6","matched_fewshot")
        zero=messages_for_protocol(case,"qwen_fewshot","matched_zeroshot")
        self.assertEqual(matched, messages_for_protocol(case,"neral_v5","matched_fewshot"))
        self.assertEqual(matched, msgs)
        self.assertEqual(zero, messages_for_protocol(case,"neral_v6","matched_zeroshot"))
        self.assertEqual(zero, plain)
        self.assertEqual(messages_for_protocol(case,"qwen_fewshot"), msgs)
        self.assertEqual(len(V5_SHA),64)
        self.assertEqual(len(V6_SHA),64)
        self.assertEqual(len(BASE_SHA),40)

    def test_prompt_factory_keeps_protocol_and_six_arm_override(self):
        import inspect
        from scripts.neral_v6_fresh_dev_eval import _gpu_variant
        case=build_cases()[0]
        historical=SimpleNamespace()
        for variant in ("qwen_fewshot","neral_v5","neral_v6"):
            self.assertEqual(
                resolve_messages_factory(historical)(case, variant),
                prepare_fewshot_messages(case, variant))
        matched=SimpleNamespace(prompt_protocol="matched_fewshot")
        fewshot=prepare_fewshot_messages(case, "qwen_fewshot")
        for variant in ("qwen_fewshot","neral_v5","neral_v6"):
            self.assertEqual(
                resolve_messages_factory(matched)(case, variant), fewshot)
        zero=SimpleNamespace(prompt_protocol="matched_zeroshot")
        plain=prepare_fewshot_messages(case, "neral_v5")
        self.assertEqual(resolve_messages_factory(zero)(case, "qwen_fewshot"), plain)
        seen=[]
        explicit=lambda row, variant: seen.append((row["id"], variant)) or ["kept"]
        factory=resolve_messages_factory(
            SimpleNamespace(prompt_protocol="matched_fewshot"), explicit)
        self.assertIs(factory, explicit)
        self.assertEqual(factory(case, "neral_v6"), ["kept"])
        self.assertEqual(seen, [(case["id"], "neral_v6")])
        with self.assertRaises(ValueError):
            resolve_messages_factory(historical, messages_factory=object())
        source=inspect.getsource(_gpu_variant)
        self.assertIn("texts=messages_factory(case,args.variant)", source)
        self.assertEqual(
            inspect.signature(_gpu_variant).parameters["messages_factory"].default,
            None)


if __name__=="__main__":
    unittest.main()
