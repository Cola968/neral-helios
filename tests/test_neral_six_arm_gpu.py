"""CPU RED: Qwen/v5/v61, 0/3-shot fair comparison with reuse of pinned raw."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.neral_v61_new_dev_eval import cases72
from scripts.neral_six_arm_eval import SIX_ARMS
from scripts.neral_six_arm_gpu import (
    REUSED, MISSING, build_private_run, checked_private_run,
    combine_private_run, model_slot_for_arm,
)


def answers(cases):
    return [{"id":c["id"],"text":c["answer"],
             "complete":True,"new_tokens":5} for c in cases]


class SixArmGPUContracts(unittest.TestCase):
    def test_only_three_missing_arms_need_new_inference(self):
        self.assertEqual(set(REUSED),{"qwen_three","v5_zero","v61_zero"})
        self.assertEqual(set(MISSING),{"qwen_zero","v5_three","v61_three"})
        self.assertEqual(set(REUSED)|set(MISSING),set(SIX_ARMS))
        self.assertEqual(model_slot_for_arm("qwen_zero"),"qwen_fewshot")
        self.assertEqual(model_slot_for_arm("v5_three"),"neral_v5")
        self.assertEqual(model_slot_for_arm("v61_three"),"neral_v6")
        with self.assertRaises(ValueError):
            model_slot_for_arm("qwen_four")

    def test_checked_reuse_requires_exact_prior_cases_and_score(self):
        cases=cases72()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); prior=root/"prior";target=root/"private_six"
            prior.mkdir()
            (prior/"cases.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in cases),encoding="utf8")
            from scripts.neral_v6_fresh_dev_eval import suite_fingerprint
            sha=suite_fingerprint(cases)
            prior_report={
                "status":"NERAL_V61_NEW_72_STRUCTURAL_DEV_NOT_RELEASE",
                "suite_sha256":sha,
                "candidate_adapter_sha256":"33e939f6063d4215807f593816a057e88d8938b30615c6000d6a0bf8c7965279",
                "release_validated":False,
                "scores":{"qwen_fewshot":{"correct":19},
                          "neral_v5":{"correct":16},
                          "neral_v61":{"correct":16}},
            }
            (prior/"summary.json").write_text(json.dumps(prior_report),encoding="utf8")
            for src in REUSED.values():
                (prior/(src+".jsonl")).write_text("".join(json.dumps(row)+"\n" for row in answers(cases)),encoding="utf8")
            # Forged prior score must not be silently accepted.
            with self.assertRaises(ValueError):
                build_private_run(prior,target)
            self.assertFalse(target.exists())
            # Then construct data with exact original results and verify reuse.
            corrected={"qwen_fewshot":19,"neral_v5":16,"neral_v6":16}
            for name,correct in corrected.items():
                rows=answers(cases)
                for k in range(correct,72):
                    rows[k]["text"]="FINAL_ANSWER: 999999"
                (prior/(name+".jsonl")).write_text("".join(json.dumps(x)+"\n" for x in rows),encoding="utf8")
            summary=build_private_run(prior,target)
            self.assertEqual(summary["reused"],sorted(REUSED))
            self.assertEqual(summary["new_gpu_arms"],sorted(MISSING))
            self.assertTrue(checked_private_run(target))
            self.assertEqual(set(x.name for x in target.iterdir()),
                             {"cases.jsonl","manifest.json",*(arm+".jsonl" for arm in REUSED)})
            with self.assertRaises(ValueError):
                build_private_run(prior,target)
            # Add the missing arms (model-only response files remain private).
            for name in MISSING:
                (target/(name+".jsonl")).write_text("".join(json.dumps(x)+"\n" for x in answers(cases)),encoding="utf8")
            result=combine_private_run(target)
            self.assertEqual(result["scores"]["qwen_three"]["correct"],19)
            self.assertEqual(result["scores"]["v5_zero"]["correct"],16)
            self.assertEqual(result["scores"]["v61_zero"]["correct"],16)
            self.assertEqual(result["scores"]["qwen_zero"]["correct"],72)
            self.assertFalse(result["release_validated"])
            self.assertTrue(result["historical_dev72_reused"])
            self.assertTrue(result["cross_shot_weight_comparisons_reported"] is False)
            self.assertNotIn(cases[0]["prompt"],json.dumps(result))
            with self.assertRaises(ValueError):
                combine_private_run(target)

    def test_reject_reordered_cases_and_output_corruption(self):
        cases=cases72()
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t)/"six";folder.mkdir()
            from scripts.neral_v6_fresh_dev_eval import suite_fingerprint
            (folder/"cases.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in cases),encoding="utf8")
            from hashlib import sha256
            hashes={}
            for arm in REUSED:
                response_path=folder/(arm+".jsonl")
                response_path.write_text("".join(json.dumps(row)+"\\n" for row in answers(cases)).replace("\\n","\n"),encoding="utf8")
                hashes[arm]=sha256(response_path.read_bytes()).hexdigest()
            (folder/"manifest.json").write_text(json.dumps({
                "status":"NERAL_6ARM_REUSE_CONSUMED_DEV72",
                "suite_sha256":suite_fingerprint(cases),
                "reused":sorted(REUSED),
                "reused_files_sha256":hashes,
                "candidate_adapter_sha256":"33e939f6063d4215807f593816a057e88d8938b30615c6000d6a0bf8c7965279",
                "new_gpu_arms":sorted(MISSING),
                "release_validated":False,
            }),encoding="utf8")
            self.assertTrue(checked_private_run(folder))
            with (folder/"cases.jsonl").open("w",encoding="utf8") as f:
                for c in (cases[1],cases[0],*cases[2:]):
                    f.write(json.dumps(c,ensure_ascii=False)+"\n")
            with self.assertRaises(ValueError):
                checked_private_run(folder)

    def test_existing_gpu_inference_allows_explicit_prompt_factory(self):
        from scripts import neral_v6_fresh_dev_eval as original
        import inspect
        self.assertIn("messages_factory",inspect.signature(original._gpu_variant).parameters)
        self.assertIn("texts=messages_factory(case,args.variant)",inspect.getsource(original._gpu_variant))
        self.assertEqual(inspect.signature(original._gpu_variant).parameters["messages_factory"].default,None)


if __name__=="__main__":
    unittest.main()
