"""Red/green contracts for source-pinned offline Qwen 2048 follow-up."""
from __future__ import annotations
import json
import unittest

from scripts.neral_foundation_2048_retest import (
    GROUP_BOTH_CAPPED, GROUP_ONLY_Q35_CAPPED, GROUP_BOTH_FINISHED,
    EXPECTED_GROUP_COUNTS, classify_tail, compute_original_groups,
    classify_review_ends, chat_payload, parse_gpu_memory,
    summarise_retest, RetestError
)

def old_records():
    rows=[]
    for domain in ("math","reasoning","code_trace"):
        for fam in range(4):
            for idx in range(6):
                cid=f"{domain}-family{fam}-{idx}"
                for model in ("qwen3:8b","qwen3.5:9b"):
                    j=len(rows)//2
                    capped=(j<31 if model=="qwen3:8b" else j<62)
                    answer=100+j
                    raw=("Compute 4+5=9. Next, calculate 2*3=6."
                         if capped else f"FINAL_ANSWER: {answer}")
                    rows.append({
                        "id":cid, "domain":domain,"family":f"family{fam}",
                        "model":model,"prompt_private":f"secret prompt {cid}",
                        "answer_private":answer,"raw_private":raw,
                        "grade":({"status":"incomplete","value":None,"format_valid":False}
                                 if capped else {"status":"correct","value":answer,"format_valid":True}),
                        "done_reason":"length" if capped else "stop"
                    })
    return rows

def review_records(original):
    return [{
        "case_id":r["id"], "model":r["model"],
        "prompt_private":r["prompt_private"],"gold_private":r["answer_private"],
        "last_800_chars_private":r["raw_private"][-800:],
        "requires_manual_reasoning_review":True
    } for r in original if r["done_reason"]=="length"]

class Qwen2048Contracts(unittest.TestCase):
    def test_original_three_groups_are_pinned_and_mutually_exclusive(self):
        groups=compute_original_groups(old_records())
        self.assertEqual({name:len(ids) for name,ids in groups.items()},
                         EXPECTED_GROUP_COUNTS)
        self.assertEqual(len(set().union(*map(set,groups.values()))),72)
        self.assertEqual(len(groups[GROUP_BOTH_CAPPED]),31)
        self.assertEqual(len(groups[GROUP_ONLY_Q35_CAPPED]),31)
        self.assertEqual(len(groups[GROUP_BOTH_FINISHED]),10)

    def test_private_tail_patterns_are_heuristics_not_proof(self):
        repeated=("We need to calculate. The next calculation is " * 6)
        self.assertEqual(classify_tail(repeated),"repetition_like")
        structured="1) compute 40+2=42\n2) check 7*6=42\nThus the answer is"
        self.assertEqual(classify_tail(structured),"structured_progress_candidate")
        self.assertEqual(classify_tail("Perhaps, but also, maybe, next..."),
                         "unstructured_or_digression")

    def test_original_private_review_matches_original_records_and_stays_private(self):
        old=old_records()
        report,private=classify_review_ends(review_records(old),old)
        self.assertEqual(report["total_incomplete"],93)
        self.assertEqual(report["models"]["qwen3:8b"]["total"],31)
        self.assertEqual(report["models"]["qwen3.5:9b"]["total"],62)
        self.assertFalse(report["reasoning_correctness_proven"])
        self.assertNotIn("secret prompt",json.dumps(report))
        self.assertIn("secret prompt",json.dumps(private))

    def test_refuse_private_review_without_matching_source(self):
        old=old_records()
        review=review_records(old)
        review[0]["last_800_chars_private"]="modified"
        with self.assertRaises(RetestError):
            classify_review_ends(review,old)
        with self.assertRaises(RetestError):
            classify_review_ends(review_records(old)[:-1],old)

    def test_modelcard_non_thinking_options_and_prompt_parity(self):
        prompt="Solve 1+2."
        a=chat_payload("qwen3:8b",prompt)
        b=chat_payload("qwen3.5:9b",prompt)
        self.assertEqual(a["messages"],b["messages"])
        self.assertEqual(a["think"],False)
        self.assertEqual(b["think"],False)
        for payload in (a,b):
            opt=payload["options"]
            self.assertEqual(opt["temperature"],0.7)
            self.assertEqual(opt["top_p"],0.8)
            self.assertEqual(opt["top_k"],20)
            self.assertEqual(opt["min_p"],0)
            self.assertEqual(opt["presence_penalty"],1.5)
            self.assertEqual(opt["num_predict"],2048)
            self.assertEqual(opt["num_ctx"],4096)
            self.assertEqual(opt["seed"],20261108)
        self.assertNotEqual(a["model"],b["model"])

    def test_gpu_memory_parse_and_reject_wrong_gpu_or_missing_samples(self):
        self.assertEqual(parse_gpu_memory(" 5200, 12282, NVIDIA GeForce RTX 4070"),
                         (5200,12282))
        for s in ("0, 8192, AMD","garbage",
                  "99999, 12282, NVIDIA RTX 4070",
                  "5000, 12282, NVIDIA RTX 4070\n123,456,other"):
            with self.subTest(s=s),self.assertRaises(RetestError):
                parse_gpu_memory(s)

    def test_scores_separately_by_original_group_and_do_not_leak_prompts(self):
        groups=compute_original_groups(old_records())
        answer={name:{} for name in ("qwen3:8b","qwen3.5:9b")}
        for model in answer:
            for i,cid in enumerate(sorted(set().union(*map(set,groups.values())))):
                answer[model][cid]={
                    "status":"correct" if i%2==0 else "incomplete",
                    "done_reason":"stop" if i%2==0 else "length",
                    "eval_count":120 if i%2==0 else 2048,
                    "duration_ns":1000000,
                }
        gpu={model:{"peak_used_mib":8200,"total_mib":12282,
                    "sample_count":100} for model in answer}
        result=summarise_retest(groups,answer,gpu)
        self.assertEqual(result["n_per_model"],72)
        self.assertEqual(result["models"]["qwen3:8b"]["by_original_group"]
                         [GROUP_BOTH_CAPPED]["total"],31)
        self.assertEqual(result["models"]["qwen3.5:9b"]["by_original_group"]
                         [GROUP_ONLY_Q35_CAPPED]["total"],31)
        self.assertEqual(result["models"]["qwen3:8b"]["sampled_peak_vram_gib"],
                         round(8200/1024,3))
        self.assertFalse(result["release_validated"])
        self.assertFalse(result["training_performed"])
        self.assertNotIn("secret prompt",json.dumps(result))

    def test_reject_missing_or_malformed_gpu_peak_and_scores(self):
        groups=compute_original_groups(old_records())
        responses={name:{cid:{"status":"correct","done_reason":"stop",
                               "eval_count":70,"duration_ns":1000000}
                         for cid in set().union(*map(set,groups.values()))}
                   for name in ("qwen3:8b","qwen3.5:9b")}
        vram={name:{"peak_used_mib":9000,"total_mib":12282,"sample_count":10}
              for name in responses}
        with self.assertRaises(RetestError):
            summarise_retest(groups,responses,{})
        del responses["qwen3:8b"][next(iter(responses["qwen3:8b"]))]
        with self.assertRaises(RetestError):
            summarise_retest(groups,responses,vram)

if __name__=="__main__":
    unittest.main()
