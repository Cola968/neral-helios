"""Offline E1 DPO/ORPO 30-step smoke contracts. No GPU required for CI."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.neral_e1_smoke import validate_pairs, smoke_plan, preflight, require_gpu_opt_in


def pairs():
    result = []
    for split, count in (("train", 50), ("validation", 10)):
        for idx in range(count):
            result.append({
                "id": f"{split}-{idx:03}", "domain": "math" if idx % 2 else "code",
                "family": f"unique-method-{idx%5}", "source_id": f"original-{split}-{idx:03}",
                "prompt": f"An original task with id {split}-{idx} and unique parameters {idx+100}.",
                "chosen": f"FINAL_ANSWER: {idx+1}",
                "rejected": f"FINAL_ANSWER: {idx+2}",
                "chosen_verified": True, "rejected_verified": False,
                "verifier_sha256": "b"*64,
                "verifier_type": "numeric_oracle" if idx % 2 else "sandbox_unit",
                "split": split,
            })
    return result


def suite(name):
    result = []
    for domain in ("math", "code", "reasoning"):
        for idx in range(200):
            kind = {"math":"numeric", "code":"python", "reasoning":"boolean"}[domain]
            result.append({
                "id": f"{name}-{domain}-{idx}",
                "domain": domain, "family": f"method-{idx%10}",
                "source_id": f"{name}-{domain}-{idx}",
                "kind": kind,
                "prompt": f"{name} isolated {domain} question number {idx} with a completely separate vocabulary.",
                "answer": "17" if kind != "python" else "def f():return 17",
                "tests_sha256": "c"*64 if kind == "python" else None,
            })
    return result


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in rows), encoding="utf8")
    return path


class E1SmokeTests(unittest.TestCase):
    def test_verified_split_and_fixed_smoke_plan(self):
        rows = pairs()
        result = validate_pairs(rows)
        self.assertEqual(result["train"], 50)
        self.assertEqual(result["validation"], 10)
        self.assertEqual(result["pairs"], 60)
        self.assertFalse(result["external_verifier_independence_proven"])
        for method in ("dpo","orpo"):
            plan = smoke_plan(method)
            self.assertEqual(plan["max_steps"], 30)
            self.assertEqual(plan["micro_batch_size"], 1)
            self.assertEqual(plan["gradient_accumulation_steps"], 4)
            self.assertEqual(plan["max_length"], 512)
            self.assertEqual(plan["run_gpu"], False)
            self.assertFalse(plan["training_performed"])
        with self.assertRaises(ValueError):
            smoke_plan("grpo")

    def test_rejects_false_verification_bad_labels_and_duplicated_source(self):
        base = pairs()
        for updated in (
            dict(base[0], chosen_verified=False),
            dict(base[0], rejected_verified=True),
            dict(base[0], chosen=base[0]["rejected"]),
            dict(base[0], verifier_sha256="c"),
            dict(base[0], verifier_type="unverified"),
            dict(base[0], split="release"),
            dict(base[0], source_id=base[1]["source_id"]),
            dict(base[0], prompt="  "),
        ):
            rows = [updated,*base[1:]]
            with self.assertRaises(ValueError):
                validate_pairs(rows)

    def test_imbalanced_family_rejected(self):
        bad = pairs()
        bad[0] = dict(bad[0],family="unique-method-1")
        with self.assertRaises(ValueError):
            validate_pairs(bad)

    def test_v6_suite_tasks_never_flow_into_preference_training(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            pair_file=write_jsonl(root/"pairs.jsonl",pairs())
            dev=write_jsonl(root/"dev.jsonl",suite("dev"))
            release=write_jsonl(root/"release.jsonl",suite("release"))
            report=preflight(pair_file,dev,release,method="dpo")
            self.assertFalse(report["training_performed"])
            self.assertFalse(report["release_validated"])
            self.assertFalse(report["human_verifier_correctness_proven"])
            self.assertEqual(report["excluded_suite_rows"],1200)
            self.assertEqual(len(report["pairs_sha256"]),64)
            self.assertNotIn("An original task", json.dumps(report))
            changed=pairs()
            changed[0]=dict(changed[0],source_id="release-code-0")
            write_jsonl(pair_file,changed)
            with self.assertRaises(ValueError):
                preflight(pair_file,dev,release,method="orpo")
            changed=pairs()
            changed[0]=dict(changed[0],prompt=suite("dev")[0]["prompt"])
            write_jsonl(pair_file,changed)
            with self.assertRaises(ValueError):
                preflight(pair_file,dev,release,method="dpo")

    def test_gpu_training_denied_without_explicit_authorization(self):
        for method in ("dpo","orpo"):
            with self.assertRaises(ValueError):
                require_gpu_opt_in(method, "no")
            with self.assertRaises(ValueError):
                require_gpu_opt_in(method, "")
            self.assertTrue(require_gpu_opt_in(method, "RUN_NERAL_E1_30_STEP_GPU_SMOKE"))
        with self.assertRaises(ValueError):
            require_gpu_opt_in("grpo","RUN_NERAL_E1_30_STEP_GPU_SMOKE")


if __name__ == "__main__":
    unittest.main()
