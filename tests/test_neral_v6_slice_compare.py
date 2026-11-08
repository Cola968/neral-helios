"""CPU contract for the retention/transfer comparison. Not a score gain."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.neral_release_gate import exact_mcnemar_p
from scripts.neral_v6_fresh_dev_eval import FEWSHOT, experiment_decision, messages_for_protocol
from scripts.neral_v6_measurement_slices import slice_sha256
from scripts.neral_v6_slice_compare import (
    ARMS, AUTHORIZE, DECODING, OUTCOMES, PROTOCOL, RETENTION_SHA, TRANSFER_SHA,
    arm_request, create_fresh, decoding_sha256, dry_run, load_visible_slices,
    main, minimum_one_sided_discordances, paired_stats, refuse_nonvisible,
    run_gpu_arm,
)

ROOT = Path(__file__).resolve().parents[1]
GPU_WORKFLOW = ROOT / ".github/workflows/neral-v6-slice-compare-gpu.yml"
CPU_WORKFLOW = ROOT / ".github/workflows/neral-v6-slice-compare-ci.yml"


class SliceCompareContracts(unittest.TestCase):
    def test_pins_and_shared_protocol_are_checked_up_front(self):
        retention, transfer = load_visible_slices()
        self.assertEqual(slice_sha256(retention), RETENTION_SHA)
        self.assertEqual(slice_sha256(transfer), TRANSFER_SHA)
        self.assertEqual((len(retention), len(transfer)), (164, 64))
        self.assertEqual(minimum_one_sided_discordances(), 6)
        self.assertEqual(exact_mcnemar_p(6, 0), 2 / 64)
        self.assertEqual(exact_mcnemar_p(5, 0), 2 / 32)
        case = transfer[0]
        messages = None
        for arm in ARMS:
            built, decoding = arm_request(case, arm)
            self.assertIs(decoding, DECODING)
            self.assertFalse(decoding["do_sample"])
            self.assertEqual(decoding["max_new_tokens"], 160)
            if messages is None:
                messages = built
            self.assertEqual(built, messages)
        self.assertEqual(
            [item["content"] for item in messages if item["role"] == "user"][:3],
            [question for question, _answer in FEWSHOT])
        self.assertNotEqual(
            messages_for_protocol(case, "neral_v5", "historical"), messages)
        self.assertEqual(messages_for_protocol(case, "qwen_fewshot", PROTOCOL), messages)
        with self.assertRaises(ValueError):
            refuse_nonvisible([{"id": "locked-holdout-x", "role": "locked_holdout",
                                "required_prompt_protocol": PROTOCOL,
                                "decoding": "greedy", "max_new_tokens": 160}])

    def test_paired_stats_match_exact_mcnemar_and_a_zero_gap(self):
        reference = [True] * 10 + [False] * 6
        candidate = [True] * 16
        families = ["same"] * 16
        stats = paired_stats(reference, candidate, families)
        again = paired_stats(reference, candidate, families)
        self.assertEqual(stats["improvements"], 6)
        self.assertEqual(stats["regressions"], 0)
        self.assertEqual(stats["exact_mcnemar_p"], exact_mcnemar_p(6, 0))
        self.assertEqual(stats, again)
        low, high = stats["item_bootstrap_95_ci"]
        self.assertGreater(low, 0)
        self.assertLess(high, 1)
        self.assertLessEqual(low, stats["gap"])
        self.assertLessEqual(stats["gap"], high)
        tied = paired_stats([True, False, True], [True, False, True], ["a", "b", "a"])
        self.assertEqual(tied["exact_mcnemar_p"], 1.0)
        self.assertEqual(tied["item_bootstrap_95_ci"], [0.0, 0.0])
        self.assertEqual(tied["family_cluster_bootstrap_95_ci"], [0.0, 0.0])

    def test_stub_dry_run_scores_reports_and_gates_without_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = "torch" in sys.modules
            summary = dry_run(root)
            self.assertEqual("torch" in sys.modules, before)
            created = [path for path in root.iterdir() if path.is_dir()]
            self.assertEqual(len(created), 1)
            run_dir = created[0]
            self.assertTrue(run_dir.name.startswith("slice-compare-"))
            stored = json.loads((run_dir / "summary.json").read_text("utf8"))
            self.assertEqual(stored, summary)
            self.assertEqual(summary["model_backend"], "deterministic_stub")
            self.assertFalse(summary["gpu_inference_performed"])
            self.assertFalse(summary["weights_loaded"])
            self.assertFalse(summary["training_authorized"])
            self.assertFalse(summary["claimed_score_gain"])
            self.assertFalse(summary["sealed_slice_loaded"])
            self.assertTrue(summary["arms_share_prompt_sha256"])
            self.assertTrue(summary["arms_share_decoding_sha256"])
            self.assertEqual(summary["decoding_sha256"], decoding_sha256())
            self.assertEqual(summary["phase2_gate"]["decision"], "ok")
            self.assertEqual(summary["phase2_gate"]["phase2_training"],
                             "v61_comparison_not_in_report")
            self.assertFalse(summary["phase2_gate"]["training_authorized"])
            self.assertEqual(summary["sample_size"]["transfer_minimum_one_sided_gap_pp"],
                             9.375)
            for role, count in (("retention", 164), ("transfer", 64)):
                block = summary["slices"][role]
                for arm in ARMS:
                    partition = block["by_arm"][arm]
                    self.assertEqual(sum(partition[name] for name in OUTCOMES), count)
                pair = block["paired"]["neral_v6_minus_neral_v5"]
                self.assertEqual(pair["arm_a"], "neral_v5")
                self.assertEqual(pair["arm_b"], "neral_v6")
                self.assertIn("exact_mcnemar_p", pair)
                self.assertLessEqual(pair["family_cluster_bootstrap_95_ci"][0],
                                     pair["family_cluster_bootstrap_95_ci"][1])
            self.assertEqual(len(summary["items"]), (164 + 64) * 3)
            self.assertTrue(all("text" not in row for row in summary["items"]))
            private = (run_dir / "items.jsonl").read_text("utf8")
            self.assertIn("\"text\"", private)
            self.assertNotIn("locked_holdout", private)
            self.assertNotIn("locked-holdout", json.dumps(summary))
            leaked = dict(summary, locked_holdout_item_ids=["locked-holdout-secret"])
            self.assertEqual(experiment_decision(leaked)["decision"], "refuse")
            with self.assertRaises(ValueError):
                create_fresh(root, run_dir.name)
            second = dry_run(root)
            self.assertFalse(second["gpu_inference_performed"])
            self.assertEqual(len(list(root.iterdir())), 2)

    def test_gpu_mode_refuses_the_wrong_permit_before_loading_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "not-created"
            before = "torch" in sys.modules
            with self.assertRaisesRegex(ValueError, "exact read-only permit"):
                run_gpu_arm(dest, "neral_v6", Path(tmp), Path(tmp), Path(tmp),
                            "TRAIN_NERAL_V6_VERIFIED_STEPS_120_EXPERIMENTAL")
            self.assertFalse(dest.exists())
            self.assertEqual("torch" in sys.modules, before)
            with self.assertRaisesRegex(ValueError, "exact read-only permit"):
                main(["--mode", "run", "--run-dir", str(dest), "--arm", "qwen_fewshot",
                      "--base", str(tmp), "--v5-adapter", str(tmp),
                      "--v6-adapter", str(tmp), "--authorize", "not-the-permit"])

    def test_cli_dry_run_and_output_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            with self.assertRaisesRegex(ValueError, "adapter directory"):
                create_fresh(adapter, "slice-compare-inside")
            main(["--mode", "dry-run", "--out-root", str(root)])
            created = [path for path in root.iterdir() if path.is_dir() and path != adapter]
            self.assertEqual(len(created), 1)
            summary = json.loads((created[0] / "summary.json").read_text("utf8"))
            self.assertEqual(summary["status"], "NERAL_V6_SLICE_COMPARE_STUB_DRY_RUN")
            self.assertFalse(summary["training_authorized"])
            self.assertFalse(summary["claimed_score_gain"])

    def test_workflows_cannot_start_from_this_branch(self):
        gpu = GPU_WORKFLOW.read_text(encoding="utf8")
        cpu = CPU_WORKFLOW.read_text(encoding="utf8")
        module = (ROOT / "scripts/neral_v6_slice_compare.py").read_text(encoding="utf8")
        self.assertIn("workflow_dispatch:", gpu)
        for banned in ("pull_request:", "push:", "schedule:", "workflow_call:",
                       "workflow_run:"):
            self.assertNotIn(banned, gpu)
        self.assertIn("refs/heads/main", gpu)
        self.assertEqual(gpu.count(AUTHORIZE), 5)
        self.assertIn("neral-gpu", gpu)
        self.assertNotIn("TRAIN_NERAL", gpu)
        self.assertNotIn("neral_v6_locked_holdout", gpu)
        self.assertNotIn("locked_holdout_cases", gpu)
        for line in cpu.splitlines():
            if "runs-on:" in line:
                self.assertNotIn("neral-gpu", line)
        self.assertIn("pull_request:", cpu)
        self.assertIn("push:", cpu)
        self.assertIn("branches: [main]", cpu)
        self.assertNotIn("workflow_dispatch:", cpu)
        self.assertNotIn("neral_v6_locked_holdout", module)
        self.assertNotIn("locked_holdout_cases", module)
        self.assertNotIn("--prompt-protocol", module)


if __name__ == "__main__":
    unittest.main()
