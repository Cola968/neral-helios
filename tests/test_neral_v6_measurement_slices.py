"""CPU contracts for verified-step measurement slices. Not a score gain."""
import ast
import json
import re
import unittest
from pathlib import Path

from scripts.neral_decontamination import normalize, shingles
from scripts.neral_reasoning_train_candidate import candidate, verify_case
from scripts.neral_v6_compositional import FAMILIES, PER_FAMILY, make_case
from scripts.neral_v6_dev72_regression_gate import analyze_dev72
from scripts.neral_v6_fresh_dev_eval import (
    build_cases, experiment_decision, phase2_training_gate,
)
from scripts.neral_v6_locked_holdout import (
    HOLDOUT_FAMILIES, holdout_commitment, locked_holdout_cases,
    sealed_holdout_counts,
)
from scripts.neral_v6_measurement_slices import (
    MEASUREMENT_SEED, RETENTION_PAIR_COUNT, TRANSFER_FAMILIES,
    partition_visible, public_manifest, retention_cases,
    safe_integer_expression, slice_sha256, transfer_cases,
)
from scripts.neral_v6_r16_gpu import _compact_trace, _verified_step_target, build_experiment
from scripts.neral_v6_regression_gate import RegressionRejected, audit_candidate_report
from scripts.neral_v61_new_dev_eval import cases72

ROOT = Path(__file__).resolve().parents[1]
RETENTION_SHA = "333bfcb4e92a9b9d6d27c2d9a6e6181c13f9c629a3a8d2bb35885a104a319054"
TRANSFER_SHA = "9604fe1910698e572d408e58e478f7446f393409d0997db3ab2bd99f769b48a2"
HOLDOUT_SHA = "6332c37ec3927475aba83da312f7943064d7c400be516caaf1dcf085cf9c3270"
COMPACT_TRAIN_SHA = "c47256233a40ad6587e702732819b0166c586a1e55804f74eb06a2071dbd3b9b"
VERIFIED_BOUND_TRAIN_SHA = "72b9c75ad154c19a941e072db6f57fa769f3aa8b3d221647ff08c5c8559fb5d8"
VALIDATION_SHA = "2e416b4206039f6305d305cc3b5a8ff4b6292ff3bc4ae30c759c854d100aa299"
DECISION_FILES = (
    "scripts/neral_v6_r16_gpu.py",
    "scripts/neral_v6_fresh_dev_eval.py",
    "scripts/neral_v61_new_dev_eval.py",
    "scripts/neral_v6_dev72_regression_gate.py",
    "scripts/neral_v6_regression_gate.py",
    "scripts/neral_v6_dev72_regression_audit.py",
    "scripts/neral_six_arm_eval.py",
    "scripts/neral_six_arm_gpu.py",
    "scripts/neral_v6_slice_compare.py",
)


def _grams(texts):
    found = set()
    for text in texts:
        found |= shingles(text, n=10)
    return found


def _hits(prompts, gram_set):
    return sum(1 for prompt in prompts if shingles(prompt, n=10) & gram_set)


def _walk_expression(node):
    if isinstance(node, ast.Expression):
        return _walk_expression(node.body)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _walk_expression(node.left) + _walk_expression(node.right)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _walk_expression(node.left) * _walk_expression(node.right)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    raise AssertionError("Unexpected expression node")


def _list_in(prompt):
    return ast.literal_eval(re.search(r"\[[^\]]+\]", prompt).group(0))


class MeasurementSliceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retention = retention_cases()
        cls.transfer = transfer_cases()
        cls.holdout = locked_holdout_cases()
        cls.manifest = public_manifest()

    def test_assignment_is_stable_and_pinned(self):
        self.assertEqual(self.retention, retention_cases())
        self.assertEqual(self.transfer, transfer_cases())
        self.assertEqual(self.holdout, locked_holdout_cases())
        self.assertEqual(slice_sha256(self.retention), RETENTION_SHA)
        self.assertEqual(slice_sha256(self.transfer), TRANSFER_SHA)
        self.assertEqual(slice_sha256(self.holdout), HOLDOUT_SHA)
        self.assertEqual(self.manifest["retention_sha256"], RETENTION_SHA)
        self.assertEqual(self.manifest["transfer_sha256"], TRANSFER_SHA)
        self.assertEqual(holdout_commitment()["sha256"], HOLDOUT_SHA)
        self.assertEqual((len(self.retention), len(self.transfer), len(self.holdout)),
                         (164, 64, 96))
        self.assertFalse(self.manifest["assignment_depends_on_scores"])
        self.assertFalse(self.manifest["locked_holdout_ids_visible_here"])
        self.assertFalse(self.manifest["training_authorized"])
        self.assertFalse(self.manifest["claimed_score_gain"])
        self.assertIn("pair index 0..15", self.manifest["assignment_rule"])
        self.assertIn("pair index 16..31", self.manifest["assignment_rule"])
        for case in self.retention + self.transfer + self.holdout:
            self.assertEqual(case["required_prompt_protocol"], "matched_fewshot")
            self.assertEqual(case["decoding"], "greedy")
            self.assertEqual(case["max_new_tokens"], 160)

    def test_slices_are_disjoint_and_families_follow_the_rule(self):
        ids = [case["id"] for case in self.retention + self.transfer + self.holdout]
        prompts = [normalize(case["prompt"]) for case in
                   self.retention + self.transfer + self.holdout]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(prompts), len(set(prompts)))
        composition = [case for case in self.retention if case["stratum"] == "composition"]
        kept = [case for case in self.retention if case["stratum"] == "counterexample"]
        sealed_pairs = [case for case in self.holdout if case["stratum"] == "counterexample"]
        self.assertEqual(len(composition), 100)
        self.assertEqual({case["family"] for case in composition}, set(FAMILIES))
        self.assertEqual(
            {(case["family"], case["index"]) for case in composition},
            {(family, index) for family in FAMILIES
             for index in range(PER_FAMILY["holdout"])})
        self.assertEqual({case["pair_index"] for case in kept}, set(range(RETENTION_PAIR_COUNT)))
        self.assertEqual({case["pair_index"] for case in sealed_pairs},
                         set(range(RETENTION_PAIR_COUNT, 32)))
        self.assertEqual(len(kept), 64)
        self.assertEqual(len(sealed_pairs), 64)
        proof_ids = {case["measurement_proof_id"] for case in kept + sealed_pairs}
        self.assertEqual(len(proof_ids), 128)
        self.assertTrue(all(item.startswith(f"{MEASUREMENT_SEED}:") for item in proof_ids))
        self.assertEqual([case["family"] for case in self.transfer][::8],
                         list(TRANSFER_FAMILIES))
        unseen = [case for case in self.holdout if case["stratum"] == "unseen_family"]
        self.assertEqual([case["family"] for case in unseen][::8], list(HOLDOUT_FAMILIES))
        training_families = set(FAMILIES) | {"ordered_price", "segmented_fuel"}
        self.assertTrue(set(TRANSFER_FAMILIES).isdisjoint(training_families))
        self.assertTrue(set(HOLDOUT_FAMILIES).isdisjoint(training_families))
        self.assertTrue(set(HOLDOUT_FAMILIES).isdisjoint(TRANSFER_FAMILIES))

    def test_answers_match_independent_oracles(self):
        for case in self.retention:
            if case["stratum"] != "composition":
                continue
            source = make_case(case["family"], "holdout", case["index"])
            self.assertEqual(case["prompt"], source["messages"][1]["content"])
            assistant = source["messages"][2]["content"]
            expected = (assistant.split("FINAL_ANSWER:", 1)[1].strip()
                        if "FINAL_ANSWER:" in assistant else assistant.strip())
            self.assertEqual(case["answer"], expected)
        rows, proofs = candidate(MEASUREMENT_SEED)
        by_id = {f"{MEASUREMENT_SEED}:{proof['id']}": (row, proof)
                 for row, proof in zip(rows, proofs)}
        for case in self.retention + self.holdout:
            if case["stratum"] != "counterexample":
                continue
            row, proof = by_id[case["measurement_proof_id"]]
            self.assertTrue(verify_case(proof, row))
            self.assertEqual(case["prompt"], row["messages"][1]["content"])
            self.assertEqual(case["answer"], proof["answer"])
        for case in self.transfer:
            self.assertEqual(case["answer"], _transfer_oracle(case))
        for case in self.holdout:
            if case["stratum"] == "unseen_family":
                self.assertEqual(case["answer"], _holdout_oracle(case))
        self.assertNotEqual(
            safe_integer_expression("((2 + 3) * (4 + 1))"),
            safe_integer_expression("2 + 3 * 4 + 1"))

    def test_exact_exclusion_and_where_10grams_are_expected(self):
        train, val, _ = build_experiment(
            counterfactual_target="verified_steps", length_source="bound")
        users = [row["messages"][1]["content"] for row in train + val]
        assistants = [row["messages"][2]["content"] for row in train + val]
        screens = ([case["prompt"] for case in build_cases()]
                   + [case["prompt"] for case in cases72()])
        user_exact = {normalize(text) for text in users}
        screen_exact = {normalize(text) for text in screens}
        user_grams = _grams(users)
        assistant_grams = _grams(assistants)
        screen_grams = _grams(screens)
        retention_prompts = [case["prompt"] for case in self.retention]
        transfer_prompts = [case["prompt"] for case in self.transfer]
        holdout_prompts = [case["prompt"] for case in self.holdout]
        unseen_prompts = [case["prompt"] for case in self.holdout
                          if case["stratum"] == "unseen_family"]
        for prompts in (retention_prompts, transfer_prompts, holdout_prompts):
            self.assertTrue(all(normalize(prompt) not in user_exact for prompt in prompts))
            self.assertTrue(all(normalize(prompt) not in screen_exact for prompt in prompts))
        self.assertEqual(_hits(retention_prompts, user_grams), 138)
        self.assertEqual(_hits(holdout_prompts, user_grams), 64)
        self.assertEqual(_hits(transfer_prompts, user_grams), 0)
        self.assertEqual(_hits(unseen_prompts, user_grams), 0)
        self.assertEqual(_hits(transfer_prompts, assistant_grams), 0)
        self.assertEqual(_hits(unseen_prompts, assistant_grams), 0)
        self.assertEqual(_hits(transfer_prompts, screen_grams), 0)
        self.assertEqual(_hits(unseen_prompts, screen_grams), 0)
        self.assertEqual(_hits(retention_prompts, screen_grams), 0)
        self.assertEqual(_hits(transfer_prompts, _grams(retention_prompts)), 0)
        self.assertEqual(_hits(unseen_prompts, _grams(retention_prompts)), 0)
        self.assertEqual(_hits(unseen_prompts, _grams(transfer_prompts)), 0)
        composition = [case["prompt"] for case in self.retention
                       if case["stratum"] == "composition"]
        self.assertEqual(_hits(composition, user_grams), 74)

    def test_holdout_ids_stay_out_of_manifest_commitment_and_decision_code(self):
        hidden = []
        for case in self.holdout:
            hidden.extend((case["id"], case["prompt"]))
        public = json.dumps(self.manifest, ensure_ascii=False)
        commitment = json.dumps(holdout_commitment(), ensure_ascii=False)
        for secret in hidden:
            self.assertNotIn(secret, public)
            self.assertNotIn(secret, commitment)
        measurement = (ROOT / "scripts/neral_v6_measurement_slices.py").read_text(encoding="utf8")
        self.assertNotIn("from scripts.neral_v6_locked_holdout", measurement)
        self.assertNotIn("import neral_v6_locked_holdout", measurement)
        for relative in DECISION_FILES:
            text = (ROOT / relative).read_text(encoding="utf8")
            self.assertNotIn("neral_v6_locked_holdout", text)
            self.assertNotIn("locked_holdout_cases(", text)
        counts = sealed_holdout_counts(
            [{"text": f"FINAL_ANSWER: {case['answer']}", "complete": True}
             for case in self.holdout])
        encoded = json.dumps(counts, ensure_ascii=False)
        for case in self.holdout:
            self.assertNotIn(case["id"], encoded)
            self.assertNotIn(case["prompt"], encoded)
            if len(case["answer"]) >= 8:
                self.assertNotIn(case["answer"], encoded)
        self.assertEqual(counts["strict_correct"], 96)
        self.assertFalse(counts["claimed_score_gain"])
        self.assertFalse(counts["ids_included"])

    def test_visible_partition_and_gate_refuse_holdout_material(self):
        completions = []
        for index, case in enumerate(self.transfer):
            if index == 0:
                completions.append({"text": "", "complete": False})
            elif index == 1:
                completions.append({"text": "FINAL_ANSWER: 999999", "complete": True})
            elif index == 2:
                completions.append({"text": "no numeric token", "complete": True})
            else:
                completions.append({"text": case["answer"], "complete": True})
        counts = partition_visible(self.transfer, completions)
        self.assertEqual(counts["incomplete"], 1)
        self.assertEqual(counts["wrong_readable_number"], 1)
        self.assertEqual(counts["unreadable"], 1)
        self.assertEqual(counts["strict_correct"], 61)
        self.assertEqual(sum(counts[name] for name in (
            "strict_correct", "wrong_readable_number", "unreadable", "incomplete")), 64)
        self.assertFalse(counts["claimed_score_gain"])
        report = {
            "prompt_protocol": "matched_fewshot",
            "release_validated": False, "adapter_promoted": False,
            "weights_promoted": False, "claimed_score_gain": False,
            "general_improvement_proven": False, "total": 32,
            "scores": {name: {
                "strict_correct": 1, "wrong_readable_number": 0,
                "unreadable": 0, "incomplete": 0, "correct": 1,
                "ungraded_completed": 0, "total": 1,
            } for name in ("qwen_fewshot", "neral_v5", "neral_v6")},
            "retention_slice": {"role": "retention", "count": 164},
        }
        self.assertEqual(phase2_training_gate(report)["decision"], "ok")
        secret = self.holdout[0]["id"]
        leaked = dict(report, locked_holdout_item_ids=[secret])
        gate = phase2_training_gate(leaked)
        decision = experiment_decision(leaked)
        self.assertEqual(gate["decision"], "refuse")
        self.assertEqual(decision["decision"], "refuse")
        self.assertIn("locked_holdout_is_not_a_gate_input", gate["reasons"])
        self.assertFalse(decision["reads_locked_holdout_item_ids"])
        self.assertEqual(decision["measurement_change"],
                         "scorer_protocol_not_a_learning_change")
        blob = json.dumps({"gate": gate, "decision": decision}, ensure_ascii=False)
        self.assertNotIn(secret, blob)
        self.assertNotIn(self.holdout[0]["prompt"], blob)
        aggregate = dict(report, locked_holdout={"strict_correct": 90, "total": 96})
        self.assertEqual(experiment_decision(aggregate)["decision"], "refuse")
        self.assertEqual(phase2_training_gate(holdout_commitment())["decision"], "refuse")
        with self.assertRaisesRegex(ValueError, "not an input to this decision"):
            analyze_dev72({"locked_holdout_item_ids": [secret]})
        with self.assertRaisesRegex(RegressionRejected, "not an input to this decision"):
            audit_candidate_report({"locked_holdout_cases": [{"id": secret}]})


def _transfer_oracle(case):
    if case["family"] == "inclusive_range_sum":
        low, high = [int(item) for item in re.findall(r"\d+", case["prompt"])]
        return str((high - low + 1) * (low + high) // 2)
    if case["family"] == "nesting_depth":
        words = case["prompt"].split(": ", 1)[1].split(". Report", 1)[0].split()
        depth = maximum = 0
        for word in words:
            if word == "open":
                depth += 1
                maximum = max(maximum, depth)
            elif word == "close":
                depth -= 1
            else:
                raise AssertionError(word)
        return str(maximum)
    if case["family"] == "inversion_pairs":
        values = _list_in(case["prompt"])
        return str(sum(1 for i, left in enumerate(values)
                       for right in values[i + 1:] if left > right))
    if case["family"] == "digit_sum":
        number = int(re.search(r"integer (\d+)", case["prompt"]).group(1))
        return str(sum(int(digit) for digit in str(number)))
    if case["family"] == "adjacent_ascents":
        values = _list_in(case["prompt"])
        return str(sum(1 for pos in range(1, len(values)) if values[pos] > values[pos - 1]))
    if case["family"] == "alternating_sum":
        values = _list_in(case["prompt"])
        return str(sum(value if pos % 2 == 0 else -value
                       for pos, value in enumerate(values)))
    if case["family"] == "capped_sum":
        values = _list_in(case["prompt"])
        cap = int(re.search(r"ceiling (\d+)", case["prompt"]).group(1))
        return str(sum(min(value, cap) for value in values))
    values = _list_in(case["prompt"])
    best, changes = values[0], 1
    for value in values[1:]:
        if value > best:
            best = value
            changes += 1
    return str(changes)


def _holdout_oracle(case):
    if case["family"] == "odd_median":
        ordered = sorted(_list_in(case["prompt"]))
        return str(ordered[len(ordered) // 2])
    if case["family"] == "population_bitcount":
        number = int(re.search(r"\d+", case["prompt"]).group(0))
        return str(number.bit_count())
    if case["family"] == "distinct_values":
        return str(len(set(_list_in(case["prompt"]))))
    expression = case["prompt"].split(": ", 1)[1].split(". Use ", 1)[0]
    return str(_walk_expression(ast.parse(expression, mode="eval")))


class TrainingMixtureAudit(unittest.TestCase):
    def test_only_128_counterexamples_drop_verified_operators(self):
        compact_train, compact_val, compact_meta = build_experiment()
        verified_train, verified_val, verified_meta = build_experiment(
            counterfactual_target="verified_steps", length_source="bound")
        self.assertEqual(compact_meta["train_sha256"], COMPACT_TRAIN_SHA)
        self.assertEqual(verified_meta["train_sha256"], VERIFIED_BOUND_TRAIN_SHA)
        self.assertEqual(compact_meta["validation_sha256"], VALIDATION_SHA)
        self.assertEqual(verified_meta["validation_sha256"], VALIDATION_SHA)
        self.assertEqual(compact_val, verified_val)
        compact = {row["messages"][1]["content"]: row for row in compact_train}
        verified = {row["messages"][1]["content"]: row for row in verified_train}
        self.assertEqual(set(compact), set(verified))
        changed = [prompt for prompt, row in compact.items() if row != verified[prompt]]
        rows, proofs = candidate()
        by_prompt = {row["messages"][1]["content"]: (row, proof)
                     for row, proof in zip(rows, proofs)}
        self.assertEqual(set(changed), set(by_prompt))
        self.assertEqual(len(changed), 128)
        for prompt in changed:
            raw, proof = by_prompt[prompt]
            verified_text = _verified_step_target(proof, raw)
            compact_text = _compact_trace(proof, raw)
            self.assertIn("*", verified_text)
            self.assertIn("+", verified_text)
            self.assertNotIn("*", compact_text)
            self.assertNotIn("+", compact_text)
            self.assertEqual(
                compact_text.split("FINAL_ANSWER:", 1)[1].strip(), proof["answer"])
            self.assertEqual(
                verified_text.split("FINAL_ANSWER:", 1)[1].strip(), proof["answer"])
            self.assertIn("*", verified[prompt]["messages"][2]["content"])
            self.assertIn("+", verified[prompt]["messages"][2]["content"])
        answer_only = check_rows = 0
        for family in FAMILIES:
            for index in range(PER_FAMILY["train"]):
                text = make_case(family, "train", index)["messages"][2]["content"]
                if text.startswith("CHECK:"):
                    check_rows += 1
                else:
                    answer_only += 1
                    self.assertNotIn("first=", text)
                    self.assertNotIn("a_per_min=", text)
        self.assertEqual((check_rows, answer_only), (200, 400))
        for row in verified_train:
            if row["messages"][1]["content"] in by_prompt:
                continue
            text = row["messages"][2]["content"]
            self.assertNotIn("first=", text)
            self.assertNotIn("a_per_min=", text)
            self.assertNotIn("*", text)

    def test_train_and_dev_prompts_do_not_match(self):
        train, val, _ = build_experiment(
            counterfactual_target="verified_steps", length_source="bound")
        users = [row["messages"][1]["content"] for row in train + val]
        screens = ([case["prompt"] for case in build_cases()]
                   + [case["prompt"] for case in cases72()])
        exact_users = {normalize(text) for text in users}
        exact_screens = {normalize(text) for text in screens}
        self.assertEqual(sum(normalize(text) in exact_screens for text in users), 0)
        self.assertEqual(sum(normalize(text) in exact_users for text in screens), 0)
        self.assertEqual(_hits(users, _grams(screens)), 0)
        self.assertEqual(_hits(screens, _grams(users)), 0)
        holdout = [make_case(family, "holdout", index)["messages"][1]["content"]
                   for family in FAMILIES for index in range(PER_FAMILY["holdout"])]
        self.assertEqual(sum(normalize(text) in exact_users for text in holdout), 0)
        self.assertEqual(_hits(holdout, _grams(users)), 74)


if __name__ == "__main__":
    unittest.main()
