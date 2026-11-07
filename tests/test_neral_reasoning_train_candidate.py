"""Arithmetic invariants and adversarial mutations, without model or holdout data."""
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.neral_reasoning_train_candidate import (
    FAMILIES, GATES, PAIRS_PER_FAMILY, _graph, _completion, _prompt,
    SYSTEM, build, candidate, verify_case,
)


class ReasoningCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.proofs = candidate()

    def test_every_step_replays_and_pairs_change_only_task_rule(self):
        self.assertEqual(len(self.rows), 2 * 2 * PAIRS_PER_FAMILY)
        self.assertEqual((self.rows, self.proofs), candidate())
        self.assertEqual(set(p["family"] for p in self.proofs), set(FAMILIES))
        self.assertEqual(len({p["id"] for p in self.proofs}), len(self.proofs))
        self.assertEqual(len({r["messages"][1]["content"] for r in self.rows}), len(self.rows))
        for row, proof in zip(self.rows, self.proofs):
            self.assertTrue(verify_case(proof, row))
        for offset in range(0, len(self.proofs), 2):
            left, right = self.proofs[offset:offset + 2]
            self.assertEqual(left["facts"], right["facts"])
            self.assertEqual(left["pair_id"], right["pair_id"])
            self.assertEqual(left["style"], right["style"])
            self.assertEqual((left["variant"], right["variant"]), (0, 1))
            self.assertNotEqual(left["answer"], right["answer"])

    def test_independent_closed_form_decimal_price_and_order_effect(self):
        with localcontext() as ctx:
            ctx.prec = 60
            for proof in self.proofs:
                if proof["family"] != "ordered_price":
                    continue
                f = proof["facts"]
                start, fee = Decimal(f["base_cents"]), Decimal(f["fee_cents"])
                down, up = 1 - Decimal(f["discount_pct"]) / 100, 1 + Decimal(f["surcharge_pct"]) / 100
                # Independent factored expression; no producer graph or Fraction calculation.
                exact = start * down * up + fee * (up if proof["variant"] == 0 else down)
                cents = exact.quantize(Decimal(1), rounding=ROUND_HALF_UP)
                answer = f"{cents / 100:.2f} EUR".replace(".", ",")
                self.assertEqual(proof["answer"], answer)
                claimed = Fraction(proof["nodes"][-1]["value"])
                self.assertEqual(exact, Decimal(claimed.numerator) / claimed.denominator)

    def test_fuel_unit_conversion_and_closed_form_numerator(self):
        for proof in self.proofs:
            if proof["family"] != "segmented_fuel":
                continue
            f = proof["facts"]
            self.assertNotEqual(f["rate_a"] % 60, 0)
            self.assertNotEqual(f["rate_b"] % 60, 0)
            reserve_rate = f["rate_a"] if proof["variant"] == 0 else f["rate_b"]
            expected_numerator = (f["rate_a"] * f["minutes_a"] +
                                  f["rate_b"] * f["minutes_b"] +
                                  reserve_rate * f["reserve_min"] + 60 * f["taxi_kg"])
            final = Fraction(proof["answer"].removesuffix(" kg"))
            # Integer cross-multiplication avoids the producer's sequence of divisions.
            self.assertEqual(60 * final.numerator, expected_numerator * final.denominator)
            self.assertGreater(final.denominator, 1)

    def test_round_half_up_tie_at_final_cent_only(self):
        facts = {"base_cents": 100, "discount_pct": 25, "surcharge_pct": 50, "fee_cents": 2}
        nodes, answer = _graph("ordered_price", facts, 0)
        self.assertEqual(nodes[-1]["value"], "231/2")
        self.assertEqual(answer, "1,16 EUR")
        proof = {"family": "ordered_price", "facts": facts, "variant": 0,
                 "style": 0, "nodes": nodes, "answer": answer}
        row = {"messages": [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": _prompt("ordered_price", facts, 0, 0)},
            {"role": "assistant", "content": _completion("ordered_price", facts, nodes, answer)}]}
        self.assertTrue(verify_case(proof, row))
        proof["answer"] = "1,15 EUR"
        with self.assertRaisesRegex(ValueError, "rounding"):
            verify_case(proof, row)

    def test_mutations_of_intermediate_graph_answer_and_prompt_are_rejected(self):
        mutations = (
            lambda p, r: p["nodes"][0].update(value=str(Fraction(p["nodes"][0]["value"]) + 1)),
            lambda p, r: p["nodes"][0].update(value=str(Fraction(p["nodes"][0]["value"]) + Fraction(1, 10**100))),
            lambda p, r: p["nodes"][0].update(op="add"),
            lambda p, r: p["nodes"][0].update(args=list(reversed(p["nodes"][0]["args"]))),
            lambda p, r: p.update(answer="0 EUR"),
            lambda p, r: r["messages"][1].update(content="Andere Aufgabe"),
            lambda p, r: r["messages"][2].update(content="CHECK: 1 = 2\nFINAL_ANSWER: 0"),
            lambda p, r: p["facts"].update(fee_cents=p["facts"]["fee_cents"] + 1),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                proof, row = deepcopy(self.proofs[0]), deepcopy(self.rows[0])
                mutate(proof, row)
                with self.assertRaises(ValueError):
                    verify_case(proof, row)
        offset = next(i for i, p in enumerate(self.proofs) if p["family"] == "segmented_fuel")
        proof, row = deepcopy(self.proofs[offset]), deepcopy(self.rows[offset])
        proof["nodes"][4]["args"][0] = "b_per_min"
        with self.assertRaisesRegex(ValueError, "contradicts"):
            verify_case(proof, row)

    def test_only_train_files_and_explicit_pending_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "candidate"
            manifest = build(out)
            self.assertEqual({p.name for p in out.iterdir()}, {"train.jsonl", "proofs.jsonl", "manifest.json"})
            self.assertEqual(manifest["splits"], ["train"])
            self.assertEqual(manifest["pending_gates"], GATES)
            self.assertTrue(manifest["decimal_replay_verified"])
            for flag in ("training_allowed", "training_performed", "human_quality_reviewed",
                         "tokenizer_verified", "decontamination_verified", "general_improvement_proven"):
                self.assertIs(manifest[flag], False)
            for name, digest in manifest["sha256"].items():
                self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), digest)
            rows = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
            self.assertTrue(all(set(row) == {"messages"} for row in rows))
            self.assertLess((out / "train.jsonl").stat().st_size, 1_048_576)
            with self.assertRaises(ValueError):
                build(out)

    def test_seed_validation_and_distinct_candidates(self):
        for bad in (-1, 2**63, True, "seed"):
            with self.subTest(seed=bad), self.assertRaises(ValueError):
                candidate(bad)
        self.assertNotEqual(self.rows, candidate(20261108)[0])


if __name__ == "__main__":
    unittest.main()
