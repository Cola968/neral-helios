"""Train-only E1 pair miner tests. No GPU, no private release answers exported."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.neral_v6_compositional import FAMILIES, make_case
from scripts.neral_e1_pair_miner import (
    assess_responses, build_pairs, _answer_value, _final_line,
)
from scripts.neral_e1_smoke import validate_pairs
from tests.test_neral_v6_suite_intake import make_suite


ADAPTER_SHA = "a" * 64
DECODING_SHA = "b" * 64
BAD = {
    "money_order": "999999,99 EUR",
    "two_leg_fuel": "999999 kg",
    "timezone_stop": "00:00 (Tag +999)",
    "inventory_flow": "999999 Stück",
    "mixture_percent": "999999 %",
}


def response(family, split, index, *, raw=None):
    case = make_case(family, split, index)
    content = BAD[family] if raw is None else raw
    return {
        "id": f"v6-{family}-{split}-{index:03}",
        "family": family,
        "split": split,
        "index": index,
        "prompt_sha256": hashlib.sha256(
            case["messages"][1]["content"].encode("utf8")).hexdigest(),
        "adapter_sha256": ADAPTER_SHA,
        "decoding_sha256": DECODING_SHA,
        "generation_source": "neral_v5_local",
        "model_generated": True,
        "response": content,
    }


def candidate_responses():
    return [response(family, split, idx)
            for split, per_family in (("train", 6), ("validation", 2))
            for family in FAMILIES for idx in range(per_family)]


def to_file(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                            for row in rows), encoding="utf8")
    return path


class E1VerifiedPairMiningTests(unittest.TestCase):
    def test_40_true_wrong_answers_convert_to_smoke_compatible_pairs(self):
        rows = candidate_responses()
        selected, report = assess_responses(rows, ADAPTER_SHA, DECODING_SHA)
        self.assertEqual(len(selected), 40)
        self.assertEqual(report["source_responses_checked"], 40)
        self.assertEqual(report["verified_wrong"], 40)
        self.assertEqual(report["semantically_correct"], 0)
        self.assertEqual(report["ungradeable"], 0)
        self.assertEqual(report["external_on_policy_proven"], False)
        self.assertEqual(report["external_independence_proven"], False)
        self.assertEqual(validate_pairs(selected)["train"], 30)
        self.assertEqual(validate_pairs(selected)["validation"], 10)
        self.assertEqual({p["family"] for p in selected}, set(FAMILIES))
        self.assertTrue(all(p["chosen_verified"] and not p["rejected_verified"] for p in selected))

    def test_semantically_equivalent_formats_are_not_false_failures(self):
        money = make_case("money_order","train",0)["messages"][2]["content"]
        # The generator's chosen response may contain CHECK+FINAL; only final is graded.
        correct_money = _final_line(money)
        self.assertIsNotNone(correct_money)
        alt_money = correct_money.replace(",", ".").replace(" EUR", " €")
        expected_fuel = _final_line(make_case("two_leg_fuel","train",0)["messages"][2]["content"])
        self.assertIsNotNone(expected_fuel)
        expected_mix = _final_line(make_case("mixture_percent","train",0)["messages"][2]["content"])
        self.assertIsNotNone(expected_mix)
        rows = [
            response("money_order", "train", 0, raw="FINAL_ANSWER: " + alt_money),
            response("two_leg_fuel", "train", 0, raw="FINAL: " + expected_fuel),
            response("mixture_percent", "train", 0, raw=expected_mix),
            response("inventory_flow", "train", 0, raw="I cannot produce a final value."),
        ]
        selected, report = assess_responses(rows, ADAPTER_SHA, DECODING_SHA)
        self.assertEqual(selected, [])
        self.assertEqual(report["semantically_correct"], 3)
        self.assertEqual(report["ungradeable"], 1)

    def test_malformed_duplicate_spoofed_and_holdout_rows_denied(self):
        good = candidate_responses()
        for changed in (
            dict(good[0], split="holdout"),
            dict(good[0], model_generated=False),
            dict(good[0], adapter_sha256="c" * 64),
            dict(good[0], prompt_sha256="d" * 64),
            dict(good[0], index=-1),
            dict(good[0], id="evil-other-id"),
            dict(good[0], generation_source="synthetic_wrong"),
            dict(good[0], response=""),
        ):
            with self.assertRaises(ValueError):
                assess_responses([changed], ADAPTER_SHA, DECODING_SHA)
        with self.assertRaises(ValueError):
            assess_responses([good[0], good[0]], ADAPTER_SHA, DECODING_SHA)

    def test_no_selection_if_only_format_missing_or_safety_suspicious(self):
        for bad in ("Antwort: 32", "Die Rechnung ergibt 10\nIch weiß nicht",
                    "FINAL: 1\nFINAL: 2", "123\x00",
                    "FINAL_ANSWER: 1\nweiterer Text"):
            selected, report = assess_responses(
                [response("inventory_flow","train",0,raw=bad)],
                ADAPTER_SHA, DECODING_SHA)
            self.assertEqual(selected, [])
            self.assertEqual(report["ungradeable"], 1)

    def test_file_build_only_training_splits_and_decon_against_private_suites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            responses=to_file(root/"onpolicy.jsonl",candidate_responses())
            dev=to_file(root/"dev.jsonl",make_suite("dev"))
            release=to_file(root/"release.jsonl",make_suite("release"))
            out=root/"pair-candidate"
            report=build_pairs(responses,dev,release,out,
                               adapter_sha256=ADAPTER_SHA,decoding_sha256=DECODING_SHA)
            self.assertTrue((out/"pairs.jsonl").exists())
            self.assertEqual(report["train_pairs"],30)
            self.assertEqual(report["validation_pairs"],10)
            self.assertFalse(report["training_allowed"])
            self.assertFalse(report["human_verified"])
            self.assertFalse(report["release_validated"])
            self.assertEqual(report["release_cases_used_for_training"],0)
            self.assertEqual(len(report["pairs_sha256"]),64)
            with self.assertRaises(ValueError):
                build_pairs(responses,dev,release,out,adapter_sha256=ADAPTER_SHA,
                            decoding_sha256=DECODING_SHA)
            modified=make_suite("release")
            modified[0]["prompt"] = make_case(FAMILIES[0],"train",0)["messages"][1]["content"]
            to_file(release,modified)
            with self.assertRaises(ValueError):
                build_pairs(responses,dev,release,root/"other",
                            adapter_sha256=ADAPTER_SHA,decoding_sha256=DECODING_SHA)
            self.assertFalse((root/"other").exists())


if __name__ == "__main__":
    unittest.main()
