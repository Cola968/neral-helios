"""Synthetic, CPU-only regression cases; no frozen suites or model data."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


MODULE = "frontier350.chat350_dev_generation_diagnostics"
ROOT = Path(__file__).resolve().parents[1]


def report():
    # Hand-calculated eval_chat350.py schema, NOT a proposed future schema.
    # EOS=1 is known only to this fixture; it is deliberately absent on disk.
    return {
        "checkpoint_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "case_sha256": "c" * 64,
        "evaluation_contract": "segment_bpe_mask_specials_v1",
        "summary": {
            "cases": 3, "eos_fraction": 2 / 3,
            "mean_repeated_trigram_fraction": 0.25,
            "mean_distinct2_fraction": 11 / 15,
            "mean_generated_tokens_before_eos": 8 / 3,
            "utf8_fraction": 1.0, "exact_instruction_pct": 100.0,
        },
        "rows": [
            {"id": "synthetic-empty", "category": "de", "text": " \n\t", "generated_ids": [1]},
            {"id": "synthetic-short", "category": "de", "text": "Grüße!", "generated_ids": [7, 8, 1], "expected_exact": "Grüße!"},
            {"id": "synthetic-loop", "category": "math", "text": "eins zwei drei vier", "generated_ids": [9, 9, 9, 9, 9, 9]},
        ],
    }


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec(MODULE), "CPU diagnostics module must exist")
        self.api = __import__(MODULE, fromlist=["diagnose_report"])

    def test_unicode_whitespace_normalized_echo_preserves_semantic_differences(self):
        flags = self.api.measure_prompt_echo("User: Grüße aus Köln?\nAssistant:", " Gru\u0308ße\u00a0aus\nKo\u0308ln? ")
        self.assertFalse(flags["exact_echo"])
        self.assertTrue(flags["normalized_echo"])
        self.assertTrue(flags["prefix_echo"])
        for answer in ("Grüße aus Koln?", "Grüsse aus Köln?", "grüße aus Köln?", "Nein, Grüße aus Köln!", "Grüße aus Köln?"):
            value = self.api.measure_prompt_echo("User: Grüße aus Köln!\nAssistant:", answer)
            self.assertFalse(value["normalized_echo"])

    def test_exact_prefix_and_word_boundaries_are_distinct(self):
        prefix = "User: Ja\nAssistant:"
        self.assertTrue(self.api.measure_prompt_echo(prefix, "Ja")["exact_echo"])
        flags = self.api.measure_prompt_echo(prefix, "Ja, mit Erklärung.")
        self.assertTrue(flags["prefix_echo"])
        self.assertFalse(flags["normalized_echo"])
        self.assertFalse(self.api.measure_prompt_echo(prefix, "Jahr")["prefix_echo"])

    def test_only_latest_user_turn_and_no_stripped_assistant_header(self):
        prefix = "User: Alt\nAssistant: Antwort\nUser: Neu\nAssistant:"
        self.assertFalse(self.api.measure_prompt_echo(prefix, "Alt")["exact_echo"])
        self.assertTrue(self.api.measure_prompt_echo(prefix, "Neu")["exact_echo"])
        self.assertFalse(self.api.measure_prompt_echo(prefix, "Assistant: Neu")["normalized_echo"])
        self.assertTrue(self.api.measure_prompt_echo(prefix, "Neu\nUser: erfunden\nAssistant: ja")["normalized_echo"])
        self.assertFalse(self.api.measure_prompt_echo(prefix, "Antwort\nUser: Neu")["normalized_echo"])

    def test_malformed_or_empty_prefix_cannot_create_echo(self):
        for prefix in ("", "User: \nAssistant:", "User: Neu", "Assistant: Neu"):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                self.api.measure_prompt_echo(prefix, "")

    def test_counts_sums_and_categories_use_stored_text_and_raw_ids(self):
        value = self.api.diagnose_report(report(), min_words=3)
        total = value["overall"]
        self.assertEqual(total["cases"], 3)
        self.assertEqual(total["word_count"], {"sum": 5, "mean": 5 / 3, "basis": "stored_decoded_text"})
        self.assertEqual(total["raw_token_count"]["sum"], 10)
        self.assertEqual(total["empty"]["numerator"], 1)
        self.assertEqual(total["very_short"]["numerator"], 2)
        self.assertEqual(total["very_short"]["denominator"], 3)
        self.assertEqual(total["very_short"]["rate"], 2 / 3)
        self.assertAlmostEqual(total["repeated_trigram"]["sum"], 0.75)
        self.assertAlmostEqual(total["repeated_trigram"]["mean"], 0.25)
        self.assertEqual(value["by_category"]["de"]["cases"], 2)
        self.assertEqual(value["by_category"]["math"]["very_short"]["numerator"], 0)

    def test_legacy_metadata_is_missing_never_inferred_from_summary_or_text(self):
        value = self.api.diagnose_report(report())
        self.assertEqual(value["status"], "incomplete_metadata")
        for metric in ("exact_echo", "normalized_echo", "prefix_echo", "eos", "max_new_without_eos"):
            row = value["overall"][metric]
            self.assertEqual(row["status"], "not_measured")
            self.assertIsNone(row["rate"])
            self.assertIsNone(row["numerator"])
            self.assertEqual(row["denominator"], 0)
        self.assertEqual(value["reported_eos"]["status"], "reported_unverified")
        self.assertEqual(value["reported_eos"]["rate"], 2 / 3)
        self.assertIsNone(value["reported_eos"]["numerator"])
        self.assertIn("eos_token_id", value["missing_metadata"])
        self.assertIn("generation_settings", value["missing_metadata"])

    def test_no_prompt_response_expected_answer_or_token_arrays_in_summary(self):
        value = self.api.diagnose_report(report())
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        for private in ("Grüße", "eins zwei", "generated_ids", "expected_exact", '"text"'):
            self.assertNotIn(private, encoded)

    def test_repetition_must_not_strip_a_guessed_eos_or_mutate_input(self):
        source = report()
        before = copy.deepcopy(source)
        result = self.api.diagnose_report(source)
        self.assertEqual(source, before)
        self.assertEqual(result["overall"]["raw_token_count"]["sum"], 10)
        self.assertEqual(result["overall"]["repeated_trigram"]["basis"], "raw_generated_token_ids_including_stop")

    def test_word_counts_normalize_combining_umlauts_and_keep_short_empty_separate(self):
        source = report()
        source["rows"][0]["text"] = "..."
        source["rows"][2]["text"] = "Gru\u0308ße aus Köln"
        value = self.api.diagnose_report(source, min_words=3)
        self.assertEqual(value["overall"]["word_count"]["sum"], 4)
        self.assertEqual(value["overall"]["empty"]["numerator"], 0)
        self.assertEqual(value["overall"]["very_short"]["numerator"], 2)

    def test_minimum_word_threshold_must_be_positive_integer(self):
        for bad in (0, -1, True, 1.5, "3"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.api.diagnose_report(report(), min_words=bad)

    def test_missing_duplicate_and_extra_or_missing_case_ids_fail_closed(self):
        for field, bad in (("id", None), ("id", ""), ("id", True), ("id", []), ("category", ""), ("text", None), ("generated_ids", None)):
            source = report()
            source["rows"][0][field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                self.api.diagnose_report(source)
        source = report()
        source["rows"][1]["id"] = source["rows"][0]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.api.diagnose_report(source)
        for rows in ([], report()["rows"][:-1], None):
            source = report()
            source["rows"] = rows
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.api.diagnose_report(source)

    def test_generated_ids_are_actual_nonnegative_integers(self):
        for ids in ([True], ["1"], [-1], [1.5], "123"):
            source = report()
            source["rows"][0]["generated_ids"] = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.api.diagnose_report(source)

    def test_nonfinite_numbers_anywhere_are_rejected(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            for target in ("summary", "extra", "row"):
                source = report()
                if target == "summary":
                    source["summary"]["eos_fraction"] = bad
                elif target == "extra":
                    source["metadata"] = {"nested": [bad]}
                else:
                    source["rows"][0]["extra"] = bad
                with self.subTest(target=target, bad=bad), self.assertRaises(ValueError):
                    self.api.diagnose_report(source)

    def test_huge_summary_integer_is_rejected_as_invalid_not_an_overflow(self):
        source = report()
        source["summary"]["eos_fraction"] = 10 ** 1000
        with self.assertRaises(ValueError):
            self.api.diagnose_report(source)

    def test_missing_required_row_fields_cannot_be_defaulted(self):
        for key in ("id", "category", "text", "generated_ids"):
            source = report()
            del source["rows"][0][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.api.diagnose_report(source)

    def test_unknown_eos_keeps_raw_repetition_instead_of_source_content_average(self):
        # Raw [4,4,4,4] has one repeated trigram out of two. With EOS=4
        # the content would be empty. No EOS is declared in the input schema.
        source = report()
        source["rows"] = [{"id": "repeat", "category": "de", "text": "", "generated_ids": [4, 4, 4, 4]}]
        source["summary"] = {"cases": 1, "eos_fraction": 0,
                             "mean_repeated_trigram_fraction": 0.5,
                             "mean_distinct2_fraction": 1 / 3,
                             "mean_generated_tokens_before_eos": 4,
                             "utf8_fraction": 1, "exact_instruction_pct": None}
        value = self.api.diagnose_report(source)
        self.assertEqual(value["overall"]["repeated_trigram"]["mean"], 0.5)
        self.assertEqual(value["overall"]["eos"]["status"], "not_measured")

    def test_tokenless_context_stop_is_unknown_instead_of_eos_or_limit(self):
        source = report()
        source["rows"][0]["text"] = ""
        source["rows"][0]["generated_ids"] = []
        source["summary"]["eos_fraction"] = 1 / 3
        value = self.api.diagnose_report(source)
        self.assertEqual(value["overall"]["empty"]["numerator"], 1)
        self.assertIsNone(value["overall"]["max_new_without_eos"]["rate"])

    def test_checked_hashes_are_report_claims_not_a_verification_of_source_files(self):
        value = self.api.diagnose_report(report())
        self.assertEqual(value["provenance_verification"], "report_claims_only_no_source_artifacts_opened")

    def test_invalid_top_level_and_invalid_rows_are_rejected(self):
        for bad in (None, [], {}, {**report(), "rows": [None]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.api.diagnose_report(bad)

    def test_summary_missing_inconsistent_counts_and_rates_are_rejected(self):
        for key, bad in (("cases", 2), ("cases", True), ("eos_fraction", 0.5), ("eos_fraction", True), ("utf8_fraction", 0.5), ("mean_generated_tokens_before_eos", 10), ("exact_instruction_pct", 50), ("mean_repeated_trigram_fraction", -1), ("mean_distinct2_fraction", 1.1)):
            source = report()
            source["summary"][key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                self.api.diagnose_report(source)
        for key in report()["summary"]:
            source = report()
            del source["summary"][key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.api.diagnose_report(source)

    def test_unknown_contract_missing_hashes_and_forged_future_metadata_rejected(self):
        for key, bad in (("evaluation_contract", None), ("evaluation_contract", "segment_bpe_mask_specials_v2"), ("tokenizer_sha256", "bad"), ("case_sha256", "0" * 64), ("checkpoint_sha256", None)):
            source = report()
            source[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.api.diagnose_report(source)
        source = report()
        source["schema_version"] = 2
        with self.assertRaises(ValueError):
            self.api.diagnose_report(source)

    def test_untrimmed_hallucinated_user_boundary_is_inconsistent_with_contract(self):
        source = report()
        source["rows"][0]["text"] = "Hallo\nUser: neu\nAssistant: erfunden"
        with self.assertRaises(ValueError):
            self.api.diagnose_report(source)

    def test_comparison_checks_all_models_ids_categories_and_hashes_before_averages(self):
        base = report()
        for mutation in ("id", "category", "tokenizer_sha256", "case_sha256", "evaluation_contract"):
            candidate = report()
            if mutation == "id":
                candidate["rows"][0]["id"] = "other"
            elif mutation == "category":
                candidate["rows"][0]["category"] = "other"
            elif mutation == "evaluation_contract":
                candidate[mutation] = "future"
            else:
                candidate[mutation] = "d" * 64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.api.compare_reports([base, copy.deepcopy(base), candidate])

    def test_matching_legacy_reports_still_cannot_prove_valid_generation_comparison(self):
        candidate = report()
        candidate["checkpoint_sha256"] = "d" * 64
        value = self.api.compare_reports([report(), candidate])
        self.assertEqual(value["status"], "not_comparable")
        self.assertFalse(value["valid_comparison"])
        self.assertFalse(value["automatic_promotion"])
        self.assertEqual(len(value["reports"]), 2)
        self.assertEqual(value["checked_identity"]["category_counts"], {"de": 2, "math": 1})
        self.assertIn("generation_settings", value["missing_metadata"])

    def test_comparison_is_order_independent_within_reports_and_does_not_mutate(self):
        first, second = report(), report()
        second["rows"].reverse()
        before = copy.deepcopy(second)
        value = self.api.compare_reports([first, second])
        self.assertEqual(second, before)
        self.assertEqual(value["reports"][0], value["reports"][1])

    def test_comparison_requires_at_least_two_nonempty_reports(self):
        for reports in ([], [report()], None):
            with self.subTest(reports=reports), self.assertRaises(ValueError):
                self.api.compare_reports(reports)


class DiagnosticsCliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-m", MODULE, *map(str, args)], cwd=ROOT, capture_output=True, text=True, encoding="utf8")

    def test_legacy_cli_writes_deterministic_json_but_exits_nonzero_for_missing_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            source, out = Path(folder) / "input.json", Path(folder) / "out.json"
            source.write_text(json.dumps(report(), ensure_ascii=False), encoding="utf8")
            first = self.run_cli("--report", source, "--out", out)
            self.assertEqual(first.returncode, 2, first.stderr)
            self.assertTrue(out.exists(), first.stderr)
            encoded = out.read_bytes()
            value = json.loads(encoded)
            self.assertEqual(value["status"], "incomplete_metadata")
            self.assertNotIn("Grüße", encoded.decode("utf8"))
            second = self.run_cli("--report", source, "--out", out)
            self.assertEqual(second.returncode, 2)
            self.assertEqual(out.read_bytes(), encoded)

    def test_cli_multiple_reports_never_accepts_legacy_as_a_valid_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            source, out = Path(folder) / "input.json", Path(folder) / "out.json"
            source.write_text(json.dumps(report()), encoding="utf8")
            result = self.run_cli("--report", source, "--report", source, "--out", out)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(json.loads(out.read_text())["valid_comparison"])

    def test_cli_invalid_json_never_leaks_payload_and_replaces_stale_success(self):
        with tempfile.TemporaryDirectory() as folder:
            source, out = Path(folder) / "input.json", Path(folder) / "out.json"
            for payload in ('{"secret prompt": NaN}', '{"secret prompt": Infinity}', '{"rows": [], "rows": []}', 'secret prompt'):
                source.write_text(payload, encoding="utf8")
                out.write_text('{"valid_comparison":true}', encoding="utf8")
                result = self.run_cli("--report", source, "--out", out)
                self.assertEqual(result.returncode, 2, result.stderr)
                value = json.loads(out.read_text())
                self.assertEqual(value["status"], "invalid_report")
                self.assertFalse(value["valid_comparison"])
                self.assertNotIn("secret prompt", result.stdout + result.stderr + out.read_text())

    def test_cli_never_overwrites_input_report(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.json"
            source.write_text(json.dumps(report()), encoding="utf8")
            before = source.read_bytes()
            result = self.run_cli("--report", source, "--out", source)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
