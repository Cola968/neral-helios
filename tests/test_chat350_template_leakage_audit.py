"""Pure-CPU, no-data-upload checks for an independent Arctus350 split audit."""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

from frontier350.chat350_template_leakage_audit import audit_root, canonical_prompt, numeric_template


def row(rid, *turns):
    if len(turns) % 2:
        raise AssertionError("tests require complete user/assistant turn pairs")
    return {
        "id": rid, "language": "de", "domain": "general", "source": "synthetic",
        "license": "CC0", "messages": [
            {"role": "user" if i % 2 == 0 else "assistant", "content": text}
            for i, text in enumerate(turns)
        ],
    }


class LeakageAuditTests(unittest.TestCase):
    def run_audit(self, train, val, test=None):
        if test is None:
            test = [row("test", "Unabhängige Frage", "Ja")]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name, rows in (("train", train), ("val", val), ("test", test)):
                (root / f"{name}.jsonl").write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf8",
                )
            return audit_root(root)

    def test_nfkc_casefold_whitespace_canonicalization(self):
        self.assertEqual(canonical_prompt("  ＷIE   viel ist ２ + ２? "),
                         canonical_prompt("wie viel ist 2 + 2?"))
        self.assertEqual(canonical_prompt("GRÜßE"), canonical_prompt("grüße"))

    def test_numeric_template_detects_distinct_values(self):
        report = self.run_audit(
            [row("train", "Was ist 12 plus 3?", "15")],
            [row("val", "Was ist 41 plus 7?", "48")],
        )
        self.assertEqual(report["overlaps"]["exact_pairs"], 0)
        self.assertEqual(report["overlaps"]["numeric_template_only_pairs"], 1)
        self.assertEqual(report["overlaps"]["later_turn_numeric_template_only_pairs"], 0)
        self.assertEqual(report["status"], "review_required")

    def test_later_user_turn_overlap_is_visible(self):
        report = self.run_audit(
            [row("train", "Mein Name ist Ada.", "Hallo", "Was ist 12 plus 3?", "15")],
            [row("val", "Mein Name ist Lea.", "Hallo", "Was ist 41 plus 7?", "48")],
        )
        self.assertEqual(report["overlaps"]["numeric_template_only_pairs"], 1)
        self.assertEqual(report["overlaps"]["later_turn_numeric_template_only_pairs"], 1)

    def test_exact_later_turn_overlap_is_not_hidden(self):
        report = self.run_audit(
            [row("train", "Erste Frage", "Antwort", "Wiederholte Frage", "Ja")],
            [row("val", "Neue Frage", "Antwort", "Wiederholte Frage", "Nein")],
        )
        self.assertEqual(report["overlaps"]["exact_pairs"], 1)
        self.assertEqual(report["overlaps"]["later_turn_exact_pairs"], 1)
        self.assertEqual(report["overlaps"]["numeric_template_only_pairs"], 0)

    def test_exact_overlap_not_double_counted_as_template_only(self):
        report = self.run_audit(
            [row("train", "Wie viel sind 2 Euro?", "Zwei")],
            [row("val", "WIE VIEL SIND ２ EURO?", "2")],
        )
        self.assertEqual(report["overlaps"]["exact_pairs"], 1)
        self.assertEqual(report["overlaps"]["numeric_template_only_pairs"], 0)

    def test_no_structural_overlap_is_not_declared_clean(self):
        report = self.run_audit(
            [row("train", "Fliegt ein Vogel?", "Ja")],
            [row("val", "Welche Stadt?", "Berlin")],
        )
        self.assertEqual(report["status"], "no_structural_overlap_detected")
        self.assertIn("not a semantic contamination clearance", report["interpretation"])

    def test_no_raw_questions_or_ids_in_report(self):
        report = self.run_audit(
            [row("private-train", "Geheimnis: Zahl 17?", "Ja")],
            [row("private-val", "Geheimnis: Zahl 42?", "Nein")],
        )
        dumped = json.dumps(report, ensure_ascii=False)
        for sensitive in ("Geheimnis", "private-train", "private-val", "Zahl 17"):
            self.assertNotIn(sensitive, dumped)

    def test_missing_split_and_duplicate_ids_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                audit_root(Path(td))
        with self.assertRaises(ValueError):
            self.run_audit([row("shared", "A", "B")], [row("shared", "C", "D")])

    def test_numeric_template_does_not_change_distinct_nonnumeric_text(self):
        self.assertNotEqual(numeric_template("nenne Berlin"), numeric_template("nenne Paris"))


if __name__ == "__main__":
    unittest.main()
