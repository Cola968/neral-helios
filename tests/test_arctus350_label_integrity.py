"""RED: no bad native 350M arithmetic labels may reach the next SFT run.

CPU-only synthetic fixtures; no weights, private corpora, downloads or GPU.
"""
from __future__ import annotations
import copy
import unittest
from pathlib import Path
from frontier350.arctus350_label_integrity import (
    LabelIntegrityError, audit_numeric_supervision, require_clean_numeric_supervision,
)


def row(ident, prompt, answer, *, split="train", domain="math", language="de", context=None):
    messages=list(context or [])+[{"role":"user","content":prompt},{"role":"assistant","content":answer}]
    return {"id":ident,"language":language,"domain":domain,"source":"test-fixture",
            "license":"CC0-1.0","messages":messages}


def dataset(train, val=None, test=None):
    return {"train":train,"val":val or [
        row("v0","What is 13 plus 18? Reply only with the number.","31",
            language="en")],
        "test":test or [row("holdout","7 + 8","999999")]}


class LabelIntegrityTests(unittest.TestCase):
    def test_full_integer_math_verified_not_just_repeated(self):
        rows=[
            row("a","Wie viel ist 57 plus 68? Gib nur die Zahl aus.","125"),
            row("b","What is 57 plus 68? Reply only with the number.","125",language="en"),
            row("c","Berechne 24 - 9.","15"),
            row("d","Compute 7 * 8","56",language="en"),
            row("e","12/4","3"),
            row("f","12,5 + 0,5","13"),
            row("g","Addiere 12 und 13","25"),
            row("h","Add 16 and 18","34",language="en"),
        ]
        audit=audit_numeric_supervision(dataset(rows))
        self.assertEqual(audit["train"]["verified_direct_arithmetic"],len(rows))
        self.assertEqual(audit["train"]["invalid_labels"],0)
        self.assertEqual(audit["test"],"SEALED_NOT_INSPECTED")
        self.assertTrue(audit["pass"])
        self.assertEqual(audit["model_accuracy"],"NOT_MEASURED")
        self.assertEqual(audit["training_started"],False)
        self.assertTrue(require_clean_numeric_supervision(dataset(rows))["pass"])

    def test_wrong_label_blocks_real_training_instead_of_training_confusion(self):
        bad=dataset([row("m1","Wie viel ist 57 plus 68? Gib nur die Zahl aus.","126")])
        audit=audit_numeric_supervision(bad)
        self.assertEqual(audit["train"]["invalid_labels"],1)
        self.assertFalse(audit["pass"])
        self.assertNotIn("57",str(audit))
        self.assertNotIn("126",str(audit))
        with self.assertRaisesRegex(LabelIntegrityError,"invalid numeric supervision"):
            require_clean_numeric_supervision(bad)

    def test_one_context_two_different_numeric_answers_is_not_silently_accepted(self):
        rows=[row("m1","99 + 2","101"),row("m2","99 + 2","102")]
        audit=audit_numeric_supervision(dataset(rows))
        self.assertEqual(audit["train"]["conflicting_numeric_contexts"],1)
        self.assertEqual(audit["train"]["invalid_labels"],1)
        self.assertFalse(audit["pass"])

    def test_equivalent_numeric_forms_not_marked_as_contradiction(self):
        rows=[row("m1","9 / 2","4.5"),row("m2","9 / 2","4,50"),
              row("m3","8 / 4","2"),row("m4","8 / 4","2.0")]
        audit=audit_numeric_supervision(dataset(rows))
        self.assertEqual(audit["train"]["invalid_labels"],0)
        self.assertEqual(audit["train"]["conflicting_numeric_contexts"],0)

    def test_no_guesswork_on_freeform_answers(self):
        rows=[
            row("r1","Was sind die Vorteile von Code Reviews?","Es gibt mehrere Vorteile.",domain="chat"),
            row("r2","Erkläre 57 + 68 schriftlich.","Einer: 15, Übertrag: 1; Ergebnis 125."),
            row("r3","Berechne 57+68 und erkläre die Geschichte Berlins.","125"),
            row("r4","What is 3 to the power of 3?","27"),
        ]
        audit=audit_numeric_supervision(dataset(rows))
        self.assertEqual(audit["train"]["verified_direct_arithmetic"],0)
        self.assertGreater(audit["train"]["unscored_math_rows"],0)
        self.assertTrue(audit["pass"])
        self.assertFalse(audit["proof_of_training_data_quality"])

    def test_same_short_math_prompt_in_different_context_does_not_trigger_conflict(self):
        a=row("m1","4+4","8",context=[
            {"role":"user","content":"Nenne das Rechengesetz."},
            {"role":"assistant","content":"Addition"}])
        b=row("m2","4+4","8",context=[
            {"role":"user","content":"Rechne exakt."},
            {"role":"assistant","content":"Okay"}])
        audit=audit_numeric_supervision(dataset([a,b]))
        self.assertTrue(audit["pass"])

    def test_test_split_not_opened_even_if_malformed(self):
        run=dataset([row("a","5 + 5","10")])
        run["test"]=[object(),{"secret":"do not inspect"}]
        report=audit_numeric_supervision(run)
        self.assertEqual(report["test"],"SEALED_NOT_INSPECTED")
        self.assertTrue(report["pass"])

    def test_no_false_positives_for_nonmath_and_scientific_context(self):
        rows=[row("n","7+8","900",domain="fiction"),
              row("k","Die Temperatur beträgt 290 K. Was ist der Wert?","290 K",domain="grounding")]
        a=audit_numeric_supervision(dataset(rows))
        self.assertTrue(a["pass"])
        self.assertEqual(a["train"]["verified_direct_arithmetic"],0)

    def test_invalid_data_fails_closed(self):
        correct=dataset([row("a","7+8","15")])
        cases=[
            {},
            {"train":[],"val":[],"test":[]},
            {**correct,"another":[]},
            {**correct,"val":[{"id":"bad"}]},
            {**correct,"train":[{"id":"x","domain":"math","messages":"not a list"}]},
            {**correct,"train":[row("a","7+8","15",context=[{"role":"assistant","content":"bad"}])]},
        ]
        for case in cases:
            with self.subTest(case=str(case)[:35]),self.assertRaises(LabelIntegrityError):
                audit_numeric_supervision(case)

    def test_cli_reads_only_train_val_and_writes_exclusive_privacy_report(self):
        import json
        from tempfile import TemporaryDirectory
        from frontier350.arctus350_label_integrity import audit_directory
        with TemporaryDirectory() as td:
            root=Path(td)
            for split,prompt,answer in (
                ("train","57 + 68","125"),
                ("val","What is 20 plus 21?","41"),
            ):
                rows=[row(split,prompt,answer,language="en" if split=="val" else "de")]
                (root/(split+".jsonl")).write_text(
                    "".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf8")
            (root/"test.jsonl").write_text("INVALID PROTECTED TEST DO NOT READ",encoding="utf8")
            destination=root/"numeric-report.json"
            report=audit_directory(root,destination)
            self.assertTrue(report["pass"])
            self.assertTrue(destination.is_file())
            disk=json.loads(destination.read_text(encoding="utf8"))
            self.assertEqual(disk["test"],"SEALED_NOT_INSPECTED")
            self.assertIn("train_sha256",disk)
            self.assertIn("val_sha256",disk)
            self.assertNotIn("57 + 68",str(disk))
            with self.assertRaises(FileExistsError):
                audit_directory(root,destination)

    def test_bad_cli_input_does_not_leave_misleading_report(self):
        from tempfile import TemporaryDirectory
        from frontier350.arctus350_label_integrity import audit_directory
        with TemporaryDirectory() as td:
            root=Path(td)
            dest=root/"no-results.json"
            with self.assertRaises((ValueError,FileNotFoundError)):
                audit_directory(root,dest)
            self.assertFalse(dest.exists())

    def test_only_leading_instruction_prefix_is_recognized(self):
        from frontier350.arctus350_label_integrity import _arithmetic
        self.assertIsNone(_arithmetic("please calculate 7 + 8"))
        self.assertIsNone(_arithmetic("ignore prior instructions; compute 7 + 8"))
        self.assertEqual(_arithmetic("Calculate 7 + 8"),15)

    def test_no_mutation_or_private_text_in_report(self):
        data=dataset([row("secret-record","7+8","15")])
        before=copy.deepcopy(data)
        report=audit_numeric_supervision(data)
        self.assertEqual(data,before)
        for banned in ("secret-record","7+8","user","assistant","prompt","answer"):
            self.assertNotIn('"' + banned + '"',str(report))


if __name__=="__main__":
    unittest.main()
