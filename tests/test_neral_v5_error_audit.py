"""Audited local-only NERAL v5 failure diagnosis; synthetic fixture (no GPU)."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_v5_error_audit import audit, scan_saved_rows
from scripts.neral_v5_style_eval import CASES, case_hash, BASE_REVISION, grade


class V5ErrorAuditTests(unittest.TestCase):
    def fixture(self, root: Path, *, mutate=None) -> None:
        for name in ("base", "neral_v5"):
            folder = root / name
            folder.mkdir()
            scores = []
            for case in CASES:
                raw = case["answer"]
                if case["id"] == "affine-00":
                    raw = "Antwort: " + raw if name == "base" else raw
                if case["id"] == "money-00":
                    raw = "999,99 EUR" if name == "neral_v5" else raw
                if case["id"] == "fuel-00":
                    raw = "placeholder " * 20 + "x" * 32 if name == "base" else raw
                correct, formatted = grade(case, raw)
                score = {
                    "id": case["id"], "correct": bool(correct),
                    "format_valid": bool(formatted),
                    "answer_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "local_raw": raw[:250],
                }
                if mutate:
                    mutate(name, score)
                scores.append(score)
            (folder / "local.json").write_text(json.dumps({
                "variant": name, "adapter_sha256": "f" * 64,
                "base_revision": BASE_REVISION,
                "case_hash": case_hash(), "scores": scores,
            }), encoding="utf8")

    def test_exact_scoring_and_recoverable_last_line_without_uploading_text(self):
        with tempfile.TemporaryDirectory() as d:
            self.fixture(Path(d))
            a = audit(Path(d), adapter_sha="f" * 64)
            self.assertEqual(a["total"], 320)
            self.assertEqual(a["base"]["correct"], 318)
            self.assertEqual(a["neral_v5"]["correct"], 319)
            self.assertEqual(a["base"]["format_invalid"], 2)
            self.assertEqual(a["neral_v5"]["format_valid_wrong"], 1)
            self.assertEqual(a["base"]["recoverable_last_line"], 1)
            self.assertEqual(a["base"]["truncated_raw"], 1)
            self.assertEqual(a["paired"]["improvements"], 2)
            self.assertEqual(a["paired"]["regressions"], 1)
            self.assertFalse(a["independent_external_benchmark"])
            self.assertFalse(a["general_improvement_proven"])
            output = json.dumps(a, ensure_ascii=False)
            for sensitive in ("local_raw", "Antwort: ", "999,99 EUR", "placeholder",
                              "prompt", "completion", "answer_sha256"):
                self.assertNotIn(sensitive, output)

    def test_rejects_modified_model_origin_and_score(self):
        for tamper in ("wrong_base", "wrong_sha", "tampered_score", "injected_key"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                self.fixture(root)
                p = root / "base" / "local.json"
                data = json.loads(p.read_text(encoding="utf8"))
                if tamper == "wrong_base":
                    data["base_revision"] = "0" * 40
                elif tamper == "wrong_sha":
                    data["adapter_sha256"] = "0" * 64
                elif tamper == "tampered_score":
                    data["scores"][0]["correct"] = False
                elif tamper == "injected_key":
                    data["scores"][0]["training_secret"] = "SECRET"
                p.write_text(json.dumps(data), encoding="utf8")
                with self.assertRaises(ValueError):
                    audit(root, adapter_sha="f" * 64)

    def test_only_exact_complete_ids_and_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            p = root / "base" / "local.json"
            data = json.loads(p.read_text(encoding="utf8"))
            data["scores"][0]["id"] = "unexpected-00"
            p.write_text(json.dumps(data), encoding="utf8")
            with self.assertRaises(ValueError):
                audit(root, adapter_sha="f" * 64)

    def test_no_raw_export_and_read_only_workflow(self):
        workflow = Path(".github/workflows/neral-v5-local-error-audit.yml").read_text(encoding="utf8")
        self.assertIn("runs-on: [self-hosted, windows, x64, neral-gpu]", workflow)
        self.assertIn("neral-qwen7b-r32-gpu-exclusive", workflow)
        self.assertIn("42383240688", workflow)
        self.assertIn("42378021782", workflow)
        self.assertIn("NERAL_V5_LOCAL_ERROR_AUDIT", workflow)
        self.assertIn("actions/upload-artifact", workflow)
        self.assertIn("push:", workflow)
        self.assertNotIn("git push", workflow)
        self.assertNotIn("--run-local-gpu", workflow)
        self.assertNotIn("trainer.train", workflow)
        self.assertNotIn("model.generate", workflow)
        self.assertNotIn("adapter_model.safetensors", workflow)
        self.assertIn("if-no-files-found: error", workflow)


if __name__ == "__main__":
    unittest.main()
