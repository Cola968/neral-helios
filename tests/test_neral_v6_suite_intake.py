"""Fail-closed 600/600 NERAL v6 dev/release suite intake CPU contracts."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_v6_suite_intake import (
    audit_suite, audit_pair, audit_files, read_training_prompts,
)

DOMAINS = ("math", "code", "reasoning")


def make_suite(label):
    rows = []
    for domain in DOMAINS:
        for index in range(200):
            kind = {"math": "numeric", "code": "python", "reasoning": "boolean"}[domain]
            rows.append({
                "id": f"{label}-{domain}-{index:03}",
                "domain": domain,
                "family": f"{domain}-method-{index % 10}",
                "source_id": f"{label}-original-{domain}-{index:03}",
                "kind": kind,
                "prompt": f"Task {label} {domain} {index} with private unique input token K{label}_{domain}_{index:03}.",
                "answer": {"numeric": str(index + 11),
                           "python": "def solution(x): return x+1",
                           "boolean": "1"}[kind],
                "tests_sha256": "a" * 64 if kind == "python" else None,
            })
    return rows


class V6SuiteIntakeTests(unittest.TestCase):
    def test_audit_two_completely_separate_600_case_suites(self):
        dev, release = make_suite("dev"), make_suite("release")
        report = audit_pair(dev, release,
                            ["Independent training entry not matching the private questions."])
        self.assertEqual(report["counts"], {
            "dev": {"math": 200, "code": 200, "reasoning": 200},
            "release": {"math": 200, "code": 200, "reasoning": 200},
        })
        self.assertEqual(report["normalized_10gram_training_overlap"], 0)
        self.assertFalse(report["semantically_unseen_proven"])
        self.assertFalse(report["human_independence_verified"])
        self.assertFalse(report["grader_correctness_verified"])
        self.assertFalse(report["release_validated"])

    def test_wrong_sizes_domains_duplicate_ids_and_lack_of_families_fail(self):
        suite = make_suite("dev")
        bad = [
            suite[:-1],
            [*suite[:-1], suite[0]],
            [dict(x, family="only-one") if x["domain"] == "math" else x for x in suite],
            [dict(suite[0], domain="unknown"), *suite[1:]],
            [dict(suite[0], tests_sha256="b" * 64), *suite[1:]],
        ]
        for rows in bad:
            with self.subTest(variant=len(rows), example=rows[0]["id"]):
                with self.assertRaises(ValueError):
                    audit_suite(rows)

    def test_code_needs_nonempty_hidden_unit_test_fingerprint(self):
        suite = make_suite("release")
        idx = next(i for i, x in enumerate(suite) if x["kind"] == "python")
        suite[idx] = dict(suite[idx], tests_sha256=None)
        with self.assertRaises(ValueError):
            audit_suite(suite)

    def test_overlap_between_dev_release_and_training_fail_closed(self):
        dev, release = make_suite("dev"), make_suite("release")
        collision = [dict(x) for x in release]
        collision[0]["prompt"] = dev[0]["prompt"]
        with self.assertRaises(ValueError):
            audit_pair(dev, collision, ["Truly separate training task here."])
        collision = [dict(x) for x in release]
        collision[0]["source_id"] = dev[0]["source_id"]
        with self.assertRaises(ValueError):
            audit_pair(dev, collision, ["Truly separate training task here."])
        with self.assertRaises(ValueError):
            audit_pair(dev, release, [dev[0]["prompt"]])
        with self.assertRaises(ValueError):
            audit_pair(dev, release, [release[0]["prompt"]])

    def test_ten_gram_overlap_catches_verbatim_but_not_short_exact_only(self):
        dev, release = make_suite("dev"), make_suite("release")
        long_text = "unique sequence of words alpha beta gamma delta epsilon zeta eta theta iota"
        release[25]["prompt"] = f"{long_text} followed by private original problem."
        with self.assertRaises(ValueError):
            audit_pair(dev, release, [f"Training example: {long_text} from model."])
        short = release[2]["prompt"]
        with self.assertRaises(ValueError):
            audit_pair(dev, release, [short.lower()])

    def test_v5_prompt_and_v6_messages_train_formats_supported(self):
        samples = [
            {"prompt": [{"role": "system", "content": "x"}, {"role": "user", "content": "v5 prompt"}],
             "completion": [{"role": "assistant", "content": "17"}]},
            {"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "v6 prompt"},
                          {"role": "assistant", "content": "18"}]},
            {"prompt": "plain training prompt"},
        ]
        self.assertEqual(read_training_prompts(samples),
                         ["v5 prompt", "v6 prompt", "plain training prompt"])
        for bad in ({"messages": [{"role": "assistant", "content": "x"}]},
                    {"prompt": [{"role": "user", "content": "x"},
                                {"role": "user", "content": "y"}]},
                    {"payload": "unknown"},
                    {"prompt": ""}):
            with self.assertRaises(ValueError):
                read_training_prompts([bad])

    def test_create_only_digest_pins_and_only_aggregate_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dev, release = make_suite("dev"), make_suite("release")
            for label, rows in (("dev", dev), ("release", release)):
                (root / f"{label}.jsonl").write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf8")
            train = root / "v5.jsonl"
            train.write_text(json.dumps({"prompt": [
                {"role": "system", "content": "x"},
                {"role": "user", "content": "Old curriculum example"}
            ]}) + "\n", encoding="utf8")
            dest = root / "audit.json"
            result = audit_files(root / "dev.jsonl", root / "release.jsonl",
                                 [train], dest)
            self.assertEqual(result["training_files_checked"], 1)
            self.assertEqual(result["training_rows_checked"], 1)
            self.assertEqual(result["release_sha256"],
                             hashlib.sha256((root / "release.jsonl").read_bytes()).hexdigest())
            self.assertEqual(result["training_sha256"][0],
                             hashlib.sha256(train.read_bytes()).hexdigest())
            exported = dest.read_text(encoding="utf8")
            self.assertNotIn("Old curriculum", exported)
            self.assertNotIn("private unique", exported)
            self.assertFalse(result["release_validated"])
            with self.assertRaises(ValueError):
                audit_files(root / "dev.jsonl", root / "release.jsonl", [train], dest)
            modified = root / "modified.jsonl"
            modified.write_bytes((root / "release.jsonl").read_bytes().replace(
                b"Krelease_math_001", b"Kdev_math_001"))
            with self.assertRaises(ValueError):
                audit_files(root / "dev.jsonl", modified, [train], root / "other.json")


if __name__ == "__main__":
    unittest.main()
