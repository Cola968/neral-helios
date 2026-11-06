"""Contracts for import of historic, local NeralTalk350 train examples."""
import json
import subprocess
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.neral_legacy_transfer import (
    choose_candidates, scan_legacy_dataset, publish_candidate, validate_export,
)

SYSTEM = "Du bist NERAL auf offengelegter Qwen-Basis. Antworte präzise."


def row(idx, prompt=None, domain="code", source="neral-generated-code-de", answer="Eine fachlich korrekte Antwort.", lang="de"):
    return {"id": f"example-{idx}", "language": lang, "domain": domain,
            "source": source, "license": "Project-generated deterministic data; no external corpus",
            "messages": [{"role": "user", "content": prompt or f"Frage {idx}: Welche Python-Funktion ist richtig?"},
                         {"role": "assistant", "content": answer}]}


def make_root(parent, train, val=None, test=None):
    root = Path(parent) / "neraltalk"
    root.mkdir()
    for name, rows in {"train": train, "val": val or [row(900)], "test": test or [row(901)]}.items():
        (root / (name + ".jsonl")).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    return root


class LegacyTransfer(unittest.TestCase):
    def test_direct_script_execution_resolves_sibling_comparison_module(self):
        with TemporaryDirectory() as tmp:
            root = make_root(tmp, [row(i+1000) for i in range(105)])
            out = Path(tmp) / "new_candidate"
            original = Path(__file__).resolve().parents[1] / "scripts" / "neral_legacy_transfer.py"
            scripts = Path(tmp) / "standalone" / "scripts"
            scripts.mkdir(parents=True)
            script = scripts / "neral_legacy_transfer.py"
            shutil.copyfile(original, script)
            (scripts / "neral_r32_compare.py").write_text("TASKS = [('x','Test-only prompt','')]\n", encoding="utf-8")
            process = subprocess.run([sys.executable, "-B", str(script),
                                      "--source-root", str(root), "--output-dir", str(out)],
                                     capture_output=True, text=True, timeout=15)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(validate_export(out / "candidate.jsonl"), 105)
            self.assertFalse(json.loads((out / "summary.json").read_text())["approved_for_training"])

    def test_no_import_from_val_test_even_with_distinct_answers(self):
        with TemporaryDirectory() as tmp:
            train = [row(1, prompt="Doppelte Frage"), row(2, prompt="Einzigartige Frage")]
            root = make_root(tmp, train, val=[row(3, prompt="doppelte   frage", answer="Andere Antwort")])
            scan = scan_legacy_dataset(root)
            selected, stats = choose_candidates(scan, quotas={"code": 10})
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["messages"][1]["content"], "Einzigartige Frage")
            self.assertEqual(stats["excluded_heldout_prompt"], 1)

    def test_deterministic_and_no_reused_train_prompts(self):
        with TemporaryDirectory() as tmp:
            a, b, c = row(1, prompt="Wiederholt!"), row(2, prompt="WIEDERHOLT!"), row(3, prompt="Neue einzigartige Frage")
            root = make_root(tmp, [a,b,c])
            scan = scan_legacy_dataset(root)
            first, _ = choose_candidates(scan, quotas={"code": 10})
            second, _ = choose_candidates(scan, quotas={"code": 10})
            self.assertEqual(first, second)
            self.assertEqual(len(first), 2)

    def test_no_multiturn_or_secrets_or_think_tags(self):
        with TemporaryDirectory() as tmp:
            multi = row(2)
            multi["messages"] += [{"role":"user","content":"Erkläre weiter"},{"role":"assistant","content":"Mehr Text"}]
            root = make_root(tmp, [row(1), multi, row(3, answer="sk-"+"x"*24), row(4, answer="<think>privat</think>")])
            scan = scan_legacy_dataset(root)
            selected, stats = choose_candidates(scan, quotas={"code": 10})
            self.assertEqual(len(selected), 1)
            self.assertEqual(stats["excluded_multiturn"], 1)
            self.assertEqual(stats["excluded_sensitive_or_invalid"], 2)

    def test_requires_explicit_source_metadata_and_supported_license(self):
        with TemporaryDirectory() as tmp:
            bad = row(1)
            bad["source"] = "unknown_source"
            root = make_root(tmp, [bad])
            with self.assertRaises(ValueError):
                scan_legacy_dataset(root)

    def test_output_is_qwen_trainer_shape_and_not_approved(self):
        with TemporaryDirectory() as tmp:
            root = make_root(tmp, [row(1), row(2)])
            selected, stats = choose_candidates(scan_legacy_dataset(root), quotas={"code": 2})
            destination = Path(tmp) / "candidate"
            report = publish_candidate(destination, selected, stats, source_root=root)
            self.assertFalse(report["approved_for_training"])
            self.assertEqual(report["rows"], 2)
            self.assertNotIn("Frage", json.dumps(report))
            self.assertEqual(validate_export(destination / "candidate.jsonl"), 2)
            first = json.loads((destination / "candidate.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual([x["role"] for x in first["messages"]], ["system","user","assistant"])
            with self.assertRaises(FileExistsError):
                publish_candidate(destination, selected, stats, source_root=root)

    def test_disallows_symlinked_train_file(self):
        with TemporaryDirectory() as tmp:
            root = make_root(tmp, [row(1)])
            src = root / "train.jsonl"
            other = root / "backup.jsonl"
            src.rename(other)
            try:
                src.symlink_to(other)
            except OSError:
                self.skipTest("Symlinks unavailable")
            with self.assertRaises(ValueError):
                scan_legacy_dataset(root)

    def test_stable_domain_quota_and_no_injected_system(self):
        with TemporaryDirectory() as tmp:
            root = make_root(tmp, [row(1), row(2), row(3,domain="math",source="neral-generated-math-de")])
            candidates, stats = choose_candidates(scan_legacy_dataset(root), quotas={"code":1,"math":1})
            self.assertEqual(len(candidates), 2)
            self.assertEqual(sum(c["messages"][0]["content"] == SYSTEM for c in candidates), 0)
            self.assertEqual(set(stats["selected_by_domain"]), {"code", "math"})


if __name__ == "__main__": unittest.main()
