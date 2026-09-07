"""I1 SFT builder with an additional post-B3 Targeted-v1 leakage screen.

The base SFT builder already screens all frozen pre-I1 release benchmarks. This wrapper also
screens the new targeted diagnostic from instruction data. It intentionally does not claim
that Targeted-v1 was excluded from the already-frozen B3 corpus; therefore Targeted-v1 remains
supplemental evidence, not a replacement for the pre-existing promotion suites.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import prepare_sft50 as base


def targeted_literals(repo_root: Path):
    path = repo_root / "benchmarks50" / "targeted_v1.jsonl"
    phrases = set()
    files = {}
    if path.is_file():
        files[str(path.relative_to(repo_root))] = base.sha256_file(path)
        for line in path.read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("prompt", "preferred", "distractor"):
                value = row.get(key)
                if isinstance(value, str):
                    n = base.normalize(value)
                    if len(n) >= 20:
                        phrases.add(n)
    return phrases, files


def build(source_root: Path, out: Path, target_train_examples: int, repo_root: Path, max_seq=1024):
    manifest_path = source_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    sources = base.validate_manifest(manifest, source_root)
    phrases, bench_files = base.benchmark_literals(repo_root)
    extra, extra_files = targeted_literals(repo_root)
    phrases = sorted(set(phrases) | extra, key=len, reverse=True)
    bench_files |= extra_files
    reject = collections.Counter()
    rows = base.load_rows(sources, phrases, reject, max_seq)
    base.audit_prompt_group_splits(rows)
    target = target_train_examples or base.auto_train_target(rows, sources, manifest.get("recommended_train_examples"))
    if target < 1000:
        raise ValueError("Use at least 1000 train examples for I1 instruction tuning")
    selected, selection = base.select_train(rows, sources, target)
    result = base.write_outputs(rows, selected, sources, out, target, reject, selection, bench_files, base.sha256_file(manifest_path))
    result["max_training_sequence"] = max_seq
    result["i1_targeted_v1_screened_from_sft"] = True
    result["targeted_v1_pretraining_contamination_status"] = "not_certified; B3 corpus was frozen before Targeted-v1"
    (out / "manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", default="sft50_sources")
    ap.add_argument("--out", default="sft50_data")
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--target-train-examples", type=int, default=0)
    args = ap.parse_args()
    result = build(Path(args.source_root), Path(args.out), args.target_train_examples, Path(__file__).resolve().parent, args.max_seq)
    print(json.dumps({"target_train_examples": result["target_train_examples"], "train": result["splits"]["train"], "val": result["splits"]["val"], "rejected": result["rejected"], "targeted_screened": True}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
