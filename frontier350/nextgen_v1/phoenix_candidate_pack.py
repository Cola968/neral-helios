"""Build a PRIVATE screened Phoenix candidate pack from pinned OASST2.

This is deliberately NOT a training pack: no holdout has been checked and no
human/source approval is inferred. Raw text stays inside the private asset root.
Stdout contains only aggregate counts and hashes. CPU-only; never loads a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

# fetch_oasst2_sft50.py lives at the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phoenix_clean_sampler import choose_screened_diverse, POLICY
from phoenix_oasst2_bridge import load_archive, select_tree_disjoint
from sft_contract import norm, sha

EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE = re.compile(r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\d[ .-]?){10,14}(?!\w)")
ROLE = re.compile(r"(?im)^\s*(?:user|nutzer|assistant|system)\s*:")


def _private_file(path: Path, root: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError("required private input missing or symlinked")
    p = path.resolve(strict=True)
    if not p.is_relative_to(root):
        raise ValueError("private input outside assets root")
    return p


def _repeat_ratio(text: str) -> float:
    words = re.findall(r"\w+", text.casefold())
    grams = [tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2))]
    if not grams:
        return 0.0
    return (len(grams) - len(set(grams))) / len(grams)


def audit(splits: dict[str, list[dict]]) -> dict:
    seen_prompt: set[str] = set()
    seen_answer: set[str] = set()
    ids: set[str] = set()
    flags = Counter()
    total = 0
    for split in ("train", "validation"):
        for row in splits[split]:
            total += 1
            ident = row["id"]
            if ident in ids:
                flags["duplicate_id"] += 1
            ids.add(ident)
            p = norm(row["user"])
            a = norm(row["assistant"])
            if p in seen_prompt:
                flags["duplicate_prompt"] += 1
            if a in seen_answer:
                flags["duplicate_answer"] += 1
            seen_prompt.add(p); seen_answer.add(a)
            text = row["user"] + "\n" + row["assistant"]
            if EMAIL.search(text): flags["email_pattern"] += 1
            if PHONE.search(text): flags["phone_pattern"] += 1
            if ROLE.search(row["assistant"]): flags["role_marker"] += 1
            if len(row["user"]) > 512: flags["long_prompt"] += 1
            if len(row["assistant"].strip()) < 40: flags["short_answer"] += 1
            if len(row["assistant"]) > 2500: flags["long_answer"] += 1
            if _repeat_ratio(row["assistant"]) > 0.30: flags["high_trigram_repetition"] += 1
    return {"rows": total, "flags": dict(sorted(flags.items())),
            "distinct_ids": len(ids), "distinct_prompts": len(seen_prompt),
            "distinct_answers": len(seen_answer)}


def build(root: Path, archive: Path, out: Path, max_de: int = 400, max_en: int = 800) -> dict:
    root = root.resolve(strict=True)
    archive = _private_file(archive, root)
    if out.exists() or out.is_symlink() or not out.parent.resolve(strict=True).is_relative_to(root):
        raise ValueError("candidate output must be a new directory inside private root")
    if any((d / ".git").exists() for d in (root, *root.parents)):
        raise ValueError("private root may not be inside a git worktree")
    messages = load_archive(archive)
    selected, selection = choose_screened_diverse(messages, max_en=max_en, max_de=max_de)
    # No frozen benchmark is supplied here on purpose. This is a candidate-only export.
    splits, split_summary = select_tree_disjoint(selected, messages, frozen=set())
    measured = audit(splits)
    if measured["rows"] != max_de + max_en:
        raise ValueError("candidate population changed during split conversion")
    if measured["distinct_ids"] != measured["rows"] or measured["distinct_prompts"] != measured["rows"]:
        raise ValueError("candidate identity/prompt uniqueness failed")
    hard = {"duplicate_id", "duplicate_prompt", "email_pattern", "phone_pattern", "role_marker", "long_prompt", "short_answer", "long_answer"}
    if any(measured["flags"].get(k, 0) for k in hard):
        raise ValueError("hard candidate hygiene flag reappeared")
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema": 1, "classification": "private_candidate_only_not_training_pack",
              "selector_policy": POLICY, "selected_de": max_de, "selected_en": max_en,
              "selection_inventory_sha256": selection["selected_inventory_sha256"],
              "split": split_summary, "audit": measured,
              "holdout_checked": False, "human_content_review_completed": False,
              "license_approved": False, "privacy_approved": False,
              "training_approved": False, "training_started": False,
              "raw_text_uploaded": False}
    created: list[Path] = []
    try:
        for split, rows in splits.items():
            path = out / f"{split}.jsonl"
            with path.open("x", encoding="utf-8") as f:
                created.append(path)
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            report[f"{split}_sha256"] = sha(path)
        rpath = out / "candidate_report.json"
        with rpath.open("x", encoding="utf-8") as f:
            created.append(rpath)
            json.dump(report, f, sort_keys=True, indent=2)
            f.write("\n")
    except Exception:
        for p in created:
            p.unlink(missing_ok=True)
        try: out.rmdir()
        except OSError: pass
        raise
    return {"status": "candidate_pack_created_not_approved", "rows": measured["rows"],
            "train_rows": len(splits["train"]), "validation_rows": len(splits["validation"]),
            "flags": measured["flags"], "training_approved": False, "training_started": False,
            "raw_text_uploaded": False, "report_sha256": sha(out / "candidate_report.json")}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets-root", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-de", type=int, default=400)
    p.add_argument("--max-en", type=int, default=800)
    a = p.parse_args()
    try:
        print(json.dumps(build(a.assets_root, a.archive, a.out, a.max_de, a.max_en), sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"status":"candidate_pack_blocked","training_approved":False,"training_started":False,"raw_text_uploaded":False}, sort_keys=True))
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
