"""Extract *only locally* repeated-answer cases for independent human triage.

Not a dataset export, human review, source approval or training. Source is exact
SHA-pinned OASST2; selection is the separately pinned screened EXPERIMENT. Raw
text and IDs may appear only in a new JSONL under a private non-git root; stdout
contains counts only. Neither a matching answer nor a zero flag proves quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phoenix_clean_sampler import choose_screened_diverse
from phoenix_oasst2_bridge import UPSTREAM_SHA256, load_archive, local_input, select_tree_disjoint
from sft_contract import norm

PINNED_CANDIDATE_INVENTORY = "e1594d6a1e9487eb9b019c93f5bbc74155e7dc22623bb6359e53704b7fc463bf"
MAX_TRIAGE_ROWS = 32


def collect_duplicates(splits: dict, language_by_id: dict) -> tuple[list[dict], dict]:
    """All repeated normalized answers; no arbitrary cap/truncation of flagged cases."""
    if set(splits) != {"train", "validation"}:
        raise ValueError("unexpected selection splits")
    groups: dict[str, list[tuple[str, dict]]] = {}
    ids: set[str] = set()
    total = Counter()
    for split in ("train", "validation"):
        rows = splits[split]
        if not isinstance(rows, list):
            raise ValueError("unexpected split type")
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"id", "source_id", "user", "assistant"}:
                raise ValueError("unexpected source row schema")
            ident = row["id"]
            if (not isinstance(ident, str) or not ident or ident in ids
                    or language_by_id.get(ident) not in {"de", "en"}
                    or any(not isinstance(row[k], str) or not row[k].strip()
                           for k in ("user", "assistant"))):
                raise ValueError("invalid or duplicated source row")
            ids.add(ident)
            total[split] += 1
            groups.setdefault(norm(row["assistant"]), []).append((split, row))
    # Counter omits absent keys from .values(): min(total.values()) alone would
    # mistakenly accept a completely EMPTY validation or training split.
    if any(total[split] < 8 for split in ("train", "validation")):
        raise ValueError("source selection missing a proper split")
    repeated = [(fingerprint, members) for fingerprint, members in groups.items() if len(members) > 1]
    repeated.sort(key=lambda group: hashlib.sha256(group[0].encode("utf-8")).hexdigest())
    flagged_rows = sum(len(members) for _, members in repeated)
    if flagged_rows > MAX_TRIAGE_ROWS:
        raise ValueError("repeated-answer triage exceeds bounded private review size")
    out = []
    across_split = 0
    for number, (_, members) in enumerate(repeated, 1):
        members = sorted(members, key=lambda member: member[1]["id"])
        crossing = len({split for split, _ in members}) > 1
        across_split += int(crossing)
        for split, row in members:
            out.append({"id": row["id"], "split": split,
                        "language": language_by_id[row["id"]],
                        "duplicate_group": number, "cross_split": crossing,
                        "user": row["user"], "assistant": row["assistant"],
                        "content_review": "pending", "license_review": "pending"})
    return out, {"selected_rows": sum(total.values()),
                 "repeated_answer_groups": len(repeated),
                 "repeated_answer_rows": flagged_rows,
                 "extra_duplicate_answers": flagged_rows - len(repeated),
                 "cross_split_repeated_answer_groups": across_split}


def prepare(root: Path, archive: Path, out: Path, receipt: Path) -> dict:
    root = root.resolve(strict=True)
    if any((part / ".git").exists() for part in (root, *root.parents)):
        raise ValueError("private root must not live inside a git tree")
    archive = local_input(archive, root)
    if (out.exists() or out.is_symlink() or receipt.exists() or receipt.is_symlink()
            or out.resolve() == receipt.resolve()
            or not out.parent.resolve(strict=True).is_relative_to(root)
            or not receipt.parent.resolve(strict=True).is_relative_to(root)
            or out.suffix != ".jsonl" or receipt.suffix != ".json"):
        raise ValueError("distinct new private JSONL and JSON outputs required")
    messages = load_archive(archive)
    selected, identity = choose_screened_diverse(messages)
    if (identity.get("selected_inventory_sha256") != PINNED_CANDIDATE_INVENTORY
            or len(selected.get("de", [])) != 400 or len(selected.get("en", [])) != 800):
        raise ValueError("screened candidate inventory changed")
    splits, split_report = select_tree_disjoint(selected, messages, frozen=set())
    if split_report.get("split_disjoint") is not True:
        raise ValueError("conversation tree leakage")
    language = {row["id"]: lang for lang in ("de", "en") for row in selected[lang]}
    if len(language) != 1200:
        raise ValueError("duplicate source selection identifier")
    triage, totals = collect_duplicates(splits, language)
    if totals["selected_rows"] != 1200 or totals["extra_duplicate_answers"] != 1:
        raise ValueError("unexpected repeated-answer population; investigate privately")
    if not triage:
        raise ValueError("no duplicate candidates to triage")
    created = []
    try:
        with out.open("x", encoding="utf-8") as stream:
            created.append(out)
            for row in triage:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        sha = hashlib.sha256(out.read_bytes()).hexdigest()
        report = {"schema": 1, "status": "private_duplicate_answer_triage_pending",
                  "original_archive_sha256": UPSTREAM_SHA256,
                  "selected_inventory_sha256": PINNED_CANDIDATE_INVENTORY,
                  "triage_sha256": sha, **totals,
                  "human_reviews_completed": False, "quality_verified": False,
                  "privacy_approved": False, "license_approved": False,
                  "semantic_holdout_checked": False, "training_approved": False,
                  "training_started": False, "raw_text_uploaded": False}
        with receipt.open("x", encoding="utf-8") as stream:
            created.append(receipt)
            json.dump(report, stream, sort_keys=True, indent=2)
            stream.write("\n")
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return {"status": report["status"], **totals,
            "human_reviews_completed": False, "training_approved": False,
            "training_started": False, "raw_text_uploaded": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    for argument in ("assets-root", "archive", "out", "receipt"):
        parser.add_argument("--" + argument, type=Path, required=True)
    args = parser.parse_args()
    if not args.execute:
        parser.error("explicit --execute required to create private files")
    try:
        report = prepare(args.assets_root, args.archive, args.out, args.receipt)
    except Exception:
        print(json.dumps({"status": "duplicate_triage_blocked", "training_approved": False,
                          "training_started": False, "raw_text_uploaded": False}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
