"""Read-only, CPU-only structural cross-split prompt leakage audit for Arctus350.

This is a heuristic REVIEW tool, not a training gate or proof of semantic
contamination. Keeps prompt contents in memory only; outputs aggregate counts.
It does not import torch, access model weights, or modify input JSONL files.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import unicodedata
from pathlib import Path

SPLITS = ("train", "val", "test")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def canonical_prompt(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("user message must be nonempty text")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def numeric_template(text: str) -> str:
    """Heuristic normalization; distinct numbers may change question semantics."""
    return NUMBER.sub("<num>", canonical_prompt(text))


def _cross_pairs(by_split: dict[str, int]) -> int:
    return sum(by_split.get(a, 0) * by_split.get(b, 0)
               for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:])


def _check_row(row: object) -> tuple[str, list[str]]:
    if not isinstance(row, dict):
        raise ValueError("dataset row must be an object")
    rid = row.get("id")
    if not isinstance(rid, str) or not rid.strip():
        raise ValueError("row needs a nonempty text id")
    if "messages" in row:
        turns = row["messages"]
        if not isinstance(turns, list) or not (2 <= len(turns) <= 32) or len(turns) % 2:
            raise ValueError("messages must be 2..32 alternating turns ending with assistant")
        result = []
        for i, msg in enumerate(turns):
            if not isinstance(msg, dict) or msg.get("role") != ("user" if i % 2 == 0 else "assistant"):
                raise ValueError("invalid role order")
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip() or len(content) > 16_384:
                raise ValueError("invalid turn content")
            if i % 2 == 0:
                result.append(content)
        return rid, result
    prompt, answer = row.get("prompt"), row.get("answer")
    if not all(isinstance(v, str) and v.strip() and len(v) <= 16_384 for v in (prompt, answer)):
        raise ValueError("single-turn row needs nonempty prompt/answer")
    return rid, [prompt]


def audit_root(root: Path) -> dict:
    root = Path(root)
    # Counts are occurrences, not de-duplicated unique questions. Keep all
    # grouping keys in memory, and never serialize prompt or row identifiers.
    groups = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0, 0])))
    ids: set[str] = set()
    row_counts = {}
    turn_counts = {}

    for split in SPLITS:
        path = root / (split + ".jsonl")
        count, users = 0, 0
        with path.open("r", encoding="utf8") as stream:
            for line_num, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    rid, utterances = _check_row(json.loads(line))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{split}.jsonl line {line_num}: {exc}") from exc
                if rid in ids:
                    raise ValueError("duplicate row id across/within splits")
                ids.add(rid)
                count += 1
                for i, prompt in enumerate(utterances):
                    exact = canonical_prompt(prompt)
                    template = numeric_template(prompt)
                    groups[template][exact][split][int(i > 0)] += 1
                    users += 1
        if count == 0:
            raise ValueError("empty split: " + split)
        row_counts[split], turn_counts[split] = count, users

    exact_pairs = template_only = later_exact = later_template = 0
    for template, exacts in groups.items():
        by_split = collections.Counter()
        first_by_split = collections.Counter()
        exact_all = exact_first = 0
        for occurrence in exacts.values():
            totals = {split: first + later for split, (first, later) in occurrence.items()}
            first = {split: occ[0] for split, occ in occurrence.items()}
            for split, n in totals.items():
                by_split[split] += n
            for split, n in first.items():
                first_by_split[split] += n
            exact_all += _cross_pairs(totals)
            exact_first += _cross_pairs(first)
        exact_pairs += exact_all
        later_exact += exact_all - exact_first
        if "<num>" in template:
            all_pairs = _cross_pairs(by_split)
            first_pairs = _cross_pairs(first_by_split)
            template_only += all_pairs - exact_all
            later_template += (all_pairs - first_pairs) - (exact_all - exact_first)

    overlaps = {
        "exact_pairs": exact_pairs,
        "numeric_template_only_pairs": template_only,
        "later_turn_exact_pairs": later_exact,
        "later_turn_numeric_template_only_pairs": later_template,
    }
    status = "review_required" if exact_pairs or template_only else "no_structural_overlap_detected"
    return {
        "schema": "arctus350-template-leakage-audit-v1",
        "status": status,
        "rows_by_split": row_counts,
        "user_turns_by_split": turn_counts,
        "overlaps": overlaps,
        "normalization": "NFKC+casefold+whitespace; numeric placeholders for Unicode decimal digit runs",
        "interpretation": "Heuristic review only; not a semantic contamination clearance. "
                          "Numeric template matches may be false positives; missing matches do not prove independence.",
        "privacy": "aggregate counts only; no raw prompts, user messages, or row IDs in output",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="local NeralTalk train/val/test JSONL folder")
    parser.add_argument("--out", type=Path, required=True, help="new report JSON file; must not exist")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("report output already exists; refusing to overwrite")
    report = audit_root(args.root)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
