"""Prospective NERAL V4.1 200-case blind-set validator and optional local runner.

DO NOT ship a public answer key, ingest these cases into training, or run
the 600-case live benchmark as part of a normal deployment.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

CATEGORIES = {
    "everyday": 35,
    "multiturn": 40,
    "live_web": 35,
    "math_logic": 30,
    "programming": 25,
    "writing": 20,
    "uncertainty": 15,
}
TIERS = ("Amber", "Quartz", "Obsidian")
TOTAL = sum(CATEGORIES.values())


def validate_suite(path: Path) -> dict:
    """Reject incomplete, repeated, leaked or unscorable records."""
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 2_000_000:
        raise ValueError("Sealed private JSONL file missing or oversized")
    data = path.read_bytes()
    cases = [json.loads(row) for row in data.decode("utf-8").splitlines() if row.strip()]
    if len(cases) != TOTAL:
        raise ValueError(f"Expected {TOTAL} cases, found {len(cases)}")
    ids, prompt_hashes = set(), set()
    counts = Counter()
    for case in cases:
        if (type(case) is not dict or
            not {"id", "category", "prompt", "history", "grader"} <= set(case) or
                set(case) - {"id", "category", "prompt", "history", "grader",
                             "web_query", "sealed_reference"}):
            raise ValueError("Case schema invalid")
        case_id, category, prompt = case["id"], case["category"], case["prompt"]
        if (type(case_id) is not str or not 4 <= len(case_id) <= 70
            or case_id in ids or category not in CATEGORIES or
            type(prompt) is not str or not 5 <= len(prompt.encode("utf-8")) <= 4096):
            raise ValueError("Case identity/category/prompt invalid")
        ids.add(case_id)
        normalized = " ".join(prompt.casefold().split())
        digest=sha256(normalized.encode("utf-8")).hexdigest()
        if digest in prompt_hashes:
            raise ValueError("Duplicate prompt detected")
        prompt_hashes.add(digest)
        turns=case["history"]
        if type(turns) is not list or len(turns)>12 or len(turns)%2:
            raise ValueError("Invalid completed history")
        for i,turn in enumerate(turns):
            if (type(turn) is not dict or set(turn)!={"role","content"} or
                turn["role"] != ("user" if i%2==0 else "assistant") or
                type(turn["content"]) is not str or not turn["content"].strip()
                or len(turn["content"].encode("utf-8"))>2048):
                raise ValueError("Malformed history turn")
        if category=="multiturn" and not turns:
            raise ValueError("Multi-turn must contain prior dialogue")
        if category=="live_web":
            web=case.get("web_query")
            if (type(web) is not str or not 2 <= len(web) <= 160):
                raise ValueError("Web case lacks explicit consented query")
        elif "web_query" in case:
            raise ValueError("Unexpected web query in non-web task")
        if case["grader"] not in ("independent_exact","blind_pairwise",
                                   "source_freshness","test_execution"):
            raise ValueError("Unsupported grader")
        if case["grader"]=="independent_exact" and "sealed_reference" not in case:
            raise ValueError("Exact-graded case has no private reference")
        counts[category]+=1
    if dict(counts) != CATEGORIES:
        raise ValueError(f"Unbalanced category distribution: {dict(counts)}")
    return {"cases":len(cases),"categories":dict(counts),
            "sha256":sha256(data).hexdigest(), "modes":list(TIERS)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sealed",required=True,type=Path)
    parser.add_argument("--manifest-out",type=Path)
    args=parser.parse_args()
    manifest=validate_suite(args.sealed)
    if args.manifest_out:
        if args.manifest_out.resolve()==args.sealed.resolve():
            raise ValueError("Cannot overwrite sealed input")
        args.manifest_out.parent.mkdir(parents=True,exist_ok=True)
        args.manifest_out.write_text(
            json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,sort_keys=True))


if __name__=="__main__":
    main()
