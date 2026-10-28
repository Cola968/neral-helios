"""Read-only native Arctus350 semantic arithmetic split integrity audit.

Exact prompt deduplication misses equivalent mathematical *questions*:
"Add 5 and 6" and "Was ist 6 + 5?" describe the same commutative fact.
This audit accepts only known, independently re-verifiable synthetic add
sources, and checks unordered operand-pair identity across TRAIN/VAL/TEST.

- No model, GPU, network, disk write, tokenizer, or checkpoint is needed.
- It does NOT modify or move data. It NEVER prints raw source content/IDs.
- Non-generated math is OPAQUE: not silently declared leakage-free.
- This is a development-curriculum check, NOT a frozen independent eval.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .chat350_data import SPLITS
from .arctus350_exact_math_curriculum import SOURCE_LICENSE
from .arctus350_balanced_arithmetic_v2 import (
    SOURCE as V2_SOURCE, parse_reference as parse_v2,
)
from .arctus350_carry_curriculum_v3 import (
    SOURCE as V3_SOURCE, parse_reference as parse_v3,
)

V1_SOURCE = "neral-generated-small-add-v1"
_GENERATED_SOURCES = frozenset((V1_SOURCE,V2_SOURCE,V3_SOURCE))
_V1_DE = re.compile(
    r"Berechne (\d+) plus (\d+)\. Gib ausschließlich das Ergebnis als Zahl an\.\Z")
_V1_EN = re.compile(
    r"Add (\d+) and (\d+)\. Give only the numerical result\.\Z")


def _verified_pair(row: dict) -> tuple[int,int] | None:
    """One fact key, iff BOTH question and answer match the owned generator."""
    source = row.get("source")
    if source == V1_SOURCE:
        turns = row["messages"]
        if len(turns) != 2 or [t["role"] for t in turns] != ["user","assistant"]:
            return None
        prompt = turns[0]["content"]
        pattern = _V1_DE if row.get("language") == "de" else _V1_EN if row.get("language") == "en" else None
        match = pattern.fullmatch(prompt) if isinstance(prompt,str) and pattern else None
        if match is None:
            return None
        a,b = map(int, match.groups())
        if not (0 <= a <= 19 and 0 <= b <= 19) or (a,b) in ((7,8),(8,7)):
            return None
        if turns[1]["content"] != str(a+b):
            return None
    elif source == V2_SOURCE:
        pair = parse_v2(row)
        if pair is None:
            return None
        a,b = pair
    elif source == V3_SOURCE:
        pair = parse_v3(row)
        if pair is None:
            return None
        a,b = pair
    else:
        return None
    if row.get("domain") != "math" or row.get("license") != SOURCE_LICENSE:
        return None
    return (min(a,b), max(a,b))


def audit_math_splits(rows: dict) -> dict:
    """Return aggregate-only report. No raw prompts, IDs or operand pairs."""
    if not isinstance(rows,dict) or set(rows) != set(SPLITS):
        raise ValueError("expected exact train/val/test splits")
    per_split = {}
    keys = {}
    frequencies = {}
    for split in SPLITS:
        group = rows[split]
        if not isinstance(group,list) or not group:
            raise ValueError("missing or empty split")
        counts = Counter()
        signatures = set()
        freq = Counter()
        for row in group:
            if not isinstance(row,dict):
                raise ValueError("invalid row in split")
            source = row.get("source")
            if not isinstance(source,str) or not isinstance(row.get("domain"),str):
                raise ValueError("invalid source/domain metadata")
            if source in _GENERATED_SOURCES:
                try:
                    pair = _verified_pair(row)
                except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                    pair = None
                if pair is None:
                    # Fail closed rather than trusting metadata or a wrong label.
                    raise ValueError("unverifiable generated math row; source validation failed")
                signatures.add(pair)
                freq[pair] += 1
                counts["verified_generated_math_rows"] += 1
                counts[source] += 1
            elif row["domain"] == "math":
                counts["opaque_math_rows"] += 1
            else:
                counts["other_domain_rows"] += 1
        keys[split] = signatures
        frequencies[split] = freq
        per_split[split] = {
            "total_rows":len(group),
            "verified_generated_math_rows":counts["verified_generated_math_rows"],
            "opaque_math_rows":counts["opaque_math_rows"],
            "other_domain_rows":counts["other_domain_rows"],
            "unique_unordered_pairs":len(signatures),
            "by_known_source":{s:counts[s] for s in sorted(_GENERATED_SOURCES)},
        }
    train,val,test = (keys[s] for s in SPLITS)
    overlap = {
        "train_val":len(train&val),
        "train_test":len(train&test),
        "val_test":len(val&test),
        "train_with_holdout":len(train&(val|test)),
        "unique_unordered_pairs_any_split":len((train&val)|(train&test)|(val&test)),
        "affected_rows":{
            split:sum(n for pair,n in frequencies[split].items()
                      if any(pair in keys[other] for other in SPLITS if other != split))
            for split in SPLITS
        },
    }
    opaque = sum(per_split[s]["opaque_math_rows"] for s in SPLITS)
    disjoint = overlap["unique_unordered_pairs_any_split"] == 0
    return {
        "schema":"arctus350-semantic-math-holdout-v1",
        "status":"read_only_aggregates",
        "per_split":per_split,
        "overlap":overlap,
        "gate":{
            "disjoint_math_pairs":disjoint,
            "unverified_generated_math_rows":0, # corrupt known sources raise instead
            "opaque_math_rows":opaque,
            "known_math_coverage_complete":opaque == 0,
            "safe_for_controlled_sft":disjoint and opaque == 0,
            "ready_for_release":False,
        },
        "normalization":"canonical_commutative_addition_pair_from_verified_source",
        "caveat":"not_an_independent_frozen_benchmark; unrecognized or paraphrased natural math remains opaque; this does not detect semantic knowledge contamination.",
        "privacy":"only counts; source texts, private IDs, answers and operands NEVER exported",
        "training_performed":False,
        "checkpoint_modified":False,
    }


def require_disjoint_math_pairs(report: dict) -> None:
    if (not isinstance(report,dict)
        or report.get("schema")!="arctus350-semantic-math-holdout-v1"
        or not isinstance(report.get("gate"),dict)
        or report["gate"].get("disjoint_math_pairs") is not True
        or report["gate"].get("known_math_coverage_complete") is not True):
        raise ValueError("semantic arithmetic train/val/test leakage or opaque math prevents approval")



def sanitize_math_splits(rows: dict) -> tuple[dict,dict]:
    """Make a NEW in-memory split copy, TEST > VAL > TRAIN priority.

    A leaked commutative sum is excluded from the lower-priority partition,
    while all variants of the same fact inside one partition stay together.
    The original rows and TEST split are unchanged; no file write is made.
    Non-generated math cannot be safely filtered and forces a stop.
    """
    from copy import deepcopy

    original_report = audit_math_splits(rows)
    if original_report["gate"]["opaque_math_rows"]:
        raise ValueError("opaque math requires human data review before numeric split sanitization")
    kept = {}
    reserved = set()
    dropped = {}
    for split in ("test","val","train"):
        accepted = []
        seen_in_split = set()
        for row in rows[split]:
            source = row.get("source")
            pair = _verified_pair(row) if source in _GENERATED_SOURCES else None
            if pair is not None and pair in reserved:
                continue
            accepted.append(deepcopy(row))
            if pair is not None:
                seen_in_split.add(pair)
        if not accepted:
            raise ValueError("semantic math sanitization would leave split empty")
        kept[split] = accepted
        dropped[split] = len(rows[split])-len(accepted)
        reserved.update(seen_in_split)
    result = {split:kept[split] for split in SPLITS}
    post = audit_math_splits(result)
    require_disjoint_math_pairs(post)
    # Preserve TEST exactly, without sharing mutable references to the source.
    if result["test"] != rows["test"]:
        raise AssertionError("TEST split changed unexpectedly")
    return result,{
        "schema":"arctus350-semantic-math-sanitization-v1",
        "priority":"TEST > VAL > TRAIN; higher-priority original questions preserved",
        "dropped":{split:dropped[split] for split in SPLITS},
        "original_counts":{split:len(rows[split]) for split in SPLITS},
        "retained_counts":{split:len(result[split]) for split in SPLITS},
        "original_numeric_overlap":original_report["overlap"]["unique_unordered_pairs_any_split"],
        "remaining_numeric_overlap":post["overlap"]["unique_unordered_pairs_any_split"],
        "immutable_test_rows":True,
        "source_modified":False,
        "raw_text_or_record_ids_exported":False,
        "training_started":False,
        "release_allowed":False,
        "caveat":"Owned deterministic arithmetic only; not a general semantic deduplicator or independent benchmark.",
    }



def write_sanitized_generated_v1(destination: Path) -> dict:
    """Create a NEW local v1 copy with commutative-split leaks removed.

    This uses ONLY the repo's owned deterministic generator; no private local
    dataset/weights are accessed. It never overwrites the source or destination.
    """
    from .arctus350_exact_math_curriculum import make_splits as v1
    from .chat350_data import validate_dataset_root, validate_prompt_holdouts
    destination=Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("refuse to overwrite an existing dataset or symlink")
    original=v1()
    clean,stats=sanitize_math_splits(original)
    require_disjoint_math_pairs(audit_math_splits(clean))
    if clean["test"]!=original["test"]:
        raise AssertionError("immutable TEST has changed")
    # Never create a dataset until every purely in-memory semantic check passes.
    destination.mkdir(parents=False,exist_ok=False)
    for split in SPLITS:
        path=destination/(split+".jsonl")
        with path.open("x",encoding="utf8") as out:
            for row in clean[split]:
                out.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
    manifest,loaded=validate_dataset_root(destination)
    validate_prompt_holdouts(loaded)
    require_disjoint_math_pairs(audit_math_splits(loaded))
    report={
        "schema":"arctus350-v1-semantic-sanitized-derived-v1",
        "original_numeric_overlap":stats["original_numeric_overlap"],
        "remaining_numeric_overlap":stats["remaining_numeric_overlap"],
        "priority":"TEST > VAL > TRAIN",
        "dropped":stats["dropped"],
        "original_counts":stats["original_counts"],
        "retained_counts":stats["retained_counts"],
        "derived_manifest_sha256":manifest["manifest_sha256"],
        "read_only_source":True,
        "immutable_test_rows":True,
        "raw_text_or_record_ids_exported":False,
        "training_started":False,
        "release_allowed":False,
        "caveat":"Synthetic derived curriculum, not independent validation or legal clearance.",
    }
    (destination/"metadata_report.json").write_text(
        json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+"\n",
        encoding="utf8",
    )
    return report


def generated_audit_reports() -> dict:
    """Recompute from native deterministic generators; development only."""
    from .arctus350_exact_math_curriculum import make_splits as v1
    from .arctus350_balanced_arithmetic_v2 import make_splits as v2
    from .arctus350_carry_curriculum_v3 import make_splits as v3
    return {version:audit_math_splits(builder()) for version,builder in (
        ("v1",v1),("v2",v2),("v3",v3)
    )}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--generated-audit",action="store_true",
                        help="Audit three deterministic existing generators; no local files read.")
    choice.add_argument("--data",type=Path,
                        help="Read local prepared train/val/test JSONL; read-only.")
    choice.add_argument("--sanitize-generated-v1-to",type=Path,
                        help="Write a NEW locally derived synthetic v1 dataset with semantic split repair.")
    parser.add_argument("--out",type=Path,
                        help="Write aggregate-only report to a NEW file; never overwrite.")
    parser.add_argument("--strict",action="store_true",
                        help="Reject any semantic split overlap or unknown math.")
    args=parser.parse_args()
    if args.sanitize_generated_v1_to is not None:
        if args.out is not None or args.strict:
            parser.error("--out/--strict are incompatible with --sanitize-generated-v1-to")
        report=write_sanitized_generated_v1(args.sanitize_generated_v1_to)
        print(json.dumps(report,sort_keys=True,ensure_ascii=False))
        return
    if args.out is not None and args.out.exists():
        raise FileExistsError("refuse to overwrite evidence")
    if args.generated_audit:
        report=generated_audit_reports()
        if args.strict:
            for version,findings in report.items():
                require_disjoint_math_pairs(findings)
    else:
        from .chat350_data import validate_dataset_root, validate_prompt_holdouts
        manifest, split_rows=validate_dataset_root(args.data)
        validate_prompt_holdouts(split_rows)
        findings=audit_math_splits(split_rows)
        report={"manifest_sha256":manifest["manifest_sha256"],"findings":findings}
        if args.strict:
            require_disjoint_math_pairs(findings)
    output=json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n"
    if args.out is None:
        print(output,end="")
    else:
        args.out.write_text(output,encoding="utf8")
        print(json.dumps({"aggregate_report_written":True,"source_text_exported":False}))
if __name__=="__main__":
    main()
