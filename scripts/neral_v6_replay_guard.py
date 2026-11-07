"""NERAL v6 anti-forgetting candidate: train-only verified corrections and v5 replay.

No model calls, GPU, optimizer, training, automatic promotion or release claims.
Runs entirely on CPU. NEVER run on held-out or release prompts. Source model
labels from a submitted local pair file remain untrusted until independently
replayed; this script recomputes the deterministic task oracle.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from scripts.neral_e1_pair_miner import _answer_value
from scripts.neral_v6_compositional import FAMILIES, make_case
from scripts.neral_v6_r16_gpu import build_experiment
from scripts.neral_v5_transfer_curriculum import _generated, SEED as V5_SEED
from scripts.neral_decontamination import normalize, shingles

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SOURCES = ("v6_clean", "v5_replay", "verified_correction")


class ReplayRejected(ValueError):
    """Invalid source, unverifiable correction, or possible holdout leakage."""


def _prompt(row: dict) -> str:
    return row["messages"][1]["content"]


def _payload(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":")) + "\n" for row in rows).encode("utf8")


def _ranked(values: list[dict], seed: int) -> list[dict]:
    return sorted(values, key=lambda r: sha256(
        (str(seed) + "|" + _prompt(r)).encode("utf8")).hexdigest())


def _balanced(values: list[dict], count: int, seed: int, family_key) -> list[dict]:
    """Deterministic round-robin prevents any one source family dominating."""
    buckets = {}
    for item in values:
        buckets.setdefault(family_key(item), []).append(item)
    for name in buckets:
        buckets[name] = _ranked(buckets[name], seed)
    names = sorted(buckets)
    chosen = []
    while len(chosen) < count:
        active = False
        for name in names:
            if buckets[name]:
                chosen.append(buckets[name].pop(0))
                active = True
                if len(chosen) == count:
                    break
        if not active:
            raise ReplayRejected("Not enough independent source-family rows")
    return chosen


def _validate_v6(rows: list[dict]) -> list[dict]:
    canonical = build_experiment()[0]
    if rows != canonical or len(rows) != 728:
        raise ReplayRejected("Expected exactly the pinned 728 v6 TRAIN rows, no validation or holdout")
    return rows


def _validate_v5(rows: list[dict]) -> list[dict]:
    canonical = [json.loads(line)
                 for line in _generated(V5_SEED)["train"].decode("utf8").splitlines()]
    if rows != canonical or len(rows) != 1920:
        raise ReplayRejected("Expected pinned original v5 TRAIN split only")
    converted = []
    for row in rows:
        if (set(row) != {"prompt", "completion"} or
                not isinstance(row["prompt"], list) or
                not isinstance(row["completion"], list)):
            raise ReplayRejected("Unexpected v5 SFT schema")
        turns = row["prompt"] + row["completion"]
        if [t.get("role") for t in turns] != ["system", "user", "assistant"]:
            raise ReplayRejected("Unexpected v5 role order")
        converted.append({"messages": turns})
    return converted


def _verified_pairs(pairs: list[dict], v6: list[dict]) -> list[dict]:
    if not isinstance(pairs, list) or not 4 <= len(pairs) <= 200:
        raise ReplayRejected("At least four actual, verified training failures required")
    all_v6 = {_prompt(row): row for row in v6}
    seen = set()
    converted = []
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ReplayRejected("Invalid preference record")
        try:
            family = pair["family"]
            key = pair["id"]
            source_id = pair["source_id"]
            prompt = pair["prompt"]
            chosen = pair["chosen"]
            rejected = pair["rejected"]
            split = pair["split"]
            index = int(key.rsplit("-", 1)[-1])
            previous = make_case(family, "train", index)
            if (family not in FAMILIES or split != "train"
                    or key != f"v6-{family}-train-{index:03}"
                    or source_id != f"v6-original-{family}-train-{index:03}"
                    or key in seen
                    or pair["chosen_verified"] is not True
                    or pair["rejected_verified"] is not False
                    or pair["verifier_type"] != "numeric_oracle"
                    or not isinstance(pair["verifier_sha256"], str)
                    or HEX64.fullmatch(pair["verifier_sha256"]) is None
                    or prompt != previous["messages"][1]["content"]
                    or chosen != previous["messages"][2]["content"]
                    or prompt not in all_v6
                    or _prompt(all_v6[prompt]) != prompt):
                raise ReplayRejected("Untrusted train-only pair provenance")
            expected = _answer_value(family, chosen)
            observed = _answer_value(family, rejected)
            if expected is None or observed is None or observed == expected:
                raise ReplayRejected("Pair must contain a parseable, genuinely wrong numeric result")
            seen.add(key)
            # Reuse original supervised target, but explicitly expose the
            # genuinely wrong attempt as context for correction training.
            turns = [dict(t) for t in previous["messages"]]
            turns[1]["content"] = (
                prompt + "\nFehlerhafter vorheriger Versuch, nicht übernehmen:\n"
                + rejected + "\nPrüfe das Problem und korrigiere die Endantwort.")
            converted.append({"messages": turns, "source": "verified_correction",
                              "source_family": family})
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayRejected("Pair failed independently replayed oracle or train-only identity") from exc
    return converted


def _guard_protected(rows: list[dict], protected_prompts: list[str]) -> None:
    if (not isinstance(protected_prompts, list) or not protected_prompts or
            any(not isinstance(p, str) or not p.strip() for p in protected_prompts)):
        raise ReplayRejected("Protected evaluation prompts are mandatory")
    protected_exact = {normalize(p) for p in protected_prompts}
    protected_grams = set()
    for p in protected_prompts:
        protected_grams.update(shingles(p, n=10))
    for row in rows:
        prompt = _prompt(row)
        if normalize(prompt) in protected_exact or shingles(prompt, n=10) & protected_grams:
            raise ReplayRejected("Protected development/release overlap detected")


def build_candidate(v6_rows: list[dict], v5_rows: list[dict],
                    pairs: list[dict], *, protected_prompts: list[str],
                    seed: int = 20261107) -> tuple[list[dict], dict]:
    """Return an unapproved 50/30/20 replay candidate without holding any test data."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ReplayRejected("Seed out of range")
    v6 = _validate_v6(v6_rows)
    v5 = _validate_v5(v5_rows)
    correction = _verified_pairs(pairs, v6)
    k = len(correction)
    new_count = (5*k)//2
    old_count = 5*k - new_count - k
    correction_prompts = {
        p["prompt"] for p in pairs
    }
    clean = [{"messages": [dict(t) for t in r["messages"]],
              "source": "v6_clean",
              "source_family": (
                  next((name for name in FAMILIES if _prompt(r) ==
                        make_case(name, "train", j)["messages"][1]["content"]
                        for j in range(120)), "reasoning_counterfactual")
              )} for r in v6 if _prompt(r) not in correction_prompts]
    legacy = [{"messages": [dict(t) for t in r["messages"]],
               "source": "v5_replay",
               "source_family": "v5-" + next((
                   family for family in
                   ("affine","fractions","money","fuel","timezone","quantities",
                    "python_trace","python_debug")
                   if family in _prompt(r).lower()
               ), "original")} for r in v5]
    new_sel = _balanced(clean, new_count, seed, lambda r: r["source_family"])
    old_sel = _balanced(legacy, old_count, seed, lambda r: r["source_family"])
    combined = new_sel + old_sel + correction
    # Deduplicate by the full correction-context prompt; no exact repeated row.
    if len({normalize(_prompt(r)) for r in combined}) != len(combined):
        raise ReplayRejected("Duplicate training prompts after replay mixing")
    _guard_protected(combined, protected_prompts)
    shuffled = _ranked(combined, seed)
    counts = dict(Counter(row["source"] for row in shuffled))
    if counts != {"v6_clean": new_count,
                  "v5_replay": old_count, "verified_correction": k}:
        raise ReplayRejected("Mixed sampling contract failed")
    digest = sha256(_payload(shuffled)).hexdigest()
    report = {
        "schema": 1, "status": "NERAL_V6_REPLAY_DATA_CANDIDATE_ONLY",
        "source_counts": counts, "total": len(shuffled),
        "seed": seed, "candidate_sha256": digest,
        "v5_train_rows_checked": len(v5_rows),
        "v6_train_rows_checked": len(v6_rows),
        "verified_pair_records_replayed": k,
        "protected_prompt_count_checked": len(protected_prompts),
        "overlap_found": False,
        "general_improvement_proven": False,
        "release_validated": False,
        "training_performed": False,
        "training_allowed": False,
        "human_reviewed": False,
        "on_policy_origin_independently_verified": False,
        "note": ("Exact and 10-gram private holdout comparison; not semantic "
                 "decontamination. Corrected labels replayed against deterministic "
                 "train-only oracle; source metadata not cryptographically authenticated."),
    }
    return shuffled, report


def export_candidate(folder: Path, rows: list[dict], report: dict) -> dict:
    """Write a private create-only candidate and aggregate manifest."""
    folder = Path(folder)
    if (folder.is_symlink() or folder.exists() or not folder.parent.is_dir()
            or folder.parent.is_symlink()
            or folder.resolve().is_relative_to(Path(__file__).resolve().parents[1])):
        raise ReplayRejected("Private destination must be new and outside repository")
    raw = _payload(rows)
    if sha256(raw).hexdigest() != report.get("candidate_sha256"):
        raise ReplayRejected("Candidate SHA mismatch")
    if report.get("training_allowed") is not False:
        raise ReplayRejected("Unreviewed candidates must never authorize training")
    folder.mkdir(exist_ok=False)
    (folder/"candidate.jsonl").write_bytes(raw)
    (folder/"manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf8")
    return report


def main(argv: list[str] | None = None) -> None:
    """Recreate pinned train-only sources; require local pair and protected suites."""
    import argparse
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--verified-pairs", required=True, type=Path)
    p.add_argument("--protected-jsonl", required=True, action="append", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=20261107)
    a = p.parse_args(argv)
    from scripts.neral_v6_suite_intake import _read_private_jsonl
    v6 = build_experiment()[0]
    v5 = [json.loads(l) for l in _generated(V5_SEED)["train"].decode("utf8").splitlines()]
    _, pairs = _read_private_jsonl(a.verified_pairs, 8_000_000)
    protected = []
    for file in a.protected_jsonl:
        _, rows = _read_private_jsonl(file, 8_000_000)
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("prompt"), str):
                p.error("Protected private suites must contain prompt strings")
            protected.append(row["prompt"])
    candidate, manifest = build_candidate(v6, v5, pairs,
                                          protected_prompts=protected, seed=a.seed)
    export_candidate(a.out, candidate, manifest)
    print(json.dumps({key: manifest[key] for key in
        ("status", "total", "source_counts", "candidate_sha256", "training_allowed")},
        sort_keys=True))


if __name__ == "__main__":
    main()
