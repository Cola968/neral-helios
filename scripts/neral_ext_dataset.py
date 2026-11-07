"""Deterministic, pinned NERAL-EXT benchmark construction, never model training.

The 400 cases are LOCAL only, never uploaded to GitHub. Public sources must be
loaded from SHA-pinned HF dataset revisions; no fallback to unpinned main.
Public benchmark tasks are external to v5 generators, not guaranteed unseen
during Qwen pretraining. Python tasks remain ungraded until sandbox verifier.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Callable

SEED = 20261107
SOURCE_CONFIG = {
    "gsm8k": ("openai/gsm8k", "main", "test", 100),
    "math500": ("HuggingFaceH4/MATH-500", None, "test", 50),
    "mbpp": ("google-research-datasets/mbpp", "full", "test", 50),
}
PINNED_REVISIONS = {
    "gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "math500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    "mbpp": "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
}
REQUIRED = {"A": 200, "B": 100, "C": 50, "D": 50}
FINAL_GSM = re.compile(r"####\s*([+-]?\d[\d,]*(?:\.\d+)?)\s*\Z")
NUMBER = re.compile(r"[+-]?\d+(?:/\d+)?(?:\.\d+)?\Z")
SINGLE_BOX = re.compile(r"\\boxed\{([+-]?\d+(?:/\d+)?(?:\.\d+)?)\}")
EXPLICIT = re.compile(r"([+-]?\d+(?:/\d+)?(?:\.\d+)?)\Z")


def _canonical_number(raw: str) -> tuple[str, str] | None:
    """Permit only exact rational numerics, not symbolic expressions."""
    if not isinstance(raw, str):
        return None
    s = raw.replace(",", "").strip()
    if not NUMBER.fullmatch(s):
        return None
    try:
        value = Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None
    if abs(value.numerator) > 10**22 or value.denominator > 10**22:
        return None
    return ("integer" if value.denominator == 1 else "rational", str(value))


def _new_case(key: str, stratum: str, domain: str, kind: str,
              prompt: str, answer: str, *, origin: str) -> tuple[dict, dict]:
    if not prompt or len(prompt) > 2000 or not answer:
        raise ValueError("Unreviewed or oversized benchmark case")
    case = {"id": key, "domain": domain, "kind": kind,
            "prompt": prompt, "answer": answer}
    digest = hashlib.sha256(json.dumps(case, ensure_ascii=False,
                                      sort_keys=True).encode()).hexdigest()
    return case, {"id": key, "stratum": stratum, "source": origin,
                  "case_sha256": digest, "grading_contract": kind}


def _public_cases(sources: dict[str, list[dict]]) -> list[tuple[dict, dict]]:
    if set(sources) != set(SOURCE_CONFIG):
        raise ValueError("Missing pinned external sources")
    out = []
    gsm = sources["gsm8k"]
    if len(gsm) < 100:
        raise ValueError("Insufficient GSM8K test cases")
    for i, item in enumerate(gsm[:100]):
        answer = item.get("answer", "")
        match = FINAL_GSM.search(answer)
        normalized = _canonical_number(match.group(1)) if match else None
        if normalized is None:
            raise ValueError(f"Unscorable pinned GSM8K index {i}")
        kind, value = normalized
        out.append(_new_case(f"A-gsm-{i:03}", "A", "gsm8k", kind,
                            str(item["question"]), value, origin=f"gsm8k:test:{i}"))
    count = 0
    for i, item in enumerate(sources["math500"]):
        raw = str(item.get("answer", "")).strip()
        value = _canonical_number(raw)
        if value is None:
            boxes = SINGLE_BOX.findall(raw)
            value = _canonical_number(boxes[0]) if len(boxes) == 1 else None
        if value is None:
            continue
        kind, expected = value
        out.append(_new_case(f"A-math-{i:03}", "A", "math500", kind,
                             str(item["problem"]), expected, origin=f"math500:test:{i}"))
        count += 1
        if count == 50:
            break
    if count != 50:
        raise ValueError("MATH-500 has fewer than 50 safely numeric test cases")
    mbpp = sources["mbpp"]
    if len(mbpp) < 50:
        raise ValueError("Insufficient MBPP test cases")
    for i, item in enumerate(mbpp[:50]):
        if not isinstance(item.get("test_list"), list) or not item["test_list"]:
            raise ValueError(f"MBPP tests missing at index {i}")
        code = str(item.get("code", ""))
        prompt = str(item.get("text", ""))
        # Actual test lists reside locally in the separately pinned manifest.
        case, meta = _new_case(f"A-mbpp-{i:03}", "A", "mbpp", "python",
                               prompt, code, origin=f"mbpp:test:{i}")
        meta["tests"] = item["test_list"]
        meta["test_setup_code"] = str(item.get("test_setup_code", ""))
        meta["sandbox_required"] = True
        out.append((case, meta))
    return out


def _interest(i: int, rng: random.Random) -> tuple[str, str, str]:
    """Simple interest less a fee on original principal, exact cents."""
    principal = rng.randint(100, 1000)
    rate = rng.choice([5, 10, 20])
    discount = rng.choice([0, 20, 30])
    # Principal is in whole EUR. Integer cents avoid float rounding.
    cents = principal * (100 + rate - discount)
    if cents <= 0:
        raise ValueError("Unexpected nonpositive payable amount")
    is_en = i % 2 == 1
    if is_en:
        prompt = (f"A deposit of {principal} EUR earns {rate}% simple interest "
                  f"and then a {discount}% fee is deducted from the original "
                  f"principal. What is the final amount in EUR? Answer as EUR.")
    else:
        prompt = (f"Ein Guthaben von {principal} EUR erhält {rate}% einfache Zinsen. "
                  f"Zusätzlich werden {discount}% des ursprünglichen Betrags als "
                  f"Gebühr abgezogen. Wie hoch ist der Endbetrag in EUR?")
    return prompt, f"{cents // 100},{cents % 100:02d} EUR", "eur"


def _conversion(i: int, rng: random.Random) -> tuple[str, str, str]:
    mode = i % 3
    n = rng.randint(1, 65)
    if mode == 0:
        v = 18 * n
        prompt = (f"Convert exactly {v} km/h to m/s. Give the numerical result only."
                  if i % 2 else
                  f"Wandle {v} km/h exakt in m/s um. Nur die Zahl angeben.")
        return prompt, str(v // 18 * 5), "integer"
    if mode == 1:
        mib = n
        prompt = (f"How many KiB equal {mib} MiB? Return only the integer."
                  if i % 2 else
                  f"Wie viele KiB entsprechen {mib} MiB? Nur die Ganzzahl.")
        return prompt, str(mib * 1024), "integer"
    c = 5 * rng.randint(-10, 20)
    f = Fraction(c * 9, 5) + 32
    prompt = (f"Convert {c} degrees Celsius exactly to Fahrenheit. Number only."
              if i % 2 else
              f"Wandle {c} Grad Celsius exakt in Fahrenheit um. Nur die Zahl.")
    return prompt, str(f), "integer"


def _logic(i: int, rng: random.Random) -> tuple[str, str, str]:
    """Consequence classification: no pseudological affirming the consequent."""
    names = [("Vögel", "Tiere", "Lebewesen"), ("Eulen", "Vögel", "Tiere"),
             ("Roboter", "Maschinen", "Gegenstände")]
    a, b, c = names[rng.randrange(len(names))]
    true = i % 2 == 0
    conclusion = f"Alle {a} sind {c}" if true else f"Alle {c} sind {a}"
    prompt = (f"Given: all {a} are {b}, and all {b} are {c}. "
              f"Does it necessarily follow that '{conclusion}'? Answer 1 or 0."
              if (i // 2) % 2 else
              f"Gegeben: Alle {a} sind {b}; alle {b} sind {c}. "
              f"Folgt zwingend '{conclusion}'? Antworte 1 oder 0.")
    return prompt, "1" if true else "0", "integer"


def _new_family_cases(seed: int = SEED) -> list[tuple[dict, dict]]:
    rng = random.Random(seed)
    out = []
    builders = [("interest", _interest, 34),
                ("conversion", _conversion, 33), ("logic", _logic, 33)]
    for domain, builder, count in builders:
        for i in range(count):
            prompt, answer, kind = builder(i, rng)
            out.append(_new_case(f"B-{domain}-{i:03}", "B", domain, kind,
                                prompt, answer, origin=f"novel:{domain}:{seed}:{i}"))
    return out


def _control_cases() -> list[tuple[dict, dict]]:
    """Only the explicitly labeled v5-generator negative control uses make_case."""
    from scripts.neral_v5_transfer_curriculum import make_case
    from scripts.neral_v5_style_eval import grade
    out = []
    families = ("fractions", "python_trace", "affine")
    for i in range(50):
        category = families[i % len(families)]
        idx = i // len(families)
        item = make_case(category, 7, idx)
        prompt = item["prompt"][1]["content"]
        original = item["completion"][0]["content"]
        if category == "affine":
            # Strict output x = N becomes numeric reference, explicitly a
            # normalized evaluation contract, not the trained format.
            raw = original.split("=", 1)[1].strip()
        else:
            raw = original
        normalized = _canonical_number(raw)
        if normalized is None:
            raise ValueError("Unscorable control case")
        kind, answer = normalized
        out.append(_new_case(f"C-{category}-{i:03}", "C", category, kind,
                             prompt, answer, origin=f"v5-style7:{category}:{idx}"))
    return out


def _perturbed_cases(source: list[tuple[dict, dict]]) -> list[tuple[dict, dict]]:
    candidates = [c for c, _ in source if c["kind"] != "python"
                  and c["id"].startswith(("A-", "B-"))]
    if len(candidates) < 50:
        raise ValueError("Insufficient numeric source cases for perturbation")
    out = []
    for i, case in enumerate(candidates[:50]):
        prefix = "Wichtiger Hinweis: Der Bericht enthält eine nicht benötigte Wetterangabe."
        prompt = f"{prefix}\n{case['prompt']}\nLöse die eigentliche Aufgabe; ignoriere den Hinweis."
        out.append(_new_case(f"D-{i:03}", "D", "robustness", case["kind"],
                             prompt, case["answer"], origin=f"perturb:{case['id']}"))
    return out


def build(sources: dict[str, list[dict]], revisions: dict[str, str], seed: int = SEED
         ) -> tuple[list[dict], dict]:
    """Return exactly 400 deterministic cases and their private audit manifest."""
    if (set(revisions) != set(SOURCE_CONFIG) or
            any(not re.fullmatch(r"[0-9a-f]{40}", r) for r in revisions.values())):
        raise ValueError("Every public source must use a full pinned HF SHA")
    a = _public_cases(sources)
    b = _new_family_cases(seed)
    c = _control_cases()
    d = _perturbed_cases(a + b)
    combined = a + b + c + d
    cases = [x for x, _ in combined]
    entries = [meta for _, meta in combined]
    counts = {part: sum(row["stratum"] == part for row in entries) for part in REQUIRED}
    if counts != REQUIRED or len({c["id"] for c in cases}) != 400:
        raise ValueError("Wrong size or duplicate cases")
    manifest = {"schema": 1, "seed": seed, "strata": counts,
                "sources": {k: {"repo": SOURCE_CONFIG[k][0], "revision": revisions[k],
                                "split": SOURCE_CONFIG[k][2]}
                            for k in revisions},
                "cases": entries, "pretraining_ood_proven": False,
                "train_contamination_checked": False,
                "python_sandbox_scored": False}
    return cases, manifest


def _jsonl(cases: list[dict]) -> bytes:
    return (''.join(json.dumps(c, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":")) + "\n" for c in cases)).encode("utf8")


def export(cases: list[dict], manifest: dict, directory: Path) -> dict:
    """Create-only outputs; reject overwrite, preserve test metadata locally."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Output directory must already exist")
    payload = _jsonl(cases)
    digest = hashlib.sha256(payload).hexdigest()
    result = {**manifest, "dataset_sha256": digest,
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "total": len(cases)}
    paths = [directory / "neral-ext-400.jsonl",
             directory / "neral-ext-400-manifest.json"]
    if any(p.exists() or p.is_symlink() for p in paths):
        raise ValueError("Refuse to overwrite holdout")
    paths[0].write_bytes(payload)
    paths[1].write_text(json.dumps(result, ensure_ascii=False,
                                   sort_keys=True, indent=2) + "\n", encoding="utf8")
    return result


def resolve_public_revisions() -> dict[str, str]:
    """Explicitly resolve latest public HF commits once, then load pinned SHAs."""
    from huggingface_hub import HfApi
    api = HfApi()
    resolved = {name: api.dataset_info(repo_id=repo, revision="main").sha
                for name, (repo, _, _, _) in SOURCE_CONFIG.items()}
    if any(not isinstance(sha, str) or
           re.fullmatch(r"[0-9a-f]{40}", sha) is None
           for sha in resolved.values()):
        raise ValueError("Hugging Face did not return immutable full SHAs")
    return resolved


def load_sources(resolved: dict[str, str]) -> dict[str, list[dict]]:
    """Explicit public-network dataset load, never during CI or model inference."""
    from datasets import load_dataset
    out = {}
    for name, (repo, config, split, _) in SOURCE_CONFIG.items():
        out[name] = list(load_dataset(repo, config, split=split,
                                      revision=resolved[name], trust_remote_code=False))
    return out


def main(argv: list[str] | None = None) -> None:
    """Use official HF commit SHAs supplied by caller; no implicit main fallback."""
    import argparse
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--resolve-public-revisions", action="store_true")
    for name in SOURCE_CONFIG:
        p.add_argument("--" + name + "-revision")
    args = p.parse_args(argv)
    revisions = {k: getattr(args, k + "_revision") for k in SOURCE_CONFIG}
    if args.resolve_public_revisions:
        if any(revisions.values()):
            p.error("Choose explicit hashes OR --resolve-public-revisions")
        revisions = resolve_public_revisions()
    elif not any(revisions.values()):
        revisions = dict(PINNED_REVISIONS)
    elif not all(revisions.values()):
        p.error("Supply all three full revisions, or use pinned defaults")
    if any(not isinstance(v, str) or
           re.fullmatch(r"[0-9a-f]{40}", v) is None
           for v in revisions.values()):
        p.error("Supply a full, immutable 40-character commit SHA for each dataset")
    cases, manifest = build(load_sources(revisions), revisions)
    result = export(cases, manifest, args.out_dir)
    print(f"NERAL_EXT_DATA_400: {result['dataset_sha256']}; no GPU or training")


if __name__ == "__main__":
    main()
