from __future__ import annotations

"""Pure-CPU contracts for the next Neral EOS + math repair lane.

This module does not train a model. It defines deterministic data balancing and hard development
criteria that a future GPU repair stage must satisfy. It intentionally keeps free-generation EOS,
teacher-forced terminal EOS and repetition as separate signals so a candidate cannot pass by
learning only one termination proxy.
"""

from dataclasses import asdict, dataclass
import random
import re


@dataclass(frozen=True)
class EOSRepairPolicy:
    min_free_eos_gain_pp: float = 5.0
    min_terminal_eos_top1_gain_pp: float = 5.0
    min_free_eos_pct: float = 40.0
    max_repetition_regression_pp: float = 0.50
    min_utf8_pct: float = 99.0
    min_mean_generated_tokens: float = 5.0
    min_short_answer_share: float = 0.35
    max_answer_bytes: int = 384


POLICY = EOSRepairPolicy()

# Loss/objective planning only. Actual promotion still requires measured held-out evidence.
E3_OBJECTIVE_MIX = {
    "screened_human_sft": 0.35,
    "rawmath_id": 0.20,
    "rawmath_transfer": 0.25,
    "base_language_replay": 0.20,
}

# Data-lane mix for math specialization. Transfer/OOD outweigh pure ID memorization.
MATH_REPAIR_DATA_MIX = {
    "math_id": 0.25,
    "math_transfer": 0.35,
    "math_ood_physics": 0.20,
    "human_math_dialogue": 0.10,
    "general_language_replay": 0.10,
}

_DIALOGUE_MARKER = re.compile(r"(?im)^\s*(user|assistant|human|prompter)\s*:")


def answer_length_bucket(answer: str) -> str:
    n = len(answer.strip().encode("utf-8"))
    if n <= 64:
        return "short"
    if n <= 192:
        return "medium"
    return "long"


def validate_eos_row(row: dict, *, max_answer_bytes: int = POLICY.max_answer_bytes) -> dict:
    prompt = row.get("prompt")
    answer = row.get("answer")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("EOS row requires non-empty prompt")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("EOS row requires non-empty answer")
    try:
        raw = answer.strip().encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("EOS answer has invalid Unicode") from exc
    if not 2 <= len(raw) <= max_answer_bytes:
        raise ValueError("EOS answer length outside admitted range")
    if "\ufffd" in answer:
        raise ValueError("EOS answer contains replacement character")
    if _DIALOGUE_MARKER.search(answer):
        raise ValueError("EOS answer contains embedded dialogue-turn marker")
    language = row.get("language", "unknown")
    domain = row.get("domain", "general")
    if not isinstance(language, str) or not isinstance(domain, str):
        raise ValueError("EOS row language/domain must be strings")
    return {
        "bucket": answer_length_bucket(answer),
        "answer_bytes": len(raw),
        "language": language,
        "domain": domain,
    }


def balance_eos_rows(rows: list[dict], *, max_examples: int, seed: int = 968_780) -> list[dict]:
    """Deterministically bias SFT toward short/medium answers without duplicating rows.

    Short answers expose the model to many clean terminal boundaries per token. Long answers remain
    represented so EOS repair does not collapse into premature stopping. No row is fabricated.
    """
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    buckets = {"short": [], "medium": [], "long": []}
    for row in rows:
        meta = validate_eos_row(row)
        buckets[meta["bucket"]].append(row)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)

    target = min(max_examples, sum(len(v) for v in buckets.values()))
    quotas = {
        "short": round(target * 0.40),
        "medium": round(target * 0.40),
    }
    quotas["long"] = max(0, target - quotas["short"] - quotas["medium"])

    picked: list[dict] = []
    leftovers: list[dict] = []
    for name in ("short", "medium", "long"):
        take = min(quotas[name], len(buckets[name]))
        picked.extend(buckets[name][:take])
        leftovers.extend(buckets[name][take:])
    rng.shuffle(leftovers)
    picked.extend(leftovers[: max(0, target - len(picked))])
    rng.shuffle(picked)
    return picked


def eos_bucket_report(rows: list[dict]) -> dict:
    counts = {"short": 0, "medium": 0, "long": 0}
    for row in rows:
        counts[validate_eos_row(row)["bucket"]] += 1
    total = max(1, sum(counts.values()))
    return {
        "examples": sum(counts.values()),
        "counts": counts,
        "shares": {k: v / total for k, v in counts.items()},
    }


def assess_eos_candidate(anchor: dict, candidate: dict, policy: EOSRepairPolicy = POLICY) -> dict:
    required = {
        "free_eos_pct",
        "terminal_eos_top1_pct",
        "repeated_trigram_pct",
        "utf8_pct",
        "mean_generated_tokens",
    }
    missing = sorted(required - set(anchor) | (required - set(candidate)))
    if missing:
        return {"accepted": False, "missing": missing, "blockers": ["missing:" + x for x in missing]}

    free_gain = float(candidate["free_eos_pct"]) - float(anchor["free_eos_pct"])
    top1_gain = float(candidate["terminal_eos_top1_pct"]) - float(anchor["terminal_eos_top1_pct"])
    repetition_regression = float(candidate["repeated_trigram_pct"]) - float(anchor["repeated_trigram_pct"])
    blockers: list[str] = []
    if free_gain < policy.min_free_eos_gain_pp:
        blockers.append("free_eos_gain_too_small")
    if top1_gain < policy.min_terminal_eos_top1_gain_pp:
        blockers.append("terminal_eos_top1_gain_too_small")
    if float(candidate["free_eos_pct"]) < policy.min_free_eos_pct:
        blockers.append("free_eos_below_floor")
    if repetition_regression > policy.max_repetition_regression_pp:
        blockers.append("repetition_regressed")
    if float(candidate["utf8_pct"]) < policy.min_utf8_pct:
        blockers.append("utf8_below_floor")
    if float(candidate["mean_generated_tokens"]) < policy.min_mean_generated_tokens:
        blockers.append("premature_stop_risk")
    return {
        "accepted": not blockers,
        "blockers": sorted(blockers),
        "free_eos_gain_pp": free_gain,
        "terminal_eos_top1_gain_pp": top1_gain,
        "repetition_regression_pp": repetition_regression,
        "policy": asdict(policy),
    }


def validate_policy() -> dict:
    if abs(sum(E3_OBJECTIVE_MIX.values()) - 1.0) > 1e-9:
        raise ValueError("E3 objective mix must sum to 1")
    if abs(sum(MATH_REPAIR_DATA_MIX.values()) - 1.0) > 1e-9:
        raise ValueError("math repair data mix must sum to 1")
    if E3_OBJECTIVE_MIX["rawmath_transfer"] <= E3_OBJECTIVE_MIX["rawmath_id"]:
        raise ValueError("transfer objective must exceed ID objective")
    if MATH_REPAIR_DATA_MIX["math_transfer"] <= MATH_REPAIR_DATA_MIX["math_id"]:
        raise ValueError("math transfer data must exceed ID math")
    return {
        "policy": asdict(POLICY),
        "e3_objective_mix": E3_OBJECTIVE_MIX,
        "math_repair_data_mix": MATH_REPAIR_DATA_MIX,
        "principle": "EOS improvement must appear in free generation and terminal-EOS top1 without repetition or premature-stop regression; math transfer/OOD outweigh ID memorization.",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(validate_policy(), indent=2))
