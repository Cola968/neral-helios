from __future__ import annotations

"""Evidence-gated Phase-E scoring for the Neral 600M architecture search.

The weighted score requested by the project brief is only emitted when both a real
RTX speed report and matched learning-pilot summaries are present.  It never invents
missing hardware or quality data. Long-context and inference terms use explicitly
labelled analytical proxies until separate long-sequence measurements are supplied.
"""

import argparse
import json
import math
from pathlib import Path

from architecture_search600 import CANDIDATES, PHASE_C, WEIGHTS


def minmax(values: dict[str, float], *, higher_is_better: bool) -> dict[str, float]:
    lo, hi = min(values.values()), max(values.values())
    if math.isclose(lo, hi, rel_tol=1e-12, abs_tol=1e-12):
        return {k: 1.0 for k in values}
    if higher_is_better:
        return {k: (v - lo) / (hi - lo) for k, v in values.items()}
    return {k: (hi - v) / (hi - lo) for k, v in values.items()}


def load_speed(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf8"))
    rows = data.get("rows", [])
    best: dict[str, dict] = {}
    for row in rows:
        if row.get("oom") or not row.get("finite", False):
            continue
        cid = row.get("candidate")
        if cid not in CANDIDATES:
            continue
        # Compare the fastest *successful* profile for each architecture.
        if cid not in best or row["tokens_per_second"] > best[cid]["tokens_per_second"]:
            best[cid] = row
    return best


def load_pilots(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf8"))
    return {r["candidate"]: r for r in rows if r.get("candidate") in CANDIDATES}


def kv_cache_bytes(cid: str, seq: int = 8192, batch: int = 1, bytes_per_value: int = 2) -> int:
    c = CANDIDATES[cid]
    # K + V for every layer, BF16 inference cache.
    return batch * c.n_layers * 2 * c.n_kv_heads * seq * c.head_dim * bytes_per_value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", required=True)
    ap.add_argument("--pilots", required=True)
    ap.add_argument("--out", default="frontier600_report/phase_e_score.json")
    args = ap.parse_args()

    speed = load_speed(Path(args.speed))
    pilots = load_pilots(Path(args.pilots))
    eligible = [c for c in PHASE_C if c in speed and c in pilots]
    missing = {
        c: {"speed": c in speed, "pilot": c in pilots}
        for c in PHASE_C if c not in eligible
    }
    if len(eligible) < 2:
        raise RuntimeError(f"insufficient real evidence for scoring: {missing}")

    # Primary quality evidence: lower final/best held-out validation CE is better.
    val = {c: float(pilots[c]["best_validation_loss"]) for c in eligible}
    quality = minmax(val, higher_is_better=False)

    # Quality per parameter: compare the same normalized quality while lightly correcting
    # for exact parameter count. Counts differ by <1%, so this term cannot dominate by size noise.
    qp_raw = {c: (quality[c] + 1.0) / (CANDIDATES[c].parameter_breakdown()["total"] / 600_000_000)
              for c in eligible}
    quality_per_param = minmax(qp_raw, higher_is_better=True)

    tps = {c: float(speed[c]["tokens_per_second"]) for c in eligible}
    train_speed = minmax(tps, higher_is_better=True)

    vram_raw = {c: float(speed[c]["peak_allocated_gb"]) for c in eligible}
    vram = minmax(vram_raw, higher_is_better=False)

    # Explicit analytical proxy until a separate 8K execution gate exists: 50% KV cache,
    # 50% 8K forward FLOPs/token. Higher score means lower long-context cost.
    kv = {c: float(kv_cache_bytes(c)) for c in eligible}
    flops8k = {c: float(CANDIDATES[c].analytical_forward_flops_per_token(8192)) for c in eligible}
    kv_score = minmax(kv, higher_is_better=False)
    flops8k_score = minmax(flops8k, higher_is_better=False)
    long_context = {c: 0.5 * kv_score[c] + 0.5 * flops8k_score[c] for c in eligible}

    # Inference proxy: analytical forward FLOPs/token at seq=512 plus KV-cache size.
    flops512 = {c: float(CANDIDATES[c].analytical_forward_flops_per_token(512)) for c in eligible}
    f512_score = minmax(flops512, higher_is_better=False)
    inference = {c: 0.75 * f512_score[c] + 0.25 * kv_score[c] for c in eligible}

    scores = {}
    for c in eligible:
        components = {
            "model_quality_learning_curve": quality[c],
            "quality_per_parameter": quality_per_param[c],
            "training_speed": train_speed[c],
            "vram": vram[c],
            "long_context_efficiency": long_context[c],
            "inference_efficiency": inference[c],
        }
        total = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
        scores[c] = {
            "weighted_score": total,
            "components": components,
            "measured": {
                "best_validation_loss": val[c],
                "training_tokens_per_second": tps[c],
                "peak_allocated_gb": vram_raw[c],
            },
            "analytical_proxies": {
                "kv_cache_bytes_8k_bf16_batch1": int(kv[c]),
                "forward_flops_per_token_seq512": int(flops512[c]),
                "forward_flops_per_token_seq8192": int(flops8k[c]),
            },
        }

    ranking = sorted(eligible, key=lambda c: scores[c]["weighted_score"], reverse=True)
    payload = {
        "status": "provisional_after_short_pilot",
        "winner_is_final_architecture": False,
        "reason": "5M-token pilots can reject weak designs but close races require 25M/100M confirmation",
        "weights": WEIGHTS,
        "eligible": eligible,
        "missing": missing,
        "ranking": ranking,
        "scores": scores,
        "long_context_term": "analytical proxy until dedicated long-sequence hardware + learning test",
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
