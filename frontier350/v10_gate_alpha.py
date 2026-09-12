"""Combine unchanged V9 development gates for one V10 alpha candidate."""
import argparse
import json
import math
from pathlib import Path

from recovery_identity import file_sha

BASELINE_DIR = Path(__file__).parent / "v10_baseline_evidence"
CANDIDATE_TOKENS = 1_008_003_584


def load(path):
    obj = json.loads(Path(path).read_text(encoding="utf8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def strict_bilingual(base, candidate, dev):
    for obj in (base, candidate):
        if obj.get("cases") != 400 or set(obj.get("directions", {})) != {"de_to_en", "en_to_de"}:
            raise ValueError("incomplete bilingual evidence")
        total = correct = 0
        for direction in obj["directions"].values():
            n, k = direction["total"], direction["correct"]
            if type(n) is not int or type(k) is not int or n != 200 or not 0 <= k <= n:
                raise ValueError("invalid bilingual counts")
            pct = direction["accuracy_pct"]
            if isinstance(pct, bool) or not math.isfinite(float(pct)) or abs(float(pct) - 100 * k / n) > 1e-9:
                raise ValueError("invalid bilingual percentage")
            total += n
            correct += k
        if abs(float(obj["overall_accuracy_pct"]) - 100 * correct / total) > 1e-9:
            raise ValueError("inconsistent bilingual total")
    for key in ("cases_sha256", "tokenizer_sha256", "protocol_sha256"):
        if not base.get(key) or base[key] != candidate.get(key):
            raise ValueError("bilingual protocol/data changed")
    for key in ("checkpoint_sha256", "checkpoint_processed_tokens", "tokenizer_sha256"):
        if candidate[key] != dev[key]:
            raise ValueError("mixed candidate checkpoint evidence")
    return candidate["overall_accuracy_pct"] >= base["overall_accuracy_pct"] and all(
        candidate["directions"][d]["accuracy_pct"] >= base["directions"][d]["accuracy_pct"]
        for d in base["directions"]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--alpha", required=True, type=float)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--bilingual", required=True)
    ap.add_argument("--legacy-gate", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    checkpoint_sha = file_sha(args.checkpoint)
    dev = load(args.dev)
    val = load(args.validation)
    bilingual = load(args.bilingual)
    legacy = load(args.legacy_gate)
    base_dev = load(BASELINE_DIR / "v9_anchor_dev.json")
    base_val = load(BASELINE_DIR / "v9_anchor_validation.json")
    base_bilingual = load(BASELINE_DIR / "v9_anchor_bilingual.json")

    for report in (dev, val, bilingual):
        if report.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("candidate report does not match staged checkpoint")
        if int(report.get("checkpoint_processed_tokens", -1)) != CANDIDATE_TOKENS:
            raise ValueError("candidate report has unexpected token count")

    if len(legacy.get("candidates", [])) != 1:
        raise ValueError("legacy gate must describe exactly one candidate")
    bilingual_pass = strict_bilingual(base_bilingual, bilingual, dev)
    improved = (
        val["validation_loss"] < base_val["validation_loss"]
        or bilingual["overall_accuracy_pct"] > base_bilingual["overall_accuracy_pct"]
        or any(
            dev["categories"][cat]["accuracy_pct"] > base_dev["categories"][cat]["accuracy_pct"]
            for cat in base_dev["categories"]
        )
    )
    passed = bool(legacy.get("selection_allowed")) and bilingual_pass and improved
    result = {
        "schema": 1,
        "alpha": args.alpha,
        "tag": "v10_a" + str(args.alpha).replace(".", "p"),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_processed_tokens": CANDIDATE_TOKENS,
        "legacy_gate": bool(legacy.get("selection_allowed")),
        "legacy_failures": legacy["candidates"][0].get("failures", []),
        "bilingual_gate": bilingual_pass,
        "strict_improvement": improved,
        "pass": passed,
        "validation_loss": val["validation_loss"],
        "overall_pct": dev["overall_accuracy_pct"],
        "categories": {k: v["accuracy_pct"] for k, v in dev["categories"].items()},
        "bilingual_pct": bilingual["overall_accuracy_pct"],
        "de_to_en_pct": bilingual["directions"]["de_to_en"]["accuracy_pct"],
        "en_to_de_pct": bilingual["directions"]["en_to_de"]["accuracy_pct"],
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
