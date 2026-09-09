from __future__ import annotations

"""Sample-audit future Hugging Face pretraining candidates for Neral-50M.

This script is deliberately non-promoting.  It records immutable revision, gating/license
metadata, schema/config resolution and a deterministic quality/contamination sample.  A green
technical audit does NOT change hf_source_policy50.py and therefore cannot make a source
trainable by itself.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import HfApi

import build_mobile_data_pack as core

PREFERENCES = {
    "HuggingFaceTB/finemath": ("finemath-3plus", "finemath-4plus", "default"),
    "open-web-math/open-web-math": ("default",),
    "HuggingFaceTB/stack-edu": ("default", "Python", "python"),
    "allenai/dolma": ("default",),
    "codeparrot/codeparrot-clean": ("default",),
}


def choose_config(dataset_id: str, revision: str) -> tuple[str | None, list[str]]:
    names = get_dataset_config_names(dataset_id, revision=revision)
    if not names:
        return None, []
    prefs = PREFERENCES.get(dataset_id, ("default",))
    for pref in prefs:
        if pref in names:
            return pref, names
    low = {n.lower(): n for n in names}
    for pref in prefs:
        if pref.lower() in low:
            return low[pref.lower()], names
    return names[0], names


def license_from_info(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is not None:
        try:
            value = card.get("license")
            if value:
                return str(value)
        except Exception:
            pass
    for tag in getattr(info, "tags", []) or []:
        if str(tag).startswith("license:"):
            return str(tag).split(":", 1)[1]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-rows", type=int, default=2500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    api = HfApi()
    info = api.dataset_info(args.dataset)
    revision = str(info.sha)
    gated = bool(getattr(info, "gated", False))
    license_id = license_from_info(info)
    report = {
        "dataset_id": args.dataset,
        "revision": revision,
        "gated": gated,
        "license": license_id,
        "automatic_training_admission": False,
        "sample_rows_requested": args.max_rows,
    }

    if gated:
        report.update({"status": "blocked_gated", "sampled_rows": 0})
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
        print(json.dumps(report, indent=2))
        return

    try:
        config, configs = choose_config(args.dataset, revision)
        report["config"] = config
        report["available_config_count"] = len(configs)
        benchmark_hashes = core.benchmark_shingles(Path(__file__).resolve().parent)
        kwargs = {"split": "train", "streaming": True, "revision": revision}
        ds = load_dataset(args.dataset, config, **kwargs) if config else load_dataset(args.dataset, **kwargs)

        reasons = Counter()
        exact_seen: set[str] = set()
        sampled = text_rows = clean = benchmark_hits = 0
        clean_bytes = 0
        clean_word_count = 0
        min_bytes = None
        max_bytes = 0

        for row in ds:
            sampled += 1
            text = core.row_text(row)
            if text is None:
                reasons["missing_text"] += 1
            else:
                text_rows += 1
                text = core.normalize(text)
                digest = core.digest_text(text)
                if digest in exact_seen:
                    reasons["sample_exact_duplicate"] += 1
                else:
                    exact_seen.add(digest)
                    reason = core.quality_reason(text, benchmark_hashes)
                    if reason:
                        reasons[reason] += 1
                        if reason == "benchmark_phrase":
                            benchmark_hits += 1
                    else:
                        b = len(text.encode("utf8"))
                        clean += 1
                        clean_bytes += b
                        clean_word_count += len(core.WORD_RE.findall(text))
                        min_bytes = b if min_bytes is None else min(min_bytes, b)
                        max_bytes = max(max_bytes, b)
            if sampled >= args.max_rows:
                break

        clean_rate = clean / max(1, text_rows)
        contamination_rate = benchmark_hits / max(1, text_rows)
        report.update({
            "status": "sampled",
            "sampled_rows": sampled,
            "text_rows": text_rows,
            "clean_rows": clean,
            "clean_rate": clean_rate,
            "benchmark_phrase_hits": benchmark_hits,
            "benchmark_phrase_rate": contamination_rate,
            "clean_utf8_bytes": clean_bytes,
            "mean_clean_bytes": clean_bytes / max(1, clean),
            "mean_clean_words": clean_word_count / max(1, clean),
            "min_clean_bytes": min_bytes,
            "max_clean_bytes": max_bytes,
            "rejections": dict(reasons),
            "technical_candidate": bool(
                license_id and clean_rate >= 0.60 and contamination_rate <= 0.002
            ),
            "note": "Technical candidate only; license/source-policy review is still mandatory.",
        })
    except Exception as exc:
        report.update({"status": "audit_error", "error_type": type(exc).__name__, "error": str(exc)[:1000]})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
