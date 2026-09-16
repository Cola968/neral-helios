"""Benchmark-only hotfix: V6 emits a training metric every TEN updates, not every update.

Keep the production trainer, checkpoint cadence and continuity untouched. Gate A measures
100 updates and excludes windows ending during warmup; Gate B measures full 10-update
windows after the first 30 updates. Exact-match patch anchors fail closed on drift.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def change(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if text.count(old) != 1:
        raise RuntimeError(f"V8.2 metrics fix anchor mismatch ({label}): {text.count(old)}")
    return text.replace(old, new, 1)


def patch_gate(text: str) -> str:
    text = change(text,
                  "    if not 16 <= args.updates <= 60:\n        raise ValueError(\"updates must be 16..60\")",
                  "    if not 80 <= args.updates <= 180:\n        raise ValueError(\"updates must be 80..180 for 10-update telemetry windows\")",
                  "benchmark duration")
    text = change(text,
                  "    if not 3 <= args.warmup_updates < args.updates - 5:\n",
                  "    if not 10 <= args.warmup_updates < args.updates - 30:\n",
                  "warmup bounds")
    text = change(text,
                  "            if len(train) != args.updates:\n                raise RuntimeError(f\"{name}: expected {args.updates} train rows, got {len(train)}\")\n            stable_rows = train[args.warmup_updates:]\n",
                  "            # Fused V6 records one aggregate metric per 10 updates (plus endpoint).\n            if len(train) < 8 or int(train[-1].get('step', -1)) != args.updates:\n                raise RuntimeError(f\"{name}: incomplete telemetry; expected endpoint step {args.updates}, got {len(train)} rows\")\n            stable_rows = [r for r in train if int(r.get('step', 0)) > args.warmup_updates\n                           and int(r.get('tokens_per_update', 0)) == TOKENS_PER_UPDATE]\n            if len(stable_rows) < 6:\n                raise RuntimeError(f\"{name}: too few post-warmup full metric windows: {len(stable_rows)}\")\n",
                  "gate A telemetry sampling")
    text = change(text,
                  "                \"checkpoint_sha256\": summary.get(\"checkpoint_sha256\"),\n                \"losses\": losses,\n",
                  "                \"checkpoint_sha256\": summary.get(\"checkpoint_sha256\"),\n                \"metric_steps\": [int(x.get('step', -1)) for x in train],\n                \"losses\": losses,\n",
                  "metric identities")
    text = change(text,
                  "        if len(losses) != len(plain_losses):\n",
                  "        if len(losses) != len(plain_losses) or row['metric_steps'] != plain['metric_steps']:\n",
                  "aligned loss windows")
    return text


def patch_soak(text: str) -> str:
    text = change(text,
                  "        stable_rows = full[10:]\n        if len(stable_rows) < 50:\n            raise RuntimeError(f\"too few post-warmup full updates: {len(stable_rows)}\")\n",
                  "        # V6 emits 10-update aggregate metrics, so 5M tokens yields ~20 rows.\n        stable_rows = [r for r in full if int(r.get('step', 0)) > 30]\n        if len(stable_rows) < 12:\n            raise RuntimeError(f\"too few post-warmup full metric windows: {len(stable_rows)}\")\n",
                  "gate B telemetry sampling")
    return text


def main() -> None:
    for name, patch in (("v8_2_throughput_gate.py", patch_gate),
                        ("v8_2_thermal_soak.py", patch_soak)):
        path = ROOT / name
        original = path.read_text(encoding="utf-8")
        updated = patch(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
        print(f"V8.2 metric cadence fix verified: {name}", flush=True)


if __name__ == "__main__":
    main()
