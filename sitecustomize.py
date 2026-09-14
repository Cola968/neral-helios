from __future__ import annotations

"""GitHub Actions-only Scale V4 rollback cap and 5M-stage I/O optimization.

Fresh Scale V4 workflow checkouts are limited to 5M continuity stages. A hard reject
returns to the immediately previous qualified SAFE/SHADOW continuity anchor instead of
discarding the entire SHADOW trajectory. SAFE promotion gates are not changed.

For 5M stages the already-persisted continuity anchor is sufficient crash recovery, so
the trainer skips the redundant full checkpoint write before the first update. The final
stage checkpoint is still written and validated normally. This reduces disk/CPU stalls
without changing model math, data, replay weighting, LR, or promotion criteria.
"""

import os
from pathlib import Path


if os.environ.get("GITHUB_ACTIONS") == "true":
    root = Path(__file__).resolve().parent
    controller = root / "frontier350" / "run_scale_v4.py"
    trainer = root / "frontier350" / "train_guarded_v9.py"

    if controller.is_file():
        text = controller.read_text(encoding="utf-8")
        original = text

        for old_ladder in (
            "LADDER = (5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000, 250_000_000)",
            "LADDER = (5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000)",
        ):
            text = text.replace(old_ladder, "LADDER = (5_000_000,)")

        text = text.replace(
            '    block_index = int(state.get("block_index", 0))',
            '    block_index = 0  # hard cap: every new continuity stage is 5M tokens',
        )

        marker = "        round_no += 1\n        remaining = a.target_tokens - train_tokens"
        replacement = (
            "        round_no += 1\n"
            "        round_anchor, round_anchor_sha, round_anchor_tokens = train_anchor, train_sha, train_tokens\n"
            "        remaining = a.target_tokens - train_tokens"
        )
        if "round_anchor, round_anchor_sha, round_anchor_tokens" not in text:
            if marker not in text:
                raise RuntimeError("Scale V4 rollback cap: round marker not found")
            text = text.replace(marker, replacement, 1)

        old_reset = (
            "        # Hard regression: throw away shadow trajectory, return to immutable SAFE, and\n"
            "        # retry a smaller/more-retentive stage. No unsafe checkpoint is promoted.\n"
            "        hard_failures += 1\n"
            "        train_anchor, train_sha, train_tokens = safe_anchor, safe_sha, safe_tokens"
        )
        new_reset = (
            "        # Hard regression: discard only this 5M stage and return to the immediately\n"
            "        # previous qualified continuity anchor (SAFE or healthy SHADOW).\n"
            "        hard_failures += 1\n"
            "        train_anchor, train_sha, train_tokens = round_anchor, round_anchor_sha, round_anchor_tokens"
        )
        if old_reset in text:
            text = text.replace(old_reset, new_reset, 1)

        text = text.replace('"status": "HARD_REJECT_RESET_TO_SAFE"', '"status": "HARD_REJECT_RESET_TO_CONTINUITY"')

        if text != original:
            controller.write_text(text, encoding="utf-8")
            print("Scale V4 rollback cap applied: 5M stages; reject returns to prior continuity anchor")

    if trainer.is_file():
        text = trainer.read_text(encoding="utf-8")
        original = text
        # The generic recovery patch makes 5M stages write their normal full resume.pt at
        # completion. Avoid a second full-state write at token zero; the input anchor is
        # already durable and is exactly where a crash before completion should return.
        text = text.replace(
            "    if not args.resume:save()",
            "    if not args.resume and args.new_tokens > 5_000_000:save()",
            1,
        )
        if text != original:
            trainer.write_text(text, encoding="utf-8")
            print("Scale V4 speed patch applied: skipped redundant initial full save for 5M stages")
