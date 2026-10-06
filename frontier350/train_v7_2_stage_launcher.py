from __future__ import annotations

"""V7.2 stage launcher: gated Fast2 first, proven V7.1/V5 fallbacks preserved.

Fast2 is materialized as a temporary trainer copy. Execution-profile fallbacks may
resume the newest discovered exact recovery candidate, but NEVER delete or silently
rewind a populated output directory if a recovery candidate is unavailable.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from apply_v6_fast2_patch import patch_trainer as patch_fast2
from nextgen_v1.fallback_guard import fallback_allowed
from train_v7_1_stage_launcher import arg_value, profile_args
from v7_1_runtime_hotfix import patch_trainer as patch_runtime
from v7_recovery import list_recovery_states

ROOT = Path(__file__).resolve().parent
TRAINER = ROOT / "train_guarded_v9.py"
FAST2_TRAINER = ROOT / "_v72_fast2_train_guarded_v9.py"


def run_one(trainer: Path, args: list[str], label: str, recovery: Path | None) -> int:
    print(json.dumps({
        "event": "v7_2_execution_profile",
        "profile": label,
        "resume_recovery": str(recovery) if recovery else None,
    }), flush=True)
    return int(subprocess.run([sys.executable, "-u", str(trainer), *args]).returncode)


def main() -> None:
    base_args = sys.argv[1:]
    out_value = arg_value(base_args, "--out")
    if not out_value:
        raise SystemExit("missing --out")
    out = Path(out_value)
    replay_only = arg_value(base_args, "--mode") == "replay_only"
    input_profile = os.environ.get("NERAL_V71_INPUT_PROFILE", "off").strip().lower()
    use_fast2 = os.environ.get("NERAL_V72_FAST2", "0").strip().lower() in ("1", "true", "yes")
    if input_profile not in ("off", "coalesced-pin", "coalesced-pageable"):
        raise SystemExit(f"invalid NERAL_V71_INPUT_PROFILE={input_profile!r}")

    # Keep the existing V7/V7.1 fallback implementation and separate Fast2 file.
    base_text = TRAINER.read_text(encoding="utf-8")
    fallback_text = patch_runtime(base_text)
    TRAINER.write_text(fallback_text, encoding="utf-8")
    if use_fast2 and replay_only:
        fast2_text = patch_runtime(patch_fast2(base_text))
        FAST2_TRAINER.write_text(fast2_text, encoding="utf-8")

    legacy = (("fast3", "turbo60", "turbo50", "safe") if input_profile != "off"
              else ("turbo60", "turbo50", "safe"))
    profiles = (("fast2",) + legacy) if use_fast2 and replay_only else (legacy if replay_only else ("safe",))

    last_code = 1
    try:
        for attempt, profile in enumerate(profiles, 1):
            recoveries = list_recovery_states(out) if out.exists() else []
            recovery = recoveries[0] if recoveries else None
            if attempt > 1 and not fallback_allowed(out, recovery):
                print(json.dumps({"event": "v7_2_fallback_blocked",
                                  "reason": "no_usable_exact_recovery_and_output_not_empty",
                                  "preserved_output": str(out)}), flush=True)
                raise SystemExit(last_code)

            if profile == "fast2":
                args = profile_args(base_args, "turbo50", recovery, "off")
                trainer = FAST2_TRAINER
            else:
                args = profile_args(base_args, profile, recovery, input_profile)
                trainer = TRAINER

            last_code = run_one(trainer, args, profile, recovery)
            if last_code == 0:
                print(json.dumps({
                    "event": "v7_2_execution_profile_selected",
                    "profile": profile,
                    "resumed_from": str(recovery) if recovery else None,
                }), flush=True)
                raise SystemExit(0)

            recoveries = list_recovery_states(out) if out.exists() else []
            newest = recoveries[0] if recoveries else None
            next_fallback = attempt < len(profiles) and fallback_allowed(out, newest)
            print(json.dumps({
                "event": "v7_2_execution_profile_failed",
                "profile": profile,
                "returncode": last_code,
                "recovery_available": str(newest) if newest else None,
                "next_fallback": next_fallback,
            }), flush=True)
            if not next_fallback:
                raise SystemExit(last_code)
    finally:
        FAST2_TRAINER.unlink(missing_ok=True)

    raise SystemExit(last_code)


if __name__ == "__main__":
    main()
