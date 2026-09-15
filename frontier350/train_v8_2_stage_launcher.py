from __future__ import annotations

"""Scale V8.2 execution launcher.

V8.2 combines the proven Fast2 fused objective with the V7.1 coalesced host-input path.
This is execution-only: sampled RNG order/token values, replay/main weighting, optimizer,
LR schedule, checkpoint/recovery state and quality gates are unchanged.

Fallback order deliberately isolates the new input optimization:
  fast2c -> fast2 -> proven legacy profiles
so a coalesced-input failure never forces a stage restart from an older continuity point.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from apply_v6_fast2_patch import patch_trainer as patch_fast2
from train_v7_1_stage_launcher import arg_value, profile_args
from v7_1_runtime_hotfix import patch_trainer as patch_runtime
from v7_recovery import list_recovery_states

ROOT = Path(__file__).resolve().parent
TRAINER = ROOT / "train_guarded_v9.py"
FAST2_TRAINER = ROOT / "_v82_fast2_train_guarded_v9.py"
_ALLOWED_INPUT = ("off", "coalesced-pin", "coalesced-pageable")


def execution_profiles(*, replay_only: bool, use_fast2: bool,
                       legacy_input: str, fast2_input: str) -> tuple[str, ...]:
    """Return the fail-closed execution order without touching training state."""
    if legacy_input not in _ALLOWED_INPUT:
        raise ValueError(f"invalid legacy input profile: {legacy_input!r}")
    if fast2_input not in _ALLOWED_INPUT:
        raise ValueError(f"invalid Fast2 input profile: {fast2_input!r}")
    if not replay_only:
        return ("safe",)

    legacy = (("fast3", "turbo60", "turbo50", "safe") if legacy_input != "off"
              else ("turbo60", "turbo50", "safe"))
    if not use_fast2:
        return legacy
    fast_profiles = (("fast2c", "fast2") if fast2_input != "off" else ("fast2",))
    return fast_profiles + legacy


def profile_command_args(base_args: list[str], profile: str, recovery: Path | None,
                         legacy_input: str, fast2_input: str) -> tuple[list[str], str]:
    """Build one profile's CLI and report the input profile actually selected."""
    if profile == "fast2c":
        args = profile_args(base_args, "fast3", recovery, fast2_input)
        return args, fast2_input
    if profile == "fast2":
        return profile_args(base_args, "turbo50", recovery, "off"), "off"
    args = profile_args(base_args, profile, recovery, legacy_input)
    return args, legacy_input if profile == "fast3" else "off"


def run_one(trainer: Path, args: list[str], label: str, recovery: Path | None,
            input_profile: str = "off") -> int:
    print(json.dumps({
        "event": "v8_2_execution_profile",
        "profile": label,
        "input_profile": input_profile,
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

    # Keep the old V7.1 fallback admission independent from the new Fast2 input path.
    legacy_input = os.environ.get("NERAL_V71_INPUT_PROFILE", "off").strip().lower()
    fast2_input = os.environ.get("NERAL_V82_FAST2_INPUT_PROFILE", "off").strip().lower()
    use_fast2 = os.environ.get("NERAL_V72_FAST2", "0").strip().lower() in ("1", "true", "yes")
    try:
        profiles = execution_profiles(
            replay_only=replay_only,
            use_fast2=use_fast2,
            legacy_input=legacy_input,
            fast2_input=fast2_input,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    base_text = TRAINER.read_text(encoding="utf-8")
    fallback_text = patch_runtime(base_text)
    TRAINER.write_text(fallback_text, encoding="utf-8")
    if use_fast2 and replay_only:
        # base_text already contains the V7.1 coalesced-input implementation because the
        # workflow applies apply_v7_1_speed_patch.py before routing through this launcher.
        fast2_text = patch_runtime(patch_fast2(base_text))
        FAST2_TRAINER.write_text(fast2_text, encoding="utf-8")

    last_code = 1
    try:
        for attempt, profile in enumerate(profiles, 1):
            recoveries = list_recovery_states(out) if out.exists() else []
            recovery = recoveries[0] if recoveries else None
            if attempt > 1 and recovery is None and out.exists():
                shutil.rmtree(out, ignore_errors=True)

            args, selected_input = profile_command_args(
                base_args, profile, recovery, legacy_input, fast2_input
            )
            trainer = FAST2_TRAINER if profile in ("fast2c", "fast2") else TRAINER

            last_code = run_one(trainer, args, profile, recovery, selected_input)
            if last_code == 0:
                print(json.dumps({
                    "event": "v8_2_execution_profile_selected",
                    "profile": profile,
                    "input_profile": selected_input,
                    "resumed_from": str(recovery) if recovery else None,
                }), flush=True)
                raise SystemExit(0)

            recoveries = list_recovery_states(out) if out.exists() else []
            newest = recoveries[0] if recoveries else None
            print(json.dumps({
                "event": "v8_2_execution_profile_failed",
                "profile": profile,
                "input_profile": selected_input,
                "returncode": last_code,
                "recovery_available": str(newest) if newest else None,
                "next_fallback": attempt < len(profiles),
            }), flush=True)
    finally:
        FAST2_TRAINER.unlink(missing_ok=True)

    raise SystemExit(last_code)


if __name__ == "__main__":
    main()
