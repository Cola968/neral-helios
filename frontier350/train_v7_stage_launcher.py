from __future__ import annotations

"""Scale V7 stage launcher.

Unlike V5/V6, an execution-profile failure does not automatically throw away the active
5M stage.  The next safer profile resumes the newest exact compact V7 recovery state.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from v7_recovery import list_recovery_states


def arg_value(args: list[str], name: str) -> str | None:
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError):
        return None


def replace_value(args: list[str], name: str, value: str) -> list[str]:
    out = list(args)
    if name in out:
        out[out.index(name) + 1] = value
    else:
        out.extend((name, value))
    return out


def remove_option(args: list[str], name: str, *, takes_value: bool) -> list[str]:
    out = list(args)
    while name in out:
        i = out.index(name)
        del out[i:i + (2 if takes_value else 1)]
    return out


def profile_args(base_args: list[str], profile: str, recovery: Path | None) -> list[str]:
    args = remove_option(base_args, "--v7-resume-state", takes_value=True)
    if profile == "turbo60":
        args = replace_value(args, "--compile-mode", "max-autotune")
        if "--fast-replay-only" in args and "--fused-replay-main" not in args:
            args.append("--fused-replay-main")
    elif profile == "turbo50":
        args = replace_value(args, "--compile-mode", "max-autotune-no-cudagraphs")
        if "--fast-replay-only" in args and "--fused-replay-main" not in args:
            args.append("--fused-replay-main")
    elif profile == "safe":
        args = replace_value(args, "--compile-mode", "default")
        args = remove_option(args, "--fused-replay-main", takes_value=False)
    else:
        raise ValueError(profile)
    if recovery is not None:
        args.extend(("--v7-resume-state", str(recovery)))
    return args


def run_profile(trainer: Path, base_args: list[str], profile: str, recovery: Path | None) -> int:
    args = profile_args(base_args, profile, recovery)
    print(json.dumps({
        "event": "v7_execution_profile",
        "profile": profile,
        "resume_recovery": str(recovery) if recovery else None,
    }), flush=True)
    return int(subprocess.run([sys.executable, "-u", str(trainer), *args]).returncode)


def main() -> None:
    base_args = sys.argv[1:]
    out_value = arg_value(base_args, "--out")
    if not out_value:
        raise SystemExit("missing --out")
    out = Path(out_value)
    trainer = Path(__file__).resolve().parent / "train_guarded_v9.py"
    replay_only = arg_value(base_args, "--mode") == "replay_only"
    profiles = ("turbo60", "turbo50", "safe") if replay_only else ("safe",)

    last_code = 1
    for attempt, profile in enumerate(profiles, 1):
        recoveries = list_recovery_states(out) if out.exists() else []
        recovery = recoveries[0] if recoveries else None
        if attempt > 1 and recovery is None and out.exists():
            # No validated compact state exists. Restarting from the immutable stage anchor
            # is safer than attempting to consume a partial/corrupt output directory.
            shutil.rmtree(out, ignore_errors=True)
        last_code = run_profile(trainer, base_args, profile, recovery)
        if last_code == 0:
            print(json.dumps({
                "event": "v7_execution_profile_selected",
                "profile": profile,
                "resumed_from": str(recovery) if recovery else None,
            }), flush=True)
            raise SystemExit(0)
        newest = list_recovery_states(out)[0] if out.exists() and list_recovery_states(out) else None
        print(json.dumps({
            "event": "v7_execution_profile_failed",
            "profile": profile,
            "returncode": last_code,
            "recovery_available": str(newest) if newest else None,
            "next_fallback": attempt < len(profiles),
        }), flush=True)

    raise SystemExit(last_code)


if __name__ == "__main__":
    main()
