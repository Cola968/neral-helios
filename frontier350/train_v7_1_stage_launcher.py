from __future__ import annotations

"""Scale V7.1 stage launcher.

Fast3 is an execution-only profile selected by the RTX input-path benchmark.  Any Fast3
failure falls back through the existing V7 profiles and resumes the newest exact compact
recovery state; no semantic training state is discarded merely to change execution mode.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from v7_recovery import list_recovery_states
from v7_1_runtime_hotfix import main as apply_runtime_hotfix


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


def profile_args(base_args: list[str], profile: str, recovery: Path | None, input_profile: str) -> list[str]:
    args = remove_option(base_args, "--v7-resume-state", takes_value=True)
    args = remove_option(args, "--v71-coalesced-input", takes_value=False)
    args = remove_option(args, "--v71-input-transfer", takes_value=True)
    if profile == "fast3":
        if input_profile not in ("coalesced-pin", "coalesced-pageable"):
            raise ValueError("Fast3 requires an admitted V7.1 input profile")
        args = replace_value(args, "--compile-mode", "max-autotune-no-cudagraphs")
        if "--fast-replay-only" not in args:
            args.append("--fast-replay-only")
        if "--fused-replay-main" not in args:
            args.append("--fused-replay-main")
        args.append("--v71-coalesced-input")
        args.extend(("--v71-input-transfer", "legacy-pin" if input_profile == "coalesced-pin" else "direct-pageable"))
    elif profile == "turbo60":
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


def run_profile(trainer: Path, base_args: list[str], profile: str, recovery: Path | None, input_profile: str) -> int:
    args = profile_args(base_args, profile, recovery, input_profile)
    print(json.dumps({
        "event": "v7_1_execution_profile",
        "profile": profile,
        "input_profile": input_profile if profile == "fast3" else "v7",
        "resume_recovery": str(recovery) if recovery else None,
    }), flush=True)
    return int(subprocess.run([sys.executable, "-u", str(trainer), *args]).returncode)


def main() -> None:
    # The patch chain is assembled at workflow runtime. Apply the narrowly scoped fixes
    # discovered by the first RTX 4070 smoke run only after V5/V6/V7/V7.1 have patched
    # train_guarded_v9.py, so none of their proven contracts are replaced.
    apply_runtime_hotfix()

    base_args = sys.argv[1:]
    out_value = arg_value(base_args, "--out")
    if not out_value:
        raise SystemExit("missing --out")
    out = Path(out_value)
    trainer = Path(__file__).resolve().parent / "train_guarded_v9.py"
    replay_only = arg_value(base_args, "--mode") == "replay_only"
    input_profile = os.environ.get("NERAL_V71_INPUT_PROFILE", "off").strip().lower()
    if input_profile not in ("off", "coalesced-pin", "coalesced-pageable"):
        raise SystemExit(f"invalid NERAL_V71_INPUT_PROFILE={input_profile!r}")
    if replay_only:
        profiles = (("fast3", "turbo60", "turbo50", "safe") if input_profile != "off"
                    else ("turbo60", "turbo50", "safe"))
    else:
        profiles = ("safe",)

    last_code = 1
    for attempt, profile in enumerate(profiles, 1):
        recoveries = list_recovery_states(out) if out.exists() else []
        recovery = recoveries[0] if recoveries else None
        if attempt > 1 and recovery is None and out.exists():
            shutil.rmtree(out, ignore_errors=True)
        last_code = run_profile(trainer, base_args, profile, recovery, input_profile)
        if last_code == 0:
            print(json.dumps({
                "event": "v7_1_execution_profile_selected",
                "profile": profile,
                "input_profile": input_profile if profile == "fast3" else "v7",
                "resumed_from": str(recovery) if recovery else None,
            }), flush=True)
            raise SystemExit(0)
        recoveries = list_recovery_states(out) if out.exists() else []
        newest = recoveries[0] if recoveries else None
        print(json.dumps({
            "event": "v7_1_execution_profile_failed",
            "profile": profile,
            "returncode": last_code,
            "recovery_available": str(newest) if newest else None,
            "next_fallback": attempt < len(profiles),
        }), flush=True)

    raise SystemExit(last_code)


if __name__ == "__main__":
    main()
