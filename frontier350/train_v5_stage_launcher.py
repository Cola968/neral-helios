from __future__ import annotations

"""Reliable launcher for Scale V5 turbo stages.

It tries the fastest execution profile first. If the trainer process fails (for example
because a compile profile is unsupported or runs out of memory), the partial stage output
is removed and the exact same stage is retried with progressively safer execution
profiles. The training objective, token budget, replay weighting, LR schedule and anchor
are not changed by the fallback.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path


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


def without(args: list[str], flag: str) -> list[str]:
    return [x for x in args if x != flag]


def run_profile(trainer: Path, base_args: list[str], profile: str) -> int:
    args = list(base_args)
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
        args = without(args, "--fused-replay-main")
    else:
        raise ValueError(profile)

    print(json.dumps({"event": "v5_execution_profile", "profile": profile}), flush=True)
    proc = subprocess.run([sys.executable, "-u", str(trainer), *args])
    return int(proc.returncode)


def main() -> None:
    base_args = sys.argv[1:]
    out_value = arg_value(base_args, "--out")
    if not out_value:
        raise SystemExit("missing --out")
    out = Path(out_value)
    trainer = Path(__file__).resolve().parent / "train_guarded_v9.py"

    replay_only = arg_value(base_args, "--mode") == "replay_only"
    profiles = ("turbo60", "turbo50", "safe") if replay_only else ("safe",)

    for attempt, profile in enumerate(profiles, 1):
        if attempt > 1 and out.exists():
            shutil.rmtree(out, ignore_errors=True)
        code = run_profile(trainer, base_args, profile)
        if code == 0:
            print(json.dumps({"event": "v5_execution_profile_selected", "profile": profile}), flush=True)
            raise SystemExit(0)
        print(json.dumps({"event": "v5_execution_profile_failed", "profile": profile,
                          "returncode": code, "next_fallback": attempt < len(profiles)}), flush=True)

    raise SystemExit(code)


if __name__ == "__main__":
    main()
