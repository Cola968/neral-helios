from __future__ import annotations

"""Pure, dependency-free execution contract for Scale V8.2.

Keeping profile selection/CLI construction separate from torch/recovery code lets CI prove
that the optimization cannot silently change stage semantics before any GPU code is loaded.
"""

from pathlib import Path

_ALLOWED_INPUT = ("off", "coalesced-pin", "coalesced-pageable")


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


def execution_profiles(*, replay_only: bool, use_fast2: bool,
                       legacy_input: str, fast2_input: str) -> tuple[str, ...]:
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


def _legacy_profile_args(base_args: list[str], profile: str, recovery: Path | None,
                         input_profile: str) -> list[str]:
    """Mirror the proven V7.1 execution-only argument contract."""
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
        args.extend(("--v71-input-transfer",
                     "legacy-pin" if input_profile == "coalesced-pin" else "direct-pageable"))
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


def profile_command_args(base_args: list[str], profile: str, recovery: Path | None,
                         legacy_input: str, fast2_input: str) -> tuple[list[str], str]:
    if profile == "fast2c":
        return _legacy_profile_args(base_args, "fast3", recovery, fast2_input), fast2_input
    if profile == "fast2":
        return _legacy_profile_args(base_args, "turbo50", recovery, "off"), "off"
    args = _legacy_profile_args(base_args, profile, recovery, legacy_input)
    return args, legacy_input if profile == "fast3" else "off"
