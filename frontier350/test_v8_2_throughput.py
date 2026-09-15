from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from apply_v8_2_throughput import patch_controller
from v8_2_profile_contract import execution_profiles, profile_command_args


def test_controller_routing_is_exact_and_idempotent() -> None:
    old = 'x = str(Path(__file__).parent / "train_v7_2_stage_launcher.py")\n'
    patched = patch_controller(old)
    assert 'train_v8_2_stage_launcher.py' in patched
    assert 'train_v7_2_stage_launcher.py' not in patched
    assert patch_controller(patched) == patched


def test_fast2_coalesced_falls_back_without_losing_legacy_chain() -> None:
    profiles = execution_profiles(
        replay_only=True,
        use_fast2=True,
        legacy_input="off",
        fast2_input="coalesced-pageable",
    )
    assert profiles == ("fast2c", "fast2", "turbo60", "turbo50", "safe")


def test_plain_fast2_contract_when_new_input_path_disabled() -> None:
    profiles = execution_profiles(
        replay_only=True,
        use_fast2=True,
        legacy_input="off",
        fast2_input="off",
    )
    assert profiles == ("fast2", "turbo60", "turbo50", "safe")


def test_non_replay_stage_is_never_upgraded_implicitly() -> None:
    assert execution_profiles(
        replay_only=False,
        use_fast2=True,
        legacy_input="coalesced-pageable",
        fast2_input="coalesced-pageable",
    ) == ("safe",)


def test_legacy_admitted_input_profile_remains_a_fallback() -> None:
    profiles = execution_profiles(
        replay_only=True,
        use_fast2=True,
        legacy_input="coalesced-pageable",
        fast2_input="coalesced-pageable",
    )
    assert profiles == ("fast2c", "fast2", "fast3", "turbo60", "turbo50", "safe")


def test_fast2c_cli_enables_only_admitted_coalesced_contract() -> None:
    base = [
        "--mode", "replay_only",
        "--compile-mode", "default",
        "--fast-replay-only",
        "--out", "scratch",
    ]
    args, selected = profile_command_args(
        base, "fast2c", None, "off", "coalesced-pageable"
    )
    assert selected == "coalesced-pageable"
    assert args[args.index("--compile-mode") + 1] == "max-autotune-no-cudagraphs"
    assert "--fast-replay-only" in args
    assert "--fused-replay-main" in args
    assert "--v71-coalesced-input" in args
    assert args[args.index("--v71-input-transfer") + 1] == "direct-pageable"


def test_plain_fast2_cli_explicitly_disables_coalesced_transfer() -> None:
    base = [
        "--mode", "replay_only",
        "--compile-mode", "default",
        "--fast-replay-only",
        "--out", "scratch",
    ]
    args, selected = profile_command_args(
        base, "fast2", None, "off", "coalesced-pageable"
    )
    assert selected == "off"
    assert "--v71-coalesced-input" not in args
    assert "--v71-input-transfer" not in args
    assert args[args.index("--compile-mode") + 1] == "max-autotune-no-cudagraphs"


def test_invalid_input_profile_fails_closed() -> None:
    try:
        execution_profiles(
            replay_only=True,
            use_fast2=True,
            legacy_input="off",
            fast2_input="mystery",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid V8.2 input profile must fail closed")
