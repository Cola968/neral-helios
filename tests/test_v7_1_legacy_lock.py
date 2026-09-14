from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "frontier350"


def read(name: str) -> str:
    return (F / name).read_text(encoding="utf-8")


def test_v5_turbo_profiles_remain_available():
    launcher = read("train_v7_1_stage_launcher.py")
    assert '("fast3", "turbo60", "turbo50", "safe")' in launcher
    assert '("turbo60", "turbo50", "safe")' in launcher
    assert 'profile == "turbo60"' in launcher
    assert 'profile == "turbo50"' in launcher
    assert 'profile == "safe"' in launcher


def test_v5_fused_replay_main_contract_is_retained():
    patch = read("apply_v5_turbo_patch.py")
    assert "--fused-replay-main" in patch
    assert "max-autotune" in patch
    assert "max-autotune-no-cudagraphs" in patch
    assert "replay_boost" in patch
    assert "weight_denominator" in patch


def test_v6_fast_host_sync_and_safe_baseline_optimizations_remain():
    patch = read("apply_v6_fast_patch.py")
    assert "_baseline_val_cache" in patch
    assert "step%10==0" in patch
    assert "metric_now=(not args.fused_replay_main) or step%10==0" in patch


def test_v6_fast2_exact_compiled_objective_remains():
    patch = read("apply_v6_fast2_patch.py")
    assert "_v6_fused_objective" in patch
    assert "max-autotune-no-cudagraphs" in patch
    assert "replay_boost" in patch
    assert "fused_run(x_all,y_all)" in patch


def test_v7_1_is_additive_not_replacement():
    patch = read("apply_v7_1_speed_patch.py")
    launcher = read("train_v7_1_stage_launcher.py")
    # V7.1 must still be layered after the existing V5/V6/V7 chain and preserve
    # the V7 fallback profiles if Fast3 is disabled or rejected.
    assert "v7" in patch.lower()
    assert 'else ("turbo60", "turbo50", "safe")' in launcher
    assert "--v7-resume-state" in launcher
