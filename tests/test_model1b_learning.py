from pathlib import Path

from helios1b.learning_experiment import run_experiment


def test_helios_proxy_actually_learns(tmp_path: Path):
    result = run_experiment(
        tmp_path / "learnability",
        steps=20,
        minimum_relative_improvement=0.30,
        device="cpu",
    )
    assert result["passed"]
    assert result["best_val_ce"] < result["initial_val_ce"]
