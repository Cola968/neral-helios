from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import HeliosConfig
from .train_proxy import ProxyTrainingConfig, run_training


MOTIFS = (
    tuple(range(16, 32)),
    tuple(range(32, 64, 2)),
    tuple(range(63, 47, -1)),
    (70, 71, 70, 72, 70, 73, 70, 74, 70, 75, 70, 76, 70, 77, 70, 78),
)


def build_learnability_stream(seed: int, documents: int) -> np.ndarray:
    """Create held-out deterministic language with several local grammars.

    This is not a capability benchmark. It answers one narrow question:
    can the architecture + optimizer + data cursor actually learn next-token
    structure and generalize it to a separately generated validation stream?
    """

    rng = random.Random(seed)
    tokens: list[int] = []
    for _ in range(documents):
        motif_id = rng.randrange(len(MOTIFS))
        repeats = rng.randrange(3, 6)
        tokens.extend((1, 4 + motif_id))
        tokens.extend(MOTIFS[motif_id] * repeats)
        tokens.append(2)
    return np.asarray(tokens, dtype="<u2")


def write_stream(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values.astype("<u2", copy=False).tofile(path)


def run_experiment(
    out_dir: str | Path,
    *,
    steps: int = 40,
    minimum_relative_improvement: float = 0.50,
    device: str = "auto",
) -> dict:
    if steps < 10:
        raise ValueError("learnability experiment needs at least 10 steps")
    if not 0 < minimum_relative_improvement < 1:
        raise ValueError("minimum_relative_improvement must be in (0, 1)")

    out = Path(out_dir)
    data = out / "data"
    train_bin = data / "train.bin"
    val_bin = data / "val.bin"
    write_stream(train_bin, build_learnability_stream(96810, 800))
    write_stream(val_bin, build_learnability_stream(96811, 240))

    model_cfg = replace(
        HeliosConfig.tiny_proxy(),
        vocab_size=128,
        max_seq_len=64,
    )
    train_cfg = ProxyTrainingConfig(
        train_bin=str(train_bin),
        val_bin=str(val_bin),
        out_dir=str(out / "run"),
        total_steps=steps,
        seq_len=48,
        batch_size=8,
        grad_accum=1,
        eval_every=5,
        eval_batches=8,
        save_every=5,
        seed=96812,
        peak_lr=1e-3,
        min_lr=1e-4,
        warmup_steps=max(2, steps // 10),
        weight_decay=0.05,
        grad_clip=1.0,
        device=device,
    )
    summary = run_training(train_cfg, model_cfg)

    initial = float(summary["initial_val_ce"])
    best = float(summary["best_val_ce"])
    relative = (initial - best) / initial
    passed = (
        summary["completed_step"] == steps
        and best < initial
        and relative >= minimum_relative_improvement
    )
    result = {
        "experiment": "helios1b-learnability-v1",
        "purpose": (
            "Architecture/trainer learnability gate only; not a language "
            "quality or chatbot capability benchmark."
        ),
        "steps": steps,
        "initial_val_ce": initial,
        "best_val_ce": best,
        "relative_val_improvement": relative,
        "minimum_relative_improvement": minimum_relative_improvement,
        "passed": passed,
        "training_summary": summary,
    }
    (out / "LEARNABILITY_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError(
            "Helios learnability gate failed: "
            f"initial={initial:.4f}, best={best:.4f}, relative={relative:.3%}"
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/helios1b_learnability_v1")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--minimum-relative-improvement", type=float, default=0.50)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    result = run_experiment(
        args.out,
        steps=args.steps,
        minimum_relative_improvement=args.minimum_relative_improvement,
        device=args.device,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
