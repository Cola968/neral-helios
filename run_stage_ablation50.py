"""Read-only Neral-50M stage ablation: RawMath C3 vs final Instruct D3.

Purpose: quantify whether instruction tuning traded away mathematical capability. This script
never trains, mutates weights, calls Helios/SOLVE, or uses an external model. Both checkpoints
are evaluated on the same frozen model-only suites and the delta is written as report JSON.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = sys.executable


def run(*args: str) -> None:
    cmd = [PY, *map(str, args)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def targeted_accuracy(targeted: dict) -> float:
    """Return Targeted-v1 overall accuracy across current and legacy report schemas."""
    overall = targeted.get("overall")
    if isinstance(overall, dict) and "accuracy" in overall:
        return float(overall["accuracy"])
    if "overall_accuracy" in targeted:
        return float(targeted["overall_accuracy"])
    raise KeyError("Targeted-v1 report missing overall accuracy")


def summarize(root: Path) -> dict:
    raw = load(root / "rawmath.json")
    free = load(root / "free.json")
    targeted = load(root / "targeted.json")
    v4 = load(root / "v4.json")
    return {
        "rawmath_exact_pct": float(raw["overall"]["exact_pct"]),
        "rawmath_answer_nll": float(raw["overall"]["answer_nll"]),
        "rawmath_transfer_pct": float(raw["by_distribution"]["transfer"]["exact_pct"]),
        "rawmath_ood_pct": float(raw["by_distribution"]["ood"]["exact_pct"]),
        "free_eos_pct": 100.0 * float(free["summary"]["eos_fraction"]),
        "free_repeated_trigram_pct": 100.0 * float(free["summary"]["mean_repeated_trigram_fraction"]),
        "targeted_overall_pct": 100.0 * targeted_accuracy(targeted),
        "targeted_math_pct": 100.0 * float(targeted["domains"]["math"]["accuracy"]),
        "targeted_code_pct": 100.0 * float(targeted["domains"]["code"]["accuracy"]),
        "targeted_de_pct": 100.0 * float(targeted["domains"]["de_grammar"]["accuracy"]),
        "v4_overall_pct": 100.0 * float(v4["pair_suite"]["accuracy"]),
        "v4_math_pct": 100.0 * float(v4["pair_suite"]["domains"]["math"]["accuracy"]),
        "v4_code_pct": 100.0 * float(v4["pair_suite"]["domains"]["code"]["accuracy"]),
        "context_recall_pct": 100.0 * float(v4["context_recall"]["accuracy"]),
    }


def benchmark(checkpoint: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    run("benchmark_raw_math50.py", str(checkpoint), "--out", str(out / "rawmath.json"), "--max-new-tokens", "32")
    run("free_generation50.py", "--checkpoint", str(checkpoint), "--out", str(out / "free.json"), "--max-tokens", "64")
    run("benchmark_targeted50.py", str(checkpoint), "--out", str(out / "targeted.json"))
    run("benchmark_v4.py", str(checkpoint), "--out", str(out / "v4.json"), "--max-new-tokens", "96")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", default="stage_ablation_reports")
    args = ap.parse_args()
    assets = Path(args.assets).resolve()
    out = (REPO / args.out).resolve()
    raw_ckpt = assets / "runs/neral50m_i1_rawmath_c3/best_math.pt"
    inst_ckpt = assets / "runs/neral50m_i1_instruct_d3/best_instruct.pt"
    for path in (raw_ckpt, inst_ckpt):
        if not path.is_file():
            raise FileNotFoundError(path)
    raw_out = out / "rawmath_c3"
    inst_out = out / "instruct_d3"
    benchmark(raw_ckpt, raw_out)
    benchmark(inst_ckpt, inst_out)
    raw = summarize(raw_out)
    inst = summarize(inst_out)
    delta = {key: inst[key] - raw[key] for key in raw}
    result = {
        "definition": "Frozen model-only ablation of I1 RawMath C3 versus final Instruct D3; no training or tools.",
        "rawmath_c3": raw,
        "instruct_d3": inst,
        "delta_instruct_minus_rawmath": delta,
        "diagnosis": {
            "rawmath_exact_regressed": inst["rawmath_exact_pct"] < raw["rawmath_exact_pct"],
            "rawmath_nll_regressed": inst["rawmath_answer_nll"] > raw["rawmath_answer_nll"],
            "generation_eos_improved": inst["free_eos_pct"] > raw["free_eos_pct"],
            "targeted_code_improved": inst["targeted_code_pct"] > raw["targeted_code_pct"],
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "STAGE_ABLATION.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
