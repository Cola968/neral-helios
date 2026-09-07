"""Run Neral's reserved external public evaluations without touching training data.

The runner currently evaluates GSM8K only. Public benchmark data is prepared in a temporary
working directory, never mixed into an Neral training corpus, and removed before exit. Only the
model-only JSON/Markdown result is intended for artifacts. No SOLVE/Helios/external model is
used to answer benchmark questions.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = sys.executable


def run(args):
    cmd = [str(x) for x in args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def choose_candidate(assets: Path):
    options = [
        ("instruct_d3", assets / "runs/neral50m_i1_instruct_d3/best_instruct.pt"),
        ("rawmath_c3", assets / "runs/neral50m_i1_rawmath_c3/best_math.pt"),
        ("b3b_500m", assets / "runs/neral50m_i1_b3b_500m/best_validation.pt"),
        ("b3a_300m", assets / "runs/neral50m_i1_b3a_300m/best_validation.pt"),
        ("b2_100m", assets / "runs/neral50m_b2_100m/best_validation.pt"),
    ]
    for stage, path in options:
        if path.is_file():
            return stage, path
    raise FileNotFoundError("No selected Neral-50M checkpoint found")


def render_markdown(report: dict, stage: str):
    s = report["summary"]
    manifest = report["dataset_manifest"]
    return "\n".join([
        "# Neral external evaluation",
        "",
        f"Checkpoint stage: **{stage}**",
        "",
        "## GSM8K main test",
        "",
        f"- Evaluated: {report['evaluated']} examples",
        f"- Exact numeric answers: {s['exact']} / {report['evaluated']} = **{s['exact_pct']:.2f}%**",
        f"- Numeric answer extracted: **{s['extracted_pct']:.2f}%**",
        f"- Explicit `Answer:` marker: **{s['answer_marker_pct']:.2f}%**",
        f"- EOS reached: **{s['eos_pct']:.2f}%**",
        f"- Mean generated tokens: {s['mean_generated_tokens']:.2f}",
        f"- Decode throughput: {s['mean_tokens_per_second']:.2f} generated tokens/s",
        "",
        "## Reproducibility",
        "",
        f"- Dataset: `{manifest['dataset_id']}`",
        f"- Upstream revision: `{manifest['upstream_revision']}`",
        f"- Upstream test SHA-256: `{manifest['upstream_sha256']}`",
        f"- Prepared holdout SHA-256: `{manifest['prepared_sha256']}`",
        f"- Checkpoint SHA-256: `{report['checkpoint_sha256']}`",
        "- Training allowed: **false**",
        "- Worked GSM8K rationales copied into the holdout: **false**",
        "- SOLVE used: **false**",
        "- Helios used: **false**",
        "- External answer model used: **false**",
        "",
        "> This is Neral's fixed direct-answer harness, not a claim of an official leaderboard score.",
        "",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", default="external_eval_reports")
    ap.add_argument("--gsm8k-limit", type=int, default=0, help="0 evaluates all 1319 test items")
    args = ap.parse_args()
    assets = Path(args.assets).resolve()
    if not assets.is_dir():
        raise FileNotFoundError(assets)
    stage, checkpoint = choose_candidate(assets)
    out = (REPO / args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    started = time.time()
    with tempfile.TemporaryDirectory(prefix="neral-public-eval-") as tmp:
        data = Path(tmp) / "gsm8k"
        run([PY, "prepare_gsm8k_holdout50.py", "--out", data])
        report_path = out / "gsm8k.json"
        cmd = [PY, "benchmark_gsm8k50.py", checkpoint, "--data", data, "--out", report_path]
        if args.gsm8k_limit:
            cmd += ["--limit", str(args.gsm8k_limit)]
        run(cmd)
        report = json.loads(report_path.read_text(encoding="utf8"))

    # TemporaryDirectory has now removed every copied public-eval question/test asset.
    (out / "EXTERNAL_EVAL_REPORT.md").write_text(render_markdown(report, stage), encoding="utf8")
    result = {
        "stage": stage,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "gsm8k_exact_pct": report["summary"]["exact_pct"],
        "gsm8k_examples": report["evaluated"],
        "public_eval_data_retained_in_report_directory": False,
        "training_performed": False,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
    }
    (out / "EXTERNAL_EVAL_RESULT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
