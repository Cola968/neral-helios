"""Neral-50M I1: resume-safe intensive campaign with targeted final tuning.

The campaign keeps B3a/B3b/RawMath/Instruct as internal checkpoints. It resumes interrupted
GPU work from local latest checkpoints, reaches 500M cumulative language-pretraining tokens,
adds guarded RawMath specialization, then instruction-tunes on human OASST1 plus deterministic
targeted German/math/code/reasoning data. Only the final candidate is compared with frozen
Neral-20M Preview 02 across the complete model-only benchmark bundle.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = sys.executable


def run(args, *, allow_failure=False):
    cmd = [str(x) for x in args]
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=REPO)
    if p.returncode and not allow_failure:
        raise subprocess.CalledProcessError(p.returncode, cmd)
    return p.returncode


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf8")


def complete(summary, minimum_tokens=0):
    p = Path(summary)
    if not p.is_file():
        return False
    try:
        row = read(p)
    except Exception:
        return False
    tokens = int(row.get("cumulative_tokens", row.get("source_cumulative_tokens", 0)))
    return bool(row.get("complete")) and tokens >= minimum_tokens


def stage_complete(summary):
    p = Path(summary)
    if not p.is_file():
        return False
    try:
        return bool(read(p).get("complete"))
    except Exception:
        return False


def selected(run_dir, preferred):
    for name in (preferred, "latest.pt"):
        p = Path(run_dir) / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"No checkpoint in {run_dir}")


def junction(link, target):
    link, target = Path(link).absolute(), Path(target).resolve()
    if not target.is_dir():
        raise FileNotFoundError(target)
    if link.resolve() == target:
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"Refusing to replace existing asset directory: {link}")
    if not link.is_relative_to(REPO.resolve()):
        raise ValueError("Junction must stay within source checkout")
    quote = lambda p: "'" + str(p).replace("'", "''") + "'"
    run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
         f"New-Item -ItemType Junction -Path {quote(link)} -Target {quote(target)} -ErrorAction Stop | Out-Null"])


def ensure_b3_data(assets, b2_data):
    data = assets / "pretrain50_b3_data"
    req = ("train_a_de.bin", "train_a_en.bin", "train_b_de.bin", "train_b_en.bin", "val_de.bin", "val_en.bin", "documents.jsonl", "manifest.json")
    if not all((data / x).is_file() for x in req):
        run([PY, "-m", "pip", "install", "--disable-pip-version-check", "-r", "requirements-data.txt"])
        src = assets / "b3_sources_auto"
        run([PY, "bootstrap_b3_sources.py", "--out", str(src)])
        run([PY, "prepare_pretrain50_b3.py", "--source-root", str(src), "--b2-data", str(b2_data), "--out", str(data), "--phase-a-tokens", "220000000", "--phase-b-tokens", "220000000"])
    if not all((data / x).is_file() for x in req):
        raise RuntimeError("B3 corpus incomplete")
    junction(REPO / "pretrain50_b3_data", data)
    return data


def build_sft(work):
    src = work / "sft_sources"
    data = work / "sft_data"
    req = ("train.jsonl", "val.jsonl", "test.jsonl", "holdout.jsonl", "manifest.json")
    if all((data / name).is_file() for name in req):
        try:
            prepared = read(data / "manifest.json")
            if prepared.get("i1_targeted_v1_screened_from_sft") is True and int(prepared.get("target_train_examples", 0)) >= 10_000:
                return data
        except Exception:
            pass
    if src.exists():
        shutil.rmtree(src)
    if data.exists():
        shutil.rmtree(data)
    run([PY, "fetch_oasst1_sft50.py", "--out", str(src), "--max-en", "16000", "--max-de", "9000"])
    run([PY, "build_targeted_sft50_sources.py", "--root", str(src), "--per-generated-source", "6000"])
    run([PY, "prepare_sft50_i1.py", "--source-root", str(src), "--out", str(data), "--target-train-examples", "0", "--max-seq", "1024"])
    run([PY, "validate_sft50_dataset.py", "--data", str(data), "--out", str(work / "SFT_DATA_AUDIT.json")])
    return data


def benchmark(checkpoint, out):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    run([PY, "benchmark_bundle50.py", str(checkpoint), "--out-dir", str(out)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--reports", default="i1_reports")
    args = ap.parse_args()
    assets = Path(args.assets).resolve()
    reports = (REPO / args.reports).resolve()
    reports.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        raise RuntimeError("I1 expects the Windows RTX runner")
    if not assets.is_dir():
        raise FileNotFoundError(assets)

    run_id = os.environ.get("GITHUB_RUN_ID", str(int(time.time())))
    state = {
        "campaign": "Neral-50M I1",
        "campaign_version": 2,
        "run_id": run_id,
        "started_at_unix": time.time(),
        "policy": "Resume-safe local checkpoints; one final candidate; frozen model-only promotion evidence.",
        "curriculum": {
            "B3a": "99.25M selected B2 -> 300M cumulative, seq512, LR unchanged for resume compatibility",
            "B3b": "300M -> 500M cumulative, seq1024",
            "RawMath": "30M mixed tokens, 35% math-domain loss / 65% language replay",
            "Instruct": "20k contamination-screened SFT mix, 4000 max steps, 78% assistant loss / 22% base replay",
        },
        "phases": [],
    }
    write(reports / "I1_STATE.json", state)

    b1_data = assets / "pretrain50_data"
    b2_data = assets / "pretrain50_b2_data"
    junction(REPO / "pretrain50_data", b1_data)
    junction(REPO / "pretrain50_b2_data", b2_data)
    for root, names in ((b1_data, ("val_de.bin", "val_en.bin", "documents.jsonl")),
                        (b2_data, ("train_de.bin", "train_en.bin", "val_de.bin", "val_en.bin", "documents.jsonl", "manifest.json"))):
        for name in names:
            if not (root / name).is_file():
                raise FileNotFoundError(root / name)

    base20 = assets / "export/pilot_02/neral20m.pt"
    if not base20.is_file():
        raise FileNotFoundError("Frozen Neral-20M Preview02 checkpoint required")
    b2_run = assets / "runs/neral50m_b2_100m"
    b2_summary = b2_run / "summary.json"
    if not complete(b2_summary, 100_000_000):
        raise RuntimeError("Completed B2 100M required")
    b2 = selected(b2_run, "best_validation.pt")
    b3_data = ensure_b3_data(assets, b2_data)

    # Phase 1: keep exact original hyperparameters so the already-saved B3a optimizer state remains valid.
    b3a_run = assets / "runs/neral50m_i1_b3a_300m"
    b3a_summary = b3a_run / "summary.json"
    if not complete(b3a_summary, 300_000_000):
        resume = b3a_run / "latest.pt" if (b3a_run / "latest.pt").is_file() else b2
        run([PY, "pretrain50_b3.py", "--phase", "a", "--resume", str(resume), "--data", str(b3_data), "--old-data", str(b2_data), "--out", str(b3a_run), "--target-cumulative-tokens", "300000000", "--seq", "512", "--batch", "16", "--peak-lr", "0.00006", "--min-lr", "0.000006", "--eval-every", "250", "--validation-blocks", "192"])
    if not complete(b3a_summary, 300_000_000):
        raise RuntimeError("I1 B3a incomplete")
    b3a = selected(b3a_run, "best_validation.pt")
    state["phases"].append({"phase": "B3a", "checkpoint": str(b3a), "summary": read(b3a_summary)})
    write(reports / "I1_STATE.json", state)

    # Phase 2: longer context with a conservative LR.
    b3b_run = assets / "runs/neral50m_i1_b3b_500m"
    b3b_summary = b3b_run / "summary.json"
    if not complete(b3b_summary, 500_000_000):
        resume = b3b_run / "latest.pt" if (b3b_run / "latest.pt").is_file() else b3a
        run([PY, "pretrain50_b3.py", "--phase", "b", "--resume", str(resume), "--data", str(b3_data), "--old-data", str(b2_data), "--out", str(b3b_run), "--target-cumulative-tokens", "500000000", "--seq", "1024", "--batch", "8", "--peak-lr", "0.00003", "--min-lr", "0.000003", "--eval-every", "250", "--validation-blocks", "192"])
    if not complete(b3b_summary, 500_000_000):
        raise RuntimeError("I1 B3b incomplete")
    b3b = selected(b3b_run, "best_validation.pt")
    state["phases"].append({"phase": "B3b", "checkpoint": str(b3b), "summary": read(b3b_summary)})
    write(reports / "I1_STATE.json", state)

    # Phase 3: stable output directory + resume-safe optimizer/RNG state.
    raw_run = assets / "runs/neral50m_i1_rawmath_c3"
    raw_summary = raw_run / "summary.json"
    if not stage_complete(raw_summary):
        run([PY, "train_raw_math50_i1.py", "--init-from", str(b3b), "--replay-data", str(b2_data), "--out", str(raw_run), "--tokens", "30000000", "--seq", "256", "--batch", "32", "--math-fraction", "0.35", "--peak-lr", "0.00004", "--min-lr", "0.000004", "--eval-every", "100", "--validation-blocks", "160"])
    if not stage_complete(raw_summary):
        raise RuntimeError("I1 RawMath incomplete")
    raw = selected(raw_run, "best_math.pt")
    state["phases"].append({"phase": "RawMath C3", "checkpoint": str(raw), "summary": read(raw_summary)})
    write(reports / "I1_STATE.json", state)

    # Phase 4: targeted SFT. Data and trainer are both resume-safe across separate PC sessions.
    work = assets / "i1_work_v2"
    work.mkdir(parents=True, exist_ok=True)
    sft = build_sft(work)
    inst_run = assets / "runs/neral50m_i1_instruct_d3"
    inst_summary = inst_run / "summary.json"
    if not stage_complete(inst_summary):
        run([PY, "train_instruct50_i1.py", "--init-from", str(raw), "--data", str(sft), "--replay-data", str(b2_data), "--out", str(inst_run), "--steps", "4000", "--seq", "1024", "--batch-size", "8", "--replay-batch", "8", "--sft-weight", "0.78", "--eos-weight", "2.0", "--peak-lr", "0.00002", "--min-lr", "0.000002", "--warmup-fraction", "0.05", "--eval-every", "100", "--patience", "12", "--min-delta", "0.0015"])
    if not stage_complete(inst_summary):
        raise RuntimeError("I1 Instruct incomplete")
    final = selected(inst_run, "best_instruct.pt")
    state["phases"].append({"phase": "Instruct D3", "checkpoint": str(final), "summary": read(inst_summary)})
    write(reports / "I1_STATE.json", state)

    # One final, decisive, identical frozen comparison.
    benchmark(base20, reports / "baseline20")
    benchmark(final, reports / "candidate50")
    run([PY, "build_promotion_comparison50.py",
         "--baseline20-free", str(reports / "baseline20/free.json"),
         "--baseline20-v3", str(reports / "baseline20/v3.json"),
         "--baseline20-v4", str(reports / "baseline20/v4.json"),
         "--baseline20-rawmath", str(reports / "baseline20/rawmath.json"),
         "--baseline20-targeted", str(reports / "baseline20/targeted.json"),
         "--candidate-free", str(reports / "candidate50/free.json"),
         "--candidate-v3", str(reports / "candidate50/v3.json"),
         "--candidate-v4", str(reports / "candidate50/v4.json"),
         "--candidate-rawmath", str(reports / "candidate50/rawmath.json"),
         "--candidate-targeted", str(reports / "candidate50/targeted.json"),
         "--out", str(reports / "I1_20M_VS_50M.json")])
    run([PY, "promotion_gate50.py", str(reports / "I1_20M_VS_50M.json"), "--candidate", "instruct", "--out", str(reports / "I1_PROMOTION_GATE.json")], allow_failure=True)
    run([PY, "long_context_gate50.py", str(reports / "baseline20/v5.json"), str(reports / "candidate50/v5.json"), "--mode", "progress", "--out", str(reports / "I1_LONG_CONTEXT_GATE.json")], allow_failure=True)

    comparison = read(reports / "I1_20M_VS_50M.json")
    promotion = read(reports / "I1_PROMOTION_GATE.json")
    from i1_targets50 import evaluate as evaluate_targets
    target_gate = evaluate_targets(comparison, read(reports / "baseline20/v5.json"), read(reports / "candidate50/v5.json"), promotion)
    write(reports / "I1_TARGET_GATE.json", target_gate)
    run([PY, "i1_quality_gate50.py",
         "--comparison", str(reports / "I1_20M_VS_50M.json"),
         "--promotion", str(reports / "I1_PROMOTION_GATE.json"),
         "--baseline-targeted", str(reports / "baseline20/targeted.json"),
         "--candidate-targeted", str(reports / "candidate50/targeted.json"),
         "--out", str(reports / "I1_QUALITY_GATE.json")], allow_failure=True)
    quality_gate = read(reports / "I1_QUALITY_GATE.json")

    nis20 = comparison["nis"]["baseline20"]["score"]
    nis50 = comparison["nis"]["instruct"]["score"]
    result = {
        "campaign": "Neral-50M I1",
        "campaign_version": 2,
        "final_checkpoint": str(final),
        "nis20": nis20,
        "nis50": nis50,
        "nis_delta": nis50 - nis20,
        "promotion_gate": promotion,
        "target_gate": target_gate,
        "quality_gate": quality_gate,
        "hard_target_nis55": nis50 >= 55.0,
        "completed_at_unix": time.time(),
    }
    write(reports / "I1_FINAL_RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
