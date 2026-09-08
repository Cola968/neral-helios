"""Neral-50M H1: fresh Hugging Face continued-pretraining campaign to 1B cumulative tokens.

H1 is intentionally independent of the running M2.1 repair. It starts from the strongest
pre-instruction B3b checkpoint, stages a new document-disjoint FineWeb/FineWeb2/FineWeb-Edu
corpus, continues language pretraining to 1B cumulative byte tokens, then re-runs guarded
RawMath and OASST2 instruction stages. Frozen release benchmarks are used only once after all
weight-changing stages and never for checkpoint selection.

The raw public corpus and checkpoints remain on the local Windows runner. GitHub receives only
small JSON/text reports.
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

import torch

from experiment import atomic_save, sha
from hf_source_policy50 import assert_training_allowed, validate_registry
from loss_telemetry50 import summarize_metrics

REPO = Path(__file__).resolve().parent
PY = sys.executable
EXPECTED_SOURCE_SHA = "ab48ea9266444c19c67325a9f344813f5b69692cf65749d6abf5b5a15ff222da"


def run(args, *, allow_failure: bool = False) -> int:
    cmd = [str(x) for x in args]
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO)
    if proc.returncode and not allow_failure:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf8")


def stage_complete(summary: Path) -> bool:
    if not summary.is_file():
        return False
    try:
        return bool(read(summary).get("complete"))
    except Exception:
        return False


def selected(run_dir: Path, preferred: str) -> Path:
    for name in (preferred, "latest.pt"):
        path = run_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No checkpoint in {run_dir}")


def junction(link: Path, target: Path) -> None:
    link = link.absolute()
    target = target.resolve()
    if not target.is_dir():
        raise FileNotFoundError(target)
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"Refusing to replace existing asset link: {link}")
    if not link.is_relative_to(REPO.resolve()):
        raise ValueError("Junction must stay inside trusted checkout")
    q = lambda p: "'" + str(p).replace("'", "''") + "'"
    run([
        "powershell", "-NoProfile", "-NonInteractive", "-Command",
        f"New-Item -ItemType Junction -Path {q(link)} -Target {q(target)} -ErrorAction Stop | Out-Null",
    ])


def loss_snapshot(run_dir: Path) -> dict:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return {"evaluations": 0, "status": "no_metrics"}
    try:
        return summarize_metrics(path)
    except Exception as exc:
        return {"evaluations": 0, "status": "telemetry_error", "error": type(exc).__name__}


def merge_exclusion_indexes(out: Path, roots: list[Path]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "documents.jsonl"
    rows = 0
    seen = set()
    with dest.open("w", encoding="utf8") as writer:
        for root in roots:
            src = root / "documents.jsonl"
            if not src.is_file():
                raise FileNotFoundError(src)
            for line in src.read_text(encoding="utf8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                key = (
                    obj.get("normalized_sha256"), obj.get("source_id"), str(obj.get("id")), obj.get("url")
                )
                if key in seen:
                    continue
                seen.add(key)
                writer.write(json.dumps(obj, ensure_ascii=False) + "\n")
                rows += 1
    return {"roots": [str(x) for x in roots], "rows": rows, "sha256": sha(dest)}


def ensure_h1_data(assets: Path, b2_data: Path, b3_data: Path) -> tuple[Path, dict]:
    # Same already-approved human-web source families used by B3, but a larger upstream window.
    for dataset in (
        "HuggingFaceFW/fineweb",
        "HuggingFaceFW/fineweb-2",
        "HuggingFaceFW/fineweb-edu",
    ):
        assert_training_allowed(dataset)
    registry = validate_registry()

    data = assets / "pretrain50_h1_data"
    required = (
        "train_a_de.bin", "train_a_en.bin", "train_b_de.bin", "train_b_en.bin",
        "val_de.bin", "val_en.bin", "documents.jsonl", "manifest.json",
    )
    if not all((data / name).is_file() for name in required):
        run([PY, "-m", "pip", "install", "--disable-pip-version-check", "-r", "requirements-data.txt"])
        source = assets / "h1_sources_auto_v1"
        source_required = (
            "fineweb-en-general.jsonl", "fineweb2-de-general.jsonl", "fineweb-edu-en.jsonl", "manifest.json"
        )
        if not all((source / name).is_file() for name in source_required):
            if source.exists():
                shutil.rmtree(source)
            run([PY, "bootstrap_b3_sources.py", "--out", str(source), "--scale", "3.0"])

        exclusions = assets / "h1_exclusions_v1"
        if exclusions.exists():
            shutil.rmtree(exclusions)
        exclusion_report = merge_exclusion_indexes(exclusions, [b2_data, b3_data])

        if data.exists():
            shutil.rmtree(data)
        run([
            PY, "prepare_pretrain50_b3.py",
            "--source-root", str(source),
            "--b2-data", str(exclusions),
            "--out", str(data),
            "--phase-a-tokens", "20000000",
            "--phase-b-tokens", "560000000",
        ])
        write(data / "H1_EXCLUSION_PROVENANCE.json", exclusion_report)

    if not all((data / name).is_file() for name in required):
        raise RuntimeError("H1 corpus incomplete")
    manifest = read(data / "manifest.json")
    actual_b = int(manifest.get("actual_phase_b_tokens", 0))
    if actual_b < 520_000_000:
        raise RuntimeError(f"H1 phase-B corpus too small: {actual_b:,}")
    junction(REPO / "pretrain50_h1_data", data)
    return data, {"registry": registry, "manifest": manifest}


def sanitize_b3_source(source: Path, out: Path) -> Path:
    """Remove B3 resume metadata so a new disjoint phase-B corpus starts a fresh optimizer."""
    source_sha = sha(source)
    if out.is_file():
        payload = torch.load(out, map_location="cpu", weights_only=True)
        if payload.get("h1_rebased_from_sha256") == source_sha:
            return out
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if "model" not in payload or "config" not in payload:
        raise ValueError("B3 source missing model/config")
    tokens = int(payload.get("tokens", 0))
    if tokens < 450_000_000:
        raise ValueError(f"B3 source token count unexpectedly low: {tokens:,}")
    clean = {
        "model": payload["model"],
        "config": payload["config"],
        "tokens": tokens,
        "h1_rebased_from_sha256": source_sha,
        "h1_rebased_from": str(source.resolve()),
    }
    atomic_save(clean, out)
    return out


def ensure_sft(assets: Path) -> Path:
    data = assets / "i1_work_v3_oasst2" / "sft_data"
    required = ("train.jsonl", "val.jsonl", "test.jsonl", "holdout.jsonl", "manifest.json")
    if all((data / name).is_file() for name in required):
        return data
    # Reuse the exact provenance-controlled I1 builder; it fetches human OASST2 rows and
    # deterministic project-generated targeted rows while screening frozen-suite overlap.
    from run_50m_i1 import build_sft
    return build_sft(assets / "i1_work_v3_oasst2")


def benchmark(checkpoint: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    run([PY, "benchmark_bundle50.py", str(checkpoint), "--out-dir", str(out)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--reports", default="h1_reports")
    args = ap.parse_args()

    if os.name != "nt":
        raise RuntimeError("H1 intensive campaign expects the Windows RTX runner")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("H1 requires BF16 CUDA")

    assets = Path(args.assets).resolve()
    reports = (REPO / args.reports).resolve()
    reports.mkdir(parents=True, exist_ok=True)
    if not assets.is_dir():
        raise FileNotFoundError(assets)

    run_id = os.environ.get("GITHUB_RUN_ID", str(int(time.time())))
    state = {
        "campaign": "Neral-50M H1 1B-token intensive",
        "campaign_version": 1,
        "run_id": run_id,
        "started_at_unix": time.time(),
        "source_policy": "Approved Hugging Face human-web sources only; no foreign model weights or inference.",
        "benchmark_policy": "Frozen bundle is post-training evidence only and never selects checkpoints.",
        "phases": [],
    }
    write(reports / "H1_STATE.json", state)

    b1_data = assets / "pretrain50_data"
    b2_data = assets / "pretrain50_b2_data"
    b3_data = assets / "pretrain50_b3_data"
    for root, names in (
        (b1_data, ("val_de.bin", "val_en.bin", "documents.jsonl")),
        (b2_data, ("val_de.bin", "val_en.bin", "documents.jsonl", "manifest.json")),
        (b3_data, ("train_b_de.bin", "train_b_en.bin", "val_de.bin", "val_en.bin", "documents.jsonl", "manifest.json")),
    ):
        for name in names:
            if not (root / name).is_file():
                raise FileNotFoundError(root / name)
    junction(REPO / "pretrain50_data", b1_data)
    junction(REPO / "pretrain50_b2_data", b2_data)
    junction(REPO / "pretrain50_b3_data", b3_data)

    i1 = assets / "runs" / "neral50m_i1_instruct_d3" / "best_instruct.pt"
    if not i1.is_file() or sha(i1) != EXPECTED_SOURCE_SHA:
        raise RuntimeError("Immutable D3/I1 comparison checkpoint missing or changed")

    b3b_run = assets / "runs" / "neral50m_i1_b3b_500m"
    if not stage_complete(b3b_run / "summary.json"):
        raise RuntimeError("Completed I1 B3b source required")
    b3b = selected(b3b_run, "best_validation.pt")

    h1_data, data_report = ensure_h1_data(assets, b2_data, b3_data)
    write(reports / "H1_DATA.json", data_report)

    source = sanitize_b3_source(b3b, assets / "runs" / "neral50m_h1_rebased_b3b.pt")
    pre_run = assets / "runs" / "neral50m_h1_pretrain_1b"
    pre_summary = pre_run / "summary.json"
    if not stage_complete(pre_summary):
        resume = pre_run / "latest.pt" if (pre_run / "latest.pt").is_file() else source
        run([
            PY, "pretrain50_b3.py", "--phase", "b", "--resume", str(resume),
            "--data", str(h1_data), "--old-data", str(b3_data), "--out", str(pre_run),
            "--target-cumulative-tokens", "1000000000", "--seq", "1024", "--batch", "8",
            "--peak-lr", "0.000012", "--min-lr", "0.0000012", "--warmup-fraction", "0.02",
            "--eval-every", "250", "--early-eval-every", "100", "--early-eval-until", "1500",
            "--selection-min-delta", "0.0004", "--validation-blocks", "256",
        ])
    if not stage_complete(pre_summary):
        raise RuntimeError("H1 1B-token continued pretraining incomplete")
    pre_best = selected(pre_run, "best_validation.pt")
    state["phases"].append({"phase": "H1 continued pretraining", "checkpoint_sha256": sha(pre_best), "summary": read(pre_summary), "loss_telemetry": loss_snapshot(pre_run)})
    write(reports / "H1_STATE.json", state)

    raw_run = assets / "runs" / "neral50m_h1_rawmath"
    raw_summary = raw_run / "summary.json"
    if not stage_complete(raw_summary):
        run([
            PY, "train_raw_math50_i1.py", "--init-from", str(pre_best),
            "--replay-data", str(b2_data), "--out", str(raw_run),
            "--tokens", "40000000", "--seq", "256", "--batch", "32",
            "--math-fraction", "0.30", "--peak-lr", "0.000025", "--min-lr", "0.0000025",
            "--eval-every", "100", "--validation-blocks", "192",
        ])
    if not stage_complete(raw_summary):
        raise RuntimeError("H1 RawMath stage incomplete")
    raw_best = selected(raw_run, "best_math.pt")
    state["phases"].append({"phase": "H1 RawMath", "checkpoint_sha256": sha(raw_best), "summary": read(raw_summary)})
    write(reports / "H1_STATE.json", state)

    sft = ensure_sft(assets)
    inst_run = assets / "runs" / "neral50m_h1_instruct"
    inst_summary = inst_run / "summary.json"
    if not stage_complete(inst_summary):
        run([
            PY, "train_instruct50_i1.py", "--init-from", str(raw_best),
            "--data", str(sft), "--replay-data", str(b2_data), "--out", str(inst_run),
            "--steps", "3200", "--seq", "1024", "--batch-size", "8", "--replay-batch", "10",
            "--sft-weight", "0.76", "--eos-weight", "2.0",
            "--peak-lr", "0.000015", "--min-lr", "0.0000015", "--warmup-fraction", "0.05",
            "--eval-every", "100", "--patience", "16", "--min-delta", "0.0006",
        ])
    if not stage_complete(inst_summary):
        raise RuntimeError("H1 instruction stage incomplete")
    final = selected(inst_run, "best_instruct.pt")
    state["phases"].append({"phase": "H1 instruct", "checkpoint_sha256": sha(final), "summary": read(inst_summary)})
    state["candidate_sha256"] = sha(final)
    write(reports / "H1_STATE.json", state)

    benchmark(i1, reports / "before_i1")
    benchmark(final, reports / "after_h1")
    gate = run([
        PY, "h1_gate50.py", "--before-dir", str(reports / "before_i1"),
        "--after-dir", str(reports / "after_h1"), "--out", str(reports / "H1_GATE.json"),
    ], allow_failure=True)
    if gate not in (0, 2):
        raise RuntimeError(f"H1 gate execution error: {gate}")
    result = read(reports / "H1_GATE.json")
    final_report = {
        "campaign": state["campaign"],
        "complete": True,
        "candidate_sha256": sha(final),
        "frozen_gate_pass": gate == 0,
        "promotion_review_eligible": bool(result.get("promotion_review_eligible")),
        "nis_before": result["before"]["nis"],
        "nis_after": result["after"]["nis"],
        "nis_70_reached": result["stretch"]["nis_ge_70"],
        "next_action": result["next_action"],
        "foreign_weights": False,
        "external_model_calls": False,
    }
    write(reports / "H1_RESULT.json", final_report)
    print(json.dumps(final_report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
