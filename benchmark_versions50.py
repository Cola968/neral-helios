"""Benchmark the Neral lineage on one identical frozen model-only bundle.

This is an evaluation orchestrator only. It never trains, mutates weights, calls Helios/SOLVE,
or uses an external model. Every locally available completed checkpoint is benchmarked with the
same current harness, then summarized into JSON, CSV and Markdown for stage-by-stage comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from m1_gate50 import HIGHER_IS_BETTER, LOWER_IS_BETTER, pareto_compare
from weakness_repair_gate50 import extract

REPO = Path(__file__).resolve().parent
PY = sys.executable


@dataclass(frozen=True)
class VersionSpec:
    order: int
    key: str
    label: str
    checkpoint_rel: str
    summary_rel: str | None = None


VERSION_SPECS = (
    VersionSpec(10, "preview02_20m", "20M Preview02", "export/pilot_02/neral20m.pt"),
    VersionSpec(20, "b2_100m", "50M B2 100M", "runs/neral50m_b2_100m/best_validation.pt", "runs/neral50m_b2_100m/summary.json"),
    VersionSpec(30, "b3a_300m", "50M B3a 300M", "runs/neral50m_i1_b3a_300m/best_validation.pt", "runs/neral50m_i1_b3a_300m/summary.json"),
    VersionSpec(40, "b3b_500m", "50M B3b 500M", "runs/neral50m_i1_b3b_500m/best_validation.pt", "runs/neral50m_i1_b3b_500m/summary.json"),
    VersionSpec(50, "c3_rawmath", "50M C3 RawMath", "runs/neral50m_i1_rawmath_c3/best_math.pt", "runs/neral50m_i1_rawmath_c3/summary.json"),
    VersionSpec(60, "d3_i1", "50M D3 / I1", "runs/neral50m_i1_instruct_d3/best_instruct.pt", "runs/neral50m_i1_instruct_d3/summary.json"),
    VersionSpec(70, "i2_e1", "50M I2 E1", "runs/neral50m_i2_weakness_e1/best_repair.pt", "runs/neral50m_i2_weakness_e1/summary.json"),
    VersionSpec(80, "i2_e2", "50M I2 E2", "runs/neral50m_i2_weakness_e2/best_repair.pt", "runs/neral50m_i2_weakness_e2/summary.json"),
    VersionSpec(90, "m1_i1", "50M M1 from I1", "runs/neral50m_math_generalization_m1_i1_instruct/best_math_generalization.pt", "runs/neral50m_math_generalization_m1_i1_instruct/summary.json"),
    VersionSpec(91, "m1_e2", "50M M1 from I2 E2", "runs/neral50m_math_generalization_m1_i2_e2/best_math_generalization.pt", "runs/neral50m_math_generalization_m1_i2_e2/summary.json"),
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def completed(spec: VersionSpec, assets: Path) -> tuple[bool, str]:
    checkpoint = assets / spec.checkpoint_rel
    if not checkpoint.is_file():
        return False, "checkpoint_missing"
    if spec.summary_rel is None:
        return True, "checkpoint_present"
    summary_path = assets / spec.summary_rel
    if not summary_path.is_file():
        return False, "summary_missing"
    summary = read_json(summary_path)
    if summary.get("complete") is not True:
        return False, "summary_not_complete"
    return True, "complete"


def available_versions(assets: Path) -> list[VersionSpec]:
    return [spec for spec in VERSION_SPECS if completed(spec, assets)[0]]


def run(*args: str) -> None:
    cmd = [str(x) for x in args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def junction(link: Path, target: Path) -> None:
    link, target = link.absolute(), target.resolve()
    if not target.is_dir():
        raise FileNotFoundError(target)
    if link.exists():
        try:
            if link.resolve() == target:
                return
        except OSError:
            pass
        raise FileExistsError(f"Refusing to replace existing path: {link}")
    if os.name != "nt":
        raise RuntimeError("Version matrix currently expects the Windows self-hosted Neral runner")
    q = lambda p: "'" + str(p).replace("'", "''") + "'"
    run("powershell", "-NoProfile", "-NonInteractive", "-Command", f"New-Item -ItemType Junction -Path {q(link)} -Target {q(target)} -ErrorAction Stop | Out-Null")


def best_by_metric(rows: list[dict]) -> dict:
    result: dict[str, dict] = {}
    for key in HIGHER_IS_BETTER:
        candidates = [(row["metrics"].get(key), row) for row in rows if row["metrics"].get(key) is not None]
        if candidates:
            value, row = max(candidates, key=lambda item: float(item[0]))
            result[key] = {"key": row["key"], "label": row["label"], "value": float(value), "direction": "higher"}
    for key in LOWER_IS_BETTER:
        candidates = [(row["metrics"].get(key), row) for row in rows if row["metrics"].get(key) is not None]
        if candidates:
            value, row = min(candidates, key=lambda item: float(item[0]))
            result[key] = {"key": row["key"], "label": row["label"], "value": float(value), "direction": "lower"}
    return result


def write_csv(rows: list[dict], path: Path) -> None:
    metrics = list(HIGHER_IS_BETTER) + list(LOWER_IS_BETTER)
    with path.open("w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["order", "key", "label", *metrics])
        for row in rows:
            writer.writerow([row["order"], row["key"], row["label"], *[row["metrics"].get(k) for k in metrics]])


def fmt(value) -> str:
    if value is None:
        return "—"
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def write_markdown(rows: list[dict], comparisons: list[dict], path: Path) -> None:
    cols = (
        ("NIS", "nis"), ("RawMath", "rawmath_exact_pct"), ("Transfer", "rawmath_transfer_pct"),
        ("OOD", "rawmath_ood_pct"), ("EOS", "free_eos_pct"), ("Rep↓", "free_repeated_trigram_pct"),
        ("Code v4", "v4_code_pct"), ("Context", "context_recall_pct"), ("DE", "targeted_de_pct"),
        ("Reasoning", "targeted_reasoning_pct"), ("BPB4095↓", "v5_bpb_4095"),
    )
    lines = [
        "# Neral multi-version benchmark matrix",
        "",
        "Every row uses the identical current frozen model-only benchmark bundle. No training or tools are used by this report.",
        "",
        "| Version | " + " | ".join(name for name, _ in cols) + " |",
        "|---|" + "|".join("---:" for _ in cols) + "|",
    ]
    for row in rows:
        lines.append("| " + row["label"] + " | " + " | ".join(fmt(row["metrics"].get(key)) for _, key in cols) + " |")
    lines += ["", "## Stage-to-stage Pareto checks", ""]
    for item in comparisons:
        status = "PASS" if item["pareto"]["pass"] else "FAIL"
        regressions = ", ".join(item["pareto"]["regressions"]) or "none"
        improvements = ", ".join(item["pareto"]["improvements"]) or "none"
        lines.append(f"- **{item['from_label']} → {item['to_label']}**: {status}; improvements: {improvements}; regressions: {regressions}.")
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assets = Path(args.assets).resolve()
    out = Path(args.out).resolve()
    if not assets.is_dir():
        raise FileNotFoundError(assets)
    out.mkdir(parents=True, exist_ok=True)

    b1 = assets / "pretrain50_data"
    for name in ("documents.jsonl", "val_de.bin", "val_en.bin"):
        if not (b1 / name).is_file():
            raise FileNotFoundError(b1 / name)
    junction(REPO / "pretrain50_data", b1)

    specs = available_versions(assets)
    if len(specs) < 2:
        raise RuntimeError("Need at least two completed Neral versions for a version matrix")

    rows: list[dict] = []
    skipped = []
    for spec in VERSION_SPECS:
        ok, reason = completed(spec, assets)
        if not ok:
            skipped.append({"key": spec.key, "label": spec.label, "reason": reason})
            continue
        checkpoint = assets / spec.checkpoint_rel
        bundle = out / "versions" / spec.key
        run(PY, "benchmark_bundle50.py", str(checkpoint), "--out-dir", str(bundle))
        manifest = read_json(bundle / "bundle_manifest.json")
        rows.append({
            "order": spec.order,
            "key": spec.key,
            "label": spec.label,
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "metrics": extract(bundle),
        })

    rows.sort(key=lambda row: row["order"])
    comparisons = []
    for before, after in zip(rows, rows[1:]):
        comparisons.append({
            "from_key": before["key"], "from_label": before["label"],
            "to_key": after["key"], "to_label": after["label"],
            "pareto": pareto_compare(before["metrics"], after["metrics"]),
        })

    result = {
        "definition": "Identical current frozen model-only bundle across completed Neral versions; no training/tools/external model.",
        "created_at_unix": time.time(),
        "versions": rows,
        "skipped": skipped,
        "stage_comparisons": comparisons,
        "best_by_metric": best_by_metric(rows),
    }
    (out / "VERSION_MATRIX.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    write_csv(rows, out / "VERSION_MATRIX.csv")
    write_markdown(rows, comparisons, out / "VERSION_MATRIX.md")
    print(json.dumps({"versions": [row["key"] for row in rows], "skipped": skipped, "out": str(out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
