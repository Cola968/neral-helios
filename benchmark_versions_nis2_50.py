"""Run NIS 2.0 across every locally available completed Neral lineage checkpoint.

This is evaluation-only. Frozen benchmarks are not used for checkpoint selection or training.
The report emits JSON/CSV/Markdown and a dependency-free SVG trajectory chart.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from neral_intelligence_score import components_from_reports as nis1_components
from neral_intelligence_score import score_from_components as nis1_score
from neral_intelligence_score_v2 import score_report_dir

REPO = Path(__file__).resolve().parent
PY = sys.executable


@dataclass(frozen=True)
class VersionSpec:
    order: int
    key: str
    label: str
    checkpoint_rel: str
    summary_rel: str | None = None


VERSIONS = (
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
    VersionSpec(100, "m2_original", "50M M2 original", "runs/neral50m_m2_transactional_34240905142/best_m2.pt", "runs/neral50m_m2_transactional_34240905142/summary.json"),
    VersionSpec(110, "m21_guarded", "50M M2.1 guarded", "runs/neral50m_m21_transactional_34255600571/best_m2.pt", "runs/neral50m_m21_transactional_34255600571/summary.json"),
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def complete(spec: VersionSpec, assets: Path) -> tuple[bool, str]:
    cp = assets / spec.checkpoint_rel
    if not cp.is_file():
        return False, "checkpoint_missing"
    if spec.summary_rel is None:
        return True, "checkpoint_present"
    summary = assets / spec.summary_rel
    if not summary.is_file():
        return False, "summary_missing"
    try:
        if read_json(summary).get("complete") is not True:
            return False, "summary_not_complete"
    except Exception:
        return False, "summary_unreadable"
    return True, "complete"


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
        raise RuntimeError("NIS2 lineage benchmark expects the Windows self-hosted Neral runner")
    q = lambda p: "'" + str(p).replace("'", "''") + "'"
    run("powershell", "-NoProfile", "-NonInteractive", "-Command", f"New-Item -ItemType Junction -Path {q(link)} -Target {q(target)} -ErrorAction Stop | Out-Null")


def fmt(v) -> str:
    return "—" if v is None else f"{float(v):.3f}".rstrip("0").rstrip(".")


def write_svg(rows: list[dict], path: Path) -> None:
    width, height = 1280, 640
    left, right, top, bottom = 90, 35, 55, 150
    pw, ph = width-left-right, height-top-bottom
    scores = [float(r["nis2"]["score"]) for r in rows]
    low = max(0.0, min(scores)-8.0); high = min(100.0, max(scores)+8.0)
    if high-low < 20:
        mid=(high+low)/2; low=max(0,mid-10); high=min(100,mid+10)
    def x(i): return left + (pw * i / max(1, len(rows)-1))
    def y(v): return top + ph * (high-v) / max(1e-9, high-low)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="90" y="32" font-family="sans-serif" font-size="24" font-weight="600">Neral NIS 2.0 – Entwicklung über alle verfügbaren Modelle</text>']
    for j in range(6):
        val=low+(high-low)*j/5; yy=y(val)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left-12}" y="{yy+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="13">{val:.1f}</text>')
    pts=' '.join(f'{x(i):.1f},{y(v):.1f}' for i,v in enumerate(scores))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="black" stroke-width="3"/>')
    for i,(row,v) in enumerate(zip(rows,scores)):
        xx,yy=x(i),y(v)
        parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5" fill="black"/>')
        parts.append(f'<text x="{xx:.1f}" y="{yy-12:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{v:.2f}</text>')
        label=row["label"].replace("&","&amp;")
        parts.append(f'<text transform="translate({xx:.1f},{height-bottom+24}) rotate(35)" font-family="sans-serif" font-size="12">{label}</text>')
    parts.append(f'<text x="20" y="{top+ph/2:.1f}" transform="rotate(-90 20 {top+ph/2:.1f})" text-anchor="middle" font-family="sans-serif" font-size="15">NIS 2.0 (0–100)</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()
    assets=Path(args.assets).resolve(); out=Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    junction(REPO/"pretrain50_data", assets/"pretrain50_data")

    rows=[]; skipped=[]
    for spec in VERSIONS:
        ok,reason=complete(spec, assets)
        if not ok:
            skipped.append({"key":spec.key,"label":spec.label,"reason":reason}); continue
        checkpoint=assets/spec.checkpoint_rel
        report=out/"versions"/spec.key; report.mkdir(parents=True, exist_ok=True)
        run(PY,"benchmark_bundle50.py",str(checkpoint),"--out-dir",str(report))
        run(PY,"benchmark_nis2_chat50.py",str(checkpoint),"--out",str(report/"nis2_chat.json"))
        n1c=nis1_components(report); n1=float(nis1_score(n1c))
        n2=score_report_dir(report)
        rows.append({"order":spec.order,"key":spec.key,"label":spec.label,"checkpoint":str(checkpoint),"nis1":n1,"nis2":n2})

    rows.sort(key=lambda r:r["order"])
    result={
        "definition":"NIS1 and full NIS2 on identical frozen model-only suites; no tools/external model/training feedback.",
        "versions":rows,"skipped":skipped,
    }
    (out/"NIS2_VERSION_MATRIX.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    with (out/"NIS2_VERSION_MATRIX.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["order","key","label","nis1","nis2","bottleneck","bottleneck_value"])
        for r in rows:
            b=r["nis2"]["bottleneck"]; w.writerow([r["order"],r["key"],r["label"],r["nis1"],r["nis2"]["score"],b["component"],b["value"]])
    md=["# NIS 2.0 lineage matrix","","| Modell | NIS 1.0 | NIS 2.0 | Engpass |","|---|---:|---:|---|"]
    for r in rows:
        b=r["nis2"]["bottleneck"]; md.append(f"| {r['label']} | {fmt(r['nis1'])} | {fmt(r['nis2']['score'])} | {b['component']} ({fmt(b['value'])}) |")
    if skipped:
        md += ["","## Übersprungen",""] + [f"- {r['label']}: {r['reason']}" for r in skipped]
    (out/"NIS2_VERSION_MATRIX.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    if rows: write_svg(rows,out/"NIS2_TRAJECTORY.svg")
    print(json.dumps({"benchmarked":[r["key"] for r in rows],"skipped":skipped,"out":str(out)},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
