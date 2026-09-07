"""Render detailed Neral benchmark values and dependency-free SVG graphics.

Consumes already-produced frozen benchmark reports. It does not load a model and never changes
weights. SVG is generated directly so the report has no matplotlib/pandas dependency.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


def load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf8")) if p.is_file() else None


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def fmt(v, digits=2):
    return "—" if not finite(v) else f"{float(v):.{digits}f}"


def pct(v):
    return None if not finite(v) else 100.0 * float(v)


def svg_bar(path: Path, title: str, rows, width=1100, row_h=46):
    rows = [(label, b, c, lo, hi, suffix) for label, b, c, lo, hi, suffix in rows if finite(b) or finite(c)]
    height = 100 + row_h * len(rows)
    left, right = 240, 90
    chart_w = width - left - right
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#0b0d12"/>',
           f'<text x="30" y="38" fill="#f5f7fb" font-family="system-ui,Segoe UI,sans-serif" font-size="24" font-weight="700">{html.escape(title)}</text>',
           '<text x="30" y="66" fill="#9aa4b2" font-family="system-ui,Segoe UI,sans-serif" font-size="13">Neral-20M Preview 02 = hell · aktueller 50M-Kandidat = dunkel</text>']
    y0 = 92
    for i, (label, b, c, lo, hi, suffix) in enumerate(rows):
        y = y0 + i * row_h
        out.append(f'<text x="30" y="{y+22}" fill="#dce2ea" font-family="system-ui,Segoe UI,sans-serif" font-size="14">{html.escape(label)}</text>')
        out.append(f'<rect x="{left}" y="{y+5}" width="{chart_w}" height="28" rx="6" fill="#151a22"/>')
        for value, yy, opacity in ((b, y+8, .65), (c, y+20, 1.0)):
            if finite(value) and hi > lo:
                frac = min(1.0, max(0.0, (float(value)-lo)/(hi-lo)))
                out.append(f'<rect x="{left}" y="{yy}" width="{max(2, chart_w*frac):.1f}" height="9" rx="4.5" fill="#d9e1ec" opacity="{opacity}"/>')
        btxt = "—" if not finite(b) else f"{b:.2f}{suffix}"
        ctxt = "—" if not finite(c) else f"{c:.2f}{suffix}"
        out.append(f'<text x="{width-right+8}" y="{y+15}" fill="#aeb8c6" font-family="ui-monospace,Consolas,monospace" font-size="12">{html.escape(btxt)}</text>')
        out.append(f'<text x="{width-right+8}" y="{y+30}" fill="#f5f7fb" font-family="ui-monospace,Consolas,monospace" font-size="12">{html.escape(ctxt)}</text>')
    out.append('</svg>')
    path.write_text("\n".join(out), encoding="utf8")


def svg_line(path: Path, title: str, xs, base, cand, y_label: str, lower_is_better=False, width=1100, height=460):
    valid = [v for v in base + cand if finite(v)]
    if not valid:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="800" height="200"><text x="20" y="50">No data</text></svg>', encoding="utf8")
        return
    ymin, ymax = min(valid), max(valid)
    pad = max((ymax-ymin)*.15, .05)
    ymin -= pad; ymax += pad
    left, right, top, bottom = 85, 45, 75, 70
    cw, ch = width-left-right, height-top-bottom
    def point(i, v):
        x = left + (cw * i / max(1, len(xs)-1))
        y = top + ch * (1 - (float(v)-ymin)/max(1e-9, ymax-ymin))
        return x, y
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         '<rect width="100%" height="100%" fill="#0b0d12"/>',
         f'<text x="30" y="38" fill="#f5f7fb" font-family="system-ui,Segoe UI,sans-serif" font-size="24" font-weight="700">{html.escape(title)}</text>',
         f'<text x="30" y="62" fill="#9aa4b2" font-family="system-ui,Segoe UI,sans-serif" font-size="13">{html.escape(y_label)} · {"niedriger ist besser" if lower_is_better else "höher ist besser"}</text>']
    for j in range(5):
        value = ymin + (ymax-ymin)*j/4
        y = top + ch*(1-j/4)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#202631"/>')
        out.append(f'<text x="15" y="{y+4:.1f}" fill="#8e99a8" font-family="ui-monospace,Consolas,monospace" font-size="11">{value:.3f}</text>')
    for i,xv in enumerate(xs):
        x = left + cw*i/max(1,len(xs)-1)
        out.append(f'<text x="{x:.1f}" y="{height-32}" text-anchor="middle" fill="#9aa4b2" font-family="ui-monospace,Consolas,monospace" font-size="12">{html.escape(str(xv))}</text>')
    for series, opacity, dash in ((base,.60,'6 5'),(cand,1.0,'')):
        pts=[]
        for i,v in enumerate(series):
            if finite(v): pts.append(point(i,v))
        if len(pts)>=2:
            p=' '.join(f'{x:.1f},{y:.1f}' for x,y in pts)
            out.append(f'<polyline points="{p}" fill="none" stroke="#e3e9f2" stroke-width="3" opacity="{opacity}" stroke-dasharray="{dash}"/>')
        for i,v in enumerate(series):
            if finite(v):
                x,y=point(i,v);out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#f5f7fb" opacity="{opacity}"/>')
    out.append(f'<text x="{width-300}" y="36" fill="#aeb8c6" font-family="system-ui,Segoe UI,sans-serif" font-size="12">gestrichelt = 20M · durchgezogen = 50M</text>')
    out.append('</svg>')
    path.write_text("\n".join(out), encoding="utf8")


def v5_bpb(report, seq):
    if not report: return None
    return report.get("document_bounded_true_byte_bpb",{}).get("macro_bits_per_byte",{}).get(str(seq))


def v5_needle(report, kind):
    if not report: return None
    return pct(report.get(kind,{}).get("accuracy"))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--comparison",required=True)
    ap.add_argument("--baseline-v5",required=True)
    ap.add_argument("--candidate-v5",required=True)
    ap.add_argument("--promotion")
    ap.add_argument("--target-gate")
    ap.add_argument("--quality-gate")
    ap.add_argument("--state")
    ap.add_argument("--out-dir",required=True)
    a=ap.parse_args()
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    comp=load(a.comparison);b5=load(a.baseline_v5);c5=load(a.candidate_v5)
    promotion=load(a.promotion) if a.promotion else None
    target=load(a.target_gate) if a.target_gate else None
    quality=load(a.quality_gate) if a.quality_gate else None
    state=load(a.state) if a.state and Path(a.state).is_file() else None
    b=comp.get("stages",{}).get("baseline20",{});c=comp.get("stages",{}).get("instruct",{})
    nis_b=comp.get("nis",{}).get("baseline20",{}).get("score");nis_c=comp.get("nis",{}).get("instruct",{}).get("score")

    core=[
        ("NIS 1.0",nis_b,nis_c,0,100,""),
        ("v3 Gesamt",b.get("v3_pair_overall_pct"),c.get("v3_pair_overall_pct"),0,100,"%"),
        ("v4 Gesamt",b.get("v4_pair_overall_pct"),c.get("v4_pair_overall_pct"),0,100,"%"),
        ("RawMath 500 exact",b.get("rawmath_v1_exact_pct"),c.get("rawmath_v1_exact_pct"),0,100,"%"),
        ("Kontext-Recall v4",b.get("context_recall_overall_pct"),c.get("context_recall_overall_pct"),0,100,"%"),
        ("Targeted v1 Gesamt",b.get("targeted_overall_pct"),c.get("targeted_overall_pct"),0,100,"%"),
    ]
    svg_bar(out/"benchmark_core.svg","Neral 20M vs 50M · Kernmetriken",core)
    targeted=[
        ("Deutsch Grammatik",b.get("targeted_de_grammar_pct"),c.get("targeted_de_grammar_pct"),0,100,"%"),
        ("Mathematik",b.get("targeted_math_pct"),c.get("targeted_math_pct"),0,100,"%"),
        ("Code",b.get("targeted_code_pct"),c.get("targeted_code_pct"),0,100,"%"),
        ("Reasoning",b.get("targeted_reasoning_pct"),c.get("targeted_reasoning_pct"),0,100,"%"),
    ]
    svg_bar(out/"benchmark_targeted.svg","Targeted v1 · Deutsch / Mathe / Code / Reasoning",targeted)
    health=[
        ("UTF-8 gültig",b.get("utf8_valid_pct"),c.get("utf8_valid_pct"),0,100,"%"),
        ("EOS / selbstständiges Stoppen",b.get("eos_pct"),c.get("eos_pct"),0,100,"%"),
        ("Trigramm-Wiederholung ↓",b.get("repeated_trigram_pct"),c.get("repeated_trigram_pct"),0,100,"%"),
        ("v4 greedy EOS",b.get("v4_greedy_eos_pct"),c.get("v4_greedy_eos_pct"),0,100,"%"),
    ]
    svg_bar(out/"benchmark_generation_health.svg","Generierungsqualität",health)

    seqs=[512,1024,2048,3072,4095]
    svg_line(out/"benchmark_long_context_bpb.svg","Long-Context · True-byte BPB",seqs,[v5_bpb(b5,s) for s in seqs],[v5_bpb(c5,s) for s in seqs],"bits pro Byte",True)
    needle_rows=[
        ("Single-needle",v5_needle(b5,"single_needle"),v5_needle(c5,"single_needle"),0,100,"%"),
        ("Multi-needle",v5_needle(b5,"multi_needle"),v5_needle(c5,"multi_needle"),0,100,"%"),
    ]
    svg_bar(out/"benchmark_long_context_recall.svg","Long-Context · Needle Recall",needle_rows)

    directions={
        "utf8_valid_pct":"higher","eos_pct":"higher","repeated_trigram_pct":"lower","rawmath_v1_exact_pct":"higher",
        "v3_pair_overall_pct":"higher","v4_pair_overall_pct":"higher","context_recall_overall_pct":"higher","true_byte_bpb_macro_1024":"lower",
        "v4_math_pct":"higher","v4_code_pct":"higher","targeted_overall_pct":"higher","targeted_de_grammar_pct":"higher","targeted_math_pct":"higher","targeted_code_pct":"higher","targeted_reasoning_pct":"higher"
    }
    rows=[]
    keys=sorted(set(b)|set(c))
    for key in keys:
        bv,cv=b.get(key),c.get(key)
        if finite(bv) or finite(cv):
            rows.append((key,bv,cv,(cv-bv) if finite(bv) and finite(cv) else None,directions.get(key,"diagnostic")))
    for seq in seqs:
        bv,cv=v5_bpb(b5,seq),v5_bpb(c5,seq)
        rows.append((f"v5_bpb_{seq}",bv,cv,(cv-bv) if finite(bv) and finite(cv) else None,"lower"))
    for kind in ("single_needle","multi_needle"):
        bv,cv=v5_needle(b5,kind),v5_needle(c5,kind)
        rows.append((f"v5_{kind}_pct",bv,cv,(cv-bv) if finite(bv) and finite(cv) else None,"higher"))
    with (out/"benchmark_values.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.writer(f);w.writerow(["metric","baseline20","candidate50","delta_candidate_minus_baseline","direction"]);w.writerows(rows)

    important=[
        ("NIS 1.0",nis_b,nis_c,"higher"),
        ("v3 Gesamt",b.get("v3_pair_overall_pct"),c.get("v3_pair_overall_pct"),"higher"),
        ("v4 Gesamt",b.get("v4_pair_overall_pct"),c.get("v4_pair_overall_pct"),"higher"),
        ("RawMath exact",b.get("rawmath_v1_exact_pct"),c.get("rawmath_v1_exact_pct"),"higher"),
        ("RawMath Transfer/OOD",b.get("rawmath_transfer_ood_pct"),c.get("rawmath_transfer_ood_pct"),"higher"),
        ("v4 Mathe",b.get("v4_math_pct"),c.get("v4_math_pct"),"higher"),
        ("v4 Code",b.get("v4_code_pct"),c.get("v4_code_pct"),"higher"),
        ("Kontext-Recall",b.get("context_recall_overall_pct"),c.get("context_recall_overall_pct"),"higher"),
        ("EOS",b.get("eos_pct"),c.get("eos_pct"),"higher"),
        ("Trigramm-Wiederholung",b.get("repeated_trigram_pct"),c.get("repeated_trigram_pct"),"lower"),
        ("Targeted Gesamt",b.get("targeted_overall_pct"),c.get("targeted_overall_pct"),"higher"),
        ("Targeted Deutsch",b.get("targeted_de_grammar_pct"),c.get("targeted_de_grammar_pct"),"higher"),
        ("Targeted Mathe",b.get("targeted_math_pct"),c.get("targeted_math_pct"),"higher"),
        ("Targeted Code",b.get("targeted_code_pct"),c.get("targeted_code_pct"),"higher"),
        ("Targeted Reasoning",b.get("targeted_reasoning_pct"),c.get("targeted_reasoning_pct"),"higher"),
        ("v5 Single-needle",v5_needle(b5,"single_needle"),v5_needle(c5,"single_needle"),"higher"),
        ("v5 Multi-needle",v5_needle(b5,"multi_needle"),v5_needle(c5,"multi_needle"),"higher"),
    ]
    md=["# Neral ausführlicher Benchmarkbericht","",
        "Dieser Bericht vergleicht Neral-20M Preview 02 mit dem ausgewählten Neral-50M-Kandidaten. Alle Modellwerte stammen aus den festgelegten model-only Harnesses; SOLVE, Helios-Tools und externe Modelle sind ausgeschlossen.","",
        "## Grafiken","",
        "![Kernmetriken](benchmark_core.svg)","","![Targeted](benchmark_targeted.svg)","","![Generierungsqualität](benchmark_generation_health.svg)","","![Long Context BPB](benchmark_long_context_bpb.svg)","","![Long Context Recall](benchmark_long_context_recall.svg)","",
        "## Wichtigste Werte","","| Metrik | 20M | 50M | Δ 50M−20M | Richtung |","|---|---:|---:|---:|---|"]
    for label,bv,cv,direction in important:
        d=(cv-bv) if finite(bv) and finite(cv) else None
        md.append(f"| {label} | {fmt(bv)} | {fmt(cv)} | {fmt(d)} | {'↑' if direction=='higher' else '↓'} |")
    md += ["","## Long-Context BPB","","| Kontext | 20M bits/Byte | 50M bits/Byte | Δ |","|---:|---:|---:|---:|"]
    for seq in seqs:
        bv,cv=v5_bpb(b5,seq),v5_bpb(c5,seq);d=(cv-bv) if finite(bv) and finite(cv) else None
        md.append(f"| {seq} | {fmt(bv,3)} | {fmt(cv,3)} | {fmt(d,3)} |")
    md += ["","## Gates",""]
    if promotion:
        md.append(f"- Legacy Promotion Gate: **{'PASS' if promotion.get('ready_for_rc_review') else 'FAIL'}** ({promotion.get('strict_wins','?')}/{promotion.get('strict_wins_required','?')} strict wins)")
    if target:
        md.append(f"- I1 Target Gate: **{'PASS' if target.get('targets_met') else 'FAIL'}**")
    if quality:
        md.append(f"- Supplemental Quality Gate: **{'PASS' if quality.get('quality_targets_met') else 'FAIL'}**")
    md += ["","## Vollständige numerische Tabelle","","Alle numerischen Metriken inklusive zusätzlicher Domänenwerte stehen in `benchmark_values.csv`. Negative BPB-Deltas sind Verbesserungen; bei Wiederholungsmetriken ist ebenfalls niedriger besser.",""]
    if state:
        md += ["## Trainingsstand",""]
        md.append(f"Campaign: `{state.get('campaign','Neral-50M')}` · Version `{state.get('campaign_version','?')}`")
        for phase in state.get("phases",[]):
            s=phase.get("summary",{});md.append(f"- {phase.get('phase')}: complete={s.get('complete')} · cumulative_tokens={s.get('cumulative_tokens','—')} · best_validation={s.get('best_validation',s.get('best_math_dev_nll',s.get('best_sft_macro_answer_nll','—')))}")
    md += ["","## Methodische Einordnung","",
           "Targeted v1 ist ein zusätzlicher Diagnose-Benchmark, der nach Beginn von B3 eingefroren wurde. Er wird deshalb nicht rückwirkend als kontaminationssicherer Release-Benchmark behandelt und fließt nicht in NIS 1.0 ein. Die alten Promotion-Gates bleiben entscheidend."]
    (out/"BENCHMARK_REPORT.md").write_text("\n".join(md)+"\n",encoding="utf8")
    summary={"nis20":nis_b,"nis50":nis_c,"nis_delta":nis_c-nis_b if finite(nis_b) and finite(nis_c) else None,
             "promotion_ready":promotion.get("ready_for_rc_review") if promotion else None,
             "targets_met":target.get("targets_met") if target else None,
             "quality_targets_met":quality.get("quality_targets_met") if quality else None,
             "charts":["benchmark_core.svg","benchmark_targeted.svg","benchmark_generation_health.svg","benchmark_long_context_bpb.svg","benchmark_long_context_recall.svg"],
             "csv":"benchmark_values.csv","markdown":"BENCHMARK_REPORT.md"}
    (out/"benchmark_report_manifest.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__":main()
