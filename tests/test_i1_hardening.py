from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from benchmark_targeted50 import load_suite
from i1_quality_gate50 import evaluate as quality_evaluate
from run_current_benchmark50 import choose_candidate


def test_targeted_suite_is_frozen_balanced_80_cases():
    rows = load_suite()
    assert len(rows) == 80
    counts = Counter(r["domain"] for r in rows)
    assert counts == {"de_grammar": 20, "math": 20, "code": 20, "reasoning": 20}
    assert len({r["id"] for r in rows}) == 80


def test_quality_gate_requires_suite_match_and_all_domains():
    domains = {d: {"accuracy": .60} for d in ("de_grammar", "math", "code", "reasoning")}
    baseline = {"suite_sha256": "x", "overall": {"accuracy": .55}, "domains": domains}
    candidate = {"suite_sha256": "x", "overall": {"accuracy": .65}, "domains": domains}
    comparison = {"targeted_suite_sha256": "x", "nis": {"instruct": {"score": 56.0}}}
    promotion = {"ready_for_rc_review": True}
    assert quality_evaluate(comparison, promotion, baseline, candidate)["quality_targets_met"]
    candidate = json.loads(json.dumps(candidate));candidate["domains"]["code"]["accuracy"] = .49
    assert not quality_evaluate(comparison, promotion, baseline, candidate)["quality_targets_met"]


def test_candidate_locator_prefers_most_advanced_selected_checkpoint(tmp_path: Path):
    b2 = tmp_path / "runs/neral50m_b2_100m/best_validation.pt";b2.parent.mkdir(parents=True);b2.write_bytes(b"x")
    stage, path = choose_candidate(tmp_path);assert stage == "b2_100m" and path == b2
    b3 = tmp_path / "runs/neral50m_i1_b3a_300m/best_validation.pt";b3.parent.mkdir(parents=True);b3.write_bytes(b"x")
    stage, path = choose_candidate(tmp_path);assert stage == "b3a_300m" and path == b3


def test_report_renderer_emits_markdown_csv_and_five_svgs(tmp_path: Path):
    base = {"v3_pair_overall_pct": 50.0, "v4_pair_overall_pct": 60.0, "rawmath_v1_exact_pct": 20.0,
            "context_recall_overall_pct": 50.0, "utf8_valid_pct": 100.0, "eos_pct": 20.0,
            "repeated_trigram_pct": 8.0, "v4_math_pct": 50.0, "v4_code_pct": 50.0,
            "targeted_overall_pct": 50.0, "targeted_de_grammar_pct": 50.0, "targeted_math_pct": 50.0,
            "targeted_code_pct": 50.0, "targeted_reasoning_pct": 50.0}
    cand = dict(base);cand.update({"v3_pair_overall_pct": 62.0, "rawmath_v1_exact_pct": 61.0, "eos_pct": 70.0,
                                  "repeated_trigram_pct": 3.0, "targeted_overall_pct": 65.0})
    comparison = {"stages": {"baseline20": base, "instruct": cand}, "nis": {"baseline20": {"score": 44.0}, "instruct": {"score": 56.0}}}
    v5 = {"single_needle": {"accuracy": .6}, "multi_needle": {"accuracy": .5},
          "document_bounded_true_byte_bpb": {"macro_bits_per_byte": {str(s): 3.0 - i*.1 for i,s in enumerate((512,1024,2048,3072,4095))}}}
    c5 = json.loads(json.dumps(v5));c5["single_needle"]["accuracy"] = .8;c5["document_bounded_true_byte_bpb"]["macro_bits_per_byte"]["4095"] = 2.4
    files = {"comparison.json": comparison, "b5.json": v5, "c5.json": c5,
             "promotion.json": {"ready_for_rc_review": True, "strict_wins": 8, "strict_wins_required": 8},
             "target.json": {"targets_met": True}, "quality.json": {"quality_targets_met": True}}
    for name, obj in files.items():(tmp_path/name).write_text(json.dumps(obj), encoding="utf8")
    subprocess.run([sys.executable, "render_benchmark_report50.py", "--comparison", str(tmp_path/"comparison.json"),
                    "--baseline-v5", str(tmp_path/"b5.json"), "--candidate-v5", str(tmp_path/"c5.json"),
                    "--promotion", str(tmp_path/"promotion.json"), "--target-gate", str(tmp_path/"target.json"),
                    "--quality-gate", str(tmp_path/"quality.json"), "--out-dir", str(tmp_path/"out")], check=True)
    out = tmp_path/"out"
    assert (out/"BENCHMARK_REPORT.md").is_file()
    assert (out/"benchmark_values.csv").is_file()
    assert len(list(out.glob("*.svg"))) == 5
    assert "Neral ausführlicher Benchmarkbericht" in (out/"BENCHMARK_REPORT.md").read_text(encoding="utf8")
