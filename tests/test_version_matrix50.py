from pathlib import Path

from benchmark_versions50 import VersionSpec, best_by_metric, completed


def test_completed_requires_summary_when_declared(tmp_path: Path):
    spec = VersionSpec(1, "x", "X", "runs/x.pt", "runs/summary.json")
    checkpoint = tmp_path / "runs/x.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")
    assert completed(spec, tmp_path) == (False, "summary_missing")
    (tmp_path / "runs/summary.json").write_text('{"complete": false}', encoding="utf8")
    assert completed(spec, tmp_path) == (False, "summary_not_complete")
    (tmp_path / "runs/summary.json").write_text('{"complete": true}', encoding="utf8")
    assert completed(spec, tmp_path) == (True, "complete")


def test_best_by_metric_respects_direction():
    rows = [
        {"key": "a", "label": "A", "metrics": {"nis": 60.0, "free_repeated_trigram_pct": 5.0, "v5_bpb_4095": 1.7}},
        {"key": "b", "label": "B", "metrics": {"nis": 65.0, "free_repeated_trigram_pct": 4.0, "v5_bpb_4095": 1.8}},
    ]
    result = best_by_metric(rows)
    assert result["nis"]["key"] == "b"
    assert result["free_repeated_trigram_pct"]["key"] == "b"
    assert result["v5_bpb_4095"]["key"] == "a"
