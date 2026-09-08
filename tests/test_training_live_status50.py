import json
from pathlib import Path

from training_live_status50 import collect_status, compact_line, read_last_jsonl


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


def append_jsonl(path: Path, *rows: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_reads_only_last_complete_jsonl_row(tmp_path):
    p = tmp_path / "metrics.jsonl"
    append_jsonl(p, {"step": 1}, {"step": 2, "train_loss": 1.25})
    with p.open("ab") as f:
        f.write(b'{"step":3')
    assert read_last_jsonl(p) == {"step": 2, "train_loss": 1.25}


def test_b3a_status_uses_existing_metrics_without_touching_training(tmp_path):
    run = tmp_path / "runs/neral50m_i1_b3a_300m"
    write_json(run / "run.json", {"source_cumulative_tokens": 100_000_000})
    append_jsonl(
        run / "metrics.jsonl",
        {
            "step": 120,
            "cumulative_tokens": 150_000_000,
            "train_loss": 1.12,
            "tokens_per_second": 75_000,
            "validation": {"macro": 1.18},
            "old_validation": {"macro": 1.16},
        },
    )
    status = collect_status(tmp_path, include_gpu=False)
    assert status["stage"] == "B3a"
    assert status["current"] == 150_000_000
    assert status["progress_pct"] == 25.0
    assert status["train_loss"] == 1.12
    assert status["validation_loss"] == 1.18
    assert status["tokens_per_second"] == 75_000
    assert status["eta_seconds_optimizer_only"] == 2000.0
    assert "stage=B3a" in compact_line(status)


def test_moves_to_b3b_after_b3a_summary_is_complete(tmp_path):
    b3a = tmp_path / "runs/neral50m_i1_b3a_300m"
    write_json(b3a / "summary.json", {"complete": True, "cumulative_tokens": 300_000_000})
    b3b = tmp_path / "runs/neral50m_i1_b3b_500m"
    write_json(b3b / "run.json", {"source_cumulative_tokens": 300_000_000})
    append_jsonl(
        b3b / "metrics.jsonl",
        {"step": 10, "cumulative_tokens": 320_000_000, "train_loss": 1.08, "tokens_per_second": 60_000},
    )
    status = collect_status(tmp_path, include_gpu=False)
    assert status["stage"] == "B3b"
    assert status["progress_pct"] == 10.0


def test_rawmath_and_instruct_schema_are_understood(tmp_path):
    for name, tokens in (("neral50m_i1_b3a_300m", 300_000_000), ("neral50m_i1_b3b_500m", 500_000_000)):
        write_json(tmp_path / f"runs/{name}/summary.json", {"complete": True, "cumulative_tokens": tokens})
    raw = tmp_path / "runs/neral50m_i1_rawmath_c3"
    append_jsonl(raw / "metrics.jsonl", {"step": 50, "stage_c_tokens": 10_000_000, "train_loss": 0.9, "math_dev": {"nll": 1.1}})
    status = collect_status(tmp_path, include_gpu=False)
    assert status["stage"] == "RawMath C3"
    assert round(status["progress_pct"], 2) == 33.33
    assert status["math_dev_nll"] == 1.1

    write_json(raw / "summary.json", {"complete": True, "stage_c_tokens": 30_000_000})
    inst = tmp_path / "runs/neral50m_i1_instruct_d3"
    append_jsonl(inst / "metrics.jsonl", {"step": 1000, "loss": 0.75, "optimizer_tokens_per_second": 12_000})
    status = collect_status(tmp_path, include_gpu=False)
    assert status["stage"] == "Instruct D3"
    assert status["progress_pct"] == 25.0
    assert status["train_loss"] == 0.75
