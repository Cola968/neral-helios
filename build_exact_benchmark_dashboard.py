from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "evidence" / "reports"
OUT = ROOT / "benchmark_dashboard"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf8"))


def pct(x):
    return 100.0 * float(x)


def safe_mean(values):
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else None


def summarize_rows(rows):
    by_domain = defaultdict(list)
    by_language = defaultdict(list)
    for r in rows:
        by_domain[r["domain"]].append(r)
        by_language[r["language"]].append(r)

    def one(group):
        n = len(group)
        numeric = [r for r in group if "numeric_exact" in r]
        return {
            "n": n,
            "utf8_valid_pct": 100.0 * sum(r["diagnostics"]["valid_utf8"] for r in group) / n,
            "eos_pct": 100.0 * sum(r["diagnostics"]["eos"] for r in group) / n,
            "repeated_trigram_pct": 100.0 * safe_mean(r["diagnostics"]["repeated_word_ngrams"]["3"] for r in group),
            "non_repetition_trigram_pct": 100.0 - 100.0 * safe_mean(r["diagnostics"]["repeated_word_ngrams"]["3"] for r in group),
            "decode_tokens_per_second": safe_mean(r["tokens_per_second"] for r in group),
            "mean_latency_seconds": safe_mean(r["seconds"] for r in group),
            "mean_output_bytes": safe_mean(r["diagnostics"]["output_bytes"] for r in group),
            "collapse_pct": 100.0 * sum(
                (r["diagnostics"]["repeated_word_ngrams"]["3"] >= 0.25)
                or (r["diagnostics"]["longest_identical_token_run"] >= 8)
                for r in group
            ) / n,
            "language_match_heuristic_pct": 100.0 * sum(
                r["diagnostics"].get("language_heuristic") == r.get("language") for r in group
            ) / n,
            "raw_math_exact_pct": (100.0 * sum(r["numeric_exact"] for r in numeric) / len(numeric)) if numeric else None,
        }

    return {
        "overall": one(rows),
        "by_domain": {k: one(v) for k, v in sorted(by_domain.items())},
        "by_language": {k: one(v) for k, v in sorted(by_language.items())},
    }


def overall_from_report(report):
    s = report["summary"]
    return {
        "responses": int(s["responses"]),
        "utf8_valid_pct": pct(s["valid_utf8_fraction"]),
        "eos_pct": pct(s["eos_fraction"]),
        "repeated_trigram_pct": pct(s["mean_repeated_trigram_fraction"]),
        "non_repetition_trigram_pct": 100.0 - pct(s["mean_repeated_trigram_fraction"]),
        "raw_math_exact_pct": 100.0 * s["numeric_exact"] / s["numeric_total"],
        "decode_tokens_per_second": float(s["mean_tokens_per_second"]),
        "mean_latency_seconds": float(s["mean_seconds"]),
    }


def save_grouped_bar(path: Path, title: str, categories, a, b, ylabel: str):
    x = list(range(len(categories)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.25), 5.2))
    ax.bar([i - width / 2 for i in x], a, width, label="20M Preview 02")
    ax.bar([i + width / 2 for i in x], b, width, label="50M Pilot A")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_single_bar(path: Path, title: str, labels, values, ylabel: str):
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.05), 5.0))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def f(v, suffix=""):
    return "—" if v is None else f"{v:.2f}{suffix}"


def main():
    OUT.mkdir(exist_ok=True)
    r20 = load_json(REPORTS / "free50_baseline20.json")
    r50 = load_json(REPORTS / "free50_pilot_a.json")
    manual = load_json(REPORTS / "manual_free50.json")
    throughput = load_json(REPORTS / "throughput50.json")

    direct20 = overall_from_report(r20)
    direct50 = overall_from_report(r50)
    direct20["manual_comprehensibility_pct"] = 25.0 * float(manual["mean20"])
    direct50["manual_comprehensibility_pct"] = 25.0 * float(manual["mean50"])
    direct20["parameters"] = int(r20["parameters"]["total"])
    direct50["parameters"] = int(r50["parameters"]["total"])

    detailed20 = summarize_rows(r20["responses"])
    detailed50 = summarize_rows(r50["responses"])

    speed_ratio = 100.0 * direct50["decode_tokens_per_second"] / direct20["decode_tokens_per_second"]
    latency_change = 100.0 * (direct50["mean_latency_seconds"] / direct20["mean_latency_seconds"] - 1.0)
    repetition_ratio = direct50["repeated_trigram_pct"] / direct20["repeated_trigram_pct"]

    training = {
        "50m_parameter_count": throughput["parameters"],
        "selected_seq": throughput["selected"]["seq"],
        "selected_batch": throughput["selected"]["batch"],
        "selected_tokens_per_step": throughput["selected"]["tokens_per_step"],
        "selected_training_tokens_per_second": throughput["selected"]["tokens_per_second"],
        "selected_peak_allocated_gib": throughput["selected"]["peak_allocated"] / 2**30,
        "selected_peak_reserved_gib": throughput["selected"]["peak_reserved"] / 2**30,
        "pilot_a_tokens": 999424,
        "pilot_a_validation_loss_start": 5.6234,
        "pilot_a_validation_loss_end": 2.3636,
        "pilot_a_validation_loss_reduction_pct": 100.0 * (1.0 - 2.3636 / 5.6234),
        "pilot_a_training_loop_seconds": 18.38,
        "pilot_a_optimizer_seconds": 13.24,
        "pilot_a_optimizer_tokens_per_second": 75507.0,
        "pilot_a_peak_allocated_gib": 3.34,
        "pilot_a_peak_reserved_gib": 3.60,
    }

    report = {
        "definition": "Exact current comparison from the same frozen 100-prompt greedy suite. No composite intelligence score.",
        "suite_sha256_20m": r20["suite_sha256"],
        "suite_sha256_50m": r50["suite_sha256"],
        "same_suite": r20["suite_sha256"] == r50["suite_sha256"],
        "generation_20m": r20["generation"],
        "generation_50m": r50["generation"],
        "same_generation_settings": r20["generation"] == r50["generation"],
        "20m": {"overall": direct20, **detailed20},
        "50m": {"overall": direct50, **detailed50},
        "derived": {
            "50m_decode_speed_as_pct_of_20m": speed_ratio,
            "50m_latency_change_vs_20m_pct": latency_change,
            "50m_repetition_multiple_vs_20m": repetition_ratio,
            "parameter_multiple_50m_vs_20m": direct50["parameters"] / direct20["parameters"],
        },
        "50m_training": training,
        "manual_review": manual,
    }
    (OUT / "exact_20m_vs_50m.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")

    quality_labels = ["UTF-8", "EOS", "Non-repetition", "Raw math", "Manual clarity"]
    quality20 = [direct20["utf8_valid_pct"], direct20["eos_pct"], direct20["non_repetition_trigram_pct"], direct20["raw_math_exact_pct"], direct20["manual_comprehensibility_pct"]]
    quality50 = [direct50["utf8_valid_pct"], direct50["eos_pct"], direct50["non_repetition_trigram_pct"], direct50["raw_math_exact_pct"], direct50["manual_comprehensibility_pct"]]
    save_grouped_bar(OUT / "quality_percentages.png", "Neral 20M vs 50M — higher is better", quality_labels, quality20, quality50, "Percent")

    domains = sorted(detailed20["by_domain"].keys())
    save_grouped_bar(
        OUT / "domain_eos.png",
        "EOS completion by domain",
        domains,
        [detailed20["by_domain"][d]["eos_pct"] for d in domains],
        [detailed50["by_domain"][d]["eos_pct"] for d in domains],
        "Percent",
    )
    save_grouped_bar(
        OUT / "domain_repetition.png",
        "Repeated trigram rate by domain — lower is better",
        domains,
        [detailed20["by_domain"][d]["repeated_trigram_pct"] for d in domains],
        [detailed50["by_domain"][d]["repeated_trigram_pct"] for d in domains],
        "Percent",
    )
    save_grouped_bar(
        OUT / "domain_decode_speed.png",
        "Decode speed by domain",
        domains,
        [detailed20["by_domain"][d]["decode_tokens_per_second"] for d in domains],
        [detailed50["by_domain"][d]["decode_tokens_per_second"] for d in domains],
        "Byte tokens/s",
    )
    save_grouped_bar(
        OUT / "collapse_rate.png",
        "Collapse diagnostic by domain — lower is better",
        domains,
        [detailed20["by_domain"][d]["collapse_pct"] for d in domains],
        [detailed50["by_domain"][d]["collapse_pct"] for d in domains],
        "Percent",
    )

    configs = [f"{c['seq']}×{c['batch']}" for c in throughput["candidates"]]
    speeds = [c["tokens_per_second"] for c in throughput["candidates"]]
    save_single_bar(OUT / "training_throughput_50m.png", "Neral-50M training throughput on RTX 4070", configs, speeds, "Tokens/s")
    save_single_bar(OUT / "validation_loss_50m.png", "Neral-50M Pilot A validation loss", ["Start", "End"], [5.6234, 2.3636], "Cross-entropy loss")

    lines = [
        "# Neral exact benchmark dashboard",
        "",
        "This report compares **20M Preview 02** and **50M Pilot A** on the same frozen 100-prompt greedy benchmark. No synthetic intelligence score is used.",
        "",
        f"Suite hash identical: **{report['same_suite']}**. Generation settings identical: **{report['same_generation_settings']}**.",
        "",
        "## Exact overall comparison",
        "",
        "| Metric | 20M Preview 02 | 50M Pilot A |",
        "|---|---:|---:|",
        f"| Parameters | {direct20['parameters']:,} | {direct50['parameters']:,} |",
        f"| Valid UTF-8 | {f(direct20['utf8_valid_pct'], '%')} | {f(direct50['utf8_valid_pct'], '%')} |",
        f"| EOS completion | {f(direct20['eos_pct'], '%')} | {f(direct50['eos_pct'], '%')} |",
        f"| Repeated trigram rate ↓ | {f(direct20['repeated_trigram_pct'], '%')} | {f(direct50['repeated_trigram_pct'], '%')} |",
        f"| Non-repetition trigram score ↑ | {f(direct20['non_repetition_trigram_pct'], '%')} | {f(direct50['non_repetition_trigram_pct'], '%')} |",
        f"| Raw math exact | {f(direct20['raw_math_exact_pct'], '%')} | {f(direct50['raw_math_exact_pct'], '%')} |",
        f"| Manual comprehensibility | {f(direct20['manual_comprehensibility_pct'], '%')} | {f(direct50['manual_comprehensibility_pct'], '%')} |",
        f"| Decode speed | {f(direct20['decode_tokens_per_second'])} tok/s | {f(direct50['decode_tokens_per_second'])} tok/s |",
        f"| Mean latency | {f(direct20['mean_latency_seconds'])} s | {f(direct50['mean_latency_seconds'])} s |",
        "",
        "## Derived deltas",
        "",
        f"- 50M decode speed is **{speed_ratio:.2f}%** of 20M.",
        f"- 50M mean latency is **{latency_change:+.2f}%** versus 20M.",
        f"- 50M repeated-trigram rate is **{repetition_ratio:.2f}×** the 20M rate.",
        f"- 50M has **{report['derived']['parameter_multiple_50m_vs_20m']:.2f}×** as many parameters.",
        "",
        "## By domain",
        "",
        "| Domain | N | 20M EOS | 50M EOS | 20M repetition ↓ | 50M repetition ↓ | 20M collapse ↓ | 50M collapse ↓ | 20M tok/s | 50M tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in domains:
        a = detailed20["by_domain"][d]
        b = detailed50["by_domain"][d]
        lines.append(
            f"| {d} | {a['n']} | {a['eos_pct']:.2f}% | {b['eos_pct']:.2f}% | {a['repeated_trigram_pct']:.2f}% | {b['repeated_trigram_pct']:.2f}% | {a['collapse_pct']:.2f}% | {b['collapse_pct']:.2f}% | {a['decode_tokens_per_second']:.2f} | {b['decode_tokens_per_second']:.2f} |"
        )

    lines += [
        "",
        "## By language",
        "",
        "| Language | N | 20M EOS | 50M EOS | 20M repetition ↓ | 50M repetition ↓ | 20M heuristic language match | 50M heuristic language match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lang in sorted(detailed20["by_language"]):
        a = detailed20["by_language"][lang]
        b = detailed50["by_language"][lang]
        lines.append(
            f"| {lang} | {a['n']} | {a['eos_pct']:.2f}% | {b['eos_pct']:.2f}% | {a['repeated_trigram_pct']:.2f}% | {b['repeated_trigram_pct']:.2f}% | {a['language_match_heuristic_pct']:.2f}% | {b['language_match_heuristic_pct']:.2f}% |"
        )

    lines += [
        "",
        "## 50M training benchmark",
        "",
        f"- Selected training configuration: **seq {training['selected_seq']} × batch {training['selected_batch']} = {training['selected_tokens_per_step']} tokens/update**.",
        f"- Selected measured throughput: **{training['selected_training_tokens_per_second']:.2f} tokens/s**.",
        f"- Selected PyTorch peak allocated/reserved: **{training['selected_peak_allocated_gib']:.2f}/{training['selected_peak_reserved_gib']:.2f} GiB**.",
        f"- Pilot A validation loss: **{training['pilot_a_validation_loss_start']:.4f} → {training['pilot_a_validation_loss_end']:.4f}**, a **{training['pilot_a_validation_loss_reduction_pct']:.2f}%** reduction.",
        f"- Pilot A trained on **{training['pilot_a_tokens']:,} tokens**.",
        "",
        "## Graphs",
        "",
        "- `quality_percentages.png`",
        "- `domain_eos.png`",
        "- `domain_repetition.png`",
        "- `domain_decode_speed.png`",
        "- `collapse_rate.png`",
        "- `training_throughput_50m.png`",
        "- `validation_loss_50m.png`",
        "",
        "Collapse diagnostic definition: repeated trigram fraction >= 25% **or** longest identical-token run >= 8. This is a transparent diagnostic, not an intelligence score.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print(json.dumps({"out": str(OUT), "same_suite": report["same_suite"], "domains": domains}, ensure_ascii=False))


if __name__ == "__main__":
    main()
