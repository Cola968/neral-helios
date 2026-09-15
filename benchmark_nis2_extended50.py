"""NIS 2.0 Extended diagnostic suite.

This deliberately does NOT modify the frozen 48-case NIS2 core or the NIS2 score weights.
It adds harder model-only diagnostics for 2B-3B training checkpoints: multi-step math,
code debugging/semantics, conflicting context, calibrated uncertainty, and longer dependency
chains. Results are reported separately as NIS2 Extended so historical Core scores remain
comparable.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import torch

from benchmark_v3 import _conditional_loss
from experiment import load_model, sha

SUITE_VERSION = "nis2-extended-v1"

PAIR_CASES = [
    # Multi-step math: two or more operations / constraints before the final answer.
    ("multi_step_math", "User: A box has 6 rows of 8 bolts. 13 bolts are removed. How many remain?\nAssistant:", " 35", " 48"),
    ("multi_step_math", "User: Ein Zug fährt 120 km in 2 h und danach 90 km in 1 h. Wie groß ist die Durchschnittsgeschwindigkeit über die gesamte Strecke?\nAssistant:", " 70 km/h", " 75 km/h"),
    ("multi_step_math", "User: Start with 5, triple it, subtract 4, then square the result. What is the result?\nAssistant:", " 121", " 11"),
    ("multi_step_math", "User: Ein Rechteck ist 9 cm lang und 4 cm breit. Die Länge wird um 3 cm erhöht. Wie groß ist die neue Fläche?\nAssistant:", " 48 cm²", " 36 cm²"),
    ("multi_step_math", "User: 40% of 250 students take physics. One quarter of those also take astronomy. How many take both?\nAssistant:", " 25", " 100"),
    ("multi_step_math", "User: Ein Preis von 80 Euro wird erst um 25% erhöht und danach um 20% gesenkt. Endpreis?\nAssistant:", " 80 Euro", " 76 Euro"),
    ("multi_step_math", "User: If x+3=11 and y=2x, what is y-5?\nAssistant:", " 11", " 16"),
    ("multi_step_math", "User: A tank is half full at 30 L. After adding 15 L, what fraction of its 60 L capacity is full?\nAssistant:", " Three quarters.", " Two thirds."),

    # Code semantics/debugging: prefer the actual behavior or minimal correct diagnosis.
    ("code_debugging", "User: Python: x=[1,2]; y=x; y.append(3). What is x?\nAssistant:", " [1, 2, 3]", " [1, 2]"),
    ("code_debugging", "User: Python function: def f(n): return n*n. What does f(4)+f(2) return?\nAssistant:", " 20", " 18"),
    ("code_debugging", "User: In Python, why does `if value = 3:` fail?\nAssistant:", " Assignment uses =; the comparison must use ==.", " Python requires === for comparison."),
    ("code_debugging", "User: JavaScript: const a=[1,2,3]; console.log(a.length). Output?\nAssistant:", " 3", " 2"),
    ("code_debugging", "User: Python loop `for i in range(3): print(i)` prints which values?\nAssistant:", " 0, 1, 2", " 1, 2, 3"),
    ("code_debugging", "User: A function should return the larger number. `return a if a < b else b` is wrong. Minimal fix?\nAssistant:", " Use `return a if a > b else b`.", " Replace return with print."),
    ("code_debugging", "User: Python: d={'a':1}; what does d.get('b', 7) return?\nAssistant:", " 7", " None"),
    ("code_debugging", "User: A recursive function has no base case. What is the likely failure?\nAssistant:", " Unbounded recursion until a recursion-depth error.", " It always returns zero."),

    # Conflicting context: explicit newer/local evidence must beat a tempting earlier statement.
    ("context_conflict", "User: At first I said the launch is Monday. Correction: the launch is Wednesday. When is the launch?\nAssistant:", " Wednesday.", " Monday."),
    ("context_conflict", "User: Projektregel: A=rot. Neue Regel ersetzt sie: A=grün. Welche Farbe ist A jetzt?\nAssistant:", " Grün.", " Rot."),
    ("context_conflict", "User: The database copy says port 8000, but the current configuration below says port 9000. Use the current configuration. Which port?\nAssistant:", " 9000", " 8000"),
    ("context_conflict", "User: Lea used to own the blue bike. She sold it to Niko yesterday. Who owns it now?\nAssistant:", " Niko.", " Lea."),
    ("context_conflict", "User: Früher hatte Zug X 5 Wagen. Nach dem Umbau hat er 7. Wie viele Wagen hat er jetzt?\nAssistant:", " 7", " 5"),
    ("context_conflict", "User: Ignore the draft value 12. The verified measurement is 9. What value should be reported?\nAssistant:", " 9", " 12"),
    ("context_conflict", "User: The first note calls the planet Aris. A later correction says its name is Eris. Use the correction. Name?\nAssistant:", " Eris", " Aris"),
    ("context_conflict", "User: Version 1 allowed three retries. Version 2, which supersedes it, allows two. How many retries now?\nAssistant:", " Two.", " Three."),

    # Uncertainty/calibration: answer only what follows from supplied evidence.
    ("uncertainty_calibration", "User: A sealed envelope contains an unknown number. No clue is given. Is it 42?\nAssistant:", " It cannot be determined from the information given.", " Yes, it is 42."),
    ("uncertainty_calibration", "User: Ich habe keine Wetterdaten genannt. Regnet es morgen um 14 Uhr bei mir?\nAssistant:", " Das lässt sich ohne Wetterdaten nicht bestimmen.", " Ja, sicher."),
    ("uncertainty_calibration", "User: A study reports correlation only. Does that prove causation?\nAssistant:", " No.", " Yes."),
    ("uncertainty_calibration", "User: Two possible causes fit the evidence equally well. Which one is definitely true?\nAssistant:", " Neither can be identified as definitely true from that evidence.", " The first cause is definitely true."),
    ("uncertainty_calibration", "User: Kannst du aus einer fehlenden Datei ihren exakten Inhalt zitieren?\nAssistant:", " Nein.", " Ja."),
    ("uncertainty_calibration", "User: A measurement is 10 ± 2. Is the exact true value certainly 10?\nAssistant:", " No.", " Yes."),
    ("uncertainty_calibration", "User: No source gives the author's birth year. What should you report?\nAssistant:", " That the birth year is unknown from the available information.", " A plausible year as fact."),
    ("uncertainty_calibration", "User: Ein Test mit kleiner Stichprobe zeigt einen Effekt. Ist damit sicher bewiesen, dass er allgemein gilt?\nAssistant:", " Nein.", " Ja."),

    # Dependency chains: answer requires following several stated relations, not a single lookup.
    ("dependency_chain", "User: A is east of B. B is east of C. C is east of D. Which is farthest west?\nAssistant:", " D.", " A."),
    ("dependency_chain", "User: Lio gives the key to Mara. Mara gives it to Sven. Sven puts it in Box 4. Where is the key?\nAssistant:", " In Box 4.", " With Lio."),
    ("dependency_chain", "User: R ist älter als S. S ist älter als T. T ist älter als U. Wer ist am jüngsten?\nAssistant:", " U.", " R."),
    ("dependency_chain", "User: If P implies Q, Q implies R, and P is true, what follows about R?\nAssistant:", " R is true.", " R is false."),
    ("dependency_chain", "User: Train A arrives before B; B before C; C before D. Which train arrives second?\nAssistant:", " B.", " C."),
    ("dependency_chain", "User: Datei X verweist auf Y, Y auf Z, und Z enthält den Wert 17. Welchen Wert erreicht man von X aus?\nAssistant:", " 17", " X"),
    ("dependency_chain", "User: Nara is north of Omi. Omi is west of Pera. Pera is north of Tali. Is Tali south of Pera?\nAssistant:", " Yes.", " No."),
    ("dependency_chain", "User: Rule 1: every K is an L. Rule 2: no L is an M. Object X is a K. Can X be an M under these rules?\nAssistant:", " No.", " Yes."),
]


def suite_sha256() -> str:
    payload = json.dumps({"version": SUITE_VERSION, "pairs": PAIR_CASES}, ensure_ascii=False,
                         separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evaluate(model, device: torch.device) -> dict:
    rows = []
    by_category = collections.defaultdict(list)
    margins = collections.defaultdict(list)
    for i, (category, prefix, preferred, distractor) in enumerate(PAIR_CASES):
        good = _conditional_loss(model, prefix, preferred, device)
        bad = _conditional_loss(model, prefix, distractor, device)
        margin = float(bad["mean_nll"] - good["mean_nll"])
        correct = margin > 0.0
        rows.append({"id": f"nis2-ext-{i:03d}", "category": category,
                     "correct": bool(correct), "margin_nats_per_byte": margin})
        by_category[category].append(bool(correct))
        margins[category].append(margin)
    categories = {}
    for k in sorted(by_category):
        vals = by_category[k]
        categories[k] = {
            "correct": sum(vals), "total": len(vals),
            "accuracy_pct": 100.0 * sum(vals) / len(vals),
            "mean_margin_nats_per_byte": sum(margins[k]) / len(margins[k]),
        }
    return {
        "overall_pct": 100.0 * sum(r["correct"] for r in rows) / len(rows),
        "categories": categories,
        "cases": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(92021)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_model(args.checkpoint, device)
    model.eval()
    result = evaluate(model, device)
    report = {
        "benchmark": SUITE_VERSION,
        "suite_sha256": suite_sha256(),
        "definition": "Separate 40-case NIS2 Extended diagnostic. It does not alter the frozen NIS2 Core score.",
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha(args.checkpoint),
        "step": payload.get("step"),
        "device": str(device),
        "foreign_model_or_tool_assistance": False,
        "used_for_checkpoint_selection": False,
        "overall_pct": result["overall_pct"],
        "categories": result["categories"],
        "cases": result["cases"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"benchmark": SUITE_VERSION, "overall_pct": report["overall_pct"],
                      "categories": report["categories"], "suite_sha256": report["suite_sha256"],
                      "out": str(out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
