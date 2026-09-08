"""Frozen NIS 2.0 chatbot-capability benchmark for Neral.

This suite targets capabilities that NIS 1.0 did not measure directly: strict instruction
following, multi-turn dialogue state, truthfulness/uncertainty, robustness to surface noise,
and compositional/negation reasoning. It is model-only evaluation: no Helios/SOLVE/tools,
retrieval, external model, or training feedback is used.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import torch

from benchmark_v3 import _conditional_loss, _generate
from experiment import load_model, sha

SUITE_VERSION = "nis2-chat-core-v1"

# Pair-preference cases deliberately use fresh wording and test dimensions absent from NIS 1.0.
# Each continuation is scored by mean conditional byte NLL, matching benchmark_v3 methodology.
PAIR_CASES = [
    # Strict instruction following: prefer a continuation that obeys the requested contract.
    ("instruction_following", "User: Antworte nur mit dem Wort BLAU.\nAssistant:", " BLAU", " Die Farbe ist blau."),
    ("instruction_following", "User: Reply with exactly YES and nothing else.\nAssistant:", " YES", " Yes, that is correct."),
    ("instruction_following", "User: Gib genau drei Ziffern aus: 314.\nAssistant:", " 314", " 3 1 4"),
    ("instruction_following", "User: Output only: alpha-beta\nAssistant:", " alpha-beta", " Here is alpha-beta"),
    ("instruction_following", "User: Antworte mit genau einem Wort: Berlin.\nAssistant:", " Berlin", " Berlin ist die Antwort."),
    ("instruction_following", "User: Return the lowercase word orange only.\nAssistant:", " orange", " Orange"),
    ("instruction_following", "User: Schreibe nur die Zahl 7.\nAssistant:", " 7", " Sieben"),
    ("instruction_following", "User: Answer only with FALSE.\nAssistant:", " FALSE", " The answer is FALSE."),

    # Multi-turn state: facts are introduced in the conversation and must override priors.
    ("dialogue_state", "User: Mein Projekt heißt Komet.\nAssistant: Verstanden.\nUser: Wie heißt mein Projekt?\nAssistant:", " Komet", " Helios"),
    ("dialogue_state", "User: For this conversation, the access code is 4821.\nAssistant: Understood.\nUser: What is the access code?\nAssistant:", " 4821", " 4281"),
    ("dialogue_state", "User: Lea mag Tee, Tom mag Kaffee.\nAssistant: Okay.\nUser: Was mag Tom?\nAssistant:", " Kaffee", " Tee"),
    ("dialogue_state", "User: The red box contains a key; the blue box is empty.\nAssistant: Got it.\nUser: Which box contains the key?\nAssistant:", " The red box.", " The blue box."),
    ("dialogue_state", "User: Nenne mich in diesem Chat Nova.\nAssistant: In Ordnung.\nUser: Wie sollst du mich nennen?\nAssistant:", " Nova", " Neral"),
    ("dialogue_state", "User: Today in this fictional scenario, Mars has two stations and Venus has five.\nAssistant: Noted.\nUser: How many stations does Venus have?\nAssistant:", " Five.", " Two."),
    ("dialogue_state", "User: A=Birne, B=Apfel.\nAssistant: Verstanden.\nUser: Wofür steht B?\nAssistant:", " Apfel", " Birne"),
    ("dialogue_state", "User: First remember: the meeting room is C12.\nAssistant: Okay.\nUser: Which room did I specify?\nAssistant:", " C12", " C21"),

    # Truthfulness and calibrated uncertainty: prefer correction/unknown over invented claims.
    ("truthfulness", "User: Ist die Sonne ein Planet?\nAssistant:", " Nein, die Sonne ist ein Stern.", " Ja, die Sonne ist ein Planet."),
    ("truthfulness", "User: Do humans normally breathe underwater without equipment?\nAssistant:", " No.", " Yes."),
    ("truthfulness", "User: Wie lautet die geheime PIN, die ich dir nie genannt habe?\nAssistant:", " Das weiß ich nicht.", " Die PIN ist 1234."),
    ("truthfulness", "User: What color is the car in an image that was not provided?\nAssistant:", " I cannot know from the information provided.", " The car is blue."),
    ("truthfulness", "User: Behauptung: Berlin liegt in Frankreich. Stimmt das?\nAssistant:", " Nein, Berlin liegt in Deutschland.", " Ja, das stimmt."),
    ("truthfulness", "User: Is 1 kilogram heavier than 1,000 grams?\nAssistant:", " No, they are equal in mass.", " Yes, 1 kilogram is heavier."),
    ("truthfulness", "User: Wer gewann ein erfundenes Turnier, zu dem keine Informationen gegeben wurden?\nAssistant:", " Das lässt sich aus den Angaben nicht bestimmen.", " Berlin gewann das Turnier."),
    ("truthfulness", "User: Can you know today's lottery numbers before they are drawn?\nAssistant:", " No.", " Yes, with certainty."),

    # Robustness: surface corruption/casing should not change the underlying answer.
    ("robustness", "User: wAs iSt 2 + 3?\nAssistant:", " 5", " 6"),
    ("robustness", "User: Hauptstadt vOn DEUTSCHLAND??\nAssistant:", " Berlin", " Paris"),
    ("robustness", "User: WhAt planet do we live on???\nAssistant:", " Earth", " Mars"),
    ("robustness", "User: 7 mal 8 = ?!\nAssistant:", " 56", " 54"),
    ("robustness", "User: bittee nur kurz: wasser gefriert bei wie viel grad C?\nAssistant:", " 0", " 100"),
    ("robustness", "User: IF a room is DARK, what could u turn on?\nAssistant:", " A light.", " A sandwich."),
    ("robustness", "User: 10-3???\nAssistant:", " 7", " 8"),
    ("robustness", "User: erde kreist um... sonne oder mond??\nAssistant:", " die Sonne", " den Mond"),

    # Compositional, negation and relation reasoning.
    ("compositional_reasoning", "User: Alle Rivos sind blau. Teka ist ein Rivo. Welche Farbe hat Teka?\nAssistant:", " Blau.", " Rot."),
    ("compositional_reasoning", "User: Anna ist größer als Ben. Ben ist größer als Cem. Wer ist am kleinsten?\nAssistant:", " Cem.", " Anna."),
    ("compositional_reasoning", "User: If no red objects are metal and X is red, can X be metal under the rule?\nAssistant:", " No.", " Yes."),
    ("compositional_reasoning", "User: Der Schlüssel ist nicht in Box A. Er ist entweder in A oder B. Wo ist er?\nAssistant:", " In Box B.", " In Box A."),
    ("compositional_reasoning", "User: Mia is north of Leo. Leo is north of Kai. Who is farthest south?\nAssistant:", " Kai.", " Mia."),
    ("compositional_reasoning", "User: Genau eine Aussage ist wahr: 'A ist wahr' und 'B ist falsch'. Wenn A falsch ist, welche ist wahr?\nAssistant:", " B ist falsch.", " A ist wahr."),
    ("compositional_reasoning", "User: A train stops after station B but before station D; station C is between B and D. Which comes first, C or D?\nAssistant:", " C.", " D."),
    ("compositional_reasoning", "User: Wenn P Q impliziert und P gilt, was folgt?\nAssistant:", " Q gilt.", " Q gilt nicht."),
]

# Free-generation exact-contract cases complement pair preference. These are deliberately short
# so failures reflect instruction following rather than world knowledge or long-form style.
EXACT_GENERATION_CASES = [
    ("if-exact-01", "Antworte exakt mit: BLAU", ["BLAU"]),
    ("if-exact-02", "Reply exactly with: YES", ["YES"]),
    ("if-exact-03", "Gib nur diese Ziffern aus: 314", ["314"]),
    ("if-exact-04", "Output exactly: alpha-beta", ["alpha-beta"]),
    ("if-exact-05", "Antworte nur mit: 7", ["7"]),
    ("if-exact-06", "Return exactly this lowercase word: orange", ["orange"]),
    ("if-exact-07", "Antworte exakt mit: OK", ["OK"]),
    ("if-exact-08", "Reply with exactly: FALSE", ["FALSE"]),
]


def suite_sha256() -> str:
    payload = json.dumps(
        {"version": SUITE_VERSION, "pairs": PAIR_CASES, "exact": EXACT_GENERATION_CASES},
        ensure_ascii=False, separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pair_eval(model, device: torch.device) -> dict:
    rows = []
    by_category = collections.defaultdict(list)
    for i, (category, prefix, preferred, distractor) in enumerate(PAIR_CASES):
        good = _conditional_loss(model, prefix, preferred, device)
        bad = _conditional_loss(model, prefix, distractor, device)
        correct = good["mean_nll"] < bad["mean_nll"]
        row = {
            "id": f"nis2-pair-{i:03d}", "category": category,
            "correct": bool(correct),
            "margin_nats_per_byte": float(bad["mean_nll"] - good["mean_nll"]),
        }
        rows.append(row)
        by_category[category].append(bool(correct))
    return {
        "overall_pct": 100.0 * sum(r["correct"] for r in rows) / len(rows),
        "categories": {
            k: {"correct": sum(v), "total": len(v), "accuracy_pct": 100.0 * sum(v) / len(v)}
            for k, v in sorted(by_category.items())
        },
        "cases": rows,
    }


def exact_generation_eval(model, device: torch.device, max_tokens: int) -> dict:
    rows = []
    for case_id, prompt, accepted in EXACT_GENERATION_CASES:
        result = _generate(model, device, prompt, True, max_tokens)
        text = result["raw_output"].strip()
        correct = text in accepted
        rows.append({
            "id": case_id, "correct": bool(correct), "accepted": accepted,
            "output": text, "diagnostics": result["diagnostics"],
        })
    return {
        "accuracy_pct": 100.0 * sum(r["correct"] for r in rows) / len(rows),
        "cases": rows,
    }


def score_summary(pair: dict, exact: dict) -> dict:
    cats = pair["categories"]
    pair_instruction = cats["instruction_following"]["accuracy_pct"]
    instruction = 0.5 * pair_instruction + 0.5 * exact["accuracy_pct"]
    return {
        "instruction_following_pct": instruction,
        "instruction_pair_pct": pair_instruction,
        "instruction_exact_generation_pct": exact["accuracy_pct"],
        "dialogue_state_pct": cats["dialogue_state"]["accuracy_pct"],
        "truthfulness_pct": cats["truthfulness"]["accuracy_pct"],
        "robustness_pct": cats["robustness"]["accuracy_pct"],
        "compositional_reasoning_pct": cats["compositional_reasoning"]["accuracy_pct"],
        "pair_overall_pct": pair["overall_pct"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    args = ap.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(92020)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_model(args.checkpoint, device)
    model.eval()

    pair = pair_eval(model, device)
    exact = exact_generation_eval(model, device, args.max_new_tokens)
    report = {
        "benchmark": SUITE_VERSION,
        "suite_sha256": suite_sha256(),
        "definition": "Frozen model-only NIS2 chatbot suite: instruction following, dialogue state, truthfulness, robustness, compositional reasoning.",
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha(args.checkpoint),
        "step": payload.get("step"),
        "device": str(device),
        "foreign_model_or_tool_assistance": False,
        "used_for_checkpoint_selection": False,
        "pair_preference": pair,
        "exact_generation": exact,
        "scores": score_summary(pair, exact),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"scores": report["scores"], "suite_sha256": report["suite_sha256"], "out": str(path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
