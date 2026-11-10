from __future__ import annotations

"""Extended diagnostic evaluation for Arctus 350M at the 500M-token checkpoint.

This suite is intentionally separate from canonical NIS2. It measures fixed-seed
held-out validation loss, continuation-preference accuracy across multiple domains,
exact instruction following, open greedy generations, and generation health.
No external model, retrieval system, answer reranker, or post-hoc rewriting is used.
"""

import argparse
import collections
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model350 import FRONTIER_350, Neral350M  # noqa: E402
from train350_compiled import PersistentMMapPool, evaluate as heldout_evaluate  # noqa: E402

PARAMETERS = 349_825_152
SUITE_VERSION = "arctus-500m-diagnostic-v1"


def _ids(tok: Tokenizer, text: str) -> list[int]:
    return [int(x) for x in tok.encode(text, add_special_tokens=False).ids]


def conditional_mean_nll(model, tok, prefix, continuation, device, bos_id):
    prefix_ids = _ids(tok, prefix)
    cont_ids = _ids(tok, continuation)
    if not cont_ids:
        raise ValueError("empty continuation")
    seq = [bos_id, *prefix_ids, *cont_ids]
    if len(seq) > FRONTIER_350.max_seq_len:
        raise ValueError("case exceeds model context")
    x = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
    y = torch.tensor(cont_ids, dtype=torch.long, device=device)
    with torch.inference_mode():
        logits, _ = model(x)
        start = len(prefix_ids)
        pred = logits[0, start:start + len(cont_ids)]
        return float(F.cross_entropy(pred.float(), y, reduction="mean"))


# (id, category, prefix, preferred, distractors)
PREFERENCE_CASES = [
    ("fact-01", "factual", "The capital of France is", " Paris.", [" London.", " Madrid.", " Rome."]),
    ("fact-02", "factual", "The capital of Germany is", " Berlin.", [" Vienna.", " Munich.", " Prague."]),
    ("fact-03", "factual", "Berlin is the capital of", " Germany.", [" France.", " Poland.", " Austria."]),
    ("fact-04", "factual", "The largest planet in the Solar System is", " Jupiter.", [" Mars.", " Earth.", " Venus."]),
    ("fact-05", "factual", "The chemical symbol for oxygen is", " O.", [" H.", " Fe.", " N."]),
    ("fact-06", "factual", "Water freezes at", " 0 degrees Celsius.", [" 50 degrees Celsius.", " 100 degrees Celsius.", " -100 degrees Celsius."]),
    ("fact-07", "factual", "The Earth orbits", " the Sun.", [" the Moon.", " Mars.", " Jupiter."]),
    ("fact-08", "factual", "Die Hauptstadt von Deutschland ist", " Berlin.", [" Hamburg.", " München.", " Wien."]),

    ("math-01", "math", "2 + 2 =", " 4", [" 3", " 5", " 22"]),
    ("math-02", "math", "10 - 3 =", " 7", [" 6", " 8", " 13"]),
    ("math-03", "math", "6 * 7 =", " 42", [" 36", " 48", " 67"]),
    ("math-04", "math", "12 / 4 =", " 3", [" 2", " 4", " 8"]),
    ("math-05", "math", "The square root of 81 is", " 9.", [" 8.", " 10.", " 81."]),
    ("math-06", "math", "If x + 5 = 12, then x =", " 7.", [" 5.", " 12.", " 17."]),
    ("math-07", "math", "25 percent of 200 is", " 50.", [" 25.", " 75.", " 100."]),
    ("math-08", "math", "7 mal 8 ist", " 56.", [" 48.", " 54.", " 78."]),

    ("sci-01", "science", "The force that keeps planets in orbit is", " gravity.", [" magnetism.", " friction.", " buoyancy."]),
    ("sci-02", "science", "The speed of light in vacuum is approximately", " 300,000 km/s.", [" 300 km/s.", " 3,000 km/s.", " 30 km/s."]),
    ("sci-03", "science", "Photosynthesis primarily occurs in", " chloroplasts.", [" mitochondria.", " nuclei.", " ribosomes."]),
    ("sci-04", "science", "The atomic number of hydrogen is", " 1.", [" 2.", " 8.", " 10."]),
    ("sci-05", "science", "A neutron has", " no electric charge.", [" positive electric charge.", " negative electric charge.", " two positive charges."]),
    ("sci-06", "science", "Mars is commonly called the", " Red Planet.", [" Blue Planet.", " Green Planet.", " Ringed Planet."]),
    ("sci-07", "science", "In a vacuum, sound", " cannot propagate.", [" travels faster than light.", " becomes visible.", " gains electric charge."]),
    ("sci-08", "science", "Ein Schwarzes Loch besitzt eine Grenze, die man", " Ereignishorizont nennt.", [" Tropopause nennt.", " Photosphäre nennt.", " Zellmembran nennt."]),

    ("reason-01", "reasoning", "All cats are mammals. Luna is a cat. Therefore Luna is", " a mammal.", [" a reptile.", " a planet.", " a plant."]),
    ("reason-02", "reasoning", "If it is raining, the ground gets wet. It is raining. Therefore the ground is", " wet.", [" dry.", " frozen.", " invisible."]),
    ("reason-03", "reasoning", "Anna is older than Ben. Ben is older than Clara. Therefore Anna is", " older than Clara.", [" younger than Clara.", " the same age as Clara.", " unrelated to Clara."]),
    ("reason-04", "reasoning", "A train leaves at 10:00 and the trip takes two hours. It arrives at", " 12:00.", [" 08:00.", " 10:30.", " 14:00."]),
    ("reason-05", "reasoning", "Tom has 3 apples and receives 2 more. He now has", " 5 apples.", [" 1 apple.", " 3 apples.", " 6 apples."]),
    ("reason-06", "reasoning", "If every A is B and every B is C, then every A is", " C.", [" not C.", " only A.", " unrelated to C."]),
    ("reason-07", "reasoning", "No birds are mammals. A sparrow is a bird. Therefore a sparrow is", " not a mammal.", [" a mammal.", " a fish.", " a mineral."]),
    ("reason-08", "reasoning", "Wenn heute Montag ist, ist morgen", " Dienstag.", [" Sonntag.", " Freitag.", " Montag."]),

    ("lang-01", "language", "She ___ to school every day.", " goes", [" go", " going", " gone"]),
    ("lang-02", "language", "Yesterday, I ___ a book.", " read", [" reads", " reading", " will read"]),
    ("lang-03", "language", "The plural of child is", " children.", [" childs.", " childes.", " child."]),
    ("lang-04", "language", "Der Plural von Haus ist", " Häuser.", [" Hauser.", " Hauses.", " Hausen."]),
    ("lang-05", "language", "Ich ___ gestern im Kino.", " war", [" bin", " sein", " wäre morgen"]),
    ("lang-06", "language", "A synonym of large is", " big.", [" tiny.", " narrow.", " short."]),
    ("lang-07", "language", "The opposite of hot is", " cold.", [" warm.", " bright.", " fast."]),
    ("lang-08", "language", "Das Gegenteil von schnell ist", " langsam.", [" laut.", " hell.", " groß."]),

    ("bi-01", "bilingual", "The German translation of 'house' is", " Haus.", [" Baum.", " Auto.", " Straße."]),
    ("bi-02", "bilingual", "The German translation of 'water' is", " Wasser.", [" Feuer.", " Luft.", " Erde."]),
    ("bi-03", "bilingual", "Die englische Übersetzung von 'Sonne' ist", " sun.", [" moon.", " starship.", " cloud."]),
    ("bi-04", "bilingual", "Die englische Übersetzung von 'Buch' ist", " book.", [" table.", " school.", " pen."]),
    ("bi-05", "bilingual", "English: 'Good morning.' German:", " Guten Morgen.", [" Gute Nacht.", " Auf Wiedersehen.", " Danke."]),
    ("bi-06", "bilingual", "Deutsch: 'Wie geht es dir?' English:", " How are you?", [" Where are you?", " What time is it?", " Good night."]),
    ("bi-07", "bilingual", "The German word for 'book' is", " Buch.", [" Tisch.", " Schule.", " Stift."]),
    ("bi-08", "bilingual", "Das englische Wort für 'Wasser' ist", " water.", [" fire.", " earth.", " air."]),

    ("code-01", "code", "def add(a, b):\n    return", " a + b", [" a - b", " a * 0", " None"]),
    ("code-02", "code", "def square(x):\n    return", " x * x", [" x + x", " x / x", " 0"]),
    ("code-03", "code", "for i in range(3):\n    print(", "i)", ["3)", "range)", "\"i\")"]),
    ("code-04", "code", "items = [1, 2, 3]\nlength = ", "len(items)", ["sum(items)", "items[3]", "range(items)"]),
    ("code-05", "code", "x = True\nif x:\n    print(", "\"yes\")", ["\"no\")", "False)", "None)"]),
    ("code-06", "code", "def is_even(n):\n    return n % 2 ==", " 0", [" 1", " 2", " n"]),
    ("code-07", "code", "const add = (a, b) =>", " a + b;", [" a - b;", " null;", " a === b;"]),
    ("code-08", "code", "SELECT * FROM users WHERE id =", " 1;", [" 'users';", " SELECT;", " WHERE;"]),
]

EXACT_CASES = [
    ("exact-01", "Antworte nur mit dem Wort BERLIN.", {"BERLIN"}),
    ("exact-02", "Reply with exactly: HELLO", {"HELLO"}),
    ("exact-03", "Gib nur die Zahl 42 aus.", {"42"}),
    ("exact-04", "Reply with only the word blue.", {"blue", "BLUE"}),
    ("exact-05", "Antworte ausschließlich mit JA.", {"JA"}),
    ("exact-06", "Return exactly this text: alpha-beta", {"alpha-beta"}),
    ("exact-07", "Write only: 3.14", {"3.14"}),
    ("exact-08", "Gib exakt OK aus.", {"OK"}),
    ("exact-09", "Reply with exactly one word: orange", {"orange"}),
    ("exact-10", "Return only the number 9.", {"9"}),
]

OPEN_PROMPTS = [
    ("open-de-01", "german", "Berlin ist eine Stadt, die"),
    ("open-de-02", "german", "Die Sonne ist"),
    ("open-de-03", "german", "Ein Schwarzes Loch ist"),
    ("open-en-01", "english", "Berlin is a city that"),
    ("open-en-02", "english", "The Solar System contains"),
    ("open-en-03", "english", "Artificial intelligence is"),
    ("open-qa-01", "qa", "Frage: Was ist die Hauptstadt von Deutschland?\nAntwort:"),
    ("open-qa-02", "qa", "Question: What is the capital of France?\nAnswer:"),
    ("open-math-01", "math_generation", "Aufgabe: 7 * 8 =\nAntwort:"),
    ("open-math-02", "math_generation", "Problem: If x + 5 = 12, x =\nAnswer:"),
    ("open-sci-01", "science_generation", "Frage: Warum fallen Gegenstände zur Erde?\nAntwort:"),
    ("open-sci-02", "science_generation", "Question: Which planet is the largest in the Solar System?\nAnswer:"),
    ("open-code-01", "code_generation", "def fibonacci(n):\n    "),
    ("open-code-02", "code_generation", "def add(a, b):\n    "),
    ("open-inst-01", "instruction", "Nenne drei Planeten des Sonnensystems:\n"),
    ("open-inst-02", "instruction", "Write one short sentence about Berlin:\n"),
    ("open-reason-01", "reasoning_generation", "Anna is older than Ben, and Ben is older than Clara. Therefore"),
    ("open-reason-02", "reasoning_generation", "Wenn heute Montag ist, dann ist morgen"),
    ("open-chat-01", "chat", "User: Hello!\nAssistant:"),
    ("open-chat-02", "chat", "Nutzer: Hallo, wie geht es dir?\nAssistent:"),
]

NEXT_TOKEN_PROBES = [
    "The capital of Germany is",
    "2 + 2 =",
    "The largest planet in the Solar System is",
    "Berlin ist die Hauptstadt von",
    "def add(a, b):\n    return",
    "User: Hello!\nAssistant:",
]


def repeated_ngram_ratio(ids, n=3):
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return (len(grams) - len(set(grams))) / max(1, len(grams))


def generate(model, tok, prompt, device, bos_id, eos_id, max_new):
    seq = [bos_id, *_ids(tok, prompt)]
    generated = []
    eos = False
    start_time = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max_new):
            if len(seq) >= FRONTIER_350.max_seq_len:
                break
            x = torch.tensor([seq], dtype=torch.long, device=device)
            logits, _ = model(x)
            nxt = int(torch.argmax(logits[0, -1]).item())
            seq.append(nxt)
            if nxt == eos_id:
                eos = True
                break
            generated.append(nxt)
    torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    return {
        "output": tok.decode(generated, skip_special_tokens=True),
        "generated_tokens": len(generated),
        "eos_reached": eos,
        "repeated_trigram_ratio": repeated_ngram_ratio(generated, 3),
        "utf8_ok": True,
        "elapsed_s": elapsed,
        "decode_tok_s": len(generated) / elapsed,
    }


def preference_eval(model, tok, device, bos_id):
    rows = []
    by_cat = collections.defaultdict(list)
    for case_id, category, prefix, preferred, distractors in PREFERENCE_CASES:
        candidates = [preferred, *distractors]
        nlls = [conditional_mean_nll(model, tok, prefix, c, device, bos_id) for c in candidates]
        winner = min(range(len(candidates)), key=lambda i: nlls[i])
        correct = winner == 0
        sorted_nlls = sorted(nlls)
        row = {
            "id": case_id,
            "category": category,
            "prefix": prefix,
            "preferred": preferred,
            "distractors": distractors,
            "candidate_mean_nll": dict(zip(candidates, nlls)),
            "chosen": candidates[winner],
            "correct": bool(correct),
            "winner_margin_nats_per_bpe": sorted_nlls[1] - sorted_nlls[0],
        }
        rows.append(row)
        by_cat[category].append(bool(correct))
        print(json.dumps({"event": "preference", "id": case_id, "category": category, "correct": correct, "chosen": candidates[winner]}, ensure_ascii=False), flush=True)
    categories = {
        cat: {
            "correct": sum(vals),
            "total": len(vals),
            "accuracy_pct": 100.0 * sum(vals) / len(vals),
        }
        for cat, vals in sorted(by_cat.items())
    }
    return {
        "correct": sum(r["correct"] for r in rows),
        "total": len(rows),
        "accuracy_pct": 100.0 * sum(r["correct"] for r in rows) / len(rows),
        "categories": categories,
        "cases": rows,
    }


def generation_eval(model, tok, device, bos_id, eos_id, max_new):
    exact_rows = []
    for case_id, prompt, accepted in EXACT_CASES:
        g = generate(model, tok, prompt, device, bos_id, eos_id, max_new)
        stripped = g["output"].strip()
        correct = stripped in accepted
        row = {"id": case_id, "prompt": prompt, "accepted": sorted(accepted), "correct": bool(correct), **g}
        exact_rows.append(row)
        print(json.dumps({"event": "exact", "id": case_id, "correct": correct, "output": stripped}, ensure_ascii=False), flush=True)

    open_rows = []
    for case_id, category, prompt in OPEN_PROMPTS:
        g = generate(model, tok, prompt, device, bos_id, eos_id, max_new)
        row = {"id": case_id, "category": category, "prompt": prompt, **g}
        open_rows.append(row)
        print(json.dumps({"event": "open", "id": case_id, "category": category, "output": g["output"]}, ensure_ascii=False), flush=True)

    all_rows = exact_rows + open_rows
    return {
        "exact": {
            "correct": sum(r["correct"] for r in exact_rows),
            "total": len(exact_rows),
            "accuracy_pct": 100.0 * sum(r["correct"] for r in exact_rows) / len(exact_rows),
            "cases": exact_rows,
        },
        "health": {
            "prompts": len(all_rows),
            "eos_pct": 100.0 * sum(r["eos_reached"] for r in all_rows) / len(all_rows),
            "mean_repeated_trigram_pct": 100.0 * statistics.mean(r["repeated_trigram_ratio"] for r in all_rows),
            "median_repeated_trigram_pct": 100.0 * statistics.median(r["repeated_trigram_ratio"] for r in all_rows),
            "utf8_pct": 100.0 * sum(r["utf8_ok"] for r in all_rows) / len(all_rows),
            "mean_decode_tok_s": statistics.mean(r["decode_tok_s"] for r in all_rows),
            "median_decode_tok_s": statistics.median(r["decode_tok_s"] for r in all_rows),
        },
        "open_generations": open_rows,
    }


def next_token_eval(model, tok, device, bos_id):
    rows = []
    for prompt in NEXT_TOKEN_PROBES:
        seq = [bos_id, *_ids(tok, prompt)]
        x = torch.tensor([seq], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits, _ = model(x)
            probs = torch.softmax(logits[0, -1].float(), dim=-1)
            entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
            vals, inds = torch.topk(probs, k=5)
        top = []
        for p, idx in zip(vals.tolist(), inds.tolist()):
            top.append({"token_id": int(idx), "text": tok.decode([int(idx)], skip_special_tokens=False), "prob": float(p)})
        rows.append({"prompt": prompt, "entropy_nats": entropy, "top5": top})
    return rows


def markdown_report(report):
    p = report["preference"]
    g = report["generation"]
    v = report["heldout_validation"]
    lines = [
        "# Arctus 350M — 500M-token extended diagnostic",
        "",
        "> Custom project diagnostic; **not canonical NIS2** and not a public benchmark.",
        "",
        f"- Checkpoint tokens: **{report['checkpoint_processed_tokens']:,}**",
        f"- Parameters: **{report['parameters']:,}**",
        f"- Fixed held-out validation loss: **{v['loss']:.4f}**",
        f"- Fixed held-out perplexity: **{v['perplexity']:.2f}**",
        f"- Preference accuracy: **{p['accuracy_pct']:.2f}% ({p['correct']}/{p['total']})**",
        f"- Exact instruction generation: **{g['exact']['accuracy_pct']:.2f}% ({g['exact']['correct']}/{g['exact']['total']})**",
        f"- EOS rate: **{g['health']['eos_pct']:.2f}%**",
        f"- Mean repeated trigram ratio: **{g['health']['mean_repeated_trigram_pct']:.2f}%**",
        f"- Median greedy decode speed: **{g['health']['median_decode_tok_s']:.2f} tok/s**",
        "",
        "## Preference categories",
        "",
        "| Category | Correct | Accuracy |",
        "|---|---:|---:|",
    ]
    for cat, row in p["categories"].items():
        lines.append(f"| {cat} | {row['correct']}/{row['total']} | {row['accuracy_pct']:.2f}% |")
    lines.extend(["", "## Exact-instruction generations", ""])
    for row in g["exact"]["cases"]:
        status = "PASS" if row["correct"] else "FAIL"
        out = row["output"].replace("\n", "\\n")
        lines.append(f"- **{row['id']} {status}** — `{out}`")
    lines.extend(["", "## Open greedy generations", ""])
    for row in g["open_generations"]:
        prompt_display = row["prompt"].replace("\n", "\\n")
        out = row["output"].replace("\n", "\\n")
        lines.append(f"### {row['id']} — {row['category']}")
        lines.append(f"- Prompt: `{prompt_display}`")
        lines.append(f"- Output: `{out}`")
        lines.append(f"- EOS: {row['eos_reached']} · repeat-3: {100.0 * row['repeated_trigram_ratio']:.2f}% · decode: {row['decode_tok_s']:.2f} tok/s")
        lines.append("")
    lines.extend([
        "## Guardrails",
        "",
        "- Preference accuracy measures continuation likelihood, not assistant quality.",
        "- Generations are raw greedy Arctus outputs with no external answerer, retrieval, reranking, or rewriting.",
        "- This is a foundation-pretraining checkpoint; chat/SFT behavior is not expected yet.",
        "- Do not convert this custom diagnostic into a canonical NIS2 score.",
        "",
    ])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--valid-data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--expected-tokens", type=int, default=500_000_000)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--val-seq", type=int, default=512)
    ap.add_argument("--val-batch", type=int, default=4)
    ap.add_argument("--val-batches", type=int, default=64)
    args = ap.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA BF16 GPU required")
    torch.manual_seed(500_350)
    torch.cuda.manual_seed_all(500_350)
    np.random.seed(500_350)
    random.seed(500_350)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("config") != FRONTIER_350.to_dict():
        raise RuntimeError("checkpoint architecture mismatch")
    processed = int(ck.get("processed_tokens", -1))
    if not args.expected_tokens <= processed <= args.expected_tokens + 32_768:
        raise RuntimeError(f"expected bounded {args.expected_tokens:,}-token checkpoint, got {processed:,}")

    tok = Tokenizer.from_file(args.tokenizer)
    if tok.get_vocab_size() != FRONTIER_350.vocab_size:
        raise RuntimeError("tokenizer vocabulary mismatch")
    bos_id = tok.token_to_id("<|bos|>")
    eos_id = tok.token_to_id("<|eos|>")
    if bos_id is None or eos_id is None:
        raise RuntimeError("missing BOS/EOS")

    device = torch.device("cuda")
    model = Neral350M().to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    if model.parameter_report()["total"] != PARAMETERS:
        raise RuntimeError("parameter contract mismatch")

    print(json.dumps({"event": "startup", "suite": SUITE_VERSION, "checkpoint_tokens": processed, "step": int(ck.get("step", -1)), "gpu": torch.cuda.get_device_name(0)}), flush=True)

    valid_pool = PersistentMMapPool(Path(args.valid_data), FRONTIER_350.vocab_size)
    val_loss = heldout_evaluate(model, valid_pool, seq=args.val_seq, batch=args.val_batch, batches=args.val_batches, device=device, seed=500_777)
    model.eval()
    val_ppl = math.exp(min(val_loss, 20.0))
    print(json.dumps({"event": "heldout_validation", "loss": val_loss, "perplexity": val_ppl, "tokens": args.val_seq * args.val_batch * args.val_batches}), flush=True)

    preference = preference_eval(model, tok, device, int(bos_id))
    generation = generation_eval(model, tok, device, int(bos_id), int(eos_id), args.max_new)
    next_tokens = next_token_eval(model, tok, device, int(bos_id))

    report = {
        "suite": SUITE_VERSION,
        "canonical_nis2": False,
        "architecture": "frontier350-v1",
        "parameters": PARAMETERS,
        "checkpoint_processed_tokens": processed,
        "checkpoint_step": int(ck.get("step", -1)),
        "foreign_model_or_tool_assistance": False,
        "used_for_checkpoint_selection": False,
        "heldout_validation": {
            "seed": 500_777,
            "seq": args.val_seq,
            "batch": args.val_batch,
            "batches": args.val_batches,
            "tokens_evaluated": args.val_seq * args.val_batch * args.val_batches,
            "loss": val_loss,
            "perplexity": val_ppl,
        },
        "preference": preference,
        "generation": generation,
        "next_token_probes": next_tokens,
        "notes": [
            "Preference ranking uses mean continuation NLL on native Arctus 16K BPE.",
            "All generations use deterministic greedy decoding.",
            "No external model, retrieval system, answer reranker, or text rewriting is used.",
            "This diagnostic is separate from canonical NIS2.",
        ],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "arctus500m_diagnostic.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "arctus500m_diagnostic.md").write_text(markdown_report(report), encoding="utf-8")

    summary = {
        "event": "summary",
        "heldout_loss": val_loss,
        "heldout_ppl": val_ppl,
        "preference_accuracy_pct": preference["accuracy_pct"],
        "preference_categories": {k: v["accuracy_pct"] for k, v in preference["categories"].items()},
        "exact_instruction_pct": generation["exact"]["accuracy_pct"],
        "generation_eos_pct": generation["health"]["eos_pct"],
        "generation_mean_repeat3_pct": generation["health"]["mean_repeated_trigram_pct"],
        "generation_median_decode_tok_s": generation["health"]["median_decode_tok_s"],
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
