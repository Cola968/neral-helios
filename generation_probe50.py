"""Development-only multi-length free-generation probe for Neral-50M.

This probe is not a promotion benchmark. It is a diagnostic for EOS/repetition failure modes
at 64/128/256 tokens using prompts that are separate from the frozen regression suite. Greedy
decoding is left completely unchanged: no EOS bias, no repetition penalty, no no-repeat ngram,
no truncation heuristic, and no Helios/tool assistance.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import time
from pathlib import Path

PROBE_VERSION = 1
DEFAULT_BUDGETS = (64, 128, 256)
PROMPTS = (
    {"id": "de/simple/01", "language": "de", "domain": "simple", "prompt": "Erkläre in einem kurzen Satz, warum Eis schmilzt."},
    {"id": "de/simple/02", "language": "de", "domain": "simple", "prompt": "Nenne zwei Vorteile eines Fahrrads."},
    {"id": "de/aviation/01", "language": "de", "domain": "aviation", "prompt": "Was macht ein Fluglotse? Antworte kurz."},
    {"id": "de/science/01", "language": "de", "domain": "science", "prompt": "Warum umkreist die Erde die Sonne?"},
    {"id": "de/code/01", "language": "de", "domain": "code", "prompt": "Was macht eine Python-for-Schleife?"},
    {"id": "de/clarify/01", "language": "de", "domain": "clarify", "prompt": "Ich meine das andere. Frage kurz nach, was ich meine."},
    {"id": "de/list/01", "language": "de", "domain": "list", "prompt": "Nenne genau drei Planeten."},
    {"id": "de/summary/01", "language": "de", "domain": "summary", "prompt": "Fasse zusammen: Lea fährt morgens mit dem Bus zur Schule."},
    {"id": "en/simple/01", "language": "en", "domain": "simple", "prompt": "Explain in one short sentence why ice melts."},
    {"id": "en/simple/02", "language": "en", "domain": "simple", "prompt": "Name two advantages of a bicycle."},
    {"id": "en/aviation/01", "language": "en", "domain": "aviation", "prompt": "What does an air traffic controller do? Reply briefly."},
    {"id": "en/science/01", "language": "en", "domain": "science", "prompt": "Why does Earth orbit the Sun?"},
    {"id": "en/code/01", "language": "en", "domain": "code", "prompt": "What does a Python for loop do?"},
    {"id": "en/clarify/01", "language": "en", "domain": "clarify", "prompt": "I mean the other one. Ask briefly what I mean."},
    {"id": "en/list/01", "language": "en", "domain": "list", "prompt": "Name exactly three planets."},
    {"id": "en/summary/01", "language": "en", "domain": "summary", "prompt": "Summarize: Lea takes the bus to school every morning."},
)


def suite_sha256() -> str:
    raw = json.dumps({"version": PROBE_VERSION, "prompts": PROMPTS}, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf8")
    return hashlib.sha256(raw).hexdigest()


def parse_budgets(text: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not values or any(v <= 0 for v in values):
        raise ValueError("budgets must contain positive integers")
    if tuple(sorted(set(values))) != values:
        raise ValueError("budgets must be unique and strictly increasing")
    return values


def diagnostics(ids: list[int], text: str) -> dict:
    words = re.findall(r"\w+", text.casefold())
    grams = [tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2))]
    counts = collections.Counter(grams)
    repeated = sum(v - 1 for v in counts.values()) / max(1, len(grams))
    top = counts.most_common(1)[0] if counts else ((), 0)
    raw = bytes(i - 4 for i in ids if 4 <= i < 260)
    try:
        raw.decode("utf8")
        valid_utf8 = True
    except UnicodeDecodeError:
        valid_utf8 = False
    eos_positions = [i for i, token in enumerate(ids) if token == 2]
    return {
        "valid_utf8": valid_utf8,
        "eos": bool(eos_positions),
        "first_eos_generated_index": eos_positions[0] if eos_positions else None,
        "generated_tokens": len(ids),
        "repeated_trigram_fraction": repeated,
        "top_trigram": list(top[0]),
        "top_trigram_count": int(top[1]),
        "word_count": len(words),
    }


def summarize_rows(rows: list[dict], budgets: tuple[int, ...]) -> dict:
    by_budget = {}
    for budget in budgets:
        part = [r for r in rows if r["budget"] == budget]
        by_budget[str(budget)] = {
            "responses": len(part),
            "utf8_valid_fraction": sum(r["diagnostics"]["valid_utf8"] for r in part) / max(1, len(part)),
            "eos_fraction": sum(r["diagnostics"]["eos"] for r in part) / max(1, len(part)),
            "mean_repeated_trigram_fraction": sum(r["diagnostics"]["repeated_trigram_fraction"] for r in part) / max(1, len(part)),
            "mean_generated_tokens": sum(r["diagnostics"]["generated_tokens"] for r in part) / max(1, len(part)),
            "mean_tokens_per_second": sum(r["tokens_per_second"] for r in part) / max(1, len(part)),
        }
    prefix_checks = []
    for prompt in PROMPTS:
        part = sorted((r for r in rows if r["id"] == prompt["id"]), key=lambda r: r["budget"])
        for short, long in zip(part, part[1:]):
            prefix_checks.append(long["raw_token_ids"][:len(short["raw_token_ids"])] == short["raw_token_ids"])
    return {
        "by_budget": by_budget,
        "greedy_prefix_consistency_fraction": sum(prefix_checks) / max(1, len(prefix_checks)),
        "interpretation": "Higher-budget outputs should extend shorter greedy outputs unless EOS ended generation; repetition growth with zero EOS indicates a model-side termination/looping problem.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    args = ap.parse_args()
    budgets = parse_budgets(args.budgets)

    import torch
    from experiment import TOKENIZER, amp, load_model, sha

    if not torch.cuda.is_available():
        raise RuntimeError("generation probe requires CUDA")
    torch.set_num_threads(4)
    torch.manual_seed(50_013)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    model, payload = load_model(args.checkpoint, device)
    model.eval()

    rows: list[dict] = []
    for prompt in PROMPTS:
        prefix = TOKENIZER.encode("User: " + prompt["prompt"] + "\nAssistant:", bos=True)
        x = torch.tensor([prefix], device=device)
        for budget in budgets:
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode(), amp(device):
                out = model.generate(x, max_new_tokens=budget, temperature=0, eos_id=2)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            generated = out[0, len(prefix):].tolist()
            text = TOKENIZER.decode(generated)
            rows.append(prompt | {
                "budget": budget,
                "raw_output": text,
                "raw_token_ids": generated,
                "seconds": elapsed,
                "tokens_per_second": len(generated) / max(elapsed, 1e-9),
                "diagnostics": diagnostics(generated, text),
            })
        print(json.dumps({"completed_prompt": prompt["id"]}, ensure_ascii=False), flush=True)

    report = {
        "probe_version": PROBE_VERSION,
        "suite_sha256": suite_sha256(),
        "definition": "Development-only raw Neral generation length sweep; not a frozen promotion benchmark.",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha(args.checkpoint),
        "step": payload.get("step"),
        "budgets": budgets,
        "decoding": {"temperature": 0, "eos_id": 2, "repetition_penalty": None, "no_repeat_ngram": None},
        "responses": rows,
        "summary": summarize_rows(rows, budgets),
        "weights_modified": False,
        "tools_used": False,
        "selection_use": False,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
