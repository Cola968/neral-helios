"""Neral benchmark v3.

Separates base-language-model evaluation from instruction/chat behaviour.
Does not use guard, SOLVE or code tools for raw-model scores.
Runs only against locally installed Neral checkpoints.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiment import TOKENIZER, amp, load_model, sha
from free_generation50 import diagnostics


PAIR_CASES = [
    # German grammar
    ("de_grammar", "Die Kinder", " spielen im Garten.", " spielt im Garten."),
    ("de_grammar", "Der Hund", " läuft durch den Park.", " laufen durch den Park."),
    ("de_grammar", "Wir", " gehen heute nach Hause.", " geht heute nach Hause."),
    ("de_grammar", "Eine rote Ampel bedeutet:", " Man muss warten.", " Man darf immer fahren."),
    # English grammar
    ("en_grammar", "The children", " are playing outside.", " is playing outside."),
    ("en_grammar", "She", " goes to school every day.", " go to school every day."),
    ("en_grammar", "These books", " are on the table.", " is on the table."),
    ("en_grammar", "I", " am ready.", " is ready."),
    # Simple factual knowledge
    ("facts", "Berlin ist die Hauptstadt von", " Deutschland.", " Frankreich."),
    ("facts", "Die Erde umkreist", " die Sonne.", " den Mond."),
    ("facts", "Wasser gefriert bei normalem Luftdruck ungefähr bei", " 0 Grad Celsius.", " 100 Grad Celsius."),
    ("facts", "A week has", " seven days.", " eleven days."),
    ("facts", "The Earth orbits", " the Sun.", " the Moon."),
    ("facts", "A triangle has", " three sides.", " five sides."),
    # Commonsense
    ("commonsense", "Wenn es stark regnet, nimmt man draußen oft", " einen Regenschirm mit.", " einen Toaster mit."),
    ("commonsense", "Wenn ein Glas vom Tisch fällt, kann es", " zerbrechen.", " anfangen zu fliegen."),
    ("commonsense", "To cut paper, a common tool is", " scissors.", " a pillow."),
    ("commonsense", "If the room is dark, you can turn on", " a light.", " a sandwich."),
    # Code semantics/completion
    ("code", "def add(a, b):\n    return", " a + b\n", " a * b\n"),
    ("code", "for i in range(3):\n    print", "(i)\n", " = i\n"),
    ("code", "items = [1, 2, 3]\nlength =", " len(items)\n", " open(items)\n"),
    ("code", "def is_even(n):\n    return", " n % 2 == 0\n", " n % 2 == 1\n"),
    # Basic arithmetic as language-model preferences, not SOLVE
    ("math", "2 + 2 =", " 4", " 5"),
    ("math", "9 + 2 =", " 11", " 12"),
    ("math", "7 * 6 =", " 42", " 36"),
    ("math", "12 - 5 =", " 7", " 8"),
]

FREE_CHAT = [
    ("de_chat", "Hallo, wie geht es dir?"),
    ("de_chat", "Wer bist du?"),
    ("de_chat", "Was ist Wasser?"),
    ("de_chat", "Erkläre Gravitation in einem kurzen Satz."),
    ("de_chat", "Nenne drei Tiere."),
    ("en_chat", "Hello, how are you?"),
    ("en_chat", "Who are you?"),
    ("en_chat", "What is water?"),
    ("en_chat", "Explain gravity in one short sentence."),
    ("en_chat", "Name three animals."),
]

BASE_PREFIXES = [
    ("de_base", "Berlin ist die Hauptstadt von"),
    ("de_base", "Wasser ist für Menschen wichtig, weil"),
    ("de_base", "Die Erde bewegt sich um die"),
    ("de_base", "Ein Computerprogramm besteht aus"),
    ("de_base", "Am Flughafen starten und landen"),
    ("en_base", "The Earth moves around the"),
    ("en_base", "Water is important for life because"),
    ("en_base", "A computer program consists of"),
    ("en_base", "At an airport, airplanes"),
    ("en_base", "A healthy backup strategy keeps"),
]


def _conditional_loss(model, prefix: str, continuation: str, device: torch.device) -> dict:
    """Conditional byte NLL; prefix is context, only continuation bytes are scored."""
    p = TOKENIZER.encode(prefix, bos=True)
    c = TOKENIZER.encode(continuation)
    if not c:
        raise ValueError("Continuation must be non-empty")
    ids = p + c
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor(ids[1:], dtype=torch.long, device=device)
    with torch.inference_mode(), amp(device):
        logits, _ = model(x)
    # logits at position len(p)-1 predicts first continuation byte
    first = len(p) - 1
    selected = logits[0, first:first + len(c)].float()
    ce = F.cross_entropy(selected, targets[first:first + len(c)], reduction="none")
    return {
        "sum_nll": float(ce.sum()),
        "mean_nll": float(ce.mean()),
        "bits_per_byte": float(ce.mean() / math.log(2)),
        "bytes": len(c),
    }


def pair_benchmark(model, device):
    rows = []
    by_domain = collections.defaultdict(list)
    for i, (domain, prefix, good, bad) in enumerate(PAIR_CASES):
        a = _conditional_loss(model, prefix, good, device)
        b = _conditional_loss(model, prefix, bad, device)
        # Mean NLL length-normalization is explicit because candidate lengths differ.
        correct = a["mean_nll"] < b["mean_nll"]
        margin = b["mean_nll"] - a["mean_nll"]
        row = {
            "id": f"pair/{i:03d}", "domain": domain, "prefix": prefix,
            "preferred": good, "distractor": bad,
            "preferred_score": a, "distractor_score": b,
            "correct": correct, "margin_nats_per_byte": margin,
        }
        rows.append(row)
        by_domain[domain].append(correct)
    return {
        "definition": "Two-way conditional continuation preference using mean byte NLL; not an official MMLU/HellaSwag score.",
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "domains": {d: {"correct": sum(v), "total": len(v), "accuracy": sum(v)/len(v)} for d, v in by_domain.items()},
        "cases": rows,
    }


def _generate(model, device, prompt: str, chat: bool, max_tokens: int):
    formatted = f"User: {prompt}\nAssistant:" if chat else prompt
    ids = TOKENIZER.encode(formatted, bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(), amp(device):
        out = model.generate(x, max_new_tokens=max_tokens, temperature=0, eos_id=TOKENIZER.eos_id)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    generated = out[0, len(ids):].tolist()
    text = TOKENIZER.decode(generated)
    return {
        "raw_output": text,
        "raw_token_ids": generated,
        "generated_tokens": len(generated),
        "seconds": elapsed,
        "tokens_per_second": len(generated)/max(elapsed, 1e-9),
        "diagnostics": diagnostics(generated, text),
    }


def generation_benchmark(model, device, max_tokens):
    chat_rows = []
    base_rows = []
    for domain, prompt in FREE_CHAT:
        chat_rows.append({"domain": domain, "prompt": prompt} | _generate(model, device, prompt, True, max_tokens))
    for domain, prefix in BASE_PREFIXES:
        base_rows.append({"domain": domain, "prefix": prefix} | _generate(model, device, prefix, False, max_tokens))

    def summary(rows):
        texts = [r["raw_output"].strip() for r in rows]
        return {
            "responses": len(rows),
            "unique_output_fraction": len(set(texts))/max(1, len(texts)),
            "eos_fraction": statistics.mean(float(r["diagnostics"]["eos"]) for r in rows),
            "valid_utf8_fraction": statistics.mean(float(r["diagnostics"]["valid_utf8"]) for r in rows),
            "mean_repeated_trigram_fraction": statistics.mean(r["diagnostics"]["repeated_word_ngrams"]["3"] for r in rows),
            "mean_tokens_per_second": statistics.mean(r["tokens_per_second"] for r in rows),
        }
    return {
        "chat_instruction": {"summary": summary(chat_rows), "responses": chat_rows},
        "base_continuation": {"summary": summary(base_rows), "responses": base_rows},
        "note": "Base continuation is the fairer free-generation diagnostic for a checkpoint that has only undergone general-language pretraining; chat measures instruction formatting separately.",
    }


@torch.inference_mode()
def stream_bpb(model, device, split="val", sequence_lengths=(128, 256, 512, 1024), batches=8):
    """Fixed-window byte BPB excluding BOS/EOS/PAD/SEP targets."""
    out = {}
    for lang in ("de", "en"):
        path = Path(f"pretrain50_data/{split}_{lang}.bin")
        if not path.exists():
            out[lang] = {"available": False, "path": str(path)}
            continue
        data = np.memmap(path, dtype="<u2", mode="r")
        lang_rows = {}
        for seq in sequence_lengths:
            if len(data) <= seq + 1:
                continue
            rng = np.random.default_rng(7300 + seq)
            total_loss = 0.0
            byte_targets = 0
            special_targets = 0
            for _ in range(batches):
                starts = rng.integers(0, len(data)-seq-1, size=4)
                z = np.stack([data[s:s+seq+1].astype(np.int64) for s in starts])
                x = torch.tensor(z[:, :-1], device=device)
                y = torch.tensor(z[:, 1:], device=device)
                with amp(device):
                    logits, _ = model(x)
                mask = y >= 4
                special_targets += int((~mask).sum())
                byte_targets += int(mask.sum())
                if mask.any():
                    total_loss += float(F.cross_entropy(logits.float()[mask], y[mask], reduction="sum"))
            mean_nll = total_loss / max(1, byte_targets)
            lang_rows[str(seq)] = {
                "byte_nll": mean_nll,
                "bits_per_byte": mean_nll/math.log(2),
                "byte_targets": byte_targets,
                "special_targets_excluded": special_targets,
                "batches": batches,
            }
        out[lang] = {"available": True, "sequence_lengths": lang_rows}
    return {
        "definition": "Cross-entropy over raw-byte targets only (token IDs >=4); special BOS/EOS/PAD/SEP targets excluded.",
        "split": split,
        "languages": out,
    }


def runtime_forward_benchmark(model, device):
    if device.type != "cuda":
        return {"available": False, "reason": "CUDA required for meaningful RTX runtime benchmark"}
    rows = []
    for seq, batch in [(128, 32), (256, 32), (512, 16), (1024, 8), (2048, 2)]:
        if seq > model.cfg.max_seq_len:
            continue
        try:
            x = torch.randint(4, 260, (batch, seq), device=device)
            for _ in range(3):
                with amp(device): model(x)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            times=[]
            for _ in range(8):
                torch.cuda.synchronize(); t=time.perf_counter()
                with amp(device): model(x)
                torch.cuda.synchronize(); times.append(time.perf_counter()-t)
            median = statistics.median(times)
            rows.append({
                "seq": seq, "batch": batch, "tokens": seq*batch,
                "median_seconds": median,
                "forward_tokens_per_second": seq*batch/median,
                "peak_allocated": torch.cuda.max_memory_allocated(),
                "peak_reserved": torch.cuda.max_memory_reserved(),
                "oom": False,
            })
        except torch.cuda.OutOfMemoryError:
            rows.append({"seq":seq,"batch":batch,"oom":True})
            torch.cuda.empty_cache()
    return {"available": True, "cases": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(73003)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_model(args.checkpoint, device)
    model.eval()

    report = {
        "benchmark_version": 3,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha(args.checkpoint),
        "parameters": model.parameter_report(),
        "config": model.cfg.to_dict(),
        "step": payload.get("step"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "torch": str(torch.__version__),
        "pair_preference": pair_benchmark(model, device),
        "true_byte_bpb": stream_bpb(model, device),
        "forward_runtime": runtime_forward_benchmark(model, device),
    }
    if not args.skip_generation:
        report["free_generation"] = generation_benchmark(model, device, args.max_new_tokens)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "parameters": report["parameters"]["trainable"],
        "pair_accuracy": report["pair_preference"]["accuracy"],
        "device": report["device"],
        "saved": args.out,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
