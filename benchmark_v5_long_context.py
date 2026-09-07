"""Neral benchmark v5: frozen 4K byte-context diagnostic.

This benchmark is deliberately separate from v3/v4 so historical scores remain frozen.
It measures whether a Neral checkpoint actually uses the context lengths its architecture
supports. No SOLVE, Helios, guard, retrieval system, external model, or answer extractor is
used. All retrieval keys are synthetic random-looking strings chosen for this suite.

The suite reports direct metrics only:
* single-needle preference at increasing byte distances up to the model's 4096-token window,
* two-needle position-sensitive recall,
* document-bounded true-byte BPB at 512..4095 input tokens on the frozen B1 validation corpus.

It is an internal diagnostic, not a public benchmark or IQ score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_v3 import _conditional_loss
from experiment import TOKENIZER, amp, load_model, sha

SUITE_VERSION = 2
SUITE_SEED = 75_005
NEEDLES = ("ORBIT-7K4Q", "NOVA-3M8P", "LUMEN-5R2X", "VECTOR-9C6T")
SINGLE_DISTANCES = (512, 1024, 2048, 3072, 3800)
BPB_LENGTHS = (512, 1024, 2048, 3072, 4095)
FILLER_DE = " Dies ist ein neutraler Satz ueber Alltag, Sprache und Beobachtung."
FILLER_EN = " This is a neutral sentence about everyday language and observation."

# Public, frozen textual definition used to derive a stable suite fingerprint.
FROZEN_TEXT_LITERALS = (
    "Kennwort:", "Passcode:", "Erstes Kennwort:", "Zweites Kennwort:",
    "First passcode:", "Second passcode:", *NEEDLES,
)


def suite_sha256() -> str:
    payload = {
        "version": SUITE_VERSION,
        "seed": SUITE_SEED,
        "needles": NEEDLES,
        "single_distances": SINGLE_DISTANCES,
        "bpb_lengths": BPB_LENGTHS,
        "filler_de": FILLER_DE,
        "filler_en": FILLER_EN,
        "literals": FROZEN_TEXT_LITERALS,
        "validation_root_policy": "pretrain50_data_only",
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf8")
    return hashlib.sha256(raw).hexdigest()


def _filler(target_bytes: int, lang: str) -> str:
    sentence = FILLER_DE if lang == "de" else FILLER_EN
    raw = sentence.encode("utf8")
    pieces = []
    used = 0
    while used + len(raw) <= target_bytes:
        pieces.append(sentence)
        used += len(raw)
    remain = target_bytes - used
    if remain:
        pieces.append(" " + ("x" * max(0, remain - 1)))
    text = "".join(pieces)
    assert len(text.encode("utf8")) == target_bytes
    return text


def _fits(model, prefix: str, *continuations: str) -> bool:
    longest = max(len(TOKENIZER.encode(c)) for c in continuations)
    return len(TOKENIZER.encode(prefix, bos=True)) + longest <= model.cfg.max_seq_len


def _pair(model, device, prefix: str, good: str, bad: str) -> dict:
    if not _fits(model, prefix, good, bad):
        return {"available": False, "reason": "exceeds_model_context", "prefix_bytes": len(prefix.encode("utf8"))}
    a = _conditional_loss(model, prefix, good, device)
    b = _conditional_loss(model, prefix, bad, device)
    margin = b["mean_nll"] - a["mean_nll"]
    return {
        "available": True,
        "correct": margin > 0,
        "margin_nats_per_byte": margin,
        "preferred": a,
        "distractor": b,
        "prefix_bytes": len(prefix.encode("utf8")),
        "context_tokens_with_good": len(TOKENIZER.encode(prefix + good, bos=True)),
    }


def single_needle_recall(model, device) -> dict:
    rows = []
    for lang in ("de", "en"):
        for i, distance in enumerate(SINGLE_DISTANCES):
            good_key = NEEDLES[i % len(NEEDLES)]
            bad_key = NEEDLES[(i + 1) % len(NEEDLES)]
            if lang == "de":
                prefix = f"Kennwort: {good_key}.\n" + _filler(distance, lang) + "\nKennwort:"
            else:
                prefix = f"Passcode: {good_key}.\n" + _filler(distance, lang) + "\nPasscode:"
            r = _pair(model, device, prefix, f" {good_key}.", f" {bad_key}.")
            rows.append({"language": lang, "filler_bytes": distance, "needle": good_key} | r)
    available = [r for r in rows if r["available"]]
    by_distance = {}
    for d in SINGLE_DISTANCES:
        part = [r for r in available if r["filler_bytes"] == d]
        by_distance[str(d)] = {
            "correct": sum(bool(r["correct"]) for r in part),
            "total": len(part),
            "accuracy": (sum(bool(r["correct"]) for r in part) / len(part)) if part else None,
            "mean_margin": statistics.mean(r["margin_nats_per_byte"] for r in part) if part else None,
        }
    return {
        "accuracy": sum(bool(r["correct"]) for r in available) / max(1, len(available)),
        "correct": sum(bool(r["correct"]) for r in available),
        "total": len(available),
        "by_distance": by_distance,
        "rows": rows,
        "definition": "Prefer the passcode actually present near the beginning of a long synthetic context over a distractor.",
    }


def multi_needle_recall(model, device) -> dict:
    rows = []
    cases = [
        ("de", 900, 1750, 0),
        ("de", 1500, 1750, 1),
        ("en", 900, 1750, 0),
        ("en", 1500, 1750, 1),
    ]
    for index, (lang, first_gap, second_gap, ask_second) in enumerate(cases):
        key1 = NEEDLES[index % len(NEEDLES)]
        key2 = NEEDLES[(index + 2) % len(NEEDLES)]
        if key1 == key2:
            key2 = NEEDLES[(index + 1) % len(NEEDLES)]
        if lang == "de":
            prefix = (
                f"Erstes Kennwort: {key1}.\n" + _filler(first_gap, lang) +
                f"\nZweites Kennwort: {key2}.\n" + _filler(second_gap, lang) +
                ("\nZweites Kennwort:" if ask_second else "\nErstes Kennwort:")
            )
        else:
            prefix = (
                f"First passcode: {key1}.\n" + _filler(first_gap, lang) +
                f"\nSecond passcode: {key2}.\n" + _filler(second_gap, lang) +
                ("\nSecond passcode:" if ask_second else "\nFirst passcode:")
            )
        good = key2 if ask_second else key1
        bad = key1 if ask_second else key2
        r = _pair(model, device, prefix, f" {good}.", f" {bad}.")
        rows.append({"language": lang, "ask": "second" if ask_second else "first", "first": key1, "second": key2} | r)
    available = [r for r in rows if r["available"]]
    return {
        "accuracy": sum(bool(r["correct"]) for r in available) / max(1, len(available)),
        "correct": sum(bool(r["correct"]) for r in available),
        "total": len(available),
        "rows": rows,
        "definition": "Position-sensitive two-needle recall. Swapping the two values is scored as wrong.",
    }


def _validation_root() -> Path | None:
    candidate = Path("pretrain50_data")
    required = ("documents.jsonl", "val_de.bin", "val_en.bin")
    return candidate if all((candidate / name).is_file() for name in required) else None


def _docs_for(root: Path, lang: str) -> list[dict]:
    rows = []
    for line in (root / "documents.jsonl").read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") == "val" and row.get("language") == lang:
            rows.append(row)
    return rows


@torch.inference_mode()
def document_bounded_bpb(model, device) -> dict:
    root = _validation_root()
    if root is None:
        return {"available": False, "reason": "frozen_pretrain50_validation_corpus_missing"}
    report = {
        "available": True,
        "data_root": str(root),
        "validation_sha256": {name: sha(root / name) for name in ("documents.jsonl", "val_de.bin", "val_en.bin")},
        "languages": {},
    }
    for lang in ("de", "en"):
        stream = np.memmap(root / f"val_{lang}.bin", dtype="<u2", mode="r")
        docs = _docs_for(root, lang)
        lang_rows = {}
        for seq in BPB_LENGTHS:
            if seq > model.cfg.max_seq_len:
                continue
            eligible = [d for d in docs if int(d["end"]) - int(d["start"]) >= seq + 1]
            if not eligible:
                lang_rows[str(seq)] = {"available": False, "eligible_documents": 0}
                continue
            samples = 8 if seq <= 512 else 4 if seq <= 1024 else 2 if seq <= 2048 else 1
            rng = random.Random(SUITE_SEED + seq + (0 if lang == "de" else 100_000))
            total_nll = 0.0
            byte_targets = 0
            chosen_meta = []
            for _ in range(samples):
                d = eligible[rng.randrange(len(eligible))]
                low = int(d["start"])
                high = int(d["end"]) - seq - 1
                start = rng.randint(low, high) if high > low else low
                z = np.asarray(stream[start:start + seq + 1], dtype=np.int64)
                if len(z) != seq + 1:
                    raise ValueError("short document-bounded validation window")
                x = torch.tensor(z[:-1], device=device).unsqueeze(0)
                y = torch.tensor(z[1:], device=device).unsqueeze(0)
                with amp(device):
                    logits, _ = model(x)
                mask = y >= 4
                if mask.any():
                    total_nll += float(F.cross_entropy(logits.float()[mask], y[mask], reduction="sum"))
                    byte_targets += int(mask.sum())
                chosen_meta.append({"source_id": d.get("source_id"), "id": d.get("id"), "offset": start - low})
            mean_nll = total_nll / max(1, byte_targets)
            lang_rows[str(seq)] = {
                "available": True,
                "bits_per_byte": mean_nll / math.log(2),
                "byte_nll": mean_nll,
                "byte_targets": byte_targets,
                "eligible_documents": len(eligible),
                "sampled_windows": samples,
                "sampled_documents": chosen_meta,
            }
        report["languages"][lang] = lang_rows
    macro = {}
    for seq in BPB_LENGTHS:
        values = []
        for lang in ("de", "en"):
            row = report["languages"].get(lang, {}).get(str(seq), {})
            if row.get("available"):
                values.append(float(row["bits_per_byte"]))
        macro[str(seq)] = statistics.mean(values) if values else None
    report["macro_bits_per_byte"] = macro
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(SUITE_SEED)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_model(args.checkpoint, device)
    model.eval()
    report = {
        "benchmark_version": 5,
        "suite_version": SUITE_VERSION,
        "suite_sha256": suite_sha256(),
        "definition": "Internal frozen 4K byte-context diagnostic; no tools/SOLVE/Helios/external model.",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha(args.checkpoint),
        "parameters": model.parameter_report(),
        "config": model.cfg.to_dict(),
        "step": payload.get("step"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "single_needle": single_needle_recall(model, device),
        "multi_needle": multi_needle_recall(model, device),
        "document_bounded_true_byte_bpb": document_bounded_bpb(model, device),
        "solve_used": False,
        "helios_used": False,
        "external_model_used": False,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({
        "saved": args.out,
        "suite_sha256": report["suite_sha256"],
        "single_needle_accuracy": report["single_needle"]["accuracy"],
        "multi_needle_accuracy": report["multi_needle"]["accuracy"],
        "bpb_macro": report["document_bounded_true_byte_bpb"].get("macro_bits_per_byte"),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
