"""Capture real private raw Neral-350M generations for Phoenix A/B evaluation.

Explicit local opt-in; no training, network, uploads, model writes or promotion.
Print only progress counters and hashes, never questions, answers or local paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

# The native model350.py lives in frontier350/, one directory above this script.
# This must also work when run as `python frontier350/nextgen_v1/phoenix_capture.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phoenix_gate import unique_object
from sft_contract import prompt_hash
from helios350_preview import (SAFE_SHA, CONTINUITY_SHA, TOKENIZER_SHA,
                                sha256_file, trusted_asset, load_verified_model)

PROMPT_KEYS = {"id", "prompt"}


def prompt_rows(path: Path) -> list[dict]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 100_000:
        raise ValueError("missing, symlinked or oversized local prompt file")
    rows = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(rows, list) or not 8 <= len(rows) <= 128:
        raise ValueError("requires 8..128 independently prepared local prompts")
    ids = set()
    prompts = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != PROMPT_KEYS:
            raise ValueError("prompt rows must contain exactly id and prompt")
        ident, prompt = row["id"], row["prompt"]
        if (not isinstance(ident, str) or not 1 <= len(ident) <= 128
                or ident in ids or not isinstance(prompt, str)
                or not prompt.strip() or len(prompt) > 1024):
            raise ValueError("invalid or duplicate prompt ID")
        if any(marker in prompt for marker in ("<|bos|>", "<|eos|>", "\nAssistant:", "\nUser:")):
            raise ValueError("role-injected prompt rejected")
        fingerprint = prompt_hash(prompt)  # NFKC + casefold + normalized whitespace.
        if fingerprint in prompts:
            raise ValueError("duplicate evaluation prompt; do not inflate the test population")
        prompts.add(fingerprint)
        ids.add(ident)
    return rows


def protocol_sha(count: int) -> str:
    protocol = {"version": 1, "tokenizer_sha256": TOKENIZER_SHA,
                "prompt_template": "User: {prompt}\nAssistant:",
                "prompt_normalization": "strip_for_inference_hash_exact_original_utf8",
                "decoding": "greedy_argmax_native_kv_cache",
                "eos": "native_eos_token_counts_as_generated",
                "stop_sequences": [], "max_new_tokens": count,
                "model_mode": "raw_no_tools", "batch_size": 1}
    data = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def capture_generation(model, tok, bos, eos, torch, prompt: str, budget: int):
    """Return actual stop reason, never infer EOS merely from visible text length."""
    from kv_decode import cached_last_logits
    ids = [bos] + tok.encode("User: " + prompt.strip() + "\nAssistant:").ids
    if len(ids) > 384:
        raise ValueError("prompt exceeds 384-token input cap")
    device = next(model.parameters()).device
    output = []
    with torch.inference_mode():
        logits, cache = cached_last_logits(model, torch.tensor([ids], device=device))
        for step in range(budget):
            token = int(logits[0].float().argmax().item())
            if token == eos:
                return tok.decode(output, skip_special_tokens=True), "eos", step + 1
            output.append(token)
            if step + 1 < budget:
                logits, cache = cached_last_logits(
                    model, torch.tensor([[token]], device=device), cache)
    return tok.decode(output, skip_special_tokens=True), "max_tokens", budget


def idle_gpu(torch) -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA/BF16 missing")
    if "RTX 4070" not in torch.cuda.get_device_name(0):
        raise ValueError("expected local RTX 4070")
    free, _ = torch.cuda.mem_get_info()
    if free < 9_000 * 1024 * 1024:
        raise ValueError("GPU occupied; refuse to collide with training")
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu",
         "--format=csv,noheader,nounits"], text=True, timeout=10).strip().splitlines()
    if len(raw) != 1:
        raise ValueError("unexpected GPU occupancy report")
    busy, temp = (int(value.strip()) for value in raw[0].split(","))
    if busy > 15 or temp > 78:
        raise ValueError("GPU busy/hot; capture blocked")


def load_own_model(root: Path, checkpoint: Path, digest: str, kind: str, device: str):
    if kind in ("safe", "continuity-2b"):
        pinned = SAFE_SHA if kind == "safe" else CONTINUITY_SHA
        if digest != pinned:
            raise ValueError("baseline/continuity SHA differs from pinned own checkpoint")
        return load_verified_model(root, checkpoint, kind, device)
    if kind != "sft-pilot" or digest in (SAFE_SHA, CONTINUITY_SHA):
        raise ValueError("invalid experimental candidate identity")
    trusted_asset(root, checkpoint, digest)  # Hash first, deserialize only trusted own data.
    trusted_asset(root, root / "frontier350_assets" / "tokenizer.json", TOKENIZER_SHA)
    import torch
    from tokenizers import Tokenizer
    from model350 import FRONTIER_350, Neral350M
    # This format contains optimizer/RNG Python objects. Only use user-trusted,
    # SHA-pinned local model files; never torch.load arbitrary downloaded pickles.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (not isinstance(state, dict)
            or state.get("checkpoint_schema") != "nextgen_sft_pilot_v1"
            or state.get("production_eligible") is not False
            or state.get("base_sha256") not in (SAFE_SHA, CONTINUITY_SHA)
            or state.get("config") != FRONTIER_350.to_dict()
            or state.get("tokenizer_sha256") != TOKENIZER_SHA
            or not isinstance(state.get("model"), dict)):
        raise ValueError("not an own, isolated SFT pilot checkpoint")
    processed = state.get("pretraining_processed_tokens")
    if type(processed) is not int or processed <= 0:
        raise ValueError("invalid originating pretraining token count")
    tok = Tokenizer.from_file(str(root / "frontier350_assets" / "tokenizer.json"))
    if tok.get_vocab_size() != FRONTIER_350.vocab_size:
        raise ValueError("tokenizer vocabulary mismatch")
    bos, eos = tok.token_to_id("<|bos|>"), tok.token_to_id("<|eos|>")
    if bos is None or eos is None or bos == eos:
        raise ValueError("invalid Neral special tokens")
    model = Neral350M()
    model.load_state_dict(state["model"], strict=True)
    del state
    model = model.to(device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32).eval()
    return model, tok, bos, eos, processed, torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--trusted-local-checkpoint", action="store_true")
    ap.add_argument("--assets-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--checkpoint-sha256", required=True)
    ap.add_argument("--kind", choices=("safe", "continuity-2b", "sft-pilot"), required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--max-new-tokens", type=int, choices=(32, 64), default=64)
    args = ap.parse_args()
    if not args.execute or not args.trusted_local_checkpoint:
        ap.error("explicit --execute and --trusted-local-checkpoint are mandatory")
    root = args.assets_root.resolve(strict=True)
    if not args.prompts.resolve(strict=True).is_relative_to(root):
        ap.error("prompts must remain within private assets root")
    if (not args.out.parent.resolve(strict=True).is_relative_to(root)
            or args.out.exists() or args.out.is_symlink()):
        ap.error("new output path must remain inside private assets root; no overwrite or symlink")
    if len(args.checkpoint_sha256) != 64 or sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("checkpoint hash mismatch")
    rows = prompt_rows(args.prompts)  # Reject duplicates before GPU/checkpoint allocation.
    if args.device == "cuda":
        import torch
        idle_gpu(torch)
    model, tok, bos, eos, processed, torch = load_own_model(
        root, args.checkpoint, args.checkpoint_sha256, args.kind, args.device)
    cases = []
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        answer, reason, used = capture_generation(
            model, tok, bos, eos, torch, row["prompt"], args.max_new_tokens)
        cases.append({"id": row["id"],
                      "prompt_sha256": hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(),
                      "answer": answer, "finish_reason": reason,
                      "raw_model": True, "tool_calls": 0, "tokens_generated": used,
                      "max_tokens": args.max_new_tokens})
        print(json.dumps({"case": index, "of": len(rows), "tokens": used,
                          "finish_reason": reason, "training_started": False}), flush=True)
    report = {"schema": 1, "checkpoint_sha256": args.checkpoint_sha256,
              "protocol_sha256": protocol_sha(args.max_new_tokens), "cases": cases}
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"completed": len(cases), "elapsed_seconds": round(time.perf_counter()-started, 2),
                      "checkpoint_sha256": args.checkpoint_sha256,
                      "report_sha256": sha256_file(args.out),
                      "training_started": False}), flush=True)


if __name__ == "__main__":
    main()
