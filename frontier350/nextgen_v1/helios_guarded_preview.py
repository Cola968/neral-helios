"""Optional CPU-only Helios research chat with bounded repetition and role stops.

Not a quality upgrade claim or a replacement for the raw Phoenix A/B decoder.
Only SHA-pinned own checkpoints; no training, network upload, output persistence,
or automatic checkpoint copies. Bound exclusively to 127.0.0.1.
"""
from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import pickle
import re
import sys
import threading

import helios350_preview as base
from phoenix_continuity_cpu_probe import (
    EXPECTED_TOKENS, check_metadata, require_pinned_checkpoint,
)

ROLE_BOUNDARY = re.compile(r"(?im)(?:^|\n)\s*(?:user|assistant|system|nutzer|benutzer)\s*:")
PENALTY = 1.18


def stop_at_role(text: str) -> tuple[str, bool]:
    """Prevent a visible next speaker; not an assertion of a native EOS."""
    match = ROLE_BOUNDARY.search(text)
    return (text[:match.start()].rstrip(), True) if match else (text, False)


def repeated_tail(text: str) -> bool:
    """Conservative word-trigram loop signal, not a correctness metric."""
    words = re.findall(r"\w+", text.casefold())
    if len(words) < 15:
        return False
    grams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    tail = grams[-1]
    return (grams.count(tail) >= 3
            and (len(grams) - len(set(grams))) / len(grams) >= 0.28)


def penalized_value(logit: float, penalty: float = PENALTY) -> float:
    if penalty <= 1:
        raise ValueError("repetition penalty must exceed one")
    return logit / penalty if logit >= 0 else logit * penalty


def guarded_generate(model, tokenizer, bos: int, eos: int, torch, prompt: str,
                     budget: int) -> tuple[str, int, str]:
    """Keep raw model unchanged; apply only a bounded optional decode policy."""
    if type(budget) is not int or not 1 <= budget <= base.MAX_NEW_TOKENS:
        raise ValueError("invalid output budget")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > base.MAX_PROMPT_CHARS:
        raise ValueError("invalid message")
    ids = [bos] + tokenizer.encode("User: " + prompt.strip() + "\nAssistant:").ids
    if len(ids) > base.MAX_INPUT_TOKENS:
        raise ValueError("message exceeds 384 encoded tokens")
    result: list[int] = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for step in range(budget):
            logits, _ = model(torch.tensor([ids], device=device, dtype=torch.long))
            scores = logits[0, -1].float().clone()
            for token in set(result[-32:]):
                if token != eos:
                    scores[token] = penalized_value(float(scores[token].item()))
            token = int(scores.argmax().item())
            used = step + 1
            if token == eos:
                return tokenizer.decode(result, skip_special_tokens=True), used, "eos"
            result.append(token)
            ids.append(token)
            text = tokenizer.decode(result, skip_special_tokens=True)
            clipped, role = stop_at_role(text)
            if role:
                return clipped, used, "role_boundary"
            if repeated_tail(text):
                return text, used, "repetition_guard"
    return tokenizer.decode(result, skip_special_tokens=True), budget, "max_tokens"


def load_continuity_cpu(assets_root: Path, checkpoint: Path):
    """The continuity file is on D: OUTSIDE tokenizer root; verify exact SHA first."""
    checkpoint = require_pinned_checkpoint(checkpoint)
    root = assets_root.resolve(strict=True)
    token_path = base.trusted_asset(root, root / "frontier350_assets" / "tokenizer.json",
                                    base.TOKENIZER_SHA)
    import torch
    from tokenizers import Tokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350, Neral350M
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    try:
        state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    except (ValueError, RuntimeError, TypeError, pickle.UnpicklingError):
        # Explicitly opted-in, exact SHA-verified own historical checkpoint ONLY.
        state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    check_metadata(state, FRONTIER_350.to_dict())
    tok = Tokenizer.from_file(str(token_path))
    if tok.get_vocab_size() != FRONTIER_350.vocab_size:
        raise ValueError("tokenizer vocabulary mismatch")
    bos, eos = tok.token_to_id("<|bos|>"), tok.token_to_id("<|eos|>")
    if bos is None or eos is None or bos == eos:
        raise ValueError("invalid BOS/EOS")
    model = Neral350M()
    model.load_state_dict(state["model"], strict=True)
    del state
    return model.to(device=torch.device("cpu"), dtype=torch.float32).eval(), tok, int(bos), int(eos), EXPECTED_TOKENS, torch


def make_guarded_handler(model, tokenizer, bos, eos, torch, processed, kind, port):
    """Reuse original origin/Host/CSP checks; override ONLY bounded generation."""
    parent = base.make_handler(model, tokenizer, bos, eos, torch, processed, kind, "cpu", port)
    mutex = threading.Lock()

    class GuardedHandler(parent):
        def do_GET(self):
            if self.path == "/" and self._allowed():
                html = base.HTML.replace(
                    "Experimentelle eigene Gewichte.",
                    "Optionaler Guard-Modus: Wiederholungsbremse und Stopp vor neuen Rollen. "
                    "Kein Nachweis korrekter Antworten. Experimentelle eigene Gewichte.", 1,
                )
                return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            return super().do_GET()

        def do_POST(self):
            if not self._allowed():
                return self._json(403, {"error": "localhost access only"})
            if self.path != "/generate":
                return self._json(404, {"error": "not found"})
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                return self._json(415, {"error": "application/json required"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._json(400, {"error": "invalid length"})
            if not 1 <= length <= base.MAX_INPUT_BYTES:
                return self._json(413, {"error": "request too large"})
            try:
                prompt, count = base.parse_request(self.rfile.read(length))
                with mutex:
                    answer, used, reason = guarded_generate(
                        model, tokenizer, bos, eos, torch, prompt, count)
                return self._json(200, {"answer": answer, "generated_tokens": used,
                                        "finish_reason": reason, "model": kind,
                                        "decoder": "experimental_guarded_not_raw"})
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            except Exception:
                return self._json(500, {"error": "local guarded inference failed"})
    return GuardedHandler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--trusted-local-checkpoint", action="store_true")
    ap.add_argument("--assets-root", default=os.getenv("NERAL_ASSETS_ROOT"))
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--model", choices=("safe", "continuity-2b"), default="continuity-2b")
    ap.add_argument("--port", type=int, default=9682)
    ap.add_argument("--smoke", action="store_true", help="One-token local load check; no server, no answer logged")
    args = ap.parse_args()
    if not args.execute or not args.trusted_local_checkpoint or not args.assets_root:
        ap.error("explicit execution, locally trusted checkpoint and assets root required")
    if not 1024 <= args.port <= 65535:
        ap.error("invalid port")
    if args.model == "safe":
        model, tok, bos, eos, processed, torch = base.load_verified_model(
            Path(args.assets_root), args.checkpoint, "safe", "cpu")
    else:
        model, tok, bos, eos, processed, torch = load_continuity_cpu(
            Path(args.assets_root), args.checkpoint)
    if args.smoke:
        answer, used, reason = guarded_generate(model, tok, bos, eos, torch, "Hallo.", 1)
        print(json.dumps({"status": "guarded_one_token_smoke_not_quality_test",
                          "generated_tokens": used, "finish_reason": reason,
                          "nonempty": bool(answer.strip()), "training_started": False,
                          "checkpoint_modified": False}), flush=True)
        return
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_guarded_handler(
        model, tok, bos, eos, torch, processed, args.model, args.port))
    print(f"Experimental guarded local preview: http://127.0.0.1:{args.port}/ ; Ctrl+C stops.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
