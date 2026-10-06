"""Opt-in, read-only cached 350M preview; original preview remains untouched.

The standalone decoder was checked against original logits/token choices using
actual pinned SAFE weights. This is inference only and does not alter weights.
"""
from __future__ import annotations

import helios350_preview as preview


def cached_generate(model, tok, bos, eos, torch, prompt, count):
    # Reuse the same trusted tokenizer, input contract and model from preview.
    ids = [bos] + tok.encode("User: " + prompt + "\nAssistant:").ids
    if len(ids) > preview.MAX_INPUT_TOKENS:
        raise ValueError("message exceeds 384 encoded tokens")
    # model350 import paths are installed by load_verified_model during startup.
    from kv_decode import cached_last_logits
    result = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        next_logits, cache = cached_last_logits(
            model, torch.tensor([ids], device=device, dtype=torch.long)
        )
        for index in range(count):
            token = int(next_logits[0].float().argmax().item())
            if token == eos:
                break
            result.append(token)
            if index + 1 < count:
                next_logits, cache = cached_last_logits(
                    model, torch.tensor([[token]], device=device, dtype=torch.long), cache
                )
    return tok.decode(result, skip_special_tokens=True), len(result)


# Preserve the original verified SHA, Host/Origin, HTTP and request code.
# Only swap the generation function; this is a distinct explicit opt-in entrypoint.
preview.generate = cached_generate

if __name__ == "__main__":
    preview.main()
