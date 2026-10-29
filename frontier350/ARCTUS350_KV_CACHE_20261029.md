# Arctus350 · Native KV cache for efficient inference (29.10.2026)

**Status:** Opt-in experimental implementation in Draft PR #189; no changes to native training, tokenizer, checkpoints, parameter count or default inference. This is an **inference efficiency** feature, not evidence of better native learned intelligence.

## What changed

The historical `frontier350/chat_cli350.py` recomputes all prior token states for every generated token. New `Neral350M.forward_cached` reuses attention keys and values from each layer, with native **GQA (18 query heads / 2 KV heads), absolute RoPE offsets**, a causal prefill and a single-token noncausal query for incremental decoding. Key/value tensors remain transient; nothing is written into `state_dict`. Cached runs are eval-only, explicit opt-in; the historical `forward` and checkpoint loader are not changed.

**CPU-proof rather than a speed claim:** Tests compare full-reference logits vs cache at every position (including two-sample batches and differing prompts), multi-token greedy decoding, original-weight immutability, automatic-differentiation after a cache run, and malformed-cache fail-closed handling. The actual CLI path is tested with a tiny native model and a deterministic tokenizer fixture. No real checkpoint weights are present in free GitHub CI; GPU latency and factual quality still require measurement on the user's PC.

## Why it matters

At prompt length **1024** and 128 generated tokens, the legacy path processes `1024×128 + (0+...+127) = 139200` transformer token positions. With caching, after one 1024-token prefill only **127** extra decode positions are processed (the last predicted token need not be fed back), or **1151** positions in total: roughly **121× fewer repeated full-token projections**. **This is not a 121× wall-clock speedup**: attention over cached keys, kernel overhead, VRAM transfers, sampling and hardware-specific execution remain.

For the 24-layer BF16 native model at 1024-token context, the theoretical raw GQA key/value storage from `estimated_kv_cache_bytes` is approximately **12 MiB**, excluding temporary activations and allocator overhead. GPU measurement is essential before enabling this by default.

## Local opt-in test (only when PC is available and owner explicitly starts it)

```powershell
# Only the native source-locked Arctus350, NEVER an arbitrary checkpoint.
$env:PYTHONPATH="frontier350"
python frontier350/chat_cli350.py --checkpoint "<LOCAL_VERIFIED_ARCTUS350_CHECKPOINT>" --tokenizer "<LOCAL_LOCKED_16K_BPE>" --prompt "Hallo NERAL" --temperature 0 --kv-cache
```

To measure true RTX-4070 **A/B latency**, repeat an identical prompt, token limit, seed, temperature and warm-up once **without** `--kv-cache` and once with it. Record prompt tokens, generated tokens, warm-up, wall time, peak allocated VRAM, EOS and exact decoded output. For stochastic sampling, seed RNG identically; for regression first prefer `--temperature 0`. Do not assume a GPU result from CPU toy-model tests.

## Safety / release gate

- No background GPU job, new network service, cloud weight upload, paid HF compute or modification of protected model files.
- Cache must preserve native logit/greedy output under strict tolerances on CPU; later verify BF16 RTX 4070 equivalence at real contexts and report p50/p95 latency.
- If the real checkpoint exhibits a difference beyond documented BF16 tolerances or unacceptable memory use, disable `--kv-cache` and continue with existing legacy generation.
- User approval remains mandatory before training, deploying, promoting checkpoints or merging PRs.
- This feature creates headroom for *future* test-time reasoning or multi-sample verification, but does not by itself make Arctus350 rival GPT-3.
