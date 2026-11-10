# Neral Helios 1B foundation

Status: new generation under validation. The frozen 350M line is not modified by this branch.

## Production architecture

- Dense decoder-only Transformer
- 1,006,740,992 trainable parameters
- 32 blocks
- d_model 1536
- 12 query heads, 4 key/value heads, head_dim 128
- Grouped Query Attention (3 query heads per KV head)
- SwiGLU FFN with width 5120
- RMSNorm pre-norm
- learned per-head QK RMS normalization
- RoPE in every layer by default
- optional every-fourth-layer NoPE only as an ablation
- 32,768-token target vocabulary
- tied input/output embeddings
- bias-free linear projections
- dropout 0 during base pretraining
- cross entropy plus small logit Z-loss
- initial context target 4096

The production configuration is deliberately conservative: modern components with a low implementation-risk surface. MoE, Mamba, hybrid recurrence and experimental routing are excluded from the first 1B run.

## Why this is separate from legacy Neral

The current legacy model uses ordinary multi-head attention and the repository's historical 260-token byte tokenizer. Helios 1B is isolated in helios1b/ so old checkpoints and the frozen 350M line cannot be silently loaded into the new model.

The 1B weights must start from a fresh random initialization. No Qwen, Llama or other third-party model weights are part of the architecture.

## Tokenizer gate

Do not launch the full 1B run with the legacy byte tokenizer.

The next tokenizer must be trained from Neral's approved corpus and target 32,768 tokens. Before promotion it must be measured separately on:
- German prose
- English prose
- Python and common source code
- mathematics / LaTeX-like text
- URLs, numbers and technical identifiers

The decision metric is compression plus round-trip correctness, not vocabulary size alone.

## Mandatory validation order

1. Tiny proxy: forward/backward, finite gradients, resume determinism.
2. Small architecture proxy: compare baseline RoPE against optional 1-in-4 NoPE.
3. Tokenizer benchmark and data-contract validation.
4. Short 1B allocation smoke test: memory, throughput and loss sanity only.
5. 10M-token learning gate.
6. 50M and 100M gates with held-out loss and generation probes.
7. Only then promote into the long base-pretraining run.

A gate fails if loss is non-finite, gradients repeatedly explode, resume changes the token stream, validation stops improving materially, or free generation degenerates while validation appears healthy.

## Training priorities

Quality comes before throughput. Throughput is optimized only after the proxy learns correctly.

Initial implementation priorities:
1. fused/SDPA attention path
2. packed sequences with no padding waste
3. BF16 where supported
4. activation checkpointing only when required by VRAM
5. exact optimizer/scheduler/RNG/data-position resume
6. atomic checkpoints and last-known-good rollback
7. benchmark gates at fixed token counts

The first production optimizer choice remains an experiment. AdamW is the safe baseline. Muon may be tested on matrix weights only after a controlled proxy comparison; it is not required for the architecture.

## Resume contract

The proxy trainer now treats resume as a complete state transition, not just weight loading. Every resumable checkpoint contains:

- model weights
- AdamW optimizer state
- warmup/cosine scheduler state
- Python, NumPy, Torch CPU and CUDA RNG state
- exact deterministic data epoch/cursor
- completed optimizer update and token count
- model/training contract
- SHA-256 fingerprints of train and validation token files
- initial and best held-out validation CE

Checkpoints are written only after a complete optimizer update and are atomically replaced. If the process dies in the middle of an update, the previous checkpoint remains authoritative and that update is replayed after restart. Resume rejects changed data, model shapes, schedules or optimizer-critical settings.

The repository test compares a continuous proxy run against a split run followed by resume and requires tensor-for-tensor identical final model weights on CPU.

## First learning proof

The first learnability experiment intentionally uses synthetic held-out token streams. It is not a capability benchmark. Its only purpose is to prove that the new attention/normalization/loss/trainer stack can lower validation cross-entropy on structure not used as the exact training stream.

The RTX smoke gate runs 40 updates and requires at least 50% relative held-out CE improvement. It uses only temporary synthetic data and uploads JSON/JSONL evidence, never weights or token binaries.

## Immediate target

Do not train 1B for billions of tokens yet.

**First prove architecture, exact resume, tokenizer and learnability. Then run a short full-size allocation/throughput smoke. Only after that spend real token budget.**
