# Neral NextGen V1 — isolated research protocol (2026-09-16)

**Status: evidence validator, compute-only GPU probe, local SFT pack builder and a bounded opt-in own-weight SFT pilot are implemented. No NextGen GPU training has been executed or quality improvement established. A complete production-ready foundation/SFT pipeline and stable 60–70k accepted tokens/s are NOT implemented, achieved or guaranteed.** This branch does not change the active V8.1 workflow, create GPU jobs, move model checkpoints, upload corpora or publish private data.

## Diagnosis before replacement

Neral's 350M architecture has 349,825,152 parameters. The existing guarded V9 continuation normally trains the upper four blocks and freezes the rest: improving its speed does not prove full-model capability gains. The documented 1B foundation baseline had 0% exact instruction generation and 0% natural EOS despite nonzero pair-preference scores. That suggests examining instruction supervision and the evaluator, **not declaring a broken optimizer or tokenizer**. Check 2B–2.5B free outputs, training/validation loss, prompt template, EOS targets, rejection causes, and checkpoint and dataset identities. The frozen NIS2 suite is not training data and is not a hyperparameter-selection set.

## Workstream A — foundation speed and quality

The original own-weight model, tokenizer, local data manifests, safe checkpoint and runtime are the control. Independent candidates may compare top-four/top-eight/progressive-unfreeze learning under matched corpus, RNG, accepted-token budgets and validation protocols. Changing the number of trainable blocks alters training FLOPs, so compare throughput *and* quality per elapsed wall time; do not call it an equivalent execution-only optimization. BF16, fused AdamW, fixed shapes, SDPA and compile modes are candidate techniques, not proven faster here. Any fused-loss implementation needs numerical and gradient parity. Thermal <=82 C and reserved VRAM <=9 GiB are not negotiable.

`probe.py` implements a **synthetic compute-only** BF16 forward/backward/fused-AdamW microbenchmark. It checks idle RTX 4070, temperature and VRAM. It omits corpus reads, host-to-device staging, replay, evaluation, checkpoint/recovery, rejected stages and NIS2. Consequently its `compute_only_tokens_per_second` is NOT an accepted training rate. No GPU probe has yet been executed from this branch. An opt-in operator can execute it *only when all existing GPU runs are finished*; examples are in the original PR or script CLI help.

`admission.py` is a fail-closed report validator requiring three full healthy >=5M accepted-token stages, each and combined >=60k accepted tokens per end-to-end wall-second, no rejected work, matching data/evaluation hashes and dev-category/validation/repetition non-regression. >=70k is a separate stretch result. Its JSON inputs are unverified claims unless original artifacts are audited, and it cannot promote or release a checkpoint.

## Workstream B — assistant-response SFT pilot (new code)

`SFT_PILOT.md` is the exact input contract and command guide. `sft_contract.py` accepts only locally approved and SHA-pinned files, checked screening metadata, an independently supplied hashed holdout-prompt registry, and independent train/validation splits. It packs `User: ...\nAssistant:` with masked user-prefix labels, native BPE, a supervised leading response space and explicit terminal EOS. It rejects truncation, duplicate prompts across splits, role-marker injection and unapproved sources. Exact hash overlap checks do NOT replace semantic near-dedup, source licensing or privacy review.

`train_sft_pilot.py` now implements an explicit-opt-in, at most 50-update local real-weight experiment with the same architecture and tokenizer, trusted SHA-verified source checkpoint, upper-four/eight FP32 trainable blocks, BF16 compute, fused AdamW, gradient clipping, thermal/VRAM guards and isolated output. It compares masked-answer validation loss before and after on up to eight validation rows, and only saves a separate, **unpromoted** candidate checkpoint if this tiny validation metric improves. The 350M base is never overwritten. This script is not a resumable large-scale trainer, does not implement foundation training, does not prove free-generation gains, and is **not a 60–70k throughput test**. Do not launch while V8.1 or queued V8.2 is occupying the GPU.

Proper post-training needs audited human examples with consistent prompt/assistant boundaries, short/medium/long responses, answer-only and EOS supervision, foundation replay to assess forgetting, and independent measurements of EOS, exact instruction generation, repetition, math transfer, code generation and useful-answer quality. Preference training is a later option, not an automatic fix.

## Governance and decision

Only own Neral weights are allowed. Verify actual licenses, immutable upstream revisions, exact/near deduplication, secrets/PII and frozen benchmark contamination **outside** the manifest before approving any corpus. Use development suites for choosing candidates, then a sealed independent release audit. The existing 140-case chat diagnostic is not the canonical full NIS2 and cannot certify product security.

Promotion sequence: study read-only 2B/2.5B evidence -> verify original evaluator and own checkpoint -> independently review SFT corpus and holdouts -> prepare a local pack -> when the GPU is idle, run a bounded candidate -> compare with a matched untouched baseline -> frozen generation and full NIS2 audit -> only then discuss a longer, exactly resumable experimental training engine. No automatic GPU job, cloud charge, HF push or production-branch merge is enabled.

## Evidence boundary

GitHub-hosted CPU CI tests synthetic admission, SFT packing and pilot-load contracts plus Python syntax, with no Torch/GPU/model inference. Until genuine hardware and real-data evidence exists, do **not** claim 60k, 70k, improved validation, repaired EOS or good chatbot quality. An idle real-GPU test still requires output/gradient parity, equal-token A/B, and full wall-clock accounting. HF remains research-only; the linked account is non-Pro and no paid jobs or exports were authorized.

Technical references: https://docs.pytorch.org/docs/stable/generated/torch.compile.html ; https://huggingface.co/docs/transformers/main/perf_train_gpu_one ; https://huggingface.co/docs/trl/sft_trainer .
