# NextGen V1: isolated, real-weight SFT pilot

**Implemented code, NOT an executed GPU experiment.** The following commands are for a trusted local operator only, after the active V8.1 training and queued V8.2 read-only diagnostic are finished. There is no GitHub Actions GPU trigger, automatic promotion, cloud job, checkpoint upload or change to the production trainer.

## Why this exists

The 1B foundation report showed 0% exact generation and 0% self-termination (EOS) even when pair-preference tests were above zero. More foundation tokens do not directly supervise the boundary between a user message and a completed assistant response. This pilot evaluates a distinct objective: response-only cross-entropy with an explicit target for the native `<|eos|>` token. A lower SFT validation loss alone does **not** demonstrate good chat answers, nor does it establish 60–70k accepted tokens/s.

## Inputs: trusted local files only

1. A *copy* of a stable, own-weight `frontier350` checkpoint, its SHA-256, and the matching 16K tokenizer JSON. Never point the pilot at a checkpoint being written by an active trainer. `torch.load(weights_only=False)` can execute untrusted pickle content: run this only on a checkpoint whose provenance you independently trust. `--trusted-local-checkpoint` is explicit acknowledgement, not an automated trust proof.
2. Distinct local `train.jsonl` and `validation.jsonl` files. Each JSON line must have **exactly** `id`, `source_id`, `user`, `assistant`. Only text from independently reviewed and legally permitted sources may be included. Do not train on any frozen NIS2 benchmark prompts/answers or feedback from a release holdout.
3. `holdout_hashes.json`: nonempty JSON list of SHA-256 hashes of *normalized prompt strings*. Normalization uses NFKC, casefold and whitespace collapse; the registry is supplied outside the pack builder and never needs to reveal holdout text in CI. Exact hashes do not catch semantic or near-duplicate leakage.
4. An independently reviewed `screen.json` report with `version: 1`, the SHA-256 hashes of training file, validation file and holdout registry, `approved: true`, and zero `exact_overlap_count`, `near_overlap_count`, `privacy_hits`. Those are assertions by a real external review process: this code checks identities/claims, **not the truth of the near-duplicate or PII review**. No fabricated pass report is acceptable.
5. A `manifest.json` with `version: 1`, `train_sha256`, `validation_sha256`, `tokenizer_sha256`, `holdout_sha256`, `screen_report_sha256` and a nonempty `sources` array. Each source requires `id`, exact `revision_sha256`, `license`, and three explicit `true` flags: `approved_for_training`, `privacy_reviewed`, `benchmark_screened`. Inspect source licenses and original revisions; a flag is not legal authorization by itself. All hashes are lowercase SHA-256 hexadecimal.

The source and screening manifests are deliberately strict and will fail closed when a required item is absent. Sensitive text, model weights and prompt samples remain on the local machine; no input files or generated token packs are committed to GitHub/Hugging Face.

## Step 1: build EOS-supervised local pack

Run from the repository root, using Python with `tokenizers` installed (same native tokenizer package as the 350M evaluator). Replace placeholders with your *own existing trusted local paths*:

```bash
python frontier350/nextgen_v1/sft_contract.py \
  --manifest PATH_TO_MANIFEST.json \
  --train PATH_TO_TRAIN.jsonl \
  --validation PATH_TO_VALIDATION.jsonl \
  --tokenizer PATH_TO_TOKENIZER.json \
  --holdout-hashes PATH_TO_HOLDOUT_HASHES.json \
  --screening PATH_TO_SCREEN.json \
  --out LOCAL_NEW_PACK_DIR \
  --seq 512
```

Inputs use the native `User: …\nAssistant:` template; the first assistant token is preceded by a space, prompt targets are masked with -100, and the final supervised token is `<|eos|>`. No answers are truncated. Splits, exact holdout collisions, hashes and token IDs are checked. The pack is capped at 20,000 rows per split and cannot overwrite an output directory. `pack_report.json` records hashes and declares itself untrained local data.

## Step 2: bounded GPU SFT pilot, only when idle

An actual candidate can be trained with `train_sft_pilot.py` on the user's own trusted Neral checkpoint after explicit opt-in:

```bash
python frontier350/nextgen_v1/train_sft_pilot.py \
  --execute --trusted-local-checkpoint \
  --checkpoint PATH_TO_STABLE_LOCAL_COPY.pt \
  --checkpoint-sha256 EXACT_64_HEX_SHA256 \
  --pack LOCAL_NEW_PACK_DIR \
  --tokenizer PATH_TO_TOKENIZER.json \
  --out NEW_ISOLATED_RESULT_DIR \
  --train-blocks 4 --batch 2 --steps 20
```

- Safeguards: SHA-matched checkpoint and token packs, matching native tokenizer, own 349,825,152-parameter architecture, idle RTX 4070/BF16, start temperature <=78 C, runtime <=82 C, >=9000 MiB initially free, <=9 GiB reserved. Refuses existing output directories.
- Model: top four or eight blocks train in FP32-master precision with BF16 autocast, AdamW and gradient clipping. Lower layers and embeddings are frozen; this is **not** whole-model pretraining. Only answer/EOS labels contribute to loss. There is a fixed seed and isolated source/validation split.
- Scope: 1–50 updates only. It records masked-answer training loss, the average SFT validation loss on up to eight fixed rows before and after, supervised answer-token counts, wall time, temperature and VRAM. It saves an isolated candidate only if this small validation loss improves. The candidate has a separate `nextgen_sft_pilot_v1` schema and `production_eligible: false`, so it is not a drop-in production pretraining checkpoint.
- Limits: this short pilot is **not resumable** and does not benchmark full-stage accepted throughput. Do not use it for a long job. Same-data/seed A/B, real generation checks, full held-out NIS2, license and near-duplicate audits, numerical parity, exact resume and three >=5M fully accepted stages all remain necessary before any replacement decision. A negative pilot result is evidence to investigate, not a license to loosen gates.

## Verified and unverified

The GitHub-hosted CPU suite checks 15 admission contracts, 13 packing contracts and 8 pilot loader contracts, as well as Python syntax. It uses synthetic fixture rows, a fake tokenizer for packing tests and **no Torch/GPU/model training**. Successful CPU CI certifies only these explicit code paths, not that a CUDA pilot runs correctly. Do not claim 60k, 70k, EOS recovery, chat quality, or benchmark scores without separate measured evidence.
