# Neral NextGen V1 — isolated research protocol (2026-09-16)

**Status: implemented CPU evidence validator + opt-in real-model compute-only probe; complete new training pipeline NOT implemented; 60–70k accepted tokens/s NOT achieved or guaranteed.** This directory does not modify production, create GPU workflows, resume a live run, replace a checkpoint, or train on Hugging Face. Branch `experiment/nextgen-v1-isolated` is isolated from the active V8.1 runner. No checkpoints, corpora, secrets, or source prompts are uploaded.

## Root-cause boundary

The 350M architecture is 349,825,152 parameters. The existing guarded V9 continuation normally freezes all but the upper four transformer blocks: speeding this up cannot establish full-model learning quality. The documented 1B foundation baseline had 0% exact instruction generation and 0% EOS despite nonzero preference scores. This is a foundation-vs-instruction-tuning problem to investigate, **not proof** that optimizer or tokenizer is defective. First inspect frozen 2B/2.5B output samples, training/validation curves, prompt formatting, EOS targets, rejection causes and checkpoint/dataset identities. The NIS2 suite must remain excluded from training and hyperparameter tuning.

## Training-method redesign (design, not yet a runnable trainer)

1. **Independent reproducible baselines.** Preserve existing checkpoint and baseline branch unchanged. Lock architecture/tokenizer revision, trusted local checkpoint hash, dataset manifest hashes, full deterministic sample/batch order, update-token budget and evaluation protocol. Never deserialize untrusted pickle checkpoints; existing legacy loads require trusted local provenance. Run same-checkpoint diagnostics before interpreting `0/24` as a model defect. Never claim a trained model from synthetic microbenchmarks.
2. **Foundation improvement, architecture held constant.** Test top-four continuation versus top-eight or progressive layer unfreezing in independent experiments. Compare on equal accepted *training tokens* with the same deduplicated, properly licensed, benchmark-clean mix. Altered trainable-block counts change optimizer memory and FLOPs, so compare quality-per-time separately; never present their speed numbers as a pure runtime optimization. Control schedule, seeds, replay, loss weighting and stable master precision. Try only validated optimization paths: fixed-shape batches, BF16 autocast, fused AdamW, native SDPA, compile mode and microbatch shape. Any new fused loss/kernel needs numerical/gradient parity against the existing implementation before training; do not assume free speed or unchanged quality. Do not increase existing 82 C thermal limit or 9 GiB reserved-memory gate.
3. **Post-training is an independent curriculum.** Once the foundation candidate is selected on development data, prepare legally usable, provenance-checked human prompt/assistant pairs with consistent user/assistant delimiters, response-only loss, explicit EOS targets, short/medium/long answer balance and masked padding. Use supervised fine-tuning (SFT), with foundation replay to watch for forgetting. Evaluate EOS, exact instruction generation, useful-answer rate, repetitiveness, math transfer and code *generation*, not just multiple-choice preference. Consider DPO/RL only after a working SFT baseline and independently audited preference data; a new optimizer cannot guarantee instruction-following competence.
4. **Developer vs release data.** Build a genuinely held-out development suite and a separate sealed final release holdout. Check data-source license, original revision, near-duplicates, PII/secrets and benchmark contamination before any training; preserve original Neral weights only. Do not repeatedly tune against frozen NIS2 or an external holdout. The 140-case internal suite can diagnose readiness but cannot alone certify production quality, cybersecurity, truthfulness or general reasoning.
5. **Small pilots before 2–3B.** Quality diagnosis -> compute-only shape benchmark -> numerical/optimizer parity -> same-corpus 1M/5M candidate training -> development quality -> full independent NIS2 and release assessment -> carefully approved continuation. Each train candidate has an independent output directory, optimizer/RNG/data-cursor resume and atomic checkpoint/journal. Loss curve alone cannot validate chatbot quality.

## Speed experiment

`probe.py` is a deliberately **compute-only** genuine `Neral350M` forward/backward/fused-AdamW microbenchmark with synthetic fixed-shape tokens. It can study top-four/top-eight/full trainable blocks and eager/compiled modes without loading or changing checkpoints. It guards for an idle RTX 4070, BF16, <=78 C at start, <=82 C throughout and <=9 GiB peak reserved VRAM. It does NOT perform H2D transfer, real data ingestion, replay, evaluation, checkpoint/recovery, end-to-end 5M stages or any NIS2 test. Its reported `compute_only_tokens_per_second` is NOT an accepted-token rate and MUST NEVER be quoted as the requested 60–70k success. No automatic CUDA workflow exists, and an existing GPU run takes precedence.

On an **idle** local GPU, an engineer can opt in to a one-off independent measurement:

```bash
python frontier350/nextgen_v1/probe.py --execute --train-blocks 4 --execution eager --out local_probe_eager.json
python frontier350/nextgen_v1/probe.py --execute --train-blocks 4 --execution compile-default --out local_probe_compile.json
```

Do not run these commands while V8.1 or queued V8.2 is occupying the GPU. Each output path must be new. These are examples, not evidence that anything was executed. Compilation startup must be measured separately; any output variance requires investigation, and full data-pipeline testing is still necessary. An OOM, failed compile or thermal guard means rejection, not permission to relax safety thresholds.

## End-to-end admission (available now; requires genuine external evidence)

`admission.py` implements a *fail-closed evidence-report validator*, not a production trainer or release workflow. It requires three complete healthy accepted >=5M-token stages with no rejected tokens, each and combined >=60,000 accepted tokens/wall-second. `wall_seconds` must encompass training, staging, validation, checkpoint, recovery and any due audit; no GPU-only peaks or recycled/failed batches count. 70k is a separately calculated stretch. It requires matching tokenizer, validation/dev dataset and evaluation protocol hashes, unchanged safety, independent experiments, no development-category regression, no validation-loss or repetition regression. A post-training claim additionally requires strictly increased exact and EOS correct case counts (>=24 matched cases). Foundation admission NEVER certifies chat quality. Reports are self-declared until evidence artifacts/hashes are inspected; the external sealed release audit and the user's explicit decision remain mandatory.

Run *only* after a genuine independent pilot has produced the necessary input JSON (schema exercised in `test_admission.py`):

```bash
python frontier350/nextgen_v1/admission.py --evidence private_measured_evidence.json --out private_admission_report.json
```

No private evidence or checkpoint should be committed to GitHub. This validator has zero privilege to promote checkpoints and cannot detect fabricated measurements from JSON alone.

## Hugging Face scope

HF account `Cola968` was confirmed authenticated but **not Pro** on 2026-09-16. Do not submit paid cloud GPU jobs or publish checkpoints/data. HF is currently a documentation/research and candidate-source discovery channel only. Every new dataset must pass immutable revision/license/provenance, contamination, dedup and privacy checks; any existing quarantined source stays blocked. Optional local-only Trackio metrics must be explicitly opt-in and stripped of paths, prompts, weights and sensitive values before any HF Space sync. Use the own Neral model, not Qwen or any imported model weights.

## Stop conditions and what remains unproven

- No stable end-to-end >=60k (or 70k), no improved validation/Math/Code/EOS/NIS2 and no GPU parity result yet.
- No completed NextGen foundation or SFT trainer, no real-data candidate training, no GPU execution or CI pass claim until logs confirm them.
- If a complete quality-validated same-seed pilot cannot improve both accepted throughput and development quality, reject that candidate rather than scaling it. If hardware cannot hit the target at unchanged training objective and temperature/VRAM caps, revise the target or hardware assumptions openly.

References: https://docs.pytorch.org/docs/stable/generated/torch.compile.html ; https://huggingface.co/docs/transformers/main/torch_compile ; https://huggingface.co/docs/transformers/main/perf_train_gpu_one .
