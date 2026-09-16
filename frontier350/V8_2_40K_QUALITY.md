# V8.2: 40k accepted tok/s with unchanged quality — experimental work

**Status: NOT ACHIEVED / NOT PROMOTED.** A CUDA peak is not an accepted-token throughput result. This branch neither alters the running fixed-checkout V8.1 workflow nor starts any GPU workload. No training/checkpoints/corpora are uploaded to GitHub or Hugging Face.

## Definition of success

- Three successive **complete 5M accepted-token** stages, no rejected or partial-salvaged stage in the measurement window.
- Each stage >=40,000 accepted tokens per elapsed wall-clock second, and the 3-stage window >=40,000. Start time is before baseline/evaluation; end time is after checkpoint, recovery, smoke evaluation and any full audit. No cherry-picked CUDA peak.
- Each stage passes existing continuity health gates; any due full SAFE audit must be promoted. Full NIS2, generation, math, code and release gates remain **separate**: a fast continuity checkpoint is not a good chatbot by itself.
- RTX 4070 temperature, VRAM limits, optimizer/RNG resume, token/update, training data/replay mixture and quality thresholds unchanged.

`apply_v8_2_speed_evidence.py` adds **only two wall-time fields** to accepted/rejected progress entries; the future training workflow invokes it after existing patches. `verify_40k_quality.py` computes accepted rates from these records and returns a nonzero exit when unverified. The workflow saves the speed report as an artifact without weakening or substituting its existing training/release gates. `publish_live_status.ps1` now calculates progress only from accepted tokens and intentionally suppresses unsupported ETA guesses.

## Evaluation startup/I/O optimization candidate — NOT ENABLED

`eval_health_v8_2_combined.py` reads a frozen checkpoint **once**, runs the unchanged 35-case smoke logic and 4-batch validation logic, and emits reports with the **original** protocol hash inputs. The existing controller still calls its two independent evaluators; **do not switch to the combined evaluator before parity and performance tests**.

On the self-hosted GPU, **only after the current training workflow has ended** (or an explicitly approved safe pause), use a separately preserved, SHA-verified checkpoint and identical tokenizer/validation set. Execute the independent `eval_recovery_smoke_v3.py` and `eval_validation_loss350.py` with the controller's exact parameters, then execute `eval_health_v8_2_combined.py` with `--checkpoint`, `--expected-tokens`, `--rotation`, `--tokenizer`, `--valid-data`, `--smoke-out`, `--validation-out`, `--seq 512 --batch 4 --batches 4 --seed 351337`. Run `verify_health_parity_v8_2.py` on the four JSON reports: their checkpoint/tokenizer identities and **all fields must match exactly**. Any difference rejects this candidate. Preserve elapsed timings separately. A single successful parity test does **not** prove the candidate achieves 40k.

Next, only if same-checkpoint parity succeeds, propose a separate opt-in integration (with original evaluator fallback), benchmark >=3 fully validated 5M stages against the baseline with the same training configuration and record accepted-token wall throughput/quality. Promote only on demonstrated throughput gains **and** unchanged gates. Do not modify the running job or raise thermal thresholds to chase the target.

The user is not on Hugging Face Pro, and the trainable weights are stored only on the local Windows runner. Hugging Face Jobs cannot speed up this local run without an authorized cloud setup, checkpoint transfer and budget approval. No HF Jobs were started.
