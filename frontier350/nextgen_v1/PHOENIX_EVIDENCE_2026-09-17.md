# Phoenix evidence ledger — 2026-09-17 — DRAFT / NO MODEL RELEASE

This is a snapshot of verified **experimental-branch** work, not a claim of improved chat performance or approved training data. PR #118 remains draft and unmerged; the pinned own SAFE checkpoint and all production weights are unchanged.

## Actual on-machine evidence

1. [Read-only SHA-pinned OASST2 selector run #35187672093](https://github.com/Cola968/neral-helios/actions/runs/35187672093): 135,174 original messages; 400 DE and 800 EN candidates before tree/prompt dedup; 1,077 train and 120 validation rows, 3 duplicate prompts skipped, 1,061 distinct trees after selection. This was in RAM, without independent frozen holdout, data export, GPU or approval.
2. [Original-selection content risk screen #35188034264](https://github.com/Cola968/neral-helios/actions/runs/35188034264): 1 email-pattern row, 4 phone-pattern rows, 23 answer rows under 40 characters, 58 answer rows over 2,500 characters, 37 prompts over 512 characters (near-screen truncation risk), no duplicated normalized answers in selected rows. All are heuristic flags only. They are not proof of PII, failure or complete review. Raw text/IDs were not logged.
3. [Isolated hash-ordered tree-diversity experiment #35188379870](https://github.com/Cola968/neral-helios/actions/runs/35188379870): 1,200 unique trees sampled, 4 duplicate prompts skipped, 1,196 final rows / 1,196 distinct trees (1,095 train / 101 validation). Legacy 1,197 rows / 1,061 trees. Candidate flagged 2 email-pattern rows, 4 phone rows, 31 short replies, 54 long replies, 29 long prompts, 1 repeated normalized answer. Some heuristic indicators improved and some regressed; **do not claim better answer quality, do not change production selection**. Candidate inventory is identified with SHA-256 `3b3605677d3c66a91d684372b7b2d373d09fcc06c66b330f409869ef90f85983` for reproducibility; no IDs or texts exposed.
4. The first diversity workflow [#35188260632](https://github.com/Cola968/neral-helios/actions/runs/35188260632) failed before data reading due missing root-level Python import. This was fixed and covered by an isolated direct-script subprocess regression, then the second run #35188379870 succeeded. Do not hide the initial failure.

## Safety and contracts

- Earlier single GPU telemetry: RTX 4070 25% utilization, 638 MiB used, 65 C. Original utilization <=15% blocked CUDA inference; no overheating proven. The current experimental CUDA diagnostic now requires three separately spaced fully passing measurements before Torch/weights, >=9,000 MiB free VRAM, and runtime temperature <=78 C both before and after each short prompt. Any bad/missing reading refuses; thresholds unchanged. Per-prompt monitoring is not continuous physical protection and cannot guarantee against hardware faults.
- **No GPU workloads or new training were launched in the current evidence campaign.** The real source audits and comparisons used read-only CPU processing via the existing WSL interpreter; source archive remains local, no training JSONL export or checkpoint writes.
- [Hosted synthetic CI #35188559996](https://github.com/Cola968/neral-helios/actions/runs/35188559996): 110 Phoenix unit/contract tests and 30 SFT tests pass. Synthetic code tests verify behavior, not model answer quality, source safety or training speed.
- New `phoenix_holdout_rubric_check.py` validates the operator's **separately authored and private** new v2 prompt hashes, frozen receipt and one-to-one scoring rubric ID coverage, language and skill distribution. Its tests use synthetic fixture prompts only. Docs: `PHOENIX_HOLDOUT_RUBRIC_V2.md`. No real Holdout v2 has yet been created, frozen or evaluated.

## Stop/Go: unresolved, do not silently promote

1. Actually create and independently review **new** private 24–100 DE/EN questions and acceptance rubrics, freeze them *before* any candidate training, and check exact/semantic overlap. Original private holdout was lost; historical NIS2 is not directly comparable.
2. Privately inspect flagged original source rows (including potential contact information) and independent source-license/attribution, privacy, unsafe/low-quality answer, near/semantic leakage and holdout review. Neither regex screening nor tree diversity automatically approves source use.
3. Only with independently reviewed data and an explicit authorization record, consider a bounded isolated SFT on a **copy** of the SHA-pinned checkpoint, with strict GPU preflight, runtime safeguards and opt-in. Do not alter SAFE, bypass caps or start long runs by default.
4. Compare original SAFE and any real candidate on the same independently frozen holdout using blind per-question rubrics, answer correctness, stability/EOS and a new NIS2-v2. Never interpret unit-test count, lower loss, generated inference token rate or tree count as proven model quality.

**As of this evidence ledger: training_approved=false, model_quality_improvement_verified=false, GPU_training_started=false, production_weights_modified=false.**
