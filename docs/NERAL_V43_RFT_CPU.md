# NERAL V4.3 RFT-QLoRA – CPU preparation, no GPU approval

The benchmark runtime is pinned to `5877b73851c986f39f4dbeb470cd8fb08bd2707d`. This development branch prepares rejection sampling, assistant-only supervised fine-tuning, and a bounded extension to the Windows AST evaluator. It contains source code and synthetic test fixtures. A future GPU pilot requires separate authorization.

## 1. Read-only ObsidianHard diagnosis

From the repository root on W39, using its existing CPU-capable isolated environment:

```powershell
$py = "$env:LOCALAPPDATA\NERAL-private\v43-obsidianhard-tools\python\Scripts\python.exe"
$env:CUDA_VISIBLE_DEVICES='-1'
& $py -B -m scripts.neral_v43_rft_diagnose --recheck
if ($LASTEXITCODE -ne 0) { throw 'Diagnosis incomplete' }
```

The diagnosis prints aggregate family, category, AST feature, and recheck counts. Each recheck uses the restricted CPU backend on the archived first draft. Rejected repair text was outside the historical capture. Keep all benchmark tasks and replies outside training. The evaluator interprets bounded `for` loops and `break`, the `list` and `set` built-ins, and explicitly mapped math and statistics functions. The existing instruction budget, finite collection bounds, and AST validation apply to these operations.

## 2. Native CPU gate

Run both dependency-light suites and the separate required native integration suite (all in the checkout, no CUDA):

```powershell
& $py -B -m unittest discover -s tests -p 'test_neral_v43_rft_cpu.py' -v
& $py -B -m unittest discover -s tests -p 'test_neral_v43_rft_diagnose_train.py' -v
& $py -B -m unittest discover -s tests -p 'test_neral_v43_rft_native.py' -v
```

The native gate exercises SymPy, z3-solver, pytest and Hypothesis using synthetic fixtures. Stop on any nonzero exit code. Reserve fixture seeds for tests.

## 3. Separate private training corpus

Create a new owner-only root outside the repository: `%LOCALAPPDATA%\NERAL-private\v43-rft`. No uploads of this root or its logs, seeds, prompts, answers, weights, receipts. Use a private cryptographic 256-bit seed. `catalog --seed <private-hex64> --count 600 --out <new-catalog-folder>` generates a private catalog and a distinct prompts-only file. The first pilot families are independently generated exact arithmetic, bounded integer implication and progressive tariffs with test examples and Hypothesis properties. Increase difficulty and add families only after the 16-case diagnosis, on new tasks. This catalog is deliberately independent of ObsidianHard and does not ingest benchmark gold. Do not use the test fixture seed. Keep a new, separately generated and sealed development holdout in addition to the original protected sets; the current exact-hash denylist cannot establish semantic non-overlap, so require a family/template review before training.

`protected.json` is a private mapping with exact keys `Reference50`, `Blind200`, `ObsidianHard`. For each: `{ "path": "<local sealed prompt JSONL>", "sha256": "<actual SHA256>", "immutable_files": [{"path":"<protected result file>","sha256":"<known receipt hash>"}, ...] }`. Use the original Reference50 sealed50.jsonl and its baseline-receipt.json plus all 50/50 results and candidate receipts; use ObsidianHard V1/private.jsonl and the recovery responses.jsonl/manifest/measurements; include the complete Blind200 archive inventory with the actual 421 records and its independently authenticated prompt manifest covering all 200 tasks. Supply correct hashes from earlier seals, not newly asserted values. For Blind200, the actual archive inventory and coverage must be independently confirmed locally; the exact archived receipt is not available here. The tool checks all provided hashes before and after and rejects missing coverage; it cannot certify a deliberately incomplete inventory. The known Reference50 and ObsidianHard seal/receipt hashes are enforced. Use no protected data as training samples. The immutable files are read, never edited.

The later candidate-collection stage requires separate GPU approval or an independently reviewed pre-existing pool of fresh tasks only. Send `model_prompts.jsonl` without the private catalog. Candidate JSONL rows: `{"id":"RFT-00000","draft":"..."}`. Produce multiple fresh drafts per task if desired. `feedback` receives the private catalog, candidate pool, protected manifest and their SHA-256 pins; it regenerates the catalog from its seed, runs actual V4.3 checks and emits a separate repair-prompts file only when independent counterexample evidence exists. Copy each exact `repair_input_sha256` into the candidate row along with its independently generated `repair` string. No model-claimed proof is trusted. The CPU build independently rechecks repairs, rejects failed/unsupported/unknown answers, verifies hidden tests on fresh code, deduplicates source tasks and never trains the rejected draft as target.

Replay source: owner-reviewed fresh JSONL rows `{"instruction":"...","response":"...","meta":{"kind":"replay","category":"everyday|knowledge|math|code","task_hash":"<SHA256 of whitespace-normalized instruction>","reviewed":true,"source":"independent_replay"}}`. Review factual correctness and diversity outside all benchmarks. Pin the approved file hash. No automatic import from 421 Blind200 or Reference50/ObsidianHard. `build --seed ... --catalog ... --catalog-sha ... --candidates ... --candidates-sha ... --replay ... --replay-sha ... --protected ... --protected-sha ... --out <new-folder> --total 200` accepts at most one row per task, at least one successful repair, and exactly 60 replay / 140 RFT rows; fails on insufficient accepted data. `feedback` uses the same catalog/candidate/protected parameters, without replay or total. Keep SHA256 pins for both manifest.json and train.jsonl. Replay percentage is by rows, not token count, and does not guarantee zero forgetting.

## 4. CPU dry run and later pilot

`python -B -m scripts.neral_v43_rft_train --bundle <bundle> --manifest-sha <sha> --protected <manifest> --protected-sha <sha>` is CPU-only by default. Add `--check-tokenizer --checkpoint-manifest <manifest> --checkpoint-sha <sha>` for native tokenizer validation, with an owner-pinned JSON mapping of all checkpoint files to their SHA-256 values. The mask compares the tokenized native generation prefix with the complete chat. Labels include the assistant response and end-of-turn token. Inputs outside the context budget fail validation. TRL 1.18.2 preserved these labels through the CPU data loader with synthetic data and a tiny test model at zero optimizer steps. Use an isolated training environment for TRL.

Pilot configuration: 4-bit NF4 double quant, BF16 when available, LoRA rank 8/alpha16, all-linear, batch 1, gradient accumulation 10, context 1024 (512–1024 supported), paged AdamW 8-bit, LR 5e-5, checkpointing, no external reporting, no old LoRA. Use existing memory-bounded embedding preparation and exact supervised-position loss. 10,240 MiB free minimum, 9 GiB torch allocator cap, 10.5 GiB sampled total limit. A sampled limit cannot guarantee no transient overrun; stop on failed VRAM sampling or cap breach, never release an unverified adapter. Windows and driver overhead prevent all 12 GB being available. No training fit claim from CPU results.

`Start-Neral-V43-RFT-Pilot.ps1` defaults to CPU-only and stages the native tokenizer check. Its Windows PowerShell test exercised argument quoting, separate stdout/stderr, native exit codes, and the approval guard with a synthetic subprocess. A future authorized `-RunGpu` launch checks the pinned owner, VRAM and output directory before one optimizer step. Its rollback path verifies the restored owner and HTTP 200, then rechecks protected hashes. GPU compatibility and live rollback for this pilot still require an authorized run.

## 5. Release gates after future one-step pilot

Require finite loss, positive finite gradient, actual adapter hash change, correct step count, sampled GPU peak below cap, successful rollback and unchanged baseline receipts. Then separately authorize full training (not implemented as an automatic next step). Compare new adapter against raw V4.3 on fresh independent holdout; Reference50/ObsidianHard remain evaluation-only. Verify no regression on everyday tasks, all 38 Reference50 subject tasks, held-out code/math, latency, VRAM and repair harm; use a new blind hard holdout to avoid repeatedly optimizing to ObsidianHard. Never auto-promote based on training loss. The original frozen baseline and original benchmark files stay intact.

## 6. Mandatory 200-row pilot composition gate (CPU-only)

The pilot now selects exactly 50 SymPy, 45 Z3, and 45 independently checked code rows, prioritizing verified repairs; at least 20 successful repair rows must be present. Replay is exactly 15 independently reviewed rows in each of everyday, knowledge, math and code. Missing quotas fail closed; do not synthesize model failures or relabel oracle answers as sampled model responses. The 200-row build requires `--holdout <private-jsonl> --holdout-sha <sha256>`: a separately reviewed 60-row development holdout (15 rows per category, each containing `prompt` and `category`). Keep reference answers private and outside training; pin the file after independent review. The builder refuses exact normalized prompt overlap with the protected datasets and the fresh catalog, and the trainer revalidates the holdout pin, selection quotas and overlap on every preflight. This is exact-prompt screening, not proof of semantic independence. GPU smoke requires exactly 200 audited rows and separate authorization. No changes to the original benchmark data or worker loop limits.

Run `python -B -m unittest discover -s tests -p 'test_neral_v43_rft_pilot.py' -v` before the native gates. Ten CPU tests passed in the development environment using synthetic fixtures and isolated pool/bundle checks. The native W39 gate must be rerun against the new commit; no actual accepted model samples, new holdout, reviewed replay or optimizer update are claimed by these fixtures.
