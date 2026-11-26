# V4.4: CPU reasoning quality candidate

This branch is isolated from the frozen release, the accepted web branch and the
original RFT commit. It adds a separate versioned catalog and quality audit;
it never changes the owner stop guard, GPU cap, benchmark isolation, or
service configuration. No private data is included.

- Nine deterministic task families: arithmetic, linear systems, quadratic
  extrema; capacity, order chains, balance constraints; tariffs, bounded sums
  and counting multiples.
- Mathematical candidates are checked by independent SymPy computation.
  Logic and code reuse existing restricted V4.3 verification paths.
- Audit replays selected rows from original candidate records, checks actual
  failed drafts, repair prompt hashes and verified corrections, and requires
  at least 20 replayed repairs plus minimum family coverage.
- Holdout family labels must be independently reviewed; overlap with any
  catalog or replay family is rejected. Normalized prompt fingerprints also
  reject simple numerical substitutions. This is not proof against all
  paraphrases or semantic leakage.
- On this development branch only, the CPU catalog, feedback and build commands
  accept explicit `--catalog-version v2`. Legacy v1 remains reproducible.
  The v2 builder uses balanced family quotas and pins catalog and candidate
  sources. Bundle loading independently repeats the audit and refuses a
  mismatched seal or report. Original GPU gates and owner checks are unchanged.
- This is not a local training acceptance: native Windows testing, protected
  seal and service identity checks, actual model sampling, semantic human
  review and the explicitly gated one-step smoke still remain required.

Run only synthetic CPU tests:
`python -m unittest discover -s tests -p test_neral_v44_rft_quality.py -v`

## One-step telemetry and local acceptance

The smoke report records the first pre-update microbatch mean supervised-token
cross-entropy as `start_loss` (not a whole-corpus baseline), synchronized wall
clock step duration, 10 accumulation microbatches, total trainer runtime and
existing sampled/allocator peaks. It requires one effective optimizer update.
No additional forward pass or extra optimization step is introduced.

The earlier protected preflight produces a legacy v1 catalog. Do not relabel
that catalog v2. Local acceptance of the new commit, actual protected artifact
and owner backup verification, independent holdout review and the existing
GPU/owner gates remain mandatory. The approval tied to the original RFT
commit is not automatically an approval for a changed commit.

## Paired development evaluation (offline)

`scripts/neral_v44_paired_eval.py` aggregates already graded 60-case baseline
and adapter receipts. Both receipts must identify the same pinned holdout,
checkpoint, grading rubric and inference conditions; the baseline must have
no adapter and the candidate must identify its adapter. Every case must be
present exactly once, with the correct normalized prompt hash. Timeouts and
unsupported answers count as incorrect rather than being excluded.

The report records exact paired gains/regressions, category deltas, latency,
status counts and an exact two-sided McNemar p-value. This is a development
holdout check, not a replacement for the sealed benchmark or proof of general
intelligence. Grader correctness and review provenance are external claims;
this tool cannot verify those claims merely from hashes. No promotion or GPU
execution is authorized. No actual model answers are created by these tests.

The grading receipts contain no reference answers; supply private, independently
pinned receipts with rows containing id, prompt_sha256, status, correct and
latency_ms, plus holdout_sha256, checkpoint_sha256, conditions_sha256,
grader_sha256 and adapter_sha256 (null for baseline). The holdout must also have
unique case IDs, independent review and family labels.
