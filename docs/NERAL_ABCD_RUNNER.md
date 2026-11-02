# NERAL A/B/C/D runner

Queue `neral-phase22-runner-v1`, operator-authorized stopped-worker retry, attempt
2. Stacked on strict benchmark #224 `c4c6a07`; the original pin `8c80bd2` remains
an ancestor. Only this document, the runner and its CPU tests change.

## Dependencies and commands

This is a coordinated draft stack: reviewed corpus module/data #228, strict
evaluation #224, completed-history/context #223 and bounded provider #225. Do not
replace missing dependencies with generated data or an older permissive backend.
The data root is an explicitly trusted reviewed repository checkout, not an
arbitrary uploaded directory containing executable Python source.

```powershell
python scripts/run_neral_abcd_eval.py plan --corpus-root C:\path\to\reviewed-repo --output plan.json
python scripts/run_neral_abcd_eval.py score --collection collection.json --reviews reviews.json --output scores.json
python scripts/run_neral_abcd_eval.py report --collection collection.json --reviews reviews.json --output report.json --markdown-output report.md
```

Plan validates all 100 frozen task/rubric commitments and the existing corpus
review/leakage gates on CPU. An optional --model-dir hashes the existing local
snapshot without loading weights. Without it the known expected revision is
reported with base_snapshot_verified=false. Missing/invalid adapter artifacts
leave C/D SKIPPED_NO_VERIFIED_ADAPTER with no responses. A/B remain base-only.

`collect` requires --confirm-execution before corpus/model work. It additionally
requires --model-dir pointing to existing validated local files; no model, GPU
job, API, account or dependency is downloaded/created. Optional --adapter-dir is
verified against the base; runtime activation is verified again, and a base
fallback never receives C/D credit. An unreleased pilot additionally needs the
separate --allow-unvalidated-adapter experimental flag. This document's tests
did **not** execute real collect or authorize training.

A calls the base runtime directly using the fixed audited template and common
system prompt: no tools, adapter, direct SOLVE route or ReasoningController. B/C
use the existing strict provider tool protocol and reviewed read-only/static
registry. D uses the actual ReasoningController, exact context counting and
controlled project fixtures. The provider receives the same common system text;
tool protocol/template overhead is an explicit treatment, not concealed as equal
input tokens. No network search is enabled by this runner. Each task is isolated;
no benchmark answer becomes another task's history.

--track resource_equivalent (default) gives every condition the same maximum of
one model call, 256 generated tokens and 20 seconds, with zero enabled tools for
A. D then uses fast controller mode; deterministic tools can use zero model
calls. --track product permits A/B/C/D respectively 1/2/2/4 calls, D deep mode,
and reports these different total budgets. The 256-token per-call limit, common
system, decoding/seed/base and 20-second task deadline remain explicit. Provider
deadlines are clamped to the task deadline. Actual kernel interruption, equal
measured resource consumption and performance on target hardware are not proven.
Only one runtime is held at once and it is closed on each condition's exit.

Collection records actual text, completion status, known/unknown usage, call
counts, tool provenance, latency and CUDA allocation/reservation peaks. Failed
cases stay in the inventory; runtime activation failures have explicit condition
states without simulated case output. If an unexpected failure destroys tool
evidence, its count/provenance stays unknown and scoring refuses to certify it;
no zero execution is invented. No automatic retries, scoring or GPU work starts
from plan/score/report. Existing output paths are refused before execution.

## Bound reviews and reporting

Reviews map each COLLECTED condition to one row per attempted task. Every row
requires exactly id, status (full/partial/fail), unsupported_claims,
failure_causes, reviewer, prompt_hash, rubric_hash and answer_sha256. The last
three must match the frozen manifest and exact collected answer. Supply actual
reviewed judgements; string matching cannot establish rubric compliance.

Scoring reuses evaluation.py settings/result/summary contracts. All four
available conditions use its complete evaluate() comparison. Partial A/B reports
retain skipped C/D; they never substitute fabricated adapter runs. The target is
newly solved baseline failures / all baseline failures. Numerator, denominator,
percentage, zero-denominator/missing-condition status, regressions, base/revision,
adapter hash, collection/manifest hashes, usage measurement counts and total call
budgets appear in JSON and Markdown. Imported execution and integrity assertions
are not independently authenticated; this runner makes no independent quality,
learned-weight or release claim.

## Actual CPU evidence

Five initial tests failed because no runner existed. Nine focused contract tests
now pass: confirmation precedes work, absent adapters cannot reach a loader,
plain A never constructs a tool registry/controller, unknown measurements remain
unknown, resources close, activation failure is explicit, exact-output/rubric
review binding, report file generation, overwrite refusal and the 20% formula
with a separate regression. Synthetic fixture results are labelled contract_fixture.

Real authored-data plan in a temporary source composition: 100 DEV tasks in
30/25/20/15/10 categories, 29,520 lexical comparisons, no flags, C/D skipped and
model_started=false. An import guard forbade torch/transformers/peft/bitsandbytes;
none were imported. No model or tokenizer weights were loaded.

Standalone runner-branch Windows module suite: **291 passed, 2 skipped,
2 deselected** (existing symlink/O_NOFOLLOW fixtures). Temporary source composition
with context, transport, Helios, evaluator, runner and corpus: **374 passed,
1 deselected**, including the real preparation-mode Brave UI test. Counts overlap;
do not add them. Final checks/commit are recorded in the Draft PR. Five historical
missing-data/checkpoint failures in the full old repository remain separate and
were not repaired by this scope. No actual A/B/C/D model run, training, weights,
extra costs, merge or deployment.
