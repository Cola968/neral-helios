# Issue #142: report-only development diagnostics validation

Base: `105502d` on `fix/arctus350-release-integrity-20261028`.
Branch: `codex/arctus350-generation-diagnostics`.
Only synthetic cases were used. No real generated report, private checkpoint,
tokenizer, original prompt, frozen benchmark suite or model weights were loaded.
No training, remote HF job or GPU workload was started.

## Reproduce

Python standard library only for the changed module and the integrity suite:

```sh
python -m unittest tests.test_chat350_dev_generation_diagnostics -v
python -m unittest tests.test_chat350_dev_generation_diagnostics tests.test_chat350_capabilities tests.test_arctus350_release_integrity tests.test_arctus350_generation_integrity tests.test_arctus350_gemini_review_regressions tests.test_chat350_checkpoint_selection_report tests.test_gpu_precision_probe350 tests.test_roblox_checkpoint_probe350 tests.test_train_precision350 tests.test_chat350_nll_partition tests.test_roblox_fp32_norm_probe350 tests.test_roblox_fp32_full_step350 tests.test_arctus350_short_sft_pilot -v
python -m py_compile frontier350/chat350_dev_generation_diagnostics.py tests/test_chat350_dev_generation_diagnostics.py
```

For the full existing chat suite, use the unchanged final test command in
`.github/workflows/arctus350-release-integrity-ci.yml`, on Linux. That hosted
CPU workflow now also runs the new diagnostics suite.

## Observed local outcomes

- Initial RED: 24 tests; 20 expected missing-module failures and 4 CLI errors
  before implementation. No production module existed at that point.
- Initial GREEN: all 24 diagnostics tests passed after implementation. CLI
  tests initially encountered Windows sandbox temp-directory permissions;
  setting `TMP`/`TEMP` to a writable workspace temp directory resolved that
  environment issue without changing test expectations.
- Additional edge RED: 30 tests, one `OverflowError` for an enormous JSON
  integer in a summary rate. Checking numeric bounds before float finiteness
  fixes that case as an ordinary rejected input.
- Final diagnostics GREEN: **30 tests passed**.
- Release integrity plus new diagnostics: **96 tests passed**.
- Compilation and direct-script `--help`: passed.
- Existing chat + integrity + initial 24 new tests: 179 tests, 2 failures,
  4 skips. Both failures are `tests.test_external_bench350.ExternalEvalTests`
  (`test_safe_math`, `test_safe_mbpp_pass`): the unchanged MBPP child runner
  imports Unix-only `resource`, which is absent on Windows. Direct
  `python -I -c 'import resource'` confirms `ModuleNotFoundError`. Three
  optional Torch/NumPy checkpoint tests and one Windows symlink test skipped.
- Broad `python -m unittest discover -s tests -v`: 212 tests, the same 2
  Windows MBPP failures, 33 import errors for unavailable `pytest` / `torch`,
  and 4 skips. All failed module names are retained in `validation.txt`.
  This is **not** reported as a passing full-repository run. Most other
  repository tests are pytest functions and are not collected by unittest.

`validation.txt` records test names and runner outcomes from these runs. The
pre-existing Windows/dependency failures were not hidden or patched away.
The pull request's hosted Linux CPU checks provide the platform-appropriate
verification for the relevant existing suite.

## Supported conclusion

The evaluator source at the base commit does not serialize the last user text,
EOS identity, per-case token limits, generation settings or complete development
suite provenance. This is a format audit, not a measurement of 1B/2B/SFT quality.
Empty/short decoded responses and raw-ID repetition are observable. Complete
echo/stop diagnosis and a valid model comparison remain unavailable from that
format. The CLI emits explicit missing metadata and exits 2; it never fabricates
EOS counts or declares a model improved. The README proposes a minimal future
evaluator schema without changing the evaluator or accepting invented fields.

## Critical review of PR #145

Reviewed head: `1362fce0b4158e3722634c089d57008dfe92bc01`.
The follow-up found and fixed these gaps with failing regression tests first:

1. Arithmetically plausible but impossible EOS summaries were accepted: more
   EOS hits than nonempty token sequences, no possible shared terminal ID, or
   the alleged stop ID appearing before a row ended. Source ngram averages
   could also contradict every possible fixed EOS. An existential consistency
   check now rejects those contradictions. It neither authenticates an EOS ID
   nor supplies one to diagnostic metrics; EOS remains unmeasured.
2. Equal suite hashes could hide changed, missing or added `expected_exact`
   targets. Comparisons now require identical per-case target presence/values.
3. Prefix echo treated identifiers, combining-mark words and contractions as
   full prompt repeats (`foo_bar`, `x` plus combining acute, `can't`). Unicode
   word continuation is now respected; malformed Unicode echo input is rejected.
4. The CLI collapsed all validation failures into an unhelpful generic message.
   Fixed diagnostic reasons now identify the failing check without reflecting
   report values, private prompts, parser input or filesystem paths.
5. Path-resolution errors could escape as tracebacks, and output used direct
   truncating writes. Path errors now fail safely; same-directory atomic
   replacement protects existing results on IO failure and cleans temporary
   files. Callers must still honor exit/status and never reuse stale output.

Validation is recorded in `review-validation.txt`: semantic RED (39 tests;
14 failing assertions/subtests plus one missing-reason error), then IO RED
(42 tests; one failure, one error, one hardlink skip). The final diagnostics
suite has **43 tests**, and diagnostics plus existing integrity tests have
**109 tests, OK (one Windows hardlink skip)**. Compilation and diff checks pass.
A further hand-calculated fixture verifies that valid EOS-trimmed source
averages do not replace the raw-token diagnostic metrics. A compatibility
matrix uses the existing `summarize_generations` helper on synthetic rows.

The full Windows unittest-discover run now has 231 tests: the **same** two MBPP
failures and 33 missing pytest/torch import errors as before, plus five skips
(the previous four and the new hardlink check). Failure-name sets were compared
and are unchanged. The hosted Linux CPU workflow runs all new diagnostics,
including the hardlink test, plus the relevant existing integrity/chat suites.
No model/evaluator/training or GPU workflow was modified by the follow-up.
