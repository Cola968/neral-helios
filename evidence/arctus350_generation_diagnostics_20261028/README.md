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
