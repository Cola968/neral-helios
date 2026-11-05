# Contract-v3 follow-up — 2 November 2026

Predecessor: 4a833089c5d2937477e6660d454722cddc3b634b, existing Draft222
on Draft220 ef65d9c. This is incremental, scoped review rework, not another
copy of the already published modules.

## Queue and sequence

Live queue revision80 showed own task neral-shared-control-221-v1 already
REVIEW_REQUIRED, attempt2, result PR222/4a83308. No expired RUNNING transition
was applicable. Agent listing showed only current root; process inspection
excluding current ancestors found no previous owner/task executor. After the
explicit operator rework request, unmodified pinned PR214 resolve(retry) and a
snapshot-guarded claim reclaimed this same task as attempt3. No manual JSON,
force push, expiry helper, new task key or foreign lock modification. The guard
refuses foreign READY/expired RUNNING tasks or changed foreign state, so claim
cannot take unrelated work. This is the final allowed attempt.

Implementation proceeded provider -> context -> controller -> factory -> full
tests -> publication. No parallel source agents were used in this revision.
All original app/legacy assistant/local runtime/store/core UI/training paths
were compared unchanged. Only leased package/tests/docs/evidence changed.

## Executed checks

- module-tests.txt: entire tests/neral_control suite: 239 passed,1 native Windows
  skip. Includes real Chromium on actual Helios in isolated preparation mode,
  calculator/static checker, profiles, keyboard,4 widths, session/errors and
  labelled cooperative cancellation fixture. No language-model generation.
- regressions.txt: module suite plus the same20 existing source-test files from
  ../README.md:468 passed,1 native Windows skip. Test command prefix:
  PYTHONDONTWRITEBYTECODE=1 HELIOS_DEVICE=cpu CUDA_VISIBLE_DEVICES=''
  /tmp/neral-control-check/bin/python -m pytest -p no:cacheprovider -q
- Four Node suites (helios_history,arctus_history,issue195_assistant,
  qwen25_history): all passed; compileall,pip check,diff check passed.
- Fake artifacts are deliberately tiny local integrity fixtures, never loaded
  or deserialized as model weights. Actual hash checks bind snapshot/adapter
  manifest to runtime identity. A forged status label is insufficient.
- Tests first reproduced absent provenance fields and artifact acceptance,
  missing allowlist/reparse checks, old profile caps/result/factory absence,
  lost cancellation token counts and missing telemetry treated as zero.
  Corresponding implementations passed after those RED runs.

No GPU inference/training, weights/model download, paid job, merge or deployment.
Python3.12.14, torch2.8.0+cpu, pytest8.4.2, Flask3.1.3, numpy2.2.6 and
waitress3.0.2 were installed only into a temporary isolated test venv.

## Exact historical-failure isolation

The earlier "8 failures/3 errors all outside neral_control" premise was false.
Two failures were development-time in-package context tests:
- test_denied_files_never_opened_by_search
- test_source_secret_is_redacted_with_original_digest
They were fixed before predecessor4a83308; both pass there and now.

Five unchanged tests require absent ignored data/checkpoints:
- tests/test_curriculum_v2.py::test_v2_splits_and_hashes
- tests/test_curriculum_v2.py::test_pilot02_lineage
- tests/test_growth.py::test_40m_exact_count_and_inherited_weights_unchanged
- tests/test_regressions.py::test_curriculum_disjoint_and_masked
- tests/test_regressions.py::test_original_untouched_and_parameter_count
assets-baseline.txt and assets-current.txt show the same5 FileNotFoundError
failures on a source-only git archive of4a83308 and this working revision.
No missing assets were generated/downloaded, and no training was run to hide them.

One failure and3 setup errors were local-socket restrictions:
- tests/test_helios_workspace.py::test_gateway_streams_before_backend_completes
- tests/test_helios_install.py::test_gateway_blocks_unpaired_and_cross_origin
- tests/test_helios_install.py::test_pair_cookie_and_unavailable_backend
- tests/test_helios_install.py::test_pair_rate_limit
historical-baseline.txt reruns these4 plus the2 context cases on predecessor
source with network-enabled local fixtures:6 passed. Current468 selection also
passes these cases. This establishes those environmental/development causes;
it is not a claim of a fully green unfiltered repository suite.

## Remaining acceptance limits

Native Windows/NTFS symlink,junction,concurrent rename/reparse and independent
security review remain a release blocker. Linux lstat attribute/Win32-handle
mocks are not a substitute. Model tool-generation quality, real active adapter,
RTX4070 VRAM/performance,100-task A/B/C/D scores and learned improvement remain
unmeasured. The blocking Ollama transport still permits only cooperative deadline
checks, not a hard in-flight wall-clock interruption. Unknown token totals stay
null; per-call/call-count token budgets remain bounded without relying on usage
telemetry. The public factory is inert and opt-in; no default product path changes.
