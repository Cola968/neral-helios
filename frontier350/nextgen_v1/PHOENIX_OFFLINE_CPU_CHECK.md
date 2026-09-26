# Phoenix: offline synthetic CPU regression check

GitHub-hosted jobs for draft PR #120 previously failed before visible test steps. This does **not** establish whether Python code passes. `phoenix_offline_cpu_check.py` provides a local alternative using the same entrypoint as `.github/workflows/phoenix-shard-distribution-cpu.yml`.

## Execute on a reviewed checkout of the PR branch

In PowerShell, open an existing Neral checkout containing current `experiment/phoenix-blind-ab-audit-20260919` files. Preserve uncommitted changes; do not reset the working tree merely to run tests. Python 3.11+:

```powershell
python -B frontier350/nextgen_v1/phoenix_offline_cpu_check.py
```

The command emits JSON (`PASS`, `FAIL`, `TIMEOUT`), compiled-file and selected-test-module counts, actual tests run, and process exit code. It returns success only if all 26 allowlisted Python files compile, the test subprocess exits zero, at least 15 tests run, unittest reports `OK`, and no skips or expected failures appear. The threshold catches gross discovery failures; it does not prove every selected module contributed a test.

Eleven runner self-tests can also be executed:

```powershell
cd frontier350/nextgen_v1
python -B -m unittest -v test_phoenix_offline_cpu_check.py
```

The runner compiles **11 source modules and 15 synthetic test modules** without importing trainers. It executes those 15 fixed test modules in a Python child, sets `CUDA_VISIBLE_DEVICES=-1` and `PYTHONDONTWRITEBYTECODE=1`, and enforces a 180-second timeout. New coverage includes answer export, candidate checkpoint lineage/loading, blinded A/B and unblinding, paired-evidence calculations, plus previous native pilot, corpus and holdout checks. It does not intentionally open actual checkpoints or private data or start training. Test modules themselves are executable Python code: review branch changes before running them.

**Verification on 21 September 2026:** The expanded runner passed four synthetic tests in an isolated temporary directory. Its updated permanent self-test file then passed **11/11 local synthetic tests**, covering prior refusal cases, added modules and incomplete discovery. These mocked tests do not constitute a complete real-checkout regression run. Neither the complete 15-module suite nor a GPU end-to-end run was executed on the owner's PC by this change. GitHub-hosted CI remains unverified unless actual job steps execute.

Even a genuine full suite `PASS` establishes synthetic CPU contracts only, not legal/private data provenance, independent semantic holdout, 350M GPU fit, checkpoint recovery, throughput, or independently verified answer-quality improvement. No training, checkpoint modification, deployment or merge is authorized by this report.
