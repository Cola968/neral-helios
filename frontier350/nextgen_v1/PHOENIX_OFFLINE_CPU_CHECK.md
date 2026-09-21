# Phoenix: offline synthetic CPU regression check

The GitHub-hosted jobs for draft PR #120 previously failed before visible job steps. This does **not** establish whether the Python code passes. `phoenix_offline_cpu_check.py` provides a local alternative using the same entrypoint as `.github/workflows/phoenix-shard-distribution-cpu.yml`.

## Execute on a reviewed checkout of the PR branch

In PowerShell, open an existing Neral repository checkout containing the current `experiment/phoenix-blind-ab-audit-20260919` files. Preserve uncommitted changes; do not reset the working tree merely to run tests. Python 3.11+:

```powershell
python -B frontier350/nextgen_v1/phoenix_offline_cpu_check.py
```

The command prints a JSON status (`PASS`, `FAIL`, `TIMEOUT`), number of compiled files, number of **selected test modules**, number of tests actually run, and the exit code. Its process exit status is zero only if all 26 allowlisted Python files compile, the child exits successfully, at least 15 tests run, unittest reports `OK`, and no skips or expected failures appear. The test-count threshold catches gross collection failure; it does not prove that every test module contributed an individual test.

The existing eight runner self-tests can also be executed:

```powershell
cd frontier350/nextgen_v1
python -B -m unittest -v test_phoenix_offline_cpu_check.py
```

The runner compiles **11 source modules and 15 synthetic test modules** without importing trainers. It runs the 15 fixed test modules in a Python child process, sets `CUDA_VISIBLE_DEVICES=-1` and `PYTHONDONTWRITEBYTECODE=1`, and enforces a 180-second timeout. Coverage now includes answer export, candidate lineage/loading, blind A/B preparation and unblinding, paired-evidence calculations, native pilot guards, data shards and holdout gates. The runner does not intentionally open real checkpoints, private data, or holdouts or launch training. Test modules are executable Python code, however: inspect changes before running a new checkout.

A local synthetic test of this *expanded runner* passed four isolated cases (module selection and CPU isolation, partial collection refusal, added-source syntax refusal, and skipped-test refusal). **The complete 15-module regression suite has not been run on the actual checkout or owner's PC in this change.** The older eight-runner-self-test claim applied to the previous runner revision, not independently to the new revision. GitHub Actions remains unverified until real test steps run.

Even a real `PASS` establishes only synthetic CPU contracts, not human data provenance/privacy/license approval, independent holdout quality, actual 350M loading, GPU feasibility, training speed, bitwise replay, or a measured answer-quality gain. No training, merge, or checkpoint modification is authorized by the report.
