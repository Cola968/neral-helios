# Phoenix: offline synthetic CPU regression check

The GitHub-hosted synthetic jobs for draft PR #120 have repeatedly failed before any visible job steps or runner assignment. This does **not** establish whether the Python code passes. `phoenix_offline_cpu_check.py` provides a local, reproducible alternative that uses the same core regression set as `.github/workflows/phoenix-shard-distribution-cpu.yml`.

## Run on an existing, reviewed checkout of the PR branch

Open PowerShell in your Neral repository working copy **after confirming it contains the current `experiment/phoenix-blind-ab-audit-20260919` files and after saving any uncommitted changes**. Do not overwrite a working tree to run this. Use an installed Python 3.11+ interpreter:

```powershell
python -B frontier350/nextgen_v1/phoenix_offline_cpu_check.py
```

The command prints a single JSON status: `PASS`, `FAIL`, or `TIMEOUT`, the number of Python files compiled and the number of synthetic unit tests actually run. Its process exit code is zero **only** if all allowlisted files compile and unittest reports a nonempty `OK` with no skips; other conditions are nonzero. The eight runner self-tests can be run separately:

```powershell
cd frontier350/nextgen_v1
python -B -m unittest -v test_phoenix_offline_cpu_check.py
```

The runner compiles the native pilot, corpus/holdout validator and related modules **in memory** without importing the trainer, then invokes exactly eight allowlisted synthetic unittest modules in a child process with `CUDA_VISIBLE_DEVICES=-1`, `PYTHONDONTWRITEBYTECODE=1` and a 180-second timeout. It neither opens real checkpoint/dataset/holdout files nor executes the training CLI. The tests are executable code, however: review new branch changes before running them, and ensure the modules have not been replaced with untrusted code. The runner captures child test output and emits only an aggregate JSON status by default. To diagnose failures, inspect the locally run test commands or their source; do not upload private paths, credentials, data, weights or holdouts as CI logs.

**Scope and limitations:** The runner's eight self-tests passed locally with fabricated files and a mocked child execution. The full eight-module Phoenix regression suite has **not** been executed against a complete checkout in the development environment, and the actual owner's PC execution has not occurred. Even a genuine local `PASS` would verify only synthetic CPU code contracts, not source provenance, real holdout independence, 350M checkpoint load, GPU fit, training throughput, checkpoint recovery or model quality. GitHub CI stays unverified until a real runner executes the corresponding test steps. Do not launch a pilot without genuine independent dataset/holdout review and SHA-bound owner approval.
