# Phoenix audited pilot launcher — experimental; no training approval

`phoenix_audited_pilot_launcher.py` is an **optional, separate entrypoint** around the existing `phoenix_weight_only_pilot.py`. It does not start automatically, is not referenced by any training workflow, creates no approval and does not alter the old 350M weights or shards. Existing direct pilot entrypoint remains available and does **not** automatically execute this additional vocabulary audit; use this launcher for the extra gate after independent human approval.

## Prerequisites and intended invocation

Follow **all** requirements in `WEIGHT_ONLY_PILOT_OPERATOR_CONTRACT.md`: independently review source rights, licensing, privacy, content and contamination; create and freeze independent bilingual holdout; independently SHA-pin the original checkpoint, all data and explicit human-approved manifest; check the RTX 4070, disk and temperatures. The approval does not exist as of 19 September 2026, so no actual invocation has been authorized or performed.

Only once the prerequisites are real, take the *exact argument set* specified for the existing pilot entrypoint and replace only the Python program path:

```text
python frontier350/nextgen_v1/phoenix_audited_pilot_launcher.py [THE EXISTING FULL HUMAN-APPROVED PILOT ARGUMENTS]
```

No new bypass flags exist. `--execute`, `--trusted-local-checkpoint`, the separate output directory, original checkpoint SHA, human manifest plus independent SHA, source review, frozen holdout, train/validation data and bounded token/run parameters still go through the **original** pilot parser and `prepare()`. If this initial preflight rejects anything, no corpus audit or training starts.

After approval preflight, the launcher streams **every byte** of the selected uint16 train/validation shards, validates vocabulary IDs `<16384`, alignment and window availability, duplicate full-file hashes and metadata stability. It rebuilds the native `corpus_identity` hashes from the *just scanned* bytes, with identical tokenizer SHA, sampler version, file names, lengths, seeds, microbatch/evaluation settings, and refuses any mismatch against the approved `prepare()` identities. It prints only shard counts and Boolean safety status, never content, paths or per-file SHA values. It then calls the original pilot's `main` with unchanged args, which deliberately re-executes its complete approval, SHA and disk guards before loading Torch/CUDA.

**Boundaries:** This is not an immutable filesystem snapshot: files can change after the scan, including during a GPU run. A malicious attacker may restore timestamps or replace identical-byte data. Content licensing/privacy and semantic train/holdout leakage are not checked by this tool. The SHA-bound manifest is not an identity signature, and its Boolean human-approval declarations are not independently verified by the program. No actual owner corpus, real 350M GPU pilot, checkpoint/recovery, answer-quality improvement or tokens/s has been measured via this launcher. The pilot's existing 262,144-new-token cap, output separation and full exact-resume restrictions remain.

Synthetic-only test command:

```bash
cd frontier350/nextgen_v1
python -m py_compile phoenix_audited_pilot_launcher.py test_phoenix_audited_pilot_launcher.py
python -m unittest -v test_phoenix_audited_pilot_launcher.py
```

Six local tests passed on 19 September 2026 with tiny synthetic shards and a fully mocked pilot. They check byte changes of identical length, out-of-range IDs, wrong approved identities, refusal without approval and delegation *only after* matching preflight. They are **not** a real launch, and GitHub-hosted runner CI remains failing before executing steps. Keep PR #118/#120 draft and issue #119 open.
