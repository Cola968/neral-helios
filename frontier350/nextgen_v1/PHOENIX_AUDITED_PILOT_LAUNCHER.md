# Phoenix v0.4 native trainer integrity gate — experimental, no training approval

**Changed 19 September 2026:** The original `phoenix_weight_only_pilot.py` now invokes the full-shard approved-identity scanner AND frozen-holdout exact-token overlap gate **inside its own `main()`**, after the original `prepare()` operator-approval preflight and before importing `phoenix_cuda_diagnostic`, Torch, or starting CUDA. This also applies to explicit `--resume` invocations. The former claim that the direct pilot bypasses the checks is obsolete on this experimental PR branch. No automatically running workflow or approval is introduced.

## Two invocation paths

The actual, previously approved operator argument set can be passed unchanged to the native trainer:

```text
python frontier350/nextgen_v1/phoenix_weight_only_pilot.py [THE EXISTING COMPLETE APPROVED PILOT ARGUMENTS]
```

The old optional wrapper also remains compatible:

```text
python frontier350/nextgen_v1/phoenix_audited_pilot_launcher.py [THE SAME COMPLETE APPROVED PILOT ARGUMENTS]
```

**Prefer the direct entrypoint on this branch:** the wrapper currently runs its own preflight and *then* calls the native trainer, which now repeats both full-corpus and overlap checks. This is safe but creates redundant full-corpus reads; no speed improvement is claimed. The implementation preserves the wrapper for existing users rather than silently changing its invocation contract. No new bypass flags exist.

## Required operator evidence and check order

Follow `WEIGHT_ONLY_PILOT_OPERATOR_CONTRACT.md`: independently review provenance, source rights, privacy, semantic contamination and labels; author and SHA-freeze an unseen DE/EN evaluation holdout (8–500 cases with the exact validated schema); SHA-pin the unchanged original checkpoint, tokenizer, approved train and validation data, and a human-approved exact bounded manifest. Verify available disk, GPU VRAM and thermal headroom. None of those real owner approvals or real data scans has been performed by this PR.

In both direct launch and resume: (1) original `prepare()` checks explicit `--execute`, `--trusted-local-checkpoint`, pinned original weights and tokenizer, bounded dimensions, separate destination, training/validation corpus identities, human manifest, source review, frozen holdout and reserve disk; (2) `verify_snapshot` scans all selected uint16 shards for in-vocabulary IDs 0..16383, full SHA, duplicate bytes, window size and observed metadata mutation, and compares exact native data identities against the approved plan; (3) `require_zero_exact_overlap` re-reads and validates the approved frozen holdout SHA and schema, verifies tokenizer SHA/vocabulary, and searches raw and full chat-template question-token subsequences in **both** training and validation; (4) any hit, changed file, invalid data, mismatch, or incomplete diagnostic aborts before GPU probing; (5) only after success can GPU diagnostic, source-checkpoint loading, new optimizer and bounded training proceed. Terminal status and errors omit private question text, case IDs and shard paths. No successful audit grants approval.

**Scope and limits:** Exact token matching cannot certify semantic independence, paraphrase detection or unseen evaluation. Preflight file hashes are not an immutable snapshot: files might mutate after the scans. Manifest approval fields are self-reported, not cryptographically signed evidence of a review. The direct trainer is guarded on **this branch only**; other historic trainers/scripts are outside this change. A full-size model, actual own corpus, independent holdout, GPU training, checkpoint replay and blinded quality comparison remain untested. No 60–70k training tokens/s or quality gain is proven.

## Read-only verification

```powershell
cd frontier350/nextgen_v1
python -m py_compile phoenix_weight_only_pilot.py phoenix_audited_pilot_launcher.py phoenix_holdout_gate.py test_phoenix_native_pilot_gate.py
python -m unittest -v test_phoenix_native_pilot_gate.py test_phoenix_audited_pilot_launcher.py test_phoenix_holdout_gate.py test_phoenix_holdout_gate_integration.py
```

`test_phoenix_native_pilot_gate.py` parses the **actual trainer source** using Python AST and executes only the pre-CUDA prologue with mocked approval, corpus and holdout checks. Seven tests cover direct, resume and refusal sequencing; they do not run training or verify real owner data. Separate prior launcher/gate tests used tiny synthetic shards and an external tokenizer stub. Hosted GitHub Actions has repeatedly failed with no assigned runner or steps: the test command above is not a claim of CI success. Preserve draft PRs #118/#120 and issue #119 until authentic operator approval, actual measurements and independent free-answer evidence exist.
