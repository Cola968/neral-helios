# NERAL 86-example QLoRA capability bundle (review only)

This additive tool composes the three **already reviewed**, frozen SFT sources
in the exact 10 + 16 + 60 order, without editing the original datasets or the
8 old pilot holdouts. It is based on the #231 reviewed corpus stack, which is
itself based on #228, #222, #220 and #219. The training code remains #220.

It is **not training**, a new semantic review, an inference test, a learned
improvement or a 10 November release approval. These 86 short, project-centric
examples are useful for an initial training experiment but too few to claim
broadly improved reasoning or independent-model competitiveness.

Run in a trusted checkout containing all dependencies and the source-reviewed
manifest files. Requires Python stdlib and existing `corpus.validate`, which
also relies on the project's CPU support modules. The destination must be a
**new, absent directory outside the repository** with an existing parent:

    python scripts/neral_capability_train_pack.py \
      --root . --out-dir C:/NERAL/reviewed-training/neral-86-v1

Outputs:

- `neral_train_86.jsonl`: deterministic verbatim UTF-8 JSONL lines.
- `neral_train_86.review.json`: preserves and remaps every original row
  digest, evidence and source SHA-256, with full downstream compatibility.
- `neral_train_86.provenance.json`: IDs, source hashes and non-training status.

**Gates:** verify phase-2.2 corpus + leakage commitments, verify each original
review manifest including source-file hashes, reject conflicting source hashes,
verify every row and exact 86 count, and finally call the existing
`scripts/neral_qlora_pilot.py` review checker on the combined output before
publishing. Any validation failure leaves the requested destination absent.
No Torch, Transformers, model or GPU is loaded; no download or mutation of
source review files is permitted. Existing output directories are never reused.

Only after a **separately approved** GPU-training experiment, a compatible and
isolated Windows QLoRA environment and a preflighted local verified HF snapshot,
pass the bundle as the trainer's *optional* dataset and review inputs. Do not
run this command to start training without the separate approval:

    python scripts/neral_qlora_pilot.py --dataset <new-dir>/neral_train_86.jsonl \
      --review-manifest <new-dir>/neral_train_86.review.json \
      --model-dir <verified-local-qwen-snapshot> \
      --output-dir <new-empty-adapter-directory> \
      --steps 3 --max-length 512 --run-local-gpu

The original three-step authorization was **only for a single smoke test**, not
for arbitrary or repeat training. The 86-example dataset does not establish
that such a test has run. Adapter promotion requires its own real base-versus-
adapter A/B on unknown tasks, no regressions and separate user approval.

CPU unit test: `python -m unittest discover -s tests -p 'test_neral_capability_train_pack.py' -v`.
The approved source branch must be reviewed as a stacked Draft PR; do not merge
other drafts or change source/holdout hashes to make this pack pass.
