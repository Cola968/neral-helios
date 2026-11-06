# NERAL Qwen2.5-Coder-7B Rank32 — first paired inference smoke

**Draft, review required.** This is a small *development* smoke test (6 user-provided coding tasks), not the independent FINAL holdout, a training step, a model release, or proof of a quality gain.

## Execution contract

- A trusted, verified local HF snapshot at revision `c03e6d358207e414f1eca0bb1891e29f1db0e242`.
- The **existing** one-shot trained rank-32 adapter from GitHub run `42078722575` remains only on the user's Windows machine in `NERAL-QLoRA-Research/actions-r32-42078722575-1/adapter`.
- First reuses `Start-NERAL-QLoRA-Action.ps1 -Mode preflight` from main (including model integrity, dataset, environment, CUDA and VRAM checks). No new training authorization is supplied.
- Local `transformers` and `peft` run sequential deterministic NF4/BF16 inference on the same six fixed prompts; after each model is released, the other is loaded. The adapter manifest must pass the existing pinned rank-32 validator.
- Generated code and detailed results stay **local only** in a unique `NERAL-QLoRA-Research/comparison-<runid>-<attempt>` folder. It is never committed or uploaded.
- Code may be executed **only inside an already available, local** `python:3.11-alpine` Docker Linux image with network disabled, read-only filesystem, non-root user, dropped capabilities, CPU/RAM/PID quotas and per-test wall clock limit. No image download. Docker is *not* an absolute security boundary.
- If Docker or the image is unavailable, only inference and syntax checks run; runtime tests are marked `not evaluated`, **not passed**. No approximate correctness heuristics.
- Uploads only bounded `summary.json` (test counts, results and adapter identity hash); **never** generated code, checkpoints, weights, prompts or datasets.
- The new workflow is manually dispatched from `main` only and requires `RUN_READONLY_R32_COMPARISON`. No automatic runs on push or PR, no installs, merges or model activation.

After reviewed and merged, use GitHub Actions: `NERAL Qwen7B Rank32 read-only paired comparison` → `Run workflow` → `main` → enter `RUN_READONLY_R32_COMPARISON`.

## Interpreting results

Six tasks are far too few to make statistical or causal quality claims. Even 6/6 on this suite would not validate NERAL generally; no claim of real improvement or training effects follows. This is a correctness check on model loading, deterministic generation and optionally isolated runtime tests. Use independently reviewed larger holdouts and matched A/B/C/D before any release or adapter activation.

## Tests

Run from repo root: `python -m unittest discover -s tests -p 'test_neral_r32_compare.py' -v`. This suite never imports torch or uses Docker. The GPU+Docker end-to-end path is unverified until actually run on the PC. Do not execute code from an unreviewed pull-request checkout on a self-hosted machine.