# NERAL gradient-health audit — Karpathy-inspired training diagnostics

Context: Andrej Karpathy's *Neural Networks: Zero to Hero*, especially "Activations & Gradients", motivates looking at gradient scale instead of inferring model quality from training loss alone. Hugging Face Transformers Trainer logs `grad_norm` **before** gradient clipping; NERAL v6 uses `max_grad_norm=1.0`.

## Evidence on 7 November 2026

The verified NERAL v6 run [42407667120](https://github.com/Cola968/neral-helios/actions/runs/42407667120) finished 120 optimizer steps, with train loss 1.0241 and validation loss 0.9859. Its console reports 24 spaced gradient samples; every reported sample exceeded 1.0, with observed values approximately 1.27–4.29. These are **24/24 observed samples**, not proof about all 120 steps. Clipping can be expected in stable training; this observation is a reason to inspect the training dynamics, **not** proof that clipping hurt accuracy or that increasing the threshold helps.

## New local CPU-only audit

`python -m scripts.neral_gradient_audit --private-report PATH/TO/scalar-private.json --max-grad-norm 1.0`

Uses only `recent_train_logs` and `steps_completed` from a *local* scalar-private report; ignores prompts, completions, credentials, weights and unrelated fields. Returns sampled pre-clipping norms, observed threshold crossings, actual sample count and the covered step range. The v6 scalar report keeps only its **last 12** log entries; consequently a historical audit of that JSON is not a 24-sample audit. Full 24-sample evidence is in the original run's console logs. The new `gradient_health` field in future v6 experiment reports summarizes **all callback logs** before the `recent_train_logs[-12:]` truncation. Neither one changes model weights, clipping settings or the release gate.

This is not a benchmark, a controlled A/B experiment or evidence that v6 generalizes better. Preserve the existing independent Qwen/v5/v6 evaluation, functionality tests, private data rules and manual GPU authorization. Do **not** tune gradient clipping using the sealed release-test suite.

References:
- https://karpathy.ai/zero-to-hero.html
- https://huggingface.co/docs/trl/sft_trainer (logged metrics and gradient norms)
- https://github.com/Cola968/neral-helios/actions/runs/42407667120
