# Arctus 350M — post-100M execution plan

Status date: 2026-09-10

## Decision

The 100M checkpoint is a successful bounded training milestone, not a chatbot-readiness claim. The next expensive step is gated by checkpoint evaluation and a fresh locked corpus. Stage-6 did not beat the Stage-5 speed record, so speed-only sweeps are paused.

## Measured 100M training evidence

- Parameters: 349,825,152
- Start checkpoint: 25,001,984 tokens
- Stop checkpoint: 100,001,792 tokens
- Steps: 6,104
- Best validation loss observed during the run: 4.068359375
- Final validation loss: 4.09765625
- Mean train loss, first 20 logged points: 5.4669921875
- Mean train loss, last 20 logged points: 3.8432226539
- Production TPS median across logged windows: 18,463.72 BPE tok/s
- Production TPS p10/p90: 17,960.78 / 18,541.56 BPE tok/s
- Peak allocated GPU memory: 4.5297 GiB
- Compile gate: passed; eager/compiled eval-loss delta 0.0, train-loss delta 0.0, max gradient absolute delta 0.0001983643.

The final validation loss is only 0.0293 above the best value (~0.72%), which is consistent with evaluation noise and is not by itself evidence of overfitting. The old training loop sampled a different random validation window at each evaluation step, so cross-step validation comparisons are directionally useful but not a perfectly frozen benchmark. The new 100M evaluation therefore uses a fixed seed and fixed protocol.

## Execution gates

1. Run a fixed-protocol 100M checkpoint evaluation: held-out validation loss plus deterministic completion probes and generation-health metrics.
2. Build a fresh 1.10B-token locked foundation corpus: 1.05B train + 50M validation, using only the already pinned ODC-BY source revisions and the locked Neral 16K BPE tokenizer. Exact normalized-document deduplication is shared across validation and train, so identical documents cannot cross the split.
3. Run the existing fail-closed corpus admission checks (token accounting, shard SHA-256, tokenizer digest, source revisions, licenses, benchmark separation).
4. Do not start 100M→500M automatically. Start only after reviewing the 100M evaluation and the 1.10B corpus manifest.
5. When approved, resume the same 10B learning-rate schedule from the 100M checkpoint and stop at 500M cumulative tokens.

## Time budget

400M additional BPE tokens / 18,463.72 measured median tok/s = about 6.02 h of pure training compute. Allow roughly 6.3–7.0 h end-to-end for evaluation, checkpointing and I/O. This is an estimate; the 500M production run must report its own sustained TPS.

## Non-goals

- No SFT before the foundation checkpoint is adequately trained.
- No foreign model weights.
- No 10B blind run.
- No further Stage-7 speed search before the 500M capability gate unless a concrete training blocker appears.
