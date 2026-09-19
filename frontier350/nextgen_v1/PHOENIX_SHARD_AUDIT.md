# Phoenix shard-distribution audit (read-only; 19 September 2026)

The old native trainer selected each `.bin` shard uniformly, then chose a random start inside the chosen shard. A shard with one window and another with 100 windows therefore gave the short shard 50% of draws instead of 1/101. `window_sampler350.WindowSampler` now draws uniformly from all valid windows. **Actual historical-corpus bias remains unmeasured**: the old draw distribution can be calculated from the audited shard lengths, but past draws cannot be reconstructed.

Run only on the owner's computer with the real, approved-for-review shard paths and exact training sequence length:

```bash
python frontier350/nextgen_v1/phoenix_shard_distribution_audit.py --train-data PATH_TO_TRAIN_BIN_DIRECTORY --valid-data PATH_TO_VALID_BIN_DIRECTORY --seq 512
```

This is a full, streaming, read-only scan of every selected `.bin` file. One pass calculates SHA-256 and checks every little-endian uint16 token against the own Frontier 350 vocabulary (`0..16383`; 16,384 tokens). It refuses missing roots, symlinks, odd bytes, shards shorter than `seq+1` tokens, out-of-range token IDs, duplicate full-file hashes within/across splits, invalid digests and files whose size/inode/mtime/ctime differ before, while or immediately after the read. Chunk size is 4 MiB. Exact number of valid autoregressive starts in each shard is `tokens - seq`.

Output is aggregate-only: counts and distributions, old sampler bias factors and total-variation distance, plus `all_token_ids_in_vocab=true` and `file_metadata_stable_during_read=true` **only after the scan succeeds**. It does not print raw content, private file paths, token IDs or shard digests. Nonzero mathematical sampling bias does not establish actual model-quality damage or improvement.

**Limitations:** The file-metadata check is not an immutable filesystem snapshot: an adversarial rewrite restoring metadata, or a change after audit, can escape detection. Full-file hashes do not detect a repeated passage, overlapping token subsequence, paraphrase, semantic contamination or leakage into a separately authored holdout. The audit does not establish ownership/licensing/privacy, proof of human data review, successful real trainer execution, GPU memory fit, training throughput or better answers. This stand-alone diagnostic does not alter the pilot trainer or authorize a run; the operator must separately freeze and independently approve all actual data.

Local synthetic regression (no real training corpus or weights):

```bash
cd frontier350/nextgen_v1
python -m py_compile phoenix_shard_distribution_audit.py test_phoenix_shard_distribution_audit.py
python -m unittest -v test_phoenix_shard_distribution_audit.py
```

Eleven synthetic tests passed locally on 19 September 2026: distribution math, duplicate shards, odd bytes, too-short shards, out-of-vocabulary rejection, highest valid token, strict sequence/SHA checks, simulated metadata divergence and actual isolated CLI outputs. GitHub-hosted CI is **not green**: jobs failed without a runner assignment, steps or retrievable logs; root cause remains unknown. Keep both PRs draft, issue #119 open, and original model unchanged.
