# Phoenix shard-distribution audit (read-only; 19 September 2026)

The old native trainer selected each `.bin` shard with equal probability, then a random valid starting window inside that shard. With one shard contributing 1 valid window and another 100 windows, the 1-window shard received 50% of training draws instead of 1/101. The updated `window_sampler350.WindowSampler` selects uniformly from all valid windows. **The magnitude of the bias for Neral's actual historic corpus has not been measured.** This utility makes the proposed new distribution auditable before any new run; it does not reconstruct actual historical draws.

On the owner's machine only, with the exact tokenizer/sequence length and shard directories chosen for an eventual experiment:

```bash
python frontier350/nextgen_v1/phoenix_shard_distribution_audit.py --train-data PATH_TO_TRAIN_BIN_DIRECTORY --valid-data PATH_TO_VALID_BIN_DIRECTORY --seq 512
```

The script rejects absent roots, symlinks, odd byte lengths, shards lacking at least one `seq+1` token window, identical shard hashes within a split, or identical shard hashes between training and validation. It streams every file to calculate SHA-256 and prints **aggregates only**: shard count, token/window counts, extremes, highest old-sampler overrepresentation, smallest relative sampling factor, and total-variation distance between old and corrected window selection. It does not print private paths, token text, or digests. Exact window count per shard is `tokens - seq` (input uses `seq` tokens and target requires the next token). A nonzero variation indicates different mathematical draw distributions, **not** evidence of changed historical model quality.

The results do **not** establish that the running historical trainer used the code one expects, actual per-shard draw frequencies, correct uint16 token IDs, unchanged bytes during training, near-duplicate/semantic train–validation overlap, data licensing/privacy, answer ability, throughput, GPU fit or any training approval. SHA checks can miss mutations occurring after the audit. The operator must inspect source rights and content independently and freeze the proposed shard set before any separately approved pilot. This utility never authorizes or starts training.

Synthetic tests (no real data):

```bash
cd frontier350/nextgen_v1
python -m unittest -v test_phoenix_shard_distribution_audit.py
```

The six tests cover the 1-vs-100 exact distribution, duplicate shard bytes across and within splits, odd byte lengths, too-short shards, and a real temporary-file CLI run with aggregate-only output. Hosted CI status must be confirmed separately; creating tests does not mean they ran on the user's machine. Keep PR #118 draft, pilot approval closed and issue #119 open.
