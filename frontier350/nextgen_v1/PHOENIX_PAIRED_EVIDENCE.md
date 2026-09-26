# Phoenix paired evidence — exploratory only

This step consumes the **human-reported aggregate** produced by `phoenix_blind_ab_unblind.py`. It neither checks the actual model weights nor independently authenticates the holdout, grades, or grading process. It does not start or authorize training and never selects a winning checkpoint.

## Exact private command

Run on the owner's machine after freezing the private human grades and producing the scored JSON. The destination `NeralPrivateAssets/phoenix` must already exist. Never commit the scored report, human answers, blind packet, holdout, or the unblinding key.

```bash
python frontier350/nextgen_v1/phoenix_paired_evidence.py --execute --scored-report D:/NeralPrivateAssets/phoenix/phoenix_scored_pilot01.json --private-dir D:/NeralPrivateAssets/phoenix --tag pilot01
```

This creates `phoenix_paired_pilot01.json` exclusively; no existing output is overwritten. The terminal output includes only the item count and explicit `quality_improvement_proven=false` and `training_approval_granted=false`. Report identity is bound to the exact input bytes through SHA-256. SHA-256 is an integrity digest, **not** proof that reported scores or model hashes are authentic.

## Mathematical meaning

For paired questions, `candidate_only` means only the candidate was marked correct, `baseline_only` means only the original was marked correct. The observed paired accuracy difference is `(candidate_only - baseline_only) / n`. The two-sided **exact discordance/binomial (McNemar) p-value** conditions on the discordant questions; it is computed with exact integer binomial coefficients (not a potentially inaccurate large-sample approximation). Zero discordances yield p=1. This p-value is exploratory. It is not a probability that the candidate is better, and does not establish independent sampling, correct human grading, general intelligence or reproducibility. Multiple evaluated candidates / repeated holdout use invalidate naïve confirmatory interpretations.

The tool refuses malformed counts, impossible paired partitions, unexpected schema/provenance flags, non-original baseline checkpoint, equal candidate/baseline hashes, inconsistent German/English or domain subtotals, missing language or score identities. Even fully consistent input can be fabricated or contain biased judgments. The output therefore **always** marks independence, quality gain and approval false, whatever p-value is observed.

Synthetic tests, no user model/data:

```bash
cd frontier350/nextgen_v1
python -m py_compile phoenix_paired_evidence.py test_phoenix_paired_evidence.py
python -m unittest -v test_phoenix_paired_evidence.py
```

Seven synthetic tests passed locally during development, including a real create-only CLI invocation. The entire local A/B + evidence suite passed 26 tests. The hosted GitHub Actions environment remains unverified. No candidate checkpoint, real human-scored holdout, model-quality improvement or 60–70k tokens/s training result is implied.
