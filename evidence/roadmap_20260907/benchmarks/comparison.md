# Frozen local Neral comparison

Model-only performance. EOS recognition with supplied answers is not free generation.

| Stage | EOS % | Repetition % | UTF8 % | RawMath % | v3 % | v4 % | BPB ↓ | NIS | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline20 | 28.000 | 2.766 | 100.000 | 0.000 | 46.154 | 61.364 | 3.771 | 44.062 | baseline |
| b1 | 0.000 | 4.352 | 98.000 | 0.000 | 46.154 | 61.364 | 2.323 | 37.646 | 1/9 wins; False |
| termination_probe | 1.000 | 8.004 | 100.000 | 0.000 | 50.000 | 61.364 | 2.449 | 41.208 | 2/9 wins; False |
