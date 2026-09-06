# Neral exact benchmark dashboard

This report compares **20M Preview 02** and **50M Pilot A** on the same frozen 100-prompt greedy benchmark. No synthetic intelligence score is used.

Suite hash identical: **True**. Generation settings identical: **True**.

## Exact overall comparison

| Metric | 20M Preview 02 | 50M Pilot A |
|---|---:|---:|
| Parameters | 20,587,008 | 50,012,976 |
| Valid UTF-8 | 100.00% | 100.00% |
| EOS completion | 29.00% | 0.00% |
| Repeated trigram rate ↓ | 2.95% | 76.62% |
| Non-repetition trigram score ↑ | 97.05% | 23.38% |
| Raw math exact | 0.00% | 0.00% |
| Manual comprehensibility | 45.00% | 25.00% |
| Decode speed | 139.23 tok/s | 91.24 tok/s |
| Mean latency | 0.37 s | 0.71 s |

## Derived deltas

- 50M decode speed is **65.53%** of 20M.
- 50M mean latency is **+90.74%** versus 20M.
- 50M repeated-trigram rate is **25.93×** the 20M rate.
- 50M has **2.43×** as many parameters.

## By domain

| Domain | N | 20M EOS | 50M EOS | 20M repetition ↓ | 50M repetition ↓ | 20M collapse ↓ | 50M collapse ↓ | 20M tok/s | 50M tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chat | 20 | 25.00% | 0.00% | 0.00% | 78.69% | 5.00% | 100.00% | 132.32 | 89.41 |
| coding | 10 | 10.00% | 0.00% | 0.00% | 78.66% | 20.00% | 100.00% | 139.66 | 79.48 |
| explanation | 40 | 25.00% | 0.00% | 4.32% | 71.17% | 15.00% | 92.50% | 142.26 | 96.04 |
| math | 10 | 100.00% | 0.00% | 0.00% | 76.43% | 0.00% | 100.00% | 131.22 | 88.75 |
| robustness | 20 | 15.00% | 0.00% | 6.14% | 84.51% | 20.00% | 100.00% | 143.85 | 90.60 |

## By language

| Language | N | 20M EOS | 50M EOS | 20M repetition ↓ | 50M repetition ↓ | 20M heuristic language match | 50M heuristic language match |
|---|---:|---:|---:|---:|---:|---:|---:|
| de | 50 | 34.00% | 0.00% | 0.00% | 87.86% | 44.00% | 60.00% |
| en | 50 | 24.00% | 0.00% | 5.91% | 65.37% | 28.00% | 42.00% |

## 50M training benchmark

- Selected training configuration: **seq 256 × batch 32 = 8192 tokens/update**.
- Selected measured throughput: **76,613.97 tokens/s**.
- Selected PyTorch peak allocated/reserved: **3.34/3.67 GiB**.
- Pilot A validation loss: **5.6234 → 2.3636**, a **57.97%** reduction.
- Pilot A trained on **999,424 tokens**.

## Collapse diagnostic

Collapse is flagged when repeated trigram fraction is >=25% **or** the longest identical-token run is >=8. This is a transparent diagnostic, not an intelligence score.
