# Nova 1B — 2026 research basis

This file records the external research ideas that motivated Nova v1. It is not evidence
that Nova is better: every imported idea is treated as a hypothesis until a matched Neral
ablation succeeds.

## 1. Mamba-3 — recurrent backbone candidate

Paper: **Mamba-3: Improved Sequence Modeling using State Space Principles**
(arXiv:2603.15569, 2026).

Nova uses Mamba-3 because the design targets stronger recurrent sequence modeling while
retaining constant-size recurrent inference state. The official `state-spaces/mamba`
implementation is pinned to:

`e9594ce1c732d97440f0332fdc43170a2294dbfa`

Pinned upstream license: Apache-2.0.

Nova v1's Mamba settings deliberately use head dim 64, state 128, expand 2 and the MIMO
rank-4 research path. This is a code dependency only. No upstream model weights,
checkpoints, vocabularies, merges or training data are imported.

## 2. Gated DeltaNet-2 and modern linear attention

Paper: **Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention**
(arXiv:2605.22791, 2026).

This work reinforces the idea that recurrent/linear memory quality depends on how memory is
written and erased, not only on nominal O(n) scaling. Nova v1 does not stack GDN2 together
with Mamba-3: doing so would make causal attribution poor and increase implementation risk.
GDN2 remains a future mixer-control candidate if Mamba-3 fails the Pareto gate.

## 3. Linear-attention architecture/optimizer trade-offs

Paper: **Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing**
(arXiv:2607.07953, 2026).

The paper compares DeltaNet-family mixers, hybrid stacks, optimizers and routing schemes.
Two implications are used here:

1. hybrid/recurrent choices must be measured jointly on loss + throughput, not selected from
   architecture fashion;
2. lightweight cross-layer value routing can be worth testing, but must be isolated by an
   ablation.

Nova's Adaptive Memory Weave (AMW) follows the second engineering intuition while using a
Neral-specific low-rank gated bridge. It is **not** presented as a reproduction of CLVR and
is **not** claimed globally novel without a broader literature/patent search.

## 4. Exact attention is retained deliberately

Paper: **Rethinking the Role of Efficient Attention in Hybrid Architectures**
(arXiv:2606.15378, 2026).

The reported mechanism study suggests that long-range exact retrieval remains strongly tied
to full-attention layers while efficient layers shape the optimization trajectory. Nova
therefore keeps 8 exact full-attention anchors rather than becoming a pure recurrent stack.

That paper also motivates a controlled NoPE-on-exact-attention experiment. Nova exposes
`nova1b-m3-amw-nope`, but RoPE remains the default until the ablation is measured.

## 5. Full-bandwidth latent feedback — intentionally deferred

Paper: **Full-bandwidth transformer** (arXiv:2608.08888, 2026).

Latent feedback is a high-interest future track because the paper reports improved token
utilization and reasoning/code/math at 1B scale. It is not merged into Nova v1 because its
scheduled multi-pass training objective changes the optimization path substantially. The
first Nova experiment must answer one clean question: does the Mamba-3 + exact-attention +
AMW hybrid beat its matched controls?

If Nova v1 survives the first 25M/100M pilots, full-bandwidth-style latent feedback should
be evaluated in a **separate Nova v2 branch** rather than silently added to v1.

## 6. Optimizer research

Muon-family optimizers are a strong 2026 research candidate for training efficiency / final
loss, especially in matched linear-attention studies. Nova does not hard-code Muon into the
architecture. Optimizer choice is a separate training-recipe dimension:

- memory-feasible 8-bit AdamW control;
- Muon / memory-reduced Muon candidate only after numerical and VRAM validation;
- identical data, seed, tokens/update and evaluation samples for comparisons.

This separation prevents an optimizer gain from being misreported as an architecture gain.

## What counts as a Nova innovation

The project-level novelty hypothesis is **Adaptive Memory Weave (AMW)** combined with a
3-recurrent : 1-exact-attention cadence and dynamic gated exact-attention anchors, all sized
to land at ~1B parameters on the local Neral hardware target.

That is a new Neral design proposal, **not a claim that the individual primitives are new**.
A serious novelty claim would require a dedicated literature and patent search beyond this
engineering pass.
