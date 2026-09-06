# Neral family model card

Own local decoder-only byte-level language models. No external learned weights, LoRAs or cloud inference.

| Model | Parameters | Status |
|---|---:|---|
| Neral Preview 02 | 20,587,008 | Current default; free language frequently incoherent |
| Neral 40M capacity experiment | 40,051,200 | Validation regressed; historical optional experiment |
| Neral 50M pilot A | 50,012,976 | New random initialization, 999,424 language-pretraining tokens; free-language gate failed |

Architecture, evaluation, performance, limitations and actual training: [PREVIEW_50M.md](PREVIEW_50M.md). Data: [DATA_PROVENANCE.md](DATA_PROVENANCE.md). Reproduction: [REPRODUCE.md](REPRODUCE.md).

## Limitations

Not reliable for general advice, coding, math or autonomous decisions. Local deterministic SOLVE/project tools are separate software capabilities, not model intelligence. The prompt guard is heuristic. New natural-language corpus is Wikipedia only, not enough for a general assistant. Historical 20M corpus provenance is incomplete. No commercial/public-readiness claim or new blanket model license is made.

## Artifacts

50M inference SHA-256: `2bf5216613af70602a5babe49dd298d82935b550eb81f6e5b4310e9798a64a8a`.

20M Preview 02 inference SHA-256: `8ea61ecc405257c73905383e922b32cfff9ccd00f13f5677a91ec938d3129c2f`.

Full optimizer/RNG checkpoints remain local. Both train and validation source hashes are recorded in run metadata.
