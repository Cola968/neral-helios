# NERAL v6: verified error replay and anti-forgetting guard

Status: CPU-only candidate data builder. No model call, GPU use, training run, automatic adapter activation, or release validation.

## Why

NERAL v5 improved format and response completion, but its measured difference versus Qwen with three few-shot examples on the 250 numeric A+B NERAL-EXT cases was only 3 correct answers. The newer v6 120-step adapter exists, but training loss alone does not establish unseen-task accuracy.

Evidence supporting replay as a potential anti-forgetting approach:
https://arxiv.org/abs/2603.09892
This project does not claim to replicate MSSR or to have demonstrated performance gains from its replay policy.

## Verified input contract

The module scripts/neral_v6_replay_guard.py rebuilds and compares:
- Exactly 728 canonical v6 train examples from the pinned generator.
- Exactly 1,920 canonical v5 train examples from the pinned generator.
- At least four real, parseable, train-only incorrect answers from the local E1 pair miner, matched back to deterministic v6 oracle answers.

Any wrong source split, duplicate identity, incorrect claimed target, unverifiable rejection, malformed record, or collision with a protected local evaluation prompt is rejected before output is written. These checks include normalized exact match and normalized ten-word overlap. They cannot exclude paraphrases or Qwen foundation pretraining contamination.

## Candidate mixture

Default initial recipe: around 50 percent clean v6 training examples, 30 percent original v5 replay, and 20 percent verified correction examples. The actual candidate size depends on the number of legitimate error pairs, which are never fabricated. Sampling is deterministic and group-balanced; private output is SHA-256 pinned and written create-only outside the repository.

The error correction row includes the incorrect attempt as context and the original exact oracle solution as supervised target; this is a proposed SFT curriculum, not a DPO dataset.

Run on a CPU-only local machine after independently reviewing the private pair records:

python -m scripts.neral_v6_replay_guard --verified-pairs C:/private/pairs.jsonl --protected-jsonl C:/private/dev.jsonl --protected-jsonl C:/private/release.jsonl --out C:/private/replay-new

The output includes candidate.jsonl and manifest.json, both private. Neither private evaluation items nor error source texts are uploaded to GitHub.

## Non-negotiable gates

1. Complete the Qwen+Few-Shot/v5/v6 fresh 72-case dev evaluation from PR #351 and report paired regressions as well as improvements.
2. Mine actual local model failures only from TRAIN cases. Do not equate a JSON field such as model_generated=true with independent provenance.
3. Verify correction targets independently and freeze all protected family/template-disjoint evaluation partitions.
4. Run a new LoRA experiment only after separate explicit GPU authorization, with both existing v5 and v6 adapters preserved.
5. Require an improvement over both Qwen and v6 on frozen unseen families, and prohibit substantial v5 skill regressions before any release promotion.

The current test fixtures contain invented incorrect responses to test rejection logic. They are not evidence about model capabilities.

CPU regression command:

python -m unittest discover -s tests -p test_neral_v6_replay_guard.py -v
