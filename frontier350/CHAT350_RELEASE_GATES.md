# Arctus 350M release gates

A model is never promoted because a single loss or chat sample looks good.

Required machine evidence before manual release review:
1. Data contract and provenance pass.
2. NeralTalk quality gate passes.
3. SFT produced a selected validation improvement, not merely a latest checkpoint.
4. Frozen 30-case capability score has no overall regression or hard category regression.
5. SFT val NLL improves overall and no eligible language/domain/source/turn slice regresses by more than 10%.
6. Generation health does not regress: EOS, repetition, UTF-8 and exact instruction contracts.
7. Foundation replay loss stays inside the post-training guard.
8. If DPO is used, human-ranked preference margin improves and DPO replay regression stays <=3%.
9. External held-out benchmarks remain separate from training. GSM8K and MBPP are planned evaluation-only sources.
10. Manual qualitative review is mandatory.

The release checker can only emit manual_release_review_required or reject. It cannot auto-release a model.

Local chat runtime:
- frontier350/chat_cli350.py
- exact locked tokenizer and checkpoint identity checks
- same User:/Assistant: serialization used by SFT
- token-budgeted recent-history retention
- EOS stopping
- greedy or temperature/top-p decoding
- repetition penalty
- no network calls and no conversation logging by default

The current generator intentionally does not claim KV-cache acceleration. Correctness and format consistency come before inference optimization.
