# Neral 350M: post-training data-selection research (2026-10-10)

Status: research only; no dataset has been admitted and no training run is authorized by this note.

The 1200-step SFT pilot lowered validation answer NLL (3.163 to 1.736) but decreased the frozen 30-case capability score (3 to 2). Keep this candidate rejected. Lower NLL is not evidence of better instruction following.

Potential evaluation hypotheses:
- Review prompt/response format parity and sampling mix before more updates.
- Compare category-level performance on held-out instruction following, German/English, code, context and calibration.
- Consider short, high-quality dialogues suitable for sub-1B models; do not blindly scale examples.
- Reject sources without independently checked license, provenance and contamination audit.

References (methodology only, not an approval to import data):
- https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk (small-model-oriented SFT subset; inspect component-level provenance)
- https://huggingface.co/datasets/OpenAssistant/oasst2 (human conversation trees; inspect role, quality and language filters)
- https://arxiv.org/abs/2503.11441 (diversity, difficulty and dependability in instruction data selection)

Known implementation issue to resolve: frontier350/chat350_knowledge.py currently tokenizes the full passage and only afterwards truncates the displayed excerpt to 2400 characters. Tokenization must use the displayed excerpt to avoid unsupported matches. Regression test: a query that only matches text beyond character 2400 must not return a hit.

No weights, checkpoints, or training artifacts should be promoted without a non-regression result on frozen capability tests and replay gates.
