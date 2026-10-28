Inference-source snapshot from the same repository, **no checkpoint/tokenizer assets**:

- Repository: https://github.com/Cola968/neral-helios
- Commit: `105502d9be58517f87d8cfff5d9a5b0e286c22b3` (PR #140)
- `frontier350/model350.py` → `arctus350_core/model350.py`
- `frontier350/chat350_contract.py` → `arctus350_core/chat350_contract.py`
- `frontier350/chat_prompt350.py` → `arctus350_core/chat_prompt350.py`

These three files are copied without changes to model math, configuration or
prompt encoding. No experiment/training branch is merged. The standalone adapter
uses the scoped NumPy RNG allowlist from `secure_checkpoint350.py` at the same
commit, adapted to an already hashed open file descriptor. Its decoding follows
the Flask temperature/top-k contract, not the separate CLI's top-p/repetition
policy. Runtime/quality parity with that CLI is therefore **not asserted**.
