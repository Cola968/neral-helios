# Neral / Helios 350M – local browser chat

This is an optional **real local chat interface**, not a hosted or released
model. It cannot run native 350M inference without the user's own compatible
checkpoint, SHA-locked tokenizer and an available CUDA BF16 GPU.

## Installation and start

Install the repository's normal model dependencies and optional UI:

```bash
python -m pip install "gradio>=5,<7"
```

From the repository root, for the real local Neral-350M checkpoint:

```bash
python frontier350/chat_web350.py --checkpoint /trusted/path/model.pt --tokenizer /trusted/path/tokenizer.json
```

Open http://127.0.0.1:9681 on the PC. This loads only the supplied local
weights and tokenizer and uses the same history serialization and generation
implementation as `chat_cli350.py`. The checkpoint is *not* fetched from
GitHub, Hugging Face or another provider.

UI preview without any model weights or GPU:

```bash
python frontier350/chat_web350.py --tool-only
```

The preview deliberately reports that no model is loaded. It can answer only
a limited set of deterministic calculations; it is **not** evidence of chat
quality or native model intelligence.

## Access from a phone on the same private Wi-Fi

```bash
python frontier350/chat_web350.py --checkpoint /trusted/path/model.pt --tokenizer /trusted/path/tokenizer.json --lan
```

Set a password of at least 12 characters at the hidden prompt. Open
`http://<PC-LAN-IP>:9681` on the phone and log in as `neral`. Use only a
trusted local network: plain HTTP is **not encrypted**, and this mode must
not be exposed to the public internet. No Gradio share link is permitted.

## Measurement and safety boundaries

- Source labels visibly distinguish `Neral 350M (nativ)`, deterministic
  tool-only responses and unavailable-model messages.
- The frozen native 30-case capability benchmark must **never** go through
  this UI's tool layer. Neither tool-only nor UI tests measure model quality.
- The 1,200-step candidate from GitHub run 37965826440 remains rejected:
  lower answer NLL did not improve the frozen native capability score.
- Session history is in memory. No model checkpoints, tokens or user chats
  are intentionally uploaded or written to disk by the application.
- No automatic model promotion, external share links, GPU retraining, or
  checkpoint writes are performed by this interface.
