# NERAL 7B: local conversations in the existing Helios UI

## Purpose

The main chat still uses legacy NERAL 20M/50M checkpoints. The existing
model-directed agent already has a local backend: Qwen2.5-Coder-7B plus
an experimental trained NERAL LoRA adapter when local assets are present,
or local Ollama as an explicitly labelled foundation fallback.
This feature adds a separate direct conversational UI inside the Agent
panel. It reuses the backend instance: no second 7B model is loaded.

## Local setup and provenance

Use the Windows Helios application on the PC and open its local web UI.
By default the NERAL_AGENT_BACKEND environment setting is auto. It
chooses an existing local adapter when available; otherwise the fallback
is local Ollama at http://127.0.0.1:11434/api/chat with proxy and
redirect disabled. There is no remote Qwen or hosted inference fallback.

Explicit adapter selection:
- NERAL_AGENT_BACKEND=hf_adapter
- NERAL_AGENT_BASE_PATH=<local verified model snapshot>
- NERAL_AGENT_ADAPTER_PATH=<local sealed PEFT adapter>

The UI explicitly distinguishes adapter configured but not loaded,
actually loaded NERAL LoRA adapter, and local foundation without
a verified adapter. Adapter SHA-256 is shown only after a confirmed
successful local load. A Qwen-based LoRA still uses Qwen foundation
weights, and a parameter ratio is not a measured quality improvement.

## Privacy and limitations

Existing Helios session, Origin and Host checks are inherited. The
browser retains history only in this tab's memory and deletes it on
reset or reload. The server checks role ordering, rejects client
system/tool roles, and bounds prompt and history bytes. No local
prompts or generated content is uploaded to GitHub.

Ollama keeps JSON-format output for tool actions, but now returns
ordinary prose for chat. HF and Ollama inference are serialized so
the shared GPU is not loaded into overlapping generations.

This is intentionally non-streaming. A browser connection abort does
not guarantee server-side model cancellation yet. This feature
does not claim independent quality gains or multiuser readiness.

## Regression verification

The Helios app CI now runs HTTP session/Origin tests, input limits,
provenance tests, local Ollama response-mode tests and shared GPU
serialization. It also checks JavaScript syntax.

Commands:
    python -m pytest -q tests/test_neral_local_chat_api.py
    node --check static/neral-local-chat.js
