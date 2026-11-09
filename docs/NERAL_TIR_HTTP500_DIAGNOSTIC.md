# NERAL Qwen3.5 HTTP 500 / empty-content wire diagnosis (CPU-only change)

## Motivation
Manual C-only three-case GPU smoke #42662197770 returned three application-level
HTTP 503 errors. On the wire there was one Ollama HTTP 500 and three Ollama
HTTP 200 replies with `message.content` length zero,
`message.thinking` absent and `done_reason=stop`. It is unknown whether
these stops occurred after zero tokens or after other tokens: the previous
collector did not record `eval_count`.

## Confirmed outgoing JSON shape (source inspection)
Both raw A and TIR C use exactly the same `OllamaLocalBackend.respond_chat`
with `POST http://127.0.0.1:11434/api/chat` and:

```json
{
  "model": "qwen3.5:9b",
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "think": false,
  "stream": false,
  "options": {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repeat_penalty": 1.0,
    "num_predict": 2048,
    "num_ctx": 4096
  },
  "keep_alive": "5m"
}
```

There is **no** `options.stop`, top-level `stop`, `tools`, `format`
or assistant-prefill in either request. A sends the NERAL identity prompt
and the question with a terminal `FINAL_ANSWER` output-format instruction.
C appends the experimental tool-routing instruction to the system message
and removes only that exact terminal output-format instruction from the
user question. Roles remain `system`, `user` (two messages).

The source comparison does not prove Ollama is at fault; token generation,
runtime initialization, an Ollama error and a compatibility problem are
still possible.

## New local diagnostic capture
`scripts/neral_ollama_wire_probe.py` temporarily intercepts *only* the
manual three-case test's calls to loopback `/api/chat`. Its upstream
still uses the exact pre-existing proxy-free, redirect-denying opener.
It passes the identical request bytes through, buffers the returned
response bytes and replays them unmodified to NERAL.

It prints **only** safe wire metadata:
* message count, roles and number of empty messages; presence of stop,
  format, native tools or assistant prefill;
* response HTTP status, `done_reason`, `eval_count`, `prompt_eval_count`,
  `message.tool_calls` presence/count, and content/thinking character counts;
* for non-200 responses, bounded error JSON key names, safe reason class,
  exact error character count and SHA-256 of the bounded response body.

The **full Ollama HTTP error body** can contain private prompts. It is never
printed or uploaded. If Ollama returns an HTTP error, the next authorized
three-case smoke stores its exact raw body only on the user's Windows PC under:

```
%USERPROFILE%\\NERAL-QLoRA-Research\\private-ollama-diagnostics\\smoke3-ollama-error-<RUN_ID>-<ATTEMPT>-<TIME>.json
```

Do not copy this file into public GitHub issues or logs without inspecting
and removing personal information. No GPU inference is run by the PR CI.

## Acceptance
The PR runs mocked CPU-only tests. The three-case GPU job remains
`workflow_dispatch`-only, gated on a successful exact-SHA release receipt
and explicit authorization. A successful CPU job is not evidence that
the model or TIR tool routing is fixed. Never run Hard36 before the
three-case live smoke shows no failures and real successful tool calls.
