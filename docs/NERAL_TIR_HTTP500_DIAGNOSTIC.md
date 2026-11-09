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


## Root-cause candidate identified in the Ollama Qwen3.5 parser

NERAL's **old** custom TIR protocol requested JSON nested inside Ollama's
**reserved** native XML delimiter:

```text
<tool_call>{"tool":"python_sandbox","code":"print(17*13)"}</tool_call>
```

This is a protocol collision. Ollama's Qwen3.5 parser reads the same
`<tool_call>` tag as a native Qwen-Coder XML tool invocation. Its native
parser expects nested `<function=...>` and `<parameter=...>` XML markup,
**not JSON between those tags**. NERAL didn't register any native Ollama
`tools`, so the tool request wasn't a supported native call either.

Ollama upstream issue
[ollama/ollama#14986](https://github.com/ollama/ollama/issues/14986)
documents this exact tag+JSON conflict on `qwen3.5:9b`, including an
Ollama HTTP 500 with an `EOF` parsing error even without registered native
tools. Issue [ollama/ollama#19187](https://github.com/ollama/ollama/issues/19187)
documents silent empty responses when the native tool parser discards
tool-like generated text. These bugs account for both *types* of responses
observed in NERAL smoke #42662197770 (one Ollama HTTP 500 and three empty
HTTP 200), but the actual original 500 response body was not retained, so
this diagnosis is a **strong source-supported hypothesis**, not a confirmed
post-fix runtime outcome.

### Specific repair

The experimental TIR prompt and strict application parser now use a
NERAL-specific non-XML envelope:

```text
NERAL_EXECUTE_JSON: {"tool":"python_sandbox","code":"print(17*13)"}
```

No reserved native Qwen/Ollama tags appear in the prompt or in the
one-step repair instruction. Legacy XML output is *rejected* by the parser,
not executed. All tool allowlists, bounded-code limits and Docker isolation
remain unchanged. The Ollama endpoint, `think:false`, sampling parameters,
model weights, `num_ctx` and `keep_alive` are unchanged.

The new CPU regression asserts the outgoing TIR prompt contains no
reserved `<tool_call>` markers, the strict decoder accepts the new
NERAL-only JSON envelope, and unsafe extra keys/tools or legacy XML
never execute. This is a **code-path fix**, not yet proof that real
`qwen3.5:9b` now produces a completed response on the RTX 4070.

### Promotion gate

Before any full development benchmark, the manual **three-task** smoke must
confirm `http_503 == 0`, `tool_calls_triggered > 0`, at least one
successful real Python sandbox execution, and inspect the new `eval_count`
and `ollama_error_type` metadata. If there is a new 500 error, its
full raw content is retained locally **on the user's PC only**, never
printed to public GitHub logs. Do not automatically launch GPU workflows
while preparing or merging this PR.


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
