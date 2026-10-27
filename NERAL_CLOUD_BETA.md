# NERAL Cloud Beta – real model answers without your gaming PC

**Status:** Implemented and tested as a standalone CPU/cloud-ready **private-beta**
chat server. **Not published, not trained, and not proven to be a reliable public
AI chatbot yet.** No inference API token is currently configured in this repo.

## What this is

NERAL is an independent assistant product and front end. Cloud inference uses
the upstream **Qwen/Qwen3.5-9B** open-weight model, served by Hugging Face
Inference Providers. The model was originally trained by Qwen; NERAL does not
claim to have pretrained Qwen. Our from-scratch **Arctus 350M** project remains
untouched. Model license: Apache-2.0; preserve NOTICE/license and attribution
as applicable when distributing derived weights. Verify any additional asset,
training-data and provider license constraints before release.

Because Hugging Face hosts inference, the PC and local Ollama can be offline.
Cloud users never see the provider API key; the cloud gateway holds it server-side.

**There is currently no public deployment, token, live-cloud benchmark,
SLA or streaming inference.** A 9B model needs to be provisioned somewhere:
this CPU gateway does NOT run the model itself.

## Local CPU test, zero cloud cost

```powershell
python -m unittest tests.test_neral_cloud -v
python -m neral_cloud.server
```

Visit http://127.0.0.1:8765. The status notice explains that real chat is
disabled without explicit provider configuration. No misleading fake answers.

## Cloud provider and release gates

For an **authorized small private beta only** (on Render/Replit in protected
server-side environment variables; never commit or paste tokens to code):

- `NERAL_HF_TOKEN`: fine-grained Hugging Face token that has
  **Make calls to Inference Providers** permission.
- `NERAL_BETA_ACCESS_KEY`: randomly generated private invite phrase, minimum
  16 characters, shared only with approved beta testers.
- `NERAL_CLOUD_INFERENCE_ENABLED=1`: separate, intentional switch that allows
  real paid inference requests; **defaults to disabled**.
- `PORT`: hosting provider automatically assigns port (default 8765).

Actual provider route is fixed at
`https://router.huggingface.co/v1/chat/completions`; model identity is fixed
to `Qwen/Qwen3.5-9B`. Supported model/providers may change. HF credit/billing
is NOT covered by hosting provider fees; observe the current cost/limits at
https://huggingface.co/docs/inference-providers/en/pricing .
Do not enable pay-as-you-go or automatic recharge unless you understand it and
explicitly approve the cost.

The app checks an invite secret on every request and uses a bounded model
response budget (256 output tokens), input caps, per-process IP rate limits,
a shared **60-request-per-process allowance**, and a maximum of **two simultaneous
provider requests**, fixed URL, redirects disabled, proxy bypass, and no
shell/file/network tools granted to the model. Every attempted provider call
counts toward the allowance, even if it fails. These are deliberately
conservative beta defaults and cannot be overridden by a browser user. Logs intentionally omit chat contents, token and invites.

**Important**: the in-memory rate limit AND total-call allowance are per-process,
reset on restart, and are NOT globally enforceable, persistent cost ceilings. For broader release, enable
provider-side spend ceilings, external abuse controls, authenticated accounts,
central rate limiting (e.g. Redis), safe audit logs with explicit privacy
policy, TOS, GDPR review, moderation, security audit and monitoring. Do not
present beta as production-safe or guarantee factual correctness.

## Render deployment plan

The root `render.yaml` is a **proposal only** (Frankfurt, free-tier CPU
gateway, auto-deploy off, inference switch 0, secrets unfilled). It is **not
automatically deployed** by committing the file. Review the current Render
plan availability, private repository authorization, workspace, quotas and
pricing before provisioning the resource. A Render CPU instance cannot load
Qwen3.5-9B into RAM/VRAM at this tier; it only proxies allowed requests.
No Postgres or Redis is provisioned yet; neither is necessary for this first
stateless beta.

## Web chat functions

- Desktop and mobile responsive NERAL-designed interface
- Keyboard Enter to send, Shift+Enter for a new line
- Multi-turn history within current browser tab (not server-stored)
- Explicit beta access modal, clear service status and truthful model lineage
- Output rendered with textContent (no untrusted HTML)
- Provider failures, rate-limit errors and missing keys handled without fake
  answers or leaking secrets
- No cloud training and no automatic local GPU use

## Verification and publication requirements

1. Offline unit and real HTTP localhost tests green (including invalid
   credentials, roles, oversized prompts, provider errors and origin checks).
2. Run a **real** authorized HF cloud-inference smoke using a connected token
   only after operator approval. Check actual response, user experience,
   upstream compatibility, latency, billing and quota, and log sanitization.
3. Run frozen native model evaluation on non-trivial German/English,
   factuality, coding, multi-turn context, hallucination calibration and
   safety tasks. Report model-native and tool-assisted scores separately.
4. Independent accessibility/security tests, account isolation, data protection
   review and abuse/cost protection before any public URL/X product launch.
5. Release notes must accurately say **“NERAL – built on Qwen3.5-9B”**.
   An actual **NERAL fine-tuned model** can be announced only after creating,
   licensing and evaluating our own adapter/derived weights.

## Sources

- Model: https://huggingface.co/Qwen/Qwen3.5-9B
- Inference API: https://huggingface.co/docs/inference-providers/en/index
- Pricing: https://huggingface.co/docs/inference-providers/en/pricing
