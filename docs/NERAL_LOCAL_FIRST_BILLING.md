# NERAL: local-first model and data boundary

## What "my own AI" means technically

NERAL is the project and product. Qwen2.5-Coder-7B-Instruct is
**frozen, locally executed foundation weights**. The experimental NERAL
rank-32 LoRA adapter is trained locally and must be loaded and verified
before the inference identity may be called "NERAL + adapter".

A model using Qwen weights is **not** independently trained from scratch.
No arbitrary percentage such as "60% NERAL / 40% Qwen" can be established
from parameter counts, prompts, or a three-step pilot. Claim measured
improvement only from a frozen independent comparison, with transparent
baseline, adapter hash, passed cases and limitations.

## Private inference

In the reviewed rank32 comparison:
- Qwen base and tokenizer load from verified local directories with
  local_files_only=True and token=False.
- PEFT adapter loads from the local sealed directory.
- scripts/neral_local_policy.py enforces HF_HUB_OFFLINE=1,
  TRANSFORMERS_OFFLINE=1, HF_DATASETS_OFFLINE=1 and disables WANDB.
- The benchmark retains generated code/prompts on the local machine.
  The GitHub Actions artifact is a bounded scalar-only comparison summary.
- A provenance report distinguishes foundation-only from an actually
  loaded NERAL adapter and explicitly states the base is still Qwen.

**This does not promise that all optional plugins are offline.** Web search,
GitHub and other network tools must remain separately controlled, visible
and off by default for sensitive content. Their use sends data to their
respective providers only when explicitly authorized. A Python environment
flag cannot replace OS firewall enforcement or a real network egress audit.

Do not confuse "Qwen weights perform computations locally" with "no Qwen
weights are used". User text can be mathematically processed by a locally
hosted Qwen-based foundation without being transmitted to Alibaba/Qwen.

## Subscription architecture and user content

Subscription verification must be separate from inference and from user
content. A billing event needs only a pseudonymous internal user ID,
Stripe customer/subscription/price IDs and subscription state. **No prompts,
chat messages, projects, uploaded files or local model artifacts belong
in Stripe metadata, checkout URLs or webhook bodies created by NERAL.**

Current billing/subscriptions.py provides an internal SQLite subscription
ledger that:
- checks Stripe-style raw-body HMAC SHA-256 signatures and 5-minute window;
- atomically deduplicates event IDs;
- binds known customers to existing authenticated internal users;
- maps allowlisted provider price IDs to free/plus/pro on the server;
- refuses to grant access on a success redirect, checkout completion or
  invoice.paid event alone;
- denies on expired, unpaid, canceled, failed or ambiguous states;
- handles stale updates conservatively and marks conflicting same-second
  events for reconciliation instead of guessing;
- isolates sandbox versus live event mode.

Important: This is **not live billing**. It intentionally exposes no
checkout, portal, account-registration or HTTP webhook route. bind_customer
may only be called by a future trusted backend that first authenticates
the user and independently verifies Stripe customer/session ownership.
Stripe source-of-truth reconciliation is needed for out-of-order events,
upgrades, renewals and multiple subscriptions. Do not turn these helpers
into client-exposed endpoints.

## Future rollout gates

1. Finish real NERAL user accounts with server-side identity and ownership.
2. Implement authorized Checkout creation using internal plan names and
   server-only configured Stripe price IDs.
3. Verify and bind the Checkout customer to the authenticated user.
4. Build raw-body Stripe webhook endpoint and current-provider-state
   reconciliation, plus sandbox-only end-to-end tests.
5. Add Customer Portal and cancellation handling tied to the account.
6. Explicitly define allowed offline paid features and whether purchase
   verification needs a separately signed, expiring entitlement certificate.
7. Check app-store billing rules, legal requirements and data deletion.

None of these steps authorizes charging customers or transmitting prompt
data to a foundation-model provider.
