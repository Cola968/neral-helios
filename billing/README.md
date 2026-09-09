# Helios Billing

This directory defines the billing and entitlement boundary for Helios/Neral.

## Current status

Billing is **prepared, not live**. Stripe is currently connected only through the `Neral AI Sandbox` test account. No live products, prices, subscriptions, charges, or tax registrations are created by this repository setup.

## Planned commercial model

- `free` — no Stripe subscription required
- `plus` — target price: 9.99 EUR/month
- `pro` — target price: 19.99 EUR/month, introduced only when the product has genuinely higher-cost capabilities

Pricing values are product decisions, not hard-coded authorization rules. Stripe Price IDs must be provided through environment configuration.

## Architecture

Helios uses a server-side entitlement layer between Stripe and the app:

`Stripe Billing -> webhook sync -> subscription state -> entitlements -> Helios features`

The UI must never decide paid access from a Stripe Price ID directly.

Recommended Stripe flow:

1. Stripe Billing with one Stripe Product per paid tier.
2. Stripe-hosted Checkout for subscription signup.
3. Stripe Customer Portal for cancellation, payment-method updates, upgrades and downgrades where applicable.
4. Verified webhooks update local subscription state.
5. The entitlement resolver maps the local plan to feature flags.
6. All Stripe secret/restricted keys and webhook secrets stay server-side and outside Git.

## Files

- `plans.json` — product-level plan and entitlement definitions
- `.env.example` — safe environment-variable contract; contains no real credentials
- `STRIPE_INTEGRATION.md` — integration and security contract

## Security rules

- Never commit Stripe keys, webhook secrets, customer data, checkout session secrets, or subscription tokens.
- Prefer a Stripe restricted API key (`rk_...`) with only the required permissions.
- Verify webhook signatures before mutating subscription state.
- Treat webhook processing as idempotent.
- Do not grant access because a checkout page returned successfully; grant access from verified Stripe state.
- Keep `free`, `plus`, and `pro` as internal stable plan identifiers and map Stripe objects to them on the server.

## Mobile note

If Helios is distributed through mobile app stores, store-billing requirements must be evaluated separately before selling digital subscriptions inside the mobile app. Stripe support in the web product does not automatically imply that the same checkout flow can be used for every in-app purchase scenario.
