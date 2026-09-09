# Stripe integration contract

## Scope

This is the implementation contract for Helios subscription billing. It intentionally contains no live Stripe credentials and does not create charges by itself.

## Recommended Stripe products

- Stripe Billing for recurring subscriptions
- Stripe-hosted Checkout for subscription signup
- Stripe Customer Portal for self-service subscription management
- Webhooks for authoritative lifecycle synchronization

Use separate Stripe Products for `Helios Plus` and `Helios Pro`. Monthly and annual variants of the same tier should be separate Stripe Prices attached to that tier's Product.

## Server responsibilities

The server owns all Stripe interaction and authorization decisions.

### Checkout

A checkout-session endpoint should:

1. Require an authenticated Helios user.
2. Accept an internal plan ID such as `plus`, never an arbitrary client-provided Stripe Price ID.
3. Resolve the configured Stripe Price ID server-side.
4. Create a Checkout Session in subscription mode.
5. Attach a stable internal user identifier as metadata where appropriate.
6. Return only the checkout URL/session information required by the client.

Do not hard-code `payment_method_types`; let Stripe use eligible dynamic payment methods.

### Webhooks

The webhook endpoint must verify the Stripe signature using `STRIPE_WEBHOOK_SECRET` before reading the event as trusted input.

The initial event set should cover at least:

- `checkout.session.completed`
- `customer.subscription.created`
- `customer.subscription.updated`
- `customer.subscription.deleted`
- `invoice.paid`
- `invoice.payment_failed`

Webhook processing must be idempotent. Store processed Stripe event IDs or otherwise make repeated delivery safe.

Do not make `checkout.session.completed` alone the source of truth for ongoing entitlement. Persist customer/subscription identifiers and derive current subscription state from verified lifecycle events and, when necessary, a server-side Stripe read.

### Entitlements

Stripe object IDs are infrastructure identifiers, not authorization concepts. Normalize subscription state into an internal record resembling:

```json
{
  "user_id": "internal-user-id",
  "plan": "plus",
  "status": "active",
  "current_period_end": "timestamp",
  "stripe_customer_id": "cus_...",
  "stripe_subscription_id": "sub_..."
}
```

The feature layer resolves `plan` against `billing/plans.json` and must fail closed for unknown plans or invalid state.

Suggested states that grant paid access: `active` and, if product policy chooses, `trialing`. Define explicit behavior for `past_due`, `unpaid`, `canceled`, and incomplete subscriptions before launch.

### Customer Portal

Create portal sessions on the server for the authenticated user's Stripe Customer ID. Do not let the client choose another customer ID.

Recommended default cancellation behavior: cancel at period end, preserving access until the paid period expires.

## Security

- Keep Stripe keys in deployment secrets, never repository files or browser JavaScript.
- Prefer a restricted Stripe API key with only required access.
- Never log full secret keys or webhook secrets.
- Validate plan identifiers against an allowlist.
- Protect checkout and portal endpoints against cross-account access.
- Treat redirects as UX only, not payment proof.
- Use TLS in any non-local deployment.
- Store the minimum Stripe/customer data necessary for billing state.

## Offline entitlement design

Because Helios is expected to support offline functionality, a paid user may eventually need a short-lived signed entitlement issued by the Helios backend. It should contain only the minimum authorization claims (for example plan, expiry and user/device binding), have a bounded expiry, and be verifiable locally without embedding a signing secret in the client.

This is deliberately not implemented yet. A server-authoritative subscription state must exist first.

## Stripe Tax

Do not enable automatic tax merely because Stripe Tax is available. Before enabling it, verify the correct product tax classification and active tax registrations for the jurisdictions where collection is required. Sandbox registrations do not carry into live mode. Legal/tax registration decisions require appropriate tax advice.

## App-store billing boundary

Web Stripe billing and mobile store billing are separate distribution concerns. Before a Google Play or App Store release, determine which purchase flows are permitted for digital subscription access in the target storefront and region. The internal entitlement layer should support multiple billing providers so app-store billing can map to the same `free` / `plus` / `pro` authorization model without rewriting product features.

## Next implementation milestone

1. Create sandbox Stripe Products/Prices after pricing is finalized.
2. Add a persistent user/subscription store.
3. Implement authenticated Checkout endpoint.
4. Implement verified webhook endpoint and idempotency.
5. Implement entitlement resolver.
6. Add Customer Portal endpoint.
7. Add automated billing tests using Stripe sandbox/test events.
8. Only then connect the billing UI.
