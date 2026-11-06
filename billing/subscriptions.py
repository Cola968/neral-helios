"""Server-authoritative, offline-testable subscription state for NERAL.

No network calls, HTTP routes, Stripe SDK or checkout actions. Only a
correctly signed raw Stripe webhook may change subscription state.
The binding of a customer ID to an authenticated user MUST happen in a
separate server-verified Checkout flow, which is not implemented here.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Mapping

from billing.entitlements import entitlements_for

MAX_WEBHOOK_BYTES = 65536
SIGNATURE_TOLERANCE_SECONDS = 300
_ALLOWED_EVENT_TYPES = {
    'customer.subscription.created', 'customer.subscription.updated',
    'customer.subscription.deleted', 'checkout.session.completed',
    'invoice.paid', 'invoice.payment_failed',
}
_VALID_STATUSES = {
    'active', 'trialing', 'past_due', 'unpaid', 'incomplete',
    'incomplete_expired', 'canceled', 'paused',
}


class BillingError(ValueError):
    """Invalid or untrusted billing input. Never includes secret values."""


def _identifier(value: object, prefix: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(re.escape(prefix) + r'[A-Za-z0-9_-]{3,128}', value):
        raise BillingError('Invalid provider identifier')
    return value


def _user_id(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{4,128}', value):
        raise BillingError('Invalid internal user ID')
    return value


def verify_stripe_signature(payload: bytes, header: str, secret: str, *, now: int | None = None) -> dict:
    """Verify Stripe's signed timestamp and exact raw body before parsing JSON."""
    if not isinstance(payload, bytes) or len(payload) > MAX_WEBHOOK_BYTES:
        raise BillingError('Webhook body invalid or too large')
    if not isinstance(secret, str) or len(secret) < 16 or not isinstance(header, str) or len(header) > 2048:
        raise BillingError('Webhook signature configuration invalid')
    fields: dict[str, list[str]] = {}
    for component in header.split(','):
        key, sep, value = component.strip().partition('=')
        if not sep:
            raise BillingError('Malformed webhook signature')
        fields.setdefault(key, []).append(value)
    if len(fields.get('t', [])) != 1 or not fields.get('v1'):
        raise BillingError('Missing webhook signature')
    try:
        timestamp = int(fields['t'][0])
    except (ValueError, TypeError) as exc:
        raise BillingError('Invalid webhook signature timestamp') from exc
    current = int(time.time()) if now is None else now
    if type(current) is not int or timestamp <= 0 or abs(current - timestamp) > SIGNATURE_TOLERANCE_SECONDS:
        raise BillingError('Webhook signature timestamp expired')
    expected = hmac.new(secret.encode('utf-8'), f'{timestamp}.'.encode()+payload, hashlib.sha256).hexdigest()
    if not any(len(candidate) == 64 and hmac.compare_digest(candidate, expected) for candidate in fields['v1']):
        raise BillingError('Webhook signature mismatch')
    try:
        event = json.loads(payload)
    except (UnicodeError, ValueError) as exc:
        raise BillingError('Invalid webhook JSON') from exc
    if not isinstance(event, dict):
        raise BillingError('Webhook JSON must be an object')
    return event


class SubscriptionStore:
    """SQLite entitlement ledger; fails closed on ambiguous subscription state.

    A webhook is not a fully ordered billing event stream. Same-timestamp
    collisions deny access pending provider reconciliation; this module is
    NOT an independently deployable live billing integration.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS customers ('
                       'user_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL UNIQUE)')
            db.execute('CREATE TABLE IF NOT EXISTS subscriptions ('
                       'user_id TEXT PRIMARY KEY REFERENCES customers(user_id), '
                       'subscription_id TEXT NOT NULL UNIQUE, '
                       'plan TEXT NOT NULL, status TEXT NOT NULL, '
                       'period_end INTEGER NOT NULL, cancel_at_period_end INTEGER NOT NULL, '
                       'last_event_created INTEGER NOT NULL, last_event_id TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS processed_events ('
                       'event_id TEXT PRIMARY KEY, event_type TEXT NOT NULL)')

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute('PRAGMA foreign_keys=ON')
        return db

    def bind_customer(self, user_id: str, customer_id: str) -> None:
        """Trusted, server-confirmed checkout only. Never expose as a client API."""
        user_id = _user_id(user_id)
        customer_id = _identifier(customer_id, 'cus_')
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            previous = db.execute('SELECT customer_id FROM customers WHERE user_id=?',
                                  (user_id,)).fetchone()
            if previous is not None and previous[0] != customer_id:
                raise BillingError('Customer reassignment requires manual review')
            if previous is None:
                owner = db.execute('SELECT user_id FROM customers WHERE customer_id=?',
                                   (customer_id,)).fetchone()
                if owner:
                    raise BillingError('Customer already bound to a different account')
                db.execute('INSERT INTO customers VALUES (?,?)', (user_id, customer_id))

    def resolve(self, user_id: str, *, now: int | None = None) -> dict:
        """Never infer an entitlement from a caller-supplied plan or price ID."""
        user_id = _user_id(user_id)
        current = int(time.time()) if now is None else now
        if type(current) is not int:
            raise BillingError('Invalid lookup timestamp')
        with self._connect() as db:
            row = db.execute('SELECT plan,status,period_end,cancel_at_period_end '
                             'FROM subscriptions WHERE user_id=?', (user_id,)).fetchone()
        valid = bool(row and row[1] == 'active' and row[2] > current and row[0] in {'plus', 'pro'})
        plan = row[0] if valid else 'free'
        return {
            'plan': plan,
            'status': row[1] if row else 'none',
            'current_period_end': row[2] if row else None,
            'cancel_at_period_end': bool(row[3]) if row else False,
            'entitlements': entitlements_for(plan),
        }

    def apply_stripe_webhook(self, payload: bytes, signature_header: str,
                             signing_secret: str, price_to_plan: Mapping[str, str],
                             *, now: int | None = None,
                             expected_livemode: bool = False) -> str:
        """Verify then atomically apply; never rely on checkout-return URLs."""
        event = verify_stripe_signature(payload, signature_header, signing_secret, now=now)
        if (type(expected_livemode) is not bool or type(event.get('livemode')) is not bool
                or event['livemode'] is not expected_livemode):
            raise BillingError('Webhook billing environment mismatch')
        event_id = _identifier(event.get('id'), 'evt_')
        kind = event.get('type')
        created = event.get('created')
        if not isinstance(kind, str) or type(created) is not int or created <= 0:
            raise BillingError('Invalid webhook envelope')
        if not isinstance(price_to_plan, Mapping) or any(
            not isinstance(p, str) or not p.startswith('price_') or plan not in {'plus', 'pro'}
            for p, plan in price_to_plan.items()
        ):
            raise BillingError('Invalid server-side price map')
        obj = event.get('data', {}).get('object') if isinstance(event.get('data'), dict) else None
        if kind not in _ALLOWED_EVENT_TYPES:
            return 'ignored'
        if not isinstance(obj, dict):
            raise BillingError('Invalid event object')

        # Neither checkout success nor invoice.paid alone grants entitlements.
        if kind in {'checkout.session.completed', 'invoice.paid'}:
            with self._connect() as db:
                db.execute('BEGIN IMMEDIATE')
                if db.execute('SELECT 1 FROM processed_events WHERE event_id=?', (event_id,)).fetchone():
                    return 'duplicate'
                db.execute('INSERT INTO processed_events VALUES (?,?)', (event_id, kind))
            return 'ignored'

        customer = _identifier(obj.get('customer'), 'cus_')
        if kind == 'invoice.payment_failed':
            # Compatible with older and newer invoice subscription references.
            subscription = obj.get('subscription')
            if subscription is None and isinstance(obj.get('parent'), dict):
                details = obj['parent'].get('subscription_details')
                subscription = details.get('subscription') if isinstance(details, dict) else None
        else:
            subscription = obj.get('id')
        subscription = _identifier(subscription, 'sub_')

        if kind == 'invoice.payment_failed':
            plan, status, period_end, cancel_pending = 'free', 'past_due', 0, False
        else:
            status = 'canceled' if kind == 'customer.subscription.deleted' else obj.get('status')
            if status not in _VALID_STATUSES:
                status = 'needs_reconciliation'
            period_end = obj.get('current_period_end')
            if type(period_end) is not int or period_end <= 0:
                period_end = 0
                status = 'needs_reconciliation' if status == 'active' else status
            cancel_pending = obj.get('cancel_at_period_end')
            if type(cancel_pending) is not bool:
                cancel_pending = False
                status = 'needs_reconciliation'
            items = obj.get('items')
            prices = items.get('data') if isinstance(items, dict) else None
            price_id = None
            if isinstance(prices, list) and len(prices) == 1 and isinstance(prices[0], dict):
                price_obj = prices[0].get('price')
                if isinstance(price_obj, dict):
                    price_id = price_obj.get('id')
            plan = price_to_plan.get(price_id, 'free')
            if plan == 'free' and status == 'active':
                status = 'needs_reconciliation'

        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM processed_events WHERE event_id=?', (event_id,)).fetchone():
                return 'duplicate'
            owner = db.execute('SELECT user_id FROM customers WHERE customer_id=?',
                               (customer,)).fetchone()
            if not owner:
                raise BillingError('Unbound billing customer')
            user_id = owner[0]
            previous = db.execute('SELECT subscription_id,last_event_created FROM subscriptions '
                                  'WHERE user_id=?', (user_id,)).fetchone()
            if previous and previous[0] != subscription:
                raise BillingError('Subscription change requires verified reassignment')
            if kind == 'invoice.payment_failed' and not previous:
                db.execute('INSERT INTO processed_events VALUES (?,?)', (event_id, kind))
                return 'ignored'
            result = 'applied'
            if previous and created < previous[1]:
                result = 'stale'
            elif previous and created == previous[1]:
                result = 'needs_reconciliation'
                plan, status, period_end, cancel_pending = 'free', 'needs_reconciliation', 0, False
            if result != 'stale':
                db.execute('INSERT INTO subscriptions VALUES (?,?,?,?,?,?,?,?) '
                           'ON CONFLICT(user_id) DO UPDATE SET '
                           'plan=excluded.plan,status=excluded.status,period_end=excluded.period_end,'
                           'cancel_at_period_end=excluded.cancel_at_period_end,'
                           'last_event_created=excluded.last_event_created,last_event_id=excluded.last_event_id',
                           (user_id, subscription, plan, status, period_end, int(cancel_pending),
                            created, event_id))
            db.execute('INSERT INTO processed_events VALUES (?,?)', (event_id, kind))
        return result
