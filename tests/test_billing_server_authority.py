"""Synthetic signed billing tests. No Stripe API calls or live credentials."""
import hashlib
import hmac
import json
import pytest

from billing.subscriptions import BillingError, SubscriptionStore, verify_stripe_signature

NOW = 1800000000
SECRET = 'whsec_local_test_not_a_real_secret'
PRICES = {'price_plus_test': 'plus', 'price_pro_test': 'pro'}


def event(kind='customer.subscription.updated', *, id='evt_test_001', created=NOW,
          customer='cus_test_001', subscription='sub_test_001', status='active',
          price='price_plus_test', period_end=NOW+3600, livemode=False, **extra):
    obj = {
        'id': subscription, 'customer': customer, 'status': status,
        'current_period_end': period_end, 'cancel_at_period_end': False,
        'items': {'data': [{'price': {'id': price}}]}, **extra,
    }
    if kind.startswith('invoice.'):
        obj = {'id': 'in_test_001', 'customer': customer, 'subscription': subscription}
    if kind.startswith('checkout.'):
        obj = {'id': 'cs_test_001', 'customer': customer}
    return {'id': id, 'type': kind, 'created': created, 'livemode': livemode,
            'data': {'object': obj}}


def signed(e, secret=SECRET, t=NOW):
    payload = json.dumps(e, separators=(',', ':')).encode()
    digest = hmac.new(secret.encode(), str(t).encode()+b'.'+payload, hashlib.sha256).hexdigest()
    return payload, f't={t},v1={digest}'


@pytest.fixture
def store(tmp_path):
    s = SubscriptionStore(tmp_path/'billing.sqlite3')
    s.bind_customer('user_alice', 'cus_test_001')
    s.bind_customer('user_bob', 'cus_test_002')
    return s


def apply(store, e, *, expected_livemode=False, signature=None, payload=None):
    p, h = signed(e)
    return store.apply_stripe_webhook(payload if payload is not None else p,
                                      signature if signature is not None else h,
                                      SECRET, PRICES, now=NOW,
                                      expected_livemode=expected_livemode)


def test_unsigned_or_tampered_webhook_never_grants(store):
    p, _ = signed(event())
    with pytest.raises(BillingError):
        apply(store, event(), signature='t=1800000000,v1=bad')
    with pytest.raises(BillingError):
        apply(store, event(), payload=p+b' ')
    assert store.resolve('user_alice', now=NOW)['plan'] == 'free'


def test_authenticated_subscription_unlocks_only_bound_customer(store):
    assert apply(store, event()) == 'applied'
    assert store.resolve('user_alice', now=NOW)['plan'] == 'plus'
    assert store.resolve('user_alice', now=NOW)['entitlements']['offline_chat_unlimited'] is True
    assert store.resolve('user_bob', now=NOW)['plan'] == 'free'
    assert store.resolve('user_alice', now=NOW+3600)['plan'] == 'free'
    assert apply(store, event()) == 'duplicate'


def test_cancel_at_period_end_preserves_access_until_end(store):
    apply(store, event())
    apply(store, event(id='evt_test_002', created=NOW+10, cancel_at_period_end=True))
    assert store.resolve('user_alice', now=NOW+100)['plan'] == 'plus'
    assert store.resolve('user_alice', now=NOW+3600)['plan'] == 'free'
    apply(store, event(kind='customer.subscription.deleted', id='evt_test_003', created=NOW+20,
                       status='canceled'))
    assert store.resolve('user_alice', now=NOW+200)['plan'] == 'free'


def test_unknown_price_fails_closed_and_wrong_customer_is_not_bound(store):
    apply(store, event())
    apply(store, event(id='evt_test_002', created=NOW+20, price='price_fake_pro'))
    assert store.resolve('user_alice', now=NOW+100)['plan'] == 'free'
    with pytest.raises(BillingError):
        apply(store, event(id='evt_test_003', customer='cus_unbound', subscription='sub_fake'))
    assert store.resolve('user_bob', now=NOW)['plan'] == 'free'


def test_no_client_supplied_plan_can_grant_paid(store):
    e = event()
    e['data']['object']['metadata'] = {'user_id': 'user_bob', 'plan': 'pro'}
    apply(store, e)
    assert store.resolve('user_bob', now=NOW)['plan'] == 'free'
    assert store.resolve('user_alice', now=NOW)['plan'] == 'plus'


def test_checkout_and_invoice_paid_not_proof_of_subscription(store):
    for id, kind in [('evt_checkout', 'checkout.session.completed'),
                     ('evt_invoice', 'invoice.paid')]:
        assert apply(store, event(kind=kind, id=id)) == 'ignored'
    assert store.resolve('user_alice', now=NOW)['plan'] == 'free'


def test_failed_payment_blocks_and_later_authoritative_subscription_recovers(store):
    apply(store, event())
    assert apply(store, event(kind='invoice.payment_failed', id='evt_failed', created=NOW+5)) == 'applied'
    assert store.resolve('user_alice', now=NOW+6)['plan'] == 'free'
    assert apply(store, event(id='evt_back', created=NOW+10)) == 'applied'
    assert store.resolve('user_alice', now=NOW+11)['plan'] == 'plus'


def test_stale_events_do_not_regrant_and_conflicting_same_second_fails_closed(store):
    apply(store, event())
    apply(store, event(kind='customer.subscription.deleted', id='evt_delete', created=NOW+10,
                       status='canceled'))
    assert apply(store, event(id='evt_stale', created=NOW+5)) == 'stale'
    assert store.resolve('user_alice', now=NOW+11)['plan'] == 'free'
    assert apply(store, event(id='evt_same', created=NOW+10)) == 'needs_reconciliation'
    assert store.resolve('user_alice', now=NOW+11)['plan'] == 'free'


def test_wrong_subscription_must_not_take_over_existing_customer(store):
    apply(store, event())
    with pytest.raises(BillingError):
        apply(store, event(id='evt_second', created=NOW+11, subscription='sub_unexpected'))
    assert store.resolve('user_alice', now=NOW+11)['plan'] == 'plus'


def test_client_identity_binding_cannot_change_ownership(store):
    with pytest.raises(BillingError):
        store.bind_customer('user_alice', 'cus_attacker')
    with pytest.raises(BillingError):
        store.bind_customer('user_new', 'cus_test_001')


def test_webhook_signature_expiry_mode_and_missing_secret(store):
    p, h = signed(event())
    with pytest.raises(BillingError):
        verify_stripe_signature(p, h, '', now=NOW)
    with pytest.raises(BillingError):
        verify_stripe_signature(p, h, SECRET, now=NOW+500)
    with pytest.raises(BillingError):
        apply(store, event(), expected_livemode=True)
    with pytest.raises(BillingError):
        verify_stripe_signature(b'x' * 65537, h, SECRET, now=NOW)


def test_invalid_event_does_not_allow_paid_access(store):
    apply(store, event())
    bad = event(id='evt_bad', created=NOW+20)
    bad['data']['object']['items'] = {'data': [{'price': {'id': 'price_pro_test'}},
                                              {'price': {'id': 'price_plus_test'}}]}
    apply(store, bad)
    assert store.resolve('user_alice', now=NOW+20)['plan'] == 'free'
