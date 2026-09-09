from billing.entitlements import entitlement_value, entitlements_for, has_entitlement, normalize_plan


def test_unknown_plan_falls_back_to_free():
    assert normalize_plan(None) == "free"
    assert normalize_plan("unknown") == "free"


def test_plus_unlocks_assistant_features():
    assert has_entitlement("plus", "offline_chat_unlimited")
    assert has_entitlement("plus", "lockscreen_assistant")
    assert has_entitlement("plus", "voice_assistant")


def test_free_does_not_unlock_paid_assistant_features():
    assert not has_entitlement("free", "offline_chat_unlimited")
    assert not has_entitlement("free", "lockscreen_assistant")
    assert not has_entitlement("free", "voice_assistant")


def test_multilevel_entitlements_are_not_booleanized():
    assert entitlement_value("free", "cascade") == "limited"
    assert entitlement_value("plus", "cascade") == "full"
    assert entitlement_value("pro", "priority_inference") == "highest"


def test_entitlement_dict_is_copied():
    first = entitlements_for("plus")
    first["chat"] = False
    assert entitlements_for("plus")["chat"] is True
