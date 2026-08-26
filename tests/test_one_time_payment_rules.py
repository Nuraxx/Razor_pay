"""
Unit tests for policy/one_time_payment_rules.py -- pure, no DB. Mirrors
tests/test_checkout_recovery.py's style for the other Track-03 domains.
"""
from policy.decision_engine import NO_ACTION
from policy.one_time_payment_rules import (
    CANDIDATE_PAYMENT_LINK_REMINDER,
    decide_one_time_payment_recovery,
)


def test_retryable_soft_bucket_gets_payment_link_reminder():
    result = decide_one_time_payment_recovery(error_code="BAD_REQUEST_ERROR", error_reason="insufficient_fund")
    assert result.classification_bucket == "retryable_soft"
    assert result.recovery_eligible is True
    assert result.candidate_type == CANDIDATE_PAYMENT_LINK_REMINDER


def test_hard_decline_bucket_also_gets_payment_link_reminder_never_a_fake_retry():
    """Unlike the subscription path (which forces hard_decline to NO_ACTION
    because a genuine automatic retry exists for retryable_soft but never
    for hard_decline), this domain has NO live automatic retry either way --
    a payment-link reminder is equally truthful for both buckets."""
    result = decide_one_time_payment_recovery(error_code="GATEWAY_ERROR", error_reason="card_declined")
    assert result.classification_bucket == "hard_decline"
    assert result.recovery_eligible is True
    assert result.candidate_type == CANDIDATE_PAYMENT_LINK_REMINDER


def test_customer_cancelled_bucket_takes_no_action():
    result = decide_one_time_payment_recovery(error_code=None, error_reason="payment_cancelled")
    assert result.classification_bucket == "customer_cancelled"
    assert result.recovery_eligible is False
    assert result.candidate_type == NO_ACTION


def test_unmapped_reason_takes_no_action_never_guesses():
    result = decide_one_time_payment_recovery(error_code="X", error_reason="some_reason_not_in_the_table")
    assert result.classification_bucket == "unmapped"
    assert result.recovery_eligible is False
    assert result.candidate_type == NO_ACTION


def test_missing_error_reason_is_unmapped_never_guessed():
    result = decide_one_time_payment_recovery(error_code=None, error_reason=None)
    assert result.classification_bucket == "unmapped"
    assert result.candidate_type == NO_ACTION


def test_reuses_the_exact_same_classifier_as_the_subscription_path():
    """The whole point (brief item 4): no subscription-specific classifier,
    no domain-specific bucket vocabulary -- the exact same
    classification/rules.py::classify function and bucket names."""
    from classification.rules import classify

    direct = classify("BAD_REQUEST_ERROR", "insufficient_fund", "customer", "payment_authorization")
    via_rules_module = decide_one_time_payment_recovery(
        error_code="BAD_REQUEST_ERROR", error_reason="insufficient_fund", error_source="customer", error_step="payment_authorization",
    )
    assert via_rules_module.classification_bucket == direct.bucket
    assert via_rules_module.classification_confidence == direct.confidence
    assert via_rules_module.classification_reason == direct.reason


def test_result_is_deterministic():
    r1 = decide_one_time_payment_recovery(error_code="X", error_reason="insufficient_fund")
    r2 = decide_one_time_payment_recovery(error_code="X", error_reason="insufficient_fund")
    assert r1 == r2
