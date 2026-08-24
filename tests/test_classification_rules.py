"""
classification/rules.py tested in isolation — pure function, no DB.
"""
from classification.rules import (
    CUSTOMER_CANCELLED,
    HARD_DECLINE,
    RETRYABLE_SOFT,
    RULE_VERSION,
    UNMAPPED,
    classify,
)


def test_insufficient_fund_classifies_retryable_soft_with_full_confidence():
    result = classify(
        error_code="BAD_REQUEST_ERROR",
        error_reason="insufficient_fund",
        error_source="customer",
        error_step="payment_authorization",
    )
    assert result.bucket == RETRYABLE_SOFT
    assert result.confidence == 1.0
    assert result.rule_version == RULE_VERSION


def test_insufficient_funds_plural_spreadsheet_spelling_also_retryable_soft():
    """Razorpay's own official error-reasons spreadsheet spells this reason
    in the plural; both spellings must map to the same bucket."""
    result = classify(error_code="BAD_REQUEST_ERROR", error_reason="insufficient_funds")
    assert result.bucket == RETRYABLE_SOFT
    assert result.confidence == 1.0


def test_verified_hard_decline_reason_classifies_hard_decline():
    """card_expired: Razorpay docs describe this as permanent for the instrument."""
    result = classify(
        error_code="BAD_REQUEST_ERROR",
        error_reason="card_expired",
        error_source="customer",
        error_step="payment_authorization",
    )
    assert result.bucket == HARD_DECLINE
    assert result.confidence == 1.0
    assert result.rule_version == RULE_VERSION


def test_another_verified_hard_decline_reason_debit_instrument_blocked():
    result = classify(error_code="GATEWAY_ERROR", error_reason="debit_instrument_blocked")
    assert result.bucket == HARD_DECLINE
    assert result.confidence == 1.0


def test_verified_cancellation_reason_classifies_customer_cancelled():
    result = classify(
        error_code="BAD_REQUEST_ERROR",
        error_reason="payment_cancelled",
        error_source="customer",
        error_step="payment_authentication",
    )
    assert result.bucket == CUSTOMER_CANCELLED
    assert result.confidence == 1.0
    assert result.rule_version == RULE_VERSION


def test_unknown_reason_classifies_unmapped():
    result = classify(error_code="BAD_REQUEST_ERROR", error_reason="some_reason_that_does_not_exist")
    assert result.bucket == UNMAPPED
    assert result.confidence == 0.0
    assert result.rule_version == RULE_VERSION


def test_plausible_but_unverified_reason_is_never_guessed():
    """A reason that sounds like it should map somewhere but isn't in the
    verified table must still be unmapped -- this classifier never guesses."""
    result = classify(error_code="BAD_REQUEST_ERROR", error_reason="payment_declined")
    assert result.bucket == UNMAPPED
    assert result.confidence == 0.0


def test_missing_error_reason_is_unmapped_not_a_crash():
    result = classify(error_code=None, error_reason=None, error_source=None, error_step=None)
    assert result.bucket == UNMAPPED
    assert result.confidence == 0.0


def test_empty_string_error_reason_is_unmapped():
    result = classify(error_code="BAD_REQUEST_ERROR", error_reason="   ")
    assert result.bucket == UNMAPPED
    assert result.confidence == 0.0


def test_missing_error_code_and_source_and_step_does_not_prevent_matching_on_reason():
    """Only error_reason is required to match a verified rule -- the other
    three fields are context, not required keys."""
    result = classify(error_code=None, error_reason="insufficient_fund", error_source=None, error_step=None)
    assert result.bucket == RETRYABLE_SOFT
    assert result.confidence == 1.0


def test_confidence_is_binary_matched_or_unmapped():
    matched = classify(error_code="X", error_reason="card_expired")
    unmapped = classify(error_code="X", error_reason="totally_unverified_reason")
    assert matched.confidence == 1.0
    assert unmapped.confidence == 0.0
