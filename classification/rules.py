"""
Day-2 deterministic failure classifier.

raw_events.error_reason -> bucket, matched only against reason strings that
are independently verified — never guessed. Two sources:

  1. This repo's own Day-1 fixture (tests/conftest.py) and this task's spec,
     which use `insufficient_fund` (singular) — the field name real Razorpay
     payment.entity webhook payloads carry.
  2. Razorpay's official "list of possible error reasons" spreadsheet,
     linked from https://razorpay.com/docs/payment-gateway/rainy-day/errors/error-reasons/
     at https://razorpay.com/docs/build/browser/assets/images/payments_error_reasons.xlsx
     (fetched 2026-08-24; 114 documented reasons with explanations), cross-checked
     against https://razorpay.com/docs/errors/payments/cards/ for card-specific wording.
     That spreadsheet spells the same concept as (1) `insufficient_funds` (plural) —
     both spellings are mapped to the same bucket below.

Every mapping in _REASON_RULES carries the verified Razorpay explanation it is
based on. A reason string that is not in this table — including one that
sounds plausible — is `unmapped`. That is the point: this layer never guesses.
"""
from dataclasses import dataclass

RULE_VERSION = "v1"

RETRYABLE_SOFT = "retryable_soft"
HARD_DECLINE = "hard_decline"
CUSTOMER_CANCELLED = "customer_cancelled"
UNMAPPED = "unmapped"

# error_reason (verbatim, as Razorpay sends it) -> (bucket, justification).
# Justification quotes/paraphrases Razorpay's own documented explanation for
# that reason — see module docstring for the source.
_REASON_RULES: dict[str, tuple[str, str]] = {
    # -- retryable_soft: the failure is transient; the same instrument can
    #    plausibly succeed if retried later. --
    "insufficient_fund": (
        RETRYABLE_SOFT,
        "Customer's account lacked funds at the time of the attempt; balance "
        "can change, so the same instrument may succeed on retry.",
    ),
    "insufficient_funds": (
        RETRYABLE_SOFT,
        "Same condition as insufficient_fund; Razorpay's official error-reasons "
        "list spells this reason in the plural.",
    ),
    "bank_technical_error": (
        RETRYABLE_SOFT,
        "Razorpay docs: issuing bank was facing technical problems at the time "
        "of the attempt — transient infra fault, not a decision about the card.",
    ),
    "server_error": (
        RETRYABLE_SOFT,
        "Razorpay docs: technical error at Razorpay's own server — transient.",
    ),
    "request_timed_out": (
        RETRYABLE_SOFT,
        "Razorpay docs: the request timed out — transient.",
    ),
    "payment_timed_out": (
        RETRYABLE_SOFT,
        "Razorpay docs: customer/gateway did not complete the transaction in "
        "time — transient, not an explicit decline.",
    ),

    # -- hard_decline: this specific instrument was refused; retrying it
    #    as-is will not help. --
    "card_expired": (
        HARD_DECLINE,
        "Razorpay docs: the card itself is expired — permanent for this instrument.",
    ),
    "debit_instrument_blocked": (
        HARD_DECLINE,
        "Razorpay docs: card is blocked by issuer or customer — permanent for this instrument.",
    ),
    "debit_instrument_inactive": (
        HARD_DECLINE,
        "Razorpay docs: card is inactive/frozen — permanent for this instrument.",
    ),
    "payment_risk_check_failed": (
        HARD_DECLINE,
        "Razorpay docs: declined by a fraud/risk check at Razorpay, gateway, or "
        "issuer — must not be retried blindly.",
    ),
    "card_declined": (
        HARD_DECLINE,
        "Razorpay docs: issuer bank explicitly declined the card after its own "
        "checks (exact reason not shared with Razorpay).",
    ),

    # -- customer_cancelled --
    "payment_cancelled": (
        CUSTOMER_CANCELLED,
        "Razorpay docs: customer explicitly cancelled the payment during authentication.",
    ),
}


@dataclass(frozen=True)
class ClassificationResult:
    bucket: str
    confidence: float
    rule_version: str
    reason: str


def classify(
    error_code: str | None,
    error_reason: str | None,
    error_source: str | None = None,
    error_step: str | None = None,
) -> ClassificationResult:
    """
    Deterministically classify one failure. Pure function, no I/O — the
    caller (classification/service.py) is responsible for reading raw_events
    and writing failure_events/audit_log.

    Matching is keyed on error_reason, the field Razorpay documents with the
    most specific, per-reason meaning. error_code/error_source/error_step
    are carried into the returned `reason` string for audit context, but are
    not used to override a missing or unverified error_reason — that would
    be guessing.
    """
    normalized_reason = (error_reason or "").strip()
    context = f"error_code={error_code!r} error_source={error_source!r} error_step={error_step!r}"

    if not normalized_reason:
        return ClassificationResult(
            bucket=UNMAPPED,
            confidence=0.0,
            rule_version=RULE_VERSION,
            reason=f"No usable error_reason ({context}); never guessed, classified unmapped.",
        )

    match = _REASON_RULES.get(normalized_reason)
    if match is None:
        return ClassificationResult(
            bucket=UNMAPPED,
            confidence=0.0,
            rule_version=RULE_VERSION,
            reason=(
                f"error_reason={normalized_reason!r} ({context}) is not in the verified "
                "rule table; never guessed, classified unmapped."
            ),
        )

    bucket, justification = match
    return ClassificationResult(
        bucket=bucket,
        confidence=1.0,
        rule_version=RULE_VERSION,
        reason=f"error_reason={normalized_reason!r} ({context}) matched verified rule: {justification}",
    )
