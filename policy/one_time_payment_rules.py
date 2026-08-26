"""
Generalized live-pipeline extension: recovery for a `payment.failed` event
that has NO `subscription_id` but DOES carry enough authoritative payment
context (payment_id + amount) to act on -- e.g. a Razorpay Payment Link or
any other one-time-payment failure. Pure, versioned, no DB -- mirrors
policy/checkout_rules.py's style exactly.

Reuses the SAME deterministic classifier every subscription payment failure
already uses (classification/rules.py::classify) -- there is nothing
subscription-specific about mapping error_reason to a bucket; the classifier
itself never required a subscription_id in the first place.

What IS domain-specific is the candidate: a one-time payment has no live
automatic-retry mechanism the way a Razorpay Subscription does (Razorpay
never silently re-attempts a Payment Link), so the only truthful candidate
this module ever selects is a reminder-with-a-retry-option -- never an
"automatic retry" claim, for either a soft or a hard decline. `hard_decline`
is deliberately included in the eligible set for exactly this reason: unlike
the subscription path (where a real auto-retry exists for retryable_soft but
never for hard_decline, so the two buckets must be treated differently), this
domain's one and only candidate is equally truthful for both -- there is no
live retry either way, so "send a payment-link reminder" is not a stronger
claim for one bucket than the other. `customer_cancelled` (opted out) and
`unmapped` (nothing verified to act on) never get a candidate, matching the
subscription path's own guardrails.
"""
from __future__ import annotations

from dataclasses import dataclass

from classification.rules import HARD_DECLINE, RETRYABLE_SOFT, classify
from policy.decision_engine import NO_ACTION

RULE_VERSION = "one-time-payment-v1"

CANDIDATE_PAYMENT_LINK_REMINDER = "payment_link_reminder"

ONE_TIME_PAYMENT_CANDIDATE_TYPES = frozenset({CANDIDATE_PAYMENT_LINK_REMINDER, NO_ACTION})

# Buckets eligible for the reminder candidate -- see module docstring for why
# hard_decline is included here (never true for the subscription path).
_ELIGIBLE_BUCKETS = frozenset({RETRYABLE_SOFT, HARD_DECLINE})


@dataclass(frozen=True)
class OneTimePaymentDecision:
    classification_bucket: str
    classification_confidence: float
    classification_reason: str
    recovery_eligible: bool
    eligibility_reason: str
    candidate_type: str
    rule_version: str = RULE_VERSION


def decide_one_time_payment_recovery(
    *, error_code: str | None, error_reason: str | None,
    error_source: str | None = None, error_step: str | None = None,
) -> OneTimePaymentDecision:
    """Caller (policy/revenue_recovery_policy.py) is responsible for having
    already confirmed enough authoritative context exists to reach this
    domain at all (payment_id + amount present) -- see
    recovery/webhook_pipeline.py's own eligibility gate, which runs BEFORE a
    RevenueRiskEvent for this domain is ever created. This function only
    ever decides the classification-driven candidate, nothing else."""
    classification = classify(error_code, error_reason, error_source, error_step)

    if classification.bucket in _ELIGIBLE_BUCKETS:
        return OneTimePaymentDecision(
            classification_bucket=classification.bucket,
            classification_confidence=classification.confidence,
            classification_reason=classification.reason,
            recovery_eligible=True,
            eligibility_reason="eligible_for_payment_link_reminder",
            candidate_type=CANDIDATE_PAYMENT_LINK_REMINDER,
        )
    return OneTimePaymentDecision(
        classification_bucket=classification.bucket,
        classification_confidence=classification.confidence,
        classification_reason=classification.reason,
        recovery_eligible=False,
        eligibility_reason=f"bucket_not_eligible_for_recovery_action: {classification.bucket}",
        candidate_type=NO_ACTION,
    )
