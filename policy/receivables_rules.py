"""
Track-03: B2B receivables chaser -- deterministic escalation policy. Pure,
versioned, dict-driven (mirrors classification/rules.py's style exactly).
`escalation_level` is decided ONLY here -- the LLM never sees or influences
it, only receives it as an already-decided fact to write copy around (same
pattern as recovery/orchestrator.py passing `will_retry`/`retry_window_description`
to the outreach job today).
"""
from __future__ import annotations

from dataclasses import dataclass

from policy.decision_engine import NO_ACTION

RULE_VERSION = "receivables-v1"

BUCKET_DUE_SOON = "due_soon"
BUCKET_OVERDUE_SOFT = "overdue_soft"
BUCKET_OVERDUE_MEDIUM = "overdue_medium"
BUCKET_OVERDUE_HIGH = "overdue_high"
BUCKET_DISPUTED = "disputed"
BUCKET_PROMISE_TO_PAY = "promise_to_pay"

CANDIDATE_FRIENDLY_REMINDER = "friendly_reminder"
CANDIDATE_PAYMENT_REQUEST = "payment_request"
CANDIDATE_PROMISE_TO_PAY_REQUEST = "promise_to_pay_request"
CANDIDATE_ESCALATION = "escalation"
CANDIDATE_HUMAN_HANDOFF = "human_handoff"

# day-overdue thresholds
_OVERDUE_SOFT_MAX_DAYS = 7
_OVERDUE_MEDIUM_MAX_DAYS = 30


def classify_receivable(*, days_overdue: int, is_disputed: bool = False, has_active_promise: bool = False) -> str:
    """`is_disputed`/`has_active_promise` outrank the day-threshold ladder --
    a disputed invoice or one already under an active promise is never just
    "overdue_high" by day count alone."""
    if is_disputed:
        return BUCKET_DISPUTED
    if has_active_promise:
        return BUCKET_PROMISE_TO_PAY
    if days_overdue < 0:
        return BUCKET_DUE_SOON
    if days_overdue <= _OVERDUE_SOFT_MAX_DAYS:
        return BUCKET_OVERDUE_SOFT
    if days_overdue <= _OVERDUE_MEDIUM_MAX_DAYS:
        return BUCKET_OVERDUE_MEDIUM
    return BUCKET_OVERDUE_HIGH


@dataclass(frozen=True)
class ReceivableDecision:
    bucket: str
    candidate_type: str
    escalation_level: int  # 0 (none) .. 4 (human handoff) -- decided ONLY here
    requires_human_review: bool
    reason: str
    rule_version: str = RULE_VERSION


# (candidate_type, escalation_level, requires_human_review, reason) per bucket.
# "medium overdue -> reminder + promise-to-pay" (brief example) is captured
# as CANDIDATE_PROMISE_TO_PAY_REQUEST -- the copy generated for it explicitly
# includes both the reminder framing and the promise-to-pay ask.
_BUCKET_POLICY: dict[str, tuple[str, int, bool, str]] = {
    BUCKET_DUE_SOON: (NO_ACTION, 0, False, "not_yet_due_no_action"),
    BUCKET_OVERDUE_SOFT: (CANDIDATE_FRIENDLY_REMINDER, 1, False, "soft_overdue_friendly_reminder"),
    BUCKET_OVERDUE_MEDIUM: (CANDIDATE_PROMISE_TO_PAY_REQUEST, 2, False, "medium_overdue_reminder_plus_promise_to_pay"),
    BUCKET_OVERDUE_HIGH: (CANDIDATE_ESCALATION, 3, False, "high_overdue_escalation"),
    BUCKET_DISPUTED: (CANDIDATE_HUMAN_HANDOFF, 4, True, "disputed_requires_human_review"),
    BUCKET_PROMISE_TO_PAY: (NO_ACTION, 1, False, "active_promise_already_in_place_no_new_action"),
}


def decide_receivable_action(*, days_overdue: int, is_disputed: bool = False, has_active_promise: bool = False) -> ReceivableDecision:
    bucket = classify_receivable(days_overdue=days_overdue, is_disputed=is_disputed, has_active_promise=has_active_promise)
    candidate_type, escalation_level, requires_human_review, reason = _BUCKET_POLICY[bucket]
    return ReceivableDecision(bucket, candidate_type, escalation_level, requires_human_review, reason)
