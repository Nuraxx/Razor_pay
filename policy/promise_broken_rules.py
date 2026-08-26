"""
Track-03: what to do when a promise-to-pay passes its date unfulfilled.
Pure, versioned rule module -- mirrors classification/rules.py's style.
Deliberately bounded, separately from (but respecting) the ORIGINAL event's
cumulative payment-retry attempts, so a broken promise can never be used to
bypass policy/guardrails.py::MAX_RETRY_ATTEMPTS.
"""
from __future__ import annotations

from dataclasses import dataclass

from policy.decision_engine import NO_ACTION

RULE_VERSION = "promise-broken-v1"

CANDIDATE_URGENT_REMINDER = "urgent_reminder"
CANDIDATE_FINAL_NOTICE = "final_notice"

# A broken promise gets at most this many further nudges, independent of
# (and smaller than) the underlying payment-retry cap.
MAX_BROKEN_PROMISE_ATTEMPTS = 2


@dataclass(frozen=True)
class PromiseBrokenDecision:
    candidate_type: str
    reason: str
    rule_version: str = RULE_VERSION


def decide_promise_broken_action(
    *, attempts_so_far: int, cumulative_payment_attempts: int, max_payment_attempts: int,
) -> PromiseBrokenDecision:
    """`cumulative_payment_attempts`/`max_payment_attempts` are the ORIGINAL
    event's own retry-attempt count/cap (policy/guardrails.py::MAX_RETRY_ATTEMPTS)
    -- checked FIRST, so a broken promise can never squeeze out one more
    attempt once the original event has already exhausted its own retry
    budget. `attempts_so_far` is this broken-promise sub-sequence's own,
    smaller, independent cap."""
    if cumulative_payment_attempts >= max_payment_attempts:
        return PromiseBrokenDecision(
            NO_ACTION, f"cumulative_payment_attempts_exhausted: {cumulative_payment_attempts} >= {max_payment_attempts}",
        )
    if attempts_so_far >= MAX_BROKEN_PROMISE_ATTEMPTS:
        return PromiseBrokenDecision(
            NO_ACTION, f"max_broken_promise_attempts_reached: {attempts_so_far} >= {MAX_BROKEN_PROMISE_ATTEMPTS}",
        )
    if attempts_so_far == 0:
        return PromiseBrokenDecision(CANDIDATE_URGENT_REMINDER, "first_broken_promise_urgent_reminder")
    return PromiseBrokenDecision(CANDIDATE_FINAL_NOTICE, "second_broken_promise_final_notice")
