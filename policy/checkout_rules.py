"""
Track-03: checkout drop-off recovery -- deterministic eligibility + candidate
selection. Pure, versioned, threshold-driven -- mirrors classification/rules.py's
style exactly (no DB, no LLM, no randomness). Deciding whether/how to nudge an
abandoned cart is a domain rule, never something the LLM decides; the LLM
(llm/service.py::generate_outreach_microcopy) only ever writes the copy for
whatever candidate this module already selected.
"""
from __future__ import annotations

from dataclasses import dataclass

from policy.decision_engine import NO_ACTION

RULE_VERSION = "checkout-v1"

# -- deterministic eligibility thresholds (brief: "configurable rules") -----
MIN_CART_AMOUNT_RS = 199.0
MIN_INACTIVITY_MINUTES = 15.0  # below this: too early to call anything "stalled"
STALL_TO_ABANDON_MINUTES = 60.0  # below this (but >= MIN_INACTIVITY_MINUTES): stalled, not yet abandoned
MAX_OUTREACH_ATTEMPTS = 3  # no duplicate/unbounded outreach

CHECKOUT_STARTED = "CHECKOUT_STARTED"
CHECKOUT_STALLED = "CHECKOUT_STALLED"
ABANDONED = "ABANDONED"

CANDIDATE_REMINDER = "reminder"
CANDIDATE_PAYMENT_LINK_REMINDER = "payment_link_reminder"
CANDIDATE_RETRY_CHECKOUT = "retry_checkout"
CANDIDATE_ALTERNATE_PAYMENT_METHOD = "alternate_payment_method"
CANDIDATE_WAIT = "wait"

CHECKOUT_CANDIDATE_TYPES = frozenset({
    CANDIDATE_REMINDER, CANDIDATE_PAYMENT_LINK_REMINDER, CANDIDATE_RETRY_CHECKOUT,
    CANDIDATE_ALTERNATE_PAYMENT_METHOD, CANDIDATE_WAIT, NO_ACTION,
})


@dataclass(frozen=True)
class CheckoutDecision:
    state: str
    recovery_eligible: bool
    eligibility_reason: str
    candidate_type: str
    rule_version: str = RULE_VERSION


def decide_checkout_recovery(
    *, cart_amount: float, inactivity_minutes: float,
    previous_outreach_count: int = 0, payment_method: str | None = None,
) -> CheckoutDecision:
    """Deterministic state machine (brief: CHECKOUT_STARTED -> CHECKOUT_STALLED
    -> ABANDONED -> RECOVERY_ELIGIBLE/EXPIRED). RECOVERED/EXPIRED are terminal
    states set by the caller once an outcome is observed (recovery/revenue_orchestrator.py
    / recovery/demo_generator.py) -- this function only ever decides the initial
    routing, never a terminal outcome it can't actually observe."""
    if cart_amount < MIN_CART_AMOUNT_RS:
        return CheckoutDecision(
            CHECKOUT_STARTED, False,
            f"cart_amount_below_minimum: {cart_amount} < {MIN_CART_AMOUNT_RS}", NO_ACTION,
        )
    if inactivity_minutes < MIN_INACTIVITY_MINUTES:
        return CheckoutDecision(
            CHECKOUT_STARTED, False,
            f"inactivity_below_minimum: {inactivity_minutes} < {MIN_INACTIVITY_MINUTES}", NO_ACTION,
        )
    if inactivity_minutes < STALL_TO_ABANDON_MINUTES:
        return CheckoutDecision(
            CHECKOUT_STALLED, False,
            f"stalled_not_yet_abandoned: {inactivity_minutes} < {STALL_TO_ABANDON_MINUTES}", CANDIDATE_WAIT,
        )
    if previous_outreach_count >= MAX_OUTREACH_ATTEMPTS:
        return CheckoutDecision(
            ABANDONED, False,
            f"max_outreach_attempts_reached: {previous_outreach_count} >= {MAX_OUTREACH_ATTEMPTS}", NO_ACTION,
        )

    # No duplicate outreach: exactly one candidate per previous_outreach_count value.
    if previous_outreach_count == 0:
        candidate = CANDIDATE_REMINDER
    elif previous_outreach_count == 1:
        candidate = CANDIDATE_ALTERNATE_PAYMENT_METHOD if payment_method else CANDIDATE_PAYMENT_LINK_REMINDER
    else:
        candidate = CANDIDATE_RETRY_CHECKOUT

    return CheckoutDecision(ABANDONED, True, "eligible_for_recovery_outreach", candidate)
