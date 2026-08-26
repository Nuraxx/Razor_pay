"""
Closes the recovery-confirmation loop for the legacy payment_failed path:

    Razorpay payment.captured (real, HMAC-verified, event-id-deduplicated)
        -> locate the matching PENDING recovery_outcomes row
        -> confirm it RECOVERED / PARTIALLY_RECOVERED
        -> audit trail
        -> dashboard sees it on its next refresh

This module contains NO webhook parsing, signature verification, or
idempotency logic of its own -- all three already exist, unchanged, in
app/main.py::razorpay_webhook (HMAC check, x-razorpay-event-id dedup, both
run BEFORE this module is ever reached) and
recovery/webhook_pipeline.py::process_raw_event (which routes an
already-stored, already-verified RawEvent to this module by event_type,
exactly the same way it already routes payment.failed events to
recovery/orchestrator.py -- one webhook architecture, not two).

AUTHORITATIVE-ONLY (brief section 2): the only thing that can ever move
recovery_status to RECOVERED/PARTIALLY_RECOVERED is a real Razorpay
payment.captured webhook. Nothing here ever infers recovery from a
scheduled retry, a sent communication, an LLM result, or a customer promise
-- recovery/orchestrator.py's own payment_action="retry_scheduled" is
recorded, never executed, precisely because none of those signals were ever
sufficient proof of money moving.

RECONCILIATION KEY (brief section 4): Razorpay assigns a NEW payment_id to
each Subscriptions retry attempt -- the payment_id that FAILED is essentially
never the same payment_id that later succeeds. So, strongest match first:
  1. payment_id: an exact match against the payment_id of an existing
     payment.failed RawEvent that still has a PENDING outcome (covers a
     two-step authorize->capture sequence, or a same-payment-id Test Mode
     simulation).
  1b. payment_id, one-time-payment domain: the same exact-payment_id
     matching principle as (1), but for a payment.failed that had no
     subscription_id and was routed through the generalized revenue-risk
     path instead (event_type="payment_failed_no_subscription" -- see
     recovery/webhook_pipeline.py). This domain has NO subscription_id to
     fall back to (tier 2 below never applies to it) -- a Payment Link /
     one-time payment's authoritative link to its later success is its
     payment_id alone, which is exactly why webhook_pipeline.py requires it
     as a precondition to enter this domain at all.
  2. subscription_id: the most recent still-PENDING outcome for a
     payment_failed case on that subscription -- subscription_id is a real,
     authoritative Razorpay identifier already used throughout this
     codebase (PolicyDecision.subscription_id, RecoveryEventInput.subscription_id),
     not a fuzzy/customer_ref-based guess.
Scope is legacy payment_failed cases and the one-time-payment revenue-risk
domain ONLY (RecoveryOutcome.event_type in {"payment_failed",
"payment_failed_no_subscription"}). The other 4 Track-03 domains have no
authoritative Razorpay payment linkage of this kind (see
policy/receivables_rules.py etc.) and correctly stay PENDING -- this module
never touches their rows.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.logging_config import log
from app.models import AuditLog, FailureEvent, RawEvent, RecoveryOutcome, RevenueRiskEvent

LEGACY_EVENT_TYPE = "payment_failed"
ONE_TIME_PAYMENT_EVENT_TYPE = "payment_failed_no_subscription"
CONFIRMED_BY_WEBHOOK = "webhook_confirmed"
STATUS_RECOVERED = "RECOVERED"
STATUS_PARTIALLY_RECOVERED = "PARTIALLY_RECOVERED"
_RESOLVED_STATUSES = frozenset({STATUS_RECOVERED, STATUS_PARTIALLY_RECOVERED})

OUTCOME_CONFIRMED = "payment_recovery_confirmed"
OUTCOME_ALREADY_CONFIRMED = "payment_confirmation_duplicate_ignored"
OUTCOME_NO_MATCH = "payment_confirmation_no_matching_case"


def _amount_in_rupees(raw_event: RawEvent) -> float:
    """Same paise->rupees convention as recovery/webhook_pipeline.py::_amount_in_rupees."""
    return (raw_event.amount or 0) / 100.0


def _find_pending_outcome(db: Session, raw_event: RawEvent) -> RecoveryOutcome | None:
    """Returns the PENDING RecoveryOutcome (legacy or one-time-payment
    domain) this payment.captured event should confirm, or None if no
    authoritative match exists. Never guesses -- see module docstring for
    the exact match-order tiers."""
    if raw_event.payment_id:
        original = (
            db.query(RawEvent)
            .filter(RawEvent.payment_id == raw_event.payment_id, RawEvent.event_type == "payment.failed")
            .order_by(RawEvent.id.desc())
            .first()
        )
        if original is not None:
            failure = db.query(FailureEvent).filter(FailureEvent.raw_event_id == original.id).first()
            if failure is not None:
                outcome = (
                    db.query(RecoveryOutcome)
                    .filter(RecoveryOutcome.event_id == failure.id, RecoveryOutcome.event_type == LEGACY_EVENT_TYPE)
                    .first()
                )
                if outcome is not None:
                    return outcome

        # Tier 1b: the one-time-payment revenue-risk domain (no
        # subscription_id, no FailureEvent row -- see module docstring).
        one_time_event = (
            db.query(RevenueRiskEvent)
            .filter(RevenueRiskEvent.event_type == ONE_TIME_PAYMENT_EVENT_TYPE, RevenueRiskEvent.external_id == raw_event.payment_id)
            .order_by(RevenueRiskEvent.id.desc())
            .first()
        )
        if one_time_event is not None:
            outcome = (
                db.query(RecoveryOutcome)
                .filter(RecoveryOutcome.event_id == one_time_event.id, RecoveryOutcome.event_type == ONE_TIME_PAYMENT_EVENT_TYPE)
                .first()
            )
            if outcome is not None:
                return outcome

    if raw_event.subscription_id:
        match = (
            db.query(RecoveryOutcome)
            .join(FailureEvent, RecoveryOutcome.event_id == FailureEvent.id)
            .join(RawEvent, FailureEvent.raw_event_id == RawEvent.id)
            .filter(
                RecoveryOutcome.event_type == LEGACY_EVENT_TYPE,
                RecoveryOutcome.recovery_status == "PENDING",
                RawEvent.subscription_id == raw_event.subscription_id,
            )
            .order_by(RecoveryOutcome.id.desc())
            .first()
        )
        if match is not None:
            return match

    return None


def confirm_payment_recovery(db: Session, raw_event: RawEvent) -> str:
    """Entry point for a verified payment.captured RawEvent -- called from
    recovery/webhook_pipeline.py::process_raw_event, which is itself only
    ever called after app/main.py's HMAC verification and event-id
    idempotency check already passed. Returns a short outcome string
    matching webhook_pipeline.py's own OUTCOME_* convention."""
    outcome = _find_pending_outcome(db, raw_event)

    if outcome is None:
        log.info(
            "payment.captured payment_id=%s subscription_id=%s matches no PENDING recovery case -- ordinary successful payment, nothing to reconcile",
            raw_event.payment_id, raw_event.subscription_id,
        )
        db.add(AuditLog(
            raw_event_id=raw_event.id, action=OUTCOME_NO_MATCH,
            reason=f"payment_id={raw_event.payment_id} subscription_id={raw_event.subscription_id}: no matching PENDING recovery_outcomes row",
            actor="system",
        ))
        db.commit()
        return OUTCOME_NO_MATCH

    if outcome.recovery_status in _RESOLVED_STATUSES:
        # Idempotency for: (a) the exact same webhook redelivered (already
        # short-circuited earlier by app/main.py's event-id check, but this
        # is a second, independent backstop), and (b) a genuinely different
        # success event referring to the same already-confirmed case.
        # Neither may ever re-apply, double-count, or overwrite the amount.
        log.info(
            "payment.captured payment_id=%s: recovery_outcomes.id=%s already %s -- not re-applied",
            raw_event.payment_id, outcome.id, outcome.recovery_status,
        )
        db.add(AuditLog(
            raw_event_id=raw_event.id, failure_event_id=outcome.event_id, action=OUTCOME_ALREADY_CONFIRMED,
            reason=f"recovery_outcomes.id={outcome.id} already {outcome.recovery_status} confirmed_payment_id={outcome.confirmed_payment_id}; payment_id={raw_event.payment_id} ignored",
            actor="system",
        ))
        db.commit()
        return OUTCOME_ALREADY_CONFIRMED

    previous_status = outcome.recovery_status
    confirmed_amount = _amount_in_rupees(raw_event)
    is_partial = confirmed_amount < outcome.at_risk_amount

    outcome.recovery_status = STATUS_PARTIALLY_RECOVERED if is_partial else STATUS_RECOVERED
    outcome.recovered_amount = confirmed_amount
    outcome.confirmed_by = CONFIRMED_BY_WEBHOOK
    outcome.confirmed_payment_id = raw_event.payment_id

    db.add(AuditLog(
        raw_event_id=raw_event.id, failure_event_id=outcome.event_id, action=OUTCOME_CONFIRMED,
        reason=(
            f"recovery_outcomes.id={outcome.id} payment_id={raw_event.payment_id} previous_status={previous_status} "
            f"new_status={outcome.recovery_status} recovered_amount={confirmed_amount} at_risk_amount={outcome.at_risk_amount} "
            f"confirmed_by={CONFIRMED_BY_WEBHOOK}"
        ),
        actor="system",
    ))
    db.commit()

    log.info(
        "Confirmed payment recovery: recovery_outcomes.id=%s payment_id=%s status=%s->%s recovered_amount=%s",
        outcome.id, raw_event.payment_id, previous_status, outcome.recovery_status, confirmed_amount,
    )
    return OUTCOME_CONFIRMED
