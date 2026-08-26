"""
Track-03: the promise-to-pay LIFECYCLE dimension -- "did the customer
actually keep it" -- layered on top of policy/promise_to_pay.py's
validation-time status (VALID/LOW_CONFIDENCE/INVALID_DATE/EXPIRED/SUPERSEDED,
UNCHANGED). Writes ONLY to the new app.models.PromiseOutcome table;
recovery/promise_service.py and app.models.PromiseToPay are never touched by
this module -- `promises_to_pay.status` is intentionally left exactly as it
was validated, since PromiseOutcome is where the lifecycle fact belongs (see
that table's own docstring).

This module contains NO orchestration/communication logic of its own -- it
only detects a lifecycle transition and persists it, plus (for BROKEN) opens
a new RevenueRiskEvent so the recovery engine can decide what happens next.
Routing that new event through the real orchestrator is a separate step,
done by the CALLER (scripts/sweep_promise_lifecycle.py,
recovery/demo_generator.py) via recovery/revenue_orchestrator.py -- kept
separate so this module itself has no dependency on the orchestrator at all.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.logging_config import log
from app.models import AuditLog, FailureEvent, PolicyDecision, PromiseOutcome, PromiseToPay, RawEvent, RevenueRiskEvent
from policy.decision_engine import NO_ACTION
from policy.promise_to_pay import STATUS_VALID

LIFECYCLE_PROMISED = "PROMISED"
LIFECYCLE_FULFILLED = "FULFILLED"
LIFECYCLE_BROKEN = "BROKEN"
LIFECYCLE_EXPIRED = "EXPIRED"
LIFECYCLE_CANCELLED = "CANCELLED"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def get_promise_outcome(db: Session, promise_to_pay_id: int) -> PromiseOutcome | None:
    return db.query(PromiseOutcome).filter(PromiseOutcome.promise_to_pay_id == promise_to_pay_id).first()


def mark_broken_promises(db: Session, *, as_of: datetime | None = None) -> list[PromiseOutcome]:
    """Finds every still-VALID promise whose promised_date has passed with
    no lifecycle resolution yet, marks it BROKEN, and opens one
    RevenueRiskEvent(event_type="promise_to_pay_broken") per broken promise
    for the recovery engine to act on next. Idempotent: a promise that
    already has a PromiseOutcome row (of any lifecycle_status) is skipped --
    this function never re-resolves or re-opens a feedback event for it."""
    as_of = as_of or _utcnow()
    candidates = (
        db.query(PromiseToPay)
        .filter(PromiseToPay.status == STATUS_VALID, PromiseToPay.promised_date < as_of)
        .order_by(PromiseToPay.id)
        .all()
    )

    created: list[PromiseOutcome] = []
    for promise in candidates:
        if get_promise_outcome(db, promise.id) is not None:
            continue  # already swept -- BROKEN, FULFILLED, or otherwise resolved

        cumulative_payment_attempts = (
            db.query(PolicyDecision)
            .filter(PolicyDecision.subscription_id == promise.subscription_id, PolicyDecision.selected_candidate_type != NO_ACTION)
            .count()
        )

        # Best-effort lookup of the original failure event's amount (paise ->
        # rupees) via failure_events -> raw_events -- promises_to_pay itself
        # has no amount column. Falls back to 0.0 if the original event can't
        # be found (should not happen on a real pipeline, but this function
        # must never raise over a missing lookup).
        original_amount = 0.0
        failure = db.query(FailureEvent).filter(FailureEvent.id == promise.event_id).first()
        if failure is not None:
            raw = db.query(RawEvent).filter(RawEvent.id == failure.raw_event_id).first()
            if raw is not None and raw.amount is not None:
                original_amount = raw.amount / 100.0

        revenue_risk_event = RevenueRiskEvent(
            idempotency_key=f"promise_to_pay_broken:{promise.id}",
            event_type="promise_to_pay_broken",
            external_id=str(promise.id),
            customer_ref=promise.subscription_id,
            amount=original_amount,
            occurred_at=as_of,
            reason="promised_date_passed_without_fulfillment",
            context_json=json.dumps({
                "original_event_id": promise.event_id,
                "promise_to_pay_id": promise.id,
                "original_amount": original_amount,
                "cumulative_payment_attempts": cumulative_payment_attempts,
                "attempts_so_far": 0,
            }),
            recovery_eligible=None,
            status="OPEN",
        )
        db.add(revenue_risk_event)
        db.flush()

        outcome = PromiseOutcome(
            promise_to_pay_id=promise.id,
            lifecycle_status=LIFECYCLE_BROKEN,
            status_reason=f"promised_date={promise.promised_date} passed as_of={as_of} with no fulfillment signal",
            resolved_by="system_auto_expire",
            triggered_reevaluation=True,
            reevaluation_event_id=revenue_risk_event.id,
        )
        db.add(outcome)
        db.add(
            AuditLog(
                failure_event_id=promise.event_id,
                action="promise_broken",
                reason=(
                    f"promises_to_pay.id={promise.id} promised_date={promise.promised_date} | "
                    f"opened revenue_risk_events.id={revenue_risk_event.id} for re-evaluation"
                ),
                actor="promise_lifecycle",
            )
        )
        db.commit()
        created.append(outcome)
        log.info("Marked promises_to_pay.id=%s BROKEN, opened revenue_risk_events.id=%s", promise.id, revenue_risk_event.id)

    return created


def confirm_promise_fulfilled(db: Session, *, promise_to_pay_id: int, resolved_by: str = "manual") -> PromiseOutcome:
    """Manually/demo-confirms a promise was kept. No live payment-confirmation
    webhook exists in this project (see app.models.RecoveryOutcome's own
    binding rule) -- `resolved_by` must never be "webhook_confirmed" unless a
    real confirmation source actually exists; callers that don't have one
    should use "manual" or "demo_synthetic". Idempotent: a promise that
    already has a PromiseOutcome row is returned unchanged, never re-resolved."""
    existing = get_promise_outcome(db, promise_to_pay_id)
    if existing is not None:
        return existing

    outcome = PromiseOutcome(
        promise_to_pay_id=promise_to_pay_id, lifecycle_status=LIFECYCLE_FULFILLED,
        status_reason=f"confirmed_fulfilled_by={resolved_by}", resolved_by=resolved_by, triggered_reevaluation=False,
    )
    db.add(outcome)
    db.add(
        AuditLog(
            failure_event_id=None, action="promise_fulfilled",
            reason=f"promises_to_pay.id={promise_to_pay_id} resolved_by={resolved_by}", actor="promise_lifecycle",
        )
    )
    db.commit()
    log.info("Marked promises_to_pay.id=%s FULFILLED (resolved_by=%s)", promise_to_pay_id, resolved_by)
    return outcome


def cancel_promise(db: Session, *, promise_to_pay_id: int, reason: str) -> PromiseOutcome:
    """E.g. the underlying subscription itself was cancelled -- the promise
    is moot, not broken. Idempotent, same convention as the two functions above."""
    existing = get_promise_outcome(db, promise_to_pay_id)
    if existing is not None:
        return existing

    outcome = PromiseOutcome(
        promise_to_pay_id=promise_to_pay_id, lifecycle_status=LIFECYCLE_CANCELLED,
        status_reason=reason, resolved_by="manual", triggered_reevaluation=False,
    )
    db.add(outcome)
    db.add(
        AuditLog(
            failure_event_id=None, action="promise_cancelled",
            reason=f"promises_to_pay.id={promise_to_pay_id} reason={reason}", actor="promise_lifecycle",
        )
    )
    db.commit()
    log.info("Marked promises_to_pay.id=%s CANCELLED", promise_to_pay_id)
    return outcome
