"""
FIX #2: turns one already-stored, already-verified `RawEvent` into a full
recovery decision -- the missing link the full-system audit identified
between "webhook received and stored" (app/main.py, unmodified) and
"classified and orchestrated" (previously a manual, separately-run script).

Two payment.failed paths, both reaching a full recovery pipeline:

    RawEvent (already stored, already HMAC-verified, already deduplicated)

    subscription_id present:
        -> classify_raw_event()        (classification/service.py, reused as-is)
        -> orchestrate_recovery()      (recovery/orchestrator.py, reused as-is)

    subscription_id ABSENT, but payment_id + amount present (e.g. a
    Payment Link or other one-time-payment failure -- generalized, see
    "GENERALIZATION" below):
        -> RevenueRiskEvent(event_type="payment_failed_no_subscription")
        -> orchestrate_revenue_event() (recovery/revenue_orchestrator.py, reused as-is)

Neither branch is a second recovery engine: both ultimately reuse the exact
same classification/compliance/LLM/audit machinery every other event type in
this codebase already goes through -- see GENERALIZATION below for why a
one-time payment reuses the Track-03 revenue-risk path rather than a third,
bespoke pipeline.

Closed-loop hardening: a `payment.captured` RawEvent takes a different,
parallel branch straight to recovery/payment_reconciliation.py::confirm_payment_recovery
-- see that module for how a successful payment reconciles against an
existing PENDING recovery case (now including the one-time-payment domain
too). Still exactly one webhook architecture: this function remains the
single place app/main.py's webhook handler routes every already-stored
RawEvent through, by event_type.

This module contains NO classification logic, NO policy logic, NO
orchestration logic, and NO payment-confirmation logic of its own -- same
"sequences existing modules, decides nothing itself" pattern
recovery/orchestrator.py and recovery/promise_service.py already
established. Its responsibilities are: (a) deciding whether a raw event is
the kind this project's recovery workflow applies to at all, (b) deciding
WHICH of the two payment.failed paths above a given RawEvent qualifies for,
(c) translating a `RawEvent` row's fields into the input dataclass the
chosen orchestrator expects, and (d) routing a payment-success event to the
reconciliation module instead.

GENERALIZATION (removes the old subscription-only dead end): a
`payment.failed` RawEvent with no `subscription_id` used to be stored and
then permanently skipped ("Received, not orchestrated" on the dashboard) --
there was no live automatic-retry capability to fabricate for it, but that
is not the same thing as having nothing recoverable to do. When enough
authoritative context exists (a real `payment_id` and a known `amount` --
Razorpay always sends both on a genuine payment.failed delivery), this now
routes through the SAME generalized revenue-risk pipeline
checkout_abandoned/mandate_failed/receivable_overdue already use
(recovery/revenue_orchestrator.py + policy/revenue_recovery_policy.py +
policy/compliance_v2.py), via a new event_type,
`ONE_TIME_PAYMENT_EVENT_TYPE`. Classification still runs (reusing
classification/rules.py::classify -- see policy/one_time_payment_rules.py),
the LLM still only writes copy for a candidate the deterministic policy
already selected, and RecoveryOutcome still stays honestly
PENDING/unconfirmed_pending for every live row -- nothing about the
authoritative-confirmation or LLM-never-decides guarantees changes. A
`payment.failed` with NEITHER a subscription_id NOR enough authoritative
context (e.g. the payment entity itself is entirely missing from the
payload) is genuinely unprocessable and still results in
`OUTCOME_SKIPPED_INSUFFICIENT_CONTEXT`, unchanged in spirit from before.

IDEMPOTENT BY CONSTRUCTION, NOT BY ANYTHING NEW HERE: `classify_raw_event`
and `orchestrate_recovery`'s own `decide_for_failure_event_engine_v4` /
compliance / LLM-invocation checks are already each individually idempotent
(query-before-act, keyed on raw_event_id / event_id) -- and so is the
one-time-payment path (RevenueRiskEvent.idempotency_key, then
policy/policy_decision_store.py's own event_id-keyed idempotency, exactly
the same machinery the other 4 revenue-risk domains already rely on).
Calling `process_raw_event` twice for the same RawEvent -- whether from a
second webhook delivery reaching this function (it never does; see
app/main.py's duplicate-event short-circuit, which returns before this
module is ever called) or from a manual reprocessing run
(scripts/reprocess_raw_events.py) -- is always safe: whatever already
happened is skipped, whatever didn't is attempted.

UNIT CONVERSION: Razorpay sends `amount` in paise; every model/policy/cost
number in this codebase (policy/costs.py's Rs5 retry_cost, the synthetic
dataset's plan-tier pricing, etc.) is in rupees. Getting this wrong would
silently misprice every live decision by 100x -- see `_amount_in_rupees`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.logging_config import log
from app.models import AuditLog, RawEvent, RevenueRiskEvent
from classification.service import classify_raw_event
from policy.policy_decision_store import REVENUE_DOMAIN_EVENT_ID_OFFSET
from recovery.orchestrator import RecoveryEventInput, orchestrate_recovery
from recovery.payment_reconciliation import confirm_payment_recovery
from recovery.revenue_orchestrator import orchestrate_revenue_event
from recovery.revenue_schemas import RevenueRiskEventInput

# The only event type this project's recovery workflow applies to (brief
# scope: Razorpay Subscriptions and one-time/Payment Link payments,
# insufficient_fund and the handful of other error_reason values
# classification/rules.py recognizes -- all carried on `payment.failed`).
# Other event types Razorpay may deliver (e.g. `subscription.activated`,
# etc.) are stored (webhook-receiver behavior, unchanged) but never enter
# classification/orchestration.
ORCHESTRATABLE_EVENT_TYPES = frozenset({"payment.failed"})

# Recovery-confirmation closed loop (see recovery/payment_reconciliation.py):
# the one Razorpay event type authoritative enough to move a recovery case
# from PENDING to RECOVERED/PARTIALLY_RECOVERED. `payment.captured` is
# Razorpay's own documented "money has actually moved" event for a payment,
# available in Test Mode exactly as in live mode.
PAYMENT_SUCCESS_EVENT_TYPES = frozenset({"payment.captured"})

# Track-03 revenue-risk event_type for a payment.failed with no
# subscription_id but sufficient authoritative context -- see
# policy/one_time_payment_rules.py / policy/revenue_recovery_policy.py.
ONE_TIME_PAYMENT_EVENT_TYPE = "payment_failed_no_subscription"

OUTCOME_COMPLETED = "completed"
OUTCOME_SKIPPED_UNSUPPORTED_EVENT_TYPE = "skipped_unsupported_event_type"
# Kept for backward-compatible log/audit reading of old rows; no longer
# returned by this module -- see OUTCOME_SKIPPED_INSUFFICIENT_CONTEXT.
OUTCOME_SKIPPED_MISSING_SUBSCRIPTION_ID = "skipped_missing_subscription_id"
OUTCOME_SKIPPED_INSUFFICIENT_CONTEXT = "skipped_insufficient_context"
OUTCOME_ALREADY_PROCESSED = "already_processed"


def _amount_in_rupees(raw_event: RawEvent) -> float:
    """Razorpay's `amount` is paise; this codebase's model/policy/cost layers
    are entirely rupee-denominated (see module docstring)."""
    return (raw_event.amount or 0) / 100.0


def _failure_timestamp(raw_event: RawEvent) -> datetime:
    """Prefers Razorpay's own `created_at` (the payload's top-level unix
    timestamp, exactly what a live webhook carries); falls back to when
    this server received it if that field is absent. Returned as a NAIVE
    UTC datetime -- every candidate/compliance/promise datetime in this
    codebase is naive by existing convention (policy/retry_candidates.py),
    and comparing a naive value against a tz-aware one raises TypeError."""
    if raw_event.razorpay_created_at:
        return datetime.fromtimestamp(raw_event.razorpay_created_at, tz=timezone.utc).replace(tzinfo=None)
    return raw_event.received_at.replace(tzinfo=None) if raw_event.received_at.tzinfo else raw_event.received_at


def _build_one_time_payment_event(db: Session, raw_event: RawEvent) -> RevenueRiskEvent:
    """Idempotent by payment_id (item 13: "use a stable payment/order
    correlation identifier plus event id" -- the event id / dedup itself is
    already handled one layer up by app/main.py's razorpay_event_id check;
    this is the SECOND, domain-level idempotency tier the other 4 revenue-
    risk routes also each have, in case the same payment_id genuinely
    fails twice under two different Razorpay event ids)."""
    idempotency_key = f"{ONE_TIME_PAYMENT_EVENT_TYPE}:{raw_event.payment_id}"
    existing = db.query(RevenueRiskEvent).filter(RevenueRiskEvent.idempotency_key == idempotency_key).first()
    if existing is not None:
        return existing

    revenue_risk_event = RevenueRiskEvent(
        idempotency_key=idempotency_key,
        event_type=ONE_TIME_PAYMENT_EVENT_TYPE,
        external_id=raw_event.payment_id,
        # No real customer identity is available on a payment.failed
        # payload -- payment_id is the closest stable per-event reference,
        # same convention checkout_abandoned uses its cart's customer_id for.
        customer_ref=raw_event.payment_id,
        amount=_amount_in_rupees(raw_event),
        currency=raw_event.currency or "INR",
        occurred_at=_failure_timestamp(raw_event),
        reason=raw_event.error_reason,
        context_json=json.dumps({
            "raw_event_id": raw_event.id,
            "payment_id": raw_event.payment_id,
            "order_id": raw_event.order_id,
            "error_code": raw_event.error_code,
            "error_reason": raw_event.error_reason,
            "error_source": raw_event.error_source,
            "error_step": raw_event.error_step,
        }),
        status="OPEN",
    )
    db.add(revenue_risk_event)
    db.flush()
    db.add(
        AuditLog(
            raw_event_id=raw_event.id,
            failure_event_id=revenue_risk_event.id + REVENUE_DOMAIN_EVENT_ID_OFFSET,
            action="revenue_risk_event_received_and_stored",
            reason=f"event_type={ONE_TIME_PAYMENT_EVENT_TYPE} payment_id={raw_event.payment_id} order_id={raw_event.order_id}",
            actor="system",
        )
    )
    db.commit()
    return revenue_risk_event


def process_raw_event(db: Session, raw_event: RawEvent, *, model: dict | None = None) -> str:
    """
    Classify + orchestrate one already-stored raw_event. Returns a short
    outcome string (also written into the audit trail by the stages this
    function calls) -- never raises for a business reason (unsupported
    event type, insufficient context are normal, expected outcomes, not
    errors); a genuine unexpected exception (DB failure, etc.) still
    propagates to the caller, exactly like every other stage in this
    pipeline -- app/main.py is responsible for catching it so a downstream
    failure never undoes the already-committed raw event storage.

    `model` is the UNIFIED ML model (model/unified_model.py), used only for
    the no-subscription/one-time-payment branch below -- see
    policy/revenue_recovery_policy.py's EVENT-TYPE ALIASING note for why a
    payment.failed with no subscription_id still lands in that model's
    `payment_failed` schema slot. The subscription branch keeps using its
    own separate, pre-existing Model B (policy/decision_engine_v4.py),
    unmodified and untouched by this parameter.
    """
    if raw_event.event_type in PAYMENT_SUCCESS_EVENT_TYPES:
        return confirm_payment_recovery(db, raw_event)

    if raw_event.event_type not in ORCHESTRATABLE_EVENT_TYPES:
        log.info("Skipping orchestration for raw_events.id=%s: unsupported event_type=%s", raw_event.id, raw_event.event_type)
        return OUTCOME_SKIPPED_UNSUPPORTED_EVENT_TYPE

    if raw_event.subscription_id:
        failure_event, _classified_now = classify_raw_event(db, raw_event)

        required_fields_present = bool(raw_event.subscription_id and raw_event.payment_id and raw_event.amount is not None)

        event = RecoveryEventInput(
            event_id=failure_event.id,
            subscription_id=raw_event.subscription_id,
            failure_timestamp=_failure_timestamp(raw_event),
            amount=_amount_in_rupees(raw_event),
            error_code=raw_event.error_code,
            error_reason=raw_event.error_reason,
            error_source=raw_event.error_source,
            error_step=raw_event.error_step,
            # A real Razorpay webhook carries none of the synthetic dataset's
            # engineered features (payday proximity, prior self-resolved rate,
            # plan_tier, etc.) -- there is no synthetic benchmark data attached
            # to a genuinely live payment. Model B's own "insufficient_features"
            # check (policy/decision_engine.py::_predict_recovery_values) then
            # correctly treats this as a malformed-model-input condition and
            # falls back to the rule-based tier, which needs none of those
            # features (it computes purely from failure_timestamp). This is an
            # honest, working degradation, not a crash -- documented in the
            # README's known limitations, not hidden.
            failure_context=None,
            required_fields_present=required_fields_present,
        )
        orchestrate_recovery(db, event)
        return OUTCOME_COMPLETED

    # No subscription_id -- generalized one-time-payment path. Only enters
    # the pipeline with enough authoritative context to act on (never
    # invents a subscription_id, never guesses payment_id/amount).
    if not raw_event.payment_id or raw_event.amount is None:
        log.info(
            "Skipping orchestration for raw_events.id=%s: no subscription_id and insufficient authoritative "
            "context (payment_id=%s, amount=%s)", raw_event.id, raw_event.payment_id, raw_event.amount,
        )
        return OUTCOME_SKIPPED_INSUFFICIENT_CONTEXT

    revenue_risk_event = _build_one_time_payment_event(db, raw_event)
    event = RevenueRiskEventInput(
        event_type=ONE_TIME_PAYMENT_EVENT_TYPE,
        event_id=revenue_risk_event.id,
        customer_ref=raw_event.payment_id,
        occurred_at=_failure_timestamp(raw_event),
        amount=_amount_in_rupees(raw_event),
        currency=raw_event.currency or "INR",
        domain_context={
            "error_code": raw_event.error_code,
            "error_reason": raw_event.error_reason,
            "error_source": raw_event.error_source,
            "error_step": raw_event.error_step,
        },
        required_fields_present=True,
    )
    orchestrate_revenue_event(db, event, model=model)
    return OUTCOME_COMPLETED
