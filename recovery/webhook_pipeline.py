"""
FIX #2: turns one already-stored, already-verified `RawEvent` into a full
recovery decision -- the missing link the full-system audit identified
between "webhook received and stored" (app/main.py, Day 1, unmodified) and
"classified and orchestrated" (previously a manual, separately-run script).

    RawEvent (already stored, already HMAC-verified, already deduplicated)
        -> classify_raw_event()        (classification/service.py, Day 2, reused as-is)
        -> orchestrate_recovery()      (recovery/orchestrator.py, reused as-is)

This module contains NO classification logic and NO orchestration logic of
its own -- same "sequences existing modules, decides nothing itself"
pattern recovery/orchestrator.py and recovery/promise_service.py already
established. Its only two responsibilities are: (a) deciding whether a raw
event is the kind this project's recovery workflow applies to at all
(FIX #2's "only process appropriate events"), and (b) translating a
`RawEvent` row's fields into the `RecoveryEventInput` the orchestrator
expects.

IDEMPOTENT BY CONSTRUCTION, NOT BY ANYTHING NEW HERE: `classify_raw_event`
and `orchestrate_recovery`'s own `decide_for_failure_event_engine_v4` /
compliance / LLM-invocation checks are already each individually idempotent
(query-before-act, keyed on raw_event_id / event_id). Calling
`process_raw_event` twice for the same RawEvent -- whether from a second
webhook delivery reaching this function (it never does; see app/main.py's
duplicate-event short-circuit, which returns before this module is ever
called) or from a manual reprocessing run (scripts/reprocess_raw_events.py)
-- is always safe: whatever already happened is skipped, whatever didn't is
attempted.

UNIT CONVERSION: Razorpay sends `amount` in paise; every model/policy/cost
number in this codebase (policy/costs.py's Rs5 retry_cost, the synthetic
dataset's plan-tier pricing, etc.) is in rupees. Getting this wrong would
silently misprice every live decision by 100x -- see `_amount_in_rupees`.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.logging_config import log
from app.models import RawEvent
from classification.service import classify_raw_event
from recovery.orchestrator import RecoveryEventInput, orchestrate_recovery

# The only event type this project's recovery workflow applies to (brief
# scope: Razorpay Subscriptions, insufficient_fund and the handful of other
# error_reason values classification/rules.py recognizes -- all carried on
# `payment.failed`). Other event types Razorpay may deliver (e.g.
# `subscription.charged` -- a SUCCESSFUL charge, nothing to recover;
# `subscription.activated`, etc.) are stored (Day 1 behavior, unchanged)
# but never enter classification/orchestration.
ORCHESTRATABLE_EVENT_TYPES = frozenset({"payment.failed"})

OUTCOME_COMPLETED = "completed"
OUTCOME_SKIPPED_UNSUPPORTED_EVENT_TYPE = "skipped_unsupported_event_type"
OUTCOME_SKIPPED_MISSING_SUBSCRIPTION_ID = "skipped_missing_subscription_id"
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


def process_raw_event(db: Session, raw_event: RawEvent) -> str:
    """
    Classify + orchestrate one already-stored raw_event. Returns a short
    outcome string (also written into the audit trail by the stages this
    function calls) -- never raises for a business reason (unsupported
    event type, missing subscription id are normal, expected outcomes, not
    errors); a genuine unexpected exception (DB failure, etc.) still
    propagates to the caller, exactly like every other stage in this
    pipeline -- app/main.py is responsible for catching it so a downstream
    failure never undoes the already-committed raw event storage.
    """
    if raw_event.event_type not in ORCHESTRATABLE_EVENT_TYPES:
        log.info("Skipping orchestration for raw_events.id=%s: unsupported event_type=%s", raw_event.id, raw_event.event_type)
        return OUTCOME_SKIPPED_UNSUPPORTED_EVENT_TYPE

    if not raw_event.subscription_id:
        log.info("Skipping orchestration for raw_events.id=%s: no subscription_id (out of scope -- Subscriptions only)", raw_event.id)
        return OUTCOME_SKIPPED_MISSING_SUBSCRIPTION_ID

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
