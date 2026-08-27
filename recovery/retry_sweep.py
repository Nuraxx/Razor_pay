"""
MULTI-ATTEMPT PERSISTENCE (final pre-submission audit): advances a
subscription payment-failure event's own pre-computed Fixed-Retry-style
retry schedule (policy/decision_engine_v4.py::build_retry_schedule_from_decision,
persisted on policy_decisions.retry_schedule_json at decision time) one step
at a time, as each step's scheduled datetime arrives -- exactly the
capability the evaluation-side economic finding (README "policy-v4:
multi-attempt persistence") measured, now wired into the live path so the
capability the evaluation credits is the SAME one the live system runs, not
an evaluation-only fiction.

Mirrors recovery/scheduler.py + recovery/promise_sweep.py's existing
in-process asyncio-loop pattern exactly (no new scheduler framework, no
Celery/Redis) -- see recovery/scheduler.py's own docstring for why this
project keeps that deliberately simple.

STOPS EARLY, per event, the moment RecoveryOutcome confirms recovery
(recovery_status != "PENDING") -- the same "stop at first recovered
attempt" semantics evaluation/evaluate_decision_engine_v4.py::score_fixed_retry_sequence
already scores Fixed Retry AND this policy by. Consistent with the rest of
this codebase's binding rule (app.models.RecoveryOutcome's own docstring):
this backend never calls Razorpay to execute a retry -- every advance here
is RECORDED ONLY (an audit_log row + retry_schedule_next_index increment),
never a live payment action. Recovery is learned exclusively from a real
`payment.captured` webhook (recovery/payment_reconciliation.py), same as
attempt 1 -- this sweep never writes to RecoveryOutcome itself.

Deliberately does NOT re-trigger LLM communication for follow-up PAYMENT
attempts: the original decision's own communication (recovery/orchestrator.py)
already ran once; subsequent scheduled payment attempts are silent retries,
matching Fixed Retry's own "silent auto-retry" philosophy (policy/baselines.py)
rather than re-contacting the customer once per attempt.

SECOND, UNRELATED CONCERN handled by the SAME sweep pass (final pre-
submission audit, "defer, don't terminate" -- see
policy/contact_hours.py::next_contact_hours_start): a communication that was
blocked SPECIFICALLY by contact-hours (never opt-out/consent/duplicate) is
marked `communication_deferred_until` on its policy_decisions row by
recovery/orchestrator.py; `fire_one_deferred_communication` below fires it
-- exactly once, via the same `_persist`/LLMInvocation path
recovery/orchestrator.py already uses -- once that time arrives, instead of
losing it outright. Reuses this module's existing periodic loop rather than
adding a second scheduler for what is, mechanically, the same
"is-it-time-yet" check.

RE-CHECK-BEFORE-ACTING (final pre-submission audit, THIRD concern handled by
the same sweep pass): attempt 1's compliance evaluation (opt-out,
cancellation) is captured once, at decision time -- but multi-attempt
persistence spreads real attempts across real elapsed time, so a customer
who opts out (or whose subscription gets cancelled) BETWEEN attempt 1 and a
later scheduled attempt/deferred communication must not have that later
action fire anyway. `_subscription_still_eligible` re-derives this from the
ONLY durable state this codebase has for it -- `PolicyDecision.customer_opted_out`
(sticky, set by recovery/orchestrator.py; see that column's own docstring for
why this project previously had NO durable opt-out record at all) and the
MOST RECENT `PolicyDecision.classification_bucket` for the same
subscription_id (a later event reclassified as e.g. customer_cancelled is
this codebase's own existing "opt-out proxy" convention -- see
policy/compliance.py's `is_cancelled` derivation). Mirrors
recovery/promise_sweep.py's own re-check-before-acting shape: never trusts
state captured only when the sequence was first created.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.logging_config import log
from app.models import AuditLog, FailureEvent, PolicyDecision, RawEvent, RecoveryOutcome
from llm.client import LLMClient
from llm.service import generate_outreach_microcopy_and_log
from policy.decision_engine import NO_ACTION
from policy.guardrails import is_classification_allowed


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _pending_recovery(db: Session, event_id: int) -> bool:
    outcome = (
        db.query(RecoveryOutcome)
        .filter(RecoveryOutcome.event_id == event_id, RecoveryOutcome.event_type == "payment_failed")
        .first()
    )
    return outcome is not None and outcome.recovery_status == "PENDING"


def _subscription_still_eligible(db: Session, subscription_id: str) -> tuple[bool, str]:
    """RE-CHECK-BEFORE-ACTING (final pre-submission audit): re-derives
    opt-out/cancellation eligibility from durable state ACROSS EVERY
    policy_decisions row for `subscription_id` -- not just the row being
    advanced -- so an opt-out or reclassification recorded on a LATER event
    for the same subscription is honored. See module docstring."""
    rows = db.query(PolicyDecision).filter(PolicyDecision.subscription_id == subscription_id).all()
    if not rows:
        return True, "eligible"  # defensive -- the caller's own row always exists by construction
    if any(r.customer_opted_out for r in rows):
        return False, "customer_opted_out_detected_since_scheduling"
    latest = max(rows, key=lambda r: (r.decided_at or _utcnow(), r.id))
    if latest.classification_bucket is not None and not is_classification_allowed(latest.classification_bucket):
        return False, f"classification_no_longer_retryable: bucket={latest.classification_bucket!r}"
    return True, "eligible"


def _abort_remaining_schedule(db: Session, decision: PolicyDecision, *, reason: str) -> None:
    """Permanently suppresses every remaining scheduled attempt on
    `decision` -- sets `retry_schedule_next_index` to the schedule's own
    length so this row is excluded from every future sweep query
    (`run_retry_sweep_once`'s WHERE clause), never re-checked or re-aborted
    again. One audit_log row per abort, not one per sweep pass."""
    schedule_len = len(json.loads(decision.retry_schedule_json or "[]"))
    aborted_count = schedule_len - decision.retry_schedule_next_index
    decision.retry_schedule_next_index = schedule_len
    db.add(
        AuditLog(
            failure_event_id=decision.event_id,
            action="retry_schedule_aborted",
            reason=f"policy_decisions.id={decision.id} {aborted_count} remaining attempt(s) permanently suppressed: {reason}",
            actor="compliance",
        )
    )
    db.commit()
    log.info("Retry sweep: aborted %s remaining attempt(s) for policy_decisions.id=%s (%s)", aborted_count, decision.id, reason)


def advance_one_retry_schedule(db: Session, decision: PolicyDecision, *, now: datetime | None = None) -> bool:
    """Advances `decision` by exactly one schedule step if it is due.
    Returns True if a follow-up attempt was recorded, False if nothing was
    due (schedule already exhausted, next step's scheduled time hasn't
    arrived yet, or recovery is no longer PENDING). Idempotent: calling this
    again on a row that just advanced does nothing until its OWN next step
    becomes due, and a fully-exhausted schedule (`next_index >= len(schedule)`)
    always returns False -- no duplicate attempts possible from repeated
    sweep passes."""
    now = now or _utcnow()
    types: list[str] = json.loads(decision.retry_schedule_json or "[]")
    datetimes_raw: list[str] = json.loads(decision.retry_schedule_datetimes_json or "[]")
    idx = decision.retry_schedule_next_index

    if idx >= len(types):
        return False
    if not _pending_recovery(db, decision.event_id):
        return False

    next_dt = datetime.fromisoformat(datetimes_raw[idx])
    next_dt_aware = next_dt if next_dt.tzinfo is not None else next_dt.replace(tzinfo=timezone.utc)
    if next_dt_aware > now:
        return False

    eligible, ineligible_reason = _subscription_still_eligible(db, decision.subscription_id)
    if not eligible:
        _abort_remaining_schedule(db, decision, reason=ineligible_reason)
        return False

    next_type = types[idx]
    decision.retry_schedule_next_index = idx + 1
    db.add(
        AuditLog(
            failure_event_id=decision.event_id,
            action="retry_schedule_attempt_recorded",
            reason=(
                f"policy_decisions.id={decision.id} attempt_index={idx + 1} candidate_type={next_type} "
                f"scheduled_at={datetimes_raw[idx]} | multi-attempt persistence follow-up, recorded only "
                f"(no live Razorpay call -- same non-execution rule as attempt 1)"
            ),
            actor="policy",
        )
    )
    db.commit()
    log.info(
        "Retry sweep: recorded follow-up attempt %s/%s for event_id=%s (%s)",
        idx + 1, len(types), decision.event_id, next_type,
    )
    return True


def fire_one_deferred_communication(
    db: Session, decision: PolicyDecision, *, now: datetime | None = None, llm_client: LLMClient | None = None,
) -> bool:
    """Fires exactly one deferred communication if it is due. Returns True if
    fired, False if nothing was due (not yet due, already sent, or no
    deferral pending). Reuses llm/service.py::generate_outreach_microcopy_and_log
    unchanged -- the SAME function recovery/orchestrator.py's own step 5
    calls -- so the resulting LLMInvocation row makes
    `ComplianceContext.communication_already_sent` correctly True for any
    later re-orchestration of this event_id, exactly as if it had been sent
    on time. Amount is re-derived from the SAME RawEvent row the original
    decision was made from (via the failure_events join); customer_segment/
    language are not persisted anywhere reachable from a policy_decisions
    row alone, so they fall back to RecoveryEventInput's own defaults
    ("unknown"/"en") -- an honest degrade, not a fabricated value."""
    now = now or _utcnow()
    if decision.communication_deferred_until is None or decision.communication_deferred_sent:
        return False
    deferred_until = decision.communication_deferred_until
    deferred_until_aware = deferred_until if deferred_until.tzinfo is not None else deferred_until.replace(tzinfo=timezone.utc)
    if deferred_until_aware > now:
        return False

    eligible, ineligible_reason = _subscription_still_eligible(db, decision.subscription_id)
    if not eligible:
        # RE-CHECK-BEFORE-ACTING: permanently suppressed, never actually
        # sent -- mark it "sent" so this row is excluded from every future
        # sweep pass (never re-checked or re-suppressed again), same
        # exclusion mechanism _abort_remaining_schedule uses for the retry
        # schedule.
        decision.communication_deferred_sent = True
        db.add(
            AuditLog(
                failure_event_id=decision.event_id,
                action="deferred_communication_suppressed",
                reason=f"policy_decisions.id={decision.id} deferred communication permanently suppressed, never sent: {ineligible_reason}",
                actor="compliance",
            )
        )
        db.commit()
        log.info("Retry sweep: suppressed deferred communication for event_id=%s (%s)", decision.event_id, ineligible_reason)
        return False

    from recovery.orchestrator import describe_retry_window  # lazy: avoids any import-order coupling at module load
    from recovery.webhook_pipeline import _amount_in_rupees  # SAME paise->rupees conversion the original decision used

    raw_event = (
        db.query(RawEvent)
        .join(FailureEvent, FailureEvent.raw_event_id == RawEvent.id)
        .filter(FailureEvent.id == decision.event_id)
        .first()
    )
    amount = _amount_in_rupees(raw_event) if raw_event is not None else 0.0

    llm_result, _invocation = generate_outreach_microcopy_and_log(
        db, event_id=decision.event_id, failure_bucket=decision.classification_bucket or "retryable_soft",
        customer_segment="unknown", language="en", will_retry=decision.selected_candidate_type != NO_ACTION,
        retry_window_description=describe_retry_window(decision.selected_candidate_type), amount_rupees=amount,
        client=llm_client,
    )
    decision.communication_deferred_sent = True
    db.add(
        AuditLog(
            failure_event_id=decision.event_id,
            action="deferred_communication_fired",
            reason=(
                f"policy_decisions.id={decision.id} deferred_until={deferred_until.isoformat()} "
                f"llm_success={llm_result.success} | contact-hours deferral, final pre-submission audit"
            ),
            actor="policy",
        )
    )
    db.commit()
    log.info("Retry sweep: fired deferred communication for event_id=%s (llm_success=%s)", decision.event_id, llm_result.success)
    return True


def run_retry_sweep_once(db=None) -> int:
    """One sweep pass covering BOTH concerns this module owns: advancing
    multi-attempt retry schedules, and firing due deferred communications
    (see module docstring for why both live in one pass). `db=None` (the
    real, running-app case) opens and closes its own session, matching
    recovery/scheduler.py::run_promise_sweep_once. Returns the total number
    of actions actually recorded this pass (retry attempts + deferred
    communications combined) -- 0 on a quiet pass, the normal, expected case
    most of the time."""
    owns_db = db is None
    db = db or SessionLocal()
    processed = 0
    try:
        retry_candidates = db.query(PolicyDecision).filter(PolicyDecision.retry_schedule_json.isnot(None)).all()
        for decision in retry_candidates:
            types = json.loads(decision.retry_schedule_json or "[]")
            if decision.retry_schedule_next_index >= len(types):
                continue
            try:
                if advance_one_retry_schedule(db, decision):
                    processed += 1
            except Exception:
                db.rollback()
                log.exception("Retry sweep: failed to advance policy_decisions.id=%s -- will retry next cycle", decision.id)

        deferred_candidates = (
            db.query(PolicyDecision)
            .filter(PolicyDecision.communication_deferred_until.isnot(None), PolicyDecision.communication_deferred_sent.is_(False))
            .all()
        )
        for decision in deferred_candidates:
            try:
                if fire_one_deferred_communication(db, decision):
                    processed += 1
            except Exception:
                db.rollback()
                log.exception("Retry sweep: failed to fire deferred communication for policy_decisions.id=%s -- will retry next cycle", decision.id)
        return processed
    finally:
        if owns_db:
            db.close()


async def retry_sweep_background_loop(interval_seconds: int) -> None:
    """Runs run_retry_sweep_once() forever, sleeping interval_seconds between
    passes. Same two-level exception isolation as
    recovery/scheduler.py::promise_sweep_background_loop: one row's own
    failure is already caught inside run_retry_sweep_once; this loop
    additionally guards the sweep call itself, so a transient failure is
    logged and the NEXT cycle still runs."""
    while True:
        try:
            processed = run_retry_sweep_once()
            if processed:
                log.info("Retry sweep: recorded %s follow-up attempt(s)", processed)
        except asyncio.CancelledError:
            raise  # real shutdown -- must propagate, not be swallowed as a "failure"
        except Exception:
            log.exception("Retry sweep: sweep pass failed -- will retry next cycle")
        await asyncio.sleep(interval_seconds)
