"""
Promise-to-pay: customer reply -> parse -> validate -> persist.

    customer_reply_text
        -> llm/service.py::parse_promise_to_pay_and_log   (Day 11, reused as-is,
           already fail-safe/audited -- see that module for LLM failure handling)
        -> policy/promise_to_pay.py::validate_promise      (deterministic, new)
        -> app.models.PromiseToPay row                     (persisted, new)

This module contains NO LLM logic and NO validation logic of its own -- it
only sequences calls to the two modules that already do those things
(same "orchestrator contains no decision logic of its own" pattern
recovery/orchestrator.py already established) and handles persistence:
idempotency on an exact-duplicate reply, and marking a superseded promise
when a newer, distinct reply arrives for the same event.

recovery/orchestrator.py is the ONLY place a persisted promise here can
affect a payment decision -- this module never touches policy or
compliance itself.
"""
from __future__ import annotations

import hashlib
from datetime import date as _date
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.logging_config import log
from app.models import AuditLog, PromiseToPay
from llm.client import LLMClient
from llm.service import parse_promise_to_pay_and_log
from policy.promise_to_pay import DEFAULT_MIN_CONFIDENCE, STATUS_SUPERSEDED, STATUS_VALID, validate_promise


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _hash_reply(customer_reply_text: str) -> str:
    return hashlib.sha256(customer_reply_text.encode("utf-8")).hexdigest()


def get_active_promise(db: Session, event_id: int) -> PromiseToPay | None:
    """The current, still-VALID promise for this event, if any -- the only
    kind of promise recovery/orchestrator.py is ever allowed to act on.
    LOW_CONFIDENCE / INVALID_DATE / EXPIRED / SUPERSEDED rows are never
    returned here, which is precisely what keeps them from ever overriding
    anything, with no special-casing needed at the call site."""
    return (
        db.query(PromiseToPay)
        .filter(PromiseToPay.event_id == event_id, PromiseToPay.status == STATUS_VALID)
        .order_by(PromiseToPay.id.desc())
        .first()
    )


def record_customer_reply(
    db: Session,
    *,
    event_id: int,
    subscription_id: str,
    customer_reply_text: str,
    today: _date | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    client: LLMClient | None = None,
) -> tuple[PromiseToPay, bool]:
    """
    Parses, validates, and persists one customer reply as a PromiseToPay
    row. Returns (promise, created).

    Idempotent on (event_id, exact reply text): submitting byte-identical
    text twice for the same event returns the existing row unchanged
    (created=False), mirroring classification/service.py::classify_raw_event's
    established idempotency pattern -- never a duplicate LLM call, never a
    duplicate row.

    A newer, DISTINCT reply for the same event_id supersedes whatever
    promise was previously VALID for that event (SUPERSEDED, not deleted --
    the full history stays in the table and the audit trail).
    """
    today = today or datetime.now(timezone.utc).date()
    text_hash = _hash_reply(customer_reply_text)

    existing = (
        db.query(PromiseToPay)
        .filter(PromiseToPay.event_id == event_id, PromiseToPay.source_text_hash == text_hash)
        .first()
    )
    if existing is not None:
        db.add(
            AuditLog(
                failure_event_id=event_id,
                action="promise_skipped_duplicate",
                reason=f"event_id={event_id} already has an identical reply recorded as promises_to_pay.id={existing.id} (status={existing.status}); not re-parsed.",
                actor="promise",
            )
        )
        db.commit()
        log.info("Skipped duplicate customer reply for event_id=%s (already promises_to_pay.id=%s)", event_id, existing.id)
        return existing, False

    llm_result, invocation = parse_promise_to_pay_and_log(
        db, event_id=event_id, customer_reply_text=customer_reply_text, today=today, client=client,
    )
    parsed = llm_result.structured_result or {}
    validation = validate_promise(
        parsed_date=parsed.get("date"), confidence=float(parsed.get("confidence", 0.0)),
        channel=parsed.get("channel", "unspecified"), today=today, min_confidence=min_confidence,
    )

    previous_active = get_active_promise(db, event_id)
    if previous_active is not None:
        previous_active.status = STATUS_SUPERSEDED
        previous_active.status_reason = f"superseded_by_newer_promise: source_text_hash={text_hash}"
        previous_active.updated_at = _utcnow()

    promise = PromiseToPay(
        event_id=event_id,
        subscription_id=subscription_id,
        promised_date=validation.promised_datetime,
        confidence=float(parsed.get("confidence", 0.0)),
        channel=parsed.get("channel", "unspecified"),
        status=validation.status,
        status_reason=validation.reason,
        source_text_hash=text_hash,
        llm_invocation_id=invocation.id,
    )
    db.add(promise)
    db.flush()

    db.add(
        AuditLog(
            failure_event_id=event_id,
            action=f"promise_{validation.status.lower()}",
            reason=(
                f"promises_to_pay.id={promise.id} llm_success={llm_result.success} "
                f"parsed_date={parsed.get('date')!r} confidence={promise.confidence:.2f} channel={promise.channel!r} | "
                f"validation: {validation.reason}"
                + (f" | superseded promises_to_pay.id={previous_active.id}" if previous_active is not None else "")
            ),
            actor="promise",
        )
    )
    db.commit()

    log.info(
        "Recorded customer reply for event_id=%s: status=%s (promises_to_pay.id=%s)",
        event_id, validation.status, promise.id,
    )
    return promise, True
