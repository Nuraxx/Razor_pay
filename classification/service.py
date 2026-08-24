"""
Day-2 orchestration: read a raw_event, classify it deterministically
(classification/rules.py), store the result in failure_events, and record
the decision in audit_log — following the same audit_log convention Day 1
established in app/models.py ("every decision the system makes — including
deciding to do nothing — goes here").

Idempotency is enforced at the application layer (query-before-insert on
raw_event_id) rather than a DB constraint: this path is driven by a
single-threaded CLI/batch script (scripts/classify_raw_events.py), not
concurrent HTTP requests, so the race Day 1's webhook endpoint guards
against with a UNIQUE column doesn't apply here. Re-running classification
over the same raw_events is always safe.
"""
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.logging_config import log
from app.models import AuditLog, FailureEvent, RawEvent
from classification.rules import classify


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def classify_raw_event(db: Session, raw_event: RawEvent) -> tuple[FailureEvent, bool]:
    """
    Classify one raw_event and persist the result.

    Returns (failure_event, created). created is False when raw_event was
    already classified — the existing failure_events row is returned
    unchanged, no duplicate is created, but an audit_log entry recording the
    skip is still written.
    """
    existing = (
        db.query(FailureEvent)
        .filter(FailureEvent.raw_event_id == raw_event.id)
        .first()
    )
    if existing is not None:
        db.add(
            AuditLog(
                raw_event_id=raw_event.id,
                action="classification_skipped_duplicate",
                reason=(
                    f"raw_events.id={raw_event.id} already classified as "
                    f"failure_events.id={existing.id} (bucket={existing.classification_bucket}, "
                    f"rule_version={existing.rule_version}); not re-classified."
                ),
                actor="rule",
            )
        )
        db.commit()
        log.info(
            "Skipped duplicate classification for raw_events.id=%s (already failure_events.id=%s)",
            raw_event.id, existing.id,
        )
        return existing, False

    result = classify(
        error_code=raw_event.error_code,
        error_reason=raw_event.error_reason,
        error_source=raw_event.error_source,
        error_step=raw_event.error_step,
    )

    failure_event = FailureEvent(
        raw_event_id=raw_event.id,
        classification_bucket=result.bucket,
        classification_confidence=result.confidence,
        classified_at=_utcnow(),
        rule_version=result.rule_version,
    )
    db.add(failure_event)
    db.flush()  # populate failure_event.id before referencing it in the audit log

    db.add(
        AuditLog(
            raw_event_id=raw_event.id,
            action="classified",
            reason=result.reason,
            actor="rule",
        )
    )
    db.commit()

    log.info(
        "Classified raw_events.id=%s -> bucket=%s confidence=%s rule_version=%s (failure_events.id=%s)",
        raw_event.id, result.bucket, result.confidence, result.rule_version, failure_event.id,
    )
    return failure_event, True


def classify_all_raw_events(db: Session) -> dict:
    """Classify every raw_event, skipping ones already classified. Safe to re-run."""
    raw_events = db.query(RawEvent).order_by(RawEvent.id).all()

    newly_classified = 0
    already_classified_skipped = 0
    buckets: dict[str, int] = {}

    for raw_event in raw_events:
        failure_event, created = classify_raw_event(db, raw_event)
        if created:
            newly_classified += 1
        else:
            already_classified_skipped += 1
        bucket = failure_event.classification_bucket
        buckets[bucket] = buckets.get(bucket, 0) + 1

    return {
        "total_raw_events": len(raw_events),
        "newly_classified": newly_classified,
        "already_classified_skipped": already_classified_skipped,
        "buckets": buckets,
    }
