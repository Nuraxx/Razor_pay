"""
FIX #2: manual reprocessing CLI. Normally unnecessary -- app/main.py's
webhook handler now classifies and orchestrates automatically -- but this
is the "retry/reprocessing must remain possible" path for a raw_event whose
automatic orchestration failed (see app/main.py's `orchestration_failed_after_storage`
audit action) or that was stored before this fix existed.

Safe to re-run over the same raw_events any number of times: classification
and orchestration are each independently idempotent (query-before-act,
keyed on raw_event_id / event_id) -- an already-fully-processed event is
simply skipped with no duplicate action of any kind.

Usage (from the project root):

    ./venv/bin/python -m scripts.reprocess_raw_events
        Reprocess every raw_event (classify if not yet classified,
        orchestrate if not yet decided).

    ./venv/bin/python -m scripts.reprocess_raw_events --raw-event-id 42
        Reprocess a single raw_events.id.
"""
import argparse

from app.db import SessionLocal, init_db
from app.models import RawEvent
from recovery.webhook_pipeline import process_raw_event


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reprocess raw_events through classification + orchestration (idempotent, safe to re-run)."
    )
    parser.add_argument("--raw-event-id", type=int, default=None, help="Reprocess only this one raw_events.id.")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        if args.raw_event_id is not None:
            raw_event = db.query(RawEvent).filter(RawEvent.id == args.raw_event_id).first()
            if raw_event is None:
                print(f"No raw_events row with id={args.raw_event_id}")
                return
            outcome = process_raw_event(db, raw_event)
            print(f"raw_events.id={raw_event.id}: {outcome}")
        else:
            raw_events = db.query(RawEvent).order_by(RawEvent.id).all()
            outcomes: dict[str, int] = {}
            for raw_event in raw_events:
                try:
                    outcome = process_raw_event(db, raw_event)
                except Exception as exc:  # noqa: BLE001 -- one bad row must not stop the whole reprocessing run
                    db.rollback()
                    outcome = f"failed: {type(exc).__name__}"
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
            print(f"Reprocessed {len(raw_events)} raw_events:")
            for outcome, count in sorted(outcomes.items()):
                print(f"  {outcome}: {count}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
