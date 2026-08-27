"""
CLI: classify existing raw_events into failure_events without needing
a live Razorpay webhook — reads directly from the configured DATABASE_URL.

Usage (from the project root):

    ./venv/bin/python -m scripts.classify_raw_events
        Classify every raw_event that doesn't already have a failure_events
        row. Safe to re-run — already-classified rows are skipped.

    ./venv/bin/python -m scripts.classify_raw_events --raw-event-id 42
        Classify (or report the existing classification of) a single
        raw_events.id.
"""
import argparse

from app.db import SessionLocal, init_db
from app.models import RawEvent
from classification.service import classify_all_raw_events, classify_raw_event


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify raw_events into failure_events (deterministic, idempotent)."
    )
    parser.add_argument(
        "--raw-event-id",
        type=int,
        default=None,
        help="Classify only this one raw_events.id instead of every unclassified raw_event.",
    )
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        if args.raw_event_id is not None:
            raw_event = db.query(RawEvent).filter(RawEvent.id == args.raw_event_id).first()
            if raw_event is None:
                print(f"No raw_events row with id={args.raw_event_id}")
                return
            failure_event, created = classify_raw_event(db, raw_event)
            verb = "Classified" if created else "Already classified (skipped)"
            print(
                f"{verb}: raw_events.id={raw_event.id} -> "
                f"bucket={failure_event.classification_bucket} "
                f"confidence={failure_event.classification_confidence} "
                f"rule_version={failure_event.rule_version}"
            )
        else:
            summary = classify_all_raw_events(db)
            print(
                f"Processed {summary['total_raw_events']} raw_events: "
                f"{summary['newly_classified']} newly classified, "
                f"{summary['already_classified_skipped']} already classified (skipped)."
            )
            print("Bucket counts:")
            for bucket, count in sorted(summary["buckets"].items()):
                print(f"  {bucket}: {count}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
