"""
Track-03: manual sweep CLI for the promise-to-pay lifecycle -- detects
promises whose promised_date has passed with no fulfillment signal, marks
them BROKEN, opens a revenue_risk_events row for each, and routes that new
event through the REAL recovery engine (recovery/revenue_orchestrator.py) so
a broken promise concretely "feeds back into the recovery engine" (brief
section 6) rather than just sitting as an inert row.

The detect-then-orchestrate glue lives in ONE place --
recovery/promise_sweep.py::sweep_and_orchestrate_broken_promises -- shared
with recovery/scheduler.py's automatic background loop, so this CLI and that
scheduler can never drift into two different broken-promise detectors.

Mirrors scripts/reprocess_raw_events.py's shape: safe to re-run any number
of times -- mark_broken_promises is idempotent (an already-resolved promise
is skipped), and orchestrate_revenue_event is idempotent by event_id (a
revenue_risk_events row that already has a policy_decisions row is not
re-decided or re-communicated).

Usage (from the project root):

    ./venv/bin/python -m scripts.sweep_promise_lifecycle
        Sweep every promise whose promised_date has passed as of now, and
        orchestrate a recovery action for each newly-broken one.
"""
import argparse
from datetime import datetime

from app.db import SessionLocal, init_db
from model.unified_model import get_live_unified_model
from recovery.promise_sweep import sweep_and_orchestrate_broken_promises


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep promises_to_pay for broken (unfulfilled, past-date) promises, then orchestrate each one.")
    parser.add_argument("--as-of", type=str, default=None, help="ISO-8601 timestamp to sweep as of (default: now).")
    args = parser.parse_args()

    as_of = datetime.fromisoformat(args.as_of) if args.as_of else None

    init_db()
    db = SessionLocal()
    try:
        results = sweep_and_orchestrate_broken_promises(db, as_of=as_of, model=get_live_unified_model())
        print(f"Swept promise lifecycle: {len(results)} newly-broken promise(s) orchestrated.")
        for result in results:
            print(f"  revenue_risk_events.id={result.event_id} -> candidate={result.selected_candidate_type} final_status={result.final_status}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
