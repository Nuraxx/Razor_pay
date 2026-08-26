"""
Track-03 hardening pass: automatic broken-promise detection, running inside
the FastAPI process itself -- no Celery/Redis/external scheduler, per the
brief's explicit instruction to keep this buildathon-simple.

This module owns exactly ONE concern: run
recovery/promise_sweep.py::sweep_and_orchestrate_broken_promises
periodically, in the background, without blocking app startup, and without
letting one bad promise (or one bad sweep pass) take the whole loop down.
It contains NO detection logic and NO orchestration-routing logic of its own
-- both already exist, unchanged, in recovery/promise_lifecycle.py and
recovery/promise_sweep.py. Duplicating either here would create exactly the
second broken-promise detector the brief says not to build.

Wired into app/main.py's lifespan via asyncio.create_task -- the task starts
after startup completes (never blocks it) and is cancelled cleanly on
shutdown. Cadence and enable/disable are both plain settings
(app/config.py::ENABLE_PROMISE_SWEEP_SCHEDULER /
PROMISE_SWEEP_INTERVAL_SECONDS), so this can be turned off without touching
code -- which is also, incidentally, what already happens for every test:
FastAPI's lifespan never runs for a bare TestClient(app) used without an
explicit `with` block (tests/conftest.py's `client` fixture never enters it),
so this loop never starts during the test suite regardless of the flag.
"""
from __future__ import annotations

import asyncio

from app.db import SessionLocal
from app.logging_config import log
from model.unified_model import get_live_unified_model
from recovery.promise_sweep import sweep_and_orchestrate_broken_promises


def run_promise_sweep_once(db=None) -> int:
    """One sweep pass. `db=None` (the real, running-app case) opens and
    closes its own session, matching every other one-shot DB entry point in
    this codebase. Returns the number of newly-broken promises processed --
    0 on a quiet pass, which is the normal, expected case most of the time."""
    owns_db = db is None
    db = db or SessionLocal()
    try:
        results = sweep_and_orchestrate_broken_promises(db, model=get_live_unified_model())
        return len(results)
    finally:
        if owns_db:
            db.close()


async def promise_sweep_background_loop(interval_seconds: int) -> None:
    """Runs run_promise_sweep_once() forever, sleeping interval_seconds
    between passes. Exception isolation at two levels: a single promise's
    own orchestration failure is already caught inside
    recovery/revenue_orchestrator.py (never propagates here at all); this
    loop additionally guards the sweep call itself (e.g. a transient DB
    error) so that failure is logged and the NEXT cycle still runs, rather
    than silently killing the background task for the lifetime of the process."""
    while True:
        try:
            processed = run_promise_sweep_once()
            if processed:
                log.info("Promise sweep: processed %s newly-broken promise(s)", processed)
        except asyncio.CancelledError:
            raise  # real shutdown -- must propagate, not be swallowed as a "failure"
        except Exception:
            log.exception("Promise sweep: sweep pass failed -- will retry next cycle")
        await asyncio.sleep(interval_seconds)
