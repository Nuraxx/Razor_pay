"""
Track-03 hardening pass: the ONE place that combines
recovery/promise_lifecycle.py::mark_broken_promises (detection) with
recovery/revenue_orchestrator.py::orchestrate_revenue_event (routing a newly-
broken promise through policy/compliance/recovery/communication/audit).

Both scripts/sweep_promise_lifecycle.py (manual CLI) and recovery/scheduler.py
(automatic background loop) call this single function -- neither duplicates
this glue itself, and neither reimplements mark_broken_promises' detection
logic. promise_lifecycle.py itself stays free of any orchestrator dependency,
exactly as its own docstring documents ("kept separate so this module itself
has no dependency on the orchestrator at all") -- this module is the CALLER
that docstring refers to, factored out so there is exactly one such caller,
not two independent copies.
"""
from __future__ import annotations

import json

from sqlalchemy.orm import Session

from app.models import RevenueRiskEvent
from llm.client import LLMClient
from recovery.promise_lifecycle import mark_broken_promises
from recovery.revenue_orchestrator import orchestrate_revenue_event
from recovery.revenue_schemas import RevenueRecoveryResult, RevenueRiskEventInput


def sweep_and_orchestrate_broken_promises(
    db: Session, *, as_of=None, model: dict | None = None, llm_client: LLMClient | None = None,
) -> list[RevenueRecoveryResult]:
    """One full sweep pass: detect newly-broken promises (unchanged
    mark_broken_promises), then route each one through the real orchestrator.
    Safe to call repeatedly -- mark_broken_promises is idempotent (an
    already-resolved promise is skipped, so it can never be detected twice),
    and orchestrate_revenue_event is idempotent by event_id (a
    revenue_risk_events row that already has a policy_decisions row is not
    re-decided or re-communicated) -- so a second sweep over the same
    promises creates zero duplicate events and zero duplicate recovery
    actions. Returns one RevenueRecoveryResult per newly-broken promise
    processed this call (empty list if nothing was newly broken)."""
    broken_outcomes = mark_broken_promises(db, as_of=as_of)
    results: list[RevenueRecoveryResult] = []
    for outcome in broken_outcomes:
        rre = db.query(RevenueRiskEvent).filter(RevenueRiskEvent.id == outcome.reevaluation_event_id).first()
        if rre is None:  # pragma: no cover -- mark_broken_promises always creates this row in the same transaction
            continue
        domain_context = json.loads(rre.context_json or "{}")
        event = RevenueRiskEventInput(
            event_type="promise_to_pay_broken", event_id=rre.id, customer_ref=rre.customer_ref,
            occurred_at=rre.occurred_at, amount=domain_context.get("original_amount", 0.0) or 0.0,
            domain_context=domain_context,
        )
        results.append(orchestrate_revenue_event(db, event, model=model, llm_client=llm_client))
    return results
