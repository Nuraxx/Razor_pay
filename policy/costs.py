"""
Day-9 intervention-cost configuration -- the ONE place cost numbers live.

`policy/decision_engine.py` never hard-codes a cost; every cost lookup goes
through `cost_for_candidate()` below, which reads from an `InterventionCosts`
instance (defaults to `DEFAULT_COSTS`, overridable per call for what-if
analysis without touching policy logic).

IMPORTANT -- SYNTHETIC / PROJECT ASSUMPTIONS, NOT RAZORPAY PRODUCTION PRICES:
every number in `DEFAULT_COSTS` is a placeholder this project chose for
illustrative purposes. It is not sourced from any real payment-gateway
pricing, SMS/WhatsApp/voice vendor rate card, or Razorpay cost structure.
A real deployment would replace these with actual metered costs.

Only `retry_cost` is used by Day 9's candidate actions -- all 5 candidate
types (`immediate`, `plus_1_day_morning`, `payday_window`, `plus_3_days`,
`month_end_window`) are automated payment retries, so they share one cost.
`whatsapp_cost` / `sms_cost` / `voice_cost` / `operational_cost` are
configured here so a later day's communication channels can be priced
without another schema/config change, but nothing in this project selects
those channels yet -- no LLM/WhatsApp/voice logic exists (explicitly out of
scope through Day 9, same as every prior day).
"""
from __future__ import annotations

from dataclasses import dataclass

from policy.retry_candidates import CANDIDATE_TYPES


@dataclass(frozen=True)
class InterventionCosts:
    """All figures in Rs. Synthetic project assumptions -- see module docstring."""

    retry_cost: float = 5.0  # cost of one automated payment-gateway retry attempt
    whatsapp_cost: float = 0.0  # placeholder -- no WhatsApp channel implemented yet
    sms_cost: float = 0.0  # placeholder -- no SMS channel implemented yet
    voice_cost: float = 0.0  # placeholder -- no voice channel implemented yet
    operational_cost: float = 0.0  # placeholder -- fixed per-decision overhead, not yet modeled


DEFAULT_COSTS = InterventionCosts()


def cost_for_candidate(candidate_type: str, costs: InterventionCosts = DEFAULT_COSTS) -> float:
    """Every one of the 5 candidate retry actions is an automated retry --
    `retry_cost` plus any fixed `operational_cost`. Raises on an unknown
    candidate_type rather than silently defaulting to zero cost."""
    if candidate_type not in CANDIDATE_TYPES:
        raise ValueError(f"unknown candidate_type: {candidate_type!r}")
    return costs.retry_cost + costs.operational_cost
