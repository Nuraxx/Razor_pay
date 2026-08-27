"""
policy-v3 intervention-cost configuration -- the ONE place cost numbers live.

`policy/decision_engine.py` never hard-codes a cost; every cost lookup goes
through `cost_for_candidate()` below, which reads from an `InterventionCosts`
instance (defaults to `DEFAULT_COSTS`, overridable per call for what-if
analysis without touching policy logic).

IMPORTANT -- SYNTHETIC / PROJECT ASSUMPTIONS, NOT RAZORPAY PRODUCTION PRICES:
`retry_cost` remains an illustrative placeholder (not sourced from any real
payment-gateway pricing or Razorpay cost structure) -- a real deployment
would replace it with an actual metered cost.

Only `retry_cost` is used by policy-v3's candidate actions -- all 5 candidate
types (`immediate`, `plus_1_day_morning`, `payday_window`, `plus_3_days`,
`month_end_window`) are automated payment retries, so they share one cost.
`sms_cost` / `voice_cost` / `operational_cost` remain unused placeholders --
no SMS/voice/live-LLM channel is priced by any operational code path in
this project.

BASELINE-FIDELITY FIX -- `whatsapp_cost` IS NOW USED, by the Rule-Based
EVALUATION BASELINE only (`policy/baselines.py::rule_based_baseline`'s
`communication_actions`, consumed by `evaluation/evaluate_decision_engine_v4.py`'s
contact-cost/cost-per-recovery metrics) -- never by the live orchestrator
(`recovery/orchestrator.py` sends real communication through the LLM
layer, an entirely separate code path this cost has no bearing on). Value
sourced directly from the original specification's own disclosed anchor
rate ("Contact cost anchored to real 2026 rates (~₹0.135/WhatsApp utility
message with GST; ~₹0.12-0.30/SMS)") -- not invented, not re-derived.
"""
from __future__ import annotations

from dataclasses import dataclass

from policy.retry_candidates import CANDIDATE_TYPES


@dataclass(frozen=True)
class InterventionCosts:
    """All figures in Rs. Synthetic project assumptions -- see module docstring."""

    retry_cost: float = 5.0  # cost of one automated payment-gateway retry attempt
    whatsapp_cost: float = 0.135  # per spec's disclosed 2026 anchor rate; used by the Rule-Based EVALUATION baseline only -- see module docstring
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


def contact_cost(n_whatsapp_messages: int, costs: InterventionCosts = DEFAULT_COSTS) -> float:
    """Cost of `n_whatsapp_messages` WhatsApp contacts (Rule-Based baseline's
    nudge + follow-up -- see policy/baselines.py). A separate function from
    `cost_for_candidate` because communication is not a retry candidate
    action -- keeping the two costs distinct is exactly what
    policy/economics.py's GMV/cost/fee separation already requires."""
    if n_whatsapp_messages < 0:
        raise ValueError(f"n_whatsapp_messages must be >= 0: {n_whatsapp_messages}")
    return n_whatsapp_messages * costs.whatsapp_cost
