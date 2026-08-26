"""
Track-03: generalized per-intervention economics for the new revenue-risk
domains. Kept fully SEPARATE from policy/costs.py::cost_for_candidate (which
intentionally raises outside its fixed 5-candidate payment-retry list -- that
guard stays untouched, it must never silently accept a domain candidate type
it was never designed for).

SYNTHETIC / PROJECT ASSUMPTIONS, NOT REAL PRICING -- same disclosure as
policy/costs.py. `reminder_cost` reuses the project's own disclosed 2026
WhatsApp anchor rate (policy/costs.py::InterventionCosts.whatsapp_cost);
`escalation_cost`/`human_handoff_cost`/`retry_cost` are illustrative
placeholders for this buildathon demo, not sourced from any real pricing.
"""
from __future__ import annotations

from dataclasses import dataclass

from policy.costs import InterventionCosts
from policy.decision_engine import NO_ACTION


@dataclass(frozen=True)
class RevenueInterventionCosts:
    """All figures in Rs."""

    reminder_cost: float = InterventionCosts().whatsapp_cost  # reuses the project's disclosed 2026 anchor rate
    escalation_cost: float = 25.0  # illustrative -- a human-adjacent escalation touch
    human_handoff_cost: float = 75.0  # illustrative -- a full human review/contact
    retry_cost: float = InterventionCosts().retry_cost  # mirrors the existing automated-retry cost, for mandate attempt steps


DEFAULT_REVENUE_COSTS = RevenueInterventionCosts()

# candidate_type -> RevenueInterventionCosts attribute name, or None for a free action.
_CANDIDATE_COST_ATTR: dict[str, str | None] = {
    # checkout
    "reminder": "reminder_cost", "payment_link_reminder": "reminder_cost",
    "retry_checkout": "retry_cost", "alternate_payment_method": "reminder_cost", "wait": None,
    # mandate
    "attempt_1": "retry_cost", "attempt_2": "retry_cost", "final_attempt": "retry_cost",
    "communication": "reminder_cost", "alternate_window": None, "escalation": "escalation_cost",
    # receivables
    "friendly_reminder": "reminder_cost", "payment_request": "reminder_cost",
    "promise_to_pay_request": "reminder_cost", "human_handoff": "human_handoff_cost",
}


def cost_for_domain_candidate(candidate_type: str, costs: RevenueInterventionCosts = DEFAULT_REVENUE_COSTS) -> float:
    """Returns 0.0 for NO_ACTION/wait/alternate_window (free -- nothing is
    sent), and for any candidate_type this module doesn't recognize (fails
    open to zero cost rather than raising, unlike policy/costs.py's stricter
    guard -- appropriate here since this module's candidate vocabulary is
    still growing across 3 domains and a missing entry must never crash
    economics reporting)."""
    if candidate_type == NO_ACTION:
        return 0.0
    attr = _CANDIDATE_COST_ATTR.get(candidate_type)
    if attr is None:
        return 0.0
    return getattr(costs, attr)


def expected_net_value(
    *, at_risk_amount: float, recovery_probability: float, candidate_type: str,
    costs: RevenueInterventionCosts = DEFAULT_REVENUE_COSTS,
) -> float:
    """expected_recovery_value - intervention_cost -- same net-value shape
    policy/decision_engine.py already uses for payment_failed, generalized
    to a caller-supplied recovery_probability (these new domains have no
    trained model to supply one, so it's always an explicit input here, never
    invented internally)."""
    if not 0.0 <= recovery_probability <= 1.0:
        raise ValueError(f"recovery_probability must be in [0, 1]: {recovery_probability}")
    expected_recovery_value = at_risk_amount * recovery_probability
    return expected_recovery_value - cost_for_domain_candidate(candidate_type, costs)
