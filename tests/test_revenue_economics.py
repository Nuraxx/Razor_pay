"""Track-03 tests: policy/revenue_economics.py -- pure cost/net-value math, no DB."""
import pytest

from policy.decision_engine import NO_ACTION
from policy.revenue_economics import (
    DEFAULT_REVENUE_COSTS,
    RevenueInterventionCosts,
    cost_for_domain_candidate,
    expected_net_value,
)


class TestCostForDomainCandidate:
    def test_no_action_is_free(self):
        assert cost_for_domain_candidate(NO_ACTION) == 0.0

    def test_wait_is_free(self):
        assert cost_for_domain_candidate("wait") == 0.0

    def test_reminder_uses_reminder_cost(self):
        assert cost_for_domain_candidate("reminder") == DEFAULT_REVENUE_COSTS.reminder_cost

    def test_escalation_uses_escalation_cost(self):
        assert cost_for_domain_candidate("escalation") == DEFAULT_REVENUE_COSTS.escalation_cost

    def test_human_handoff_uses_human_handoff_cost(self):
        assert cost_for_domain_candidate("human_handoff") == DEFAULT_REVENUE_COSTS.human_handoff_cost

    def test_mandate_attempt_uses_retry_cost(self):
        assert cost_for_domain_candidate("attempt_1") == DEFAULT_REVENUE_COSTS.retry_cost

    def test_unrecognized_candidate_fails_open_to_zero(self):
        assert cost_for_domain_candidate("something_new_not_yet_priced") == 0.0

    def test_custom_costs_object_is_respected(self):
        custom = RevenueInterventionCosts(reminder_cost=1.5)
        assert cost_for_domain_candidate("reminder", custom) == 1.5


class TestExpectedNetValue:
    def test_basic_computation(self):
        value = expected_net_value(at_risk_amount=1000.0, recovery_probability=0.5, candidate_type="reminder")
        assert value == pytest.approx(1000.0 * 0.5 - DEFAULT_REVENUE_COSTS.reminder_cost)

    def test_no_action_has_zero_cost_component(self):
        value = expected_net_value(at_risk_amount=1000.0, recovery_probability=0.0, candidate_type=NO_ACTION)
        assert value == 0.0

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError):
            expected_net_value(at_risk_amount=1000.0, recovery_probability=1.5, candidate_type="reminder")
        with pytest.raises(ValueError):
            expected_net_value(at_risk_amount=1000.0, recovery_probability=-0.1, candidate_type="reminder")
