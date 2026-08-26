"""
Track-03 tests: policy/checkout_rules.py -- pure eligibility + candidate
selection, no DB. Full orchestration/idempotency/compliance-block/LLM-failure
coverage lives in tests/test_revenue_orchestrator.py; this file is the rule
module in isolation, mirroring tests/test_classification_rules.py's style.
"""
from policy.checkout_rules import (
    ABANDONED,
    CANDIDATE_ALTERNATE_PAYMENT_METHOD,
    CANDIDATE_PAYMENT_LINK_REMINDER,
    CANDIDATE_REMINDER,
    CANDIDATE_RETRY_CHECKOUT,
    CANDIDATE_WAIT,
    CHECKOUT_CANDIDATE_TYPES,
    CHECKOUT_STALLED,
    CHECKOUT_STARTED,
    MAX_OUTREACH_ATTEMPTS,
    MIN_CART_AMOUNT_RS,
    MIN_INACTIVITY_MINUTES,
    RULE_VERSION,
    STALL_TO_ABANDON_MINUTES,
    decide_checkout_recovery,
)
from policy.decision_engine import NO_ACTION


class TestEligibilityThresholds:
    def test_cart_below_minimum_is_not_eligible(self):
        result = decide_checkout_recovery(cart_amount=MIN_CART_AMOUNT_RS - 1, inactivity_minutes=1000)
        assert result.recovery_eligible is False
        assert result.state == CHECKOUT_STARTED
        assert result.candidate_type == NO_ACTION
        assert "cart_amount_below_minimum" in result.eligibility_reason

    def test_cart_at_minimum_passes_the_amount_check(self):
        result = decide_checkout_recovery(cart_amount=MIN_CART_AMOUNT_RS, inactivity_minutes=1000)
        assert result.eligibility_reason != "cart_amount_below_minimum"

    def test_inactivity_below_minimum_is_too_early(self):
        result = decide_checkout_recovery(cart_amount=500, inactivity_minutes=MIN_INACTIVITY_MINUTES - 1)
        assert result.recovery_eligible is False
        assert result.state == CHECKOUT_STARTED
        assert result.candidate_type == NO_ACTION

    def test_stalled_between_thresholds_gets_wait_candidate(self):
        result = decide_checkout_recovery(cart_amount=500, inactivity_minutes=MIN_INACTIVITY_MINUTES + 1)
        assert result.state == CHECKOUT_STALLED
        assert result.recovery_eligible is False
        assert result.candidate_type == CANDIDATE_WAIT

    def test_past_stall_threshold_becomes_abandoned_and_eligible(self):
        result = decide_checkout_recovery(cart_amount=500, inactivity_minutes=STALL_TO_ABANDON_MINUTES)
        assert result.state == ABANDONED
        assert result.recovery_eligible is True

    def test_max_outreach_attempts_blocks_further_outreach(self):
        result = decide_checkout_recovery(cart_amount=500, inactivity_minutes=1000, previous_outreach_count=MAX_OUTREACH_ATTEMPTS)
        assert result.recovery_eligible is False
        assert result.state == ABANDONED
        assert result.candidate_type == NO_ACTION
        assert "max_outreach_attempts_reached" in result.eligibility_reason


class TestCandidateSelectionNoDuplicates:
    def test_first_touch_is_reminder(self):
        result = decide_checkout_recovery(cart_amount=500, inactivity_minutes=1000, previous_outreach_count=0)
        assert result.candidate_type == CANDIDATE_REMINDER

    def test_second_touch_without_payment_method_is_payment_link_reminder(self):
        result = decide_checkout_recovery(cart_amount=500, inactivity_minutes=1000, previous_outreach_count=1, payment_method=None)
        assert result.candidate_type == CANDIDATE_PAYMENT_LINK_REMINDER

    def test_second_touch_with_known_payment_method_is_alternate_payment_method(self):
        result = decide_checkout_recovery(cart_amount=500, inactivity_minutes=1000, previous_outreach_count=1, payment_method="card")
        assert result.candidate_type == CANDIDATE_ALTERNATE_PAYMENT_METHOD

    def test_third_touch_is_retry_checkout(self):
        result = decide_checkout_recovery(cart_amount=500, inactivity_minutes=1000, previous_outreach_count=2)
        assert result.candidate_type == CANDIDATE_RETRY_CHECKOUT

    def test_candidate_always_in_allowed_set(self):
        for count in range(MAX_OUTREACH_ATTEMPTS + 1):
            result = decide_checkout_recovery(cart_amount=500, inactivity_minutes=1000, previous_outreach_count=count)
            assert result.candidate_type in CHECKOUT_CANDIDATE_TYPES


class TestVersioning:
    def test_rule_version_recorded(self):
        result = decide_checkout_recovery(cart_amount=500, inactivity_minutes=1000)
        assert result.rule_version == RULE_VERSION

    def test_deterministic_same_input_same_output(self):
        kwargs = dict(cart_amount=500, inactivity_minutes=1000, previous_outreach_count=1, payment_method="upi")
        assert decide_checkout_recovery(**kwargs) == decide_checkout_recovery(**kwargs)
