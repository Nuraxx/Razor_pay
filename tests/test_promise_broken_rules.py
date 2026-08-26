"""Track-03 tests: policy/promise_broken_rules.py -- pure, no DB."""
from policy.decision_engine import NO_ACTION
from policy.promise_broken_rules import (
    CANDIDATE_FINAL_NOTICE,
    CANDIDATE_URGENT_REMINDER,
    MAX_BROKEN_PROMISE_ATTEMPTS,
    RULE_VERSION,
    decide_promise_broken_action,
)


class TestBoundedByOriginalEventAttempts:
    def test_cumulative_payment_attempts_exhausted_forces_no_action(self):
        d = decide_promise_broken_action(attempts_so_far=0, cumulative_payment_attempts=3, max_payment_attempts=3)
        assert d.candidate_type == NO_ACTION
        assert "cumulative_payment_attempts_exhausted" in d.reason

    def test_cannot_bypass_original_attempts_cap_even_with_zero_broken_promise_attempts(self):
        # the whole point: a broken promise must never squeeze out one more
        # attempt once the ORIGINAL event already exhausted its own cap.
        d = decide_promise_broken_action(attempts_so_far=0, cumulative_payment_attempts=5, max_payment_attempts=3)
        assert d.candidate_type == NO_ACTION


class TestOwnSmallerCap:
    def test_first_broken_promise_is_urgent_reminder(self):
        d = decide_promise_broken_action(attempts_so_far=0, cumulative_payment_attempts=0, max_payment_attempts=3)
        assert d.candidate_type == CANDIDATE_URGENT_REMINDER

    def test_second_broken_promise_is_final_notice(self):
        d = decide_promise_broken_action(attempts_so_far=1, cumulative_payment_attempts=0, max_payment_attempts=3)
        assert d.candidate_type == CANDIDATE_FINAL_NOTICE

    def test_at_max_broken_promise_attempts_is_no_action(self):
        d = decide_promise_broken_action(attempts_so_far=MAX_BROKEN_PROMISE_ATTEMPTS, cumulative_payment_attempts=0, max_payment_attempts=3)
        assert d.candidate_type == NO_ACTION
        assert "max_broken_promise_attempts_reached" in d.reason


class TestVersioning:
    def test_rule_version_recorded(self):
        d = decide_promise_broken_action(attempts_so_far=0, cumulative_payment_attempts=0, max_payment_attempts=3)
        assert d.rule_version == RULE_VERSION
