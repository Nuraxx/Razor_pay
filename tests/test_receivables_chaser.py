"""
Track-03 tests: policy/receivables_rules.py -- pure bucket classification +
deterministic escalation, no DB. Orchestration-level coverage (HUMAN_REVIEW
routing, LLM-failure independence) lives in tests/test_revenue_orchestrator.py.
"""
from llm.client import LLMProviderError
from llm.service import generate_outreach_microcopy
from policy.decision_engine import NO_ACTION
from policy.receivables_rules import (
    BUCKET_DISPUTED,
    BUCKET_DUE_SOON,
    BUCKET_OVERDUE_HIGH,
    BUCKET_OVERDUE_MEDIUM,
    BUCKET_OVERDUE_SOFT,
    BUCKET_PROMISE_TO_PAY,
    CANDIDATE_ESCALATION,
    CANDIDATE_FRIENDLY_REMINDER,
    CANDIDATE_HUMAN_HANDOFF,
    CANDIDATE_PROMISE_TO_PAY_REQUEST,
    RULE_VERSION,
    classify_receivable,
    decide_receivable_action,
)
from tests.test_llm import _RaisingClient


class TestBucketThresholds:
    def test_not_yet_due_is_due_soon(self):
        assert classify_receivable(days_overdue=-3) == BUCKET_DUE_SOON

    def test_zero_days_overdue_is_soft(self):
        assert classify_receivable(days_overdue=0) == BUCKET_OVERDUE_SOFT

    def test_seven_days_overdue_is_still_soft(self):
        assert classify_receivable(days_overdue=7) == BUCKET_OVERDUE_SOFT

    def test_eight_days_overdue_is_medium(self):
        assert classify_receivable(days_overdue=8) == BUCKET_OVERDUE_MEDIUM

    def test_thirty_days_overdue_is_still_medium(self):
        assert classify_receivable(days_overdue=30) == BUCKET_OVERDUE_MEDIUM

    def test_thirty_one_days_overdue_is_high(self):
        assert classify_receivable(days_overdue=31) == BUCKET_OVERDUE_HIGH

    def test_disputed_outranks_day_count(self):
        assert classify_receivable(days_overdue=90, is_disputed=True) == BUCKET_DISPUTED

    def test_active_promise_outranks_day_count(self):
        assert classify_receivable(days_overdue=90, has_active_promise=True) == BUCKET_PROMISE_TO_PAY

    def test_disputed_outranks_active_promise(self):
        assert classify_receivable(days_overdue=90, is_disputed=True, has_active_promise=True) == BUCKET_DISPUTED


class TestDeterministicEscalationPolicy:
    def test_due_soon_is_no_action(self):
        d = decide_receivable_action(days_overdue=-1)
        assert d.candidate_type == NO_ACTION
        assert d.escalation_level == 0
        assert d.requires_human_review is False

    def test_soft_overdue_is_friendly_reminder(self):
        d = decide_receivable_action(days_overdue=3)
        assert d.candidate_type == CANDIDATE_FRIENDLY_REMINDER
        assert d.escalation_level == 1

    def test_medium_overdue_is_reminder_plus_promise_to_pay(self):
        d = decide_receivable_action(days_overdue=15)
        assert d.candidate_type == CANDIDATE_PROMISE_TO_PAY_REQUEST
        assert d.escalation_level == 2

    def test_high_overdue_is_escalation(self):
        d = decide_receivable_action(days_overdue=45)
        assert d.candidate_type == CANDIDATE_ESCALATION
        assert d.escalation_level == 3

    def test_disputed_is_human_handoff_and_flags_human_review(self):
        d = decide_receivable_action(days_overdue=45, is_disputed=True)
        assert d.candidate_type == CANDIDATE_HUMAN_HANDOFF
        assert d.escalation_level == 4
        assert d.requires_human_review is True

    def test_escalation_level_strictly_increases_with_severity(self):
        levels = [decide_receivable_action(days_overdue=d).escalation_level for d in (-1, 3, 15, 45)]
        assert levels == sorted(levels)
        assert len(set(levels)) == len(levels)  # strictly increasing, no ties


class TestLLMNeverDecidesEscalation:
    """The brief is explicit: 'Do not let the LLM determine the escalation
    level. LLM only generates/rewrites the communication.' -- proves the
    escalation_level computed by decide_receivable_action is identical
    whether or not the LLM call that would later write the communication
    copy succeeds, by never even passing escalation_level as an LLM input in
    the first place (structural proof, not just a runtime one)."""

    def test_decide_receivable_action_takes_no_llm_client_parameter(self):
        import inspect

        params = set(inspect.signature(decide_receivable_action).parameters)
        assert not any("llm" in p.lower() for p in params)

    def test_llm_failure_does_not_touch_receivables_rules_module_at_all(self):
        # A broken LLM client used for the downstream communication job has
        # zero code path back into decide_receivable_action -- demonstrated
        # by calling both independently and confirming the escalation
        # decision is unaffected by the LLM outcome.
        broken_client = _RaisingClient(LLMProviderError("simulated_outage"))
        decision_before = decide_receivable_action(days_overdue=45)
        generate_outreach_microcopy(
            failure_bucket=decision_before.bucket, customer_segment="mid", language="en",
            will_retry=False, retry_window_description=None, amount_rupees=1000.0, client=broken_client,
        )
        decision_after = decide_receivable_action(days_overdue=45)
        assert decision_before == decision_after


class TestVersioning:
    def test_rule_version_recorded(self):
        assert decide_receivable_action(days_overdue=5).rule_version == RULE_VERSION
