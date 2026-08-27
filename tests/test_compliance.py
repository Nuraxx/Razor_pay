"""
Day-12 tests: the deterministic compliance gate (policy/compliance.py).

Pure-function tests -- ComplianceContext in, ComplianceResult out, no DB,
no policy call, no LLM call. Integration with the rest of the Day-12 flow
is covered by tests/test_orchestrator.py's failure matrix.
"""
from datetime import datetime, timedelta

from policy.compliance import COMPLIANCE_RULE_VERSION, ComplianceContext, evaluate_compliance
from policy.contact_hours import ContactHoursConfig
from policy.decision_engine import NO_ACTION
from policy.guardrails import MAX_CANDIDATE_HORIZON_DAYS, MAX_RETRY_ATTEMPTS

FAILURE_TS = datetime(2026, 2, 24, 10, 0, 0)
VALID_CANDIDATE_DT = FAILURE_TS + timedelta(days=1)


def _valid_context(**overrides) -> ComplianceContext:
    base = dict(
        classification_bucket="retryable_soft",
        selected_candidate_type="plus_1_day_morning",
        selected_candidate_datetime=VALID_CANDIDATE_DT,
        failure_timestamp=FAILURE_TS,
        attempts_so_far=0,
    )
    base.update(overrides)
    return ComplianceContext(**base)


class TestPaymentActionGate:
    def test_fully_valid_context_allows_payment(self):
        result = evaluate_compliance(_valid_context())
        assert result.payment_action_allowed is True
        assert result.allowed is True  # brief's minimal {allowed, reason, rule_version} alias
        assert result.reason == result.payment_reason

    def test_classification_not_retryable_soft_blocks_payment(self):
        for bucket in ("hard_decline", "customer_cancelled", "unmapped"):
            result = evaluate_compliance(_valid_context(classification_bucket=bucket))
            assert result.payment_action_allowed is False
            assert "classification_not_retryable_soft" in result.payment_reason

    def test_policy_selected_no_action_blocks_payment(self):
        result = evaluate_compliance(_valid_context(selected_candidate_type=NO_ACTION, selected_candidate_datetime=None))
        assert result.payment_action_allowed is False
        assert result.payment_reason == "policy_selected_no_action"

    def test_max_retry_attempts_blocks_payment(self):
        result = evaluate_compliance(_valid_context(attempts_so_far=MAX_RETRY_ATTEMPTS))
        assert result.payment_action_allowed is False
        assert "max_retry_attempts_reached" in result.payment_reason

    def test_attempts_below_max_allows_payment(self):
        result = evaluate_compliance(_valid_context(attempts_so_far=MAX_RETRY_ATTEMPTS - 1))
        assert result.payment_action_allowed is True

    def test_candidate_not_after_failure_blocks_payment(self):
        result = evaluate_compliance(_valid_context(selected_candidate_datetime=FAILURE_TS - timedelta(hours=1)))
        assert result.payment_action_allowed is False
        assert "candidate_not_after_failure" in result.payment_reason

    def test_candidate_beyond_horizon_blocks_payment(self):
        beyond = FAILURE_TS + timedelta(days=MAX_CANDIDATE_HORIZON_DAYS + 1)
        result = evaluate_compliance(_valid_context(selected_candidate_datetime=beyond))
        assert result.payment_action_allowed is False
        assert "recovery_horizon" in result.payment_reason

    def test_missing_candidate_datetime_blocks_payment(self):
        result = evaluate_compliance(_valid_context(selected_candidate_datetime=None))
        assert result.payment_action_allowed is False
        assert result.payment_reason == "missing_candidate_datetime"

    def test_duplicate_payment_blocks_payment(self):
        result = evaluate_compliance(_valid_context(payment_already_decided=True))
        assert result.payment_action_allowed is False
        assert result.payment_reason == "duplicate_payment_action_blocked"

    def test_required_fields_missing_blocks_payment(self):
        result = evaluate_compliance(_valid_context(required_fields_present=False))
        assert result.payment_action_allowed is False
        assert result.payment_reason == "required_fields_missing"


class TestCommunicationActionGate:
    def test_fully_valid_context_allows_communication(self):
        result = evaluate_compliance(_valid_context())
        assert result.communication_action_allowed is True

    def test_customer_cancelled_bucket_auto_blocks_communication(self):
        result = evaluate_compliance(_valid_context(classification_bucket="customer_cancelled", selected_candidate_type=NO_ACTION, selected_candidate_datetime=None))
        assert result.communication_action_allowed is False
        assert "opted_out_or_cancelled" in result.communication_reason

    def test_explicit_opt_out_blocks_communication_independent_of_payment(self):
        # brief section 3's exact example: payment_retry = allowed, outreach = blocked
        result = evaluate_compliance(_valid_context(customer_opted_out=True))
        assert result.payment_action_allowed is True
        assert result.communication_action_allowed is False
        assert "opted_out_or_cancelled" in result.communication_reason

    def test_missing_consent_blocks_communication_only(self):
        result = evaluate_compliance(_valid_context(consent_for_communication=False))
        assert result.payment_action_allowed is True
        assert result.communication_action_allowed is False
        assert result.communication_reason == "consent_for_communication_missing"

    def test_duplicate_communication_blocks_communication_only(self):
        result = evaluate_compliance(_valid_context(communication_already_sent=True))
        assert result.payment_action_allowed is True
        assert result.communication_action_allowed is False
        assert result.communication_reason == "duplicate_communication_action_blocked"

    def test_required_fields_missing_blocks_both_actions(self):
        result = evaluate_compliance(_valid_context(required_fields_present=False))
        assert result.payment_action_allowed is False
        assert result.communication_action_allowed is False


class TestContactHoursGate:
    """Final pre-submission correction: proves the contact-hours gate
    (policy/contact_hours.py) is actually wired into evaluate_compliance's
    communication gate -- the README claimed this gate existed before it
    was implemented. Uses an injected `contact_hours_config` for fully
    deterministic tests (never depends on real-world "now")."""

    IST_9_TO_21 = ContactHoursConfig(enabled=True, timezone_name="Asia/Kolkata", start=datetime(2000, 1, 1, 9, 0).time(), end=datetime(2000, 1, 1, 21, 0).time())

    def test_candidate_outside_contact_hours_blocks_communication_only(self):
        # 2026-02-25 18:00 UTC = 23:30 IST -- outside [09:00, 21:00) IST.
        outside_hours_dt = datetime(2026, 2, 25, 18, 0, 0)
        result = evaluate_compliance(_valid_context(selected_candidate_datetime=outside_hours_dt), self.IST_9_TO_21)
        assert result.communication_action_allowed is False
        assert "outside_contact_hours" in result.communication_reason
        # payment (backend retry) is NOT scoped by contact hours -- see
        # policy/compliance.py's docstring for why.
        assert result.payment_action_allowed is True

    def test_candidate_inside_contact_hours_allows_communication(self):
        # 2026-02-25 10:00 UTC = 15:30 IST -- inside the window.
        inside_hours_dt = datetime(2026, 2, 25, 10, 0, 0)
        result = evaluate_compliance(_valid_context(selected_candidate_datetime=inside_hours_dt), self.IST_9_TO_21)
        assert result.communication_action_allowed is True

    def test_contact_hours_block_sets_deferred_until_the_next_window(self):
        # DEFER, DON'T TERMINATE (final pre-submission audit): a pure
        # contact-hours block must carry the next window's start, not just a
        # dead-end "blocked" verdict.
        outside_hours_dt = datetime(2026, 2, 25, 18, 0, 0)  # 23:30 IST
        result = evaluate_compliance(_valid_context(selected_candidate_datetime=outside_hours_dt), self.IST_9_TO_21)
        assert result.communication_deferred_until == datetime(2026, 2, 26, 3, 30, 0)  # 2026-02-26 09:00 IST

    def test_opt_out_block_never_sets_deferred_until(self):
        # A non-timing block (opt-out) must NOT get a deferred-until time --
        # a later retry can never fix an opt-out.
        result = evaluate_compliance(_valid_context(customer_opted_out=True))
        assert result.communication_action_allowed is False
        assert result.communication_deferred_until is None

    def test_communication_allowed_has_no_deferred_until(self):
        result = evaluate_compliance(_valid_context())
        assert result.communication_action_allowed is True
        assert result.communication_deferred_until is None

    def test_no_candidate_datetime_does_not_block_on_contact_hours(self):
        # NO_ACTION path: nothing scheduled, nothing to check -- must not be
        # blocked by a contact-hours reason (it's already blocked for the
        # real reason, policy_selected_no_action / analogous).
        result = evaluate_compliance(
            _valid_context(selected_candidate_type=NO_ACTION, selected_candidate_datetime=None), self.IST_9_TO_21,
        )
        assert "contact_hours" not in result.communication_reason

    def test_disabled_gate_never_blocks_on_contact_hours(self):
        outside_hours_dt = datetime(2026, 2, 25, 18, 0, 0)
        disabled = ContactHoursConfig(enabled=False, timezone_name="Asia/Kolkata", start=self.IST_9_TO_21.start, end=self.IST_9_TO_21.end)
        result = evaluate_compliance(_valid_context(selected_candidate_datetime=outside_hours_dt), disabled)
        assert result.communication_action_allowed is True

    def test_default_contact_hours_config_is_used_when_none_injected(self):
        # No config passed -- must fall back to app/config.py::settings
        # (policy/contact_hours.py::default_contact_hours_config), not raise
        # and not silently skip the check.
        outside_hours_dt = datetime(2026, 2, 25, 18, 0, 0)
        result = evaluate_compliance(_valid_context(selected_candidate_datetime=outside_hours_dt))
        assert result.communication_action_allowed is False
        assert "outside_contact_hours" in result.communication_reason


class TestResultShapeAndVersioning:
    def test_rule_version_is_recorded(self):
        result = evaluate_compliance(_valid_context())
        assert result.rule_version == COMPLIANCE_RULE_VERSION

    def test_to_dict_contains_minimal_and_extended_shape(self):
        result = evaluate_compliance(_valid_context())
        d = result.to_dict()
        for key in ("allowed", "reason", "rule_version", "payment_action_allowed", "payment_reason", "communication_action_allowed", "communication_reason"):
            assert key in d

    def test_payment_and_communication_gates_are_independent(self):
        # payment blocked (max attempts) but communication still allowed
        result = evaluate_compliance(_valid_context(attempts_so_far=MAX_RETRY_ATTEMPTS))
        assert result.payment_action_allowed is False
        assert result.communication_action_allowed is True
