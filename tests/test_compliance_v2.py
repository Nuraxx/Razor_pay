"""
Track-03 tests: the generalized compliance gate (policy/compliance_v2.py).

Pure-function tests, no DB. Two concerns:
  1. Delegation correctness -- evaluate_compliance_v2 must be a byte-identical
     superset of evaluate_compliance for payment_failed/subscription_payment_failed,
     never a reimplementation (see TestDelegationIsByteIdentical, which reuses
     every context permutation tests/test_compliance.py already covers).
  2. The new domain path (checkout_abandoned/mandate_failed/receivable_overdue/
     promise_to_pay_broken), including the new HUMAN_REVIEW verdict.
"""
from datetime import datetime, timedelta

import pytest

from policy.compliance import ComplianceContext, evaluate_compliance
from policy.compliance_v2 import (
    COMPLIANCE_V2_RULE_VERSION,
    GeneralizedComplianceContext,
    evaluate_compliance_v2,
)
from policy.contact_hours import ContactHoursConfig
from policy.decision_engine import NO_ACTION
from policy.guardrails import MAX_CANDIDATE_HORIZON_DAYS, MAX_RETRY_ATTEMPTS

FAILURE_TS = datetime(2026, 2, 24, 10, 0, 0)
VALID_CANDIDATE_DT = FAILURE_TS + timedelta(days=1)


def _valid_v2_context(**overrides) -> GeneralizedComplianceContext:
    base = dict(
        event_type="payment_failed",
        classification_bucket="retryable_soft",
        selected_candidate_type="plus_1_day_morning",
        selected_candidate_datetime=VALID_CANDIDATE_DT,
        occurred_at=FAILURE_TS,
        attempts_so_far=0,
    )
    base.update(overrides)
    return GeneralizedComplianceContext(**base)


def _legacy_equivalent(**overrides) -> ComplianceContext:
    base = dict(
        classification_bucket="retryable_soft",
        selected_candidate_type="plus_1_day_morning",
        selected_candidate_datetime=VALID_CANDIDATE_DT,
        failure_timestamp=FAILURE_TS,
        attempts_so_far=0,
    )
    base.update(overrides)
    return ComplianceContext(**base)


# ---------------------------------------------------------------------------
# Delegation correctness for payment_failed / subscription_payment_failed
# ---------------------------------------------------------------------------

class TestDelegationIsByteIdentical:
    """Every one of these mirrors a case from tests/test_compliance.py --
    evaluate_compliance_v2 must produce the exact payment_reason/
    communication_reason strings evaluate_compliance itself does."""

    @pytest.mark.parametrize("event_type", ["payment_failed", "subscription_payment_failed"])
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"classification_bucket": "hard_decline"},
            {"classification_bucket": "customer_cancelled"},
            {"classification_bucket": "unmapped"},
            {"selected_candidate_type": NO_ACTION, "selected_candidate_datetime": None},
            {"attempts_so_far": MAX_RETRY_ATTEMPTS},
            {"attempts_so_far": MAX_RETRY_ATTEMPTS - 1},
            {"selected_candidate_datetime": FAILURE_TS - timedelta(hours=1)},
            {"selected_candidate_datetime": FAILURE_TS + timedelta(days=MAX_CANDIDATE_HORIZON_DAYS + 1)},
            {"selected_candidate_datetime": None},
            {"payment_already_decided": True},
            {"required_fields_present": False},
            {"customer_opted_out": True},
            {"consent_for_communication": False},
            {"communication_already_sent": True},
        ],
    )
    def test_v2_matches_legacy_exactly(self, event_type, kwargs):
        legacy = evaluate_compliance(_legacy_equivalent(**kwargs))
        v2 = evaluate_compliance_v2(_valid_v2_context(event_type=event_type, **kwargs))

        assert v2.payment_action_allowed == legacy.payment_action_allowed
        assert v2.payment_reason == legacy.payment_reason
        assert v2.communication_action_allowed == legacy.communication_action_allowed
        assert v2.communication_reason == legacy.communication_reason
        assert v2.rule_version == legacy.rule_version  # proves true delegation, not relabeling
        assert v2.payment_verdict in ("ALLOWED", "BLOCKED")  # never HUMAN_REVIEW for this event type
        assert v2.communication_verdict in ("ALLOWED", "BLOCKED")

    def test_requires_human_review_flag_is_ignored_for_payment_failed(self):
        # requires_human_review is a new-domain-only concept; the legacy
        # delegation path must never surface it as HUMAN_REVIEW.
        v2 = evaluate_compliance_v2(_valid_v2_context(requires_human_review=True, human_review_reason="should be ignored"))
        assert v2.payment_verdict == "ALLOWED"
        assert v2.communication_verdict == "ALLOWED"


# ---------------------------------------------------------------------------
# New domain path
# ---------------------------------------------------------------------------

def _domain_context(**overrides) -> GeneralizedComplianceContext:
    base = dict(
        event_type="checkout_abandoned",
        classification_bucket="recovery_eligible",
        selected_candidate_type="reminder",
        selected_candidate_datetime=VALID_CANDIDATE_DT,
        occurred_at=FAILURE_TS,
        attempts_so_far=0,
    )
    base.update(overrides)
    return GeneralizedComplianceContext(**base)


class TestNewDomainGates:
    def test_fully_valid_context_allows_both_gates(self):
        result = evaluate_compliance_v2(_domain_context())
        assert result.payment_verdict == "ALLOWED"
        assert result.communication_verdict == "ALLOWED"
        assert result.rule_version == COMPLIANCE_V2_RULE_VERSION

    def test_no_action_candidate_blocks_payment_gate(self):
        result = evaluate_compliance_v2(_domain_context(selected_candidate_type=NO_ACTION, selected_candidate_datetime=None))
        assert result.payment_verdict == "BLOCKED"
        assert result.payment_reason == "policy_selected_no_action"

    def test_max_attempts_blocks_payment_gate(self):
        result = evaluate_compliance_v2(_domain_context(attempts_so_far=MAX_RETRY_ATTEMPTS))
        assert result.payment_verdict == "BLOCKED"
        assert "max_retry_attempts_reached" in result.payment_reason

    def test_candidate_beyond_horizon_blocks_payment_gate(self):
        beyond = FAILURE_TS + timedelta(days=MAX_CANDIDATE_HORIZON_DAYS + 1)
        result = evaluate_compliance_v2(_domain_context(selected_candidate_datetime=beyond))
        assert result.payment_verdict == "BLOCKED"
        assert "recovery_horizon" in result.payment_reason

    def test_duplicate_payment_blocks_payment_gate(self):
        result = evaluate_compliance_v2(_domain_context(payment_already_decided=True))
        assert result.payment_verdict == "BLOCKED"
        assert result.payment_reason == "duplicate_payment_action_blocked"

    def test_required_fields_missing_blocks_both_gates(self):
        result = evaluate_compliance_v2(_domain_context(required_fields_present=False))
        assert result.payment_verdict == "BLOCKED"
        assert result.communication_verdict == "BLOCKED"

    def test_opt_out_blocks_communication_independent_of_payment(self):
        result = evaluate_compliance_v2(_domain_context(customer_opted_out=True))
        assert result.payment_verdict == "ALLOWED"
        assert result.communication_verdict == "BLOCKED"
        assert "opted_out_or_cancelled" in result.communication_reason

    def test_missing_consent_blocks_communication_only(self):
        result = evaluate_compliance_v2(_domain_context(consent_for_communication=False))
        assert result.payment_verdict == "ALLOWED"
        assert result.communication_verdict == "BLOCKED"

    def test_duplicate_communication_blocks_communication_only(self):
        result = evaluate_compliance_v2(_domain_context(communication_already_sent=True))
        assert result.payment_verdict == "ALLOWED"
        assert result.communication_verdict == "BLOCKED"

    def test_candidate_outside_contact_hours_blocks_communication_only(self):
        # Final pre-submission correction: proves the contact-hours gate is
        # wired into the new-domain path too, not just the legacy delegation.
        ist_9_to_21 = ContactHoursConfig(enabled=True, timezone_name="Asia/Kolkata", start=datetime(2000, 1, 1, 9, 0).time(), end=datetime(2000, 1, 1, 21, 0).time())
        outside_hours_dt = datetime(2026, 2, 25, 18, 0, 0)  # 23:30 IST
        result = evaluate_compliance_v2(_domain_context(selected_candidate_datetime=outside_hours_dt), ist_9_to_21)
        assert result.communication_verdict == "BLOCKED"
        assert "outside_contact_hours" in result.communication_reason
        assert result.payment_verdict == "ALLOWED"


class TestHumanReviewVerdict:
    def test_requires_human_review_routes_payment_gate_to_human_review(self):
        result = evaluate_compliance_v2(_domain_context(requires_human_review=True, human_review_reason="disputed_invoice"))
        assert result.payment_verdict == "HUMAN_REVIEW"
        assert result.payment_reason == "disputed_invoice"
        assert result.payment_action_allowed is False  # HUMAN_REVIEW is not ALLOWED

    def test_requires_human_review_routes_communication_gate_to_human_review(self):
        result = evaluate_compliance_v2(_domain_context(requires_human_review=True, human_review_reason="disputed_invoice"))
        assert result.communication_verdict == "HUMAN_REVIEW"
        assert result.communication_action_allowed is False

    def test_human_review_without_explicit_reason_gets_a_default(self):
        result = evaluate_compliance_v2(_domain_context(requires_human_review=True))
        assert result.payment_verdict == "HUMAN_REVIEW"
        assert result.payment_reason == "flagged_for_human_review"

    def test_opt_out_outranks_human_review_on_communication_gate(self):
        # an explicit opt-out is a stronger "never contact" signal than a
        # human-review flag -- the customer must not be contacted regardless
        # of whether a human later reviews the case.
        result = evaluate_compliance_v2(_domain_context(customer_opted_out=True, requires_human_review=True))
        assert result.communication_verdict == "BLOCKED"
        assert "opted_out_or_cancelled" in result.communication_reason

    def test_duplicate_action_outranks_human_review_on_payment_gate(self):
        result = evaluate_compliance_v2(_domain_context(payment_already_decided=True, requires_human_review=True))
        assert result.payment_verdict == "BLOCKED"
        assert result.payment_reason == "duplicate_payment_action_blocked"

    def test_human_review_outranks_no_action_on_payment_gate(self):
        result = evaluate_compliance_v2(_domain_context(selected_candidate_type=NO_ACTION, selected_candidate_datetime=None, requires_human_review=True))
        assert result.payment_verdict == "HUMAN_REVIEW"


class TestMandateCandidateTimingInvariant:
    """Defense-in-depth proof for the mandate retry sequencer: even if
    policy/mandate_rules.py ever produced a same-instant or earlier
    next_action_at (see tests/test_mandate_retry_sequencer.py's
    TestCandidateAlwaysStrictlyAfterNow, including its one documented
    exception for forced escalation), this compliance gate is the
    independent backstop that guarantees such a candidate is NEVER
    ALLOWED -- proven directly against event_type="mandate_failed",
    not just the generic checkout_abandoned default used elsewhere in
    this file."""

    def _mandate_context(self, selected_candidate_datetime) -> GeneralizedComplianceContext:
        return GeneralizedComplianceContext(
            event_type="mandate_failed", classification_bucket="IN_PROGRESS",
            selected_candidate_type="attempt_1", selected_candidate_datetime=selected_candidate_datetime,
            occurred_at=FAILURE_TS, attempts_so_far=0,
        )

    def test_equal_or_earlier_candidate_is_blocked_not_allowed(self):
        for candidate_dt in (FAILURE_TS, FAILURE_TS - timedelta(hours=1), FAILURE_TS - timedelta(microseconds=1)):
            result = evaluate_compliance_v2(self._mandate_context(candidate_dt))
            assert result.payment_verdict == "BLOCKED", f"candidate_dt={candidate_dt} (occurred_at={FAILURE_TS}) must be BLOCKED"
            assert result.payment_reason == "invalid_candidate_time: candidate_not_after_event"
            assert result.payment_action_allowed is False

    def test_one_microsecond_after_is_the_earliest_valid_candidate(self):
        # Pins the exact boundary: the check is strictly "<=" rejected, not
        # some coarser (e.g. whole-second/minute) tolerance.
        just_after = FAILURE_TS + timedelta(microseconds=1)
        result = evaluate_compliance_v2(self._mandate_context(just_after))
        assert result.payment_verdict == "ALLOWED"

    def test_real_attempt_1_offset_of_one_hour_is_allowed(self):
        # The actual production shape: mandate_rules.STEP_WAIT_HOURS["attempt_1"] == 1.
        result = evaluate_compliance_v2(self._mandate_context(FAILURE_TS + timedelta(hours=1)))
        assert result.payment_verdict == "ALLOWED"


class TestResultShape:
    def test_to_dict_contains_verdicts_and_legacy_boolean_aliases(self):
        result = evaluate_compliance_v2(_domain_context())
        d = result.to_dict()
        for key in ("payment_verdict", "payment_reason", "communication_verdict", "communication_reason", "rule_version", "payment_action_allowed", "communication_action_allowed"):
            assert key in d

    def test_gates_are_independent(self):
        result = evaluate_compliance_v2(_domain_context(attempts_so_far=MAX_RETRY_ATTEMPTS))
        assert result.payment_verdict == "BLOCKED"
        assert result.communication_verdict == "ALLOWED"
