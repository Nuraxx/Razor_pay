"""
Closed-loop tests: recovery/payment_reconciliation.py -- confirms a PENDING
legacy recovery_outcomes row as RECOVERED/PARTIALLY_RECOVERED from an
authoritative Razorpay payment.captured event, and nothing else can.
"""
from datetime import datetime, timezone

from app.models import AuditLog, FailureEvent, PolicyDecision, RawEvent, RecoveryOutcome
from llm.client import LLMClient, LLMProviderError
from recovery.orchestrator import RecoveryEventInput, orchestrate_recovery
from recovery.payment_reconciliation import (
    CONFIRMED_BY_WEBHOOK,
    OUTCOME_ALREADY_CONFIRMED,
    OUTCOME_CONFIRMED,
    OUTCOME_NO_MATCH,
    STATUS_PARTIALLY_RECOVERED,
    STATUS_RECOVERED,
    confirm_payment_recovery,
)

FAILURE_TS = datetime(2026, 8, 25, 10, 0, 0)


def _make_pending_case(db, *, raw_event_id: str, payment_id: str, subscription_id: str, amount_paise: int = 100000) -> tuple[RawEvent, FailureEvent, RecoveryOutcome]:
    """Builds a full, real PENDING recovery case exactly the shape the real
    webhook->classify->orchestrate pipeline produces, via direct ORM inserts
    (fast, no LLM/model dependency needed for these reconciliation-focused
    tests -- recovery/orchestrator.py's own tests already cover how a PENDING
    outcome gets created)."""
    raw = RawEvent(
        razorpay_event_id=raw_event_id, event_type="payment.failed", payment_id=payment_id, subscription_id=subscription_id,
        amount=amount_paise, currency="INR", error_reason="insufficient_fund", signature_verified=True, raw_payload="{}",
    )
    db.add(raw)
    db.flush()
    failure = FailureEvent(raw_event_id=raw.id, classification_bucket="retryable_soft", classification_confidence=1.0, rule_version="v1")
    db.add(failure)
    db.flush()
    db.add(PolicyDecision(
        event_id=failure.id, subscription_id=subscription_id, selected_candidate_type="payday_window",
        policy_version="policy-v4", decision_reason="test fixture", decision_source="rule_based_fallback",
        classification_bucket="retryable_soft",
    ))
    outcome = RecoveryOutcome(
        event_id=failure.id, event_type="payment_failed", at_risk_amount=amount_paise / 100.0,
        recovered_amount=None, retained_amount=None, lost_amount=None,
        recovery_status="PENDING", confirmed_by="unconfirmed_pending",
    )
    db.add(outcome)
    db.commit()
    return raw, failure, outcome


def _captured_event(*, razorpay_event_id: str, payment_id: str, subscription_id: str | None, amount_paise: int) -> RawEvent:
    """A stored (not-yet-reconciled) payment.captured RawEvent, mirroring
    what app/main.py's webhook handler would have already persisted before
    calling confirm_payment_recovery -- these tests exercise the
    reconciliation function directly, one layer below the HTTP endpoint
    (see tests/test_webhook_endpoint.py for the full HTTP-level flow)."""
    return RawEvent(
        razorpay_event_id=razorpay_event_id, event_type="payment.captured", payment_id=payment_id, subscription_id=subscription_id,
        amount=amount_paise, currency="INR", signature_verified=True, raw_payload="{}",
    )


class TestOutcomeStartsPending:
    def test_recovery_outcome_starts_as_pending(self, test_db_session):
        db = test_db_session()
        _, _, outcome = _make_pending_case(db, raw_event_id="evt_f1", payment_id="pay_f1", subscription_id="sub_1")
        assert outcome.recovery_status == "PENDING"
        assert outcome.recovered_amount is None
        assert outcome.confirmed_by == "unconfirmed_pending"
        assert outcome.confirmed_payment_id is None
        db.close()


class TestSuccessfulConfirmation:
    def test_matching_payment_id_confirms_recovery(self, test_db_session):
        db = test_db_session()
        raw, failure, outcome = _make_pending_case(db, raw_event_id="evt_f2", payment_id="pay_f2", subscription_id="sub_2", amount_paise=150000)
        captured = _captured_event(razorpay_event_id="evt_s2", payment_id="pay_f2", subscription_id="sub_2", amount_paise=150000)
        db.add(captured)
        db.flush()

        result = confirm_payment_recovery(db, captured)

        assert result == OUTCOME_CONFIRMED
        db.refresh(outcome)
        assert outcome.recovery_status == STATUS_RECOVERED
        assert outcome.recovered_amount == 1500.0  # authoritative Razorpay amount, rupees
        assert outcome.confirmed_by == CONFIRMED_BY_WEBHOOK
        assert outcome.confirmed_payment_id == "pay_f2"
        db.close()

    def test_subscription_id_match_confirms_when_payment_id_differs(self, test_db_session):
        # The realistic Razorpay Subscriptions case: the retry that succeeds
        # gets a BRAND NEW payment_id, never the one that failed.
        db = test_db_session()
        raw, failure, outcome = _make_pending_case(db, raw_event_id="evt_f3", payment_id="pay_f3_failed", subscription_id="sub_3", amount_paise=99900)
        captured = _captured_event(razorpay_event_id="evt_s3", payment_id="pay_f3_RETRY_new_id", subscription_id="sub_3", amount_paise=99900)
        db.add(captured)
        db.flush()

        result = confirm_payment_recovery(db, captured)

        assert result == OUTCOME_CONFIRMED
        db.refresh(outcome)
        assert outcome.recovery_status == STATUS_RECOVERED
        assert outcome.confirmed_payment_id == "pay_f3_RETRY_new_id"
        db.close()

    def test_partial_amount_is_partially_recovered_not_recovered(self, test_db_session):
        db = test_db_session()
        raw, failure, outcome = _make_pending_case(db, raw_event_id="evt_f4", payment_id="pay_f4", subscription_id="sub_4", amount_paise=200000)
        captured = _captured_event(razorpay_event_id="evt_s4", payment_id="pay_f4", subscription_id="sub_4", amount_paise=120000)  # only 1200 of 2000 rupees
        db.add(captured)
        db.flush()

        result = confirm_payment_recovery(db, captured)

        assert result == OUTCOME_CONFIRMED
        db.refresh(outcome)
        assert outcome.recovery_status == STATUS_PARTIALLY_RECOVERED
        assert outcome.recovered_amount == 1200.0
        db.close()


class TestIdempotencyAndCorrelationSafety:
    def test_duplicate_confirmation_does_not_double_apply(self, test_db_session):
        db = test_db_session()
        raw, failure, outcome = _make_pending_case(db, raw_event_id="evt_f5", payment_id="pay_f5", subscription_id="sub_5", amount_paise=50000)
        captured = _captured_event(razorpay_event_id="evt_s5", payment_id="pay_f5", subscription_id="sub_5", amount_paise=50000)
        db.add(captured)
        db.flush()
        confirm_payment_recovery(db, captured)

        # a second, DIFFERENT success event (different razorpay_event_id) referring to the SAME payment
        captured_again = _captured_event(razorpay_event_id="evt_s5_dup", payment_id="pay_f5", subscription_id="sub_5", amount_paise=50000)
        db.add(captured_again)
        db.flush()
        result = confirm_payment_recovery(db, captured_again)

        assert result == OUTCOME_ALREADY_CONFIRMED
        db.refresh(outcome)
        assert outcome.recovery_status == STATUS_RECOVERED  # unchanged, not re-applied
        assert outcome.recovered_amount == 500.0  # unchanged, not doubled
        db.close()

    def test_unrelated_success_event_does_not_modify_an_unrelated_outcome(self, test_db_session):
        db = test_db_session()
        _, _, outcome_a = _make_pending_case(db, raw_event_id="evt_f6a", payment_id="pay_f6a", subscription_id="sub_6a")
        _, _, outcome_b = _make_pending_case(db, raw_event_id="evt_f6b", payment_id="pay_f6b", subscription_id="sub_6b")

        captured = _captured_event(razorpay_event_id="evt_s6b", payment_id="pay_f6b", subscription_id="sub_6b", amount_paise=100000)
        db.add(captured)
        db.flush()
        confirm_payment_recovery(db, captured)

        db.refresh(outcome_a)
        db.refresh(outcome_b)
        assert outcome_a.recovery_status == "PENDING"  # untouched
        assert outcome_b.recovery_status == STATUS_RECOVERED
        db.close()

    def test_success_event_with_no_matching_case_is_a_safe_no_op(self, test_db_session):
        db = test_db_session()
        captured = _captured_event(razorpay_event_id="evt_s7", payment_id="pay_ordinary_success", subscription_id="sub_never_failed", amount_paise=75000)
        db.add(captured)
        db.flush()

        result = confirm_payment_recovery(db, captured)

        assert result == OUTCOME_NO_MATCH
        assert db.query(RecoveryOutcome).filter(RecoveryOutcome.recovery_status == STATUS_RECOVERED).count() == 0
        db.close()


class TestFailedPaymentNeverFabricatesRecovery:
    def test_a_failed_payment_alone_never_produces_a_recovered_outcome(self, test_db_session):
        db = test_db_session()
        event = RecoveryEventInput(
            event_id=900201, subscription_id="sub_never_recovered", failure_timestamp=FAILURE_TS,
            amount=500.0, error_code=None, error_reason="insufficient_fund",
        )
        orchestrate_recovery(db, event)
        outcome = db.query(RecoveryOutcome).filter(RecoveryOutcome.event_id == 900201, RecoveryOutcome.event_type == "payment_failed").first()
        assert outcome.recovery_status == "PENDING"
        assert outcome.recovered_amount is None
        db.close()


class TestConfirmationNeverAffectsPolicyOrLLM:
    def test_policy_decision_is_unchanged_by_confirmation(self, test_db_session):
        db = test_db_session()
        raw, failure, outcome = _make_pending_case(db, raw_event_id="evt_f8", payment_id="pay_f8", subscription_id="sub_8")
        decision_before = db.query(PolicyDecision).filter(PolicyDecision.event_id == failure.id).first()
        selected_before = decision_before.selected_candidate_type

        captured = _captured_event(razorpay_event_id="evt_s8", payment_id="pay_f8", subscription_id="sub_8", amount_paise=100000)
        db.add(captured)
        db.flush()
        confirm_payment_recovery(db, captured)

        decision_after = db.query(PolicyDecision).filter(PolicyDecision.event_id == failure.id).first()
        assert decision_after.selected_candidate_type == selected_before  # confirmation never touches policy
        db.close()

    def test_llm_success_or_failure_never_influences_confirmation(self, test_db_session):
        # LLM communication (success or fallback) already happened, entirely
        # independently, back when the case was first orchestrated -- the
        # confirmation function never queries LLMInvocation or reads any
        # llm_success value at all (grep-verifiable: no LLMInvocation import
        # in recovery/payment_reconciliation.py). Confirm behavior is
        # identical whether or not any LLM activity exists for this event.
        db1, db2 = test_db_session(), test_db_session()

        raw1, failure1, outcome1 = _make_pending_case(db1, raw_event_id="evt_f9a", payment_id="pay_f9a", subscription_id="sub_9a")
        captured1 = _captured_event(razorpay_event_id="evt_s9a", payment_id="pay_f9a", subscription_id="sub_9a", amount_paise=100000)
        db1.add(captured1)
        db1.flush()
        result1 = confirm_payment_recovery(db1, captured1)

        raw2, failure2, outcome2 = _make_pending_case(db2, raw_event_id="evt_f9b", payment_id="pay_f9b", subscription_id="sub_9b")
        db2.add(AuditLog(failure_event_id=failure2.id, action="llm_outreach_microcopy_failed_used_fallback", reason="simulated fallback", actor="llm"))
        db2.commit()
        captured2 = _captured_event(razorpay_event_id="evt_s9b", payment_id="pay_f9b", subscription_id="sub_9b", amount_paise=100000)
        db2.add(captured2)
        db2.flush()
        result2 = confirm_payment_recovery(db2, captured2)

        assert result1 == result2 == OUTCOME_CONFIRMED
        db1.refresh(outcome1)
        db2.refresh(outcome2)
        assert outcome1.recovery_status == outcome2.recovery_status == STATUS_RECOVERED
        db1.close()
        db2.close()


class TestNewDomainsStayPendingWithoutAuthoritativeLinkage:
    def test_confirm_payment_recovery_never_touches_track03_outcomes(self, test_db_session):
        # A Track-03 RecoveryOutcome (event_type != "payment_failed") must
        # never be reachable by this function, even if a coincidence of ids
        # or subscription_id-shaped customer_ref could otherwise line up --
        # the query is explicitly scoped to event_type == "payment_failed".
        db = test_db_session()
        track03_outcome = RecoveryOutcome(
            event_id=1, event_type="checkout_abandoned", at_risk_amount=999.0,
            recovered_amount=None, recovery_status="PENDING", confirmed_by="unconfirmed_pending",
        )
        db.add(track03_outcome)
        db.commit()

        captured = _captured_event(razorpay_event_id="evt_s10", payment_id="pay_unrelated", subscription_id="demo_cust_checkout", amount_paise=99900)
        db.add(captured)
        db.flush()
        result = confirm_payment_recovery(db, captured)

        assert result == OUTCOME_NO_MATCH
        db.refresh(track03_outcome)
        assert track03_outcome.recovery_status == "PENDING"  # never touched
        db.close()


class TestAuditRecordCreated:
    def test_confirmation_writes_an_audit_row_with_the_required_fields(self, test_db_session):
        db = test_db_session()
        raw, failure, outcome = _make_pending_case(db, raw_event_id="evt_f11", payment_id="pay_f11", subscription_id="sub_11", amount_paise=100000)
        captured = _captured_event(razorpay_event_id="evt_s11", payment_id="pay_f11", subscription_id="sub_11", amount_paise=100000)
        db.add(captured)
        db.flush()
        confirm_payment_recovery(db, captured)

        audit = db.query(AuditLog).filter(AuditLog.action == "payment_recovery_confirmed", AuditLog.failure_event_id == failure.id).first()
        assert audit is not None
        assert "pay_f11" in audit.reason
        assert "previous_status=PENDING" in audit.reason
        assert "new_status=RECOVERED" in audit.reason
        assert "recovered_amount=1000.0" in audit.reason
        assert "confirmed_by=webhook_confirmed" in audit.reason
        assert audit.created_at is not None
        db.close()
