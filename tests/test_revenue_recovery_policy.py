"""
Track-03 tests: policy/revenue_recovery_policy.py -- the unified dispatcher.

Two concerns:
  1. payment_failed/subscription_payment_failed route to
     decide_for_failure_event_engine_v4 UNMODIFIED -- proven by a delegation
     test that compares the persisted row's fields against calling that
     function directly with identical inputs (see
     TestPaymentFailedDelegationUnchanged).
  2. The 3 new domains route to their own rule module and persist via
     policy/policy_decision_store.py, with dispatcher-level idempotency.
"""
from datetime import datetime

import numpy as np
import pytest

from app.models import PolicyDecision
from classification.rules import classify
from policy.decision_engine import NO_ACTION
from policy.decision_engine_v4 import decide_for_failure_event_engine_v4
from policy.revenue_recovery_policy import decide_for_revenue_risk_event

# Naive datetime -- policy/retry_candidates.py::generate_candidates does
# naive-datetime arithmetic internally (matches every other test file's
# convention, e.g. tests/test_compliance.py::FAILURE_TS).
NOW = datetime(2026, 8, 25, 10, 0, 0)

FAILURE_CONTEXT = {
    "day_of_month": 24, "days_to_nearest_payday_window": 6, "prior_if_failure_count": 0,
    "prior_if_self_resolved_rate": float("nan"), "tenure_days": 200, "plan_tier": "mid",
    "primary_instrument": "upi_autopay", "city_tier": "tier_1", "bank_network_conditions": "good",
    "issuing_bank_downtime_flag": False, "network_latency_bucket": "low", "is_month_end_settlement_rush": False,
}


class _PassthroughImputer:
    def transform(self, X):
        return X


class _FakeCatBoost:
    def predict(self, X):
        return np.array([100.0, 90.0, 80.0, 70.0, 60.0][: len(X)])


def _fake_model() -> dict:
    return {"imputer": _PassthroughImputer(), "catboost_model": _FakeCatBoost()}


def _row_fields_excluding_identity(row: PolicyDecision) -> dict:
    """Every column except the identity/timing/customer-reference ones that
    legitimately differ between two separately-created rows (id, event_id,
    decided_at, subscription_id -- the two calls below deliberately use
    different subscription_ids so each gets attempts_so_far=0 independently,
    since both share one in-memory DB via test_db_session)."""
    return {
        "selected_candidate_type": row.selected_candidate_type,
        "selected_candidate_datetime": row.selected_candidate_datetime,
        "predicted_recovery_probability": row.predicted_recovery_probability,
        "expected_recovery_value": row.expected_recovery_value,
        "expected_incremental_value": row.expected_incremental_value,
        "baseline_action": row.baseline_action,
        "policy_version": row.policy_version,
        "decision_reason": row.decision_reason,
        "classification_bucket": row.classification_bucket,
        "intervention_cost": row.intervention_cost,
        "runner_up_value": row.runner_up_value,
        "decision_margin": row.decision_margin,
        "decision_source": row.decision_source,
        "model_version": row.model_version,
        "margin_threshold_used": row.margin_threshold_used,
        "fallback_advantage_threshold": row.fallback_advantage_threshold,
        "fallback_strategy": row.fallback_strategy,
    }


class TestPaymentFailedDelegationUnchanged:
    @pytest.mark.parametrize("event_type", ["payment_failed", "subscription_payment_failed"])
    def test_dispatcher_output_identical_to_calling_v4_directly(self, test_db_session, event_type):
        bucket = classify(None, "insufficient_fund").bucket
        model = _fake_model()

        # Different subscription_ids -- both share one in-memory DB (test_db_session
        # is one engine for this whole test), so distinct subscription_ids keep
        # attempts_so_far=0 for each independently rather than the second call
        # seeing the first call's row as a prior attempt.
        db_direct = test_db_session()
        direct_row, direct_created = decide_for_failure_event_engine_v4(
            db_direct, event_id=5001, subscription_id="sub_direct_a", failure_timestamp=NOW,
            amount=1000.0, classification_bucket=bucket, failure_context=FAILURE_CONTEXT, model=model,
        )

        db_dispatch = test_db_session()
        dispatch_row, dispatch_created, _, _ = decide_for_revenue_risk_event(
            db_dispatch, event_type=event_type, event_id=5002, customer_ref="sub_direct_b",
            occurred_at=NOW, amount=1000.0, domain_context=FAILURE_CONTEXT, classification_bucket=bucket, model=model,
        )

        assert dispatch_created is True
        assert _row_fields_excluding_identity(direct_row) == _row_fields_excluding_identity(dispatch_row)
        db_direct.close()
        db_dispatch.close()

    def test_missing_classification_bucket_raises_for_payment_failed(self, test_db_session):
        db = test_db_session()
        with pytest.raises(ValueError, match="classification_bucket"):
            decide_for_revenue_risk_event(
                db, event_type="payment_failed", event_id=5003, customer_ref="sub_x",
                occurred_at=NOW, amount=500.0, domain_context={},
            )
        db.close()


class TestNewDomainDispatch:
    def test_checkout_abandoned_routes_to_checkout_rules(self, test_db_session):
        db = test_db_session()
        row, created, _, _ = decide_for_revenue_risk_event(
            db, event_type="checkout_abandoned", event_id=6001, customer_ref="cust_checkout_1",
            occurred_at=NOW, amount=500.0,
            domain_context={"cart_amount": 500.0, "inactivity_minutes": 90.0, "previous_outreach_count": 0},
        )
        assert created is True
        assert row.decision_source == "rule_checkout_abandoned"
        assert row.selected_candidate_type == "reminder"
        assert row.selected_candidate_datetime is not None
        db.close()

    def test_mandate_failed_routes_to_mandate_rules(self, test_db_session):
        db = test_db_session()
        row, created, _, _ = decide_for_revenue_risk_event(
            db, event_type="mandate_failed", event_id=6002, customer_ref="sub_mandate_1",
            occurred_at=NOW, amount=1000.0, domain_context={},
        )
        assert created is True
        assert row.decision_source == "rule_mandate_failed"
        assert row.selected_candidate_type == "attempt_1"
        db.close()

    def test_receivable_overdue_routes_to_receivables_rules(self, test_db_session):
        db = test_db_session()
        row, created, _, _ = decide_for_revenue_risk_event(
            db, event_type="receivable_overdue", event_id=6003, customer_ref="acct_1",
            occurred_at=NOW, amount=25000.0, domain_context={"days_overdue": 45},
        )
        assert created is True
        assert row.decision_source == "rule_receivable_overdue"
        assert row.selected_candidate_type == "escalation"
        db.close()

    def test_disputed_receivable_surfaces_requires_human_review(self, test_db_session):
        db = test_db_session()
        row, created, requires_human_review, human_review_reason = decide_for_revenue_risk_event(
            db, event_type="receivable_overdue", event_id=6007, customer_ref="acct_disputed",
            occurred_at=NOW, amount=25000.0, domain_context={"days_overdue": 45, "is_disputed": True},
        )
        assert created is True
        assert row.selected_candidate_type == "human_handoff"
        assert requires_human_review is True
        assert human_review_reason is not None
        db.close()

    def test_payment_failed_never_surfaces_human_review(self, test_db_session):
        db = test_db_session()
        bucket = classify(None, "insufficient_fund").bucket
        _, _, requires_human_review, human_review_reason = decide_for_revenue_risk_event(
            db, event_type="payment_failed", event_id=6008, customer_ref="sub_no_review",
            occurred_at=NOW, amount=500.0, domain_context=FAILURE_CONTEXT, classification_bucket=bucket, model=_fake_model(),
        )
        assert requires_human_review is False
        assert human_review_reason is None
        db.close()

    def test_dispatcher_level_idempotency(self, test_db_session):
        db = test_db_session()
        row1, created1, _, _ = decide_for_revenue_risk_event(
            db, event_type="checkout_abandoned", event_id=6004, customer_ref="cust_checkout_2",
            occurred_at=NOW, amount=500.0, domain_context={"cart_amount": 500.0, "inactivity_minutes": 90.0},
        )
        row2, created2, _, _ = decide_for_revenue_risk_event(
            db, event_type="checkout_abandoned", event_id=6004, customer_ref="cust_checkout_2",
            occurred_at=NOW, amount=500.0, domain_context={"cart_amount": 500.0, "inactivity_minutes": 90.0},
        )
        assert created1 is True
        assert created2 is False
        assert row1.id == row2.id
        db.close()

    def test_no_action_candidate_persists_with_no_datetime(self, test_db_session):
        db = test_db_session()
        row, _, _, _ = decide_for_revenue_risk_event(
            db, event_type="checkout_abandoned", event_id=6005, customer_ref="cust_checkout_3",
            occurred_at=NOW, amount=1.0, domain_context={"cart_amount": 1.0, "inactivity_minutes": 90.0},
        )
        assert row.selected_candidate_type == NO_ACTION
        assert row.selected_candidate_datetime is None
        db.close()

    def test_unknown_event_type_raises(self, test_db_session):
        db = test_db_session()
        with pytest.raises(ValueError, match="unknown event_type"):
            decide_for_revenue_risk_event(
                db, event_type="not_a_real_event_type", event_id=6006, customer_ref="x",
                occurred_at=NOW, amount=1.0, domain_context={},
            )
        db.close()


class TestUnifiedMLPolicyBoundary:
    """Phase-13/14: ML recommends, policy/compliance stay authoritative --
    ML must never be given the chance to fabricate an action where the rule
    decider says there is none, nor to override a human-review escalation.
    Uses the REAL trained unified model artifact (model/artifacts/unified_model.joblib)
    -- not a second, test-only inference implementation."""

    @pytest.fixture(autouse=True)
    def _unified_model(self):
        from model.unified_model import load_unified_model

        return load_unified_model()

    def test_ml_is_consulted_but_overridden_when_rule_says_no_action(self, test_db_session, _unified_model):
        """'Should ML evaluate this event' and 'should policy act on ML's
        recommendation' are different questions -- ML still runs (its
        recommendation/score is recorded for audit) even though the
        guardrail-triggered NO_ACTION eligibility gate is what actually
        wins as the FINAL candidate."""
        db = test_db_session()
        # amount=1.0 -- same guardrail-triggered NO_ACTION case as
        # test_no_action_candidate_persists_with_no_datetime above.
        row, created, _, _ = decide_for_revenue_risk_event(
            db, event_type="checkout_abandoned", event_id=7001, customer_ref="cust_ml_no_action",
            occurred_at=NOW, amount=1.0, domain_context={"cart_amount": 1.0, "inactivity_minutes": 90.0},
            model=_unified_model,
        )
        assert row.selected_candidate_type == NO_ACTION
        assert row.decision_source == "rule_checkout_abandoned"  # policy made the FINAL call, never "ml_unified_v1"
        assert "ml_consulted=True" in row.decision_reason  # but ML DID run, and that's visible in the audit trail
        assert "ml_recommendation=" in row.decision_reason
        assert row.model_version == "unified_catboost_v1"
        assert row.predicted_recovery_probability is not None  # ML's score for its (overridden) recommendation
        db.close()

    def test_ml_is_consulted_but_overridden_for_a_disputed_receivable_requiring_human_review(self, test_db_session, _unified_model):
        db = test_db_session()
        row, created, requires_human_review, _ = decide_for_revenue_risk_event(
            db, event_type="receivable_overdue", event_id=7002, customer_ref="acct_ml_disputed",
            occurred_at=NOW, amount=25000.0, domain_context={"days_overdue": 45, "is_disputed": True},
            model=_unified_model,
        )
        assert row.selected_candidate_type == "human_handoff"
        assert row.decision_source == "rule_receivable_overdue"  # never overridden by ML -- human review stays authoritative
        assert requires_human_review is True
        assert "ml_consulted=True" in row.decision_reason
        assert row.predicted_recovery_probability is not None
        db.close()

    def test_ml_not_consulted_marker_when_no_model_supplied(self, test_db_session):
        """Model genuinely unavailable (model=None, e.g. artifact missing) --
        must be visible as ml_consulted=False, never silently indistinguishable
        from 'ML ran and recommended NO_ACTION'."""
        db = test_db_session()
        row, created, _, _ = decide_for_revenue_risk_event(
            db, event_type="checkout_abandoned", event_id=7005, customer_ref="cust_ml_unavailable",
            occurred_at=NOW, amount=1.0, domain_context={"cart_amount": 1.0, "inactivity_minutes": 90.0},
            model=None,
        )
        assert row.selected_candidate_type == NO_ACTION
        assert "ml_consulted=False" in row.decision_reason
        assert row.predicted_recovery_probability is None
        db.close()

    def test_ml_consulted_and_recorded_when_rule_says_action_is_warranted(self, test_db_session, _unified_model):
        db = test_db_session()
        row, created, _, _ = decide_for_revenue_risk_event(
            db, event_type="checkout_abandoned", event_id=7003, customer_ref="cust_ml_eligible",
            occurred_at=NOW, amount=999.0, domain_context={"cart_amount": 999.0, "inactivity_minutes": 90.0},
            model=_unified_model,
        )
        assert row.decision_source == "ml_unified_v1"
        assert row.selected_candidate_type in {"reminder", "payment_link_reminder", "retry_checkout", "alternate_payment_method"}
        assert row.model_version == "unified_catboost_v1"
        assert row.predicted_recovery_probability is not None
        assert "rule_baseline_candidate=" in row.decision_reason  # Phase-13 audit trail
        db.close()

    def test_payment_link_no_subscription_event_reaches_ml_with_classification_preserved(self, test_db_session, _unified_model):
        """The ₹1 Payment Link scenario (subscription_id=NULL): reaches the
        unified model via the payment_failed_no_subscription -> payment_failed
        alias, and classification_bucket stays the real rule-computed fact,
        not the storage-level event_type string."""
        db = test_db_session()
        row, created, _, _ = decide_for_revenue_risk_event(
            db, event_type="payment_failed_no_subscription", event_id=7004, customer_ref="pay_ml_link",
            occurred_at=NOW, amount=1.0, domain_context={"error_reason": "insufficient_fund"},
            model=_unified_model,
        )
        assert row.decision_source == "ml_unified_v1"
        assert row.selected_candidate_type == "payment_link_reminder"
        assert row.classification_bucket == "retryable_soft"
        db.close()

    def test_payment_link_with_unmapped_generic_reason_still_consults_ml(self, test_db_session, _unified_model):
        """Reproduces a REAL live Razorpay Test Mode Payment Link failure
        (razorpay_event_id=TUMJKgLj36PqPW, payment_id=pay_TUMJEoCAr6tS0E,
        amount=paise 100/Rs1, subscription_id=NULL): Razorpay returned the
        generic error_reason "payment_failed", which classification/rules.py
        correctly refuses to guess a bucket for (never in the verified
        Razorpay reason table) -- classification_bucket="unmapped" ->
        eligibility gate -> NO_ACTION. Before this fix, that eligibility
        outcome silently meant ML was never even attempted; now ML IS
        consulted (a real recommendation/score is computed and recorded),
        it is simply overridden by the eligibility gate exactly like the
        disputed-receivable/opted-out cases above."""
        db = test_db_session()
        row, created, _, _ = decide_for_revenue_risk_event(
            db, event_type="payment_failed_no_subscription", event_id=7006, customer_ref="pay_TUMJEoCAr6tS0E",
            occurred_at=NOW, amount=1.0, domain_context={"error_code": "BAD_REQUEST_ERROR", "error_reason": "payment_failed", "error_source": "gateway", "error_step": "payment_authorization"},
            model=_unified_model,
        )
        assert row.classification_bucket == "unmapped"
        assert row.selected_candidate_type == NO_ACTION
        assert row.decision_source == "rule_one_time_payment_failed"  # policy's eligibility gate made the final call
        assert "ml_consulted=True" in row.decision_reason  # ML was NOT silently skipped
        assert "ml_recommendation=payment_link_reminder" in row.decision_reason  # the domain's only candidate
        assert row.model_version == "unified_catboost_v1"
        assert row.predicted_recovery_probability is not None
        db.close()
