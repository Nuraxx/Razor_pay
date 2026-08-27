"""
Day-12 tests: recovery/orchestrator.py -- the full failure matrix (brief
section 7, scenarios A-K), audit trail, final-result schema, and the
LLM-cannot-affect-payment guarantee.

Uses a hand-crafted fake Model B (same pattern as tests/test_decision_engine.py
/ test_decision_engine_v4.py) so payment-decision outcomes are fully
deterministic and controllable per test, independent of the real trained
model's behavior.
"""
from datetime import datetime, timedelta

import numpy as np
import pytest

from app.models import AuditLog, LLMInvocation, PolicyDecision
from llm.client import LLMClient, LLMProviderError
from llm.service import generate_outreach_microcopy
from policy.guardrails import MAX_RETRY_ATTEMPTS
from recovery.orchestrator import RecoveryEventInput, orchestrate_recovery
from recovery.schemas import RecoveryExecutionResult

FAILURE_TS = datetime(2026, 2, 24, 10, 0, 0)  # all 5 candidates valid (Day 7/9/10 convention)
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
    def __init__(self, values):
        self._values = values

    def predict(self, X):
        return np.array(self._values[: len(X)])


def _fake_model(values=None) -> dict:
    values = values or [500.0, 10.0, 5.0, 1.0, 0.5]  # huge margin -> confident, day8_model_b-sourced decision
    return {"imputer": _PassthroughImputer(), "catboost_model": _FakeCatBoost(values)}


def _make_event(**overrides) -> RecoveryEventInput:
    base = dict(
        event_id=1, subscription_id="sub_test", failure_timestamp=FAILURE_TS, amount=1000.0,
        error_code=None, error_reason="insufficient_fund", failure_context=FAILURE_CONTEXT,
    )
    base.update(overrides)
    return RecoveryEventInput(**base)


# ---------------------------------------------------------------------------
# A-D: classification-driven outcomes
# ---------------------------------------------------------------------------

def test_A_insufficient_fund_valid_retry_is_allowed(test_db_session):
    db = test_db_session()
    result = orchestrate_recovery(db, _make_event(event_id=1, error_reason="insufficient_fund"), model=_fake_model())
    assert result.classification_bucket == "retryable_soft"
    assert result.payment_action == "retry_scheduled"
    assert result.compliance_allowed is True
    assert result.final_status in ("RETRY_ALLOWED", "COMMUNICATION_ALLOWED", "POLICY_FALLBACK", "LLM_FALLBACK")
    db.close()


def test_B_hard_decline_no_retry(test_db_session):
    # No automatic retry -- but per the specification, hard_decline still
    # gets a payment-method-update communication (FIX #3), so this event's
    # own final_status is COMMUNICATION_ALLOWED, not NO_ACTION, precisely
    # because something real DID happen (a nudge was sent). See the
    # dedicated hard-decline-communication tests below for the full matrix.
    db = test_db_session()
    result = orchestrate_recovery(db, _make_event(event_id=2, error_reason="card_expired"), model=_fake_model())
    assert result.classification_bucket == "hard_decline"
    assert result.selected_candidate_type == "NO_ACTION"
    assert result.payment_action == "no_action"
    assert result.final_status in ("COMMUNICATION_ALLOWED", "LLM_FALLBACK")
    db.close()


def test_C_customer_cancelled_no_retry_and_communication_blocked(test_db_session):
    db = test_db_session()
    result = orchestrate_recovery(db, _make_event(event_id=3, error_reason="payment_cancelled"), model=_fake_model())
    assert result.classification_bucket == "customer_cancelled"
    assert result.payment_action == "no_action"
    assert result.communication_action == "skipped"  # NO_ACTION short-circuits communication entirely
    assert result.final_status == "NO_ACTION"
    db.close()


def test_D_unmapped_no_retry(test_db_session):
    db = test_db_session()
    result = orchestrate_recovery(db, _make_event(event_id=4, error_reason="totally_unrecognized_reason_xyz"), model=_fake_model())
    assert result.classification_bucket == "unmapped"
    assert result.payment_action == "no_action"
    db.close()


# ---------------------------------------------------------------------------
# E: max retry attempts reached
# ---------------------------------------------------------------------------

def test_E_max_retry_attempts_reached_blocks_payment(test_db_session):
    # Policy itself (policy/guardrails.py::MAX_RETRY_ATTEMPTS, enforced
    # inside decide_engine_v4) already refuses to select a candidate once
    # attempts are exhausted, returning NO_ACTION -- so end-to-end, the 4th
    # event never reaches compliance's OWN max-attempts branch at all (that
    # branch is exercised directly, at the compliance layer, by
    # tests/test_compliance.py::test_max_retry_attempts_blocks_payment,
    # exactly like invalid-candidate scenario G below). What this test
    # verifies end-to-end is that exhaustion is honored consistently the
    # whole way through the orchestrator, landing on NO_ACTION.
    db = test_db_session()
    sub_id = "sub_max_attempts"
    for i in range(MAX_RETRY_ATTEMPTS):
        orchestrate_recovery(db, _make_event(event_id=100 + i, subscription_id=sub_id, failure_timestamp=FAILURE_TS + timedelta(days=i * 20)), model=_fake_model())
    exhausted = orchestrate_recovery(db, _make_event(event_id=200, subscription_id=sub_id, failure_timestamp=FAILURE_TS + timedelta(days=100)), model=_fake_model())
    assert exhausted.payment_action == "no_action"
    assert exhausted.selected_candidate_type == "NO_ACTION"
    assert exhausted.final_status == "NO_ACTION"
    assert "max_retry_attempts" in (exhausted.decision_reason or "")
    db.close()


# ---------------------------------------------------------------------------
# F: duplicate event -> blocked/idempotent
# ---------------------------------------------------------------------------

def test_F_duplicate_event_is_blocked_and_idempotent(test_db_session):
    db = test_db_session()
    event = _make_event(event_id=5)
    first = orchestrate_recovery(db, event, model=_fake_model())
    second = orchestrate_recovery(db, event, model=_fake_model())
    assert first.payment_action == "retry_scheduled"
    assert second.payment_action == "blocked"
    assert second.compliance_reason == "duplicate_payment_action_blocked"
    assert second.final_status == "RETRY_BLOCKED"
    # exactly one policy_decisions row was ever created for this event_id
    assert db.query(PolicyDecision).filter(PolicyDecision.event_id == 5).count() == 1
    db.close()


# ---------------------------------------------------------------------------
# G: invalid candidate -> blocked (compliance-layer defense-in-depth;
# structurally prevented from reaching this point via the full flow since
# policy itself filters invalid candidates -- see tests/test_compliance.py
# for the direct compliance-layer test of this rule)
# ---------------------------------------------------------------------------

def test_G_invalid_candidate_blocked_at_compliance_layer():
    from policy.compliance import ComplianceContext, evaluate_compliance

    context = ComplianceContext(
        classification_bucket="retryable_soft", selected_candidate_type="plus_1_day_morning",
        selected_candidate_datetime=FAILURE_TS - timedelta(hours=1),  # before the failure -- invalid
        failure_timestamp=FAILURE_TS, attempts_so_far=0,
    )
    result = evaluate_compliance(context)
    assert result.payment_action_allowed is False
    assert "candidate_not_after_failure" in result.payment_reason


# ---------------------------------------------------------------------------
# H: compliance rejection -> payment/communication blocked
# ---------------------------------------------------------------------------

def test_H_compliance_rejection_blocks_payment_via_opt_out_does_not_block_it(test_db_session):
    # opt-out blocks COMMUNICATION but not payment -- the independence brief section 3 requires
    db = test_db_session()
    result = orchestrate_recovery(db, _make_event(event_id=6, customer_opted_out=True), model=_fake_model())
    assert result.payment_action == "retry_scheduled"
    assert result.communication_action == "blocked"
    assert result.final_status == "COMMUNICATION_BLOCKED"
    db.close()


def test_H_required_fields_missing_blocks_payment(test_db_session):
    db = test_db_session()
    result = orchestrate_recovery(db, _make_event(event_id=7, required_fields_present=False), model=_fake_model())
    assert result.payment_action == "blocked"
    assert result.compliance_reason == "required_fields_missing"
    assert result.final_status == "RETRY_BLOCKED"
    db.close()


# ---------------------------------------------------------------------------
# I: LLM unavailable -> payment decision unchanged, deterministic fallback
# ---------------------------------------------------------------------------

class _UnavailableClient(LLMClient):
    model_name = "unavailable"
    provider_name = "mock"

    def complete(self, system_prompt, user_prompt, *, max_tokens=512):
        raise LLMProviderError("provider_unavailable")


def test_I_llm_unavailable_payment_unaffected_deterministic_fallback(test_db_session):
    db = test_db_session()
    with_llm = orchestrate_recovery(db, _make_event(event_id=8), model=_fake_model())
    without_llm = orchestrate_recovery(db, _make_event(event_id=9, subscription_id="sub_llm_unavailable"), model=_fake_model(), llm_client=_UnavailableClient())

    assert with_llm.payment_action == without_llm.payment_action == "retry_scheduled"
    assert with_llm.selected_candidate_type == without_llm.selected_candidate_type
    assert without_llm.communication_action == "fallback_used"
    assert without_llm.llm_success is False
    assert without_llm.final_status == "LLM_FALLBACK"
    db.close()


# ---------------------------------------------------------------------------
# J: malformed LLM output -> payment decision unchanged, deterministic fallback
# ---------------------------------------------------------------------------

class _MalformedJSONClient(LLMClient):
    model_name = "malformed"
    provider_name = "mock"

    def complete(self, system_prompt, user_prompt, *, max_tokens=512):
        return "this is not valid json {{"


def test_J_malformed_llm_output_payment_unaffected_deterministic_fallback(test_db_session):
    db = test_db_session()
    result = orchestrate_recovery(db, _make_event(event_id=10), model=_fake_model(), llm_client=_MalformedJSONClient())
    assert result.payment_action == "retry_scheduled"
    assert result.compliance_allowed is True
    assert result.communication_action == "fallback_used"
    assert result.llm_success is False
    assert result.final_status == "LLM_FALLBACK"
    db.close()


# ---------------------------------------------------------------------------
# K: fully valid flow -> classification -> policy -> compliance -> communication -> audit
# ---------------------------------------------------------------------------

def test_K_fully_valid_flow_produces_full_audit_trail(test_db_session):
    db = test_db_session()
    result = orchestrate_recovery(db, _make_event(event_id=11), model=_fake_model())

    assert result.classification_bucket == "retryable_soft"
    assert result.payment_action == "retry_scheduled"
    assert result.communication_action in ("sent", "fallback_used")
    assert result.llm_task_name == "outreach_microcopy"

    actors = {row.actor for row in db.query(AuditLog).filter(AuditLog.failure_event_id == 11).all()}
    assert actors == {"classifier", "policy", "compliance", "llm", "orchestrator"}
    db.close()


# ---------------------------------------------------------------------------
# Audit trail: no secrets, exact actor labels
# ---------------------------------------------------------------------------

def test_audit_trail_actor_values_are_explicit(test_db_session):
    db = test_db_session()
    orchestrate_recovery(db, _make_event(event_id=12), model=_fake_model())
    rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 12).order_by(AuditLog.id).all()
    actions = [row.action for row in rows]
    assert "orchestrator_classification" in actions
    assert "orchestrator_compliance" in actions
    assert "orchestrator_final_status" in actions
    for row in rows:
        assert row.actor in ("classifier", "policy", "compliance", "llm", "orchestrator")
    db.close()


def test_no_secrets_in_audit_trail(test_db_session):
    db = test_db_session()
    orchestrate_recovery(db, _make_event(event_id=13), model=_fake_model())
    rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 13).all()
    for row in rows:
        text = (row.reason or "").lower()
        assert "api_key" not in text
        assert "webhook_secret" not in text
        assert "authorization" not in text
    db.close()


# ---------------------------------------------------------------------------
# Orchestrator ordering: policy always runs before LLM, LLM never affects policy
# ---------------------------------------------------------------------------

def test_policy_decision_persisted_before_any_llm_call(test_db_session):
    db = test_db_session()
    policy_row_exists_when_llm_called = {}

    class _AssertPolicyAlreadyPersistedClient(LLMClient):
        model_name = "ordering-check"
        provider_name = "mock"

        def complete(self, system_prompt, user_prompt, *, max_tokens=512):
            existing = db.query(PolicyDecision).filter(PolicyDecision.event_id == 14).first()
            policy_row_exists_when_llm_called["value"] = existing is not None
            import json

            return json.dumps({"message_text": "ok", "language": "en", "failure_bucket": "retryable_soft", "customer_segment": "mid"})

    result = orchestrate_recovery(db, _make_event(event_id=14), model=_fake_model(), llm_client=_AssertPolicyAlreadyPersistedClient())
    assert policy_row_exists_when_llm_called.get("value") is True, "LLM was called before the policy decision was persisted"
    assert result.payment_action == "retry_scheduled"
    db.close()


def test_llm_failure_never_changes_selected_candidate_or_compliance(test_db_session):
    db = test_db_session()
    event_a = _make_event(event_id=15, subscription_id="sub_ordering_a")
    event_b = _make_event(event_id=16, subscription_id="sub_ordering_b")

    result_ok = orchestrate_recovery(db, event_a, model=_fake_model())
    result_broken_llm = orchestrate_recovery(db, event_b, model=_fake_model(), llm_client=_UnavailableClient())

    assert result_ok.selected_candidate_type == result_broken_llm.selected_candidate_type
    assert result_ok.compliance_allowed == result_broken_llm.compliance_allowed == True
    assert result_ok.payment_action == result_broken_llm.payment_action == "retry_scheduled"
    db.close()


# ---------------------------------------------------------------------------
# Final result schema
# ---------------------------------------------------------------------------

def test_result_is_recovery_execution_result_with_required_fields(test_db_session):
    db = test_db_session()
    result = orchestrate_recovery(db, _make_event(event_id=17), model=_fake_model())
    assert isinstance(result, RecoveryExecutionResult)
    for field_name in (
        "event_id", "subscription_id", "classification_bucket", "policy_version", "selected_candidate_type",
        "selected_candidate_datetime", "compliance_allowed", "compliance_reason", "payment_action",
        "communication_action", "llm_task_name", "llm_success", "final_status", "created_at",
    ):
        assert hasattr(result, field_name)
    db.close()


def test_result_serializes_to_json(test_db_session):
    import json

    db = test_db_session()
    result = orchestrate_recovery(db, _make_event(event_id=18), model=_fake_model())
    parsed = json.loads(result.to_json())
    assert parsed["event_id"] == 18
    assert parsed["final_status"] in (
        "RETRY_ALLOWED", "RETRY_BLOCKED", "COMMUNICATION_ALLOWED", "COMMUNICATION_BLOCKED",
        "NO_ACTION", "POLICY_FALLBACK", "LLM_FALLBACK",
    )
    db.close()


def test_determinism_same_inputs_same_result():
    # Two GENUINELY independent in-memory databases (not two sessions
    # sharing one fixture-provided engine, which would make the second call
    # a legitimate duplicate of the first rather than an independent run).
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.db import Base

    def _fresh_session():
        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(bind=engine)
        return sessionmaker(bind=engine)()

    db1, db2 = _fresh_session(), _fresh_session()
    r1 = orchestrate_recovery(db1, _make_event(event_id=19), model=_fake_model())
    r2 = orchestrate_recovery(db2, _make_event(event_id=19), model=_fake_model())
    assert r1 == r2  # created_at is compare=False; everything else must match
    db1.close()
    db2.close()


# ---------------------------------------------------------------------------
# FIX #3: hard-decline communication (payment-method-update nudge)
# ---------------------------------------------------------------------------

def test_hard_decline_communication_uses_will_retry_false(test_db_session):
    db = test_db_session()
    result = orchestrate_recovery(db, _make_event(event_id=30, error_reason="card_expired"), model=_fake_model())
    assert result.payment_action == "no_action"  # no false retry timing is ever implied
    assert result.communication_action in ("sent", "fallback_used")
    assert result.llm_task_name == "outreach_microcopy"
    invocation = db.query(LLMInvocation).filter(LLMInvocation.event_id == 30, LLMInvocation.task_name == "outreach_microcopy").first()
    assert invocation is not None
    import json as _json

    structured = _json.loads(invocation.structured_output)
    # the mock/fallback text for will_retry=False never claims a retry will happen
    assert "retry" not in structured["message_text"].lower() or "check" in structured["message_text"].lower()
    db.close()


def test_hard_decline_customer_opted_out_blocks_communication(test_db_session):
    db = test_db_session()
    result = orchestrate_recovery(db, _make_event(event_id=31, error_reason="card_expired", customer_opted_out=True), model=_fake_model())
    assert result.payment_action == "no_action"
    assert result.communication_action == "blocked"
    assert result.final_status == "COMMUNICATION_BLOCKED"
    db.close()


def test_hard_decline_missing_consent_blocks_communication(test_db_session):
    db = test_db_session()
    result = orchestrate_recovery(db, _make_event(event_id=32, error_reason="card_expired", consent_for_communication=False), model=_fake_model())
    assert result.payment_action == "no_action"
    assert result.communication_action == "blocked"
    assert result.communication_reason == "consent_for_communication_missing"
    assert result.final_status == "COMMUNICATION_BLOCKED"
    db.close()


def test_hard_decline_llm_failure_produces_deterministic_fallback(test_db_session):
    db = test_db_session()
    result = orchestrate_recovery(db, _make_event(event_id=33, error_reason="card_expired"), model=_fake_model(), llm_client=_UnavailableClient())
    assert result.payment_action == "no_action"
    assert result.communication_action == "fallback_used"
    assert result.llm_success is False
    assert result.final_status == "LLM_FALLBACK"
    db.close()


def test_hard_decline_communication_requested_false_skips(test_db_session):
    db = test_db_session()
    result = orchestrate_recovery(db, _make_event(event_id=34, error_reason="card_expired", request_communication=False), model=_fake_model())
    assert result.payment_action == "no_action"
    assert result.communication_action == "skipped"
    assert result.final_status == "NO_ACTION"
    db.close()


def test_customer_cancelled_still_never_gets_communication(test_db_session):
    # Distinguishes hard_decline (nudge allowed) from customer_cancelled
    # (compliance's own opt-out-on-cancellation rule blocks it) -- the two
    # NO_ACTION buckets must NOT be treated identically after FIX #3.
    db = test_db_session()
    result = orchestrate_recovery(db, _make_event(event_id=35, error_reason="payment_cancelled"), model=_fake_model())
    assert result.payment_action == "no_action"
    assert result.communication_action == "skipped"
    assert result.final_status == "NO_ACTION"
    db.close()


def test_unmapped_still_never_gets_communication(test_db_session):
    db = test_db_session()
    result = orchestrate_recovery(db, _make_event(event_id=36, error_reason="totally_unrecognized_reason_xyz"), model=_fake_model())
    assert result.payment_action == "no_action"
    assert result.communication_action == "skipped"
    assert result.final_status == "NO_ACTION"
    db.close()


def test_hard_decline_audit_shows_payment_blocked_and_communication_result(test_db_session):
    db = test_db_session()
    orchestrate_recovery(db, _make_event(event_id=37, error_reason="card_expired"), model=_fake_model())
    rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 37).order_by(AuditLog.id).all()
    final_row = next(r for r in rows if r.action == "orchestrator_final_status")
    assert "payment_action=no_action" in final_row.reason
    assert "communication_action=sent" in final_row.reason or "communication_action=fallback_used" in final_row.reason
    llm_row = next((r for r in rows if r.actor == "llm"), None)
    assert llm_row is not None
    db.close()


# ---------------------------------------------------------------------------
# FIX #1: promise-to-pay override, exercised through the real orchestrator
# ---------------------------------------------------------------------------

def test_valid_promise_overrides_model_timing(test_db_session):
    from datetime import date as _date

    from recovery.promise_service import record_customer_reply

    db = test_db_session()
    event_id = 40
    record_customer_reply(
        db, event_id=event_id, subscription_id="sub_promise_ok",
        customer_reply_text="I'll pay Friday when salary comes", today=_date(2026, 2, 24),  # a Tuesday
    )
    result = orchestrate_recovery(db, _make_event(event_id=event_id, subscription_id="sub_promise_ok"), model=_fake_model())
    assert result.promise_to_pay_applied is True
    assert result.selected_candidate_type == "promise_to_pay"
    assert result.original_candidate_type != "promise_to_pay"
    assert result.payment_action == "retry_scheduled"
    db.close()


def test_low_confidence_promise_does_not_override(test_db_session):
    from datetime import date as _date

    from recovery.promise_service import record_customer_reply

    db = test_db_session()
    event_id = 41
    # No weekday/relative-day keyword and no "pay"/"salary"/"will" -> the mock
    # provider's own confidence stays 0.0, well below DEFAULT_MIN_CONFIDENCE.
    record_customer_reply(
        db, event_id=event_id, subscription_id="sub_promise_low_conf",
        customer_reply_text="ok", today=_date(2026, 2, 24),
    )
    result = orchestrate_recovery(db, _make_event(event_id=event_id, subscription_id="sub_promise_low_conf"), model=_fake_model())
    assert result.promise_to_pay_applied is False
    assert result.selected_candidate_type != "promise_to_pay"
    db.close()


def test_promise_outside_horizon_falls_back_to_original_candidate(test_db_session):
    from datetime import date as _date

    from recovery.promise_service import record_customer_reply

    db = test_db_session()
    event_id = 42
    # 2026-02-24 is a Tuesday; asking for "Friday" 10 weeks out isn't
    # possible via the mock's weekday resolver (always <=7 days), so this
    # test constructs an out-of-horizon promise directly through the
    # validation+persistence layer instead of via a real reply.
    from app.models import PromiseToPay

    db.add(PromiseToPay(
        event_id=event_id, subscription_id="sub_promise_horizon",
        promised_date=FAILURE_TS + timedelta(days=20), confidence=0.9, channel="unspecified",
        status="VALID", status_reason="test-constructed", source_text_hash="deadbeef",
    ))
    db.commit()

    result = orchestrate_recovery(db, _make_event(event_id=event_id, subscription_id="sub_promise_horizon"), model=_fake_model())
    assert result.promise_to_pay_applied is False
    assert result.selected_candidate_type != "promise_to_pay"
    assert result.payment_action == "retry_scheduled"  # falls back to the original, still-valid candidate -- not blocked
    db.close()


def test_llm_failure_produces_no_fake_promise(test_db_session):
    from datetime import date as _date

    from recovery.promise_service import record_customer_reply

    db = test_db_session()
    promise, created = record_customer_reply(
        db, event_id=43, subscription_id="sub_promise_llm_fail",
        customer_reply_text="I'll pay Friday", today=_date(2026, 2, 24), client=_UnavailableClient(),
    )
    assert created is True
    assert promise.promised_date is None
    assert promise.status == "INVALID_DATE"
    db.close()


def test_policy_remains_deterministic_after_promise_parsing(test_db_session):
    # A promise changes TIMING, never the underlying model/policy tier that
    # was chosen -- decision_source is unaffected by the override.
    from datetime import date as _date

    from recovery.promise_service import record_customer_reply

    db1, db2 = test_db_session(), test_db_session()
    record_customer_reply(db1, event_id=44, subscription_id="sub_a", customer_reply_text="I'll pay Friday when salary comes", today=_date(2026, 2, 24))
    r1 = orchestrate_recovery(db1, _make_event(event_id=44, subscription_id="sub_a"), model=_fake_model())
    r2 = orchestrate_recovery(db2, _make_event(event_id=44, subscription_id="sub_b"), model=_fake_model())
    assert r1.decision_source == r2.decision_source
    assert r1.original_candidate_type == r2.original_candidate_type
    db1.close()
    db2.close()


# ---------------------------------------------------------------------------
# Full-system audit finding: RecoveryOutcome.__doc__ says this table is
# "shared by every domain (payment_failed included)", but orchestrate_recovery
# never wrote one -- payment_failed/subscription_payment_failed were invisible
# in the Track-03 outcome/economics dashboard views. Fixed additively (no
# existing field/behavior changed); these tests prove the fix and its honesty
# guarantee (never a fabricated recovered amount) and its idempotency.
# ---------------------------------------------------------------------------

def test_legacy_payment_failed_now_writes_a_recovery_outcome_row(test_db_session):
    from app.models import RecoveryOutcome

    db = test_db_session()
    orchestrate_recovery(db, _make_event(event_id=45, amount=750.0, error_reason="insufficient_fund"), model=_fake_model())
    outcome = db.query(RecoveryOutcome).filter(RecoveryOutcome.event_id == 45, RecoveryOutcome.event_type == "payment_failed").first()
    assert outcome is not None
    assert outcome.at_risk_amount == 750.0
    db.close()


def test_legacy_recovery_outcome_is_never_fabricated_as_recovered(test_db_session):
    from app.models import RecoveryOutcome

    db = test_db_session()
    orchestrate_recovery(db, _make_event(event_id=46, error_reason="insufficient_fund"), model=_fake_model())
    outcome = db.query(RecoveryOutcome).filter(RecoveryOutcome.event_id == 46, RecoveryOutcome.event_type == "payment_failed").first()
    assert outcome.recovery_status in ("PENDING", "NO_ACTION")  # never RECOVERED/PARTIALLY_RECOVERED/LOST for a live event
    assert outcome.recovered_amount is None
    assert outcome.confirmed_by == "unconfirmed_pending"
    db.close()


def test_legacy_recovery_outcome_reflects_no_action_for_unmapped_reason(test_db_session):
    # unmapped -> NO_ACTION candidate AND communication skipped (nothing
    # truthful to tell the customer for an unrecognized reason) -> the one
    # bucket where final_status is genuinely "NO_ACTION", not just blocked.
    from app.models import RecoveryOutcome

    db = test_db_session()
    result = orchestrate_recovery(db, _make_event(event_id=47, error_reason="totally_unrecognized_reason_xyz"), model=_fake_model())
    assert result.final_status == "NO_ACTION"
    outcome = db.query(RecoveryOutcome).filter(RecoveryOutcome.event_id == 47, RecoveryOutcome.event_type == "payment_failed").first()
    assert outcome is not None
    assert outcome.recovery_status == "NO_ACTION"
    db.close()


def test_legacy_recovery_outcome_is_idempotent_across_repeated_orchestration(test_db_session):
    # orchestrate_recovery can be invoked more than once for the same event_id
    # (e.g. a manual reprocess run) -- every other write in this function is
    # already idempotent; the outcome row must not silently duplicate.
    from app.models import RecoveryOutcome

    db = test_db_session()
    orchestrate_recovery(db, _make_event(event_id=48, error_reason="insufficient_fund"), model=_fake_model())
    orchestrate_recovery(db, _make_event(event_id=48, error_reason="insufficient_fund"), model=_fake_model())
    assert db.query(RecoveryOutcome).filter(RecoveryOutcome.event_id == 48, RecoveryOutcome.event_type == "payment_failed").count() == 1
    db.close()


# ---------------------------------------------------------------------------
# DEFER, DON'T TERMINATE (final pre-submission audit): contact-hours block
# defers communication instead of losing it outright.
# ---------------------------------------------------------------------------

def test_communication_deferred_when_candidate_falls_outside_contact_hours(test_db_session):
    # immediate = failure_timestamp + 1h = 2026-02-24 21:00 UTC = 02:30 IST
    # (next day) -- outside the default [09:00, 21:00) IST window. Forced via
    # _fake_model's default values (huge margin -> "immediate", CANDIDATE_TYPES[0]).
    late_failure_ts = datetime(2026, 2, 24, 20, 0, 0)
    db = test_db_session()
    result = orchestrate_recovery(db, _make_event(event_id=900, failure_timestamp=late_failure_ts, error_reason="insufficient_fund"), model=_fake_model())

    assert result.payment_action == "retry_scheduled"  # payment itself is unaffected by contact hours
    assert result.communication_action == "deferred"
    assert result.final_status == "COMMUNICATION_DEFERRED"
    assert result.communication_deferred_until is not None
    assert result.communication_deferred_until > late_failure_ts

    row = db.query(PolicyDecision).filter(PolicyDecision.event_id == 900).first()
    assert row.communication_deferred_until is not None
    assert row.communication_deferred_sent is False
    db.close()


def test_communication_deferred_never_fires_the_llm_at_decision_time(test_db_session):
    # A deferred communication must not have already called the LLM /
    # written an LLMInvocation row -- that only happens later, when
    # recovery/retry_sweep.py fires it.
    late_failure_ts = datetime(2026, 2, 24, 20, 0, 0)
    db = test_db_session()
    result = orchestrate_recovery(db, _make_event(event_id=901, failure_timestamp=late_failure_ts, error_reason="insufficient_fund"), model=_fake_model())
    assert result.communication_action == "deferred"
    assert result.llm_task_name is None
    assert db.query(LLMInvocation).filter(LLMInvocation.event_id == 901).count() == 0
    db.close()
