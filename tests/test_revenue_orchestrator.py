"""
Track-03 tests: recovery/revenue_orchestrator.py -- end-to-end orchestration
for the 4 new event types. Mirrors tests/test_orchestrator.py's coverage
style: idempotency, compliance-block, NO_ACTION, HUMAN_REVIEW, LLM-failure
fallback, audit-trail, RecoveryOutcome creation. recovery/orchestrator.py's
own tests are untouched -- this file proves the NEW module works correctly
in complete isolation from it.
"""
from datetime import datetime

from app.models import AuditLog, LLMInvocation, PolicyDecision, RecoveryOutcome
from llm.client import LLMClient, LLMProviderError
from policy.decision_engine import NO_ACTION
from policy.policy_decision_store import REVENUE_DOMAIN_EVENT_ID_OFFSET
from recovery.revenue_orchestrator import orchestrate_revenue_event
from recovery.revenue_schemas import RevenueRiskEventInput
from recovery.voice import MockVoiceProvider
from tests.test_llm import _RaisingClient

NOW = datetime(2026, 8, 25, 10, 0, 0)


def _checkout_event(**overrides) -> RevenueRiskEventInput:
    base = dict(
        event_type="checkout_abandoned", event_id=1, customer_ref="cust_1", occurred_at=NOW, amount=500.0,
        domain_context={"cart_amount": 500.0, "inactivity_minutes": 90.0, "previous_outreach_count": 0},
    )
    base.update(overrides)
    return RevenueRiskEventInput(**base)


class TestCheckoutOrchestrationHappyPath:
    def test_eligible_checkout_gets_reminder_communication_sent(self, test_db_session):
        db = test_db_session()
        result = orchestrate_revenue_event(db, _checkout_event())
        assert result.selected_candidate_type == "reminder"
        assert result.payment_verdict == "ALLOWED"
        assert result.primary_action == "action_scheduled"
        assert result.communication_action == "sent"
        assert result.final_status == "COMMUNICATION_ALLOWED"
        assert result.llm_success is True
        db.close()

    def test_writes_recovery_outcome_pending_unconfirmed(self, test_db_session):
        db = test_db_session()
        orchestrate_revenue_event(db, _checkout_event())
        outcome = db.query(RecoveryOutcome).filter(RecoveryOutcome.event_id == 1, RecoveryOutcome.event_type == "checkout_abandoned").first()
        assert outcome is not None
        assert outcome.recovery_status == "PENDING"
        assert outcome.recovered_amount is None
        assert outcome.confirmed_by == "unconfirmed_pending"
        db.close()


class TestNoActionAndWait:
    def test_cart_below_minimum_is_no_action_and_skips_communication(self, test_db_session):
        db = test_db_session()
        result = orchestrate_revenue_event(db, _checkout_event(event_id=2, domain_context={"cart_amount": 1.0, "inactivity_minutes": 90.0}))
        assert result.selected_candidate_type == NO_ACTION
        assert result.primary_action == "no_action"
        assert result.communication_action == "skipped"
        assert result.final_status == "NO_ACTION"
        db.close()

    def test_stalled_wait_candidate_is_treated_as_no_action_not_blocked(self, test_db_session):
        db = test_db_session()
        result = orchestrate_revenue_event(db, _checkout_event(event_id=3, domain_context={"cart_amount": 500.0, "inactivity_minutes": 30.0}))
        assert result.selected_candidate_type == "wait"
        assert result.primary_action == "no_action"  # NOT "blocked" -- nothing went wrong, it's just too early
        assert result.payment_verdict == "ALLOWED"  # structurally valid (has a real datetime), just not acted on yet
        assert result.final_status == "NO_ACTION"
        assert result.communication_action == "skipped"
        db.close()


class TestIdempotency:
    def test_replaying_same_event_id_does_not_duplicate_anything(self, test_db_session):
        db = test_db_session()
        orchestrate_revenue_event(db, _checkout_event(event_id=4))
        orchestrate_revenue_event(db, _checkout_event(event_id=4))

        assert db.query(PolicyDecision).filter(PolicyDecision.event_id == 4 + REVENUE_DOMAIN_EVENT_ID_OFFSET).count() == 1
        assert db.query(LLMInvocation).filter(LLMInvocation.event_id == 4 + REVENUE_DOMAIN_EVENT_ID_OFFSET).count() == 1
        assert db.query(RecoveryOutcome).filter(RecoveryOutcome.event_id == 4).count() == 1
        db.close()

    def test_second_call_reports_communication_blocked_as_duplicate(self, test_db_session):
        db = test_db_session()
        orchestrate_revenue_event(db, _checkout_event(event_id=5))
        second = orchestrate_revenue_event(db, _checkout_event(event_id=5))
        assert second.communication_verdict == "BLOCKED"
        assert "duplicate_communication_action_blocked" in second.communication_reason
        db.close()


class TestComplianceBlock:
    def test_customer_opted_out_blocks_communication_only(self, test_db_session):
        db = test_db_session()
        result = orchestrate_revenue_event(db, _checkout_event(event_id=6, customer_opted_out=True))
        assert result.payment_verdict == "ALLOWED"
        assert result.communication_verdict == "BLOCKED"
        assert result.communication_action == "blocked"
        assert result.final_status == "COMMUNICATION_BLOCKED"
        db.close()

    def test_required_fields_missing_blocks_everything(self, test_db_session):
        db = test_db_session()
        result = orchestrate_revenue_event(db, _checkout_event(event_id=7, required_fields_present=False))
        assert result.payment_verdict == "BLOCKED"
        assert result.communication_verdict == "BLOCKED"
        assert result.final_status == "RETRY_BLOCKED"
        db.close()


class TestHumanReviewRouting:
    def test_disputed_receivable_routes_to_human_review_and_holds_communication(self, test_db_session):
        db = test_db_session()
        event = RevenueRiskEventInput(
            event_type="receivable_overdue", event_id=8, customer_ref="acct_1", occurred_at=NOW, amount=25000.0,
            domain_context={"days_overdue": 60, "is_disputed": True},
        )
        result = orchestrate_revenue_event(db, event)
        assert result.selected_candidate_type == "human_handoff"
        assert result.payment_verdict == "HUMAN_REVIEW"
        assert result.communication_verdict == "HUMAN_REVIEW"
        assert result.primary_action == "human_review"
        assert result.communication_action == "skipped"  # held pending a human, never auto-sent
        assert result.final_status == "HUMAN_REVIEW"
        db.close()


class TestLLMFailureNeverAffectsPrimaryAction:
    def test_broken_llm_client_falls_back_and_primary_action_is_unaffected(self, test_db_session):
        db = test_db_session()
        broken_client = _RaisingClient(LLMProviderError("simulated_outage"))
        result = orchestrate_revenue_event(db, _checkout_event(event_id=9), llm_client=broken_client)

        assert result.selected_candidate_type == "reminder"
        assert result.payment_verdict == "ALLOWED"
        assert result.primary_action == "action_scheduled"  # UNAFFECTED by the LLM outage
        assert result.communication_action == "fallback_used"
        assert result.llm_success is False
        assert result.final_status == "LLM_FALLBACK"
        db.close()

    def test_same_decision_with_working_or_broken_llm_client(self, test_db_session):
        db = test_db_session()
        broken_result = orchestrate_revenue_event(db, _checkout_event(event_id=10), llm_client=_RaisingClient(LLMProviderError("outage")))

        db2 = test_db_session()
        working_result = orchestrate_revenue_event(db2, _checkout_event(event_id=11, customer_ref="cust_working"))

        assert broken_result.selected_candidate_type == working_result.selected_candidate_type
        assert broken_result.payment_verdict == working_result.payment_verdict
        assert broken_result.primary_action == working_result.primary_action
        db.close()
        db2.close()


class TestReceivablesEscalationDeterministic:
    def test_high_overdue_escalates_and_llm_never_decides_escalation_level(self, test_db_session):
        db = test_db_session()
        event = RevenueRiskEventInput(
            event_type="receivable_overdue", event_id=12, customer_ref="acct_2", occurred_at=NOW, amount=50000.0,
            domain_context={"days_overdue": 45},
        )
        broken_client = _RaisingClient(LLMProviderError("outage"))
        result = orchestrate_revenue_event(db, event, llm_client=broken_client)
        assert result.selected_candidate_type == "escalation"  # unaffected by the broken LLM client
        assert result.payment_verdict == "ALLOWED"
        db.close()


class TestMandateOrchestration:
    def test_mandate_failed_starts_at_attempt_1(self, test_db_session):
        db = test_db_session()
        event = RevenueRiskEventInput(event_type="mandate_failed", event_id=13, customer_ref="sub_mandate", occurred_at=NOW, amount=1000.0)
        result = orchestrate_revenue_event(db, event)
        assert result.selected_candidate_type == "attempt_1"
        assert result.payment_verdict == "ALLOWED"
        db.close()


class TestVoiceChannel:
    def test_voice_channel_generates_script_and_places_a_mock_call(self, test_db_session):
        db = test_db_session()
        event = _checkout_event(event_id=14, channel="voice")
        result = orchestrate_revenue_event(db, event, voice_provider=MockVoiceProvider())

        assert result.llm_task_name == "voice_script_generation"
        assert result.communication_action == "sent"
        assert result.voice_call_result is not None
        assert result.voice_call_result.attempted is True
        assert result.voice_call_result.audio_available is False

        invocation = db.query(LLMInvocation).filter(LLMInvocation.event_id == 14 + REVENUE_DOMAIN_EVENT_ID_OFFSET).first()
        assert invocation.task_name == "voice_script_generation"

        voice_audit = db.query(AuditLog).filter(AuditLog.actor == "voice", AuditLog.failure_event_id == 14 + REVENUE_DOMAIN_EVENT_ID_OFFSET).first()
        assert voice_audit is not None
        db.close()

    def test_text_channel_never_calls_the_voice_provider(self, test_db_session):
        db = test_db_session()

        class _ExplodingVoiceProvider(MockVoiceProvider):
            def place_call(self, script_text, customer_ref):
                raise AssertionError("voice provider must not be called for a text-channel event")

        result = orchestrate_revenue_event(db, _checkout_event(event_id=15), voice_provider=_ExplodingVoiceProvider())
        assert result.voice_call_result is None
        db.close()


class TestAuditTrail:
    def test_writes_compliance_and_final_status_audit_rows(self, test_db_session):
        db = test_db_session()
        orchestrate_revenue_event(db, _checkout_event(event_id=16))
        actions = {row.action for row in db.query(AuditLog).filter(AuditLog.failure_event_id == 16 + REVENUE_DOMAIN_EVENT_ID_OFFSET).all()}
        assert "revenue_orchestrator_compliance" in actions
        assert "revenue_orchestrator_final_status" in actions
        db.close()
