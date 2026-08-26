"""
Hardening-pass tests: recovery/promise_sweep.py -- the single shared
"detect broken promises, then route each through the real orchestrator"
function used by both scripts/sweep_promise_lifecycle.py (manual CLI) and
recovery/scheduler.py (automatic background loop). Proves the full pipeline
(mark_broken_promises -> policy -> compliance -> recovery -> communication ->
audit) runs end to end, is idempotent across repeated calls, and creates
exactly one RevenueRiskEvent per broken promise.
"""
from datetime import datetime, timedelta

from app.models import AuditLog, PolicyDecision, PromiseToPay, RevenueRiskEvent
from policy.policy_decision_store import REVENUE_DOMAIN_EVENT_ID_OFFSET
from policy.promise_to_pay import STATUS_VALID
from recovery.promise_lifecycle import LIFECYCLE_BROKEN, get_promise_outcome
from recovery.promise_sweep import sweep_and_orchestrate_broken_promises
from recovery.revenue_schemas import RevenueRecoveryResult

NOW = datetime(2026, 8, 25, 10, 0, 0)


def _make_valid_promise(db, *, event_id: int, subscription_id: str, promised_date: datetime, source_text_hash: str = "sweep_hash1") -> PromiseToPay:
    promise = PromiseToPay(
        event_id=event_id, subscription_id=subscription_id, promised_date=promised_date,
        confidence=0.9, channel="upi_autopay", status=STATUS_VALID, status_reason="test_fixture",
        source_text_hash=source_text_hash,
    )
    db.add(promise)
    db.flush()
    db.commit()
    return promise


class TestSweepAndOrchestrate:
    def test_due_promise_becomes_broken_and_is_orchestrated(self, test_db_session):
        db = test_db_session()
        promise = _make_valid_promise(db, event_id=1, subscription_id="sub_sweep_a", promised_date=NOW - timedelta(days=2))

        results = sweep_and_orchestrate_broken_promises(db, as_of=NOW)

        assert len(results) == 1
        assert isinstance(results[0], RevenueRecoveryResult)
        assert results[0].event_type == "promise_to_pay_broken"

        outcome = get_promise_outcome(db, promise.id)
        assert outcome.lifecycle_status == LIFECYCLE_BROKEN
        db.close()

    def test_broken_promise_creates_exactly_one_revenue_risk_event(self, test_db_session):
        db = test_db_session()
        _make_valid_promise(db, event_id=2, subscription_id="sub_sweep_b", promised_date=NOW - timedelta(days=3))

        sweep_and_orchestrate_broken_promises(db, as_of=NOW)

        assert db.query(RevenueRiskEvent).filter(RevenueRiskEvent.event_type == "promise_to_pay_broken").count() == 1
        db.close()

    def test_broken_promise_goes_through_policy_compliance_and_audit(self, test_db_session):
        db = test_db_session()
        _make_valid_promise(db, event_id=3, subscription_id="sub_sweep_c", promised_date=NOW - timedelta(days=1))

        results = sweep_and_orchestrate_broken_promises(db, as_of=NOW)
        result = results[0]

        rre = db.query(RevenueRiskEvent).filter(RevenueRiskEvent.event_type == "promise_to_pay_broken").first()
        stored_id = rre.id + REVENUE_DOMAIN_EVENT_ID_OFFSET

        # policy: a real decision was persisted (not skipped/inert)
        decision = db.query(PolicyDecision).filter(PolicyDecision.event_id == stored_id).first()
        assert decision is not None
        assert decision.decision_source == "rule_promise_to_pay_broken"

        # compliance: a real verdict was reached, not silently bypassed
        assert result.payment_verdict in ("ALLOWED", "BLOCKED", "HUMAN_REVIEW")

        # audit: the pipeline's own audit rows exist, keyed correctly
        actions = {row.action for row in db.query(AuditLog).filter(AuditLog.failure_event_id == stored_id).all()}
        assert "revenue_orchestrator_final_status" in actions
        db.close()

    def test_second_sweep_pass_does_nothing(self, test_db_session):
        db = test_db_session()
        _make_valid_promise(db, event_id=4, subscription_id="sub_sweep_d", promised_date=NOW - timedelta(days=1))

        first = sweep_and_orchestrate_broken_promises(db, as_of=NOW)
        second = sweep_and_orchestrate_broken_promises(db, as_of=NOW)

        assert len(first) == 1
        assert len(second) == 0  # nothing newly broken -- already resolved
        assert db.query(RevenueRiskEvent).filter(RevenueRiskEvent.event_type == "promise_to_pay_broken").count() == 1
        db.close()

    def test_multiple_broken_promises_are_all_processed_independently(self, test_db_session):
        db = test_db_session()
        _make_valid_promise(db, event_id=5, subscription_id="sub_sweep_e1", promised_date=NOW - timedelta(days=1), source_text_hash="sweep_hash_e1")
        _make_valid_promise(db, event_id=6, subscription_id="sub_sweep_e2", promised_date=NOW - timedelta(days=2), source_text_hash="sweep_hash_e2")

        results = sweep_and_orchestrate_broken_promises(db, as_of=NOW)

        assert len(results) == 2
        assert db.query(RevenueRiskEvent).filter(RevenueRiskEvent.event_type == "promise_to_pay_broken").count() == 2
        db.close()

    def test_no_due_promises_returns_empty_list(self, test_db_session):
        db = test_db_session()
        _make_valid_promise(db, event_id=7, subscription_id="sub_sweep_f", promised_date=NOW + timedelta(days=5))

        results = sweep_and_orchestrate_broken_promises(db, as_of=NOW)

        assert results == []
        db.close()
