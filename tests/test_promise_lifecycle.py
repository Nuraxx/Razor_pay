"""
Track-03 tests: recovery/promise_lifecycle.py -- the promise-to-pay
lifecycle dimension (PROMISED/FULFILLED/BROKEN/EXPIRED/CANCELLED), layered
on top of the existing, UNCHANGED policy/promise_to_pay.py validation-time
status. Also cross-checks that promise_service.py's own contract and
tests/test_promise_to_pay.py's fixtures are untouched by this module.
"""
from datetime import date, datetime, timedelta

from app.models import AuditLog, PromiseToPay, RevenueRiskEvent
from policy.promise_to_pay import STATUS_VALID
from recovery.promise_lifecycle import (
    LIFECYCLE_BROKEN,
    LIFECYCLE_CANCELLED,
    LIFECYCLE_FULFILLED,
    cancel_promise,
    confirm_promise_fulfilled,
    get_promise_outcome,
    mark_broken_promises,
)
from recovery.promise_service import get_active_promise, record_customer_reply

NOW = datetime(2026, 8, 25, 10, 0, 0)


def _make_valid_promise(db, *, event_id: int, subscription_id: str, promised_date: datetime, source_text_hash: str = "hash1") -> PromiseToPay:
    """Direct construction (not via record_customer_reply): a promise whose
    date has ALREADY passed while status=VALID is a state only real time
    passing after a legitimate creation could produce -- record_customer_reply
    itself would classify a past date as EXPIRED at validation time, never VALID."""
    promise = PromiseToPay(
        event_id=event_id, subscription_id=subscription_id, promised_date=promised_date,
        confidence=0.9, channel="upi_autopay", status=STATUS_VALID, status_reason="test_fixture",
        source_text_hash=source_text_hash,
    )
    db.add(promise)
    db.flush()
    db.commit()
    return promise


class TestMarkBrokenPromises:
    def test_past_due_valid_promise_is_marked_broken(self, test_db_session):
        db = test_db_session()
        promise = _make_valid_promise(db, event_id=1, subscription_id="sub_a", promised_date=NOW - timedelta(days=1))

        broken = mark_broken_promises(db, as_of=NOW)

        assert len(broken) == 1
        outcome = broken[0]
        assert outcome.promise_to_pay_id == promise.id
        assert outcome.lifecycle_status == LIFECYCLE_BROKEN
        assert outcome.triggered_reevaluation is True
        assert outcome.reevaluation_event_id is not None
        db.close()

    def test_future_dated_valid_promise_is_not_swept(self, test_db_session):
        db = test_db_session()
        _make_valid_promise(db, event_id=1, subscription_id="sub_a", promised_date=NOW + timedelta(days=5))
        broken = mark_broken_promises(db, as_of=NOW)
        assert broken == []
        db.close()

    def test_opens_a_promise_to_pay_broken_revenue_risk_event(self, test_db_session):
        db = test_db_session()
        promise = _make_valid_promise(db, event_id=1, subscription_id="sub_a", promised_date=NOW - timedelta(days=1))
        broken = mark_broken_promises(db, as_of=NOW)

        rre = db.query(RevenueRiskEvent).filter(RevenueRiskEvent.id == broken[0].reevaluation_event_id).first()
        assert rre is not None
        assert rre.event_type == "promise_to_pay_broken"
        assert rre.customer_ref == "sub_a"
        assert rre.idempotency_key == f"promise_to_pay_broken:{promise.id}"
        db.close()

    def test_idempotent_second_sweep_does_not_double_process(self, test_db_session):
        db = test_db_session()
        _make_valid_promise(db, event_id=1, subscription_id="sub_a", promised_date=NOW - timedelta(days=1))

        first = mark_broken_promises(db, as_of=NOW)
        second = mark_broken_promises(db, as_of=NOW)

        assert len(first) == 1
        assert second == []  # already resolved -- not re-broken, no duplicate revenue_risk_events row
        assert db.query(RevenueRiskEvent).filter(RevenueRiskEvent.event_type == "promise_to_pay_broken").count() == 1
        db.close()

    def test_a_promise_already_confirmed_fulfilled_is_never_marked_broken(self, test_db_session):
        db = test_db_session()
        promise = _make_valid_promise(db, event_id=1, subscription_id="sub_a", promised_date=NOW - timedelta(days=1))
        confirm_promise_fulfilled(db, promise_to_pay_id=promise.id, resolved_by="demo_synthetic")

        broken = mark_broken_promises(db, as_of=NOW)

        assert broken == []
        outcome = get_promise_outcome(db, promise.id)
        assert outcome.lifecycle_status == LIFECYCLE_FULFILLED  # unchanged by the sweep
        db.close()

    def test_multiple_broken_promises_each_get_their_own_event(self, test_db_session):
        db = test_db_session()
        _make_valid_promise(db, event_id=1, subscription_id="sub_a", promised_date=NOW - timedelta(days=1), source_text_hash="h1")
        _make_valid_promise(db, event_id=2, subscription_id="sub_b", promised_date=NOW - timedelta(days=2), source_text_hash="h2")

        broken = mark_broken_promises(db, as_of=NOW)

        assert len(broken) == 2
        assert broken[0].reevaluation_event_id != broken[1].reevaluation_event_id
        db.close()

    def test_writes_an_audit_row(self, test_db_session):
        db = test_db_session()
        promise = _make_valid_promise(db, event_id=1, subscription_id="sub_a", promised_date=NOW - timedelta(days=1))
        mark_broken_promises(db, as_of=NOW)

        rows = db.query(AuditLog).filter(AuditLog.actor == "promise_lifecycle", AuditLog.action == "promise_broken").all()
        assert len(rows) == 1
        assert f"promises_to_pay.id={promise.id}" in rows[0].reason
        db.close()


class TestFulfilledAndCancelled:
    def test_confirm_fulfilled_is_idempotent(self, test_db_session):
        db = test_db_session()
        promise = _make_valid_promise(db, event_id=1, subscription_id="sub_a", promised_date=NOW - timedelta(days=1))
        first = confirm_promise_fulfilled(db, promise_to_pay_id=promise.id, resolved_by="demo_synthetic")
        second = confirm_promise_fulfilled(db, promise_to_pay_id=promise.id, resolved_by="demo_synthetic")
        assert first.id == second.id
        assert first.lifecycle_status == LIFECYCLE_FULFILLED

    def test_cancel_promise_records_cancelled_status(self, test_db_session):
        db = test_db_session()
        promise = _make_valid_promise(db, event_id=1, subscription_id="sub_a", promised_date=NOW + timedelta(days=5))
        outcome = cancel_promise(db, promise_to_pay_id=promise.id, reason="subscription_cancelled")
        assert outcome.lifecycle_status == LIFECYCLE_CANCELLED
        assert outcome.status_reason == "subscription_cancelled"

    def test_cancel_is_idempotent(self, test_db_session):
        db = test_db_session()
        promise = _make_valid_promise(db, event_id=1, subscription_id="sub_a", promised_date=NOW + timedelta(days=5))
        first = cancel_promise(db, promise_to_pay_id=promise.id, reason="a")
        second = cancel_promise(db, promise_to_pay_id=promise.id, reason="b")
        assert first.id == second.id
        assert first.status_reason == "a"  # first resolution wins, not silently overwritten


class TestPromiseToPayContractUntouched:
    """Cross-checks with the existing, unmodified promise_service.py /
    policy/promise_to_pay.py -- this module must never interfere with them."""

    def test_promises_to_pay_status_column_is_never_written_by_this_module(self, test_db_session):
        db = test_db_session()
        promise = _make_valid_promise(db, event_id=1, subscription_id="sub_a", promised_date=NOW - timedelta(days=1))
        mark_broken_promises(db, as_of=NOW)
        db.refresh(promise)
        assert promise.status == STATUS_VALID  # untouched -- the lifecycle fact lives entirely in PromiseOutcome

    def test_get_active_promise_still_works_normally_via_record_customer_reply(self, test_db_session):
        db = test_db_session()
        promise, created = record_customer_reply(
            db, event_id=42, subscription_id="sub_normal", customer_reply_text="I'll pay tomorrow via UPI", today=date(2026, 8, 24),
        )
        assert created is True
        active = get_active_promise(db, 42)
        assert active is not None
        assert active.id == promise.id
        db.close()
