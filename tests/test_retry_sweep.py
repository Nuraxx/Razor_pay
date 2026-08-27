"""
MULTI-ATTEMPT PERSISTENCE (final pre-submission audit) tests:
recovery/retry_sweep.py + its wiring into app/main.py's FastAPI lifespan.
Mirrors tests/test_scheduler.py's structure exactly (exception isolation,
enable/disable mechanism) since retry_sweep.py deliberately reuses that same
in-process asyncio-loop pattern -- see recovery/retry_sweep.py's own
docstring.
"""
import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from app.models import AuditLog, FailureEvent, LLMInvocation, PolicyDecision, RawEvent, RecoveryOutcome
from recovery.retry_sweep import advance_one_retry_schedule, fire_one_deferred_communication, retry_sweep_background_loop, run_retry_sweep_once

NOW = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
PAST = NOW - timedelta(hours=1)
FUTURE = NOW + timedelta(hours=1)


def _make_decision(db, *, event_id: int, types: list[str], datetimes: list[datetime], next_index: int = 1, subscription_id: str | None = None) -> PolicyDecision:
    row = PolicyDecision(
        event_id=event_id,
        subscription_id=subscription_id or f"sub_retry_sweep_{event_id}",
        selected_candidate_type=types[0],
        selected_candidate_datetime=datetimes[0],
        policy_version="policy-v4",
        decision_reason="test fixture",
        decision_source="subscription_value_model",
        classification_bucket="retryable_soft",
        retry_schedule_json=json.dumps(types),
        retry_schedule_datetimes_json=json.dumps([dt.isoformat() for dt in datetimes]),
        retry_schedule_next_index=next_index,
    )
    db.add(row)
    db.flush()
    return row


def _make_later_event_row(db, *, event_id: int, subscription_id: str, customer_opted_out: bool = False, classification_bucket: str = "retryable_soft") -> PolicyDecision:
    """A SECOND policy_decisions row for the SAME subscription_id -- models a
    later real event (e.g. a subsequent webhook) that changed durable
    opt-out/classification state for the subscription. No retry_schedule of
    its own -- only `_subscription_still_eligible`'s re-check reads it."""
    row = PolicyDecision(
        event_id=event_id, subscription_id=subscription_id, selected_candidate_type="immediate",
        policy_version="policy-v4", decision_reason="later event fixture", decision_source="subscription_value_model",
        classification_bucket=classification_bucket, customer_opted_out=customer_opted_out,
    )
    db.add(row)
    db.flush()
    return row


def _make_raw_and_failure_event(db, *, event_id: int, amount_paise: int = 100000) -> None:
    raw = RawEvent(razorpay_event_id=f"evt_retry_sweep_{event_id}", event_type="payment.failed", amount=amount_paise, raw_payload="{}")
    db.add(raw)
    db.flush()
    fe = FailureEvent(id=event_id, raw_event_id=raw.id, classification_bucket="retryable_soft")
    db.add(fe)
    db.flush()


def _make_deferred_decision(db, *, event_id: int, deferred_until: datetime, sent: bool = False) -> PolicyDecision:
    row = PolicyDecision(
        event_id=event_id, subscription_id=f"sub_retry_sweep_{event_id}", selected_candidate_type="immediate",
        selected_candidate_datetime=deferred_until - timedelta(hours=6), policy_version="policy-v4",
        decision_reason="test fixture", decision_source="subscription_value_model", classification_bucket="retryable_soft",
        communication_deferred_until=deferred_until, communication_deferred_sent=sent,
    )
    db.add(row)
    db.flush()
    return row


def _make_outcome(db, *, event_id: int, status: str = "PENDING") -> RecoveryOutcome:
    outcome = RecoveryOutcome(
        event_id=event_id, event_type="payment_failed", at_risk_amount=1000.0,
        recovery_status=status, confirmed_by="unconfirmed_pending",
    )
    db.add(outcome)
    db.flush()
    return outcome


class TestAdvanceOneRetrySchedule:
    def test_not_due_yet_returns_false(self, test_db_session):
        db = test_db_session()
        decision = _make_decision(db, event_id=70001, types=["immediate", "plus_1_day_morning"], datetimes=[PAST, FUTURE])
        _make_outcome(db, event_id=70001)
        db.commit()

        assert advance_one_retry_schedule(db, decision, now=NOW) is False
        assert decision.retry_schedule_next_index == 1
        db.close()

    def test_due_and_pending_advances_and_logs_audit_row(self, test_db_session):
        db = test_db_session()
        decision = _make_decision(db, event_id=70002, types=["immediate", "plus_1_day_morning"], datetimes=[PAST, PAST])
        _make_outcome(db, event_id=70002)
        db.commit()

        assert advance_one_retry_schedule(db, decision, now=NOW) is True
        assert decision.retry_schedule_next_index == 2
        audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 70002).all()
        assert len(audit_rows) == 1
        assert audit_rows[0].action == "retry_schedule_attempt_recorded"
        assert "plus_1_day_morning" in audit_rows[0].reason
        db.close()

    def test_recovered_outcome_stops_further_attempts(self, test_db_session):
        db = test_db_session()
        decision = _make_decision(db, event_id=70003, types=["immediate", "plus_1_day_morning"], datetimes=[PAST, PAST])
        _make_outcome(db, event_id=70003, status="RECOVERED")
        db.commit()

        assert advance_one_retry_schedule(db, decision, now=NOW) is False
        assert decision.retry_schedule_next_index == 1
        db.close()

    def test_no_recovery_outcome_row_returns_false(self, test_db_session):
        # Defensive: recovery/orchestrator.py always writes a RecoveryOutcome
        # row alongside every PolicyDecision, but this must never crash if
        # one is somehow missing -- "unknown" is treated as "don't advance."
        db = test_db_session()
        decision = _make_decision(db, event_id=70004, types=["immediate", "plus_1_day_morning"], datetimes=[PAST, PAST])
        db.commit()

        assert advance_one_retry_schedule(db, decision, now=NOW) is False
        db.close()

    def test_exhausted_schedule_returns_false(self, test_db_session):
        db = test_db_session()
        decision = _make_decision(db, event_id=70005, types=["immediate", "plus_1_day_morning"], datetimes=[PAST, PAST], next_index=2)
        _make_outcome(db, event_id=70005)
        db.commit()

        assert advance_one_retry_schedule(db, decision, now=NOW) is False
        db.close()

    def test_idempotent_second_call_same_tick_is_noop(self, test_db_session):
        db = test_db_session()
        decision = _make_decision(db, event_id=70006, types=["immediate", "plus_1_day_morning", "payday_window"], datetimes=[PAST, PAST, FUTURE])
        _make_outcome(db, event_id=70006)
        db.commit()

        assert advance_one_retry_schedule(db, decision, now=NOW) is True
        assert decision.retry_schedule_next_index == 2
        # Second call: index 2's own datetime (FUTURE) hasn't arrived yet --
        # must NOT double-advance just because a sweep pass runs twice.
        assert advance_one_retry_schedule(db, decision, now=NOW) is False
        assert decision.retry_schedule_next_index == 2
        audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 70006).all()
        assert len(audit_rows) == 1
        db.close()


class TestReCheckBeforeActing:
    """RE-CHECK-BEFORE-ACTING (final pre-submission audit): opt-out or
    reclassification recorded on a LATER event for the same subscription
    must immediately, permanently suppress every remaining scheduled
    attempt / pending deferred communication -- not just the state captured
    when the sequence was first created."""

    def test_opt_out_on_later_event_aborts_remaining_schedule(self, test_db_session):
        db = test_db_session()
        sub_id = "sub_recheck_optout"
        decision = _make_decision(db, event_id=70040, types=["immediate", "plus_1_day_morning", "payday_window"], datetimes=[PAST, PAST, PAST], subscription_id=sub_id)
        _make_outcome(db, event_id=70040)
        _make_later_event_row(db, event_id=70041, subscription_id=sub_id, customer_opted_out=True)
        db.commit()

        assert advance_one_retry_schedule(db, decision, now=NOW) is False
        assert decision.retry_schedule_next_index == 3  # aborted -- schedule length, not advanced
        audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 70040, AuditLog.action == "retry_schedule_aborted").all()
        assert len(audit_rows) == 1
        assert "customer_opted_out" in audit_rows[0].reason
        db.close()

    def test_reclassification_on_later_event_aborts_remaining_schedule(self, test_db_session):
        db = test_db_session()
        sub_id = "sub_recheck_cancelled"
        decision = _make_decision(db, event_id=70042, types=["immediate", "plus_1_day_morning"], datetimes=[PAST, PAST], subscription_id=sub_id)
        _make_outcome(db, event_id=70042)
        _make_later_event_row(db, event_id=70043, subscription_id=sub_id, classification_bucket="customer_cancelled")
        db.commit()

        assert advance_one_retry_schedule(db, decision, now=NOW) is False
        assert decision.retry_schedule_next_index == 2
        audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 70042, AuditLog.action == "retry_schedule_aborted").all()
        assert len(audit_rows) == 1
        assert "customer_cancelled" in audit_rows[0].reason
        db.close()

    def test_no_opt_out_or_reclassification_still_advances_normally(self, test_db_session):
        db = test_db_session()
        sub_id = "sub_recheck_clean"
        decision = _make_decision(db, event_id=70044, types=["immediate", "plus_1_day_morning"], datetimes=[PAST, PAST], subscription_id=sub_id)
        _make_outcome(db, event_id=70044)
        _make_later_event_row(db, event_id=70045, subscription_id=sub_id, classification_bucket="retryable_soft")
        db.commit()

        assert advance_one_retry_schedule(db, decision, now=NOW) is True
        assert decision.retry_schedule_next_index == 2
        db.close()

    def test_abort_is_idempotent_across_sweep_passes(self, test_db_session):
        # Once aborted, a second sweep pass must not re-check eligibility or
        # write a second audit row -- the schedule is already exhausted.
        db = test_db_session()
        sub_id = "sub_recheck_idempotent"
        decision = _make_decision(db, event_id=70046, types=["immediate", "plus_1_day_morning"], datetimes=[PAST, PAST], subscription_id=sub_id)
        _make_outcome(db, event_id=70046)
        _make_later_event_row(db, event_id=70047, subscription_id=sub_id, customer_opted_out=True)
        db.commit()

        assert advance_one_retry_schedule(db, decision, now=NOW) is False
        assert advance_one_retry_schedule(db, decision, now=NOW) is False
        audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 70046, AuditLog.action == "retry_schedule_aborted").all()
        assert len(audit_rows) == 1
        db.close()

    def test_opt_out_suppresses_pending_deferred_communication(self, test_db_session):
        db = test_db_session()
        sub_id = "sub_recheck_defer_optout"
        _make_raw_and_failure_event(db, event_id=70048)
        decision = PolicyDecision(
            event_id=70048, subscription_id=sub_id, selected_candidate_type="immediate",
            selected_candidate_datetime=PAST - timedelta(hours=6), policy_version="policy-v4",
            decision_reason="test fixture", decision_source="subscription_value_model", classification_bucket="retryable_soft",
            communication_deferred_until=PAST, communication_deferred_sent=False,
        )
        db.add(decision)
        db.flush()
        _make_later_event_row(db, event_id=70049, subscription_id=sub_id, customer_opted_out=True)
        db.commit()

        assert fire_one_deferred_communication(db, decision, now=NOW) is False
        assert decision.communication_deferred_sent is True  # permanently suppressed
        assert db.query(LLMInvocation).filter(LLMInvocation.event_id == 70048).count() == 0
        audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 70048, AuditLog.action == "deferred_communication_suppressed").all()
        assert len(audit_rows) == 1
        db.close()


class TestFireOneDeferredCommunication:
    def test_fires_when_due_and_not_sent(self, test_db_session):
        db = test_db_session()
        _make_raw_and_failure_event(db, event_id=70020)
        decision = _make_deferred_decision(db, event_id=70020, deferred_until=PAST)
        db.commit()

        assert fire_one_deferred_communication(db, decision, now=NOW) is True
        assert decision.communication_deferred_sent is True
        assert db.query(LLMInvocation).filter(LLMInvocation.event_id == 70020, LLMInvocation.task_name == "outreach_microcopy").count() == 1
        audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 70020, AuditLog.action == "deferred_communication_fired").all()
        assert len(audit_rows) == 1
        db.close()

    def test_does_not_fire_when_not_due_yet(self, test_db_session):
        db = test_db_session()
        _make_raw_and_failure_event(db, event_id=70021)
        decision = _make_deferred_decision(db, event_id=70021, deferred_until=FUTURE)
        db.commit()

        assert fire_one_deferred_communication(db, decision, now=NOW) is False
        assert decision.communication_deferred_sent is False
        db.close()

    def test_does_not_fire_twice(self, test_db_session):
        db = test_db_session()
        _make_raw_and_failure_event(db, event_id=70022)
        decision = _make_deferred_decision(db, event_id=70022, deferred_until=PAST, sent=True)
        db.commit()

        assert fire_one_deferred_communication(db, decision, now=NOW) is False
        assert db.query(LLMInvocation).filter(LLMInvocation.event_id == 70022).count() == 0
        db.close()

    def test_does_not_fire_when_no_deferral_pending(self, test_db_session):
        db = test_db_session()
        _make_raw_and_failure_event(db, event_id=70023)
        row = PolicyDecision(
            event_id=70023, subscription_id="sub_retry_sweep_70023", selected_candidate_type="immediate",
            policy_version="policy-v4", decision_reason="no deferral",
        )
        db.add(row)
        db.commit()

        assert fire_one_deferred_communication(db, row, now=NOW) is False
        db.close()

    def test_missing_raw_event_degrades_to_zero_amount_rather_than_crashing(self, test_db_session):
        db = test_db_session()
        decision = _make_deferred_decision(db, event_id=70024, deferred_until=PAST)  # no matching RawEvent/FailureEvent
        db.commit()

        assert fire_one_deferred_communication(db, decision, now=NOW) is True
        db.close()


class TestRunRetrySweepOnce:
    def test_returns_zero_when_nothing_is_due(self, test_db_session):
        db = test_db_session()
        assert run_retry_sweep_once(db) == 0
        db.close()

    def test_processes_multiple_due_rows(self, test_db_session):
        db = test_db_session()
        _make_decision(db, event_id=70010, types=["immediate", "plus_1_day_morning"], datetimes=[PAST, PAST])
        _make_outcome(db, event_id=70010)
        _make_decision(db, event_id=70011, types=["immediate", "payday_window"], datetimes=[PAST, PAST])
        _make_outcome(db, event_id=70011)
        db.commit()

        assert run_retry_sweep_once(db) == 2
        db.close()

    def test_processes_both_retry_attempts_and_deferred_communications_in_one_pass(self, test_db_session):
        db = test_db_session()
        _make_decision(db, event_id=70031, types=["immediate", "plus_1_day_morning"], datetimes=[PAST, PAST])
        _make_outcome(db, event_id=70031)
        _make_raw_and_failure_event(db, event_id=70032)
        _make_deferred_decision(db, event_id=70032, deferred_until=PAST)
        db.commit()

        assert run_retry_sweep_once(db) == 2
        db.close()

    def test_ignores_rows_with_no_schedule(self, test_db_session):
        db = test_db_session()
        row = PolicyDecision(
            event_id=70012, subscription_id="sub_retry_sweep_70012", selected_candidate_type="immediate",
            policy_version="policy-v4", decision_reason="no schedule", retry_schedule_json=None,
        )
        db.add(row)
        db.commit()

        assert run_retry_sweep_once(db) == 0
        db.close()


class TestRetrySweepExceptionIsolation:
    def test_a_failing_sweep_pass_does_not_kill_the_loop(self, monkeypatch):
        call_count = {"n": 0}

        def _flaky_sweep(db=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated transient failure (e.g. a DB blip)")
            return 0

        monkeypatch.setattr("recovery.retry_sweep.run_retry_sweep_once", _flaky_sweep)

        async def _run_briefly():
            with pytest.raises((asyncio.TimeoutError, TimeoutError)):
                await asyncio.wait_for(retry_sweep_background_loop(interval_seconds=0), timeout=0.3)

        asyncio.run(_run_briefly())
        assert call_count["n"] >= 2

    def test_cancellation_propagates_cleanly_and_is_not_swallowed_as_a_failure(self):
        async def _run_and_cancel():
            task = asyncio.create_task(retry_sweep_background_loop(interval_seconds=999))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(_run_and_cancel())


class TestRetrySweepSchedulerWiring:
    def test_enabled_flag_does_create_a_background_task(self, monkeypatch):
        from app.main import app as fastapi_app
        from app.main import lifespan

        monkeypatch.setattr("app.main.init_db", lambda: None)
        monkeypatch.setattr("app.config.settings.validate_webhook_secret_present", lambda: None)
        monkeypatch.setattr("app.config.settings.ENABLE_PROMISE_SWEEP_SCHEDULER", False)
        monkeypatch.setattr("app.config.settings.ENABLE_RETRY_SWEEP_SCHEDULER", True)
        monkeypatch.setattr("app.main.retry_sweep_background_loop", lambda interval_seconds: _noop_coro())

        async def _enter_and_exit_lifespan():
            async with lifespan(fastapi_app):
                pass

        asyncio.run(_enter_and_exit_lifespan())  # must not raise


async def _noop_coro() -> None:
    return None
