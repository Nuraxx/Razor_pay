"""
FIX #1 tests: policy/promise_to_pay.py (deterministic validation, pure) and
recovery/promise_service.py (parse -> validate -> persist, DB-backed).

Orchestrator-level override behavior (a valid promise actually changing a
real recovery decision) is tested end-to-end in tests/test_orchestrator.py;
this file covers the validation rules and the persistence/idempotency/
supersession layer directly and in isolation.
"""
from datetime import date, datetime, timedelta

import pytest

from app.models import PromiseToPay
from llm.client import LLMClient, LLMProviderError
from policy.promise_to_pay import (
    DEFAULT_MIN_CONFIDENCE,
    STATUS_EXPIRED,
    STATUS_INVALID_DATE,
    STATUS_LOW_CONFIDENCE,
    STATUS_VALID,
    validate_promise,
)
from recovery.promise_service import get_active_promise, record_customer_reply

TODAY = date(2026, 8, 24)  # a Monday, matching tests/test_llm.py's own reference date


# ---------------------------------------------------------------------------
# policy/promise_to_pay.py::validate_promise -- pure, deterministic
# ---------------------------------------------------------------------------

def test_valid_promise():
    result = validate_promise(parsed_date="2026-08-28", confidence=0.85, channel="upi_autopay", today=TODAY)
    assert result.status == STATUS_VALID
    assert result.promised_datetime == datetime(2026, 8, 28, 0, 0)


def test_missing_date():
    result = validate_promise(parsed_date=None, confidence=0.9, channel="unspecified", today=TODAY)
    assert result.status == STATUS_INVALID_DATE
    assert result.promised_datetime is None


def test_invalid_date_string():
    result = validate_promise(parsed_date="not-a-date", confidence=0.9, channel="unspecified", today=TODAY)
    assert result.status == STATUS_INVALID_DATE
    assert result.promised_datetime is None


def test_past_date():
    result = validate_promise(parsed_date="2026-08-20", confidence=0.9, channel="unspecified", today=TODAY)
    assert result.status == STATUS_EXPIRED
    assert result.promised_datetime is None


def test_same_day_date_is_not_a_future_promise():
    result = validate_promise(parsed_date=TODAY.isoformat(), confidence=0.9, channel="unspecified", today=TODAY)
    assert result.status == STATUS_EXPIRED
    assert result.promised_datetime is None


def test_low_confidence_promise():
    result = validate_promise(parsed_date="2026-08-28", confidence=0.2, channel="unspecified", today=TODAY)
    assert result.status == STATUS_LOW_CONFIDENCE
    assert result.promised_datetime is None


def test_confidence_exactly_at_threshold_is_not_low_confidence():
    result = validate_promise(parsed_date="2026-08-28", confidence=DEFAULT_MIN_CONFIDENCE, channel="unspecified", today=TODAY)
    assert result.status == STATUS_VALID


def test_invalid_channel_raises_not_a_new_status():
    with pytest.raises(ValueError):
        validate_promise(parsed_date="2026-08-28", confidence=0.9, channel="bitcoin", today=TODAY)


def test_configurable_min_confidence_threshold():
    result = validate_promise(parsed_date="2026-08-28", confidence=0.5, channel="unspecified", today=TODAY, min_confidence=0.4)
    assert result.status == STATUS_VALID


# ---------------------------------------------------------------------------
# recovery/promise_service.py::record_customer_reply -- DB-backed
# ---------------------------------------------------------------------------

def test_record_valid_reply_persists_a_valid_promise(test_db_session):
    db = test_db_session()
    promise, created = record_customer_reply(
        db, event_id=1, subscription_id="sub_a", customer_reply_text="I'll pay Friday when salary comes", today=TODAY,
    )
    assert created is True
    assert promise.status == STATUS_VALID
    assert promise.promised_date is not None
    assert promise.subscription_id == "sub_a"
    db.close()


def test_no_raw_reply_text_is_ever_stored(test_db_session):
    db = test_db_session()
    secret_text = "I'll pay Friday, my card number is 4111111111111111"
    promise, _ = record_customer_reply(db, event_id=2, subscription_id="sub_b", customer_reply_text=secret_text, today=TODAY)
    row = db.query(PromiseToPay).filter(PromiseToPay.id == promise.id).first()
    for column in PromiseToPay.__table__.columns:
        value = str(getattr(row, column.name))
        assert "4111111111111111" not in value
    db.close()


def test_duplicate_exact_reply_is_idempotent(test_db_session):
    db = test_db_session()
    text = "I'll pay Friday when salary comes"
    first, created1 = record_customer_reply(db, event_id=3, subscription_id="sub_c", customer_reply_text=text, today=TODAY)
    second, created2 = record_customer_reply(db, event_id=3, subscription_id="sub_c", customer_reply_text=text, today=TODAY)
    assert created1 is True
    assert created2 is False
    assert first.id == second.id
    assert db.query(PromiseToPay).filter(PromiseToPay.event_id == 3).count() == 1
    db.close()


def test_distinct_reply_for_same_event_supersedes_the_previous_one(test_db_session):
    db = test_db_session()
    first, _ = record_customer_reply(db, event_id=4, subscription_id="sub_d", customer_reply_text="I'll pay Friday when salary comes", today=TODAY)
    assert first.status == STATUS_VALID

    second, _ = record_customer_reply(db, event_id=4, subscription_id="sub_d", customer_reply_text="Actually I'll pay tomorrow instead", today=TODAY)

    db.refresh(first)
    assert first.status == "SUPERSEDED"
    assert second.status == STATUS_VALID
    assert get_active_promise(db, 4).id == second.id
    db.close()


def test_multiple_promises_for_same_subscription_different_events(test_db_session):
    db = test_db_session()
    p1, _ = record_customer_reply(db, event_id=5, subscription_id="sub_shared", customer_reply_text="I'll pay Friday when salary comes", today=TODAY)
    p2, _ = record_customer_reply(db, event_id=6, subscription_id="sub_shared", customer_reply_text="I'll pay tomorrow", today=TODAY)
    assert p1.id != p2.id
    assert p1.status == STATUS_VALID
    assert p2.status == STATUS_VALID
    # supersession is scoped per event_id, not per subscription -- both stay active
    assert get_active_promise(db, 5).id == p1.id
    assert get_active_promise(db, 6).id == p2.id
    db.close()


class _AlwaysFailsClient(LLMClient):
    model_name = "test-broken"
    provider_name = "mock"

    def complete(self, system_prompt, user_prompt, *, max_tokens=512):
        raise LLMProviderError("simulated_outage")


def test_llm_failure_produces_no_fake_promise(test_db_session):
    db = test_db_session()
    promise, created = record_customer_reply(
        db, event_id=7, subscription_id="sub_e", customer_reply_text="I'll pay Friday", today=TODAY, client=_AlwaysFailsClient(),
    )
    assert created is True
    assert promise.status == STATUS_INVALID_DATE  # the fail-safe {date: null, confidence: 0.0} fallback -> no fabricated date
    assert promise.promised_date is None
    assert promise.confidence == 0.0
    db.close()


def test_no_active_promise_for_unknown_event(test_db_session):
    db = test_db_session()
    assert get_active_promise(db, 999999) is None
    db.close()


def test_db_failure_while_persisting_propagates_not_silently_swallowed(test_db_session):
    # Matches the rest of this codebase's convention (classify_raw_event,
    # decide_for_failure_event_engine_v4, orchestrate_recovery): a DB-layer
    # failure propagates to the caller rather than being caught and hidden
    # here -- the caller (e.g. the webhook pipeline) is responsible for
    # rollback, exactly as it already is for every other stage. Disposing
    # the in-memory engine's pool destroys the DB entirely (StaticPool means
    # there's exactly one underlying connection/DB for the whole test) --
    # a real, not simulated, "no such table" DB failure on the next query.
    db = test_db_session()
    test_db_session.kw["bind"].dispose()
    with pytest.raises(Exception):
        record_customer_reply(db, event_id=8, subscription_id="sub_f", customer_reply_text="I'll pay Friday", today=TODAY)
