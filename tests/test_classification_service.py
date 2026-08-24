"""
classification/service.py tested against a real (in-memory) DB session --
reuses the same test_db_session fixture Day 1 defined in tests/conftest.py.
"""
from app.models import AuditLog, FailureEvent, RawEvent
from classification.rules import HARD_DECLINE, RETRYABLE_SOFT, RULE_VERSION, UNMAPPED
from classification.service import classify_all_raw_events, classify_raw_event


def _make_raw_event(db, **overrides) -> RawEvent:
    defaults = dict(
        razorpay_event_id="evt_ClassifyTest001",
        event_type="payment.failed",
        payment_id="pay_ClassifyTest001",
        error_code="BAD_REQUEST_ERROR",
        error_description="Payment failed due to insufficient funds in the customer account.",
        error_reason="insufficient_fund",
        error_source="customer",
        error_step="payment_authorization",
        signature_verified=True,
        raw_payload="{}",
    )
    defaults.update(overrides)
    raw_event = RawEvent(**defaults)
    db.add(raw_event)
    db.commit()
    db.refresh(raw_event)
    return raw_event


def test_classify_raw_event_creates_failure_event_and_audit_log(test_db_session):
    db = test_db_session()
    raw_event = _make_raw_event(db)

    failure_event, created = classify_raw_event(db, raw_event)

    assert created is True
    assert failure_event.raw_event_id == raw_event.id
    assert failure_event.classification_bucket == RETRYABLE_SOFT
    assert failure_event.classification_confidence == 1.0
    assert failure_event.rule_version == RULE_VERSION
    assert failure_event.classified_at is not None

    stored = db.query(FailureEvent).filter(FailureEvent.raw_event_id == raw_event.id).all()
    assert len(stored) == 1

    audit_rows = db.query(AuditLog).filter(AuditLog.raw_event_id == raw_event.id).all()
    assert len(audit_rows) == 1
    assert audit_rows[0].action == "classified"
    assert audit_rows[0].actor == "rule"
    assert "insufficient_fund" in audit_rows[0].reason
    db.close()


def test_example_insufficient_fund_classification_end_to_end(test_db_session):
    """The exact example from the Day-2 spec: insufficient_fund -> retryable_soft, confidence 1.0."""
    db = test_db_session()
    raw_event = _make_raw_event(db, razorpay_event_id="evt_InsufficientFundExample", error_reason="insufficient_fund")

    failure_event, _ = classify_raw_event(db, raw_event)

    assert failure_event.classification_bucket == "retryable_soft"
    assert failure_event.classification_confidence == 1.0
    db.close()


def test_duplicate_classification_does_not_create_second_failure_event(test_db_session):
    db = test_db_session()
    raw_event = _make_raw_event(db, razorpay_event_id="evt_DuplicateClassify001")

    first_result, first_created = classify_raw_event(db, raw_event)
    second_result, second_created = classify_raw_event(db, raw_event)

    assert first_created is True
    assert second_created is False
    assert first_result.id == second_result.id  # same row returned, not a new one

    stored = db.query(FailureEvent).filter(FailureEvent.raw_event_id == raw_event.id).all()
    assert len(stored) == 1  # exactly one row, not two
    db.close()


def test_duplicate_classification_still_writes_an_audit_log_entry(test_db_session):
    """Deciding not to re-classify is still a decision -- Day 1's audit_log
    convention ("every decision the system makes, including deciding to do
    nothing") applies here too."""
    db = test_db_session()
    raw_event = _make_raw_event(db, razorpay_event_id="evt_DuplicateAudit001")

    classify_raw_event(db, raw_event)
    classify_raw_event(db, raw_event)

    audit_rows = db.query(AuditLog).filter(AuditLog.raw_event_id == raw_event.id).order_by(AuditLog.id).all()
    assert len(audit_rows) == 2
    assert audit_rows[0].action == "classified"
    assert audit_rows[1].action == "classification_skipped_duplicate"
    db.close()


def test_hard_decline_reason_stores_hard_decline_bucket(test_db_session):
    db = test_db_session()
    raw_event = _make_raw_event(db, razorpay_event_id="evt_HardDecline001", error_reason="card_expired")

    failure_event, _ = classify_raw_event(db, raw_event)

    assert failure_event.classification_bucket == HARD_DECLINE
    assert failure_event.classification_confidence == 1.0
    db.close()


def test_malformed_raw_event_with_missing_error_fields_classifies_unmapped_not_crash(test_db_session):
    """A raw_event with no error_* fields at all (e.g. a non-failure event
    type accidentally passed through) must classify as unmapped, not raise."""
    db = test_db_session()
    raw_event = _make_raw_event(
        db,
        razorpay_event_id="evt_MalformedNoErrorFields001",
        event_type="payment.authorized",
        error_code=None,
        error_description=None,
        error_reason=None,
        error_source=None,
        error_step=None,
    )

    failure_event, created = classify_raw_event(db, raw_event)

    assert created is True
    assert failure_event.classification_bucket == UNMAPPED
    assert failure_event.classification_confidence == 0.0
    db.close()


def test_unverified_error_reason_classifies_unmapped(test_db_session):
    db = test_db_session()
    raw_event = _make_raw_event(db, razorpay_event_id="evt_Unverified001", error_reason="not_a_real_razorpay_reason")

    failure_event, _ = classify_raw_event(db, raw_event)

    assert failure_event.classification_bucket == UNMAPPED
    assert failure_event.classification_confidence == 0.0
    db.close()


def test_classify_all_raw_events_processes_every_row_and_skips_already_classified(test_db_session):
    db = test_db_session()
    raw_event_1 = _make_raw_event(db, razorpay_event_id="evt_BatchA", error_reason="insufficient_fund")
    raw_event_2 = _make_raw_event(db, razorpay_event_id="evt_BatchB", error_reason="card_expired")

    # Pre-classify one of the two to confirm the batch run skips it correctly.
    classify_raw_event(db, raw_event_1)

    summary = classify_all_raw_events(db)

    assert summary["total_raw_events"] == 2
    assert summary["newly_classified"] == 1  # only raw_event_2
    assert summary["already_classified_skipped"] == 1  # raw_event_1
    assert summary["buckets"][RETRYABLE_SOFT] == 1
    assert summary["buckets"][HARD_DECLINE] == 1

    # Running it again must be a full no-op in terms of new rows created.
    second_summary = classify_all_raw_events(db)
    assert second_summary["newly_classified"] == 0
    assert second_summary["already_classified_skipped"] == 2

    assert db.query(FailureEvent).count() == 2
    db.close()
