import json

from app.models import AuditLog, RawEvent
from tests.conftest import sign


def _post(client, body_bytes: bytes, signature: str | None, event_id: str | None):
    headers = {"Content-Type": "application/json"}
    if signature is not None:
        headers["x-razorpay-signature"] = signature
    if event_id is not None:
        headers["x-razorpay-event-id"] = event_id
    return client.post("/webhook/razorpay", content=body_bytes, headers=headers)


def test_valid_webhook_is_stored(client, test_db_session, sample_subscription_failure_payload):
    body = json.dumps(sample_subscription_failure_payload).encode("utf-8")
    signature = sign(body)

    response = _post(client, body, signature, event_id="evt_TestEventId001")

    assert response.status_code == 200

    db = test_db_session()
    stored = db.query(RawEvent).filter(RawEvent.razorpay_event_id == "evt_TestEventId001").first()
    assert stored is not None
    assert stored.event_type == "payment.failed"
    assert stored.payment_id == "pay_TestPaymentId001"
    assert stored.subscription_id == "sub_TestSubscriptionId001"
    assert stored.error_reason == "insufficient_fund"
    assert stored.amount == 29900
    assert stored.signature_verified is True
    assert json.loads(stored.raw_payload)["event"] == "payment.failed"

    audit_rows = db.query(AuditLog).filter(AuditLog.raw_event_id == stored.id).all()
    assert len(audit_rows) == 1
    assert audit_rows[0].action == "webhook_received_and_stored"
    db.close()


def test_invalid_signature_is_rejected_and_not_stored(client, test_db_session, sample_subscription_failure_payload):
    body = json.dumps(sample_subscription_failure_payload).encode("utf-8")
    wrong_signature = "0" * 64  # well-formed hex string, but not a valid HMAC for this body

    response = _post(client, body, wrong_signature, event_id="evt_ShouldNotBeStored")

    assert response.status_code == 400

    db = test_db_session()
    assert db.query(RawEvent).filter(RawEvent.razorpay_event_id == "evt_ShouldNotBeStored").first() is None
    db.close()


def test_malformed_json_body_with_valid_signature_is_rejected(client, test_db_session):
    """
    A body that isn't valid JSON. The signature is computed correctly over
    these exact (garbage) bytes, proving signature validity and JSON validity
    are checked independently -- a body can pass HMAC verification and still
    fail to parse.
    """
    body = b"this is not json at all {{{"
    signature = sign(body)

    response = _post(client, body, signature, event_id="evt_MalformedBody")

    assert response.status_code == 400

    db = test_db_session()
    assert db.query(RawEvent).filter(RawEvent.razorpay_event_id == "evt_MalformedBody").first() is None
    db.close()


def test_missing_event_id_header_is_rejected(client, sample_subscription_failure_payload):
    body = json.dumps(sample_subscription_failure_payload).encode("utf-8")
    signature = sign(body)

    response = _post(client, body, signature, event_id=None)

    assert response.status_code == 400


def test_duplicate_event_id_does_not_create_second_record(client, test_db_session, sample_subscription_failure_payload):
    body = json.dumps(sample_subscription_failure_payload).encode("utf-8")
    signature = sign(body)

    first = _post(client, body, signature, event_id="evt_DuplicateTest001")
    second = _post(client, body, signature, event_id="evt_DuplicateTest001")

    # Razorpay expects 2xx even for an already-processed duplicate, or it
    # will keep retrying — never return an error status for a known duplicate.
    assert first.status_code == 200
    assert second.status_code == 200

    db = test_db_session()
    matching_rows = db.query(RawEvent).filter(RawEvent.razorpay_event_id == "evt_DuplicateTest001").all()
    assert len(matching_rows) == 1  # exactly one row, not two
    db.close()
