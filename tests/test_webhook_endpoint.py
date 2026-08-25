import json

from app.models import AuditLog, LLMInvocation, PolicyDecision, RawEvent
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

    # FIX #2: storage is followed automatically by classification + full
    # orchestration -- both a raw_event_id-scoped row (this webhook's own
    # storage) and further rows keyed by failure_event_id (classification,
    # policy, compliance, llm, orchestrator) now exist for the same delivery.
    audit_rows = db.query(AuditLog).filter(AuditLog.raw_event_id == stored.id).all()
    assert any(row.action == "webhook_received_and_stored" for row in audit_rows)

    assert "orchestration=completed" in response.text
    assert db.query(PolicyDecision).count() == 1
    assert db.query(LLMInvocation).filter(LLMInvocation.task_name == "outreach_microcopy").count() == 1
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
    # FIX #2 (2D): a duplicate delivery must not create a second recovery
    # action, retry decision, or communication either -- only the FIRST
    # delivery ever reaches process_raw_event at all (the duplicate check
    # above short-circuits before that call), so orchestration output stays
    # single no matter how many times the same event_id is redelivered.
    assert db.query(PolicyDecision).count() == 1
    assert db.query(LLMInvocation).count() == 1
    db.close()


# ---------------------------------------------------------------------------
# FIX #2: webhook -> automatic orchestration integration tests
# ---------------------------------------------------------------------------

def test_unsupported_event_type_is_stored_but_not_orchestrated(client, test_db_session, sample_subscription_failure_payload):
    payload = dict(sample_subscription_failure_payload)
    payload["event"] = "subscription.charged"  # a SUCCESSFUL charge -- nothing to recover
    body = json.dumps(payload).encode("utf-8")
    signature = sign(body)

    response = _post(client, body, signature, event_id="evt_UnsupportedType")
    assert response.status_code == 200
    assert "orchestration=skipped_unsupported_event_type" in response.text

    db = test_db_session()
    assert db.query(RawEvent).filter(RawEvent.razorpay_event_id == "evt_UnsupportedType").first() is not None
    assert db.query(PolicyDecision).count() == 0
    db.close()


def test_missing_subscription_entity_is_stored_but_not_orchestrated(client, test_db_session, sample_subscription_failure_payload):
    import copy

    payload = copy.deepcopy(sample_subscription_failure_payload)
    del payload["payload"]["subscription"]
    body = json.dumps(payload).encode("utf-8")
    signature = sign(body)

    response = _post(client, body, signature, event_id="evt_NoSubscription")
    assert response.status_code == 200
    assert "orchestration=skipped_missing_subscription_id" in response.text

    db = test_db_session()
    stored = db.query(RawEvent).filter(RawEvent.razorpay_event_id == "evt_NoSubscription").first()
    assert stored is not None
    assert stored.subscription_id is None
    assert db.query(PolicyDecision).count() == 0
    db.close()


def test_missing_payment_entity_does_not_crash(client, test_db_session, sample_subscription_failure_payload):
    import copy

    payload = copy.deepcopy(sample_subscription_failure_payload)
    del payload["payload"]["payment"]
    body = json.dumps(payload).encode("utf-8")
    signature = sign(body)

    response = _post(client, body, signature, event_id="evt_NoPayment")
    assert response.status_code == 200  # stored and handled gracefully, never a raw traceback

    db = test_db_session()
    stored = db.query(RawEvent).filter(RawEvent.razorpay_event_id == "evt_NoPayment").first()
    assert stored is not None
    assert stored.error_reason is None
    db.close()


def test_a_real_webhook_falls_back_to_rule_based_tier(client, test_db_session, sample_subscription_failure_payload):
    """A genuinely live webhook carries none of the synthetic dataset's
    engineered features (payday proximity, prior self-resolved rate, etc.)
    -- Model B's own insufficient-features check correctly treats this as
    an unusable model input and falls back to the rule-based tier, which
    needs none of those features. Documented behavior, not a bug."""
    body = json.dumps(sample_subscription_failure_payload).encode("utf-8")
    signature = sign(body)
    _post(client, body, signature, event_id="evt_RuleBasedFallback")

    db = test_db_session()
    decision = db.query(PolicyDecision).first()
    assert decision is not None
    assert decision.decision_source == "rule_based_fallback"
    db.close()


def test_orchestration_failure_after_storage_preserves_raw_event(client, test_db_session, sample_subscription_failure_payload, monkeypatch):
    def _broken(db, raw_event):
        raise RuntimeError("simulated orchestration bug")

    monkeypatch.setattr("app.main.process_raw_event", _broken)

    body = json.dumps(sample_subscription_failure_payload).encode("utf-8")
    signature = sign(body)
    response = _post(client, body, signature, event_id="evt_OrchestrationFailure")

    assert response.status_code == 200  # storage succeeded; Razorpay must not be told to redeliver
    assert "orchestration=failed" in response.text

    db = test_db_session()
    stored = db.query(RawEvent).filter(RawEvent.razorpay_event_id == "evt_OrchestrationFailure").first()
    assert stored is not None  # never un-stored by the downstream failure
    failure_audit = db.query(AuditLog).filter(AuditLog.raw_event_id == stored.id, AuditLog.action == "orchestration_failed_after_storage").first()
    assert failure_audit is not None
    assert "api_key" not in failure_audit.reason.lower()
    db.close()


def test_db_failure_during_downstream_processing_preserves_raw_event(client, test_db_session, sample_subscription_failure_payload, monkeypatch):
    from sqlalchemy.exc import OperationalError

    def _broken(db, raw_event):
        raise OperationalError("simulated", {}, Exception("db down"))

    monkeypatch.setattr("app.main.process_raw_event", _broken)

    body = json.dumps(sample_subscription_failure_payload).encode("utf-8")
    signature = sign(body)
    response = _post(client, body, signature, event_id="evt_DBFailure")

    assert response.status_code == 200
    db = test_db_session()
    assert db.query(RawEvent).filter(RawEvent.razorpay_event_id == "evt_DBFailure").first() is not None
    db.close()


def test_llm_failure_after_webhook_ingestion_leaves_payment_decision_intact(client, test_db_session, sample_subscription_failure_payload, monkeypatch):
    from llm.client import LLMClient, LLMProviderError

    class _AlwaysFailsClient(LLMClient):
        model_name = "webhook-test-broken"
        provider_name = "mock"

        def complete(self, system_prompt, user_prompt, *, max_tokens=512):
            raise LLMProviderError("simulated_outage")

    monkeypatch.setattr("llm.service.get_llm_client", lambda: _AlwaysFailsClient())

    body = json.dumps(sample_subscription_failure_payload).encode("utf-8")
    signature = sign(body)
    response = _post(client, body, signature, event_id="evt_LLMFailure")

    assert response.status_code == 200
    assert "orchestration=completed" in response.text

    db = test_db_session()
    decision = db.query(PolicyDecision).first()
    assert decision is not None
    assert decision.selected_candidate_type != "NO_ACTION"  # the payment decision is unaffected by the LLM outage
    invocation = db.query(LLMInvocation).filter(LLMInvocation.task_name == "outreach_microcopy").first()
    assert invocation is not None
    assert invocation.success is False
    db.close()
