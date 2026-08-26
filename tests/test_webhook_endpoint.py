import json

from app.models import AuditLog, LLMInvocation, PolicyDecision, RawEvent, RecoveryOutcome
from tests.conftest import sign


def _post(client, body_bytes: bytes, signature: str | None, event_id: str | None):
    headers = {"Content-Type": "application/json"}
    if signature is not None:
        headers["x-razorpay-signature"] = signature
    if event_id is not None:
        headers["x-razorpay-event-id"] = event_id
    return client.post("/webhook/razorpay", content=body_bytes, headers=headers)


def _captured_payload(*, payment_id: str, subscription_id: str | None, amount: int = 29900) -> dict:
    """Same envelope shape Razorpay uses for payment.failed (see
    sample_subscription_failure_payload) -- payment.captured carries no
    error_* fields, just a successful, captured payment. subscription_id=None
    omits the subscription entity entirely, matching a real Payment Link /
    one-time payment's captured payload (no subscription at all)."""
    payload = {
        "entity": "event", "account_id": "acc_TestAccountId000", "event": "payment.captured", "contains": ["payment"],
        "payload": {
            "payment": {"entity": {
                "id": payment_id, "entity": "payment", "amount": amount, "currency": "INR", "status": "captured",
                "order_id": None, "invoice_id": None, "method": "card",
            }},
        },
        "created_at": 1755840100,
    }
    if subscription_id is not None:
        payload["payload"]["subscription"] = {"entity": {"id": subscription_id, "entity": "subscription", "plan_id": "plan_TestPlanId001", "status": "active"}}
    return payload


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


def test_missing_subscription_entity_reaches_the_generalized_one_time_payment_path(client, test_db_session, sample_subscription_failure_payload):
    """A payment.failed with no subscription_id but a real payment_id +
    amount (this fixture's payment entity is untouched) is no longer a dead
    end -- it must reach the generalized revenue-risk pipeline (recovery/
    webhook_pipeline.py's GENERALIZATION), not remain stuck at "stored"."""
    import copy

    from app.models import RevenueRiskEvent

    payload = copy.deepcopy(sample_subscription_failure_payload)
    del payload["payload"]["subscription"]
    body = json.dumps(payload).encode("utf-8")
    signature = sign(body)

    response = _post(client, body, signature, event_id="evt_NoSubscription")
    assert response.status_code == 200
    assert "orchestration=completed" in response.text

    db = test_db_session()
    stored = db.query(RawEvent).filter(RawEvent.razorpay_event_id == "evt_NoSubscription").first()
    assert stored is not None
    assert stored.subscription_id is None

    rre = db.query(RevenueRiskEvent).filter(RevenueRiskEvent.event_type == "payment_failed_no_subscription").first()
    assert rre is not None
    assert rre.external_id == stored.payment_id

    # Eligible bucket -> reaches the unified ML model, not the rule fallback.
    decision = db.query(PolicyDecision).filter(PolicyDecision.decision_source == "ml_unified_v1").first()
    assert decision is not None
    assert decision.classification_bucket == "retryable_soft"  # error_reason=insufficient_fund; a real classification fact, unchanged by ML
    assert decision.selected_candidate_type == "payment_link_reminder"
    assert decision.model_version == "unified_catboost_v1"
    db.close()


def test_payment_failed_with_no_subscription_and_no_payment_context_is_genuinely_skipped(client, test_db_session, sample_subscription_failure_payload):
    """The ONLY remaining dead end: neither a subscription_id NOR enough
    authoritative payment context (payment_id/amount) to act on at all."""
    import copy

    payload = copy.deepcopy(sample_subscription_failure_payload)
    del payload["payload"]["subscription"]
    del payload["payload"]["payment"]
    body = json.dumps(payload).encode("utf-8")
    signature = sign(body)

    response = _post(client, body, signature, event_id="evt_NoSubscriptionNoPayment")
    assert response.status_code == 200
    assert "orchestration=skipped_insufficient_context" in response.text

    db = test_db_session()
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


# ---------------------------------------------------------------------------
# Closed-loop recovery confirmation: payment.failed -> PENDING outcome ->
# payment.captured (real Razorpay success event, HMAC-verified same as any
# other webhook) -> RECOVERED. Full HTTP-level flow, not just the unit-level
# recovery/payment_reconciliation.py tests in tests/test_payment_reconciliation.py.
# ---------------------------------------------------------------------------

def test_payment_captured_webhook_confirms_a_pending_recovery_case(client, test_db_session, sample_subscription_failure_payload):
    failed_body = json.dumps(sample_subscription_failure_payload).encode("utf-8")
    _post(client, failed_body, sign(failed_body), event_id="evt_CloseLoopFailed")

    db = test_db_session()
    failure = db.query(RawEvent).filter(RawEvent.razorpay_event_id == "evt_CloseLoopFailed").first()
    outcome_before = db.query(RecoveryOutcome).filter(RecoveryOutcome.event_type == "payment_failed").order_by(RecoveryOutcome.id.desc()).first()
    assert outcome_before.recovery_status == "PENDING"
    db.close()

    # Razorpay Subscriptions retries get a NEW payment_id -- not the one that failed.
    captured_body = json.dumps(_captured_payload(payment_id="pay_RetrySucceeded001", subscription_id="sub_TestSubscriptionId001", amount=29900)).encode("utf-8")
    response = _post(client, captured_body, sign(captured_body), event_id="evt_CloseLoopCaptured")

    assert response.status_code == 200
    assert "orchestration=payment_recovery_confirmed" in response.text

    db = test_db_session()
    outcome_after = db.query(RecoveryOutcome).filter(RecoveryOutcome.id == outcome_before.id).first()
    assert outcome_after.recovery_status == "RECOVERED"
    assert outcome_after.recovered_amount == 299.0
    assert outcome_after.confirmed_by == "webhook_confirmed"
    assert outcome_after.confirmed_payment_id == "pay_RetrySucceeded001"

    # policy decision made when the payment first failed is untouched by confirmation
    decision = db.query(PolicyDecision).first()
    assert decision.selected_candidate_type is not None
    db.close()


def test_duplicate_payment_captured_webhook_is_idempotent(client, test_db_session, sample_subscription_failure_payload):
    failed_body = json.dumps(sample_subscription_failure_payload).encode("utf-8")
    _post(client, failed_body, sign(failed_body), event_id="evt_CloseLoopDupFailed")

    captured_payload = _captured_payload(payment_id="pay_RetrySucceeded002", subscription_id="sub_TestSubscriptionId001", amount=29900)
    captured_body = json.dumps(captured_payload).encode("utf-8")

    first = _post(client, captured_body, sign(captured_body), event_id="evt_CloseLoopDupCaptured")
    assert "orchestration=payment_recovery_confirmed" in first.text

    # Razorpay redelivers the EXACT same webhook (same x-razorpay-event-id) --
    # caught by the existing, unmodified event-id idempotency check, same as
    # any other webhook type.
    second = _post(client, captured_body, sign(captured_body), event_id="evt_CloseLoopDupCaptured")
    assert second.status_code == 200
    assert "duplicate, already processed" in second.text

    db = test_db_session()
    assert db.query(RawEvent).filter(RawEvent.razorpay_event_id == "evt_CloseLoopDupCaptured").count() == 1
    outcome = db.query(RecoveryOutcome).filter(RecoveryOutcome.event_type == "payment_failed").order_by(RecoveryOutcome.id.desc()).first()
    assert outcome.recovery_status == "RECOVERED"
    assert outcome.recovered_amount == 299.0  # not doubled
    db.close()


def test_unrelated_payment_captured_webhook_does_not_fabricate_recovery(client, test_db_session):
    # An ordinary successful payment with no corresponding prior failure --
    # the overwhelmingly common case in practice -- must be a safe no-op.
    captured_body = json.dumps(_captured_payload(payment_id="pay_OrdinarySuccess", subscription_id="sub_NeverFailed", amount=50000)).encode("utf-8")
    response = _post(client, captured_body, sign(captured_body), event_id="evt_OrdinarySuccess")

    assert response.status_code == 200
    assert "orchestration=payment_confirmation_no_matching_case" in response.text

    db = test_db_session()
    assert db.query(RecoveryOutcome).filter(RecoveryOutcome.recovery_status == "RECOVERED").count() == 0
    db.close()


def test_a_failed_payment_webhook_alone_never_produces_a_recovered_status(client, test_db_session, sample_subscription_failure_payload):
    body = json.dumps(sample_subscription_failure_payload).encode("utf-8")
    _post(client, body, sign(body), event_id="evt_NeverRecoveredAlone")

    db = test_db_session()
    outcome = db.query(RecoveryOutcome).filter(RecoveryOutcome.event_type == "payment_failed").order_by(RecoveryOutcome.id.desc()).first()
    assert outcome.recovery_status == "PENDING"
    assert outcome.recovered_amount is None
    db.close()


def test_no_secret_or_signature_value_is_ever_persisted_for_a_captured_event(client, test_db_session):
    body = json.dumps(_captured_payload(payment_id="pay_SecretCheck", subscription_id="sub_SecretCheck", amount=10000)).encode("utf-8")
    captured_signature = sign(body)
    _post(client, body, captured_signature, event_id="evt_SecretCheck")

    db = test_db_session()
    raw = db.query(RawEvent).filter(RawEvent.razorpay_event_id == "evt_SecretCheck").first()
    assert raw.signature_verified is True  # boolean flag only
    # the raw payload never contains the signature or webhook secret (they
    # travel as an HTTP header, never part of the JSON body Razorpay sends)
    assert captured_signature not in raw.raw_payload
    from tests.conftest import TEST_WEBHOOK_SECRET
    assert TEST_WEBHOOK_SECRET not in raw.raw_payload
    for audit in db.query(AuditLog).all():
        assert TEST_WEBHOOK_SECRET not in (audit.reason or "")
        assert captured_signature not in (audit.reason or "")
    db.close()


# ---------------------------------------------------------------------------
# Generalized one-time-payment domain (payment.failed with no
# subscription_id, e.g. a Payment Link) -- full closed loop over real HTTP,
# mirroring the legacy-path tests above exactly.
# ---------------------------------------------------------------------------

def _payment_link_failure_payload(*, payment_id: str, amount: int = 100, error_reason: str = "insufficient_fund") -> dict:
    """A payment.failed payload shaped like a real Razorpay Payment Link
    failure -- payment entity present (order_id set, since Payment Links are
    built on the Orders API), no subscription entity at all."""
    return {
        "entity": "event", "account_id": "acc_TestAccountId000", "event": "payment.failed", "contains": ["payment"],
        "payload": {
            "payment": {"entity": {
                "id": payment_id, "entity": "payment", "amount": amount, "currency": "INR", "status": "failed",
                "order_id": "order_TestPaymentLinkOrder001", "invoice_id": None, "method": "upi",
                "error_code": "BAD_REQUEST_ERROR", "error_description": "insufficient funds", "error_reason": error_reason,
                "error_source": "customer", "error_step": "payment_authorization", "created_at": 1755840000,
            }},
        },
        "created_at": 1755840000,
    }


def test_payment_link_failure_reaches_full_pipeline_and_confirms_on_capture(client, test_db_session):
    from app.models import LLMInvocation, RevenueRiskEvent

    failed_body = json.dumps(_payment_link_failure_payload(payment_id="pay_PaymentLinkRupee1", amount=100)).encode("utf-8")
    response = _post(client, failed_body, sign(failed_body), event_id="evt_PaymentLinkFailed")
    assert response.status_code == 200
    assert "orchestration=completed" in response.text

    db = test_db_session()
    rre = db.query(RevenueRiskEvent).filter(RevenueRiskEvent.external_id == "pay_PaymentLinkRupee1").first()
    assert rre is not None
    assert rre.event_type == "payment_failed_no_subscription"

    # Eligible bucket (retryable_soft) -> reaches the UNIFIED ML model, not
    # the rule-based fallback -- proves the Payment Link / subscription_id=NULL
    # path genuinely reaches ML, not just the deterministic classifier.
    decision = db.query(PolicyDecision).filter(PolicyDecision.decision_source == "ml_unified_v1").first()
    assert decision is not None
    assert decision.selected_candidate_type == "payment_link_reminder"  # the domain's one truthful action -- never a fabricated auto-retry
    assert decision.classification_bucket == "retryable_soft"  # error_reason=insufficient_fund; still a real classification fact, not an ML output
    assert decision.model_version == "unified_catboost_v1"
    assert decision.predicted_recovery_probability is not None
    assert "rule_baseline_candidate=payment_link_reminder" in decision.decision_reason  # Phase-13 audit trail: rule baseline recorded alongside ML's pick

    invocation = db.query(LLMInvocation).filter(LLMInvocation.task_name == "outreach_microcopy").order_by(LLMInvocation.id.desc()).first()
    assert invocation is not None
    assert invocation.success is True  # mock provider in tests

    outcome_before = db.query(RecoveryOutcome).filter(RecoveryOutcome.event_type == "payment_failed_no_subscription").first()
    assert outcome_before.recovery_status == "PENDING"
    assert outcome_before.recovered_amount is None
    db.close()

    captured_body = json.dumps(_captured_payload(payment_id="pay_PaymentLinkRupee1", subscription_id=None, amount=100)).encode("utf-8")
    response2 = _post(client, captured_body, sign(captured_body), event_id="evt_PaymentLinkCaptured")
    assert response2.status_code == 200
    assert "orchestration=payment_recovery_confirmed" in response2.text

    db = test_db_session()
    outcome_after = db.query(RecoveryOutcome).filter(RecoveryOutcome.id == outcome_before.id).first()
    assert outcome_after.recovery_status == "RECOVERED"
    assert outcome_after.recovered_amount == 1.0  # 100 paise -> rupee 1.0
    assert outcome_after.confirmed_by == "webhook_confirmed"
    assert outcome_after.confirmed_payment_id == "pay_PaymentLinkRupee1"
    db.close()


def test_payment_link_failure_with_hard_decline_still_gets_a_reminder_never_a_fake_retry_claim(client, test_db_session):
    body = json.dumps(_payment_link_failure_payload(payment_id="pay_PaymentLinkHardDecline", error_reason="card_declined")).encode("utf-8")
    response = _post(client, body, sign(body), event_id="evt_PaymentLinkHardDecline")
    assert response.status_code == 200
    assert "orchestration=completed" in response.text

    db = test_db_session()
    # hard_decline is still eligible for the one truthful action (see
    # policy/one_time_payment_rules.py) -- reaches the unified ML model too.
    decision = db.query(PolicyDecision).filter(PolicyDecision.decision_source == "ml_unified_v1").first()
    assert decision is not None
    assert decision.classification_bucket == "hard_decline"
    assert decision.selected_candidate_type == "payment_link_reminder"  # never a fabricated automatic retry
    assert decision.model_version == "unified_catboost_v1"
    db.close()


def test_payment_link_failure_with_unmapped_reason_takes_no_action(client, test_db_session):
    body = json.dumps(_payment_link_failure_payload(payment_id="pay_PaymentLinkUnmapped", error_reason="totally_unrecognized_reason")).encode("utf-8")
    _post(client, body, sign(body), event_id="evt_PaymentLinkUnmapped")

    db = test_db_session()
    decision = db.query(PolicyDecision).filter(PolicyDecision.decision_source == "rule_one_time_payment_failed").first()
    assert decision.classification_bucket == "unmapped"
    assert decision.selected_candidate_type == "NO_ACTION"
    # ML is still consulted for an unmapped/generic reason -- it just never
    # gets to override the eligibility gate's NO_ACTION (see
    # tests/test_revenue_recovery_policy.py::test_payment_link_with_unmapped_generic_reason_still_consults_ml).
    assert "ml_consulted=True" in decision.decision_reason
    assert decision.predicted_recovery_probability is not None
    db.close()


def test_duplicate_payment_link_failure_webhook_does_not_duplicate_anything(client, test_db_session):
    from app.models import RevenueRiskEvent

    body = json.dumps(_payment_link_failure_payload(payment_id="pay_PaymentLinkDup")).encode("utf-8")
    r1 = _post(client, body, sign(body), event_id="evt_PaymentLinkDup")
    r2 = _post(client, body, sign(body), event_id="evt_PaymentLinkDup")  # identical event_id -- exact webhook redelivery
    assert r1.status_code == 200 and r2.status_code == 200
    assert "duplicate, already processed" in r2.text

    db = test_db_session()
    assert db.query(RawEvent).filter(RawEvent.razorpay_event_id == "evt_PaymentLinkDup").count() == 1
    assert db.query(RevenueRiskEvent).filter(RevenueRiskEvent.external_id == "pay_PaymentLinkDup").count() == 1
    # default payload's error_reason=insufficient_fund is eligible -> ML path
    assert db.query(PolicyDecision).filter(PolicyDecision.decision_source == "ml_unified_v1").count() == 1
    db.close()
