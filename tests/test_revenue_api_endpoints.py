"""
Track-03 tests: app/main.py's 4 new /events/* routes. Mirrors
tests/test_webhook_endpoint.py's style: valid payload -> 200 + stored + real
orchestration; duplicate idempotency_key -> 200 "duplicate"; malformed
payload -> 422; orchestration exception never un-stores the event.
"""
from app.models import AuditLog, CheckoutSession, MandateRetrySequence, PolicyDecision, Receivable, RevenueRiskEvent
from policy.policy_decision_store import REVENUE_DOMAIN_EVENT_ID_OFFSET


def _checkout_payload(**overrides) -> dict:
    base = dict(
        cart_id="cart_1", customer_id="cust_1", cart_amount=999.0,
        checkout_started_at="2026-08-25T08:00:00", last_activity_at="2026-08-25T06:00:00",  # 2h+ inactivity, well past every checkout threshold
        payment_method="card",
    )
    base.update(overrides)
    return base


class TestCheckoutAbandonedEndpoint:
    def test_valid_payload_is_stored_and_orchestrated(self, client, test_db_session):
        resp = client.post("/events/checkout-abandoned", json=_checkout_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "processed"
        assert "orchestration" in body
        assert body["orchestration"]["event_type"] == "checkout_abandoned"

        db = test_db_session()
        rre = db.query(RevenueRiskEvent).filter(RevenueRiskEvent.id == body["revenue_risk_event_id"]).first()
        assert rre is not None
        assert rre.event_type == "checkout_abandoned"
        session = db.query(CheckoutSession).filter(CheckoutSession.revenue_risk_event_id == rre.id).first()
        assert session is not None
        assert session.cart_id == "cart_1"
        db.close()

    def test_duplicate_idempotency_key_does_not_reorchestrate(self, client, test_db_session):
        first = client.post("/events/checkout-abandoned", json=_checkout_payload(cart_id="cart_dup"))
        second = client.post("/events/checkout-abandoned", json=_checkout_payload(cart_id="cart_dup"))
        assert first.json()["status"] == "processed"
        assert second.json()["status"] == "duplicate"
        assert second.json()["revenue_risk_event_id"] == first.json()["revenue_risk_event_id"]

        db = test_db_session()
        count = db.query(RevenueRiskEvent).filter(RevenueRiskEvent.external_id == "cart_dup").count()
        assert count == 1
        db.close()

    def test_malformed_payload_is_422(self, client):
        resp = client.post("/events/checkout-abandoned", json={"cart_id": "only_this_field"})
        assert resp.status_code == 422

    def test_customer_opted_out_is_reachable_and_blocks_communication(self, client):
        # Full-system audit finding: CheckoutAbandonedRequest had no
        # customer_opted_out field, so compliance_v2's opt-out block
        # (policy/compliance_v2.py) was coded but unreachable from this
        # endpoint -- every checkout request silently defaulted to False.
        resp = client.post("/events/checkout-abandoned", json=_checkout_payload(cart_id="cart_opted_out", customer_id="cust_opted_out", customer_opted_out=True))
        assert resp.status_code == 200
        orchestration = resp.json()["orchestration"]
        assert orchestration["payment_verdict"] == "ALLOWED"  # opt-out only blocks communication
        assert orchestration["communication_verdict"] == "BLOCKED"
        assert "opted_out" in orchestration["communication_reason"]

    def test_orchestration_failure_preserves_the_stored_event(self, client, test_db_session, monkeypatch):
        def _broken(db, event, **kwargs):
            raise RuntimeError("simulated orchestration bug")

        monkeypatch.setattr("app.main.orchestrate_revenue_event", _broken)

        resp = client.post("/events/checkout-abandoned", json=_checkout_payload(cart_id="cart_broken"))
        assert resp.status_code == 200
        assert resp.json()["status"] == "stored_orchestration_failed"

        db = test_db_session()
        rre = db.query(RevenueRiskEvent).filter(RevenueRiskEvent.external_id == "cart_broken").first()
        assert rre is not None  # never un-stored by the downstream failure
        # AuditLog.failure_event_id shares its column with FailureEvent.id (payment
        # domain) -- every revenue-domain write into it must carry
        # REVENUE_DOMAIN_EVENT_ID_OFFSET, same as PolicyDecision.event_id/
        # LLMInvocation.event_id, or it risks colliding with an unrelated
        # payment_failed event's audit trail (see policy/policy_decision_store.py).
        failure_audit = db.query(AuditLog).filter(AuditLog.failure_event_id == rre.id + REVENUE_DOMAIN_EVENT_ID_OFFSET, AuditLog.action == "revenue_orchestration_failed_after_storage").first()
        assert failure_audit is not None
        # and the raw, un-offset id must NOT have been used for this write
        raw_id_audit = db.query(AuditLog).filter(AuditLog.failure_event_id == rre.id, AuditLog.action == "revenue_orchestration_failed_after_storage").first()
        assert raw_id_audit is None
        db.close()


class TestMandateFailedEndpoint:
    def test_valid_payload_is_stored_and_orchestrated(self, client, test_db_session):
        resp = client.post("/events/mandate-failed", json={"mandate_id": "mandate_1", "subscription_id": "sub_1", "amount": 1000.0, "occurred_at": "2026-08-25T10:00:00"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "processed"
        # Eligible mandate failure -> the unified ML model picks among the
        # domain's real candidate vocabulary (model/unified_model.py
        # CANDIDATE_SPACE["mandate_failed"]), not necessarily the rule
        # decider's own "attempt_1" default -- see
        # tests/test_revenue_recovery_policy.py for the rule-vs-ML boundary itself.
        assert body["orchestration"]["selected_candidate_type"] in {"attempt_1", "attempt_2", "final_attempt"}

        db = test_db_session()
        seq = db.query(MandateRetrySequence).filter(MandateRetrySequence.mandate_id == "mandate_1").first()
        assert seq is not None
        db.close()

    def test_duplicate_is_not_reorchestrated(self, client):
        payload = {"mandate_id": "mandate_dup", "amount": 1000.0, "occurred_at": "2026-08-25T10:00:00"}
        first = client.post("/events/mandate-failed", json=payload)
        second = client.post("/events/mandate-failed", json=payload)
        assert first.json()["status"] == "processed"
        assert second.json()["status"] == "duplicate"

    def test_customer_opted_out_is_reachable_and_blocks_communication(self, client):
        resp = client.post("/events/mandate-failed", json={
            "mandate_id": "mandate_opted_out", "amount": 1000.0, "occurred_at": "2026-08-25T10:00:00", "customer_opted_out": True,
        })
        orchestration = resp.json()["orchestration"]
        assert orchestration["payment_verdict"] == "ALLOWED"
        assert orchestration["communication_verdict"] == "BLOCKED"


class TestReceivableOverdueEndpoint:
    def test_valid_payload_is_stored_and_orchestrated(self, client, test_db_session):
        resp = client.post("/events/receivable-overdue", json={"invoice_id": "inv_1", "customer_account_id": "acct_1", "invoice_amount": 25000.0, "due_date": "2026-07-01T00:00:00", "days_overdue": 45})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "processed"
        # Eligible (non-disputed) receivable -> the unified ML model is live
        # and picks among the domain's real candidate vocabulary (see
        # model/unified_model.py CANDIDATE_SPACE["receivable_overdue"]).
        assert body["orchestration"]["selected_candidate_type"] in {"friendly_reminder", "payment_request", "promise_to_pay_request", "escalation"}

        db = test_db_session()
        receivable = db.query(Receivable).filter(Receivable.invoice_id == "inv_1").first()
        assert receivable is not None
        decision = db.query(PolicyDecision).filter(PolicyDecision.event_id == body["revenue_risk_event_id"] + REVENUE_DOMAIN_EVENT_ID_OFFSET).first()
        assert decision is not None
        assert decision.decision_source == "ml_unified_v1"
        assert decision.model_version == "unified_catboost_v1"
        assert decision.predicted_recovery_probability is not None
        db.close()

    def test_disputed_invoice_routes_to_human_review(self, client):
        resp = client.post("/events/receivable-overdue", json={"invoice_id": "inv_disputed", "customer_account_id": "acct_2", "invoice_amount": 5000.0, "due_date": "2026-07-01T00:00:00", "days_overdue": 45, "is_disputed": True})
        body = resp.json()
        assert body["orchestration"]["payment_verdict"] == "HUMAN_REVIEW"

    def test_duplicate_is_not_reorchestrated(self, client):
        payload = {"invoice_id": "inv_dup", "customer_account_id": "acct_3", "invoice_amount": 1000.0, "due_date": "2026-07-01T00:00:00", "days_overdue": 5}
        first = client.post("/events/receivable-overdue", json=payload)
        second = client.post("/events/receivable-overdue", json=payload)
        assert first.json()["status"] == "processed"
        assert second.json()["status"] == "duplicate"

    def test_customer_opted_out_is_reachable_and_blocks_communication(self, client):
        resp = client.post("/events/receivable-overdue", json={
            "invoice_id": "inv_opted_out", "customer_account_id": "acct_opted_out", "invoice_amount": 5000.0,
            "due_date": "2026-07-01T00:00:00", "days_overdue": 10, "customer_opted_out": True,
        })
        orchestration = resp.json()["orchestration"]
        assert orchestration["payment_verdict"] == "ALLOWED"
        assert orchestration["communication_verdict"] == "BLOCKED"


class TestPromiseToPayEndpoint:
    def test_valid_reply_is_recorded(self, client, test_db_session):
        resp = client.post("/events/promise-to-pay", json={"event_id": 999001, "subscription_id": "sub_promise_api", "customer_reply_text": "I'll pay tomorrow via UPI"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "processed"
        assert "promise_to_pay_id" in body

    def test_duplicate_exact_reply_is_idempotent(self, client):
        payload = {"event_id": 999002, "subscription_id": "sub_promise_api_2", "customer_reply_text": "I'll pay Friday"}
        first = client.post("/events/promise-to-pay", json=payload)
        second = client.post("/events/promise-to-pay", json=payload)
        assert first.json()["status"] == "processed"
        assert second.json()["status"] == "duplicate"
        assert first.json()["promise_to_pay_id"] == second.json()["promise_to_pay_id"]

    def test_malformed_payload_is_422(self, client):
        resp = client.post("/events/promise-to-pay", json={"event_id": "not_an_int"})
        assert resp.status_code == 422

    def test_does_not_duplicate_the_razorpay_webhook_endpoint(self, client):
        # this route must never touch raw_events / signature verification --
        # it's a thin wrapper over the existing promise_service, not a webhook.
        resp = client.post("/events/promise-to-pay", json={"event_id": 999003, "subscription_id": "sub_x", "customer_reply_text": "yes I will pay"})
        assert resp.status_code == 200
        assert "signature" not in resp.json()
