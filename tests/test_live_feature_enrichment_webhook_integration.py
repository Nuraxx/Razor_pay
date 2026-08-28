"""
BUG-4 webhook-level integration tests (pre-submission audit): proves the
FULL wiring --

    webhook -> build_live_features() -> feature-completeness check
        -> complete   -> Model B invoked -> policy -> compliance -> recovery action
        -> incomplete -> Model B NOT invoked -> rule-based fallback -> audit explains why

-- end to end over real HTTP (fastapi.testclient), not just at the
decide_engine()/decide_engine_v4() unit level (already covered extensively
by tests/test_decision_engine.py / tests/test_decision_engine_v4.py).

On "complete": as documented in recovery/live_feature_enrichment.py's module
docstring, a GENUINE live Razorpay webhook can never actually produce a
complete Model B feature vector -- at least 5 of the 12 required features
(bank_network_conditions, issuing_bank_downtime_flag, network_latency_bucket,
is_month_end_settlement_rush, prior_if_self_resolved_rate) have no real
source at all, live or otherwise. This test proves the CONDITIONAL WIRING is
correct (IF the enrichment boundary ever reported complete, Model B WOULD be
invoked) by injecting a simulated complete result at that boundary -- it
does not, and must not be read to, claim real traffic ever reaches this
branch. The companion "incomplete" tests below use the REAL, unmocked
build_live_features() path to prove what actually happens for a genuine
webhook today.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx

from app.models import AuditLog, PolicyDecision
from policy.decision_engine import SOURCE_FALLBACK, SOURCE_MODEL
from policy.retry_candidates import CANDIDATE_TYPES
from recovery.live_feature_enrichment import LiveFeatureResult
from tests.conftest import sign

# Same failure timestamp tests/test_decision_engine.py uses -- verified there
# that all 5 candidates are valid from this point.
FAILURE_DT = datetime(2026, 2, 24, 10, 0, 0, tzinfo=timezone.utc)
FAILURE_UNIX_TS = int(FAILURE_DT.timestamp())

COMPLETE_FAILURE_CONTEXT = {
    "day_of_month": 24, "days_to_nearest_payday_window": 6, "prior_if_failure_count": 0,
    "prior_if_self_resolved_rate": 0.3, "tenure_days": 200, "plan_tier": "mid",
    "primary_instrument": "upi_autopay", "city_tier": "tier_1", "bank_network_conditions": "good",
    "issuing_bank_downtime_flag": False, "network_latency_bucket": "low", "is_month_end_settlement_rush": False,
}


class _PassthroughImputer:
    def transform(self, X):
        return X


class _FakeModelBValues:
    """Distinct, clearly-ranked predictions (immediate wins by a wide
    margin) -- same fixture style as tests/test_decision_engine.py's
    _FakeModel, so Model B decisively beats the rule-based candidate and
    clears policy-v4's margin gate (DEFAULT_MARGIN_THRESHOLD_RS=0.0)."""

    def predict(self, X):
        import numpy as np

        values = {"immediate": 500.0, "plus_1_day_morning": 100.0, "payday_window": 50.0, "plus_3_days": 20.0, "month_end_window": 10.0}
        return np.array([values[row["candidate_type"]] for _, row in X.iterrows()])


def _fake_model_b_dict() -> dict:
    return {"imputer": _PassthroughImputer(), "catboost_model": _FakeModelBValues()}


def _webhook_payload(*, payment_id: str, subscription_id: str, event_id: str) -> tuple[bytes, str, dict]:
    payload = {
        "entity": "event", "account_id": "acc_TestAccountId000", "event": "payment.failed", "contains": ["payment"],
        "payload": {
            "payment": {"entity": {
                "id": payment_id, "entity": "payment", "amount": 29900, "currency": "INR", "status": "failed",
                "order_id": None, "invoice_id": None, "method": "upi",
                "error_code": "BAD_REQUEST_ERROR", "error_description": "insufficient funds",
                "error_reason": "insufficient_fund", "error_source": "customer", "error_step": "payment_authorization",
                "created_at": FAILURE_UNIX_TS,
            }},
            "subscription": {"entity": {"id": subscription_id, "entity": "subscription", "plan_id": "plan_TestPlanId001", "status": "active"}},
        },
        "created_at": FAILURE_UNIX_TS,
    }
    body = json.dumps(payload).encode("utf-8")
    return body, event_id, payload


def _post(client, body: bytes, event_id: str):
    return client.post(
        "/webhook/razorpay", content=body,
        headers={"Content-Type": "application/json", "x-razorpay-signature": sign(body), "x-razorpay-event-id": event_id},
    )


# ---------------------------------------------------------------------------
# #17: complete (simulated) feature vector -> Model B invoked
# ---------------------------------------------------------------------------

def test_complete_simulated_feature_vector_reaches_model_b_end_to_end(client, test_db_session, monkeypatch):
    monkeypatch.setattr(
        "recovery.webhook_pipeline.build_live_features",
        lambda db, raw_event, subscription_id, failure_timestamp: LiveFeatureResult(
            features=dict(COMPLETE_FAILURE_CONTEXT), missing_features=[], complete=True,
            enrichment_attempted=True, enrichment_succeeded=True,
        ),
    )
    monkeypatch.setattr("policy.decision_engine.load_latent_target_model", lambda target: _fake_model_b_dict())

    body, event_id, _ = _webhook_payload(payment_id="pay_CompleteFeatures", subscription_id="sub_CompleteFeatures", event_id="evt_CompleteFeatures")
    response = _post(client, body, event_id)
    assert response.status_code == 200
    assert "orchestration=completed" in response.text

    db = test_db_session()
    decision = db.query(PolicyDecision).filter(PolicyDecision.subscription_id == "sub_CompleteFeatures").first()
    assert decision is not None
    assert decision.decision_source == SOURCE_MODEL  # "subscription_value_model" -- Model B, not the fallback
    assert decision.selected_candidate_type == "immediate"  # the candidate _FakeModelBValues ranks highest
    assert decision.selected_candidate_type in CANDIDATE_TYPES
    assert decision.expected_recovery_value is not None

    audit = db.query(AuditLog).filter(AuditLog.failure_event_id == decision.event_id, AuditLog.action == "policy_decision_made").first()
    assert audit is not None
    assert SOURCE_MODEL in audit.reason
    db.close()


# ---------------------------------------------------------------------------
# #18: incomplete feature vector (the REAL, unmocked path for a genuine
# webhook) -> Model B NOT invoked -> rule-based fallback -> audit explains
# missing features
# ---------------------------------------------------------------------------

def test_incomplete_real_feature_vector_never_invokes_model_b_and_audit_explains_why(client, test_db_session, monkeypatch):
    """No mocking of build_live_features here -- this exercises the REAL
    feature-derivation live path a genuine Razorpay webhook takes today.
    Model B's own artifact-loading IS stubbed to a fake-but-loadable model
    (same pattern as tests/test_decision_engine.py's _fake_model_dict) --
    not to change the outcome (Model B never wins either way, since the real
    feature vector is genuinely incomplete), but so THIS assertion is about
    the feature-completeness check specifically, not incidentally coupled to
    whether some other test/environment happened to leave a real trained
    model/latent_target_artifacts/value/ artifact on disk."""
    monkeypatch.setattr("policy.decision_engine.load_latent_target_model", lambda target: _fake_model_b_dict())

    body, event_id, _ = _webhook_payload(payment_id="pay_IncompleteFeatures", subscription_id="sub_IncompleteFeatures", event_id="evt_IncompleteFeatures")
    response = _post(client, body, event_id)
    assert response.status_code == 200

    db = test_db_session()
    decision = db.query(PolicyDecision).filter(PolicyDecision.subscription_id == "sub_IncompleteFeatures").first()
    assert decision is not None
    assert decision.decision_source == SOURCE_FALLBACK  # Model B never wins -- fallback IS what decided this
    assert decision.selected_candidate_type != "NO_ACTION"  # still a valid recovery result, not a dead end

    # Audit trail explicitly names the missing features (policy/decision_engine.py's
    # existing, unmodified insufficient_features message) -- never sensitive data.
    audit = db.query(AuditLog).filter(AuditLog.failure_event_id == decision.event_id, AuditLog.action == "policy_decision_made").first()
    assert audit is not None
    assert "insufficient_features" in audit.reason
    assert "missing" in audit.reason
    for unavailable_key in ("bank_network_conditions", "issuing_bank_downtime_flag", "network_latency_bucket"):
        assert unavailable_key in audit.reason
    db.close()


def test_model_b_artifact_genuinely_unavailable_also_falls_back_cleanly(client, test_db_session, monkeypatch):
    """Companion scenario: on a fresh clone with NO Model B artifact trained
    at all (model/latent_target_artifacts/ absent), the fallback still fires
    cleanly for a different, equally valid reason (model_unavailable rather
    than insufficient_features) -- both are honest, both mean Model B was
    never actually used. Forces this scenario deterministically rather than
    depending on the artifact's real on-disk state."""
    def _raise_missing(target):
        raise FileNotFoundError("no such artifact")

    monkeypatch.setattr("policy.decision_engine.load_latent_target_model", _raise_missing)

    body, event_id, _ = _webhook_payload(payment_id="pay_ModelBUnavailable", subscription_id="sub_ModelBUnavailable", event_id="evt_ModelBUnavailable")
    response = _post(client, body, event_id)
    assert response.status_code == 200

    db = test_db_session()
    decision = db.query(PolicyDecision).filter(PolicyDecision.subscription_id == "sub_ModelBUnavailable").first()
    assert decision is not None
    assert decision.decision_source == SOURCE_FALLBACK
    assert decision.selected_candidate_type != "NO_ACTION"
    audit = db.query(AuditLog).filter(AuditLog.failure_event_id == decision.event_id, AuditLog.action == "policy_decision_made").first()
    assert "model_unavailable" in audit.reason
    db.close()


def test_incomplete_feature_vector_still_includes_the_genuinely_available_ones(client, test_db_session, monkeypatch):
    """Proves the enrichment boundary isn't just always returning nothing --
    the real DERIVED_FROM_AUTHORITATIVE_DATA/WEBHOOK_NATIVE features ARE
    computed and passed through, they're just not sufficient on their own
    for Model B's strict all-12-keys check. Model B's artifact loading is
    stubbed the same way as the test above, for the same reason (isolates
    this from whichever artifact state happens to exist on disk)."""
    monkeypatch.setattr("policy.decision_engine.load_latent_target_model", lambda target: _fake_model_b_dict())
    body, event_id, _ = _webhook_payload(payment_id="pay_PartialFeatures", subscription_id="sub_PartialFeatures", event_id="evt_PartialFeatures")
    _post(client, body, event_id)

    db = test_db_session()
    decision = db.query(PolicyDecision).filter(PolicyDecision.subscription_id == "sub_PartialFeatures").first()
    assert decision is not None
    audit = db.query(AuditLog).filter(AuditLog.failure_event_id == decision.event_id, AuditLog.action == "policy_decision_made").first()
    missing_list_text = audit.reason[audit.reason.index("missing ["):]
    # day_of_month/days_to_nearest_payday_window/prior_if_failure_count/
    # primary_instrument were genuinely available (extracted/derived, not
    # blocked) -- so they must NOT appear in the missing list, proving the
    # derivation logic actually ran rather than silently producing nothing.
    for available_key in ("day_of_month", "days_to_nearest_payday_window", "prior_if_failure_count", "primary_instrument"):
        assert f"'{available_key}'" not in missing_list_text
    # The genuinely UNAVAILABLE/MERCHANT_PROFILE keys must be listed.
    for unavailable_key in ("bank_network_conditions", "issuing_bank_downtime_flag", "network_latency_bucket", "plan_tier", "city_tier"):
        assert f"'{unavailable_key}'" in missing_list_text
    db.close()


# ---------------------------------------------------------------------------
# #19: enrichment failures (with LIVE_FEATURE_ENRICHMENT_ENABLED=true)
# degrade to the deterministic fallback rather than crashing the webhook
# ---------------------------------------------------------------------------

def _real_fetch_over_mock_transport(handler):
    """Exercises the REAL _default_fetch_subscription/_fetch_razorpay_enrichment
    code (the actual error-handling logic under test), routed through
    httpx.MockTransport instead of a real network call -- captures the real
    httpx.Client class BEFORE any patching, so this cannot recurse into
    itself the way patching httpx.Client globally would."""
    real_client_class = httpx.Client

    def _client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client_class(*args, **kwargs)

    return _client_factory


def test_webhook_survives_enrichment_timeout_and_still_produces_a_valid_decision(client, test_db_session, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "LIVE_FEATURE_ENRICHMENT_ENABLED", True)
    monkeypatch.setattr(settings, "RAZORPAY_KEY_ID", "rzp_test_fake")
    monkeypatch.setattr(settings, "RAZORPAY_KEY_SECRET", "fake_secret")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("simulated timeout", request=request)

    monkeypatch.setattr(httpx, "Client", _real_fetch_over_mock_transport(handler))

    body, event_id, _ = _webhook_payload(payment_id="pay_EnrichTimeout", subscription_id="sub_EnrichTimeout", event_id="evt_EnrichTimeout")
    response = _post(client, body, event_id)
    assert response.status_code == 200  # webhook survives -- never a 5xx from an enrichment failure
    assert "orchestration=completed" in response.text

    db = test_db_session()
    decision = db.query(PolicyDecision).filter(PolicyDecision.subscription_id == "sub_EnrichTimeout").first()
    assert decision is not None
    assert decision.decision_source == SOURCE_FALLBACK
    db.close()


def test_webhook_survives_enrichment_401_and_still_produces_a_valid_decision(client, test_db_session, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "LIVE_FEATURE_ENRICHMENT_ENABLED", True)
    monkeypatch.setattr(settings, "RAZORPAY_KEY_ID", "rzp_test_fake")
    monkeypatch.setattr(settings, "RAZORPAY_KEY_SECRET", "fake_secret")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    monkeypatch.setattr(httpx, "Client", _real_fetch_over_mock_transport(handler))

    body, event_id, _ = _webhook_payload(payment_id="pay_Enrich401", subscription_id="sub_Enrich401", event_id="evt_Enrich401")
    response = _post(client, body, event_id)
    assert response.status_code == 200
    assert "orchestration=completed" in response.text

    db = test_db_session()
    decision = db.query(PolicyDecision).filter(PolicyDecision.subscription_id == "sub_Enrich401").first()
    assert decision is not None
    assert decision.decision_source == SOURCE_FALLBACK
    db.close()


# ---------------------------------------------------------------------------
# #20/#21: the live webhook path never imports/invokes the synthetic
# generator's random entity-simulation functions
# ---------------------------------------------------------------------------

def test_full_webhook_pipeline_never_invokes_synthetic_generators(client, test_db_session, monkeypatch):
    import data.generate_synthetic_dataset as gen

    def _forbidden(*args, **kwargs):
        raise AssertionError("the live webhook path must never call the synthetic data generator")

    monkeypatch.setattr(gen, "generate_dataset", _forbidden)
    monkeypatch.setattr(gen, "generate_subscriptions", _forbidden)
    monkeypatch.setattr(gen, "generate_failure_events_and_outcomes", _forbidden)
    monkeypatch.setattr(gen, "generate_retry_candidates", _forbidden)

    body, event_id, _ = _webhook_payload(payment_id="pay_NoSyntheticLeakage", subscription_id="sub_NoSyntheticLeakage", event_id="evt_NoSyntheticLeakage")
    response = _post(client, body, event_id)
    assert response.status_code == 200  # would have raised AssertionError (surfaced as orchestration=failed) if leaked
    assert "orchestration=completed" in response.text
