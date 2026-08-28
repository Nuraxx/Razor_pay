"""
BUG-4 regression tests (pre-submission audit) for
recovery/live_feature_enrichment.py -- the live webhook -> Model B
feature-enrichment boundary. Covers: feature-source classification, the
DERIVED_FROM_AUTHORITATIVE_DATA / WEBHOOK_NATIVE computations, the optional
RAZORPAY_API_ENRICHED path's full failure-mode matrix (never crashes, never
raises), determinism (no hidden randomness), and the anti-leakage guarantee
that the synthetic data generator's random entity-simulation functions are
never invoked from this module.

End-to-end webhook-level integration tests (complete vs. incomplete feature
vector -> Model B invoked or not) live in
tests/test_live_feature_enrichment_webhook_integration.py.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from app.config import settings
from app.models import RawEvent
from policy.decision_engine import EVENT_FEATURE_KEYS
from recovery.live_feature_enrichment import (
    FEATURE_SOURCES,
    LiveFeatureResult,
    _count_prior_failures,
    _extract_primary_instrument,
    _extract_tenure_days,
    build_live_features,
)

FAILURE_TS = datetime(2026, 2, 24, 10, 0, 0)


def _raw_event(**overrides) -> RawEvent:
    payload = overrides.pop("raw_payload_dict", {
        "event": "payment.failed",
        "payload": {"payment": {"entity": {"id": "pay_test", "method": "upi"}}},
    })
    defaults = dict(
        razorpay_event_id="evt_test", event_type="payment.failed", payment_id="pay_test",
        subscription_id="sub_test", amount=29900, currency="INR", signature_verified=True,
        raw_payload=json.dumps(payload),
    )
    defaults.update(overrides)
    return RawEvent(**defaults)


# ---------------------------------------------------------------------------
# Feature-source classification -- every key accounted for, no duplicates
# ---------------------------------------------------------------------------

def test_feature_sources_classifies_exactly_model_bs_12_required_keys():
    assert set(FEATURE_SOURCES) == set(EVENT_FEATURE_KEYS)
    assert len(FEATURE_SOURCES) == 12


def test_feature_sources_uses_only_the_five_documented_categories():
    allowed = {"WEBHOOK_NATIVE", "DERIVED_FROM_AUTHORITATIVE_DATA", "RAZORPAY_API_ENRICHED", "MERCHANT_PROFILE", "UNAVAILABLE"}
    assert set(FEATURE_SOURCES.values()) <= allowed


def test_at_least_five_features_have_no_real_source_today():
    # Pins the honest conclusion: even with enrichment fully enabled and
    # succeeding, a genuine live subscription webhook can never produce a
    # complete Model B feature vector -- Model B is never actually invoked
    # for real live traffic (see module docstring's COMPLETENESS section).
    unavailable_today = [k for k, v in FEATURE_SOURCES.items() if v in ("UNAVAILABLE", "MERCHANT_PROFILE")]
    assert len(unavailable_today) >= 5


# ---------------------------------------------------------------------------
# DERIVED_FROM_AUTHORITATIVE_DATA
# ---------------------------------------------------------------------------

def test_count_prior_failures_counts_only_earlier_same_subscription_payment_failed_events(test_db_session):
    db = test_db_session()
    db.add(_raw_event(razorpay_event_id="e1", subscription_id="sub_A"))
    db.add(_raw_event(razorpay_event_id="e2", subscription_id="sub_A"))
    db.add(_raw_event(razorpay_event_id="e3", subscription_id="sub_B"))  # different subscription -- must not count
    db.add(_raw_event(razorpay_event_id="e4", subscription_id="sub_A", event_type="payment.captured"))  # not a failure -- must not count
    db.commit()

    third_failure_for_A = _raw_event(razorpay_event_id="e5", subscription_id="sub_A")
    db.add(third_failure_for_A)
    db.commit()

    assert _count_prior_failures(db, "sub_A", third_failure_for_A.id) == 2
    db.close()


def test_day_of_month_and_payday_window_are_derived_purely_from_the_failure_timestamp(test_db_session):
    db = test_db_session()
    event = _raw_event()
    db.add(event)
    db.commit()

    result = build_live_features(db, event, "sub_test", FAILURE_TS)
    assert result.features["day_of_month"] == 24
    assert isinstance(result.features["days_to_nearest_payday_window"], int)
    db.close()


# ---------------------------------------------------------------------------
# WEBHOOK_NATIVE
# ---------------------------------------------------------------------------

def test_primary_instrument_extracted_from_the_stored_raw_payload():
    event = _raw_event(raw_payload_dict={"event": "payment.failed", "payload": {"payment": {"entity": {"method": "netbanking"}}}})
    assert _extract_primary_instrument(event) == "netbanking"


def test_primary_instrument_is_none_not_fabricated_when_absent():
    event = _raw_event(raw_payload_dict={"event": "payment.failed", "payload": {}})
    assert _extract_primary_instrument(event) is None


def test_primary_instrument_extraction_never_raises_on_malformed_payload():
    event = _raw_event(raw_payload="not valid json {{{")
    assert _extract_primary_instrument(event) is None


# ---------------------------------------------------------------------------
# RAZORPAY_API_ENRICHED -- disabled by default (safe/offline)
# ---------------------------------------------------------------------------

def test_enrichment_is_not_attempted_when_disabled_by_default(test_db_session, monkeypatch):
    assert settings.LIVE_FEATURE_ENRICHMENT_ENABLED is False  # the documented safe default
    db = test_db_session()
    event = _raw_event()
    db.add(event)
    db.commit()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("no network call should be attempted when enrichment is disabled")

    monkeypatch.setattr(httpx, "Client", _fail_if_called)
    result = build_live_features(db, event, "sub_test", FAILURE_TS)
    assert result.enrichment_attempted is False
    assert "tenure_days" not in result.features
    db.close()


def test_incomplete_feature_vector_lists_missing_keys_honestly(test_db_session):
    db = test_db_session()
    event = _raw_event()
    db.add(event)
    db.commit()
    result = build_live_features(db, event, "sub_test", FAILURE_TS)
    assert result.complete is False
    assert set(result.missing_features) | set(result.features) == set(EVENT_FEATURE_KEYS)
    assert "bank_network_conditions" in result.missing_features  # a genuinely UNAVAILABLE key
    db.close()


# ---------------------------------------------------------------------------
# RAZORPAY_API_ENRICHED -- enabled: success + full failure-mode matrix.
# Never raises; every mode degrades to "not enriched".
# ---------------------------------------------------------------------------

@pytest.fixture()
def _enrichment_enabled(monkeypatch):
    monkeypatch.setattr(settings, "LIVE_FEATURE_ENRICHMENT_ENABLED", True)
    monkeypatch.setattr(settings, "RAZORPAY_KEY_ID", "rzp_test_fake_key_id")
    monkeypatch.setattr(settings, "RAZORPAY_KEY_SECRET", "fake_key_secret_never_logged")


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_enrichment_success_adds_tenure_days(test_db_session, _enrichment_enabled):
    db = test_db_session()
    event = _raw_event()
    db.add(event)
    db.commit()

    start_at = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/subscriptions/sub_test"
        return httpx.Response(200, json={"id": "sub_test", "start_at": start_at})

    result = build_live_features(db, event, "sub_test", FAILURE_TS, http_client=_mock_client(handler))
    assert result.enrichment_attempted is True
    assert result.enrichment_succeeded is True
    assert result.features["tenure_days"] == (FAILURE_TS - datetime(2025, 1, 1)).days
    db.close()


@pytest.mark.parametrize("status_code", [401, 403, 404, 429, 500, 503])
def test_enrichment_degrades_cleanly_on_http_error_status(test_db_session, _enrichment_enabled, status_code):
    db = test_db_session()
    event = _raw_event()
    db.add(event)
    db.commit()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"error": {"description": "nope"}})

    result = build_live_features(db, event, "sub_test", FAILURE_TS, http_client=_mock_client(handler))
    assert result.enrichment_attempted is True
    assert result.enrichment_succeeded is False
    assert "tenure_days" not in result.features
    assert result.complete is False  # never crashes; falls through to the normal missing-features path
    db.close()


def test_enrichment_degrades_cleanly_on_timeout(test_db_session, _enrichment_enabled):
    db = test_db_session()
    event = _raw_event()
    db.add(event)
    db.commit()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("simulated timeout", request=request)

    result = build_live_features(db, event, "sub_test", FAILURE_TS, http_client=_mock_client(handler))
    assert result.enrichment_succeeded is False
    assert "tenure_days" not in result.features
    db.close()


def test_enrichment_degrades_cleanly_on_dns_or_network_failure(test_db_session, _enrichment_enabled):
    db = test_db_session()
    event = _raw_event()
    db.add(event)
    db.commit()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated DNS/network failure", request=request)

    result = build_live_features(db, event, "sub_test", FAILURE_TS, http_client=_mock_client(handler))
    assert result.enrichment_succeeded is False
    db.close()


def test_enrichment_degrades_cleanly_on_malformed_json_response(test_db_session, _enrichment_enabled):
    db = test_db_session()
    event = _raw_event()
    db.add(event)
    db.commit()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"this is not json {{{")

    result = build_live_features(db, event, "sub_test", FAILURE_TS, http_client=_mock_client(handler))
    assert result.enrichment_succeeded is False
    db.close()


def test_enrichment_degrades_cleanly_on_unexpected_response_type(test_db_session, _enrichment_enabled):
    db = test_db_session()
    event = _raw_event()
    db.add(event)
    db.commit()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "a", "dict"])

    result = build_live_features(db, event, "sub_test", FAILURE_TS, http_client=_mock_client(handler))
    assert result.enrichment_succeeded is False
    db.close()


def test_enrichment_degrades_cleanly_when_subscription_response_is_missing_start_at(test_db_session, _enrichment_enabled):
    db = test_db_session()
    event = _raw_event()
    db.add(event)
    db.commit()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "sub_test", "status": "active"})  # no start_at/created_at

    result = build_live_features(db, event, "sub_test", FAILURE_TS, http_client=_mock_client(handler))
    assert result.enrichment_succeeded is False
    assert "tenure_days" not in result.features
    db.close()


def test_enrichment_not_even_attempted_without_configured_razorpay_credentials(test_db_session, monkeypatch):
    monkeypatch.setattr(settings, "LIVE_FEATURE_ENRICHMENT_ENABLED", True)
    monkeypatch.setattr(settings, "RAZORPAY_KEY_ID", "")
    monkeypatch.setattr(settings, "RAZORPAY_KEY_SECRET", "")
    db = test_db_session()
    event = _raw_event()
    db.add(event)
    db.commit()

    def _fail_if_called(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made without credentials configured")

    result = build_live_features(db, event, "sub_test", FAILURE_TS, http_client=_mock_client(_fail_if_called))
    assert result.enrichment_succeeded is False
    db.close()


def test_extract_tenure_days_returns_none_for_implausible_values():
    assert _extract_tenure_days({"start_at": "not_a_number"}, FAILURE_TS) is None
    assert _extract_tenure_days({}, FAILURE_TS) is None
    assert _extract_tenure_days({"start_at": True}, FAILURE_TS) is None  # bool is technically an int -- must not slip through


# ---------------------------------------------------------------------------
# Never logs secrets
# ---------------------------------------------------------------------------

def test_enrichment_failure_never_logs_the_api_secret(test_db_session, _enrichment_enabled, monkeypatch, caplog):
    import logging

    db = test_db_session()
    event = _raw_event()
    db.add(event)
    db.commit()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    with caplog.at_level(logging.WARNING):
        build_live_features(db, event, "sub_test", FAILURE_TS, http_client=_mock_client(handler))

    for record in caplog.records:
        assert "fake_key_secret_never_logged" not in record.getMessage()
        assert "Authorization" not in record.getMessage()
        assert "Basic " not in record.getMessage()
    db.close()


# ---------------------------------------------------------------------------
# Determinism -- no hidden randomness (guards against any accidental
# reliance on the synthetic generator's probabilistic draws)
# ---------------------------------------------------------------------------

def test_build_live_features_is_deterministic_given_identical_inputs(test_db_session):
    db = test_db_session()
    # Distinct subscriptions -- an identical scenario twice, not one event
    # observing the other as a "prior failure" (that would be a real,
    # correct difference in output, not nondeterminism).
    event1 = _raw_event(razorpay_event_id="det_1", subscription_id="sub_det_1")
    event2 = _raw_event(razorpay_event_id="det_2", subscription_id="sub_det_2")
    db.add(event1)
    db.add(event2)
    db.commit()

    result1 = build_live_features(db, event1, "sub_det_1", FAILURE_TS)
    result2 = build_live_features(db, event2, "sub_det_2", FAILURE_TS)
    assert result1.features == result2.features


# ---------------------------------------------------------------------------
# BUG-4 requirement #20/#21: the live path must never invoke the synthetic
# data generator's random entity-simulation functions.
# ---------------------------------------------------------------------------

def test_live_feature_enrichment_never_invokes_the_synthetic_generators(test_db_session, monkeypatch):
    import data.generate_synthetic_dataset as gen

    def _forbidden(*args, **kwargs):
        raise AssertionError("the live feature-enrichment path must never call the synthetic data generator")

    monkeypatch.setattr(gen, "generate_dataset", _forbidden)
    monkeypatch.setattr(gen, "generate_subscriptions", _forbidden)
    monkeypatch.setattr(gen, "generate_failure_events_and_outcomes", _forbidden)
    monkeypatch.setattr(gen, "generate_retry_candidates", _forbidden)

    db = test_db_session()
    event = _raw_event()
    db.add(event)
    db.commit()

    result = build_live_features(db, event, "sub_test", FAILURE_TS)  # must not raise
    assert isinstance(result, LiveFeatureResult)
    db.close()
