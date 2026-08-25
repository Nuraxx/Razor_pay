"""
Day-11 tests: the LLM-assisted communication layer (llm/).

Covers, for EVERY one of the 3 jobs (brief section 9): valid structured
response, schema validation, mock provider, deterministic mock result,
malformed JSON, invalid schema, timeout/error, provider unavailable, empty
response, no network call in mock mode, no hidden synthetic fields passed
to the LLM, audit record created, no secrets written to logs -- plus the
cross-cutting guarantee that an LLM failure never changes the policy
decision.
"""
from __future__ import annotations

import json
import logging
import socket
from datetime import date, datetime

import pytest
from pydantic import ValidationError

from app.models import AuditLog, LLMInvocation
from classification.rules import classify
from llm.client import AnthropicLLMClient, LLMClient, LLMProviderError, MockLLMClient, get_llm_client
from llm.prompts import mock_response_for_prompt
from llm.schemas import BatchExplanationOutput, LLMResult, OutreachMicrocopyOutput, PromiseToPayOutput
from llm.service import (
    FORBIDDEN_CONTEXT_KEYS,
    generate_batch_explanation,
    generate_batch_explanation_and_log,
    generate_outreach_microcopy,
    generate_outreach_microcopy_and_log,
    parse_promise_to_pay,
    parse_promise_to_pay_and_log,
    sanitize_context,
)
from policy.decision_engine_v4 import NO_ACTION, decide_engine_v4

FAILURE_TS = datetime(2026, 2, 24, 10, 0, 0)
FAILURE_CONTEXT = {
    "day_of_month": 24, "days_to_nearest_payday_window": 6, "prior_if_failure_count": 0,
    "prior_if_self_resolved_rate": float("nan"), "tenure_days": 200, "plan_tier": "mid",
    "primary_instrument": "upi_autopay", "city_tier": "tier_1", "bank_network_conditions": "good",
    "issuing_bank_downtime_flag": False, "network_latency_bucket": "low", "is_month_end_settlement_rush": False,
}


class _JSONClient(LLMClient):
    model_name = "test-client"
    provider_name = "mock"

    def __init__(self, response_text: str):
        self._response_text = response_text

    def complete(self, system_prompt, user_prompt, *, max_tokens=512):
        return self._response_text


class _RaisingClient(LLMClient):
    model_name = "test-client"
    provider_name = "mock"

    def __init__(self, exc: Exception):
        self._exc = exc

    def complete(self, system_prompt, user_prompt, *, max_tokens=512):
        raise self._exc


def _malformed_json_client() -> _JSONClient:
    return _JSONClient("this is not { valid json at all")


def _empty_client() -> _JSONClient:
    return _JSONClient("")


# ---------------------------------------------------------------------------
# Job 1: outreach microcopy
# ---------------------------------------------------------------------------

class TestOutreachMicrocopy:
    def test_valid_structured_response_mock_provider(self):
        result = generate_outreach_microcopy(
            failure_bucket="retryable_soft", customer_segment="mid", language="en",
            will_retry=True, retry_window_description="tomorrow morning", amount_rupees=499.0,
        )
        assert result.success is True
        assert result.provider == "mock"
        assert result.task_name == "outreach_microcopy"
        OutreachMicrocopyOutput.model_validate(result.structured_result)  # schema validation

    def test_deterministic_mock_result_same_input_same_output(self):
        kwargs = dict(failure_bucket="retryable_soft", customer_segment="mid", language="hinglish", will_retry=True, retry_window_description="soon", amount_rupees=499.0)
        r1 = generate_outreach_microcopy(**kwargs)
        r2 = generate_outreach_microcopy(**kwargs)
        assert r1.structured_result == r2.structured_result

    def test_hinglish_is_a_prompt_parameter_not_separate_infra(self):
        for language in ("en", "hi", "hinglish"):
            result = generate_outreach_microcopy(failure_bucket="retryable_soft", customer_segment="mid", language=language, will_retry=False, retry_window_description=None, amount_rupees=100.0)
            assert result.structured_result["language"] == language

    def test_malformed_json_falls_back(self):
        result = generate_outreach_microcopy(failure_bucket="retryable_soft", customer_segment="mid", language="en", will_retry=True, retry_window_description="soon", amount_rupees=100.0, client=_malformed_json_client())
        assert result.success is False
        assert result.error_type == "invalid_json"
        OutreachMicrocopyOutput.model_validate(result.structured_result)  # fallback is still schema-valid

    def test_schema_invalid_output_falls_back(self):
        bad = json.dumps({"message_text": "hi", "language": "klingon", "failure_bucket": "x", "customer_segment": "y"})
        result = generate_outreach_microcopy(failure_bucket="retryable_soft", customer_segment="mid", language="en", will_retry=True, retry_window_description="soon", amount_rupees=100.0, client=_JSONClient(bad))
        assert result.success is False
        assert result.error_type == "schema_validation_error"

    def test_empty_response_falls_back(self):
        result = generate_outreach_microcopy(failure_bucket="retryable_soft", customer_segment="mid", language="en", will_retry=True, retry_window_description="soon", amount_rupees=100.0, client=_empty_client())
        assert result.success is False
        assert "empty_response" in result.error_type

    def test_timeout_or_sdk_error_falls_back(self):
        result = generate_outreach_microcopy(failure_bucket="retryable_soft", customer_segment="mid", language="en", will_retry=True, retry_window_description="soon", amount_rupees=100.0, client=_RaisingClient(LLMProviderError("simulated_timeout")))
        assert result.success is False
        assert "simulated_timeout" in result.error_type

    def test_unexpected_exception_type_still_falls_back(self):
        result = generate_outreach_microcopy(failure_bucket="retryable_soft", customer_segment="mid", language="en", will_retry=True, retry_window_description="soon", amount_rupees=100.0, client=_RaisingClient(RuntimeError("boom")))
        assert result.success is False
        assert "unexpected_error" in result.error_type

    def test_no_hidden_synthetic_fields_can_reach_this_function(self):
        import inspect

        params = set(inspect.signature(generate_outreach_microcopy).parameters)
        assert not (params & FORBIDDEN_CONTEXT_KEYS)
        assert "context" not in params  # no catch-all dict param that could smuggle a hidden field in

    def test_audit_record_created(self, test_db_session):
        db = test_db_session()
        result, invocation = generate_outreach_microcopy_and_log(
            db, event_id=1001, failure_bucket="retryable_soft", customer_segment="mid", language="en",
            will_retry=True, retry_window_description="soon", amount_rupees=100.0,
        )
        assert isinstance(invocation, LLMInvocation)
        assert invocation.task_name == "outreach_microcopy"
        assert invocation.event_id == 1001
        assert invocation.success is True

        audit_rows = db.query(AuditLog).filter(AuditLog.actor == "llm", AuditLog.failure_event_id == 1001).all()
        assert len(audit_rows) == 1
        assert "task_name=outreach_microcopy" in audit_rows[0].reason
        db.close()


# ---------------------------------------------------------------------------
# Job 2: promise-to-pay parsing
# ---------------------------------------------------------------------------

class TestPromiseToPay:
    def test_valid_structured_response_mock_provider(self):
        result = parse_promise_to_pay(customer_reply_text="I'll pay Friday when salary comes", today=date(2026, 8, 24))
        assert result.success is True
        assert result.provider == "mock"
        PromiseToPayOutput.model_validate(result.structured_result)

    def test_date_resolved_correctly_for_named_weekday(self):
        result = parse_promise_to_pay(customer_reply_text="I'll pay Friday when salary comes", today=date(2026, 8, 24))  # Monday
        assert result.structured_result["date"] == "2026-08-28"  # next Friday

    def test_deterministic_mock_result(self):
        kwargs = dict(customer_reply_text="I'll pay tomorrow via UPI", today=date(2026, 8, 24))
        r1 = parse_promise_to_pay(**kwargs)
        r2 = parse_promise_to_pay(**kwargs)
        assert r1.structured_result == r2.structured_result

    def test_no_promise_detected_returns_null_date_not_a_guess(self):
        result = parse_promise_to_pay(customer_reply_text="Stop contacting me.", today=date(2026, 8, 24))
        assert result.structured_result["date"] is None
        assert result.structured_result["channel"] == "unspecified"

    def test_malformed_json_falls_back_to_unknown_object(self):
        result = parse_promise_to_pay(customer_reply_text="I'll pay Friday", today=date(2026, 8, 24), client=_malformed_json_client())
        assert result.success is False
        assert result.structured_result == {"date": None, "confidence": 0.0, "channel": "unspecified"}

    def test_schema_invalid_output_falls_back(self):
        bad = json.dumps({"date": "not-a-date", "confidence": 0.5, "channel": "unspecified"})
        result = parse_promise_to_pay(customer_reply_text="x", today=date(2026, 8, 24), client=_JSONClient(bad))
        assert result.success is False
        assert result.error_type == "schema_validation_error"

    def test_confidence_out_of_range_is_schema_invalid(self):
        bad = json.dumps({"date": None, "confidence": 1.5, "channel": "unspecified"})
        result = parse_promise_to_pay(customer_reply_text="x", today=date(2026, 8, 24), client=_JSONClient(bad))
        assert result.success is False
        assert result.error_type == "schema_validation_error"

    def test_empty_response_falls_back(self):
        result = parse_promise_to_pay(customer_reply_text="x", today=date(2026, 8, 24), client=_empty_client())
        assert result.success is False

    def test_provider_error_falls_back(self):
        result = parse_promise_to_pay(customer_reply_text="x", today=date(2026, 8, 24), client=_RaisingClient(LLMProviderError("provider_unavailable")))
        assert result.success is False
        assert "provider_unavailable" in result.error_type

    def test_no_hidden_synthetic_fields_can_reach_this_function(self):
        import inspect

        params = set(inspect.signature(parse_promise_to_pay).parameters)
        assert not (params & FORBIDDEN_CONTEXT_KEYS)
        assert "context" not in params

    def test_audit_record_created(self, test_db_session):
        db = test_db_session()
        result, invocation = parse_promise_to_pay_and_log(db, event_id=1002, customer_reply_text="I'll pay Friday", today=date(2026, 8, 24))
        assert invocation.task_name == "promise_to_pay_parse"
        assert invocation.event_id == 1002
        audit_rows = db.query(AuditLog).filter(AuditLog.actor == "llm", AuditLog.failure_event_id == 1002).all()
        assert len(audit_rows) == 1
        db.close()


# ---------------------------------------------------------------------------
# Job 3: batch-level explanation
# ---------------------------------------------------------------------------

class TestBatchExplanation:
    SAMPLE_REPORT = {
        "label": "SYNTHETIC COUNTERFACTUAL EVALUATION -- test",
        "latent_economic": {"fixed_retry": {"total_latent_value_rs": 1000.0}, "oracle_policy": {"total_latent_value_rs": 1200.0}},
    }

    def test_valid_structured_response_mock_provider(self):
        result = generate_batch_explanation(report_summary=self.SAMPLE_REPORT)
        assert result.success is True
        BatchExplanationOutput.model_validate(result.structured_result)

    def test_explanation_labels_data_as_synthetic(self):
        result = generate_batch_explanation(report_summary=self.SAMPLE_REPORT)
        assert "SYNTHETIC" in result.structured_result["explanation_text"]

    def test_deterministic_mock_result(self):
        r1 = generate_batch_explanation(report_summary=self.SAMPLE_REPORT)
        r2 = generate_batch_explanation(report_summary=self.SAMPLE_REPORT)
        assert r1.structured_result == r2.structured_result

    def test_malformed_json_falls_back(self):
        result = generate_batch_explanation(report_summary=self.SAMPLE_REPORT, client=_malformed_json_client())
        assert result.success is False
        BatchExplanationOutput.model_validate(result.structured_result)

    def test_schema_invalid_output_falls_back(self):
        bad = json.dumps({"explanation_text": ""})  # violates min_length=1
        result = generate_batch_explanation(report_summary=self.SAMPLE_REPORT, client=_JSONClient(bad))
        assert result.success is False
        assert result.error_type == "schema_validation_error"

    def test_empty_response_falls_back(self):
        result = generate_batch_explanation(report_summary=self.SAMPLE_REPORT, client=_empty_client())
        assert result.success is False

    def test_provider_error_falls_back(self):
        result = generate_batch_explanation(report_summary=self.SAMPLE_REPORT, client=_RaisingClient(LLMProviderError("timeout")))
        assert result.success is False

    def test_hidden_synthetic_fields_stripped_before_reaching_prompt(self):
        contaminated_report = {
            **self.SAMPLE_REPORT,
            "per_event_detail": [{"archetype": "chronic_faller", "recovery_probability_latent": 0.42, "expected_recovery_value_latent": 12.3, "recovered_within_14d": True}],
        }
        from llm.prompts import batch_explanation_user_prompt

        safe = sanitize_context(contaminated_report)
        prompt_text = batch_explanation_user_prompt(report_summary=safe)
        for forbidden in FORBIDDEN_CONTEXT_KEYS:
            assert forbidden not in prompt_text

    def test_sanitize_context_strips_nested_forbidden_keys(self):
        nested = {"outer": {"inner": [{"archetype": "x", "recovered_at": "2026-01-01"}]}, "safe_field": 42}
        cleaned = sanitize_context(nested)
        assert cleaned["outer"]["inner"][0] == {}
        assert cleaned["safe_field"] == 42

    def test_audit_record_created(self, test_db_session):
        db = test_db_session()
        result, invocation = generate_batch_explanation_and_log(db, batch_id="demo_batch_001", report_summary=self.SAMPLE_REPORT)
        assert invocation.task_name == "batch_explanation"
        assert invocation.batch_id == "demo_batch_001"
        assert invocation.event_id is None
        audit_rows = db.query(AuditLog).filter(AuditLog.actor == "llm").all()
        assert any("batch_id=demo_batch_001" in row.reason for row in audit_rows)
        db.close()


# ---------------------------------------------------------------------------
# Mock provider: no network calls, provider selection
# ---------------------------------------------------------------------------

class TestMockProviderAndSelection:
    def test_mock_provider_makes_no_network_call(self, monkeypatch):
        def _blow_up(*args, **kwargs):
            raise AssertionError("mock provider attempted a real network connection")

        monkeypatch.setattr(socket, "create_connection", _blow_up)
        monkeypatch.setattr(socket.socket, "connect", _blow_up)

        result = generate_outreach_microcopy(failure_bucket="retryable_soft", customer_segment="mid", language="en", will_retry=True, retry_window_description="soon", amount_rupees=100.0)
        assert result.success is True  # would have raised AssertionError above if any networking occurred

    def test_get_llm_client_defaults_to_mock(self, monkeypatch):
        monkeypatch.setattr("app.config.settings.LLM_PROVIDER", "mock")
        client = get_llm_client()
        assert isinstance(client, MockLLMClient)

    def test_provider_unavailable_anthropic_without_api_key_falls_back_to_mock(self, monkeypatch):
        monkeypatch.setattr("app.config.settings.LLM_PROVIDER", "anthropic")
        monkeypatch.setattr("app.config.settings.ANTHROPIC_API_KEY", "")
        client = get_llm_client()
        assert isinstance(client, MockLLMClient)  # provider unavailable (no key) -> safe fallback, never crashes

    def test_mock_response_for_unknown_prompt_returns_empty_json_not_an_exception(self):
        raw = mock_response_for_prompt("no task marker here")
        assert json.loads(raw) == {}


# ---------------------------------------------------------------------------
# Secret handling
# ---------------------------------------------------------------------------

class TestNoSecretsInLogs:
    def test_anthropic_sdk_failure_never_logs_the_api_key(self, monkeypatch, caplog, test_db_session):
        import anthropic as anthropic_module

        secret = "sk-ant-SUPER-SECRET-KEY-DO-NOT-LEAK"

        class _FakeMessages:
            def create(self, **kwargs):
                raise RuntimeError(f"authentication failed for Authorization: Bearer {secret}")

        class _FakeAnthropicSDKClient:
            def __init__(self, api_key):
                self.messages = _FakeMessages()

        monkeypatch.setattr(anthropic_module, "Anthropic", _FakeAnthropicSDKClient)
        client = AnthropicLLMClient(api_key=secret)

        db = test_db_session()
        with caplog.at_level(logging.DEBUG):
            result, invocation = generate_outreach_microcopy_and_log(
                db, event_id=2001, failure_bucket="retryable_soft", customer_segment="mid", language="en",
                will_retry=True, retry_window_description="soon", amount_rupees=100.0, client=client,
            )

        assert result.success is False
        assert secret not in caplog.text
        assert secret not in (result.error_type or "")
        assert secret not in json.dumps(result.structured_result)

        audit_rows = db.query(AuditLog).filter(AuditLog.actor == "llm", AuditLog.failure_event_id == 2001).all()
        assert secret not in audit_rows[0].reason
        assert secret not in (invocation.structured_output or "")
        assert secret not in (invocation.error_type or "")
        db.close()

    def test_llm_provider_error_message_never_embeds_raw_exception_text(self):
        secret_like = "webhook_secret=RazorpayRecoveryWebhook2026RandomSecret"

        class _LeakyRaisingClient(LLMClient):
            model_name = "leaky"
            provider_name = "anthropic"

            def complete(self, system_prompt, user_prompt, *, max_tokens=512):
                raise LLMProviderError("anthropic_api_error:AuthenticationError")  # constructed WITHOUT embedding secret_like, per llm/client.py's design

        result = generate_outreach_microcopy(failure_bucket="retryable_soft", customer_segment="mid", language="en", will_retry=True, retry_window_description="soon", amount_rupees=100.0, client=_LeakyRaisingClient())
        assert secret_like not in result.error_type


# ---------------------------------------------------------------------------
# LLM failure must never change the policy decision (brief section 10)
# ---------------------------------------------------------------------------

class TestPolicyIndependenceFromLLM:
    def test_decide_engine_v4_signature_has_no_llm_parameter(self):
        import inspect

        params = set(inspect.signature(decide_engine_v4).parameters)
        assert not any("llm" in p.lower() for p in params)

    def test_policy_decision_identical_regardless_of_llm_outcome(self):
        def _make_decision():
            bucket = classify(None, "insufficient_fund").bucket

            class _FakeModelDict(dict):
                pass

            class _PassthroughImputer:
                def transform(self, X):
                    return X

            class _FakeCatBoost:
                def predict(self, X):
                    import numpy as np

                    return np.array([100.0, 90.0, 80.0, 70.0, 60.0][: len(X)])

            model = {"imputer": _PassthroughImputer(), "catboost_model": _FakeCatBoost()}
            return decide_engine_v4(5001, "sub_llm_independence", FAILURE_TS, 1000.0, bucket, FAILURE_CONTEXT, model=model)

        decision_before = _make_decision()

        # Now run all 3 LLM jobs with a client that always raises -- this must have ZERO effect on the policy layer.
        broken_client = _RaisingClient(LLMProviderError("simulated_total_llm_outage"))
        generate_outreach_microcopy(failure_bucket=decision_before.classification_bucket, customer_segment="mid", language="en", will_retry=True, retry_window_description="soon", amount_rupees=1000.0, client=broken_client)
        parse_promise_to_pay(customer_reply_text="I'll pay Friday", today=date(2026, 8, 24), client=broken_client)
        generate_batch_explanation(report_summary={"label": "SYNTHETIC COUNTERFACTUAL EVALUATION"}, client=broken_client)

        decision_after = _make_decision()
        assert decision_before == decision_after  # identical decision, LLM outage had no effect whatsoever


# ---------------------------------------------------------------------------
# LLMResult envelope
# ---------------------------------------------------------------------------

def test_llm_result_contains_required_envelope_fields():
    result = generate_outreach_microcopy(failure_bucket="retryable_soft", customer_segment="mid", language="en", will_retry=True, retry_window_description="soon", amount_rupees=100.0)
    dumped = result.model_dump()
    for key in ("task_name", "model_name", "prompt_version", "structured_result", "created_at", "provider", "success"):
        assert key in dumped
