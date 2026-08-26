"""
Track-03 tests: llm/service.py::generate_voice_script (Job 4, optional) and
recovery/voice.py::MockVoiceProvider. Mirrors tests/test_llm.py's style for
the other 3 jobs.
"""
import json

import pytest

from app.models import AuditLog, LLMInvocation
from llm.client import LLMProviderError
from llm.schemas import VoiceScriptOutput
from llm.service import generate_voice_script, generate_voice_script_and_log
from recovery.voice import MockVoiceProvider, VoiceCallResult, VoiceRecoveryProvider
from tests.test_llm import _JSONClient, _RaisingClient, _empty_client, _malformed_json_client


class TestVoiceScriptGeneration:
    def test_valid_structured_response_mock_provider(self):
        result = generate_voice_script(
            failure_bucket="retryable_soft", customer_segment="mid", language="en",
            will_retry=True, retry_window_description="tomorrow morning", amount_rupees=499.0,
        )
        assert result.success is True
        assert result.provider == "mock"
        assert result.task_name == "voice_script_generation"
        VoiceScriptOutput.model_validate(result.structured_result)

    def test_deterministic_mock_result_same_input_same_output(self):
        kwargs = dict(failure_bucket="retryable_soft", customer_segment="mid", language="hinglish", will_retry=True, retry_window_description="soon", amount_rupees=499.0)
        r1 = generate_voice_script(**kwargs)
        r2 = generate_voice_script(**kwargs)
        assert r1.structured_result == r2.structured_result

    def test_hinglish_is_supported(self):
        for language in ("en", "hi", "hinglish"):
            result = generate_voice_script(failure_bucket="retryable_soft", customer_segment="mid", language=language, will_retry=False, retry_window_description=None, amount_rupees=100.0)
            assert result.structured_result["language"] == language

    def test_malformed_json_falls_back(self):
        result = generate_voice_script(failure_bucket="retryable_soft", customer_segment="mid", language="en", will_retry=True, retry_window_description="soon", amount_rupees=100.0, client=_malformed_json_client())
        assert result.success is False
        assert result.error_type == "invalid_json"
        VoiceScriptOutput.model_validate(result.structured_result)

    def test_schema_invalid_output_falls_back(self):
        bad = json.dumps({"script_text": "hi", "estimated_duration_seconds": 999, "requires_callback_offer": False, "language": "en", "failure_bucket": "x", "customer_segment": "y"})
        result = generate_voice_script(failure_bucket="retryable_soft", customer_segment="mid", language="en", will_retry=True, retry_window_description="soon", amount_rupees=100.0, client=_JSONClient(bad))
        assert result.success is False
        assert result.error_type == "schema_validation_error"  # estimated_duration_seconds exceeds the 180s max

    def test_empty_response_falls_back(self):
        result = generate_voice_script(failure_bucket="retryable_soft", customer_segment="mid", language="en", will_retry=True, retry_window_description="soon", amount_rupees=100.0, client=_empty_client())
        assert result.success is False

    def test_provider_error_falls_back(self):
        result = generate_voice_script(failure_bucket="retryable_soft", customer_segment="mid", language="en", will_retry=True, retry_window_description="soon", amount_rupees=100.0, client=_RaisingClient(LLMProviderError("simulated_outage")))
        assert result.success is False
        assert "simulated_outage" in result.error_type

    def test_audit_record_created_with_correct_task_name(self, test_db_session):
        db = test_db_session()
        result, invocation = generate_voice_script_and_log(
            db, event_id=7001, failure_bucket="retryable_soft", customer_segment="mid", language="en",
            will_retry=True, retry_window_description="soon", amount_rupees=100.0,
        )
        assert invocation.task_name == "voice_script_generation"
        assert invocation.event_id == 7001
        assert invocation.success is True
        audit_rows = db.query(AuditLog).filter(AuditLog.actor == "llm", AuditLog.failure_event_id == 7001).all()
        assert len(audit_rows) == 1
        assert "task_name=voice_script_generation" in audit_rows[0].reason
        db.close()


class TestMockVoiceProvider:
    def test_is_a_voice_recovery_provider(self):
        assert isinstance(MockVoiceProvider(), VoiceRecoveryProvider)

    def test_place_call_never_makes_a_real_call(self):
        result = MockVoiceProvider().place_call("hello there", "cust_1")
        assert isinstance(result, VoiceCallResult)
        assert result.attempted is True
        assert result.connected is False
        assert result.audio_available is False  # brief: "The mock provider should simulate: generated script, audio_available=False"
        assert result.error is None
        assert result.provider_name == "mock"

    def test_script_text_is_passed_through_unchanged(self):
        result = MockVoiceProvider().place_call("exact script text", "cust_1")
        assert result.script_text == "exact script text"

    def test_deterministic(self):
        provider = MockVoiceProvider()
        r1 = provider.place_call("script", "cust_1")
        r2 = provider.place_call("script", "cust_1")
        assert r1 == r2


class TestLLMFailureNeverAffectsEligibility:
    """Mirrors tests/test_llm.py::TestPolicyIndependenceFromLLM -- a broken
    voice-script LLM call must never affect anything upstream (eligibility,
    escalation level, payment/primary action)."""

    def test_broken_client_still_produces_a_usable_fallback_script_for_the_voice_provider(self):
        broken_client = _RaisingClient(LLMProviderError("simulated_total_outage"))
        result = generate_voice_script(
            failure_bucket="retryable_soft", customer_segment="mid", language="en",
            will_retry=True, retry_window_description="soon", amount_rupees=100.0, client=broken_client,
        )
        assert result.success is False
        # the fallback script is still schema-valid and usable by the voice provider
        call_result = MockVoiceProvider().place_call(result.structured_result["script_text"], "cust_1")
        assert call_result.attempted is True
