"""
Day-11 LLM job orchestration -- the only place that (a) builds a prompt,
(b) calls an `LLMClient`, (c) validates the response against a schema, (d)
falls back deterministically on any failure, and (e) persists an audit
trail. Three public pure functions, one per LLM job, plus a DB-aware
`..._and_log` wrapper for each (same pure-function / DB-wrapper split
established by policy/decision_engine.py in Day 9).

FAIL-SAFE GUARANTEE (brief section 5 + 10): every one of these functions
ALWAYS returns a valid `LLMResult` -- never raises, never blocks, never
depends on the LLM call succeeding. The recovery decision (policy/*) is
made entirely upstream, before any of these functions are ever called,
and none of them can affect it (see llm/service.py's callers in
scripts/run_llm_demo.py, which call the policy layer first and pass its
already-final decision fields in as plain data).

CONTEXT BOUNDARY (brief section 7): the two per-event job functions
(`generate_outreach_microcopy`, `parse_promise_to_pay`) take only
explicit, individually-named parameters -- never a raw context dict or
dataframe row -- so a hidden synthetic field (archetype,
recovery_probability_latent, expected_recovery_value_latent, or a
counterfactual-outcome column) cannot reach them by construction; there is
no dict for such a field to hide inside. `sanitize_context()` below is a
second, defense-in-depth layer for the one place a dict genuinely is
threaded through (`generate_batch_explanation`'s `report_summary`).
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone

from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from app.logging_config import log
from app.models import AuditLog, LLMInvocation
from llm.client import LLMClient, LLMProviderError, get_llm_client
from llm.prompts import (
    BATCH_EXPLANATION_PROMPT_VERSION,
    BATCH_EXPLANATION_SYSTEM_PROMPT,
    OUTREACH_MICROCOPY_PROMPT_VERSION,
    OUTREACH_MICROCOPY_SYSTEM_PROMPT,
    PROMISE_TO_PAY_PROMPT_VERSION,
    PROMISE_TO_PAY_SYSTEM_PROMPT,
    batch_explanation_user_prompt,
    outreach_microcopy_user_prompt,
    promise_to_pay_user_prompt,
)
from llm.schemas import BatchExplanationOutput, LLMResult, OutreachMicrocopyOutput, PromiseToPayOutput

# Per-event hidden synthetic-benchmark fields that must never reach the LLM
# layer (brief section 7). The two per-event job functions below never
# accept a dict in the first place (see module docstring), so this set is
# defense-in-depth for `sanitize_context()`, used only by the batch job's
# report-dict path.
FORBIDDEN_CONTEXT_KEYS = frozenset(
    {
        "archetype",
        "recovery_probability_latent",
        "expected_recovery_value_latent",
        "recovered_within_14d",
        "recovered_at",
        "recovered_via",
        "amount_recovered",
    }
)


def sanitize_context(context: dict) -> dict:
    """Defense-in-depth (brief section 7): recursively strip any forbidden
    key from a dict before it can be templated into a prompt. Logs (never
    raises) if anything was stripped -- generation should still proceed
    safely without those fields rather than crash the whole request."""

    def _clean(value):
        if isinstance(value, dict):
            leaked = FORBIDDEN_CONTEXT_KEYS & value.keys()
            if leaked:
                log.warning("Stripped forbidden synthetic fields before LLM call: %s", sorted(leaked))
            return {k: _clean(v) for k, v in value.items() if k not in FORBIDDEN_CONTEXT_KEYS}
        if isinstance(value, list):
            return [_clean(v) for v in value]
        return value

    return _clean(context)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _invoke(
    *, client: LLMClient, system_prompt: str, user_prompt: str,
    schema_cls: type[BaseModel], task_name: str, prompt_version: str, fallback_result: BaseModel,
) -> LLMResult:
    """Shared call/validate/fallback machinery for all 3 jobs (brief
    section 5): provider unavailable, timeout, SDK exception, invalid
    JSON, schema-invalid JSON, and empty response are ALL caught here and
    turned into a `success=False` LLMResult carrying the caller-supplied
    deterministic `fallback_result` -- never an unhandled exception."""
    error_type: str | None = None
    try:
        raw = client.complete(system_prompt, user_prompt)
        if not raw or not raw.strip():
            raise LLMProviderError("empty_response")
        parsed = json.loads(raw)
        validated = schema_cls.model_validate(parsed)
        return LLMResult(
            task_name=task_name, model_name=client.model_name, prompt_version=prompt_version,
            provider=client.provider_name, success=True, structured_result=validated.model_dump(),
            error_type=None, created_at=_utcnow(),
        )
    except LLMProviderError as exc:
        error_type = f"provider_error:{exc}"
    except json.JSONDecodeError:
        error_type = "invalid_json"
    except ValidationError:
        error_type = "schema_validation_error"
    except Exception as exc:  # noqa: BLE001 -- fail closed: no LLM failure mode may escape as an unhandled exception
        error_type = f"unexpected_error:{type(exc).__name__}"
        log.error("Unexpected LLM failure for task=%s: %s", task_name, type(exc).__name__)

    log.warning("LLM task=%s failed (%s) -- using deterministic fallback", task_name, error_type)
    return LLMResult(
        task_name=task_name, model_name=getattr(client, "model_name", "unknown"), prompt_version=prompt_version,
        provider=getattr(client, "provider_name", "unknown"), success=False,
        structured_result=fallback_result.model_dump(), error_type=error_type, created_at=_utcnow(),
    )


# ---------------------------------------------------------------------------
# Job 1: outreach microcopy generation
# ---------------------------------------------------------------------------

_FALLBACK_MICROCOPY = {
    ("en", True): "Your recent payment did not go through. We will automatically retry it soon.",
    ("en", False): "Your recent payment did not go through. Please check your payment method.",
    ("hi", True): "aapka payment poora nahi ho paaya. hum jald hi dobara koshish karenge.",
    ("hi", False): "aapka payment poora nahi ho paaya. kripya apna payment tarika check karein.",
    ("hinglish", True): "Aapka payment complete nahi hua. Hum jald hi dobara try karenge.",
    ("hinglish", False): "Aapka payment complete nahi hua. Kripya apna payment method check karein.",
}


def generate_outreach_microcopy(
    *, failure_bucket: str, customer_segment: str, language: str,
    will_retry: bool, retry_window_description: str | None, amount_rupees: float,
    client: LLMClient | None = None,
) -> LLMResult:
    """Job 1 (pure -- no DB). `will_retry` / `retry_window_description` are
    plain facts the policy layer already decided (brief: LLM is downstream
    of policy, never decides retry timing itself)."""
    client = client or get_llm_client()
    system_prompt = OUTREACH_MICROCOPY_SYSTEM_PROMPT
    user_prompt = outreach_microcopy_user_prompt(
        failure_bucket=failure_bucket, customer_segment=customer_segment, language=language,
        will_retry=will_retry, retry_window_description=retry_window_description, amount_rupees=amount_rupees,
    )
    fallback_text = _FALLBACK_MICROCOPY.get((language, will_retry), _FALLBACK_MICROCOPY[("en", will_retry)])
    fallback = OutreachMicrocopyOutput(message_text=fallback_text, language=language if language in ("en", "hi", "hinglish") else "en", failure_bucket=failure_bucket, customer_segment=customer_segment)
    return _invoke(
        client=client, system_prompt=system_prompt, user_prompt=user_prompt, schema_cls=OutreachMicrocopyOutput,
        task_name="outreach_microcopy", prompt_version=OUTREACH_MICROCOPY_PROMPT_VERSION, fallback_result=fallback,
    )


def generate_outreach_microcopy_and_log(
    db: Session, *, event_id: int, failure_bucket: str, customer_segment: str, language: str,
    will_retry: bool, retry_window_description: str | None, amount_rupees: float, client: LLMClient | None = None,
) -> tuple[LLMResult, LLMInvocation]:
    result = generate_outreach_microcopy(
        failure_bucket=failure_bucket, customer_segment=customer_segment, language=language,
        will_retry=will_retry, retry_window_description=retry_window_description, amount_rupees=amount_rupees, client=client,
    )
    return result, _persist(db, event_id=event_id, batch_id=None, result=result)


# ---------------------------------------------------------------------------
# Job 2: promise-to-pay parsing
# ---------------------------------------------------------------------------

_FALLBACK_PROMISE_TO_PAY = PromiseToPayOutput(date=None, confidence=0.0, channel="unspecified")


def parse_promise_to_pay(*, customer_reply_text: str, today: date | None = None, client: LLMClient | None = None) -> LLMResult:
    """Job 2 (pure -- no DB). Fallback is always the "we don't know"
    object -- never a fabricated date/channel (brief section 4/5)."""
    client = client or get_llm_client()
    today = today or datetime.now(timezone.utc).date()
    system_prompt = PROMISE_TO_PAY_SYSTEM_PROMPT
    user_prompt = promise_to_pay_user_prompt(customer_reply_text=customer_reply_text, today=today)
    return _invoke(
        client=client, system_prompt=system_prompt, user_prompt=user_prompt, schema_cls=PromiseToPayOutput,
        task_name="promise_to_pay_parse", prompt_version=PROMISE_TO_PAY_PROMPT_VERSION, fallback_result=_FALLBACK_PROMISE_TO_PAY,
    )


def parse_promise_to_pay_and_log(
    db: Session, *, event_id: int, customer_reply_text: str, today: date | None = None, client: LLMClient | None = None,
) -> tuple[LLMResult, LLMInvocation]:
    result = parse_promise_to_pay(customer_reply_text=customer_reply_text, today=today, client=client)
    return result, _persist(db, event_id=event_id, batch_id=None, result=result)


# ---------------------------------------------------------------------------
# Job 3: batch-level plain-English explanation
# ---------------------------------------------------------------------------

_FALLBACK_BATCH_EXPLANATION = BatchExplanationOutput(
    explanation_text=(
        "A plain-English explanation could not be generated for this batch report. "
        "See the structured report data directly for the SYNTHETIC COUNTERFACTUAL EVALUATION figures."
    )
)


def generate_batch_explanation(*, report_summary: dict, client: LLMClient | None = None) -> LLMResult:
    """Job 3 (pure -- no DB). `report_summary` is an ALREADY-COMPUTED,
    already-labeled-synthetic aggregate (e.g. evaluate_decision_engine_v4.py's
    `report` dict) -- distinct from the per-event hidden fields Jobs 1/2
    must never see (see module docstring). Still passed through
    `sanitize_context()` as defense-in-depth."""
    client = client or get_llm_client()
    safe_summary = sanitize_context(report_summary)
    system_prompt = BATCH_EXPLANATION_SYSTEM_PROMPT
    user_prompt = batch_explanation_user_prompt(report_summary=safe_summary)
    return _invoke(
        client=client, system_prompt=system_prompt, user_prompt=user_prompt, schema_cls=BatchExplanationOutput,
        task_name="batch_explanation", prompt_version=BATCH_EXPLANATION_PROMPT_VERSION, fallback_result=_FALLBACK_BATCH_EXPLANATION,
    )


def generate_batch_explanation_and_log(
    db: Session, *, batch_id: str, report_summary: dict, client: LLMClient | None = None,
) -> tuple[LLMResult, LLMInvocation]:
    result = generate_batch_explanation(report_summary=report_summary, client=client)
    return result, _persist(db, event_id=None, batch_id=batch_id, result=result)


# ---------------------------------------------------------------------------
# Shared audit persistence (brief section 8)
# ---------------------------------------------------------------------------

def _persist(db: Session, *, event_id: int | None, batch_id: str | None, result: LLMResult) -> LLMInvocation:
    """Writes BOTH a dedicated `llm_invocations` row (queryable structured
    metadata) and an `audit_log` row (actor="llm", unified narrative log,
    same table every other day's decisions use). Never stores an API key,
    webhook secret, or raw auth header -- only `result.structured_result`
    (already-validated JSON) and metadata."""
    invocation = LLMInvocation(
        event_id=event_id,
        batch_id=batch_id,
        task_name=result.task_name,
        model_name=result.model_name,
        prompt_version=result.prompt_version,
        provider=result.provider,
        success=result.success,
        structured_output=json.dumps(result.structured_result),
        error_type=result.error_type,
    )
    db.add(invocation)
    db.flush()

    db.add(
        AuditLog(
            failure_event_id=event_id,
            action=f"llm_{result.task_name}_{'succeeded' if result.success else 'failed_used_fallback'}",
            reason=(
                f"task_name={result.task_name} | model_name={result.model_name} | "
                f"prompt_version={result.prompt_version} | provider={result.provider} | "
                f"success={result.success} | error_type={result.error_type} | "
                f"batch_id={batch_id} | structured_output={json.dumps(result.structured_result)}"
            ),
            actor="llm",
        )
    )
    db.commit()

    log.info(
        "LLM invocation task=%s success=%s provider=%s (llm_invocations.id=%s)",
        result.task_name, result.success, result.provider, invocation.id,
    )
    return invocation
