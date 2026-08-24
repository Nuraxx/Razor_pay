"""
Day-11 structured LLM output schemas (Pydantic).

One schema per LLM job's structured result, plus `LLMResult` -- the
universal envelope every job returns (brief section 3: task name, model
name, prompt/version, structured result, created_at -- plus provider /
success / error_type for auditability, brief section 8).

`extra="forbid"` on every schema: an LLM response carrying an unexpected
extra field is treated the same as a missing/malformed one -- validation
failure, not silently ignored, per brief section 3's "Validation failure
must be handled explicitly."

Channel/instrument vocabulary reuses the project's own existing enums
(`INSTRUMENTS` in data/generate_synthetic_dataset.py: credit_card,
debit_card, upi_autopay, netbanking) rather than inventing a new one.
"""
from __future__ import annotations

from datetime import date as _date
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TaskName = Literal["outreach_microcopy", "promise_to_pay_parse", "batch_explanation"]
Provider = Literal["mock", "anthropic"]

ALLOWED_LANGUAGES = ("en", "hi", "hinglish")
ALLOWED_CHANNELS = ("credit_card", "debit_card", "upi_autopay", "netbanking", "unspecified")


class OutreachMicrocopyOutput(BaseModel):
    """Job 1 structured result -- a single piece of customer-facing outreach copy."""

    model_config = ConfigDict(extra="forbid")

    message_text: str = Field(..., min_length=1, max_length=800)
    language: Literal["en", "hi", "hinglish"]
    failure_bucket: str = Field(..., min_length=1)  # echoed back from input, for audit traceability
    customer_segment: str = Field(..., min_length=1)  # echoed back from input, for audit traceability


class PromiseToPayOutput(BaseModel):
    """Job 2 structured result -- parsed from a free-text customer reply.

    `date`: ISO-8601 `YYYY-MM-DD`, or None if no date could be extracted
        from the text (never guessed/invented -- see llm/prompts.py).
    `confidence`: the model's own confidence that this is a genuine,
        correctly-parsed promise-to-pay (0.0-1.0), not the customer's
        stated certainty.
    `channel`: the payment instrument the customer mentions, if any, else
        "unspecified" -- never invented (see ALLOWED_CHANNELS above).
    """

    model_config = ConfigDict(extra="forbid")

    date: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    channel: Literal["credit_card", "debit_card", "upi_autopay", "netbanking", "unspecified"]

    @field_validator("date")
    @classmethod
    def _date_must_be_iso_or_none(cls, v: str | None) -> str | None:
        if v is None:
            return v
        _date.fromisoformat(v)  # raises ValueError -> surfaces as a pydantic ValidationError
        return v


class BatchExplanationOutput(BaseModel):
    """Job 3 structured result -- a plain-English paragraph summarizing an
    already-computed, already-labeled-synthetic batch evaluation report."""

    model_config = ConfigDict(extra="forbid")

    explanation_text: str = Field(..., min_length=1, max_length=3000)


class LLMResult(BaseModel):
    """The universal envelope every LLM job call returns (brief sections 3 + 8)."""

    model_config = ConfigDict(extra="forbid")

    task_name: TaskName
    model_name: str
    prompt_version: str
    provider: Provider
    success: bool
    structured_result: dict | None  # validated schema's .model_dump(), or None if success=False
    error_type: str | None = None
    created_at: datetime
