"""
Deterministic validation for a parsed promise-to-pay object.

llm/service.py::parse_promise_to_pay extracts exactly three fields from a
customer's free-text reply -- {date, confidence, channel} -- and nothing
else; it never decides whether that parse should be trusted. This module is
the ONLY place that decision is made, deterministically, with no ML and no
LLM involved. Same separation of concerns as policy/compliance.py: the LLM
extracts, this module judges.

IMPORTANT WORDING: `confidence` here is the LLM's own self-reported
confidence that it parsed the text correctly -- not a statistically
calibrated probability. `DEFAULT_MIN_CONFIDENCE` below is a deterministic,
configurable PROJECT threshold chosen for illustrative purposes, the same
way policy/costs.py's `retry_cost=Rs5` is a project assumption, not a
calibrated statistical cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime

STATUS_VALID = "VALID"
STATUS_LOW_CONFIDENCE = "LOW_CONFIDENCE"
STATUS_INVALID_DATE = "INVALID_DATE"
STATUS_EXPIRED = "EXPIRED"
STATUS_SUPERSEDED = "SUPERSEDED"

# Project-chosen deterministic cutoff -- see module docstring. Below this, a
# parsed promise is not trusted to override the model's own retry-timing
# decision, even if the date itself parsed cleanly.
DEFAULT_MIN_CONFIDENCE = 0.6

# Reuses llm/schemas.py::ALLOWED_CHANNELS verbatim -- not redefined there to
# avoid a circular import (llm/ has no reason to depend on policy/), but kept
# in sync deliberately; tests/test_promise_to_pay.py asserts the two match.
ALLOWED_CHANNELS = ("credit_card", "debit_card", "upi_autopay", "netbanking", "unspecified")


@dataclass(frozen=True)
class PromiseValidationResult:
    status: str
    reason: str
    promised_datetime: datetime | None  # None unless status == STATUS_VALID


def validate_promise(
    *,
    parsed_date: str | None,
    confidence: float,
    channel: str,
    today: _date,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> PromiseValidationResult:
    """
    Pure function, no I/O. Checks, in this fixed order (brief section on
    promise validation, "at minimum check"):

        1. date exists
        2. date is a valid ISO date
        3. date is in the future (strictly after `today` -- a same-day
           promise is treated the same as a past one: there is no
           actionable future retry WINDOW left to schedule against it, by
           construction of this project's candidate-time architecture)
        4. confidence is above `min_confidence`

    `channel` is expected to already be one of ALLOWED_CHANNELS -- llm/schemas.py's
    PromiseToPayOutput schema guarantees this on every real code path (a
    schema-invalid channel value is rejected upstream, before this function
    is ever reached, and the LLM call is treated as a failure with the
    deterministic {date: null, confidence: 0.0, channel: "unspecified"}
    fallback). A channel outside ALLOWED_CHANNELS reaching this function
    would mean that upstream guarantee was violated -- a real bug, not a
    customer-input problem -- so it raises rather than being modeled as
    another promise status.
    """
    if channel not in ALLOWED_CHANNELS:
        raise ValueError(f"unknown channel (should be impossible via the real pipeline -- llm/schemas.py already constrains this): {channel!r}")

    if not parsed_date:
        return PromiseValidationResult(STATUS_INVALID_DATE, "no_date_extracted: the LLM did not extract a promised date from the reply", None)

    try:
        promised_date = _date.fromisoformat(parsed_date)
    except ValueError:
        return PromiseValidationResult(STATUS_INVALID_DATE, f"unparseable_date: {parsed_date!r} is not a valid ISO-8601 date", None)

    if promised_date <= today:
        return PromiseValidationResult(
            STATUS_EXPIRED,
            f"date_not_in_future: promised_date={promised_date.isoformat()} is not strictly after today={today.isoformat()}",
            None,
        )

    if confidence < min_confidence:
        return PromiseValidationResult(
            STATUS_LOW_CONFIDENCE,
            f"confidence_below_threshold: {confidence:.2f} < min_confidence={min_confidence:.2f}",
            None,
        )

    # Naive datetime, matching this project's existing convention throughout
    # policy/ and recovery/ (failure_timestamp, candidate_datetime, etc. are
    # never timezone-aware -- see policy/retry_candidates.py::generate_candidates).
    # A tz-aware value here would raise TypeError the moment it's compared
    # against a naive failure_timestamp in compliance's horizon check.
    promised_datetime = datetime.combine(promised_date, datetime.min.time())
    return PromiseValidationResult(
        STATUS_VALID,
        f"valid_promise: promised_date={promised_date.isoformat()} confidence={confidence:.2f} channel={channel}",
        promised_datetime,
    )
