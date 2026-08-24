"""
Day-5 hard guardrails -- deterministic, no LLM, no ML. Shared by the main
AI-assisted policy (policy/recovery_policy.py) and the two non-trivial
baselines (policy/baselines.py); the "No Recovery" baseline needs none of
this since it never selects an action.

Guardrail -> Day-5 brief section 7 requirement:
  ALLOWED_CLASSIFICATION_BUCKET check  -> "no action when classification is
      not retryable_soft" AND "no action after a cancellation state"
      (customer_cancelled is itself a cancellation state -- one check covers
      both; there is no separate cancellation flag in this project's schema)
  validate_candidate() horizon check   -> "no action if the selected retry
      time is invalid"
  validate_candidate() ordering check  -> "no action if candidate time is
      not after failure"
  MAX_RETRY_ATTEMPTS (enforced in policy/recovery_policy.py, which has the
      attempt-count state)                -> "maximum retry attempts"
  duplicate-decision check (enforced in policy/recovery_policy.py, which has
      the persisted decision state)        -> "duplicate-action prevention"
"""
from __future__ import annotations

from datetime import datetime, timedelta

from policy.retry_candidates import Candidate

ALLOWED_CLASSIFICATION_BUCKET = "retryable_soft"
MAX_RETRY_ATTEMPTS = 3

# A candidate scheduled beyond this many days after the failure cannot
# possibly contribute to "recovered_within_14d" -- the project's own
# objective -- so it is invalid for this policy regardless of its score.
# This has real bite: payday_window / month_end_window can legitimately
# land beyond 14 days out if a failure happens just after that window closes.
MAX_CANDIDATE_HORIZON_DAYS = 14


def is_classification_allowed(classification_bucket: str) -> bool:
    return classification_bucket == ALLOWED_CLASSIFICATION_BUCKET


def validate_candidate(candidate: Candidate, failure_timestamp: datetime) -> tuple[bool, str | None]:
    """Returns (is_valid, reason_if_invalid)."""
    if candidate.candidate_datetime <= failure_timestamp:
        return False, "candidate_not_after_failure"
    if candidate.candidate_datetime > failure_timestamp + timedelta(days=MAX_CANDIDATE_HORIZON_DAYS):
        return False, f"candidate_beyond_{MAX_CANDIDATE_HORIZON_DAYS}_day_recovery_horizon"
    return True, None
