"""
Day-5 baselines -- for comparison against the AI-assisted policy in
evaluation/evaluate_policy.py. All three respect the same classification
gate as the main policy (retrying a hard-declined or already-cancelled
subscription doesn't make sense under any policy, naive or not) and the
same candidate-validity check. None of them tracks max-attempts or
duplicate-decision state -- those are specifically the AI-assisted policy's
guardrails (Day-5 brief section 7 frames them under "Our Policy" / section
6), not properties of a simple comparison rule evaluated per-event.
"""
from __future__ import annotations

from datetime import datetime

from policy.guardrails import is_classification_allowed, validate_candidate
from policy.retry_candidates import generate_candidates
from policy.scoring import score_candidate

NO_ACTION = "NO_ACTION"

PAYDAY_PROXIMITY_THRESHOLD_DAYS = 2  # baseline 3's "very close to payday" cutoff


def _no_action_result(event_id, subscription_id: str, reason: str) -> dict:
    return {
        "event_id": event_id,
        "subscription_id": subscription_id,
        "selected_candidate_type": NO_ACTION,
        "selected_candidate_datetime": None,
        "predicted_recovery_probability": None,
        "expected_recovery_value": None,
        "expected_incremental_value": None,
        "decision_reason": reason,
    }


def no_recovery_baseline(event_id, subscription_id: str, **_ignored) -> dict:
    """BASELINE 1: never takes any action, regardless of context."""
    return _no_action_result(event_id, subscription_id, "baseline_no_recovery: no retry action is ever taken")


def fixed_retry_baseline(
    event_id,
    subscription_id: str,
    failure_timestamp: datetime,
    amount: float,
    classification_bucket: str,
    base_probability: float,
) -> dict:
    """BASELINE 2: always chooses plus_1_day_morning (subject to the shared classification/validity gates)."""
    if not is_classification_allowed(classification_bucket):
        return _no_action_result(event_id, subscription_id, f"blocked_by_classification: bucket={classification_bucket!r}")

    candidates = {c.candidate_type: c for c in generate_candidates(failure_timestamp)}
    target = candidates["plus_1_day_morning"]
    valid, invalid_reason = validate_candidate(target, failure_timestamp)
    if not valid:
        return _no_action_result(event_id, subscription_id, f"blocked_invalid_candidate: {invalid_reason}")

    scored = score_candidate(base_probability, target, amount)
    return {
        "event_id": event_id,
        "subscription_id": subscription_id,
        "selected_candidate_type": scored["candidate_type"],
        "selected_candidate_datetime": scored["candidate_datetime"],
        "predicted_recovery_probability": scored["predicted_recovery_probability"],
        "expected_recovery_value": scored["expected_recovery_value"],
        "expected_incremental_value": scored["expected_incremental_value"],
        "decision_reason": "baseline_fixed_retry: always plus_1_day_morning",
    }


def rule_based_baseline(
    event_id,
    subscription_id: str,
    failure_timestamp: datetime,
    amount: float,
    classification_bucket: str,
    base_probability: float,
) -> dict:
    """
    BASELINE 3: very close to payday -> payday_window, otherwise ->
    plus_1_day_morning. Never selects immediate / plus_3_days /
    month_end_window -- deliberately simple and explainable.
    """
    if not is_classification_allowed(classification_bucket):
        return _no_action_result(event_id, subscription_id, f"blocked_by_classification: bucket={classification_bucket!r}")

    candidates = {c.candidate_type: c for c in generate_candidates(failure_timestamp)}
    payday = candidates["payday_window"]
    morning = candidates["plus_1_day_morning"]

    if payday.candidate_days_to_payday <= PAYDAY_PROXIMITY_THRESHOLD_DAYS:
        preferred, rule = payday, f"very close to payday (<= {PAYDAY_PROXIMITY_THRESHOLD_DAYS}d) -> payday_window"
    else:
        preferred, rule = morning, "not close to payday -> plus_1_day_morning"

    valid, invalid_reason = validate_candidate(preferred, failure_timestamp)
    if not valid:
        # Simple, explainable fallback -- try the other of the two rule candidates before giving up.
        fallback = morning if preferred is payday else payday
        fallback_valid, fallback_invalid_reason = validate_candidate(fallback, failure_timestamp)
        if not fallback_valid:
            return _no_action_result(
                event_id, subscription_id, f"blocked_invalid_candidate: {invalid_reason}; fallback also invalid: {fallback_invalid_reason}"
            )
        preferred, rule = fallback, f"{rule} was invalid ({invalid_reason}); fell back to the other candidate"

    scored = score_candidate(base_probability, preferred, amount)
    return {
        "event_id": event_id,
        "subscription_id": subscription_id,
        "selected_candidate_type": scored["candidate_type"],
        "selected_candidate_datetime": scored["candidate_datetime"],
        "predicted_recovery_probability": scored["predicted_recovery_probability"],
        "expected_recovery_value": scored["expected_recovery_value"],
        "expected_incremental_value": scored["expected_incremental_value"],
        "decision_reason": f"baseline_rule_based: {rule}",
    }
