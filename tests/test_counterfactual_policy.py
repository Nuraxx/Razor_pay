"""
Candidate-aware policy tests: decide_candidate_aware() and its
DB-aware wrapper. Mirrors tests/test_policy.py's structure for the heuristic
`decide()` / `decide_for_failure_event()` pair -- same guardrails,
determinism, audit logging, and idempotency guarantees must hold here too.
"""
from datetime import datetime

import pytest

from app.models import AuditLog, PolicyDecision
from policy.guardrails import MAX_RETRY_ATTEMPTS
from policy.recovery_policy import (
    NO_ACTION,
    POLICY_VERSION_CANDIDATE_AWARE,
    decide_candidate_aware,
    decide_for_failure_event_candidate_aware,
)
from policy.retry_candidates import CANDIDATE_TYPES

FAILURE_TS = datetime(2026, 3, 5, 14, 0, 0)
# All 5 candidates land within the 14-day recovery horizon from this
# timestamp (verified: payday_window/month_end_window can otherwise legitimately
# fall beyond it, per policy/guardrails.py -- irrelevant for FAILURE_TS-based
# tests above, but matters for the "does selection track the given
# probabilities" tests below, which need every candidate eligible).
FAILURE_TS_ALL_VALID = datetime(2026, 2, 24, 10, 0, 0)
UNIFORM_PROBS = {ct: 0.5 for ct in CANDIDATE_TYPES}


def _distinct_probs(best: str) -> dict:
    """One clearly-highest candidate, the rest lower -- for testing that
    selection actually tracks the supplied probabilities."""
    probs = {ct: 0.2 for ct in CANDIDATE_TYPES}
    probs[best] = 0.9
    return probs


# ---------------------------------------------------------------------------
# Candidate ranking / selection works
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("winner", CANDIDATE_TYPES)
def test_decide_candidate_aware_selects_the_highest_probability_valid_candidate(winner):
    result = decide_candidate_aware(
        event_id=1, subscription_id="sub_1", failure_timestamp=FAILURE_TS_ALL_VALID,
        amount=1000.0, classification_bucket="retryable_soft",
        candidate_probabilities=_distinct_probs(winner),
    )
    assert result.selected_candidate_type == winner
    assert result.predicted_recovery_probability == pytest.approx(0.9)
    assert result.expected_recovery_value == pytest.approx(900.0)


def test_policy_version_is_recorded_as_candidate_aware():
    result = decide_candidate_aware(
        event_id=2, subscription_id="sub_2", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", candidate_probabilities=UNIFORM_PROBS,
    )
    assert result.policy_version == POLICY_VERSION_CANDIDATE_AWARE
    assert result.policy_version != "policy-v1"  # distinct from the heuristic policy


# ---------------------------------------------------------------------------
# Oracle selection: feeding LATENT probabilities through the exact same
# function is how evaluation/evaluate_counterfactual_policy.py computes
# oracle_action -- this confirms that composition actually works.
# ---------------------------------------------------------------------------

def test_oracle_selection_via_latent_probabilities():
    latent_probs = _distinct_probs("payday_window")
    oracle_result = decide_candidate_aware(
        event_id=3, subscription_id="sub_3", failure_timestamp=FAILURE_TS_ALL_VALID,
        amount=500.0, classification_bucket="retryable_soft", candidate_probabilities=latent_probs,
    )
    assert oracle_result.selected_candidate_type == "payday_window"


# ---------------------------------------------------------------------------
# Guardrails (same as the heuristic policy)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bucket", ["hard_decline", "customer_cancelled", "unmapped"])
def test_no_action_for_non_retryable_soft_buckets(bucket):
    result = decide_candidate_aware(
        event_id=4, subscription_id="sub_4", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket=bucket, candidate_probabilities=UNIFORM_PROBS,
    )
    assert result.selected_candidate_type == NO_ACTION


def test_no_action_beyond_max_retry_attempts():
    result = decide_candidate_aware(
        event_id=5, subscription_id="sub_5", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", candidate_probabilities=UNIFORM_PROBS,
        attempts_so_far=MAX_RETRY_ATTEMPTS,
    )
    assert result.selected_candidate_type == NO_ACTION
    assert "max_retry_attempts" in result.decision_reason


def test_probabilities_stay_within_unit_interval_even_if_supplied_out_of_range():
    out_of_range = {ct: 1.5 for ct in CANDIDATE_TYPES}  # a buggy upstream model could do this
    result = decide_candidate_aware(
        event_id=6, subscription_id="sub_6", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", candidate_probabilities=out_of_range,
    )
    assert 0.0 <= result.predicted_recovery_probability <= 1.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_decide_candidate_aware_is_deterministic():
    kwargs = dict(
        event_id=7, subscription_id="sub_7", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", candidate_probabilities=_distinct_probs("plus_3_days"),
    )
    assert decide_candidate_aware(**kwargs) == decide_candidate_aware(**kwargs)


# ---------------------------------------------------------------------------
# DB-backed: audit logging + idempotency, same guarantees as the heuristic policy
# ---------------------------------------------------------------------------

def test_decide_for_failure_event_candidate_aware_creates_decision_and_audit_rows(test_db_session):
    db = test_db_session()
    row, created = decide_for_failure_event_candidate_aware(
        db, event_id=101, subscription_id="sub_cf_1", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", candidate_probabilities=_distinct_probs("immediate"),
    )
    assert created is True
    assert isinstance(row, PolicyDecision)
    assert row.policy_version == POLICY_VERSION_CANDIDATE_AWARE
    assert row.selected_candidate_type == "immediate"

    audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 101).all()
    assert len(audit_rows) == 1
    assert audit_rows[0].actor == "policy"
    db.close()


def test_decide_for_failure_event_candidate_aware_is_idempotent(test_db_session):
    db = test_db_session()
    first, first_created = decide_for_failure_event_candidate_aware(
        db, event_id=102, subscription_id="sub_cf_2", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", candidate_probabilities=UNIFORM_PROBS,
    )
    second, second_created = decide_for_failure_event_candidate_aware(
        db, event_id=102, subscription_id="sub_cf_2", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", candidate_probabilities=UNIFORM_PROBS,
    )
    assert first_created is True
    assert second_created is False
    assert first.id == second.id
    assert db.query(PolicyDecision).filter(PolicyDecision.event_id == 102).count() == 1

    audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 102).order_by(AuditLog.id).all()
    assert len(audit_rows) == 2
    assert audit_rows[1].action == "policy_decision_skipped_duplicate"
    db.close()


def test_idempotency_shared_across_policy_versions(test_db_session):
    """An event_id decided once by decide_for_failure_event_candidate_aware
    must never be re-decided by the heuristic decide_for_failure_event either --
    idempotency is keyed on event_id alone, not per policy version."""
    from policy.recovery_policy import decide_for_failure_event

    db = test_db_session()
    decide_for_failure_event_candidate_aware(
        db, event_id=103, subscription_id="sub_cf_3", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", candidate_probabilities=UNIFORM_PROBS,
    )
    row, created = decide_for_failure_event(
        db, event_id=103, subscription_id="sub_cf_3", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", base_probability=0.5,
    )
    assert created is False
    assert row.policy_version == POLICY_VERSION_CANDIDATE_AWARE  # the original v2 decision, untouched
    db.close()


def test_max_retry_attempts_enforced_across_candidate_aware_calls(test_db_session):
    db = test_db_session()
    from datetime import timedelta

    for i in range(MAX_RETRY_ATTEMPTS):
        decide_for_failure_event_candidate_aware(
            db, event_id=200 + i, subscription_id="sub_cf_maxattempts",
            failure_timestamp=FAILURE_TS + timedelta(days=i * 20),
            amount=1000.0, classification_bucket="retryable_soft", candidate_probabilities=UNIFORM_PROBS,
        )

    blocked_row, created = decide_for_failure_event_candidate_aware(
        db, event_id=300, subscription_id="sub_cf_maxattempts",
        failure_timestamp=FAILURE_TS + timedelta(days=100),
        amount=1000.0, classification_bucket="retryable_soft", candidate_probabilities=UNIFORM_PROBS,
    )
    assert created is True
    assert blocked_row.selected_candidate_type == NO_ACTION
    assert "max_retry_attempts" in blocked_row.decision_reason
    db.close()
