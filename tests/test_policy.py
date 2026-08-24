"""
Day-5 policy layer tests: candidate generation, scoring, guardrails,
baselines, the AI-assisted policy, and its DB-backed audit trail.

Pure-function tests (candidate generation, scoring, guardrails, baselines,
`decide()`) need no DB. DB-backed tests reuse the `test_db_session` fixture
Day 1 defined in tests/conftest.py, mirroring tests/test_classification_service.py.
"""
from datetime import datetime, timedelta

import pytest

from app.models import AuditLog, PolicyDecision
from policy.baselines import (
    NO_ACTION as BASELINE_NO_ACTION,
    fixed_retry_baseline,
    no_recovery_baseline,
    rule_based_baseline,
)
from policy.guardrails import (
    ALLOWED_CLASSIFICATION_BUCKET,
    MAX_CANDIDATE_HORIZON_DAYS,
    MAX_RETRY_ATTEMPTS,
    is_classification_allowed,
    validate_candidate,
)
from policy.recovery_policy import NO_ACTION, POLICY_VERSION, decide, decide_for_failure_event
from policy.retry_candidates import CANDIDATE_TYPES, Candidate, generate_candidates
from policy.scoring import Day4ModelUnavailable, load_calibrated_model, predict_base_recovery_probability, score_candidate

FAILURE_TS = datetime(2026, 3, 5, 14, 0, 0)


# ---------------------------------------------------------------------------
# 1. Candidate feature generation
# ---------------------------------------------------------------------------

def test_generate_candidates_returns_all_five_types_in_order():
    candidates = generate_candidates(FAILURE_TS)
    assert [c.candidate_type for c in candidates] == CANDIDATE_TYPES


def test_every_candidate_occurs_after_failure():
    for c in generate_candidates(FAILURE_TS):
        assert c.candidate_datetime > FAILURE_TS


def test_candidate_features_are_internally_consistent():
    for c in generate_candidates(FAILURE_TS):
        expected_hours = round((c.candidate_datetime - FAILURE_TS).total_seconds() / 3600, 2)
        assert c.hours_from_failure == expected_hours
        assert c.candidate_day_of_month == c.candidate_datetime.day
        assert c.candidate_day_of_week == c.candidate_datetime.strftime("%A")
        assert c.candidate_days_to_payday >= 0


def test_payday_window_candidate_is_flagged_payday_aligned():
    candidates = {c.candidate_type: c for c in generate_candidates(FAILURE_TS)}
    assert candidates["payday_window"].candidate_is_payday_aligned is True


def test_month_end_window_candidate_is_flagged_month_end_aligned():
    candidates = {c.candidate_type: c for c in generate_candidates(FAILURE_TS)}
    assert candidates["month_end_window"].candidate_is_month_end_aligned is True


def test_generate_candidates_is_deterministic():
    assert generate_candidates(FAILURE_TS) == generate_candidates(FAILURE_TS)


# ---------------------------------------------------------------------------
# 2. Expected recovery value calculation
# ---------------------------------------------------------------------------

def test_expected_recovery_value_formula():
    # Mid-month failure -> plus_1_day_morning lands >7 days from any payday
    # window, so PAYDAY_PROXIMITY_MAX_BOOST is exactly 0 and the only
    # variable left is the formula itself: expected_recovery_value =
    # predicted_recovery_probability * amount.
    mid_month = datetime(2026, 3, 15, 14, 0, 0)
    candidate = {c.candidate_type: c for c in generate_candidates(mid_month)}["plus_1_day_morning"]
    assert candidate.candidate_days_to_payday > 7

    scored = score_candidate(base_probability=0.4, candidate=candidate, amount=1000.0, intervention_cost=0.0)
    assert scored["heuristic_adjustment"] == pytest.approx(0.0)
    assert scored["predicted_recovery_probability"] == pytest.approx(0.4)
    assert scored["expected_recovery_value"] == pytest.approx(0.4 * 1000.0)
    assert scored["expected_incremental_value"] == pytest.approx(scored["expected_recovery_value"])


def test_expected_incremental_value_subtracts_intervention_cost():
    candidate = generate_candidates(FAILURE_TS)[1]
    scored = score_candidate(base_probability=0.4, candidate=candidate, amount=1000.0, intervention_cost=50.0)
    assert scored["expected_incremental_value"] == pytest.approx(scored["expected_recovery_value"] - 50.0)


def test_intervention_cost_defaults_to_zero():
    candidate = generate_candidates(FAILURE_TS)[1]
    scored = score_candidate(base_probability=0.4, candidate=candidate, amount=1000.0)
    assert scored["expected_incremental_value"] == pytest.approx(scored["expected_recovery_value"])


@pytest.mark.parametrize("base_probability", [0.0, 0.5, 0.97, 1.0])
def test_predicted_recovery_probability_stays_within_unit_interval(base_probability):
    for candidate in generate_candidates(FAILURE_TS):
        scored = score_candidate(base_probability, candidate, amount=500.0)
        assert 0.0 <= scored["predicted_recovery_probability"] <= 1.0


def test_immediate_retry_penalty_reduces_adjustment_relative_to_other_types():
    from policy.scoring import IMMEDIATE_RETRY_PENALTY, heuristic_adjustment

    days_to_payday = 5  # same proximity for both, so only the immediate-specific term should differ
    immediate_adjustment = heuristic_adjustment("immediate", days_to_payday)
    other_adjustment = heuristic_adjustment("plus_1_day_morning", days_to_payday)
    assert immediate_adjustment == pytest.approx(other_adjustment - IMMEDIATE_RETRY_PENALTY)


# ---------------------------------------------------------------------------
# 3. Guardrails
# ---------------------------------------------------------------------------

def test_only_retryable_soft_is_an_allowed_classification():
    assert is_classification_allowed(ALLOWED_CLASSIFICATION_BUCKET) is True
    for bucket in ("hard_decline", "customer_cancelled", "unmapped"):
        assert is_classification_allowed(bucket) is False


def test_candidate_before_failure_is_invalid():
    bad_candidate = Candidate(
        candidate_type="immediate",
        candidate_datetime=FAILURE_TS - timedelta(hours=1),
        hours_from_failure=-1.0,
        candidate_day_of_month=FAILURE_TS.day,
        candidate_day_of_week=FAILURE_TS.strftime("%A"),
        candidate_is_payday_aligned=False,
        candidate_is_month_end_aligned=False,
        candidate_days_to_payday=5,
    )
    valid, reason = validate_candidate(bad_candidate, FAILURE_TS)
    assert valid is False
    assert reason == "candidate_not_after_failure"


def test_candidate_beyond_recovery_horizon_is_invalid():
    far_future = FAILURE_TS + timedelta(days=MAX_CANDIDATE_HORIZON_DAYS + 1)
    bad_candidate = Candidate(
        candidate_type="month_end_window",
        candidate_datetime=far_future,
        hours_from_failure=(far_future - FAILURE_TS).total_seconds() / 3600,
        candidate_day_of_month=far_future.day,
        candidate_day_of_week=far_future.strftime("%A"),
        candidate_is_payday_aligned=False,
        candidate_is_month_end_aligned=True,
        candidate_days_to_payday=10,
    )
    valid, reason = validate_candidate(bad_candidate, FAILURE_TS)
    assert valid is False
    assert "horizon" in reason


def test_normal_future_candidate_within_horizon_is_valid():
    candidate = generate_candidates(FAILURE_TS)[1]  # plus_1_day_morning
    valid, reason = validate_candidate(candidate, FAILURE_TS)
    assert valid is True
    assert reason is None


# ---------------------------------------------------------------------------
# 4. The AI-assisted policy (`decide`, pure)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bucket", ["hard_decline", "customer_cancelled", "unmapped"])
def test_decide_returns_no_action_for_non_retryable_soft_buckets(bucket):
    result = decide(
        event_id=1, subscription_id="sub_1", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket=bucket, base_probability=0.6,
    )
    assert result.selected_candidate_type == NO_ACTION
    assert result.selected_candidate_datetime is None
    assert result.predicted_recovery_probability is None
    assert "blocked_by_classification" in result.decision_reason


def test_decide_selects_a_real_candidate_for_retryable_soft():
    result = decide(
        event_id=2, subscription_id="sub_2", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", base_probability=0.6,
    )
    assert result.selected_candidate_type in CANDIDATE_TYPES
    assert result.selected_candidate_datetime > FAILURE_TS
    assert 0.0 <= result.predicted_recovery_probability <= 1.0
    assert result.expected_recovery_value == pytest.approx(result.predicted_recovery_probability * 1000.0)
    assert result.policy_version == POLICY_VERSION


def test_decide_blocks_after_max_retry_attempts():
    result = decide(
        event_id=3, subscription_id="sub_3", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", base_probability=0.6,
        attempts_so_far=MAX_RETRY_ATTEMPTS,
    )
    assert result.selected_candidate_type == NO_ACTION
    assert "max_retry_attempts" in result.decision_reason


def test_decide_is_idempotent_when_already_decided():
    result = decide(
        event_id=4, subscription_id="sub_4", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", base_probability=0.6,
        already_decided=True,
    )
    assert result.selected_candidate_type == NO_ACTION
    assert "duplicate_decision_skipped" in result.decision_reason


def test_decide_is_deterministic_for_identical_inputs():
    kwargs = dict(
        event_id=5, subscription_id="sub_5", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", base_probability=0.6,
    )
    assert decide(**kwargs) == decide(**kwargs)


def test_decide_blocks_when_all_candidates_are_invalid(monkeypatch):
    def all_invalid_candidates(_failure_timestamp):
        return [
            Candidate("immediate", FAILURE_TS - timedelta(hours=1), -1.0, FAILURE_TS.day, "Thursday", False, False, 5)
        ]

    monkeypatch.setattr("policy.recovery_policy.generate_candidates", all_invalid_candidates)
    result = decide(
        event_id=6, subscription_id="sub_6", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", base_probability=0.6,
    )
    assert result.selected_candidate_type == NO_ACTION
    assert "blocked_no_valid_candidates" in result.decision_reason


def test_decide_records_baseline_action_for_comparison():
    result = decide(
        event_id=7, subscription_id="sub_7", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", base_probability=0.6,
    )
    assert result.baseline_action in CANDIDATE_TYPES + [NO_ACTION]


# ---------------------------------------------------------------------------
# 5. Baselines
# ---------------------------------------------------------------------------

def test_no_recovery_baseline_always_no_action():
    result = no_recovery_baseline(event_id=8, subscription_id="sub_8")
    assert result["selected_candidate_type"] == BASELINE_NO_ACTION


def test_fixed_retry_baseline_always_picks_plus_1_day_morning():
    result = fixed_retry_baseline(
        event_id=9, subscription_id="sub_9", failure_timestamp=FAILURE_TS,
        amount=500.0, classification_bucket="retryable_soft", base_probability=0.3,
    )
    assert result["selected_candidate_type"] == "plus_1_day_morning"


def test_fixed_retry_baseline_blocked_for_hard_decline():
    result = fixed_retry_baseline(
        event_id=10, subscription_id="sub_10", failure_timestamp=FAILURE_TS,
        amount=500.0, classification_bucket="hard_decline", base_probability=0.3,
    )
    assert result["selected_candidate_type"] == BASELINE_NO_ACTION


def test_rule_based_baseline_prefers_payday_window_when_close_to_payday():
    near_payday = datetime(2026, 3, 1, 8, 0, 0)  # candidates() lands payday_window within 2 days
    result = rule_based_baseline(
        event_id=11, subscription_id="sub_11", failure_timestamp=near_payday,
        amount=500.0, classification_bucket="retryable_soft", base_probability=0.3,
    )
    assert result["selected_candidate_type"] == "payday_window"


def test_rule_based_baseline_falls_back_to_plus_1_day_morning_otherwise():
    mid_month = datetime(2026, 3, 15, 8, 0, 0)  # far from any payday window
    result = rule_based_baseline(
        event_id=12, subscription_id="sub_12", failure_timestamp=mid_month,
        amount=500.0, classification_bucket="retryable_soft", base_probability=0.3,
    )
    assert result["selected_candidate_type"] == "plus_1_day_morning"


def test_rule_based_baseline_blocked_for_customer_cancelled():
    result = rule_based_baseline(
        event_id=13, subscription_id="sub_13", failure_timestamp=FAILURE_TS,
        amount=500.0, classification_bucket="customer_cancelled", base_probability=0.3,
    )
    assert result["selected_candidate_type"] == BASELINE_NO_ACTION


# ---------------------------------------------------------------------------
# 6. All candidate types handled end-to-end by scoring
# ---------------------------------------------------------------------------

def test_all_candidate_types_score_without_error():
    scored_types = set()
    for candidate in generate_candidates(FAILURE_TS):
        scored = score_candidate(0.5, candidate, amount=1000.0)
        assert 0.0 <= scored["predicted_recovery_probability"] <= 1.0
        scored_types.add(scored["candidate_type"])
    assert scored_types == set(CANDIDATE_TYPES)


# ---------------------------------------------------------------------------
# 7. Day-4 model integration (uses committed model/artifacts/ if present)
# ---------------------------------------------------------------------------

def test_base_probability_from_real_calibrated_model_stays_within_unit_interval():
    try:
        model, imputer = load_calibrated_model()
    except Day4ModelUnavailable:
        pytest.skip("model/artifacts/ not present -- run model/train.py first")

    import pandas as pd
    from model.preprocessing import load_processed_splits, select_features_and_target

    _train, _val, test_df = load_processed_splits()
    X_test, _y = select_features_and_target(test_df.head(20))
    probs = predict_base_recovery_probability(X_test, model, imputer)
    assert isinstance(probs, pd.Series)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


# ---------------------------------------------------------------------------
# 8. DB-backed: audit log, idempotency, policy version, max attempts
# ---------------------------------------------------------------------------

def test_decide_for_failure_event_creates_decision_and_audit_rows(test_db_session):
    db = test_db_session()
    row, created = decide_for_failure_event(
        db, event_id=101, subscription_id="sub_db_1", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", base_probability=0.6,
    )
    assert created is True
    assert isinstance(row, PolicyDecision)
    assert row.policy_version == POLICY_VERSION

    audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 101).all()
    assert len(audit_rows) == 1
    assert audit_rows[0].actor == "policy"
    assert audit_rows[0].action in ("policy_decision_made", "policy_no_action")
    db.close()


def test_decide_for_failure_event_is_idempotent_on_second_call(test_db_session):
    db = test_db_session()
    first, first_created = decide_for_failure_event(
        db, event_id=102, subscription_id="sub_db_2", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", base_probability=0.6,
    )
    second, second_created = decide_for_failure_event(
        db, event_id=102, subscription_id="sub_db_2", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", base_probability=0.6,
    )
    assert first_created is True
    assert second_created is False
    assert first.id == second.id

    assert db.query(PolicyDecision).filter(PolicyDecision.event_id == 102).count() == 1
    audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 102).order_by(AuditLog.id).all()
    assert len(audit_rows) == 2
    assert audit_rows[1].action == "policy_decision_skipped_duplicate"
    db.close()


def test_decide_for_failure_event_no_action_still_writes_audit_row(test_db_session):
    db = test_db_session()
    row, created = decide_for_failure_event(
        db, event_id=103, subscription_id="sub_db_3", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="hard_decline", base_probability=0.6,
    )
    assert created is True
    assert row.selected_candidate_type == NO_ACTION

    audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 103).all()
    assert len(audit_rows) == 1
    assert audit_rows[0].action == "policy_no_action"
    db.close()


def test_decide_for_failure_event_enforces_max_retry_attempts_across_calls(test_db_session):
    db = test_db_session()
    for i in range(MAX_RETRY_ATTEMPTS):
        decide_for_failure_event(
            db, event_id=200 + i, subscription_id="sub_db_maxattempts",
            failure_timestamp=FAILURE_TS + timedelta(days=i * 20),
            amount=1000.0, classification_bucket="retryable_soft", base_probability=0.6,
        )

    blocked_row, created = decide_for_failure_event(
        db, event_id=300, subscription_id="sub_db_maxattempts",
        failure_timestamp=FAILURE_TS + timedelta(days=100),
        amount=1000.0, classification_bucket="retryable_soft", base_probability=0.6,
    )
    assert created is True
    assert blocked_row.selected_candidate_type == NO_ACTION
    assert "max_retry_attempts" in blocked_row.decision_reason
    db.close()
