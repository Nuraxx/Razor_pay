"""
Policy integration tests: the ranking model's scores flow into
the SAME `policy/recovery_policy.py::decide_candidate_aware` /
`decide_for_failure_event_candidate_aware` the candidate-aware model already
built and tested (tests/test_counterfactual_policy.py) -- no new policy code
exists for the ranking model by design. These tests confirm that composition
actually works end-to-end with REAL ranking-model scores, and that every
guardrail still applies.
"""
import numpy as np
import pytest

from app.models import AuditLog, PolicyDecision
from classification.rules import classify
from data.generate_counterfactual_dataset import COUNTERFACTUAL_SEED_OFFSET, generate_counterfactual_outcomes
from data.generate_synthetic_dataset import generate_dataset
from model.candidate_preprocessing import build_candidate_level_dataset_from_tables, split_candidate_dataset
from model.train_ranking_model import fit_pipeline, predict_ranking_scores
from policy.guardrails import MAX_RETRY_ATTEMPTS
from policy.recovery_policy import (
    NO_ACTION,
    POLICY_VERSION_CANDIDATE_AWARE,
    decide_candidate_aware,
    decide_for_failure_event_candidate_aware,
)
from policy.retry_candidates import CANDIDATE_TYPES

TEST_SEED = 42
TEST_N = 120


@pytest.fixture(scope="module")
def base_dataset():
    return generate_dataset(seed=TEST_SEED, n_subscriptions=TEST_N)


@pytest.fixture(scope="module")
def candidate_dataset(base_dataset):
    rng = np.random.default_rng(TEST_SEED + COUNTERFACTUAL_SEED_OFFSET)
    counterfactual = generate_counterfactual_outcomes(rng, base_dataset["subscriptions"], base_dataset["failure_events"], base_dataset["retry_candidates"])
    return build_candidate_level_dataset_from_tables(counterfactual, base_dataset["retry_candidates"], base_dataset["failure_events"], base_dataset["subscriptions"])


@pytest.fixture(scope="module")
def candidate_splits(candidate_dataset):
    return split_candidate_dataset(candidate_dataset)


@pytest.fixture(scope="module")
def ranking_fitted(candidate_splits):
    train_df, val_df, _test_df = candidate_splits
    return fit_pipeline(train_df, val_df)


@pytest.fixture(scope="module")
def one_event_scores(candidate_splits, ranking_fitted):
    """Real ranking-model scores for one held-out test event -- not a
    hand-crafted toy dict, so this exercises the true score distribution
    (which can be close together, unlike the candidate-aware model's tests'
    deliberately separated toy probabilities)."""
    _train, _val, test_df = candidate_splits
    scores = predict_ranking_scores(test_df, ranking_fitted)
    test_df = test_df.assign(_score=scores)
    event_id = test_df["event_id"].iloc[0]
    group = test_df[test_df["event_id"] == event_id]
    first = group.iloc[0]
    probs = dict(zip(group["candidate_type"], group["_score"]))
    return {
        "event_id": event_id,
        "subscription_id": first["subscription_id"],
        "failure_timestamp": first["failure_timestamp"],
        "amount": float(first["amount"]),
        "classification_bucket": classify(None, first["error_reason"]).bucket,
        "probs": probs,
    }


# ---------------------------------------------------------------------------
# Policy integration with real ranking-model scores
# ---------------------------------------------------------------------------

def test_ranking_scores_produce_exactly_one_selection_per_event(one_event_scores):
    e = one_event_scores
    result = decide_candidate_aware(e["event_id"], e["subscription_id"], e["failure_timestamp"], e["amount"], e["classification_bucket"], e["probs"])
    assert result.selected_candidate_type in set(CANDIDATE_TYPES) | {NO_ACTION}


def test_ranking_scores_selection_is_a_real_candidate_when_retryable_soft(one_event_scores):
    e = one_event_scores
    result = decide_candidate_aware(e["event_id"], e["subscription_id"], e["failure_timestamp"], e["amount"], "retryable_soft", e["probs"])
    assert result.selected_candidate_type != NO_ACTION  # every candidate is generated fresh and within-horizon for a typical failure timestamp


def test_ranking_scores_policy_version_is_candidate_aware(one_event_scores):
    e = one_event_scores
    result = decide_candidate_aware(e["event_id"], e["subscription_id"], e["failure_timestamp"], e["amount"], "retryable_soft", e["probs"])
    assert result.policy_version == POLICY_VERSION_CANDIDATE_AWARE


def test_ranking_scores_decision_is_deterministic(one_event_scores):
    e = one_event_scores
    kwargs = dict(event_id=e["event_id"], subscription_id=e["subscription_id"], failure_timestamp=e["failure_timestamp"], amount=e["amount"], classification_bucket="retryable_soft", candidate_probabilities=e["probs"])
    assert decide_candidate_aware(**kwargs) == decide_candidate_aware(**kwargs)


@pytest.mark.parametrize("bucket", ["hard_decline", "customer_cancelled", "unmapped"])
def test_ranking_scores_still_blocked_by_classification_guardrail(one_event_scores, bucket):
    e = one_event_scores
    result = decide_candidate_aware(e["event_id"], e["subscription_id"], e["failure_timestamp"], e["amount"], bucket, e["probs"])
    assert result.selected_candidate_type == NO_ACTION


def test_ranking_scores_blocked_after_max_retry_attempts(one_event_scores):
    e = one_event_scores
    result = decide_candidate_aware(
        e["event_id"], e["subscription_id"], e["failure_timestamp"], e["amount"], "retryable_soft", e["probs"], attempts_so_far=MAX_RETRY_ATTEMPTS
    )
    assert result.selected_candidate_type == NO_ACTION
    assert "max_retry_attempts" in result.decision_reason


# ---------------------------------------------------------------------------
# Idempotency + audit logging with real ranking-model scores
# ---------------------------------------------------------------------------

def test_ranking_scores_decision_creates_audit_and_decision_rows(one_event_scores, test_db_session):
    e = one_event_scores
    db = test_db_session()
    row, created = decide_for_failure_event_candidate_aware(
        db, event_id=90001, subscription_id="sub_RankingAuditTest", failure_timestamp=e["failure_timestamp"],
        amount=e["amount"], classification_bucket="retryable_soft", candidate_probabilities=e["probs"],
    )
    assert created is True
    assert isinstance(row, PolicyDecision)
    assert row.policy_version == POLICY_VERSION_CANDIDATE_AWARE

    audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 90001).all()
    assert len(audit_rows) == 1
    assert audit_rows[0].actor == "policy"
    db.close()


def test_ranking_scores_decision_is_idempotent(one_event_scores, test_db_session):
    e = one_event_scores
    db = test_db_session()
    first, first_created = decide_for_failure_event_candidate_aware(
        db, event_id=90002, subscription_id="sub_RankingIdempotent", failure_timestamp=e["failure_timestamp"],
        amount=e["amount"], classification_bucket="retryable_soft", candidate_probabilities=e["probs"],
    )
    second, second_created = decide_for_failure_event_candidate_aware(
        db, event_id=90002, subscription_id="sub_RankingIdempotent", failure_timestamp=e["failure_timestamp"],
        amount=e["amount"], classification_bucket="retryable_soft", candidate_probabilities=e["probs"],
    )
    assert first_created is True
    assert second_created is False
    assert first.id == second.id
    assert db.query(PolicyDecision).filter(PolicyDecision.event_id == 90002).count() == 1
    db.close()
