"""
Day-8 policy integration tests: Model A's (probability) and Model B's
(value, converted to a probability-equivalent via /amount) scores flow into
the SAME `policy/recovery_policy.py::decide_candidate_aware` /
`decide_for_failure_event_candidate_aware` Day 6 built and Day 7 already
reused -- no new policy code exists for Day 8 either. These tests confirm
that composition works end-to-end with REAL Day-8 model scores, and that
every Day-5 guardrail still applies.
"""
import numpy as np
import pytest

from app.models import AuditLog, PolicyDecision
from classification.rules import classify
from data.generate_counterfactual_dataset import COUNTERFACTUAL_SEED_OFFSET, generate_counterfactual_outcomes
from data.generate_synthetic_dataset import generate_dataset
from model.candidate_preprocessing import build_candidate_level_dataset_from_tables, split_candidate_dataset
from model.latent_target_preprocessing import add_latent_targets, prepare_for_catboost, select_features_and_target
from model.train_latent_target_model import fit_pipeline_for_target
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
def latent_splits(base_dataset):
    rng = np.random.default_rng(TEST_SEED + COUNTERFACTUAL_SEED_OFFSET)
    counterfactual = generate_counterfactual_outcomes(rng, base_dataset["subscriptions"], base_dataset["failure_events"], base_dataset["retry_candidates"])
    joined = build_candidate_level_dataset_from_tables(counterfactual, base_dataset["retry_candidates"], base_dataset["failure_events"], base_dataset["subscriptions"])
    df = add_latent_targets(joined)
    return split_candidate_dataset(df)


@pytest.fixture(scope="module")
def model_b_fitted(latent_splits):
    train_df, val_df, _test_df = latent_splits
    return fit_pipeline_for_target(train_df, val_df, "value")


@pytest.fixture(scope="module")
def one_event_probability_equivalent_scores(latent_splits, model_b_fitted):
    """Real Model-B predictions for one held-out test event, converted to a
    probability-equivalent (predicted_value / amount, clipped to [0,1]) --
    the same conversion evaluation/evaluate_latent_target_policy.py uses to
    plug Model B into the unmodified Day-6 policy architecture."""
    _train, _val, test_df = latent_splits
    X, _y = select_features_and_target(test_df, "value")
    X_imp = model_b_fitted["imputer"].transform(X)
    X_cb = prepare_for_catboost(X_imp)
    preds = model_b_fitted["catboost_model"].predict(X_cb)

    df = test_df.assign(_pred_value=preds)
    event_id = df["event_id"].iloc[0]
    group = df[df["event_id"] == event_id]
    first = group.iloc[0]
    prob_equivalent = (group["_pred_value"] / group["amount"]).clip(0.0, 1.0)
    probs = dict(zip(group["candidate_type"], prob_equivalent))
    return {
        "event_id": event_id,
        "subscription_id": first["subscription_id"],
        "failure_timestamp": first["failure_timestamp"],
        "amount": float(first["amount"]),
        "classification_bucket": classify(None, first["error_reason"]).bucket,
        "probs": probs,
    }


# ---------------------------------------------------------------------------
# Policy integration with real Model-B scores
# ---------------------------------------------------------------------------

def test_model_b_scores_produce_a_valid_selection(one_event_probability_equivalent_scores):
    e = one_event_probability_equivalent_scores
    result = decide_candidate_aware(e["event_id"], e["subscription_id"], e["failure_timestamp"], e["amount"], e["classification_bucket"], e["probs"])
    assert result.selected_candidate_type in set(CANDIDATE_TYPES) | {NO_ACTION}


def test_model_b_scores_probabilities_stay_within_unit_interval(one_event_probability_equivalent_scores):
    e = one_event_probability_equivalent_scores
    assert all(0.0 <= p <= 1.0 for p in e["probs"].values())


def test_model_b_scores_select_a_real_candidate_when_retryable_soft(one_event_probability_equivalent_scores):
    e = one_event_probability_equivalent_scores
    result = decide_candidate_aware(e["event_id"], e["subscription_id"], e["failure_timestamp"], e["amount"], "retryable_soft", e["probs"])
    assert result.selected_candidate_type != NO_ACTION


def test_model_b_scores_decision_is_deterministic(one_event_probability_equivalent_scores):
    e = one_event_probability_equivalent_scores
    kwargs = dict(event_id=e["event_id"], subscription_id=e["subscription_id"], failure_timestamp=e["failure_timestamp"], amount=e["amount"], classification_bucket="retryable_soft", candidate_probabilities=e["probs"])
    assert decide_candidate_aware(**kwargs) == decide_candidate_aware(**kwargs)


@pytest.mark.parametrize("bucket", ["hard_decline", "customer_cancelled", "unmapped"])
def test_model_b_scores_still_blocked_by_classification_guardrail(one_event_probability_equivalent_scores, bucket):
    e = one_event_probability_equivalent_scores
    result = decide_candidate_aware(e["event_id"], e["subscription_id"], e["failure_timestamp"], e["amount"], bucket, e["probs"])
    assert result.selected_candidate_type == NO_ACTION


def test_model_b_scores_blocked_after_max_retry_attempts(one_event_probability_equivalent_scores):
    e = one_event_probability_equivalent_scores
    result = decide_candidate_aware(
        e["event_id"], e["subscription_id"], e["failure_timestamp"], e["amount"], "retryable_soft", e["probs"], attempts_so_far=MAX_RETRY_ATTEMPTS
    )
    assert result.selected_candidate_type == NO_ACTION
    assert "max_retry_attempts" in result.decision_reason


# ---------------------------------------------------------------------------
# Idempotency + audit logging with real Model-B scores
# ---------------------------------------------------------------------------

def test_model_b_decision_creates_audit_and_decision_rows(one_event_probability_equivalent_scores, test_db_session):
    e = one_event_probability_equivalent_scores
    db = test_db_session()
    row, created = decide_for_failure_event_candidate_aware(
        db, event_id=95001, subscription_id="sub_LatentTargetAuditTest", failure_timestamp=e["failure_timestamp"],
        amount=e["amount"], classification_bucket="retryable_soft", candidate_probabilities=e["probs"],
    )
    assert created is True
    assert isinstance(row, PolicyDecision)
    assert row.policy_version == POLICY_VERSION_CANDIDATE_AWARE

    audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 95001).all()
    assert len(audit_rows) == 1
    assert audit_rows[0].actor == "policy"
    db.close()


def test_model_b_decision_is_idempotent(one_event_probability_equivalent_scores, test_db_session):
    e = one_event_probability_equivalent_scores
    db = test_db_session()
    first, first_created = decide_for_failure_event_candidate_aware(
        db, event_id=95002, subscription_id="sub_LatentTargetIdempotent", failure_timestamp=e["failure_timestamp"],
        amount=e["amount"], classification_bucket="retryable_soft", candidate_probabilities=e["probs"],
    )
    second, second_created = decide_for_failure_event_candidate_aware(
        db, event_id=95002, subscription_id="sub_LatentTargetIdempotent", failure_timestamp=e["failure_timestamp"],
        amount=e["amount"], classification_bucket="retryable_soft", candidate_probabilities=e["probs"],
    )
    assert first_created is True
    assert second_created is False
    assert first.id == second.id
    assert db.query(PolicyDecision).filter(PolicyDecision.event_id == 95002).count() == 1
    db.close()
