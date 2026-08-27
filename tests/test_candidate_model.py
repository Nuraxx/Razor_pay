"""
Candidate-aware model pipeline tests.

Uses a small synthetic dataset generated fresh (not the committed
data/raw/*.csv) so these tests are fast and self-contained -- same pattern
as tests/test_model_pipeline.py.
"""
import numpy as np
import pandas as pd
import pytest

from data.generate_counterfactual_dataset import COUNTERFACTUAL_SEED_OFFSET, generate_counterfactual_outcomes
from data.generate_synthetic_dataset import generate_dataset
from model.candidate_preprocessing import (
    ALL_BOOLEAN_FEATURES,
    ALL_CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_candidate_level_dataset_from_tables,
    prepare_for_catboost,
    select_features_and_target,
    split_candidate_dataset,
)
from model.train_candidate_model import fit_pipeline

TEST_SEED = 42
TEST_N = 120


@pytest.fixture(scope="module")
def base_dataset() -> dict[str, pd.DataFrame]:
    return generate_dataset(seed=TEST_SEED, n_subscriptions=TEST_N)


@pytest.fixture(scope="module")
def candidate_dataset(base_dataset) -> pd.DataFrame:
    rng = np.random.default_rng(TEST_SEED + COUNTERFACTUAL_SEED_OFFSET)
    counterfactual = generate_counterfactual_outcomes(rng, base_dataset["subscriptions"], base_dataset["failure_events"], base_dataset["retry_candidates"])
    return build_candidate_level_dataset_from_tables(counterfactual, base_dataset["retry_candidates"], base_dataset["failure_events"], base_dataset["subscriptions"])


@pytest.fixture(scope="module")
def candidate_splits(candidate_dataset) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return split_candidate_dataset(candidate_dataset)


@pytest.fixture(scope="module")
def fitted(candidate_splits) -> dict:
    train_df, val_df, _test_df = candidate_splits
    return fit_pipeline(train_df, val_df)


# ---------------------------------------------------------------------------
# No leakage
# ---------------------------------------------------------------------------

def test_hidden_archetype_excluded_from_candidate_dataset(candidate_dataset):
    assert "archetype" not in candidate_dataset.columns


def test_hidden_archetype_excluded_from_feature_columns():
    assert "archetype" not in FEATURE_COLUMNS


def test_feature_columns_exclude_label_and_post_treatment_and_latent_columns():
    leaky = {"recovered_within_14d", "recovered_at", "recovered_via", "amount_recovered", "recovery_probability_latent", "archetype"}
    assert leaky.isdisjoint(set(FEATURE_COLUMNS))


def test_excluded_columns_documents_every_leaky_and_identifier_field(candidate_dataset):
    accounted_for = set(FEATURE_COLUMNS) | set(EXCLUDED_COLUMNS.keys())
    missing = set(candidate_dataset.columns) - accounted_for - {"split"}
    assert missing == set(), f"columns present in the joined table but neither a feature nor documented as excluded: {missing}"


def test_no_split_leakage_across_subscriptions(candidate_dataset):
    by_sub = candidate_dataset.groupby("subscription_id")["split"].nunique()
    assert (by_sub == 1).all()


def test_train_validation_test_share_no_subscriptions(candidate_splits, candidate_dataset):
    train_df, val_df, test_df = candidate_splits
    # split_candidate_dataset drops the split column but subscription_id survives -- recover membership via it
    train_subs = set(train_df["subscription_id"])
    val_subs = set(val_df["subscription_id"])
    test_subs = set(test_df["subscription_id"])
    assert not (train_subs & val_subs)
    assert not (train_subs & test_subs)
    assert not (val_subs & test_subs)


def test_select_features_and_target_returns_only_declared_columns(candidate_splits):
    train_df, _val, _test = candidate_splits
    X, y = select_features_and_target(train_df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert y.name == TARGET_COLUMN
    for col in ALL_BOOLEAN_FEATURES:
        assert X[col].dtype.kind in "iu"  # cast to int, not bool


# ---------------------------------------------------------------------------
# Candidate structure
# ---------------------------------------------------------------------------

def test_five_candidate_rows_per_event(candidate_dataset):
    counts = candidate_dataset.groupby("event_id").size()
    assert (counts == 5).all()


def test_prepare_for_catboost_casts_categoricals_to_object(candidate_splits):
    train_df, _val, _test = candidate_splits
    X, _y = select_features_and_target(train_df)
    X_cb = prepare_for_catboost(X)
    for col in ALL_CATEGORICAL_FEATURES:
        assert X_cb[col].dtype == object


# ---------------------------------------------------------------------------
# Model trains and produces valid probabilities
# ---------------------------------------------------------------------------

def test_candidate_aware_model_trains(fitted):
    assert fitted["catboost_model"] is not None
    assert fitted["logreg_model"] is not None
    assert fitted["model_config"]["train_rows"] > 0


def test_model_probabilities_valid_on_validation(fitted, candidate_splits):
    _train, val_df, _test = candidate_splits
    X_val_raw, _y = select_features_and_target(val_df)
    X_val = fitted["imputer"].transform(X_val_raw)
    X_val_cb = prepare_for_catboost(X_val)
    probs = fitted["sigmoid_calibrator"].predict_proba(X_val_cb)[:, 1]
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert not np.isnan(probs).any()


def test_reproducibility_same_inputs_produce_identical_predictions(candidate_splits):
    train_df, val_df, _test = candidate_splits
    fitted_a = fit_pipeline(train_df, val_df)
    fitted_b = fit_pipeline(train_df, val_df)

    X_val_raw, _y = select_features_and_target(val_df)
    X_val_a = prepare_for_catboost(fitted_a["imputer"].transform(X_val_raw))
    X_val_b = prepare_for_catboost(fitted_b["imputer"].transform(X_val_raw))

    preds_a = fitted_a["catboost_model"].predict_proba(X_val_a)[:, 1]
    preds_b = fitted_b["catboost_model"].predict_proba(X_val_b)[:, 1]
    np.testing.assert_array_almost_equal(preds_a, preds_b)


def test_fit_pipeline_signature_has_no_test_parameter():
    import inspect

    params = list(inspect.signature(fit_pipeline).parameters)
    assert params == ["train_df", "val_df"]
