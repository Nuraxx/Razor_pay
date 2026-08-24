"""
Day-8 latent-target construction + regression model tests.

Uses a small synthetic dataset generated fresh (not the committed
data/raw/*.csv), same pattern as tests/test_candidate_model.py /
tests/test_ranking_model.py.
"""
import numpy as np
import pandas as pd
import pytest

from data.generate_counterfactual_dataset import COUNTERFACTUAL_SEED_OFFSET, generate_counterfactual_outcomes
from data.generate_synthetic_dataset import generate_dataset
from model.candidate_preprocessing import build_candidate_level_dataset_from_tables, split_candidate_dataset
from model.latent_target_preprocessing import (
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    LATENT_PROBABILITY_COLUMN,
    LATENT_RATE_COLUMN,
    LATENT_VALUE_COLUMN,
    add_latent_targets,
    select_features_and_target,
    validate_latent_targets,
)
from model.train_latent_target_model import fit_pipeline_for_target, load_latent_target_model, prepare_for_catboost, save_artifacts

TEST_SEED = 42
TEST_N = 120


@pytest.fixture(scope="module")
def base_dataset() -> dict[str, pd.DataFrame]:
    return generate_dataset(seed=TEST_SEED, n_subscriptions=TEST_N)


@pytest.fixture(scope="module")
def latent_dataset(base_dataset) -> pd.DataFrame:
    rng = np.random.default_rng(TEST_SEED + COUNTERFACTUAL_SEED_OFFSET)
    counterfactual = generate_counterfactual_outcomes(rng, base_dataset["subscriptions"], base_dataset["failure_events"], base_dataset["retry_candidates"])
    joined = build_candidate_level_dataset_from_tables(counterfactual, base_dataset["retry_candidates"], base_dataset["failure_events"], base_dataset["subscriptions"])
    return add_latent_targets(joined)


@pytest.fixture(scope="module")
def latent_splits(latent_dataset):
    return split_candidate_dataset(latent_dataset)


# ---------------------------------------------------------------------------
# Target construction (A/B/C concepts)
# ---------------------------------------------------------------------------

def test_target_construction_matches_formula(latent_dataset):
    expected = latent_dataset[LATENT_PROBABILITY_COLUMN] * latent_dataset["amount"]
    pd.testing.assert_series_equal(latent_dataset[LATENT_VALUE_COLUMN], expected, check_names=False)


def test_rate_column_is_exact_alias_of_probability_column(latent_dataset):
    pd.testing.assert_series_equal(latent_dataset[LATENT_RATE_COLUMN], latent_dataset[LATENT_PROBABILITY_COLUMN], check_names=False)


def test_validate_latent_targets_reports_no_issues(latent_dataset):
    assert validate_latent_targets(latent_dataset) == []


def test_target_bounds_probability_in_unit_interval(latent_dataset):
    assert latent_dataset[LATENT_PROBABILITY_COLUMN].between(0.0, 1.0).all()


def test_target_bounds_value_never_negative(latent_dataset):
    assert (latent_dataset[LATENT_VALUE_COLUMN] >= 0).all()


def test_target_bounds_value_never_exceeds_amount(latent_dataset):
    assert (latent_dataset[LATENT_VALUE_COLUMN] <= latent_dataset["amount"] + 1e-6).all()


def test_validate_latent_targets_catches_a_corrupted_value_column(latent_dataset):
    corrupted = latent_dataset.copy()
    corrupted.loc[corrupted.index[0], LATENT_VALUE_COLUMN] = corrupted["amount"].iloc[0] * 2  # break the formula
    issues = validate_latent_targets(corrupted)
    assert any("expected_recovery_value_latent" in issue for issue in issues)


def test_validate_latent_targets_catches_out_of_range_probability(latent_dataset):
    corrupted = latent_dataset.copy()
    corrupted.loc[corrupted.index[0], LATENT_PROBABILITY_COLUMN] = 1.5
    issues = validate_latent_targets(corrupted)
    assert any("recovery_probability_latent" in issue for issue in issues)


# ---------------------------------------------------------------------------
# No leakage
# ---------------------------------------------------------------------------

def test_feature_columns_exclude_all_latent_and_outcome_columns():
    leaky = {LATENT_PROBABILITY_COLUMN, LATENT_RATE_COLUMN, LATENT_VALUE_COLUMN, "recovered_within_14d", "recovered_at", "recovered_via", "amount_recovered", "archetype"}
    assert leaky.isdisjoint(set(FEATURE_COLUMNS))
    for col in leaky:
        assert col in EXCLUDED_COLUMNS, f"{col} is neither a feature nor documented as excluded"


def test_select_features_and_target_never_leaks_the_other_target(latent_splits):
    train_df, _val, _test = latent_splits
    Xa, ya = select_features_and_target(train_df, "probability")
    Xb, yb = select_features_and_target(train_df, "value")
    assert LATENT_VALUE_COLUMN not in Xa.columns and LATENT_PROBABILITY_COLUMN not in Xa.columns
    assert LATENT_VALUE_COLUMN not in Xb.columns and LATENT_PROBABILITY_COLUMN not in Xb.columns
    assert ya.name == LATENT_PROBABILITY_COLUMN
    assert yb.name == LATENT_VALUE_COLUMN


def test_select_features_and_target_rejects_unknown_target(latent_splits):
    train_df, _val, _test = latent_splits
    with pytest.raises(ValueError):
        select_features_and_target(train_df, "not_a_real_target")


# ---------------------------------------------------------------------------
# Split isolation / candidate groups
# ---------------------------------------------------------------------------

def test_candidate_groups_remain_exactly_five(latent_dataset):
    counts = latent_dataset.groupby("event_id").size()
    assert (counts == 5).all()


def test_no_event_crosses_splits(latent_dataset):
    by_sub = latent_dataset.groupby("subscription_id")["split"].nunique()
    assert (by_sub == 1).all()


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", ["probability", "value"])
def test_regression_model_trains(latent_splits, target):
    train_df, val_df, _test = latent_splits
    fitted = fit_pipeline_for_target(train_df, val_df, target)
    assert fitted["catboost_model"] is not None
    assert fitted["model_config"]["train_rows"] > 0


@pytest.mark.parametrize("target", ["probability", "value"])
def test_predictions_are_finite_and_non_negative(latent_splits, target):
    train_df, val_df, test_df = latent_splits
    fitted = fit_pipeline_for_target(train_df, val_df, target)
    X, _y = select_features_and_target(test_df, target)
    X_imp = fitted["imputer"].transform(X)
    X_cb = prepare_for_catboost(X_imp)
    preds = fitted["catboost_model"].predict(X_cb)
    assert np.isfinite(preds).all()
    assert (preds >= -1e-6).all()  # regressor isn't clipped at train time; policy layer clips downstream


@pytest.mark.parametrize("target", ["probability", "value"])
def test_deterministic_training(latent_splits, target):
    train_df, val_df, test_df = latent_splits
    fitted_a = fit_pipeline_for_target(train_df, val_df, target)
    fitted_b = fit_pipeline_for_target(train_df, val_df, target)

    X, _y = select_features_and_target(test_df, target)
    X_imp_a = prepare_for_catboost(fitted_a["imputer"].transform(X))
    X_imp_b = prepare_for_catboost(fitted_b["imputer"].transform(X))
    preds_a = fitted_a["catboost_model"].predict(X_imp_a)
    preds_b = fitted_b["catboost_model"].predict(X_imp_b)
    np.testing.assert_array_almost_equal(preds_a, preds_b)


def test_save_and_load_round_trips_predictions(latent_splits, tmp_path, monkeypatch):
    train_df, val_df, test_df = latent_splits
    fitted = fit_pipeline_for_target(train_df, val_df, "value")

    import model.train_latent_target_model as train_module

    monkeypatch.setattr(train_module, "ARTIFACTS_DIR", tmp_path)
    save_artifacts(fitted, "value")
    loaded = load_latent_target_model("value")

    X, _y = select_features_and_target(test_df, "value")
    X_imp_orig = prepare_for_catboost(fitted["imputer"].transform(X))
    X_imp_loaded = prepare_for_catboost(loaded["imputer"].transform(X))
    np.testing.assert_array_almost_equal(fitted["catboost_model"].predict(X_imp_orig), loaded["catboost_model"].predict(X_imp_loaded))
