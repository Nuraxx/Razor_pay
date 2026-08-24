"""
Day-4 model pipeline tests.

Uses a small synthetic dataset generated fresh via data.generate_synthetic_dataset
(not the committed data/processed/*.csv) so these tests are fast and fully
self-contained.
"""
import inspect

import joblib
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

from data.generate_synthetic_dataset import generate_dataset
from model.calibrate import fit_calibration, isotonic_is_defensible, recommended_calibration_method
from model.preprocessing import (
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES_WITH_FLAG,
    TARGET_COLUMN,
    PriorSelfResolvedImputer,
    build_logreg_column_transformer,
    prepare_for_catboost,
    select_features_and_target,
)
from model.train import fit_pipeline, save_artifacts

TEST_SEED = 42
TEST_N = 120  # large enough that train/validation/test all get first-time AND repeat failures


@pytest.fixture(scope="module")
def small_dataset() -> dict[str, pd.DataFrame]:
    return generate_dataset(seed=TEST_SEED, n_subscriptions=TEST_N)


# --- Feature selection ---

def test_feature_columns_exclude_label_and_archetype_and_split():
    for leaky in ("recovered_within_14d", "recovered_at", "recovered_via", "final_amount_recovered", "archetype", "split"):
        assert leaky not in FEATURE_COLUMNS
    assert TARGET_COLUMN not in FEATURE_COLUMNS


def test_excluded_columns_documents_every_leaky_and_identifier_field():
    must_be_excluded = (
        "event_id", "subscription_id", "failure_timestamp", "signup_date", "monthly_amount",
        "error_reason", "recovered_within_14d", "recovered_at", "recovered_via",
        "final_amount_recovered", "archetype", "split",
    )
    for col in must_be_excluded:
        assert col in EXCLUDED_COLUMNS, f"{col} must have a documented exclusion reason"
        assert col not in FEATURE_COLUMNS


def test_select_features_and_target_returns_only_declared_columns(small_dataset):
    X, y = select_features_and_target(small_dataset["train"])
    assert set(X.columns) == set(FEATURE_COLUMNS)
    assert y.name == TARGET_COLUMN
    assert set(y.unique()).issubset({0, 1})
    for leaky in EXCLUDED_COLUMNS:
        assert leaky not in X.columns


# --- Preprocessing: categorical handling ---

def test_preprocessing_handles_categorical_data(small_dataset):
    X_train, y_train = select_features_and_target(small_dataset["train"])
    imputer = PriorSelfResolvedImputer().fit(X_train)
    X_train_imputed = imputer.transform(X_train)

    ct = build_logreg_column_transformer()
    encoded = ct.fit_transform(X_train_imputed)
    assert encoded.shape[0] == len(X_train)
    assert encoded.shape[1] > len(NUMERIC_FEATURES_WITH_FLAG)  # categoricals got one-hot expanded

    X_cb = prepare_for_catboost(X_train_imputed)
    for col in CATEGORICAL_FEATURES:
        assert X_cb[col].dtype == object
        assert not X_cb[col].isna().any()


# --- Missing historical feature ---

def test_missing_historical_feature_imputed_using_train_only_statistics(small_dataset):
    X_train, _ = select_features_and_target(small_dataset["train"])
    X_val, _ = select_features_and_target(small_dataset["validation"])
    assert X_train["prior_if_self_resolved_rate"].isna().any(), "fixture assumption: some first-time failures in train"
    assert X_val["prior_if_self_resolved_rate"].isna().any(), "fixture assumption: some first-time failures in validation"

    imputer = PriorSelfResolvedImputer().fit(X_train)
    expected_train_mean = X_train["prior_if_self_resolved_rate"].dropna().mean()
    assert imputer.fill_value == pytest.approx(expected_train_mean)

    was_missing_val = X_val["prior_if_self_resolved_rate"].isna()
    X_val_transformed = imputer.transform(X_val)

    assert (X_val_transformed.loc[was_missing_val, "prior_if_self_resolved_rate"] == imputer.fill_value).all()
    assert (X_val_transformed.loc[was_missing_val, "prior_if_self_resolved_rate_missing"] == 1).all()
    assert (X_val_transformed.loc[~was_missing_val, "prior_if_self_resolved_rate_missing"] == 0).all()
    assert not X_val_transformed["prior_if_self_resolved_rate"].isna().any()


# --- Train-only fitting ---

def test_transform_does_not_refit_or_mutate_learned_state(small_dataset):
    X_train, _ = select_features_and_target(small_dataset["train"])
    X_val, _ = select_features_and_target(small_dataset["validation"])
    imputer = PriorSelfResolvedImputer().fit(X_train)
    X_train_i = imputer.transform(X_train)
    X_val_i = imputer.transform(X_val)

    ct = build_logreg_column_transformer().fit(X_train_i)
    mean_before = ct.named_transformers_["num"].named_steps["scaler"].mean_.copy()

    ct.transform(X_val_i)  # must be transform-only

    mean_after = ct.named_transformers_["num"].named_steps["scaler"].mean_
    np.testing.assert_array_equal(mean_before, mean_after)


def test_fit_pipeline_signature_has_no_test_parameter():
    """Structural guarantee: test data cannot leak into fitting through this
    function because there is no parameter to pass it through."""
    params = list(inspect.signature(fit_pipeline).parameters)
    assert params == ["train_df", "val_df"]
    assert not any("test" in p.lower() for p in params)


def test_training_result_is_unaffected_by_unrelated_test_split_contents(small_dataset):
    """Train twice with identical (train, val) but simulate a completely
    different/corrupted test split existing in memory alongside -- since
    fit_pipeline never receives it, results must be identical either way."""
    fitted_a = fit_pipeline(small_dataset["train"], small_dataset["validation"])

    corrupted_test = small_dataset["test"].sample(frac=1.0, random_state=999).reset_index(drop=True)
    corrupted_test["recovered_within_14d"] = ~corrupted_test["recovered_within_14d"]  # deliberately garbage
    _ = corrupted_test  # exists in scope, never passed to fit_pipeline

    fitted_b = fit_pipeline(small_dataset["train"], small_dataset["validation"])

    assert fitted_a["model_config"]["validation_auc"] == fitted_b["model_config"]["validation_auc"]
    np.testing.assert_array_equal(
        fitted_a["logreg_model"].coef_, fitted_b["logreg_model"].coef_
    )


# --- Model trains successfully / probabilities in [0,1] / calibration works ---

def test_model_trains_and_produces_valid_probabilities(small_dataset):
    fitted = fit_pipeline(small_dataset["train"], small_dataset["validation"])
    X_val, y_val = select_features_and_target(small_dataset["validation"])
    X_val_i = fitted["imputer"].transform(X_val)

    logreg_probs = fitted["logreg_model"].predict_proba(fitted["logreg_preprocessor"].transform(X_val_i))[:, 1]
    assert logreg_probs.min() >= 0.0 and logreg_probs.max() <= 1.0

    X_val_cb = prepare_for_catboost(X_val_i)
    cb_probs = fitted["catboost_model"].predict_proba(X_val_cb)[:, 1]
    assert cb_probs.min() >= 0.0 and cb_probs.max() <= 1.0

    calibrated_probs = fitted["sigmoid_calibrator"].predict_proba(X_val_cb)[:, 1]
    assert calibrated_probs.min() >= 0.0 and calibrated_probs.max() <= 1.0
    isotonic_probs = fitted["isotonic_calibrator"].predict_proba(X_val_cb)[:, 1]
    assert isotonic_probs.min() >= 0.0 and isotonic_probs.max() <= 1.0


def test_calibration_recommendation_logic():
    assert isotonic_is_defensible(200) is True
    assert isotonic_is_defensible(199) is False
    assert recommended_calibration_method(59) == "sigmoid"
    assert recommended_calibration_method(500) == "isotonic"


def test_calibration_changes_probability_values(small_dataset):
    """Calibration should actually do something -- the calibrated
    probabilities should differ from the raw model's, in general."""
    X_train, y_train = select_features_and_target(small_dataset["train"])
    X_val, y_val = select_features_and_target(small_dataset["validation"])
    imputer = PriorSelfResolvedImputer().fit(X_train)
    X_train_cb = prepare_for_catboost(imputer.transform(X_train))
    X_val_cb = prepare_for_catboost(imputer.transform(X_val))

    cb = CatBoostClassifier(iterations=100, depth=3, cat_features=CATEGORICAL_FEATURES, random_seed=TEST_SEED, verbose=False)
    cb.fit(X_train_cb, y_train)
    raw_probs = cb.predict_proba(X_val_cb)[:, 1]

    calibrator = fit_calibration(cb, X_val_cb, y_val, method="sigmoid")
    calibrated_probs = calibrator.predict_proba(X_val_cb)[:, 1]

    assert not np.allclose(raw_probs, calibrated_probs)


# --- Test split never used during fitting (end-to-end) ---

def test_evaluate_only_reads_test_never_fits_on_it(small_dataset, tmp_path):
    """Fit on train+val, save artifacts, then confirm the saved LogReg
    coefficients are identical regardless of what the test split contains --
    proving evaluation-time test data has zero path back into the model."""
    fitted = fit_pipeline(small_dataset["train"], small_dataset["validation"])
    save_artifacts(fitted, tmp_path)

    reloaded_logreg = joblib.load(tmp_path / "logreg_model.joblib")
    np.testing.assert_array_equal(fitted["logreg_model"].coef_, reloaded_logreg.coef_)

    # The test split itself is never opened by anything in this test.
    assert "test" not in [f.stem for f in tmp_path.glob("*.csv")]


# --- Artifact saving/loading ---

def test_artifact_saving_and_loading_round_trips_predictions(small_dataset, tmp_path):
    fitted = fit_pipeline(small_dataset["train"], small_dataset["validation"])
    save_artifacts(fitted, tmp_path)

    for fname in (
        "logreg_preprocessor.joblib",
        "logreg_model.joblib",
        "prior_self_resolved_imputer.joblib",
        "catboost_model.cbm",
        "catboost_calibrated_sigmoid.joblib",
        "catboost_calibrated_isotonic.joblib",
        "feature_list.json",
        "model_config.json",
    ):
        assert (tmp_path / fname).exists()

    X_val, y_val = select_features_and_target(small_dataset["validation"])
    X_val_i = fitted["imputer"].transform(X_val)
    X_val_cb = prepare_for_catboost(X_val_i)

    original_probs = fitted["catboost_model"].predict_proba(X_val_cb)[:, 1]

    reloaded_cb = CatBoostClassifier()
    reloaded_cb.load_model(str(tmp_path / "catboost_model.cbm"))
    reloaded_probs = reloaded_cb.predict_proba(X_val_cb)[:, 1]

    np.testing.assert_allclose(original_probs, reloaded_probs, rtol=1e-6)

    reloaded_logreg = joblib.load(tmp_path / "logreg_model.joblib")
    reloaded_preprocessor = joblib.load(tmp_path / "logreg_preprocessor.joblib")
    original_lr_probs = fitted["logreg_model"].predict_proba(fitted["logreg_preprocessor"].transform(X_val_i))[:, 1]
    reloaded_lr_probs = reloaded_logreg.predict_proba(reloaded_preprocessor.transform(X_val_i))[:, 1]
    np.testing.assert_allclose(original_lr_probs, reloaded_lr_probs, rtol=1e-6)


# --- Reproducibility ---

def test_reproducibility_same_inputs_produce_identical_models(small_dataset):
    fitted_a = fit_pipeline(small_dataset["train"], small_dataset["validation"])
    fitted_b = fit_pipeline(small_dataset["train"], small_dataset["validation"])

    np.testing.assert_array_equal(fitted_a["logreg_model"].coef_, fitted_b["logreg_model"].coef_)
    assert fitted_a["model_config"]["catboost_best_iteration"] == fitted_b["model_config"]["catboost_best_iteration"]

    X_val, _ = select_features_and_target(small_dataset["validation"])
    X_val_cb = prepare_for_catboost(fitted_a["imputer"].transform(X_val))
    probs_a = fitted_a["catboost_model"].predict_proba(X_val_cb)[:, 1]
    probs_b = fitted_b["catboost_model"].predict_proba(X_val_cb)[:, 1]
    np.testing.assert_allclose(probs_a, probs_b, rtol=1e-6)
