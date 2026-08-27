"""
Latent-target regression training.

    ./venv/bin/python model/train_latent_target_model.py

Trains TWO independent regressors, controlled experiment (brief section 6):

    Model A: target = recovery_probability_latent   (predicts a probability)
    Model B: target = expected_recovery_value_latent (predicts rupees -- the PRIMARY target, brief section 2)

Each gets a CatBoostRegressor (main model) and a plain LinearRegression
baseline, fit on TRAIN, selected/early-stopped on VALIDATION only -- the
test split is never touched here (evaluation/evaluate_latent_target_policy.py
is the only script that reads it, matching every prior day's discipline).

NOT a classifier: these targets are continuous (a probability in [0,1] for
Model A, rupees for Model B) and are fit with regression loss (RMSE), not
log loss. Per the brief section 9, NO classifier-style sigmoid/isotonic
calibration is applied here -- that machinery is built for binary
classifiers and would be a category error on a regression target. If a
probability-flavored output is later needed for legacy policy plumbing
(see evaluation/evaluate_latent_target_policy.py), it is obtained by
simple, documented conversion (predicted_value / amount for Model B),
never by forcing CalibratedClassifierCV onto a regressor.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model.latent_target_preprocessing import (
    ALL_BOOLEAN_FEATURES,
    ALL_CATEGORICAL_FEATURES,
    ALL_NUMERIC_FEATURES,
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    PriorSelfResolvedImputer,
    PROJECT_ROOT,
    SEED,
    TARGET_COLUMNS,
    prepare_for_catboost,
    select_features_and_target,
)

ARTIFACTS_DIR = PROJECT_ROOT / "model" / "latent_target_artifacts"

CATBOOST_REGRESSOR_PARAMS = dict(
    iterations=500,
    depth=4,
    learning_rate=0.05,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=SEED,
    early_stopping_rounds=50,
    use_best_model=True,
    verbose=False,
)


def build_regression_column_transformer() -> ColumnTransformer:
    """LinearRegression baseline's encoder -- same shape as every prior
    day's *_column_transformer, fit on training data only by the caller."""
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
    numeric_with_flag = ALL_NUMERIC_FEATURES + ALL_BOOLEAN_FEATURES + ["prior_if_self_resolved_rate_missing"]
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_with_flag),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ALL_CATEGORICAL_FEATURES),
        ]
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_pipeline_for_target(train_df: pd.DataFrame, val_df: pd.DataFrame, target: str) -> dict:
    """target: 'probability' (Model A) or 'value' (Model B). Deterministic
    given the same (train_df, val_df, target)."""
    X_train_raw, y_train = select_features_and_target(train_df, target)
    X_val_raw, y_val = select_features_and_target(val_df, target)

    imputer = PriorSelfResolvedImputer().fit(X_train_raw)  # fit on train only
    X_train = imputer.transform(X_train_raw)
    X_val = imputer.transform(X_val_raw)
    # Boolean features are already cast to 0/1 int by select_features_and_target() above.

    transformer = build_regression_column_transformer()
    X_train_encoded = transformer.fit_transform(X_train)  # fit on train only
    X_val_encoded = transformer.transform(X_val)  # transform only, never fit

    linreg = LinearRegression()
    linreg.fit(X_train_encoded, y_train)
    linreg_val_pred = linreg.predict(X_val_encoded)
    linreg_val_metrics = regression_metrics(y_val.to_numpy(), linreg_val_pred)

    X_train_cb = prepare_for_catboost(X_train)
    X_val_cb = prepare_for_catboost(X_val)

    catboost_model = CatBoostRegressor(cat_features=ALL_CATEGORICAL_FEATURES, **CATBOOST_REGRESSOR_PARAMS)
    catboost_model.fit(X_train_cb, y_train, eval_set=(X_val_cb, y_val))
    cb_val_pred = catboost_model.predict(X_val_cb)
    cb_val_metrics = regression_metrics(y_val.to_numpy(), cb_val_pred)

    model_config = {
        "target": target,
        "target_column": TARGET_COLUMNS[target],
        "seed": SEED,
        "feature_columns": FEATURE_COLUMNS,
        "catboost_params": dict(CATBOOST_REGRESSOR_PARAMS),
        "catboost_best_iteration": catboost_model.get_best_iteration(),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "validation_metrics": {"catboost_regressor": cb_val_metrics, "linear_regression": linreg_val_metrics},
        "prior_self_resolved_rate_fill_value": imputer.fill_value,
    }

    return {
        "imputer": imputer,
        "transformer": transformer,
        "linreg_model": linreg,
        "catboost_model": catboost_model,
        "feature_list": {"feature_columns": FEATURE_COLUMNS, "excluded_columns": EXCLUDED_COLUMNS, "target_column": TARGET_COLUMNS[target]},
        "model_config": model_config,
    }


def save_artifacts(fitted: dict, target: str) -> None:
    target_dir = ARTIFACTS_DIR / target
    target_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted["imputer"], target_dir / "prior_self_resolved_imputer.joblib")
    joblib.dump(fitted["transformer"], target_dir / "column_transformer.joblib")
    joblib.dump(fitted["linreg_model"], target_dir / "linreg_model.joblib")
    fitted["catboost_model"].save_model(str(target_dir / "catboost_regressor.cbm"))
    with open(target_dir / "feature_list.json", "w") as f:
        json.dump(fitted["feature_list"], f, indent=2)
    with open(target_dir / "model_config.json", "w") as f:
        json.dump(fitted["model_config"], f, indent=2, default=str)


def load_latent_target_model(target: str) -> dict:
    target_dir = ARTIFACTS_DIR / target
    imputer = joblib.load(target_dir / "prior_self_resolved_imputer.joblib")
    catboost_model = CatBoostRegressor()
    catboost_model.load_model(str(target_dir / "catboost_regressor.cbm"))
    return {"imputer": imputer, "catboost_model": catboost_model}


def main() -> None:
    from model.candidate_preprocessing import split_candidate_dataset
    from model.latent_target_preprocessing import build_candidate_level_dataset_with_latent_targets, validate_latent_targets

    df = build_candidate_level_dataset_with_latent_targets()
    issues = validate_latent_targets(df)
    if issues:
        raise SystemExit(f"Latent target validation FAILED: {issues}")

    train_df, val_df, _test_df_unused = split_candidate_dataset(df)
    del _test_df_unused  # test is never used for anything training-related

    for target in ("probability", "value"):
        fitted = fit_pipeline_for_target(train_df, val_df, target)
        save_artifacts(fitted, target)
        cfg = fitted["model_config"]
        cb = cfg["validation_metrics"]["catboost_regressor"]
        lr = cfg["validation_metrics"]["linear_regression"]
        print(f"=== Model {'A (probability)' if target == 'probability' else 'B (value)'} -- target={cfg['target_column']} ===")
        print(f"  Train rows: {cfg['train_rows']} | Validation rows: {cfg['validation_rows']}")
        print(f"  CatBoostRegressor  (best_iteration={cfg['catboost_best_iteration']}): MAE={cb['mae']:.4f} RMSE={cb['rmse']:.4f} R2={cb['r2']:.4f}")
        print(f"  LinearRegression baseline:                     MAE={lr['mae']:.4f} RMSE={lr['rmse']:.4f} R2={lr['r2']:.4f}")
        print(f"  Artifacts written to {ARTIFACTS_DIR / target}")
        print()


if __name__ == "__main__":
    main()
