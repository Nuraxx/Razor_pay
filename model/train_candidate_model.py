"""
Candidate-aware training entry point.

    ./venv/bin/python model/train_candidate_model.py

Structurally identical to model/train.py (same imputer -> LogReg -> CatBoost
-> calibration pipeline, same train/validation separation discipline), but
trained on the candidate-level table (model/candidate_preprocessing.py) so
the resulting model answers a genuinely different question:

    Calibrated model:      P(recovered_within_14d | failure/customer context)
    Candidate-aware model: P(recovered_within_14d | failure/customer context, candidate retry action)

`fit_pipeline(train_df, val_df)` takes no test data -- test is loaded and
immediately discarded in main(), read only by
evaluation/evaluate_counterfactual_policy.py.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from model.calibrate import fit_calibration, isotonic_is_defensible, recommended_calibration_method
from model.candidate_preprocessing import (
    ALL_CATEGORICAL_FEATURES,
    ALL_NUMERIC_FEATURES_WITH_FLAG,
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    PriorSelfResolvedImputer,
    build_candidate_logreg_column_transformer,
    load_candidate_splits,
    prepare_for_catboost,
    select_features_and_target,
)
from model.preprocessing import SEED

ARTIFACTS_DIR = PROJECT_ROOT / "model" / "candidate_artifacts"

CATBOOST_PARAMS = dict(
    iterations=500,
    depth=4,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=SEED,
    early_stopping_rounds=50,
    use_best_model=True,
    verbose=False,
)

LOGREG_BASE_PARAMS = dict(
    max_iter=1000,
    random_state=SEED,
    solver="lbfgs",
)
LOGREG_C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


def fit_pipeline(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Fit everything on the candidate-level table. Deterministic given the
    same (train_df, val_df) -- see SEED usage throughout."""
    X_train_raw, y_train = select_features_and_target(train_df)
    X_val_raw, y_val = select_features_and_target(val_df)

    imputer = PriorSelfResolvedImputer().fit(X_train_raw)  # fit on train only
    X_train = imputer.transform(X_train_raw)
    X_val = imputer.transform(X_val_raw)

    logreg_preprocessor = build_candidate_logreg_column_transformer()
    X_train_encoded = logreg_preprocessor.fit_transform(X_train)  # fit on train only
    X_val_encoded = logreg_preprocessor.transform(X_val)  # transform only, never fit

    logreg_selection = []
    for C in LOGREG_C_GRID:
        candidate = LogisticRegression(C=C, **LOGREG_BASE_PARAMS)
        candidate.fit(X_train_encoded, y_train)
        val_auc = roc_auc_score(y_val, candidate.predict_proba(X_val_encoded)[:, 1])
        logreg_selection.append({"C": C, "validation_auc": val_auc})

    best = max(logreg_selection, key=lambda row: row["validation_auc"])
    logreg_params = {"C": best["C"], **LOGREG_BASE_PARAMS}
    logreg = LogisticRegression(**logreg_params)
    logreg.fit(X_train_encoded, y_train)
    logreg_val_auc = best["validation_auc"]

    X_train_cb = prepare_for_catboost(X_train)
    X_val_cb = prepare_for_catboost(X_val)

    catboost_model = CatBoostClassifier(cat_features=ALL_CATEGORICAL_FEATURES, **CATBOOST_PARAMS)
    catboost_model.fit(X_train_cb, y_train, eval_set=(X_val_cb, y_val))
    catboost_val_auc = roc_auc_score(y_val, catboost_model.predict_proba(X_val_cb)[:, 1])

    sigmoid_calibrator = fit_calibration(catboost_model, X_val_cb, y_val, method="sigmoid")
    isotonic_calibrator = fit_calibration(catboost_model, X_val_cb, y_val, method="isotonic")
    chosen_method = recommended_calibration_method(len(X_val_cb))

    feature_list = {
        "numeric_features": ALL_NUMERIC_FEATURES_WITH_FLAG,
        "categorical_features": ALL_CATEGORICAL_FEATURES,
        "target_column": "recovered_within_14d",
        "excluded_columns": EXCLUDED_COLUMNS,
    }

    model_config = {
        "seed": SEED,
        "feature_columns_before_imputation_flag": FEATURE_COLUMNS,
        "logreg_params": logreg_params,
        "logreg_C_selection": logreg_selection,
        "catboost_params": dict(CATBOOST_PARAMS),
        "catboost_best_iteration": catboost_model.get_best_iteration(),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "train_recovery_rate": float(y_train.mean()),
        "prior_self_resolved_rate_fill_value": imputer.fill_value,
        "calibration": {
            "candidates_fit": ["sigmoid", "isotonic"],
            "isotonic_defensible_at_this_validation_size": isotonic_is_defensible(len(X_val_cb)),
            "chosen_method": chosen_method,
            "reason": (
                f"validation set has {len(X_val_cb)} rows; isotonic regression fits an unconstrained "
                f"step function and is only considered defensible at >= 200 rows in this project."
            ),
        },
        "validation_auc": {
            "logistic_regression": logreg_val_auc,
            "catboost_uncalibrated": catboost_val_auc,
        },
    }

    return {
        "imputer": imputer,
        "logreg_preprocessor": logreg_preprocessor,
        "logreg_model": logreg,
        "catboost_model": catboost_model,
        "sigmoid_calibrator": sigmoid_calibrator,
        "isotonic_calibrator": isotonic_calibrator,
        "feature_list": feature_list,
        "model_config": model_config,
    }


def save_artifacts(fitted: dict, artifacts_dir) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted["logreg_preprocessor"], artifacts_dir / "logreg_preprocessor.joblib")
    joblib.dump(fitted["logreg_model"], artifacts_dir / "logreg_model.joblib")
    joblib.dump(fitted["imputer"], artifacts_dir / "prior_self_resolved_imputer.joblib")
    fitted["catboost_model"].save_model(str(artifacts_dir / "catboost_model.cbm"))
    joblib.dump(fitted["sigmoid_calibrator"], artifacts_dir / "catboost_calibrated_sigmoid.joblib")
    joblib.dump(fitted["isotonic_calibrator"], artifacts_dir / "catboost_calibrated_isotonic.joblib")
    with open(artifacts_dir / "feature_list.json", "w") as f:
        json.dump(fitted["feature_list"], f, indent=2)
    with open(artifacts_dir / "model_config.json", "w") as f:
        json.dump(fitted["model_config"], f, indent=2)


def main() -> None:
    train_df, val_df, _test_df_unused = load_candidate_splits()
    del _test_df_unused  # test is never used for anything training-related

    fitted = fit_pipeline(train_df, val_df)
    save_artifacts(fitted, ARTIFACTS_DIR)

    cfg = fitted["model_config"]
    print(f"Train rows: {cfg['train_rows']} | Validation rows: {cfg['validation_rows']}")
    print(f"Logistic Regression (C={cfg['logreg_params']['C']}) -- validation ROC-AUC: {cfg['validation_auc']['logistic_regression']:.4f}")
    print(
        f"CatBoost (uncalibrated) -- validation ROC-AUC: {cfg['validation_auc']['catboost_uncalibrated']:.4f} "
        f"(best_iteration={cfg['catboost_best_iteration']})"
    )
    print(f"Calibration: fit sigmoid + isotonic on validation ({cfg['validation_rows']} rows); recommended = {cfg['calibration']['chosen_method']!r}")
    print(f"Artifacts written to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
