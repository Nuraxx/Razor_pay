"""
Day-4 final evaluation entry point.

    ./venv/bin/python evaluation/evaluate_models.py

Loads every artifact model/train.py saved (never retrains anything) and
evaluates all of them against data/processed/test.csv -- the one dataset
neither training nor calibration has ever seen. This is the only script in
Day 4 that touches the test split.

Also evaluates two trivial baselines (majority-class, training-set base
rate) and reports bootstrap confidence intervals on the primary metrics,
since the dataset is intentionally small.
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from model.preprocessing import (
    CATEGORICAL_FEATURES,
    DISTRACTOR_FEATURES,
    PROJECT_ROOT,
    SEED,
    load_processed_splits,
    prepare_for_catboost,
    select_features_and_target,
)

ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "model" / "reports"
N_BOOTSTRAP = 1000


def load_artifacts() -> dict:
    logreg_preprocessor = joblib.load(ARTIFACTS_DIR / "logreg_preprocessor.joblib")
    logreg_model = joblib.load(ARTIFACTS_DIR / "logreg_model.joblib")
    imputer = joblib.load(ARTIFACTS_DIR / "prior_self_resolved_imputer.joblib")
    catboost_model = CatBoostClassifier()
    catboost_model.load_model(str(ARTIFACTS_DIR / "catboost_model.cbm"))
    sigmoid_calibrator = joblib.load(ARTIFACTS_DIR / "catboost_calibrated_sigmoid.joblib")
    isotonic_calibrator = joblib.load(ARTIFACTS_DIR / "catboost_calibrated_isotonic.joblib")
    with open(ARTIFACTS_DIR / "model_config.json") as f:
        model_config = json.load(f)
    return {
        "logreg_preprocessor": logreg_preprocessor,
        "logreg_model": logreg_model,
        "imputer": imputer,
        "catboost_model": catboost_model,
        "sigmoid_calibrator": sigmoid_calibrator,
        "isotonic_calibrator": isotonic_calibrator,
        "model_config": model_config,
    }


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_prob_clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)  # keep log loss finite for degenerate (0/1) predictions
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "log_loss": float(log_loss(y_true, y_prob_clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_fn, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_t, y_p = y_true[idx], y_prob[idx]
        if len(set(y_t)) < 2:
            continue  # a metric like AUC is undefined for a single-class resample; skip it
        scores.append(metric_fn(y_t, y_p))
    scores = np.array(scores)
    return {
        "mean": float(scores.mean()),
        "ci_2.5": float(np.percentile(scores, 2.5)),
        "ci_97.5": float(np.percentile(scores, 97.5)),
        "n_bootstrap_valid": int(len(scores)),
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = load_artifacts()
    config = artifacts["model_config"]

    _train_df_unused, _val_df_unused, test_df = load_processed_splits()
    del _train_df_unused, _val_df_unused  # evaluation only ever reads test.csv's rows for predictions

    X_test_raw, y_test = select_features_and_target(test_df)
    X_test = artifacts["imputer"].transform(X_test_raw)  # transform only -- fitted on train in model/train.py
    y_test_arr = y_test.to_numpy()

    X_test_logreg = artifacts["logreg_preprocessor"].transform(X_test)  # transform only
    X_test_cb = prepare_for_catboost(X_test)

    predictions = {
        "trivial_majority_class": np.full(len(y_test), 1.0 if config["train_recovery_rate"] >= 0.5 else 0.0),
        "trivial_train_recovery_rate": np.full(len(y_test), config["train_recovery_rate"]),
        "logistic_regression": artifacts["logreg_model"].predict_proba(X_test_logreg)[:, 1],
        "catboost_uncalibrated": artifacts["catboost_model"].predict_proba(X_test_cb)[:, 1],
        "catboost_calibrated_sigmoid": artifacts["sigmoid_calibrator"].predict_proba(X_test_cb)[:, 1],
        "catboost_calibrated_isotonic": artifacts["isotonic_calibrator"].predict_proba(X_test_cb)[:, 1],
    }

    for name, probs in predictions.items():
        assert probs.min() >= 0.0 and probs.max() <= 1.0, f"{name} produced a probability outside [0,1]"

    metrics_report = {}
    for name, probs in predictions.items():
        m = compute_metrics(y_test_arr, probs)
        if m["roc_auc"] is not None:
            m["roc_auc_bootstrap_ci"] = bootstrap_ci(y_test_arr, probs, roc_auc_score)
            m["pr_auc_bootstrap_ci"] = bootstrap_ci(y_test_arr, probs, average_precision_score)
        metrics_report[name] = m

    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_report, f, indent=2)

    flat_rows = []
    for name, m in metrics_report.items():
        row = {"model": name, **{k: v for k, v in m.items() if k not in ("confusion_matrix", "roc_auc_bootstrap_ci", "pr_auc_bootstrap_ci")}}
        row.update({f"cm_{k}": v for k, v in m["confusion_matrix"].items()})
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(REPORTS_DIR / "metrics.csv", index=False)

    # --- Feature importance (CatBoost) ---
    importances = artifacts["catboost_model"].get_feature_importance()
    feature_names = artifacts["catboost_model"].feature_names_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    fi_df["is_distractor"] = fi_df["feature"].isin(DISTRACTOR_FEATURES)
    fi_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    # --- Calibration curve plot: uncalibrated vs sigmoid-calibrated, on test ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    for label, key in [("CatBoost (uncalibrated)", "catboost_uncalibrated"), ("CatBoost (sigmoid-calibrated)", "catboost_calibrated_sigmoid")]:
        frac_pos, mean_pred = calibration_curve(y_test_arr, predictions[key], n_bins=5, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed recovery rate")
    ax.set_title("Calibration curve (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "calibration_curve.png", dpi=150)
    plt.close(fig)

    # --- Confusion matrices ---
    for name in ("logistic_regression", "catboost_uncalibrated"):
        y_pred = (predictions[name] >= 0.5).astype(int)
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(y_test_arr, y_pred, ax=ax, colorbar=False)
        ax.set_title(f"Confusion matrix -- {name} (test)")
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / f"confusion_matrix_{name}.png", dpi=150)
        plt.close(fig)

    # --- Console summary ---
    print(f"Test rows: {len(test_df)} | test recovery rate: {y_test_arr.mean():.4f}")
    print()
    for name, m in metrics_report.items():
        auc_str = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "n/a"
        pr_str = f"{m['pr_auc']:.4f}" if m["pr_auc"] is not None else "n/a"
        print(f"{name:32s} ROC-AUC={auc_str:>7s} PR-AUC={pr_str:>7s} LogLoss={m['log_loss']:.4f} Brier={m['brier_score']:.4f} Acc={m['accuracy']:.4f}")
    print()
    print("Feature importance (top 10):")
    print(fi_df.head(10).to_string(index=False))
    print()
    distractor_ranks = fi_df.reset_index(drop=True)
    distractor_ranks["rank"] = distractor_ranks.index + 1
    print("Distractor feature ranks (out of", len(fi_df), "features):")
    print(distractor_ranks[distractor_ranks["is_distractor"]][["feature", "importance", "rank"]].to_string(index=False))
    print()
    print(f"Reports written to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
