"""
Within-event ranking model.

    ./venv/bin/python model/train_ranking_model.py

ROOT CAUSE THIS FIXES (see model/diagnose_ranking_failure.py for the full
writeup, reproducible directly): the candidate-aware model's CatBoost was
trained with pointwise log-loss on POOLED rows and selected by
POOLED validation AUC. `hours_from_failure` alone captured 93.3% of its
feature importance -- mostly a proxy for "does this candidate's own
scheduled time still leave room to recover within 14 days" -- which crowded
out the actual causal candidate-timing signal (`candidate_days_to_payday`:
0.03% importance, `candidate_is_payday_aligned`/`is_month_end_aligned`:
0.0%). A model can nail pooled AUC/log-loss entirely off that one dominant,
easy-to-fit split while getting WITHIN-event ordering wrong -- which is
exactly what happened (mean within-event rank correlation: -0.149, worse
than the ~+0.11 achievable from `candidate_days_to_payday` alone). Platt
calibration cannot fix this: it's a monotonic transform, provably unable to
change within-group rank order (verified empirically -- see diagnosis).

THE FIX (brief section 4, Option A: pairwise ranking, chosen as the
simplest defensible method -- CatBoost's own specialized ranking Pool API
was considered but explicit pair construction is more testable and
transparent, and doesn't require a new dependency):

For every failure event, for every (higher-label, lower-label) candidate
pair -- i.e. one candidate that recovered and one that didn't, in the SAME
event -- build a training example whose feature vector is the DIFFERENCE
of the two candidates' (one-hot-encoded, scaled) feature vectors, and whose
target is "did the first candidate rank higher" (1) or not (0). Both
directions of every informative pair are included, so the training set is
symmetric. A model trained this way must explicitly compare two candidates
FROM THE SAME EVENT to get any signal at all -- it structurally cannot
"solve" the problem by fitting a global threshold on one dominant feature,
because a constant additive offset cancels out in the subtraction.

At inference time, an event's 5 candidates are scored via a round-robin
tournament: for each candidate, its score is the mean predicted "beats"
probability against the other 4 candidates of the SAME event
(`score_candidates_for_event`). This produces a bounded [0,1] per-candidate
score, directly usable wherever the candidate-aware model's pointwise
probability was used -- including
`policy/recovery_policy.py::decide_candidate_aware`, unchanged.
"""
from __future__ import annotations

import json
from itertools import product

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model.candidate_preprocessing import PriorSelfResolvedImputer
from model.preprocessing import PROJECT_ROOT, SEED
from model.ranking_preprocessing import (
    ALL_BOOLEAN_FEATURES,
    ALL_CATEGORICAL_FEATURES,
    ALL_NUMERIC_FEATURES,
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    select_features_and_target,
)

ARTIFACTS_DIR = PROJECT_ROOT / "model" / "ranking_artifacts"

CATBOOST_PARAMS = dict(
    iterations=300,
    depth=4,
    learning_rate=0.05,
    loss_function="Logloss",  # the pairwise MODEL is itself an ordinary binary classifier -- see module docstring; "pairwise" describes the DATASET, not a specialized CatBoost ranking loss
    eval_metric="AUC",
    random_seed=SEED,
    early_stopping_rounds=50,
    use_best_model=True,
    verbose=False,
)


def build_column_transformer() -> ColumnTransformer:
    """Encodes a candidate-level feature row into a purely numeric vector --
    required so pairwise DIFFERENCES are well-defined (you cannot subtract
    two categorical values). Must be `.fit()` on training rows only."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean", add_indicator=False)),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_and_boolean = ALL_NUMERIC_FEATURES + ALL_BOOLEAN_FEATURES + ["prior_if_self_resolved_rate_missing"]
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_and_boolean),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ALL_CATEGORICAL_FEATURES),
        ]
    )


def build_pairwise_dataset(Z: np.ndarray, y: np.ndarray, event_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Z: (n, d) already-encoded numeric feature matrix.
    y: (n,) binary labels (recovered_within_14d).
    event_ids: (n,) group identifiers -- a "ranking group" per the brief.

    For every event, for every pair (i, j) in that event with y_i != y_j,
    emits BOTH directions: (Z_i - Z_j, 1) and (Z_j - Z_i, 0). Pairs are never
    formed across two different event_ids -- each event's 5 candidates form
    one ranking group, exactly as the brief requires. Events where every
    candidate shares the same label contribute zero pairs (no informative
    comparison exists) -- see model/diagnose_ranking_failure.py for how
    large that fraction is and why it is not the dominant driver of the
    candidate-aware model's failure.
    """
    y = np.asarray(y)
    event_ids = np.asarray(event_ids)
    diffs: list[np.ndarray] = []
    labels: list[int] = []

    for event_id in np.unique(event_ids):
        idx = np.where(event_ids == event_id)[0]
        pos_idx = idx[y[idx] == 1]
        neg_idx = idx[y[idx] == 0]
        for i, j in product(pos_idx, neg_idx):
            diffs.append(Z[i] - Z[j])
            labels.append(1)
            diffs.append(Z[j] - Z[i])
            labels.append(0)

    if not diffs:
        return np.empty((0, Z.shape[1])), np.empty((0,), dtype=int)
    return np.vstack(diffs), np.array(labels, dtype=int)


def fit_pipeline(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Fit everything: imputer -> column transformer -> pairwise CatBoost
    classifier, with early stopping on a pairwise-encoded VALIDATION set.
    Deterministic given the same (train_df, val_df). Takes no test data."""
    X_train_raw, y_train = select_features_and_target(train_df)
    X_val_raw, y_val = select_features_and_target(val_df)

    imputer = PriorSelfResolvedImputer().fit(X_train_raw)  # fit on train only
    X_train = imputer.transform(X_train_raw)
    X_val = imputer.transform(X_val_raw)

    transformer = build_column_transformer()
    Z_train = transformer.fit_transform(X_train)  # fit on train only
    Z_val = transformer.transform(X_val)  # transform only, never fit
    if hasattr(Z_train, "toarray"):
        Z_train, Z_val = Z_train.toarray(), Z_val.toarray()

    pair_X_train, pair_y_train = build_pairwise_dataset(Z_train, y_train.to_numpy(), train_df["event_id"].to_numpy())
    pair_X_val, pair_y_val = build_pairwise_dataset(Z_val, y_val.to_numpy(), val_df["event_id"].to_numpy())

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(pair_X_train, pair_y_train, eval_set=(pair_X_val, pair_y_val))
    pairwise_val_auc = roc_auc_score(pair_y_val, model.predict_proba(pair_X_val)[:, 1])

    val_scores = predict_ranking_scores(val_df, {"imputer": imputer, "transformer": transformer, "model": model})
    val_ranking = _quick_top1_accuracy(val_df, val_scores)

    model_config = {
        "seed": SEED,
        "feature_columns": FEATURE_COLUMNS,
        "catboost_params": dict(CATBOOST_PARAMS),
        "catboost_best_iteration": model.get_best_iteration(),
        "train_events": train_df["event_id"].nunique(),
        "train_rows": len(train_df),
        "validation_events": val_df["event_id"].nunique(),
        "validation_rows": len(val_df),
        "n_pairwise_train_examples": len(pair_y_train),
        "n_pairwise_validation_examples": len(pair_y_val),
        "pairwise_validation_auc": float(pairwise_val_auc),
        "validation_top1_accuracy": val_ranking,
    }

    return {
        "imputer": imputer,
        "transformer": transformer,
        "model": model,
        "feature_list": {"feature_columns": FEATURE_COLUMNS, "excluded_columns": EXCLUDED_COLUMNS, "target_column": "recovered_within_14d"},
        "model_config": model_config,
    }


def _encode(df: pd.DataFrame, fitted: dict) -> np.ndarray:
    X_raw, _y = select_features_and_target(df)
    X = fitted["imputer"].transform(X_raw)
    Z = fitted["transformer"].transform(X)
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    return Z


def score_candidates_for_event(Z_group: np.ndarray, model: CatBoostClassifier) -> np.ndarray:
    """Round-robin tournament: candidate i's score = mean predicted
    P(i beats j) over every other candidate j in the SAME group. Returns
    one score per row of Z_group, in [0, 1] (each score is itself a mean of
    predicted probabilities, so it can never leave that range)."""
    n = len(Z_group)
    if n == 1:
        return np.array([0.5])  # no opponent to compare against -- neutral score, not a claim of merit

    pair_diffs = []
    pair_owner = []
    for i, j in product(range(n), range(n)):
        if i == j:
            continue
        pair_diffs.append(Z_group[i] - Z_group[j])
        pair_owner.append(i)

    pair_diffs = np.vstack(pair_diffs)
    win_probs = model.predict_proba(pair_diffs)[:, 1]

    scores = np.zeros(n)
    counts = np.zeros(n)
    for owner, p in zip(pair_owner, win_probs):
        scores[owner] += p
        counts[owner] += 1
    return scores / counts


def predict_ranking_scores(df: pd.DataFrame, fitted: dict) -> pd.Series:
    """One ranking score per row of `df`, computed group-by-group (event_id)
    via round-robin tournament scoring. Index-aligned with `df`."""
    Z = _encode(df, fitted)
    scores = np.empty(len(df))
    df_reset = df.reset_index(drop=True)
    for event_id in df_reset["event_id"].unique():
        idx = df_reset.index[df_reset["event_id"] == event_id].to_numpy()
        scores[idx] = score_candidates_for_event(Z[idx], fitted["model"])
    return pd.Series(scores, index=df.index)


def _quick_top1_accuracy(df: pd.DataFrame, scores: pd.Series) -> float:
    """Lightweight validation-only ranking check (model selection signal,
    NOT tuned against test) -- top-1 accuracy against the REALIZED label
    (the only label available without peeking at recovery_probability_latent,
    which the model must never see)."""
    d = df.assign(_score=scores)
    hits = 0
    n = 0
    for _event_id, g in d.groupby("event_id"):
        if g["recovered_within_14d"].nunique() < 2:
            continue  # uninformative group -- no defined "correct" top-1 among ties
        predicted_best = g.loc[g["_score"].idxmax(), "candidate_type"]
        actual_best_candidates = set(g.loc[g["recovered_within_14d"] == 1, "candidate_type"])
        hits += int(predicted_best in actual_best_candidates)
        n += 1
    return hits / n if n else None


def save_artifacts(fitted: dict, artifacts_dir) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted["imputer"], artifacts_dir / "prior_self_resolved_imputer.joblib")
    joblib.dump(fitted["transformer"], artifacts_dir / "column_transformer.joblib")
    fitted["model"].save_model(str(artifacts_dir / "pairwise_catboost_model.cbm"))
    with open(artifacts_dir / "feature_list.json", "w") as f:
        json.dump(fitted["feature_list"], f, indent=2)
    with open(artifacts_dir / "model_config.json", "w") as f:
        json.dump(fitted["model_config"], f, indent=2, default=str)


def load_ranking_model() -> dict:
    imputer = joblib.load(ARTIFACTS_DIR / "prior_self_resolved_imputer.joblib")
    transformer = joblib.load(ARTIFACTS_DIR / "column_transformer.joblib")
    model = CatBoostClassifier()
    model.load_model(str(ARTIFACTS_DIR / "pairwise_catboost_model.cbm"))
    return {"imputer": imputer, "transformer": transformer, "model": model}


def main() -> None:
    from model.candidate_preprocessing import load_candidate_splits

    train_df, val_df, _test_df_unused = load_candidate_splits()
    del _test_df_unused  # test is never used for anything training-related

    fitted = fit_pipeline(train_df, val_df)
    save_artifacts(fitted, ARTIFACTS_DIR)

    cfg = fitted["model_config"]
    print(f"Train rows: {cfg['train_rows']} ({cfg['train_events']} events) | Validation rows: {cfg['validation_rows']} ({cfg['validation_events']} events)")
    print(f"Pairwise training examples: {cfg['n_pairwise_train_examples']} | validation: {cfg['n_pairwise_validation_examples']}")
    print(f"Pairwise CatBoost (best_iteration={cfg['catboost_best_iteration']}) -- validation pairwise AUC: {cfg['pairwise_validation_auc']:.4f}")
    print(f"Validation top-1 accuracy (round-robin scoring, vs. realized label): {cfg['validation_top1_accuracy']}")
    print(f"Artifacts written to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
