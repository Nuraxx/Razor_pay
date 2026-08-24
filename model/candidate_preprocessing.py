"""
Day-6 candidate-level feature selection + preprocessing.

Mirrors model/preprocessing.py's structure and every one of its leakage
safeguards, extended with candidate-time features so the model can condition
on WHICH retry candidate is being scored, not just the failure context.

No new data/processed/*.csv files are written -- the candidate-level table
(5 rows per failure event) is built at load time by joining the raw tables
Day 3 and Day 6 already produce:

    data/raw/counterfactual_outcomes.csv   (Day 6 -- target + candidate_type)
    data/raw/retry_candidates.csv          (Day 3 -- candidate_datetime, offset/alignment)
    data/raw/failure_events.csv            (Day 3 -- failure-time/customer features)
    data/raw/subscriptions.csv             (Day 3 -- leakage-safe subscription columns + split)

Every fit happens on the training split only, exactly like Day 4 -- see
`load_candidate_splits` and the leakage tests in tests/test_candidate_model.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.generate_synthetic_dataset import days_to_nearest_payday_window
from model.preprocessing import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PriorSelfResolvedImputer,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

TARGET_COLUMN = "recovered_within_14d"

# Candidate-action features (Day 6, section 2 of the brief) -- everything
# about WHEN/WHICH candidate, layered on top of Day 4's unchanged
# failure-time/customer features (NUMERIC_FEATURES / CATEGORICAL_FEATURES /
# BOOLEAN_FEATURES, imported above, reused verbatim).
CANDIDATE_NUMERIC_FEATURES = ["hours_from_failure", "candidate_day_of_month", "candidate_days_to_payday"]
CANDIDATE_CATEGORICAL_FEATURES = ["candidate_type", "candidate_day_of_week"]
CANDIDATE_BOOLEAN_FEATURES = ["candidate_is_payday_aligned", "candidate_is_month_end_aligned"]

ALL_NUMERIC_FEATURES = NUMERIC_FEATURES + CANDIDATE_NUMERIC_FEATURES
ALL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + CANDIDATE_CATEGORICAL_FEATURES
ALL_BOOLEAN_FEATURES = BOOLEAN_FEATURES + CANDIDATE_BOOLEAN_FEATURES

FEATURE_COLUMNS = ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES + ALL_BOOLEAN_FEATURES

# Every column present in the joined candidate-level table that is NOT a
# model feature, with the reason -- same convention as
# model/preprocessing.py::EXCLUDED_COLUMNS, reviewed explicitly per the
# Day-6 brief section 3 ("avoid post-treatment leakage").
EXCLUDED_COLUMNS = {
    "counterfactual_id": "identifier -- unique per row, not predictive.",
    "event_id": "identifier -- same failure event repeats across its 5 candidate rows; not a feature.",
    "subscription_id": "identifier -- same subscription can appear across multiple failures; not a feature.",
    "candidate_datetime": (
        "raw high-cardinality timestamp. Its useful signal is already captured by the derived "
        "candidate_day_of_month / candidate_day_of_week / candidate_days_to_payday / "
        "candidate_is_payday_aligned / candidate_is_month_end_aligned / hours_from_failure columns."
    ),
    "failure_timestamp": "same reasoning as Day 4 -- captured by day_of_month / days_to_nearest_payday_window / tenure_days.",
    "signup_date": "raw high-cardinality timestamp; captured by tenure_days.",
    "monthly_amount": "exact duplicate of `amount` in this dataset -- redundant.",
    "error_reason": "constant ('insufficient_fund') across this dataset -- zero variance.",
    "recovery_probability_latent": (
        "POST-TREATMENT / label-adjacent -- this is the hidden generation mechanism's own latent "
        "probability for this exact candidate. Using it as a feature would let the model read the "
        "answer off a column instead of learning it from legitimate context."
    ),
    "recovered_within_14d": "THE TARGET LABEL -- used as y, never as a feature.",
    "recovered_at": "post-outcome information -- only known after the candidate retry resolves (or doesn't).",
    "recovered_via": "post-outcome information -- same reason as recovered_at.",
    "amount_recovered": "post-outcome information -- same reason as recovered_at.",
    "archetype": "hidden, generation-only field. Must never be a model feature.",
    "split": "dataset-partition bookkeeping, not a real-world signal.",
}


def _load_raw_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counterfactual = pd.read_csv(RAW_DIR / "counterfactual_outcomes.csv", parse_dates=["candidate_datetime", "recovered_at"])
    retry_candidates = pd.read_csv(RAW_DIR / "retry_candidates.csv", parse_dates=["candidate_datetime"])
    failure_events = pd.read_csv(RAW_DIR / "failure_events.csv", parse_dates=["failure_timestamp"])
    subscriptions = pd.read_csv(RAW_DIR / "subscriptions.csv", parse_dates=["signup_date"])
    return counterfactual, retry_candidates, failure_events, subscriptions


def build_candidate_level_dataset_from_tables(
    counterfactual: pd.DataFrame, retry_candidates: pd.DataFrame, failure_events: pd.DataFrame, subscriptions: pd.DataFrame
) -> pd.DataFrame:
    """Pure join logic, no I/O -- separated out from build_candidate_level_dataset()
    so tests can exercise it against a small in-memory dataset (see
    tests/test_candidate_model.py), the same pattern Day 4's
    tests/test_model_pipeline.py uses via data.generate_synthetic_dataset.generate_dataset().

    One row per (failure event, candidate_type) -- 5x failure_events'
    row count. Includes `split`, inherited from the underlying subscription
    exactly like Day 3's processed splits (a subscription and every one of
    its candidate rows lands in exactly one split)."""
    candidate_cols = retry_candidates[
        ["event_id", "candidate_type", "offset_hours_from_failure", "is_payday_aligned", "is_month_end_aligned"]
    ].rename(columns={"offset_hours_from_failure": "hours_from_failure"})

    df = counterfactual.merge(candidate_cols, on=["event_id", "candidate_type"], how="left")
    df = df.merge(failure_events.drop(columns=["subscription_id"]), on="event_id", how="left")

    sub_cols = subscriptions[
        ["subscription_id", "plan_tier", "monthly_amount", "primary_instrument", "city_tier", "signup_date", "split"]
    ]
    df = df.merge(sub_cols, on="subscription_id", how="left")

    df["candidate_day_of_month"] = df["candidate_datetime"].dt.day
    df["candidate_day_of_week"] = df["candidate_datetime"].dt.day_name()
    df["candidate_days_to_payday"] = df["candidate_datetime"].apply(days_to_nearest_payday_window)
    df["candidate_is_payday_aligned"] = df["is_payday_aligned"]
    df["candidate_is_month_end_aligned"] = df["is_month_end_aligned"]
    df = df.drop(columns=["is_payday_aligned", "is_month_end_aligned"])  # superseded by the candidate_is_* columns above

    return df


def build_candidate_level_dataset() -> pd.DataFrame:
    """Reads the committed raw tables from disk and delegates to the pure
    join logic above."""
    counterfactual, retry_candidates, failure_events, subscriptions = _load_raw_tables()
    return build_candidate_level_dataset_from_tables(counterfactual, retry_candidates, failure_events, subscriptions)


def split_candidate_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits an already-built candidate-level table (must have a `split`
    column) into train/validation/test, dropping `split` itself -- pure,
    no I/O, reused by both load_candidate_splits() and tests."""
    train = df[df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    validation = df[df["split"] == "validation"].drop(columns=["split"]).reset_index(drop=True)
    test = df[df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
    return train, validation, test


def load_candidate_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return split_candidate_dataset(build_candidate_level_dataset())


def select_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select model-input columns and cast booleans to 0/1 ints. No fitting happens here."""
    X = df[FEATURE_COLUMNS].copy()
    for col in ALL_BOOLEAN_FEATURES:
        X[col] = X[col].astype(int)
    y = df[TARGET_COLUMN].astype(int)
    return X, y


# PriorSelfResolvedImputer is reused unchanged from model/preprocessing.py --
# `prior_if_self_resolved_rate` is a failure-time feature, identical meaning
# and identical missingness pattern here as in Day 4 (first-time failures
# have no prior history regardless of which candidate is being scored).
ALL_NUMERIC_FEATURES_WITH_FLAG = ALL_NUMERIC_FEATURES + ALL_BOOLEAN_FEATURES + ["prior_if_self_resolved_rate_missing"]


def build_candidate_logreg_column_transformer() -> ColumnTransformer:
    """Same shape as model/preprocessing.py::build_logreg_column_transformer,
    extended to the candidate-level numeric/categorical feature lists above.
    Must be `.fit()` on training data only -- the caller is responsible for
    that (see model/train_candidate_model.py)."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean", add_indicator=False)),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, ALL_NUMERIC_FEATURES_WITH_FLAG),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ALL_CATEGORICAL_FEATURES),
        ]
    )


def prepare_for_catboost(X: pd.DataFrame) -> pd.DataFrame:
    """CatBoost takes categoricals natively -- see model/preprocessing.py's
    identical function for the reasoning behind the double .astype() cast."""
    out = X.copy()
    for col in ALL_CATEGORICAL_FEATURES:
        out[col] = out[col].astype(str).astype(object)
    return out
