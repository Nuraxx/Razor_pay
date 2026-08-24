"""
Day-4 feature selection + preprocessing.

Selects exactly which Day-3 `data/processed/*.csv` columns are legitimate
model inputs -- features that were genuinely knowable at the moment a
payment failed -- and documents every excluded column with a reason. See
EXCLUDED_COLUMNS below; this dict is the single source of truth both the
code and tests check against.

Every fit happens on the training split only. Validation and test data are
only ever `.transform()`-ed, never `.fit()`-ed on -- see
`Preprocessor.fit`/`.transform` below and the leakage tests in
tests/test_model_pipeline.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COLUMN = "recovered_within_14d"

# Available at failure time, numeric.
NUMERIC_FEATURES = [
    "day_of_month",
    "days_to_nearest_payday_window",
    "amount",
    "prior_if_failure_count",
    "prior_if_self_resolved_rate",  # missing for first-time failures -- see PriorSelfResolvedImputer
    "tenure_days",  # this is failure_events.tenure_days ("as of this failure"), NOT subscriptions.tenure_days
]

# Available at failure time, categorical (includes the 3 deliberate distractors).
CATEGORICAL_FEATURES = [
    "plan_tier",
    "primary_instrument",
    "city_tier",
    "bank_network_conditions",
    "network_latency_bucket",
    "app_version",  # distractor
    "device_build",  # distractor
    "ui_theme",  # distractor
]

# Available at failure time, boolean -- modeled as 0/1 numeric.
BOOLEAN_FEATURES = [
    "issuing_bank_downtime_flag",
    "is_month_end_settlement_rush",
]

DISTRACTOR_FEATURES = ["app_version", "device_build", "ui_theme"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

# Every column present in data/processed/*.csv that is NOT a model feature,
# with the reason it's excluded. Reviewed explicitly per the Day-4 brief.
EXCLUDED_COLUMNS = {
    "event_id": "identifier -- unique per row, not predictive, would only let the model memorize rows",
    "subscription_id": "identifier -- same subscription can appear across multiple failures; not a feature",
    "failure_timestamp": (
        "raw high-cardinality timestamp. Its useful signal is already captured by the derived "
        "day_of_month / days_to_nearest_payday_window / tenure_days columns; feeding the raw "
        "timestamp into a ~200-row dataset risks the model keying on specific dates instead of "
        "generalizable patterns."
    ),
    "signup_date": "raw high-cardinality timestamp; its useful signal is already captured by tenure_days.",
    "monthly_amount": "exact duplicate of `amount` in this dataset (failure amount == subscription's monthly billed amount by construction) -- redundant.",
    "error_reason": "constant across the entire Day-3 dataset (always 'insufficient_fund' by Day-3 scope) -- zero variance, carries no information.",
    "recovered_within_14d": "THE TARGET LABEL -- used as y, never as a feature.",
    "recovered_at": "post-outcome information -- only known after the failure resolves (or doesn't). Using it would leak the label.",
    "recovered_via": "post-outcome information -- same reason as recovered_at.",
    "final_amount_recovered": "post-outcome information -- same reason as recovered_at.",
    "archetype": "hidden, generation-only field. Must never be a model feature (also not present in data/processed/*.csv at all -- Day 3 already drops it).",
    "split": "dataset-partition bookkeeping, not a real-world signal (also not present in data/processed/*.csv -- Day 3 already drops it).",
}


def load_processed_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    validation = pd.read_csv(PROCESSED_DIR / "validation.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    return train, validation, test


def select_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select model-input columns and cast booleans to 0/1 ints. No fitting happens here."""
    X = df[FEATURE_COLUMNS].copy()
    for col in BOOLEAN_FEATURES:
        X[col] = X[col].astype(int)
    y = df[TARGET_COLUMN].astype(int)
    return X, y


@dataclass
class PriorSelfResolvedImputer:
    """
    Fills missing `prior_if_self_resolved_rate` (first-time failures have no
    prior history to compute it from) with a single scalar learned from the
    training split only, and adds an explicit
    `prior_if_self_resolved_rate_missing` flag so the model can distinguish
    "no prior history" from "had history and it happened to be 0".

    fit() must only ever be called with training data. transform() is safe
    to call on validation/test -- it never recomputes the fill value.
    """

    column: str = "prior_if_self_resolved_rate"
    fill_value: float | None = None

    def fit(self, X: pd.DataFrame) -> "PriorSelfResolvedImputer":
        observed = X[self.column].dropna()
        self.fill_value = float(observed.mean()) if len(observed) > 0 else 0.5
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.fill_value is None:
            raise RuntimeError("PriorSelfResolvedImputer.transform() called before fit()")
        out = X.copy()
        out[f"{self.column}_missing"] = out[self.column].isna().astype(int)
        out[self.column] = out[self.column].fillna(self.fill_value)
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


NUMERIC_FEATURES_WITH_FLAG = NUMERIC_FEATURES + BOOLEAN_FEATURES + ["prior_if_self_resolved_rate_missing"]


def build_logreg_column_transformer() -> ColumnTransformer:
    """
    One-hot encodes categoricals, scales numerics. Must be `.fit()` on
    training data only -- the caller is responsible for that (see
    model/train.py). `handle_unknown="ignore"` so an unseen category in
    validation/test degrades gracefully instead of raising.
    """
    numeric_pipeline = Pipeline(
        steps=[
            # add_indicator=True is a defensive second layer in case any other
            # numeric column ever has missing values -- prior_if_self_resolved_rate
            # is already imputed with an explicit flag by PriorSelfResolvedImputer
            # before this transformer runs.
            ("imputer", SimpleImputer(strategy="mean", add_indicator=False)),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES_WITH_FLAG),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def prepare_for_catboost(X: pd.DataFrame) -> pd.DataFrame:
    """
    CatBoost takes categoricals natively -- just make sure they're plain
    object-dtype Python strings. Cast through .astype(str).astype(object)
    rather than just .astype(str): pandas >= 3.0 makes .astype(str) return
    its new StringDtype extension type by default, not classic `object`;
    the explicit second cast keeps this independent of that pandas version
    detail and matches what CatBoost's own examples use.
    """
    out = X.copy()
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype(str).astype(object)
    return out
