"""
Day-7 ranking-model feature selection.

Reuses Day 6's join logic (model/candidate_preprocessing.py::build_candidate_level_dataset_from_tables)
unchanged -- no new raw tables, no re-derivation of the candidate-level
join. The only thing that differs is FEATURE_COLUMNS: the Day-7 brief
(section 5) specifies an explicit feature list that excludes the 3
distractor columns Day 4/6 carried along (app_version, device_build,
ui_theme) -- honored literally here rather than silently reusing Day 6's
broader list, since a ranking model with already-weak within-group signal
benefits from not spending capacity on columns already proven non-predictive.

Day 6's model/candidate_preprocessing.py is otherwise completely untouched
-- this module only adds a narrower feature view on top of the same table.
"""
from __future__ import annotations

from model.candidate_preprocessing import (
    EXCLUDED_COLUMNS as _DAY6_EXCLUDED_COLUMNS,
    PriorSelfResolvedImputer,
    RAW_DIR,
    TARGET_COLUMN,
    build_candidate_level_dataset,
    build_candidate_level_dataset_from_tables,
    split_candidate_dataset,
)
from model.preprocessing import PROJECT_ROOT, SEED  # noqa: F401 -- re-exported for train_ranking_model.py

# Failure-time/customer context (brief section 5, first list).
EVENT_NUMERIC_FEATURES = [
    "day_of_month",
    "days_to_nearest_payday_window",
    "amount",
    "prior_if_failure_count",
    "prior_if_self_resolved_rate",
    "tenure_days",
]
EVENT_CATEGORICAL_FEATURES = ["plan_tier", "primary_instrument", "city_tier", "bank_network_conditions", "network_latency_bucket"]
EVENT_BOOLEAN_FEATURES = ["issuing_bank_downtime_flag", "is_month_end_settlement_rush"]

# Candidate-action features (brief section 5, second list).
CANDIDATE_NUMERIC_FEATURES = ["hours_from_failure", "candidate_day_of_month", "candidate_days_to_payday"]
CANDIDATE_CATEGORICAL_FEATURES = ["candidate_type", "candidate_day_of_week"]
CANDIDATE_BOOLEAN_FEATURES = ["candidate_is_payday_aligned", "candidate_is_month_end_aligned"]

ALL_NUMERIC_FEATURES = EVENT_NUMERIC_FEATURES + CANDIDATE_NUMERIC_FEATURES
ALL_CATEGORICAL_FEATURES = EVENT_CATEGORICAL_FEATURES + CANDIDATE_CATEGORICAL_FEATURES
ALL_BOOLEAN_FEATURES = EVENT_BOOLEAN_FEATURES + CANDIDATE_BOOLEAN_FEATURES

FEATURE_COLUMNS = ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES + ALL_BOOLEAN_FEATURES

# Distractors and everything Day 6 already excludes (post-outcome fields,
# hidden archetype, identifiers, split) apply unchanged here too -- see
# model/candidate_preprocessing.py::EXCLUDED_COLUMNS for the full reasoning.
EXCLUDED_COLUMNS = dict(_DAY6_EXCLUDED_COLUMNS)
EXCLUDED_COLUMNS.update(
    {
        "app_version": "distractor -- proven non-predictive (Day 4/6); explicitly not in the Day-7 brief's feature list.",
        "device_build": "distractor -- same reason as app_version.",
        "ui_theme": "distractor -- same reason as app_version.",
    }
)


def select_features_and_target(df):
    """Same shape as model/candidate_preprocessing.py::select_features_and_target,
    restricted to the Day-7 feature list."""
    X = df[FEATURE_COLUMNS].copy()
    for col in ALL_BOOLEAN_FEATURES:
        X[col] = X[col].astype(int)
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def prepare_for_catboost(X):
    out = X.copy()
    for col in ALL_CATEGORICAL_FEATURES:
        out[col] = out[col].astype(str).astype(object)
    return out
