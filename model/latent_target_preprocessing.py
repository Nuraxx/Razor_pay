"""
Latent-target construction: fixing the OBJECTIVE, not adding another model.

The ranking model's root-cause finding: `recovered_within_14d` is a single
noisy Bernoulli realization of a latent probability, further confounded by
14-day horizon truncation. A model can get measurably better at predicting
that noisy/confounded target without that improvement transferring to
agreement with the true underlying preference ordering. The candidate-aware
model's own synthetic generator already computed a continuous latent
probability for every candidate (`data/generate_counterfactual_dataset.py`)
-- this module trains directly against THAT, instead of against its noisy
sample.

THREE DISTINCT CONCEPTS -- documented explicitly per the brief, never
treated as interchangeable:

  A. OBSERVED OUTCOME -- `recovered_within_14d` (data/raw/counterfactual_outcomes.csv)
     A noisy Bernoulli REALIZATION: one coin-flip per (event, candidate),
     sampled from the latent probability, further constrained to False
     whenever the candidate's own scheduled time falls beyond the 14-day
     recovery horizon. This is what the candidate-aware and ranking models
     trained against.

  B. LATENT RECOVERY PROBABILITY -- `recovery_probability_latent`
     The synthetic GROUND TRUTH the generator itself computed before
     sampling (A) from it. Continuous, in [0.02, 0.98]. Exists only because
     this is a synthetic benchmark with a known generating mechanism --
     see IMPORTANT note below.

  C. LATENT EXPECTED MONEY VALUE -- `expected_recovery_value_latent`
     = recovery_probability_latent * amount. The synthetic ground-truth
     ECONOMIC objective: not "how likely is this candidate to recover" but
     "how many rupees do we expect this candidate to recover." This is what
     the hackathon track actually cares about, and is this module's PRIMARY
     ranking target (brief section 2/5).

IMPORTANT -- why (B) and (C) are legitimate SYNTHETIC BENCHMARK TARGETS but
must never reach a production feature list:
`recovery_probability_latent` is the hidden mechanism this project's own
data generator (data/generate_counterfactual_dataset.py) used to draw (A).
Training against it here is training against a KNOWN, SELF-AUTHORED ground
truth for benchmarking purposes -- legitimate for measuring "how good could
a candidate-ranking model be, in principle, on this synthetic environment,"
exactly as an oracle upper bound already does. It is NOT a claim
that recovery_probability_latent is observable, computable, or available in
any real Razorpay deployment -- a production system has no such column and
would need an equivalent OBSERVABLE target built from historical retries
and their outcomes (see README §16 for the explicit statement of this).
It must never appear as a model INPUT feature (that would be the hidden
archetype leaking through a back door) -- see EXCLUDED_COLUMNS below, which
documents both new latent columns as excluded for exactly this reason.
"""
from __future__ import annotations

import pandas as pd

from model.ranking_preprocessing import (  # noqa: F401 -- re-exported for train_latent_target_model.py
    ALL_BOOLEAN_FEATURES,
    ALL_CATEGORICAL_FEATURES,
    ALL_NUMERIC_FEATURES,
    FEATURE_COLUMNS,
    PriorSelfResolvedImputer,
    PROJECT_ROOT,
    SEED,
    build_candidate_level_dataset,
    split_candidate_dataset,
)
from model.ranking_preprocessing import EXCLUDED_COLUMNS as _RANKING_MODEL_EXCLUDED_COLUMNS

RECOVERY_HORIZON_DAYS = 14

# Column (A): the noisy observed outcome the candidate-aware and ranking models trained against.
OBSERVED_OUTCOME_COLUMN = "recovered_within_14d"
# Column (B): synthetic ground-truth latent probability -- Model A's target.
LATENT_PROBABILITY_COLUMN = "recovery_probability_latent"
# Column (C), derived: synthetic ground-truth latent expected value -- Model B's target, and the brief's PRIMARY ranking target.
LATENT_VALUE_COLUMN = "expected_recovery_value_latent"
LATENT_RATE_COLUMN = "expected_recovery_rate_latent"  # explicit alias of (B), per brief section 2

TARGET_COLUMNS = {"probability": LATENT_PROBABILITY_COLUMN, "value": LATENT_VALUE_COLUMN}

# Both new latent columns, plus the rate alias, are excluded from
# FEATURE_COLUMNS exactly like recovery_probability_latent already was in
# the candidate-aware/ranking models -- see model/candidate_preprocessing.py::EXCLUDED_COLUMNS.
EXCLUDED_COLUMNS = dict(_RANKING_MODEL_EXCLUDED_COLUMNS)
EXCLUDED_COLUMNS.update(
    {
        LATENT_VALUE_COLUMN: (
            "SYNTHETIC BENCHMARK TARGET (Model B's y) -- recovery_probability_latent * amount. "
            "Ground truth only because this project's own generator authored it; never a production feature."
        ),
        LATENT_RATE_COLUMN: (
            "explicit alias of recovery_probability_latent (Model A's y) -- same exclusion reasoning; "
            "kept as a separate documented name per brief section 2, not a new independent quantity."
        ),
    }
)
assert LATENT_PROBABILITY_COLUMN in EXCLUDED_COLUMNS  # inherited from the candidate-aware/ranking models -- re-asserted here so this module fails loudly if that ever changes


def add_latent_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Adds columns (C) and the (B) alias to a candidate-level DataFrame
    that already has `recovery_probability_latent` and `amount` (i.e. the
    output of model.candidate_preprocessing.build_candidate_level_dataset()).
    Pure, deterministic, no fitting."""
    out = df.copy()
    out[LATENT_RATE_COLUMN] = out[LATENT_PROBABILITY_COLUMN]
    out[LATENT_VALUE_COLUMN] = out[LATENT_PROBABILITY_COLUMN] * out["amount"]
    return out


def validate_latent_targets(df: pd.DataFrame) -> list[str]:
    """Sanity checks (brief section 10). Returns a list of issue strings;
    empty means everything checked out."""
    issues: list[str] = []

    if not df[LATENT_PROBABILITY_COLUMN].between(0.0, 1.0).all():
        issues.append(f"{LATENT_PROBABILITY_COLUMN} outside [0, 1]")

    recomputed = df[LATENT_PROBABILITY_COLUMN] * df["amount"]
    if not (df[LATENT_VALUE_COLUMN] - recomputed).abs().max() < 1e-6:
        issues.append(f"{LATENT_VALUE_COLUMN} != {LATENT_PROBABILITY_COLUMN} * amount for at least one row")

    if not (df[LATENT_RATE_COLUMN] == df[LATENT_PROBABILITY_COLUMN]).all():
        issues.append(f"{LATENT_RATE_COLUMN} is not an exact alias of {LATENT_PROBABILITY_COLUMN}")

    if (df[LATENT_VALUE_COLUMN] < 0).any():
        issues.append(f"{LATENT_VALUE_COLUMN} is negative for at least one row")

    if (df[LATENT_VALUE_COLUMN] > df["amount"] + 1e-6).any():
        issues.append(f"{LATENT_VALUE_COLUMN} exceeds the original amount for at least one row")

    return issues


def build_candidate_level_dataset_with_latent_targets() -> pd.DataFrame:
    return add_latent_targets(build_candidate_level_dataset())


def select_features_and_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """target: 'probability' -> Model A's y (recovery_probability_latent);
    'value' -> Model B's y (expected_recovery_value_latent). Selecting
    FEATURE_COLUMNS here is identical to the ranking model's -- the feature set is
    unchanged, only the target differs. Boolean features are cast to 0/1
    int HERE (once), so downstream callers never need to re-detect them."""
    if target not in TARGET_COLUMNS:
        raise ValueError(f"target must be one of {list(TARGET_COLUMNS)}, got {target!r}")
    X = df[FEATURE_COLUMNS].copy()
    for col in ALL_BOOLEAN_FEATURES:
        X[col] = X[col].astype(int)
    y = df[TARGET_COLUMNS[target]].astype(float)
    return X, y


def prepare_for_catboost(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in ALL_CATEGORICAL_FEATURES:
        out[col] = out[col].astype(str).astype(object)
    return out
