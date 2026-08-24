"""
Day-5 candidate scoring.

IMPORTANT MODELING HONESTY NOTE (see also README "Day 5" and section 12 of
the Day-5 brief this module implements):

Day 4's calibrated CatBoost model was trained ONLY on failure-time/customer
features (model/preprocessing.py::FEATURE_COLUMNS) -- it has never seen a
candidate retry time as an input. Day 3's synthetic dataset records exactly
ONE observed outcome per failure event, not what would have happened under
each of the 5 candidate retry times -- there is no genuine counterfactual
label to train a true candidate-aware model against, and fabricating one
(e.g. by pretending the single observed outcome "belongs" to whichever
candidate its timestamp happens to be closest to) would misrepresent
correlation for a candidate we didn't actually simulate as causal evidence
for it. So this module does NOT retrain or repurpose the Day-4 model to
"predict conditional on candidate time" -- that would be Option A from the
brief, and it isn't defensible with this dataset.

Instead (Option B): `base_probability` is Day-4's calibrated model output,
unchanged, representing "how likely is this failure to recover at all." A
small, fixed, fully transparent HEURISTIC -- not a learned effect -- nudges
that probability up or down per candidate based on payday proximity and
whether the candidate is an immediate retry. It is a policy rule, sized by
domain reasoning matching Day 3's own generator assumptions (funds are more
likely available near a payday window; an insufficient-funds failure is
unlikely to self-resolve within the same hour), not a validated model
prediction. Every function below documents this distinction in its
docstring so it's never confused with `base_probability`.

DAY-6 UPDATE: the objection above is now fixed. `data/generate_counterfactual_dataset.py`
generates a genuine simulated outcome for every (event, candidate) pair, so
Option A -- a model trained directly on candidate-time features -- is now
defensible; see `model/train_candidate_model.py` and
`model/candidate_preprocessing.py`. `load_candidate_aware_model()`,
`predict_candidate_aware_recovery_probability()`, and
`score_candidate_with_model_probability()` below are that new path, used by
`policy/recovery_policy.py::decide_candidate_aware`. The Day-5 heuristic
path above (`score_candidate`, `heuristic_adjustment`,
`load_calibrated_model`) is left completely unchanged and still works --
it remains the honest choice for any context where only failure-time
features (no genuine candidate-level counterfactual data) are available.
"""
from __future__ import annotations

import joblib
import pandas as pd
from catboost import CatBoostClassifier

from model.preprocessing import PROJECT_ROOT, prepare_for_catboost
from policy.retry_candidates import Candidate

ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"

# Heuristic adjustment constants -- deliberately small and bounded so the
# heuristic can only nudge the model's probability, never dominate it.
IMMEDIATE_RETRY_PENALTY = 0.05  # an insufficient_fund failure is unlikely to self-resolve within the hour
PAYDAY_PROXIMITY_MAX_BOOST = 0.08  # bounded; scaled linearly by closeness to the nearest payday window
PAYDAY_PROXIMITY_HORIZON_DAYS = 7  # beyond this many days from a payday window, no boost is applied


class Day4ModelUnavailable(RuntimeError):
    """Raised when model/artifacts/ doesn't exist yet -- run model/train.py first."""


def load_calibrated_model() -> tuple[CatBoostClassifier, object]:
    """Loads Day-4's sigmoid-calibrated CatBoost model and the train-fit imputer it needs upstream of it."""
    model_path = ARTIFACTS_DIR / "catboost_calibrated_sigmoid.joblib"
    imputer_path = ARTIFACTS_DIR / "prior_self_resolved_imputer.joblib"
    if not model_path.exists() or not imputer_path.exists():
        raise Day4ModelUnavailable(
            f"Day-4 model artifacts not found in {ARTIFACTS_DIR}. Run `./venv/bin/python model/train.py` first."
        )
    return joblib.load(model_path), joblib.load(imputer_path)


def predict_base_recovery_probability(failure_time_features: pd.DataFrame, model, imputer) -> "pd.Series[float]":
    """
    `base_probability` = Day-4's calibrated model's prediction, using ONLY
    the failure-time/customer features it was trained on. No candidate-time
    information is or can be passed in here -- see module docstring.
    """
    X_imputed = imputer.transform(failure_time_features)
    X_cb = prepare_for_catboost(X_imputed)
    return pd.Series(model.predict_proba(X_cb)[:, 1], index=failure_time_features.index)


def heuristic_adjustment(candidate_type: str, candidate_days_to_payday: int) -> float:
    """
    A documented POLICY HEURISTIC, not a learned/validated effect (see module
    docstring). Returns a signed adjustment to be added to base_probability,
    bounded to [-IMMEDIATE_RETRY_PENALTY, +PAYDAY_PROXIMITY_MAX_BOOST].
    """
    adjustment = 0.0
    if candidate_type == "immediate":
        adjustment -= IMMEDIATE_RETRY_PENALTY
    proximity_fraction = max(
        0.0,
        (PAYDAY_PROXIMITY_HORIZON_DAYS - min(candidate_days_to_payday, PAYDAY_PROXIMITY_HORIZON_DAYS))
        / PAYDAY_PROXIMITY_HORIZON_DAYS,
    )
    adjustment += PAYDAY_PROXIMITY_MAX_BOOST * proximity_fraction
    return adjustment


CANDIDATE_ARTIFACTS_DIR = PROJECT_ROOT / "model" / "candidate_artifacts"


def load_candidate_aware_model() -> tuple[CatBoostClassifier, object]:
    """
    Day-6: loads the candidate-aware calibrated CatBoost model
    (model/train_candidate_model.py) -- trained directly on candidate-time
    features (see model/candidate_preprocessing.py), unlike Day-5's
    `load_calibrated_model()` above which only ever saw failure-time
    features. Its output is a genuine per-candidate probability, not a
    heuristic adjustment of one shared failure-time probability.
    """
    model_path = CANDIDATE_ARTIFACTS_DIR / "catboost_calibrated_sigmoid.joblib"
    imputer_path = CANDIDATE_ARTIFACTS_DIR / "prior_self_resolved_imputer.joblib"
    if not model_path.exists() or not imputer_path.exists():
        raise Day4ModelUnavailable(
            f"Day-6 candidate-aware model artifacts not found in {CANDIDATE_ARTIFACTS_DIR}. "
            "Run `./venv/bin/python model/train_candidate_model.py` first."
        )
    return joblib.load(model_path), joblib.load(imputer_path)


def predict_candidate_aware_recovery_probability(candidate_features: pd.DataFrame, model, imputer) -> "pd.Series[float]":
    """
    Day-6 equivalent of `predict_base_recovery_probability` above, except
    `candidate_features` is a candidate-LEVEL row (model/candidate_preprocessing.py::FEATURE_COLUMNS)
    -- failure-time features AND candidate-time features together -- so the
    returned probability is already conditioned on a specific candidate
    retry time. No heuristic_adjustment is layered on top; see
    policy/recovery_policy.py::decide_candidate_aware.
    """
    from model.candidate_preprocessing import prepare_for_catboost as prepare_candidates_for_catboost

    X_imputed = imputer.transform(candidate_features)
    X_cb = prepare_candidates_for_catboost(X_imputed)
    return pd.Series(model.predict_proba(X_cb)[:, 1], index=candidate_features.index)


def score_candidate_with_model_probability(predicted_recovery_probability: float, candidate: Candidate, amount: float, intervention_cost: float = 0.0) -> dict:
    """
    Day-6: the candidate-aware model already outputs a probability
    conditioned on THIS candidate's timing -- unlike `score_candidate` below
    (Day 5), there is no separate heuristic_adjustment layered on top.
    `predicted_recovery_probability` is clipped to [0, 1] defensively (model
    output should already be in range; see tests/test_candidate_model.py).

    expected_recovery_value = predicted_recovery_probability * amount
    expected_incremental_value = expected_recovery_value - intervention_cost
    """
    clipped = min(1.0, max(0.0, predicted_recovery_probability))
    expected_recovery_value = clipped * amount
    expected_incremental_value = expected_recovery_value - intervention_cost
    return {
        "candidate_type": candidate.candidate_type,
        "candidate_datetime": candidate.candidate_datetime,
        "predicted_recovery_probability": clipped,
        "expected_recovery_value": expected_recovery_value,
        "expected_incremental_value": expected_incremental_value,
    }


def score_candidate(base_probability: float, candidate: Candidate, amount: float, intervention_cost: float = 0.0) -> dict:
    """
    expected_recovery_value = predicted_recovery_probability * amount
    expected_incremental_value = expected_recovery_value - intervention_cost

    `intervention_cost` defaults to 0 (Day 5 doesn't model a real cost yet)
    but the parameter exists so a later day can supply a non-zero one
    without changing this function's shape.

    This is an ESTIMATE, not a claim about actual money recovered -- see
    README "Day 5: expected recovery value".
    """
    adjustment = heuristic_adjustment(candidate.candidate_type, candidate.candidate_days_to_payday)
    predicted_recovery_probability = min(1.0, max(0.0, base_probability + adjustment))
    expected_recovery_value = predicted_recovery_probability * amount
    expected_incremental_value = expected_recovery_value - intervention_cost

    return {
        "candidate_type": candidate.candidate_type,
        "candidate_datetime": candidate.candidate_datetime,
        "base_probability": base_probability,
        "heuristic_adjustment": adjustment,
        "predicted_recovery_probability": predicted_recovery_probability,
        "expected_recovery_value": expected_recovery_value,
        "expected_incremental_value": expected_incremental_value,
    }
