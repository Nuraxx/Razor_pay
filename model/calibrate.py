"""
Day-4 probability calibration.

Both sigmoid (Platt) and isotonic calibrators are fit -- each using ONLY the
validation split, via an already-trained (train-only) CatBoost model frozen
with sklearn's FrozenEstimator so CalibratedClassifierCV cannot accidentally
re-fit the base model. Neither calibrator ever sees train or test data.

Method choice is not automatic: `isotonic_is_defensible()` encodes the
rule (validation set must have enough points to fit a free-form step
function without just memorizing validation noise) and train.py logs the
result. Both calibrators are still saved and both are evaluated on the test
set in evaluation/evaluate_models.py -- the choice is explained, not hidden.
"""
from __future__ import annotations

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

# Conservative rule of thumb: isotonic regression fits an unconstrained
# step function, so it needs enough validation points per step to reflect
# real structure rather than sampling noise. There's no universal constant
# for this; 200 is a common, conservative floor for binary-outcome
# calibration sets in small-data settings like this one.
MIN_VALIDATION_SIZE_FOR_ISOTONIC = 200


def fit_calibration(model, X_val: pd.DataFrame, y_val: pd.Series, method: str) -> CalibratedClassifierCV:
    """Calibrate an already-trained `model` using validation data only. method: 'sigmoid' | 'isotonic'."""
    frozen = FrozenEstimator(model)
    calibrator = CalibratedClassifierCV(estimator=frozen, method=method)
    calibrator.fit(X_val, y_val)
    return calibrator


def isotonic_is_defensible(n_validation: int) -> bool:
    return n_validation >= MIN_VALIDATION_SIZE_FOR_ISOTONIC


def recommended_calibration_method(n_validation: int) -> str:
    return "isotonic" if isotonic_is_defensible(n_validation) else "sigmoid"
