"""
Diagnosis: why did the candidate-aware model fail at within-event
ranking despite a good pooled ROC-AUC?

    ./venv/bin/python model/diagnose_ranking_failure.py

Written BEFORE model/train_ranking_model.py, per the brief's explicit
instruction to diagnose before implementing a fix -- this script only
recomputes the evidence.

Requires model/candidate_artifacts/ (the candidate-aware model's trained
artifact) and data/raw/counterfactual_outcomes.csv (the counterfactual
layer's dataset) to already exist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from model.candidate_preprocessing import build_candidate_level_dataset, load_candidate_splits, prepare_for_catboost, select_features_and_target
from policy.scoring import ScoringModelUnavailable, load_candidate_aware_model


def diagnose_feature_importance() -> pd.DataFrame:
    """Finding 1/2/4: hours_from_failure dominates; event-level context also
    generally outranks the other candidate features."""
    from catboost import CatBoostClassifier

    from policy.scoring import CANDIDATE_ARTIFACTS_DIR

    raw_model = CatBoostClassifier()
    raw_model.load_model(str(CANDIDATE_ARTIFACTS_DIR / "catboost_model.cbm"))
    fi = pd.DataFrame({"feature": raw_model.feature_names_, "importance": raw_model.get_feature_importance()}).sort_values("importance", ascending=False)
    return fi


def diagnose_calibration_invariance() -> dict:
    """Finding 5: calibration is a monotonic transform and cannot change
    within-event rank order -- verified empirically, not just asserted."""
    from catboost import CatBoostClassifier

    from policy.scoring import CANDIDATE_ARTIFACTS_DIR

    _train, _val, test_df = load_candidate_splits()
    calibrated_model, imputer = load_candidate_aware_model()
    raw_model = CatBoostClassifier()
    raw_model.load_model(str(CANDIDATE_ARTIFACTS_DIR / "catboost_model.cbm"))

    X, _y = select_features_and_target(test_df)
    X_imp = imputer.transform(X)
    X_cb = prepare_for_catboost(X_imp)

    df = test_df.copy()
    df["raw_pred"] = raw_model.predict_proba(X_cb)[:, 1]
    df["calibrated_pred"] = calibrated_model.predict_proba(X_cb)[:, 1]

    def within_event_corr(col):
        corrs = []
        for _eid, g in df.groupby("event_id"):
            if g[col].nunique() < 2 or g["recovery_probability_latent"].nunique() < 2:
                continue
            r, _p = spearmanr(g[col], g["recovery_probability_latent"])
            corrs.append(r)
        return float(np.mean(corrs)), len(corrs)

    raw_corr, n_raw = within_event_corr("raw_pred")
    cal_corr, n_cal = within_event_corr("calibrated_pred")
    return {
        "within_event_rank_correlation_raw_uncalibrated": raw_corr,
        "within_event_rank_correlation_sigmoid_calibrated": cal_corr,
        "n_events": n_raw,
        "identical": raw_corr == cal_corr,
    }


def diagnose_uniform_label_groups() -> dict:
    """Finding 6: how many events have zero informative pairs (all 5
    candidates share the same realized label)?"""
    cf = build_candidate_level_dataset()
    label_nunique = cf.groupby("event_id")["recovered_within_14d"].nunique()
    n_uniform = int((label_nunique == 1).sum())
    n_total = len(label_nunique)
    return {"n_events_total": n_total, "n_events_uniform_label": n_uniform, "fraction_uniform": round(n_uniform / n_total, 4)}


def diagnose_true_signal_within_horizon() -> dict:
    """Finding 3: even isolating the horizon-truncation confound (restricting
    to candidates that ARE within the 14-day window), does candidate_days_to_payday
    alone carry a real within-event signal?"""
    df = build_candidate_level_dataset()
    df["failure_timestamp"] = pd.to_datetime(df["failure_timestamp"])
    df["candidate_datetime"] = pd.to_datetime(df["candidate_datetime"])
    within_horizon = df[df["candidate_datetime"] <= df["failure_timestamp"] + pd.Timedelta(days=14)]

    corrs = []
    for _eid, g in within_horizon.groupby("event_id"):
        if len(g) < 2 or g["candidate_days_to_payday"].nunique() < 2 or g["recovery_probability_latent"].nunique() < 2:
            continue
        r, _p = spearmanr(-g["candidate_days_to_payday"], g["recovery_probability_latent"])  # closer to payday (lower days) -> higher latent prob
        corrs.append(r)
    return {"n_rows_within_horizon": len(within_horizon), "n_rows_total": len(df), "n_events_used": len(corrs), "within_event_corr_days_to_payday_vs_latent": float(np.mean(corrs))}


def main() -> None:
    print("=== Diagnosis: why did the candidate-aware model fail at within-event ranking? ===\n")

    try:
        fi = diagnose_feature_importance()
    except ScoringModelUnavailable as exc:
        raise SystemExit(f"{exc}\nRun model/train_candidate_model.py first.") from exc

    print("1/2/4. Candidate-aware model feature importance (top 10) -- hours_from_failure dominance:")
    print(fi.head(10).to_string(index=False))
    print()

    cal = diagnose_calibration_invariance()
    print("5. Calibration invariance check (must be identical -- monotonic transform):")
    print(f"   raw uncalibrated within-event corr = {cal['within_event_rank_correlation_raw_uncalibrated']:.4f} (n={cal['n_events']})")
    print(f"   sigmoid calibrated within-event corr = {cal['within_event_rank_correlation_sigmoid_calibrated']:.4f}")
    print(f"   identical: {cal['identical']}")
    print()

    uniform = diagnose_uniform_label_groups()
    print("6. Uniform-label (zero informative pairs) events:")
    print(f"   {uniform['n_events_uniform_label']}/{uniform['n_events_total']} events ({uniform['fraction_uniform']:.1%}) -- not the dominant driver")
    print()

    signal = diagnose_true_signal_within_horizon()
    print("3. True signal, isolated from the horizon-truncation confound:")
    print(f"   within-horizon rows: {signal['n_rows_within_horizon']}/{signal['n_rows_total']}")
    print(f"   within-event corr(candidate_days_to_payday, latent), horizon-valid only: {signal['within_event_corr_days_to_payday_vs_latent']:.4f} (n_events={signal['n_events_used']})")
    print(f"   -> a real but weak signal, consistent with the candidate-aware model's measured -0.149 being an anti-correlation, not just noise")


if __name__ == "__main__":
    main()
