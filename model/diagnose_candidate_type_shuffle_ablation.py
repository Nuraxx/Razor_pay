"""
CRITICAL ABLATION: does the candidate-aware model (model/train_candidate_model.py)
genuinely learn a candidate_type -> recovery relationship, or does it reach its
pooled ROC-AUC purely from event-level context features (duplicated identically
across an event's 5 candidate rows) plus the already-attached candidate-timing
numeric features (hours_from_failure, candidate_days_to_payday, etc.), with the
`candidate_type` categorical LABEL itself contributing nothing?

    ./venv/bin/python model/diagnose_candidate_type_shuffle_ablation.py

METHOD: within each TRAINING event, randomly permute the `candidate_type`
labels among that event's 5 rows (seeded, deterministic) -- every other
column (event context, the candidate-timing numeric/boolean features
`hours_from_failure` / `candidate_days_to_payday` / `candidate_is_*_aligned`,
and the target `recovered_within_14d`) stays attached to its original row,
unchanged. This isolates exactly one question: does the `candidate_type`
categorical label carry independent signal beyond what the already-present
candidate-timing features encode? Retrain the EXACT SAME `fit_pipeline()`
from model/train_candidate_model.py, unmodified, on this shuffled training
set; evaluate the resulting model on the REAL (unshuffled) validation set,
side-by-side with the real-trained model, using the same within-event top-1
selection / rank-correlation-vs-latent-truth methodology
evaluation/evaluate_counterfactual_policy.py already uses.

If the shuffled-trained model performs approximately as well as the
real-trained model, `candidate_type` is not contributing genuine, learnable
signal in this architecture and the candidate-aware reformulation must not
be trusted on that basis alone -- consistent with the pre-existing finding
in model/diagnose_ranking_failure.py (hours_from_failure captured 93.3% of
feature importance; candidate_days_to_payday only 0.03%).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from model.candidate_preprocessing import load_candidate_splits, prepare_for_catboost, select_features_and_target
from model.train_candidate_model import fit_pipeline

# Distinct from the model's own training SEED (42, model/preprocessing.py) --
# this is the ablation's own shuffle randomness, documented here so a re-run
# reproduces byte-identical shuffled assignments.
SHUFFLE_SEED = 20260828


def shuffle_candidate_type_within_event(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permutes `candidate_type` among each event's rows independently, in
    place of nothing else -- every other column (including the numeric/boolean
    candidate-timing features and the target) stays attached to its original
    row. Returns a new DataFrame; input is not mutated."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for _event_id, group in out.groupby("event_id"):
        idx = group.index.to_numpy()
        out.loc[idx, "candidate_type"] = rng.permutation(group["candidate_type"].to_numpy())
    return out


def top1_accuracy_and_rank_corr(model, imputer, df: pd.DataFrame) -> dict:
    """Within-event top-1 selection accuracy (predicted argmax vs latent-truth
    argmax) and mean within-event Spearman rank correlation (predicted vs
    latent probability) -- same methodology as
    evaluation/evaluate_counterfactual_policy.py's within_event_rank_correlation
    / top1_match, applied directly to a candidate-level DataFrame."""
    X, _y = select_features_and_target(df)
    X_imp = imputer.transform(X)
    X_cb = prepare_for_catboost(X_imp)
    preds = model.predict_proba(X_cb)[:, 1]

    d = df.copy()
    d["_pred"] = preds

    hits, n_top1 = 0, 0
    corrs = []
    for _eid, g in d.groupby("event_id"):
        pred_best = g.loc[g["_pred"].idxmax(), "candidate_type"]
        latent_best = g.loc[g["recovery_probability_latent"].idxmax(), "candidate_type"]
        hits += int(pred_best == latent_best)
        n_top1 += 1
        if g["_pred"].nunique() >= 2 and g["recovery_probability_latent"].nunique() >= 2:
            corr, _p = spearmanr(g["_pred"], g["recovery_probability_latent"])
            corrs.append(corr)

    return {
        "top1_accuracy_vs_latent_truth": hits / n_top1 if n_top1 else None,
        "n_events": n_top1,
        "mean_within_event_rank_correlation": float(np.mean(corrs)) if corrs else None,
        "n_events_with_rank_correlation": len(corrs),
    }


def realized_money_for_top1(model, imputer, df: pd.DataFrame) -> dict:
    """Top-1-only realized-money diagnostic (no guardrails/costs -- this is an
    ablation signal check, not a policy evaluation): for each event, sum
    amount_recovered for whichever candidate the model's predicted probability
    ranks highest."""
    X, _y = select_features_and_target(df)
    X_imp = imputer.transform(X)
    X_cb = prepare_for_catboost(X_imp)
    preds = model.predict_proba(X_cb)[:, 1]

    d = df.copy()
    d["_pred"] = preds

    total_recovered = 0.0
    n_recovered = 0
    n_events = 0
    for _eid, g in d.groupby("event_id"):
        row = g.loc[g["_pred"].idxmax()]
        total_recovered += float(row["amount_recovered"])
        n_recovered += int(bool(row["recovered_within_14d"]))
        n_events += 1

    return {
        "total_amount_recovered_rs": total_recovered,
        "recovery_rate": n_recovered / n_events if n_events else None,
        "n_events": n_events,
    }


def main() -> None:
    train_df, val_df, _test_df_unused = load_candidate_splits()
    del _test_df_unused  # test is never touched by this ablation either

    print("=== CRITICAL ABLATION: candidate_type shuffle ===\n")
    print(f"Train events: {train_df['event_id'].nunique()} ({len(train_df)} rows) | Validation events: {val_df['event_id'].nunique()} ({len(val_df)} rows)\n")

    print("--- Training REAL model (unchanged candidate_type) ---")
    real_fitted = fit_pipeline(train_df, val_df)
    real_model, real_imputer = real_fitted["catboost_model"], real_fitted["imputer"]
    print(f"validation AUC (pooled): {real_fitted['model_config']['validation_auc']['catboost_uncalibrated']:.4f}\n")

    shuffled_train_df = shuffle_candidate_type_within_event(train_df, seed=SHUFFLE_SEED)
    real_counts = train_df["candidate_type"].value_counts().sort_index()
    shuffled_counts = shuffled_train_df["candidate_type"].value_counts().sort_index()
    assert real_counts.equals(shuffled_counts), "shuffle must preserve the marginal candidate_type distribution"
    n_changed = int((shuffled_train_df["candidate_type"].to_numpy() != train_df["candidate_type"].to_numpy()).sum())
    print(f"--- Training SHUFFLED model (candidate_type permuted within each of {train_df['event_id'].nunique()} training events; {n_changed}/{len(train_df)} rows actually reassigned) ---")
    shuffled_fitted = fit_pipeline(shuffled_train_df, val_df)
    shuffled_model, shuffled_imputer = shuffled_fitted["catboost_model"], shuffled_fitted["imputer"]
    print(f"validation AUC (pooled): {shuffled_fitted['model_config']['validation_auc']['catboost_uncalibrated']:.4f}\n")

    real_ranking = top1_accuracy_and_rank_corr(real_model, real_imputer, val_df)
    shuffled_ranking = top1_accuracy_and_rank_corr(shuffled_model, shuffled_imputer, val_df)
    real_money = realized_money_for_top1(real_model, real_imputer, val_df)
    shuffled_money = realized_money_for_top1(shuffled_model, shuffled_imputer, val_df)

    print("=== RESULTS (both models evaluated on the REAL, unshuffled VALIDATION set) ===")
    print(f"{'metric':45s} {'real':>12s} {'shuffled':>12s}")
    print(f"{'top1_accuracy_vs_latent_truth':45s} {real_ranking['top1_accuracy_vs_latent_truth']:>12.4f} {shuffled_ranking['top1_accuracy_vs_latent_truth']:>12.4f}")
    print(f"{'mean_within_event_rank_correlation':45s} {real_ranking['mean_within_event_rank_correlation']:>12.4f} {shuffled_ranking['mean_within_event_rank_correlation']:>12.4f}")
    print(f"{'top1_realized_amount_recovered_rs':45s} {real_money['total_amount_recovered_rs']:>12.2f} {shuffled_money['total_amount_recovered_rs']:>12.2f}")
    print(f"{'top1_realized_recovery_rate':45s} {real_money['recovery_rate']:>12.4f} {shuffled_money['recovery_rate']:>12.4f}")
    print()

    top1_gap = real_ranking["top1_accuracy_vs_latent_truth"] - shuffled_ranking["top1_accuracy_vs_latent_truth"]
    money_gap = real_money["total_amount_recovered_rs"] - shuffled_money["total_amount_recovered_rs"]
    print(f"top1_accuracy gap (real - shuffled): {top1_gap:+.4f}")
    print(f"realized money gap (real - shuffled): Rs{money_gap:+.2f}")
    if abs(top1_gap) < 0.03 and abs(money_gap) < 0.02 * abs(real_money["total_amount_recovered_rs"]):
        print(
            "\nVERDICT: candidate_type shuffle does NOT materially degrade performance -- "
            "candidate_type is not carrying genuine, learnable within-event signal in this "
            "architecture. DO NOT trust the candidate-aware improvement on this basis."
        )
    else:
        print("\nVERDICT: candidate_type shuffle DOES materially degrade performance -- candidate_type appears to carry genuine signal.")


if __name__ == "__main__":
    main()
