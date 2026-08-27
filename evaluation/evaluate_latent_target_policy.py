"""
Latent-target model + policy evaluation.

    ./venv/bin/python evaluation/evaluate_latent_target_policy.py

"SYNTHETIC COUNTERFACTUAL EVALUATION" -- same disclaimer as the earlier
candidate-aware/ranking model evaluations: every number here comes from
data/raw/counterfactual_outcomes.csv, a hand-designed simulation. It does
not measure real Razorpay recovery performance. `recovery_probability_latent`
/ `expected_recovery_value_latent` are SYNTHETIC BENCHMARK TARGETS this
project's own generator authored -- see model/latent_target_preprocessing.py's
module docstring for the full A/B/C distinction. A production system has no
such column.

Compares 8 policies (brief section 7, nothing from the earlier model
evaluations deleted): Random, Fixed Retry, Rule-Based, the candidate-aware
probability model, the pairwise ranking model, Model A (predicts
recovery_probability_latent), Model B (predicts expected_recovery_value_latent,
the brief's PRIMARY target), Oracle.

Four report sections (brief section 8), kept explicitly separate:
  A. Regression metrics (Model A/B only, vs. their own synthetic target)
  B. Ranking metrics (all policies, PRIMARY ground truth = expected_recovery_value_latent)
  C. Economic metrics (latent, synthetic -- "what the generator's own ground truth says")
  D. Realized counterfactual metrics (the stochastic sampled outcome -- what the earlier model evaluations already reported)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from classification.rules import classify
from evaluation.ranking_metrics import ndcg_at_5, pairwise_concordant_pairs, reciprocal_rank, regret, top_k_accuracy, within_event_rank
from model.candidate_preprocessing import prepare_for_catboost as candidate_model_prepare_cb
from model.candidate_preprocessing import select_features_and_target as select_candidate_probability_features
from model.latent_target_preprocessing import (
    LATENT_PROBABILITY_COLUMN,
    LATENT_VALUE_COLUMN,
    PROJECT_ROOT,
    build_candidate_level_dataset_with_latent_targets,
    prepare_for_catboost,
    select_features_and_target,
    split_candidate_dataset,
)
from model.train_latent_target_model import load_latent_target_model, regression_metrics
from model.train_ranking_model import load_ranking_model, predict_ranking_scores
from policy.baselines import fixed_retry_baseline, rule_based_baseline
from policy.guardrails import validate_candidate
from policy.recovery_policy import NO_ACTION, decide_candidate_aware
from policy.retry_candidates import CANDIDATE_TYPES, Candidate
from policy.scoring import load_candidate_aware_model, predict_candidate_aware_recovery_probability

REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"
RANDOM_SEED = 42
POLICY_NAMES = [
    "random_candidate",
    "fixed_retry",
    "rule_based",
    "candidate_probability_model",
    "ranking_model",
    "model_a_probability",
    "model_b_value",
    "oracle_policy",
]


# ---------------------------------------------------------------------------
# Section A: regression metrics for Model A / Model B
# ---------------------------------------------------------------------------

def compute_regression_metrics(test_df: pd.DataFrame) -> dict:
    out = {}
    for target, label in (("probability", LATENT_PROBABILITY_COLUMN), ("value", LATENT_VALUE_COLUMN)):
        fitted = load_latent_target_model(target)
        X, y = select_features_and_target(test_df, target)
        X_imp = fitted["imputer"].transform(X)
        X_cb = prepare_for_catboost(X_imp)
        preds = fitted["catboost_model"].predict(X_cb)
        out[f"model_{'a' if target == 'probability' else 'b'}_{target}"] = {
            "target_column": label,
            **regression_metrics(y.to_numpy(), preds),
            "predictions_finite": bool(np.isfinite(preds).all()),
            "predictions_non_negative": bool((preds >= -1e-6).all()),
        }
    return out


# ---------------------------------------------------------------------------
# Score computation for every policy
# ---------------------------------------------------------------------------

def compute_all_scores(test_df: pd.DataFrame) -> dict[str, pd.Series]:
    candidate_model, candidate_model_imputer = load_candidate_aware_model()
    X6, _y = select_candidate_probability_features(test_df)
    X6_imp = candidate_model_imputer.transform(X6)
    candidate_probability_scores = predict_candidate_aware_recovery_probability(X6_imp, candidate_model, candidate_model_imputer)

    ranking_model_fitted = load_ranking_model()
    ranking_model_scores = predict_ranking_scores(test_df, ranking_model_fitted)

    model_a = load_latent_target_model("probability")
    Xa, _ya = select_features_and_target(test_df, "probability")
    Xa_imp = model_a["imputer"].transform(Xa)
    Xa_cb = prepare_for_catboost(Xa_imp)
    model_a_scores = pd.Series(model_a["catboost_model"].predict(Xa_cb), index=test_df.index)

    model_b = load_latent_target_model("value")
    Xb, _yb = select_features_and_target(test_df, "value")
    Xb_imp = model_b["imputer"].transform(Xb)
    Xb_cb = prepare_for_catboost(Xb_imp)
    model_b_value_scores = pd.Series(model_b["catboost_model"].predict(Xb_cb), index=test_df.index)
    # Model B predicts RUPEES directly. decide_candidate_aware()'s existing
    # architecture (score_candidate_with_model_probability) expects a
    # probability and multiplies by amount internally -- converting back via
    # /amount round-trips correctly (value = prob * amount => prob = value / amount)
    # and lets Model B plug into the SAME unmodified policy code Model A and
    # the candidate-aware/ranking models already use. Clipped to [0,1] defensively: a
    # regressor's raw output is not guaranteed to stay in range.
    model_b_prob_equivalent = (model_b_value_scores / test_df["amount"]).clip(0.0, 1.0)

    rng = np.random.default_rng(RANDOM_SEED)
    random_scores = pd.Series(rng.random(len(test_df)), index=test_df.index)

    return {
        "candidate_probability_model": candidate_probability_scores,
        "ranking_model": ranking_model_scores,
        "model_a_probability": model_a_scores.clip(0.0, 1.0),
        "model_b_value": model_b_prob_equivalent,
        "random_candidate": random_scores,
        "oracle_policy": test_df[LATENT_PROBABILITY_COLUMN],
    }


# ---------------------------------------------------------------------------
# Section B/C/D: per-event ranking + economic + realized evaluation
# ---------------------------------------------------------------------------

def _row_to_candidate(row: pd.Series) -> Candidate:
    return Candidate(
        candidate_type=row["candidate_type"],
        candidate_datetime=row["candidate_datetime"],
        hours_from_failure=row["hours_from_failure"],
        candidate_day_of_month=int(row["candidate_day_of_month"]),
        candidate_day_of_week=row["candidate_day_of_week"],
        candidate_is_payday_aligned=bool(row["candidate_is_payday_aligned"]),
        candidate_is_month_end_aligned=bool(row["candidate_is_month_end_aligned"]),
        candidate_days_to_payday=int(row["candidate_days_to_payday"]),
    )


def evaluate_events(test_df: pd.DataFrame, all_scores: dict[str, pd.Series]) -> pd.DataFrame:
    df = test_df.copy()
    for name, series in all_scores.items():
        df[f"_score_{name}"] = series

    records = []
    for event_id, group in df.groupby("event_id"):
        first = group.iloc[0]
        subscription_id = first["subscription_id"]
        failure_timestamp = first["failure_timestamp"]
        amount = float(first["amount"])
        classification_bucket = classify(None, first["error_reason"]).bucket

        candidate_types = group["candidate_type"].to_numpy()
        # PRIMARY ground truth (brief section 5): expected_recovery_value_latent, not the probability.
        truth_value = group[LATENT_VALUE_COLUMN].to_numpy()
        realized_recovered = dict(zip(group["candidate_type"], group["recovered_within_14d"]))
        realized_amount = dict(zip(group["candidate_type"], group["amount_recovered"]))
        latent_value = dict(zip(group["candidate_type"], group[LATENT_VALUE_COLUMN]))
        valid_mask = np.array([validate_candidate(_row_to_candidate(r), failure_timestamp)[0] for _, r in group.iterrows()])

        record = {"event_id": event_id, "subscription_id": subscription_id, "amount": amount, "classification_bucket": classification_bucket}

        fixed_selected = fixed_retry_baseline(event_id, subscription_id, failure_timestamp, amount, classification_bucket, 0.0)["selected_candidate_type"]
        rule_selected = rule_based_baseline(event_id, subscription_id, failure_timestamp, amount, classification_bucket, 0.0)["selected_candidate_type"]
        fixed_scores = np.array([1.0 if ct == fixed_selected else 0.0 for ct in candidate_types])
        rule_scores = np.array([1.0 if ct == rule_selected else 0.0 for ct in candidate_types])

        for policy_name in POLICY_NAMES:
            if policy_name == "fixed_retry":
                scores, selected = fixed_scores, fixed_selected
            elif policy_name == "rule_based":
                scores, selected = rule_scores, rule_selected
            else:
                scores = group[f"_score_{policy_name}"].to_numpy()
                probs = dict(zip(candidate_types, scores))
                selected = decide_candidate_aware(event_id, subscription_id, failure_timestamp, amount, classification_bucket, probs).selected_candidate_type

            record[f"{policy_name}__selected_candidate_type"] = selected
            record[f"{policy_name}__realized_recovered"] = bool(realized_recovered.get(selected, False)) if selected != NO_ACTION else False
            record[f"{policy_name}__realized_amount_recovered"] = float(realized_amount.get(selected, 0.0)) if selected != NO_ACTION else 0.0
            record[f"{policy_name}__latent_value_selected"] = float(latent_value.get(selected, 0.0)) if selected != NO_ACTION else 0.0

            if group["candidate_type"].nunique() >= 2:
                record[f"{policy_name}__top1"] = top_k_accuracy(scores, truth_value, candidate_types, 1)
                record[f"{policy_name}__top2"] = top_k_accuracy(scores, truth_value, candidate_types, 2)
                record[f"{policy_name}__rr"] = reciprocal_rank(scores, truth_value, candidate_types)
                record[f"{policy_name}__ndcg5"] = ndcg_at_5(scores, truth_value)
                record[f"{policy_name}__rank"] = within_event_rank(scores, truth_value, candidate_types)
                concordant, total = pairwise_concordant_pairs(scores, truth_value)
                record[f"{policy_name}__pairwise_concordant"] = concordant
                record[f"{policy_name}__pairwise_total"] = total
                record[f"{policy_name}__regret"] = regret(scores, truth_value, candidate_types, 1.0, valid_mask)  # truth is already in Rs, amount multiplier = 1
            else:
                for suffix in ("top1", "top2", "rr", "ndcg5", "rank", "pairwise_concordant", "pairwise_total", "regret"):
                    record[f"{policy_name}__{suffix}"] = None

        records.append(record)

    return pd.DataFrame(records)


def summarize_ranking(events: pd.DataFrame) -> dict:
    summary = {}
    for policy_name in POLICY_NAMES:
        top1 = events[f"{policy_name}__top1"].dropna()
        top2 = events[f"{policy_name}__top2"].dropna()
        rr = events[f"{policy_name}__rr"].dropna()
        ndcg5 = events[f"{policy_name}__ndcg5"].dropna()
        rank = events[f"{policy_name}__rank"].dropna()
        concordant = events[f"{policy_name}__pairwise_concordant"].dropna().sum()
        total = events[f"{policy_name}__pairwise_total"].dropna().sum()
        reg = events[f"{policy_name}__regret"].dropna()

        summary[policy_name] = {
            "n_events_with_variance": int(len(top1)),
            "top1_accuracy": float(top1.mean()) if len(top1) else None,
            "top2_accuracy": float(top2.mean()) if len(top2) else None,
            "mrr": float(rr.mean()) if len(rr) else None,
            "ndcg_at_5": float(ndcg5.mean()) if len(ndcg5) else None,
            "mean_within_event_rank": float(rank.mean()) if len(rank) else None,
            "pairwise_accuracy": float(concordant / total) if total > 0 else None,
            "avg_regret_rs": float(reg.mean()) if len(reg) else None,
            "sum_regret_rs": float(reg.sum()) if len(reg) else None,
        }
    return summary


def summarize_economic_metrics(events: pd.DataFrame) -> dict:
    """Section C: LATENT (synthetic ground-truth) economic value -- distinct
    from section D's realized/sampled outcome."""
    n_events = len(events)
    summary = {}
    for policy_name in POLICY_NAMES:
        col = f"{policy_name}__latent_value_selected"
        total = float(events[col].sum())
        summary[policy_name] = {
            "total_latent_expected_value_selected_rs": total,
            "average_latent_rs_per_failed_event": total / n_events if n_events else 0.0,
        }
    fixed_total = summary["fixed_retry"]["total_latent_expected_value_selected_rs"]
    oracle_total = summary["oracle_policy"]["total_latent_expected_value_selected_rs"]
    for policy_name in POLICY_NAMES:
        summary[policy_name]["improvement_vs_fixed_retry_rs"] = summary[policy_name]["total_latent_expected_value_selected_rs"] - fixed_total
        summary[policy_name]["regret_vs_oracle_rs"] = oracle_total - summary[policy_name]["total_latent_expected_value_selected_rs"]
    return summary


def summarize_realized_metrics(events: pd.DataFrame) -> dict:
    """Section D: the STOCHASTIC, sampled counterfactual outcome -- what
    the earlier model evaluations already reported. Kept explicitly separate from section C."""
    n_events = len(events)
    summary = {}
    for policy_name in POLICY_NAMES:
        recovered_col = f"{policy_name}__realized_recovered"
        amount_col = f"{policy_name}__realized_amount_recovered"
        total_recovered = float(events[amount_col].sum())
        summary[policy_name] = {
            "total_amount_recovered_rs": total_recovered,
            "recovery_rate": float(events[recovered_col].mean()),
            "average_rs_recovered_per_failed_payment": total_recovered / n_events if n_events else 0.0,
        }
    fixed_rate = summary["fixed_retry"]["recovery_rate"]
    fixed_amount = summary["fixed_retry"]["total_amount_recovered_rs"]
    for policy_name in POLICY_NAMES:
        summary[policy_name]["recovery_lift_vs_fixed_retry_pp"] = round((summary[policy_name]["recovery_rate"] - fixed_rate) * 100, 4)
        summary[policy_name]["incremental_rs_vs_fixed_retry"] = summary[policy_name]["total_amount_recovered_rs"] - fixed_amount
    return summary


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def sanity_checks(events: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    checks = {}
    counts = test_df.groupby("event_id").size()
    checks["every_event_has_exactly_5_candidates"] = bool((counts == 5).all())

    for policy_name in POLICY_NAMES:
        selected_col = f"{policy_name}__selected_candidate_type"
        valid_values = set(CANDIDATE_TYPES) | {NO_ACTION}
        checks[f"{policy_name}_exactly_one_selection_per_event"] = bool(events[selected_col].notna().all())
        checks[f"{policy_name}_selection_is_known"] = bool(events[selected_col].isin(valid_values).all())
        checks[f"{policy_name}_candidate_distribution"] = events[selected_col].value_counts().to_dict()

    # Verifies the "amount is constant within an event" fact this module's
    # docstring relies on: ranking by probability vs. by value must agree
    # within an event when both come from the SAME underlying score.
    agree = 0
    total = 0
    for _eid, g in test_df.groupby("event_id"):
        if g["candidate_type"].nunique() < 2:
            continue
        by_prob = g.loc[g[LATENT_PROBABILITY_COLUMN].idxmax(), "candidate_type"]
        by_value = g.loc[g[LATENT_VALUE_COLUMN].idxmax(), "candidate_type"]
        agree += int(by_prob == by_value)
        total += 1
    checks["latent_probability_and_value_argmax_always_agree_within_event"] = {"agree": agree, "total": total, "always_true": agree == total}

    return checks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = build_candidate_level_dataset_with_latent_targets()
    _train, _val, test_df = split_candidate_dataset(df)

    regression_metrics_summary = compute_regression_metrics(test_df)
    all_scores = compute_all_scores(test_df)
    events = evaluate_events(test_df, all_scores)
    events.to_csv(REPORTS_DIR / "latent_target_policy_decisions_test_set.csv", index=False)

    ranking_summary = summarize_ranking(events)
    economic_summary = summarize_economic_metrics(events)
    realized_summary = summarize_realized_metrics(events)
    checks = sanity_checks(events, test_df)

    report = {
        "label": "SYNTHETIC COUNTERFACTUAL EVALUATION -- does not measure real Razorpay recovery performance. recovery_probability_latent / expected_recovery_value_latent are synthetic benchmark targets, not production features.",
        "section_A_regression_metrics": regression_metrics_summary,
        "section_B_ranking_metrics_ground_truth_expected_recovery_value_latent": ranking_summary,
        "section_C_economic_metrics_latent_synthetic": economic_summary,
        "section_D_realized_counterfactual_metrics_stochastic": realized_summary,
        "sanity_checks": checks,
    }
    with open(REPORTS_DIR / "latent_target_policy_evaluation.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("=== SYNTHETIC COUNTERFACTUAL EVALUATION (not real Razorpay performance) ===")
    print(f"Test candidate rows: {len(test_df)} | events: {len(events)}")
    print()
    print("Section A -- Regression metrics (Model A / Model B, vs. their own synthetic target):")
    for name, m in regression_metrics_summary.items():
        print(f"  {name:28s} target={m['target_column']:32s} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} R2={m['r2']:.4f} finite={m['predictions_finite']} non_negative={m['predictions_non_negative']}")
    print()
    print("Section B -- Ranking metrics (ground truth = expected_recovery_value_latent):")
    for name in POLICY_NAMES:
        s = ranking_summary[name]
        print(f"  {name:26s} top1={s['top1_accuracy']:.4f} top2={s['top2_accuracy']:.4f} MRR={s['mrr']:.4f} NDCG@5={s['ndcg_at_5']:.4f} mean_rank={s['mean_within_event_rank']:.2f} pairwise_acc={s['pairwise_accuracy']:.4f} avg_regret=Rs{s['avg_regret_rs']:.2f}")
    print()
    print("Section C -- Economic metrics (LATENT, synthetic ground truth):")
    for name in POLICY_NAMES:
        s = economic_summary[name]
        print(f"  {name:26s} total=Rs{s['total_latent_expected_value_selected_rs']:>10.2f} avg/event=Rs{s['average_latent_rs_per_failed_event']:>7.2f} vs_fixed=Rs{s['improvement_vs_fixed_retry_rs']:>+9.2f} regret_vs_oracle=Rs{s['regret_vs_oracle_rs']:>8.2f}")
    print()
    print("Section D -- Realized counterfactual metrics (STOCHASTIC, sampled outcome):")
    for name in POLICY_NAMES:
        s = realized_summary[name]
        print(f"  {name:26s} recovered=Rs{s['total_amount_recovered_rs']:>10.2f} rate={s['recovery_rate']:.4f} lift_vs_fixed={s['recovery_lift_vs_fixed_retry_pp']:+.2f}pp incremental=Rs{s['incremental_rs_vs_fixed_retry']:>+9.2f}")
    print()
    print("Sanity checks:")
    print(f"  every_event_has_exactly_5_candidates: {checks['every_event_has_exactly_5_candidates']}")
    print(f"  latent probability/value argmax agreement within event: {checks['latent_probability_and_value_argmax_always_agree_within_event']}")
    print()
    print(f"Reports written to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
