"""
Day-7 ranking-model + money evaluation.

    ./venv/bin/python evaluation/evaluate_ranking_policy.py

"SYNTHETIC COUNTERFACTUAL EVALUATION" -- same disclaimer as Day 6: every
number here comes from data/raw/counterfactual_outcomes.csv, a hand-designed
simulation. It does not measure real Razorpay recovery performance.

Compares 5 approaches (brief section 7): Random candidate, Fixed Retry,
Day-6 probability model, Day-7 ranking model, Oracle -- on both ranking
metrics (top-1/top-2 accuracy, MRR, NDCG@5, mean within-event rank, pairwise
accuracy, regret; ground truth = recovery_probability_latent, same Oracle
definition Day 6 used, for continuity) and money metrics (section 8,
realized counterfactual outcomes). Also runs the section-9 ablation study
(event-only / candidate-only / event+candidate pointwise / Day-6 / Day-7)
to separate "weak candidate features" from "wrong modeling objective" as the
driver of Day 6's failure.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from classification.rules import classify
from evaluation.evaluate_models import compute_metrics
from evaluation.ranking_metrics import (
    ndcg_at_5,
    pairwise_concordant_pairs,
    reciprocal_rank,
    regret,
    top_k_accuracy,
    within_event_rank,
)
from model.candidate_preprocessing import PriorSelfResolvedImputer, load_candidate_splits
from model.preprocessing import PROJECT_ROOT, SEED
from model.ranking_preprocessing import CANDIDATE_CATEGORICAL_FEATURES as R_CANDIDATE_CATEGORICAL_FEATURES
from model.ranking_preprocessing import CANDIDATE_BOOLEAN_FEATURES as R_CANDIDATE_BOOLEAN_FEATURES
from model.ranking_preprocessing import CANDIDATE_NUMERIC_FEATURES as R_CANDIDATE_NUMERIC_FEATURES
from model.ranking_preprocessing import EVENT_BOOLEAN_FEATURES as R_EVENT_BOOLEAN_FEATURES
from model.ranking_preprocessing import EVENT_CATEGORICAL_FEATURES as R_EVENT_CATEGORICAL_FEATURES
from model.ranking_preprocessing import EVENT_NUMERIC_FEATURES as R_EVENT_NUMERIC_FEATURES
from model.train_ranking_model import load_ranking_model, predict_ranking_scores
from policy.baselines import fixed_retry_baseline
from policy.guardrails import validate_candidate
from policy.recovery_policy import NO_ACTION, decide_candidate_aware
from policy.retry_candidates import CANDIDATE_TYPES, Candidate
from policy.scoring import load_candidate_aware_model, predict_candidate_aware_recovery_probability

REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"
RANDOM_SEED = SEED
POLICY_NAMES = ["random_candidate", "fixed_retry", "day6_probability_model", "day7_ranking_model", "oracle_policy"]


# ---------------------------------------------------------------------------
# Score computation for each of the 5 approaches
# ---------------------------------------------------------------------------

def compute_all_scores(test_df: pd.DataFrame) -> dict[str, pd.Series]:
    from model.candidate_preprocessing import select_features_and_target as day6_select
    from model.candidate_preprocessing import prepare_for_catboost as day6_prepare_cb

    day6_model, day6_imputer = load_candidate_aware_model()
    X6, _y = day6_select(test_df)
    X6_imp = day6_imputer.transform(X6)
    day6_scores = predict_candidate_aware_recovery_probability(X6_imp, day6_model, day6_imputer)

    ranking_fitted = load_ranking_model()
    day7_scores = predict_ranking_scores(test_df, ranking_fitted)

    rng = np.random.default_rng(RANDOM_SEED)
    random_scores = pd.Series(rng.random(len(test_df)), index=test_df.index)

    oracle_scores = test_df["recovery_probability_latent"]

    return {
        "day6_probability_model": day6_scores,
        "day7_ranking_model": day7_scores,
        "random_candidate": random_scores,
        "oracle_policy": oracle_scores,
    }


def compute_pooled_classification_metrics(test_df: pd.DataFrame, score_columns: dict[str, pd.Series]) -> dict:
    y_true = test_df["recovered_within_14d"].astype(int).to_numpy()
    out = {}
    for name in ("day6_probability_model", "day7_ranking_model"):
        probs = score_columns[name].to_numpy()
        m = compute_metrics(y_true, np.clip(probs, 0.0, 1.0))
        out[name] = {k: v for k, v in m.items() if k != "confusion_matrix"}
    return out


# ---------------------------------------------------------------------------
# Per-event ranking + money evaluation
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
        truth = group["recovery_probability_latent"].to_numpy()
        realized_recovered = dict(zip(group["candidate_type"], group["recovered_within_14d"]))
        realized_amount = dict(zip(group["candidate_type"], group["amount_recovered"]))
        valid_mask = np.array([validate_candidate(_row_to_candidate(r), failure_timestamp)[0] for _, r in group.iterrows()])

        record = {"event_id": event_id, "subscription_id": subscription_id, "amount": amount, "classification_bucket": classification_bucket}

        # Fixed Retry: naive baseline, no continuous score -- a one-hot
        # "score" vector so it still gets a well-defined rank/NDCG/etc.
        fixed_selected = fixed_retry_baseline(event_id, subscription_id, failure_timestamp, amount, classification_bucket, 0.0)["selected_candidate_type"]
        fixed_scores = np.array([1.0 if ct == fixed_selected else 0.0 for ct in candidate_types])

        for policy_name in POLICY_NAMES:
            if policy_name == "fixed_retry":
                scores = fixed_scores
                selected = fixed_selected
            else:
                scores = group[f"_score_{policy_name}"].to_numpy()
                probs = dict(zip(candidate_types, scores))
                selected = decide_candidate_aware(event_id, subscription_id, failure_timestamp, amount, classification_bucket, probs).selected_candidate_type

            record[f"{policy_name}__selected_candidate_type"] = selected
            record[f"{policy_name}__realized_recovered"] = bool(realized_recovered.get(selected, False)) if selected != NO_ACTION else False
            record[f"{policy_name}__realized_amount_recovered"] = float(realized_amount.get(selected, 0.0)) if selected != NO_ACTION else 0.0

            if group["candidate_type"].nunique() >= 2:
                record[f"{policy_name}__top1"] = top_k_accuracy(scores, truth, candidate_types, 1)
                record[f"{policy_name}__top2"] = top_k_accuracy(scores, truth, candidate_types, 2)
                record[f"{policy_name}__rr"] = reciprocal_rank(scores, truth, candidate_types)
                record[f"{policy_name}__ndcg5"] = ndcg_at_5(scores, truth)
                record[f"{policy_name}__rank"] = within_event_rank(scores, truth, candidate_types)
                concordant, total = pairwise_concordant_pairs(scores, truth)
                record[f"{policy_name}__pairwise_concordant"] = concordant
                record[f"{policy_name}__pairwise_total"] = total
                record[f"{policy_name}__regret"] = regret(scores, truth, candidate_types, amount, valid_mask)
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


def summarize_money_metrics(events: pd.DataFrame) -> dict:
    n_events = len(events)
    total_eligible_amount = float(events["amount"].sum())

    summary = {}
    for policy_name in POLICY_NAMES:
        recovered_col = f"{policy_name}__realized_recovered"
        amount_col = f"{policy_name}__realized_amount_recovered"
        selected_col = f"{policy_name}__selected_candidate_type"

        total_recovered = float(events[amount_col].sum())
        recovery_rate = float(events[recovered_col].mean())
        summary[policy_name] = {
            "n_events": n_events,
            "n_actions_taken": int((events[selected_col] != NO_ACTION).sum()),
            "total_eligible_payment_amount_rs": total_eligible_amount,
            "total_amount_recovered_rs": total_recovered,
            "recovery_rate": recovery_rate,
            "average_rs_recovered_per_failed_payment": total_recovered / n_events if n_events else 0.0,
        }

    fixed_rate = summary["fixed_retry"]["recovery_rate"]
    fixed_amount = summary["fixed_retry"]["total_amount_recovered_rs"]
    oracle_amount = summary["oracle_policy"]["total_amount_recovered_rs"]
    for policy_name in POLICY_NAMES:
        summary[policy_name]["recovery_lift_vs_fixed_retry_pp"] = round((summary[policy_name]["recovery_rate"] - fixed_rate) * 100, 4)
        summary[policy_name]["incremental_rs_vs_fixed_retry"] = summary[policy_name]["total_amount_recovered_rs"] - fixed_amount
        summary[policy_name]["realized_regret_vs_oracle_rs"] = oracle_amount - summary[policy_name]["total_amount_recovered_rs"]

    return summary


# ---------------------------------------------------------------------------
# Section 9: ablation study
# ---------------------------------------------------------------------------

def _fit_quick_pointwise_model(train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str], boolean_features: list[str]) -> dict:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    all_features = numeric_features + categorical_features + boolean_features

    def select(df):
        X = df[all_features].copy()
        for col in boolean_features:
            X[col] = X[col].astype(int)
        y = df["recovered_within_14d"].astype(int)
        return X, y

    X_train_raw, y_train = select(train_df)
    X_val_raw, y_val = select(val_df)

    needs_imputer = "prior_if_self_resolved_rate" in numeric_features
    if needs_imputer:
        imputer = PriorSelfResolvedImputer().fit(X_train_raw)
        X_train_raw = imputer.transform(X_train_raw)
        X_val_raw = imputer.transform(X_val_raw)
        numeric_with_flag = numeric_features + ["prior_if_self_resolved_rate_missing"]
    else:
        numeric_with_flag = numeric_features

    if categorical_features:
        cb_categorical = categorical_features
        for col in cb_categorical:
            X_train_raw[col] = X_train_raw[col].astype(str).astype(object)
            X_val_raw[col] = X_val_raw[col].astype(str).astype(object)
        model = CatBoostClassifier(cat_features=cb_categorical, iterations=200, depth=4, learning_rate=0.05, loss_function="Logloss", eval_metric="AUC", random_seed=SEED, early_stopping_rounds=30, use_best_model=True, verbose=False)
    else:
        model = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05, loss_function="Logloss", eval_metric="AUC", random_seed=SEED, early_stopping_rounds=30, use_best_model=True, verbose=False)

    model.fit(X_train_raw, y_train, eval_set=(X_val_raw, y_val))
    return {"model": model, "select": select, "needs_imputer": needs_imputer, "imputer": imputer if needs_imputer else None}


def _score_with_quick_model(fitted: dict, df: pd.DataFrame) -> pd.Series:
    X, _y = fitted["select"](df)
    if fitted["needs_imputer"]:
        X = fitted["imputer"].transform(X)
    return pd.Series(fitted["model"].predict_proba(X)[:, 1], index=df.index)


def run_ablation_study(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, day6_scores: pd.Series, day7_scores: pd.Series) -> dict:
    """A: event context only. B: candidate features only. C: event+candidate,
    POINTWISE objective (isolates features from objective -- D/E already
    exist as the real Day-6/Day-7 models)."""
    results = {}

    configs = {
        "A_event_context_only": (R_EVENT_NUMERIC_FEATURES, R_EVENT_CATEGORICAL_FEATURES, R_EVENT_BOOLEAN_FEATURES),
        "B_candidate_features_only": (R_CANDIDATE_NUMERIC_FEATURES, R_CANDIDATE_CATEGORICAL_FEATURES, R_CANDIDATE_BOOLEAN_FEATURES),
        "C_event_plus_candidate_pointwise": (
            R_EVENT_NUMERIC_FEATURES + R_CANDIDATE_NUMERIC_FEATURES,
            R_EVENT_CATEGORICAL_FEATURES + R_CANDIDATE_CATEGORICAL_FEATURES,
            R_EVENT_BOOLEAN_FEATURES + R_CANDIDATE_BOOLEAN_FEATURES,
        ),
    }

    ablation_scores = {}
    for name, (numeric, categorical, boolean) in configs.items():
        fitted = _fit_quick_pointwise_model(train_df, val_df, numeric, categorical, boolean)
        ablation_scores[name] = _score_with_quick_model(fitted, test_df)

    ablation_scores["D_day6_probability_model"] = day6_scores
    ablation_scores["E_day7_ranking_model"] = day7_scores

    for name, scores in ablation_scores.items():
        df = test_df.copy()
        df["_score"] = scores
        top1s, ndcg5s, ranks = [], [], []
        for _event_id, g in df.groupby("event_id"):
            if g["recovery_probability_latent"].nunique() < 2:
                continue
            s = g["_score"].to_numpy()
            t = g["recovery_probability_latent"].to_numpy()
            ct = g["candidate_type"].to_numpy()
            top1s.append(top_k_accuracy(s, t, ct, 1))
            ndcg5s.append(ndcg_at_5(s, t))
            ranks.append(within_event_rank(s, t, ct))
        results[name] = {
            "n_events_with_variance": len(top1s),
            "top1_accuracy": float(np.mean(top1s)) if top1s else None,
            "ndcg_at_5": float(np.mean(ndcg5s)) if ndcg5s else None,
            "mean_within_event_rank": float(np.mean(ranks)) if ranks else None,
        }

    return results


# ---------------------------------------------------------------------------
# Section 10: sanity checks
# ---------------------------------------------------------------------------

def sanity_checks(events: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    checks = {}
    counts = test_df.groupby("event_id").size()
    checks["every_event_has_exactly_5_candidates"] = bool((counts == 5).all())

    for policy_name in POLICY_NAMES:
        selected_col = f"{policy_name}__selected_candidate_type"
        checks[f"{policy_name}_exactly_one_selection_per_event"] = bool(events[selected_col].notna().all())
        valid_values = set(CANDIDATE_TYPES) | {NO_ACTION}
        checks[f"{policy_name}_selection_is_a_known_candidate_or_no_action"] = bool(events[selected_col].isin(valid_values).all())

        distribution = events[selected_col].value_counts(normalize=True).to_dict()
        max_share = max(distribution.values()) if distribution else 0.0
        checks[f"{policy_name}_candidate_distribution"] = events[selected_col].value_counts().to_dict()
        checks[f"{policy_name}_no_single_candidate_dominates"] = bool(max_share < 0.90)

    return checks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        train_df, val_df, test_df = load_candidate_splits()
        all_scores = compute_all_scores(test_df)
    except FileNotFoundError as exc:
        raise SystemExit(f"Model artifacts not found: {exc}\nRun model/train_candidate_model.py and model/train_ranking_model.py first.") from exc

    pooled_metrics = compute_pooled_classification_metrics(test_df, all_scores)
    events = evaluate_events(test_df, all_scores)
    events.to_csv(REPORTS_DIR / "ranking_policy_decisions_test_set.csv", index=False)

    ranking_summary = summarize_ranking(events)
    money_summary = summarize_money_metrics(events)
    ablation_results = run_ablation_study(train_df, val_df, test_df, all_scores["day6_probability_model"], all_scores["day7_ranking_model"])
    checks = sanity_checks(events, test_df)

    report = {
        "label": "SYNTHETIC COUNTERFACTUAL EVALUATION -- does not measure real Razorpay recovery performance",
        "pooled_classification_metrics": pooled_metrics,
        "ranking_metrics_by_policy": ranking_summary,
        "money_metrics_by_policy": money_summary,
        "ablation_study": ablation_results,
        "sanity_checks": checks,
    }
    with open(REPORTS_DIR / "ranking_policy_evaluation.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("=== SYNTHETIC COUNTERFACTUAL EVALUATION (not real Razorpay performance) ===")
    print(f"Test candidate rows: {len(test_df)} | events: {len(events)}")
    print()
    print("Pooled classification metrics:")
    for name, m in pooled_metrics.items():
        print(f"  {name:24s} ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  LogLoss={m['log_loss']:.4f}")
    print()
    print("Ranking metrics by policy:")
    for name in POLICY_NAMES:
        s = ranking_summary[name]
        print(f"  {name:24s} top1={s['top1_accuracy']:.4f}  top2={s['top2_accuracy']:.4f}  MRR={s['mrr']:.4f}  NDCG@5={s['ndcg_at_5']:.4f}  mean_rank={s['mean_within_event_rank']:.2f}  pairwise_acc={s['pairwise_accuracy']:.4f}  avg_regret=Rs{s['avg_regret_rs']:.2f}")
    print()
    print("Money metrics by policy:")
    for name in POLICY_NAMES:
        s = money_summary[name]
        print(f"  {name:24s} recovered=Rs{s['total_amount_recovered_rs']:>10.2f}  rate={s['recovery_rate']:.4f}  lift_vs_fixed={s['recovery_lift_vs_fixed_retry_pp']:+.2f}pp  regret_vs_oracle=Rs{s['realized_regret_vs_oracle_rs']:>8.2f}")
    print()
    print("Ablation study (top1_accuracy / ndcg_at_5 / mean_within_event_rank, vs. Oracle):")
    for name, r in ablation_results.items():
        print(f"  {name:34s} top1={r['top1_accuracy']:.4f}  ndcg5={r['ndcg_at_5']:.4f}  mean_rank={r['mean_within_event_rank']:.2f}")
    print()
    print("Sanity checks:")
    for k in ("every_event_has_exactly_5_candidates",):
        print(f"  {k}: {checks[k]}")
    for name in POLICY_NAMES:
        print(f"  {name}_no_single_candidate_dominates: {checks[f'{name}_no_single_candidate_dominates']}  distribution={checks[f'{name}_candidate_distribution']}")
    print()
    print(f"Reports written to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
