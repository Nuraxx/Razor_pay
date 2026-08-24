"""
Day-6 candidate-aware model + counterfactual policy evaluation.

    ./venv/bin/python evaluation/evaluate_counterfactual_policy.py

"SYNTHETIC COUNTERFACTUAL EVALUATION" -- every number in this script's
output is computed against data/raw/counterfactual_outcomes.csv, a
hand-designed simulation (see data/generate_counterfactual_dataset.py). It
is legitimate WITHIN that synthetic environment -- Day 6 finally has a real
outcome for every candidate, not just the one that was observed -- but it
does not measure real Razorpay recovery performance. See README "Day 6".

Three things this script does, on the untouched candidate-level test split:

1. Candidate-aware MODEL metrics (ROC-AUC, PR-AUC, log loss, Brier,
   calibration, accuracy/precision/recall/F1) -- same battle-tested
   `compute_metrics`/`bootstrap_ci` evaluation/evaluate_models.py already
   uses, reused here rather than reimplemented.
2. RANKING quality per failure event: rank the 5 candidates by predicted
   probability, compare the argmax (`policy_action`, guardrail-respecting)
   against `oracle_action` (argmax of the true latent probability, also
   guardrail-respecting -- computed by feeding LATENT probabilities through
   the exact same `decide_candidate_aware()` used for real decisions).
3. MONEY metrics for five policies (No Recovery / Fixed Retry / Rule-Based /
   AI-Assisted / Oracle) using the REALIZED counterfactual outcome for
   whichever candidate each policy selects -- the one thing Day 5 could not
   honestly do (see evaluation/evaluate_policy.py's module docstring).
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

from classification.rules import classify
from evaluation.evaluate_models import bootstrap_ci, compute_metrics
from model.candidate_preprocessing import (
    ALL_CATEGORICAL_FEATURES,
    load_candidate_splits,
    prepare_for_catboost,
    select_features_and_target,
)
from model.preprocessing import PROJECT_ROOT
from policy.baselines import fixed_retry_baseline, rule_based_baseline
from policy.guardrails import validate_candidate
from policy.recovery_policy import NO_ACTION, decide_candidate_aware
from policy.retry_candidates import CANDIDATE_TYPES, Candidate, generate_candidates
from policy.scoring import (
    CANDIDATE_ARTIFACTS_DIR,
    Day4ModelUnavailable,
    load_candidate_aware_model,
    predict_candidate_aware_recovery_probability,
)

REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"
N_BOOTSTRAP = 1000
SEED = 42

POLICY_NAMES = ["no_recovery", "fixed_retry", "rule_based", "ai_assisted_policy", "oracle_policy"]


# ---------------------------------------------------------------------------
# 1. Candidate-aware model metrics
# ---------------------------------------------------------------------------

def load_all_candidate_artifacts() -> dict:
    logreg_preprocessor = joblib.load(CANDIDATE_ARTIFACTS_DIR / "logreg_preprocessor.joblib")
    logreg_model = joblib.load(CANDIDATE_ARTIFACTS_DIR / "logreg_model.joblib")
    imputer = joblib.load(CANDIDATE_ARTIFACTS_DIR / "prior_self_resolved_imputer.joblib")
    catboost_model = CatBoostClassifier()
    catboost_model.load_model(str(CANDIDATE_ARTIFACTS_DIR / "catboost_model.cbm"))
    sigmoid_calibrator = joblib.load(CANDIDATE_ARTIFACTS_DIR / "catboost_calibrated_sigmoid.joblib")
    isotonic_calibrator = joblib.load(CANDIDATE_ARTIFACTS_DIR / "catboost_calibrated_isotonic.joblib")
    with open(CANDIDATE_ARTIFACTS_DIR / "model_config.json") as f:
        model_config = json.load(f)
    return {
        "logreg_preprocessor": logreg_preprocessor,
        "logreg_model": logreg_model,
        "imputer": imputer,
        "catboost_model": catboost_model,
        "sigmoid_calibrator": sigmoid_calibrator,
        "isotonic_calibrator": isotonic_calibrator,
        "model_config": model_config,
    }


def compute_candidate_model_metrics(test_df: pd.DataFrame, artifacts: dict) -> tuple[dict, pd.Series]:
    """Returns (metrics_report, sigmoid_calibrated_predictions) -- the
    predictions are reused downstream for ranking/money metrics so the model
    is only ever scored once."""
    X_test_raw, y_test = select_features_and_target(test_df)
    X_test = artifacts["imputer"].transform(X_test_raw)
    y_test_arr = y_test.to_numpy()

    X_test_logreg = artifacts["logreg_preprocessor"].transform(X_test)
    X_test_cb = prepare_for_catboost(X_test)

    predictions = {
        "logistic_regression": artifacts["logreg_model"].predict_proba(X_test_logreg)[:, 1],
        "catboost_uncalibrated": artifacts["catboost_model"].predict_proba(X_test_cb)[:, 1],
        "catboost_calibrated_sigmoid": artifacts["sigmoid_calibrator"].predict_proba(X_test_cb)[:, 1],
        "catboost_calibrated_isotonic": artifacts["isotonic_calibrator"].predict_proba(X_test_cb)[:, 1],
    }
    for name, probs in predictions.items():
        assert probs.min() >= 0.0 and probs.max() <= 1.0, f"{name} produced a probability outside [0,1]"

    metrics_report = {}
    for name, probs in predictions.items():
        m = compute_metrics(y_test_arr, probs)
        if m["roc_auc"] is not None:
            m["roc_auc_bootstrap_ci"] = bootstrap_ci(y_test_arr, probs, roc_auc_score, n_bootstrap=N_BOOTSTRAP, seed=SEED)
        frac_pos, mean_pred = calibration_curve(y_test_arr, probs, n_bins=5, strategy="quantile")
        m["calibration_curve"] = {"mean_predicted": mean_pred.tolist(), "observed_fraction_positive": frac_pos.tolist()}
        metrics_report[name] = m

    predicted_series = pd.Series(predictions["catboost_calibrated_sigmoid"], index=test_df.index)
    return metrics_report, predicted_series


# ---------------------------------------------------------------------------
# 2 & 3. Per-event ranking + money-metric evaluation
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


def _check_guardrail_consistency(test_df: pd.DataFrame) -> int:
    """Sanity check (Day-6 brief section 10 territory): decide_candidate_aware()
    regenerates its own candidates internally (policy/retry_candidates.py's
    fixed offsets), rather than reading test_df's actual (Day-3, lightly
    jittered) candidate_datetime values. Confirms that never flips a
    candidate's validity (after-failure / within-14-day-horizon) for this
    test set, so reusing decide_candidate_aware() unmodified is safe."""
    mismatches = 0
    for event_id, group in test_df.groupby("event_id"):
        failure_ts = group["failure_timestamp"].iloc[0]
        actual_valid = {row["candidate_type"]: validate_candidate(_row_to_candidate(row), failure_ts)[0] for _, row in group.iterrows()}
        regenerated_valid = {c.candidate_type: validate_candidate(c, failure_ts)[0] for c in generate_candidates(failure_ts)}
        if actual_valid != regenerated_valid:
            mismatches += 1
    return mismatches


def evaluate_events(test_df: pd.DataFrame, predicted_probabilities: pd.Series) -> pd.DataFrame:
    """One row per failure event with every policy's selection, realized
    counterfactual outcome, and the ranking quantities (regret, top-1/top-2)
    needed for sections 6 and 7 of the Day-6 brief."""
    df = test_df.copy()
    df["predicted_recovery_probability"] = predicted_probabilities

    records = []
    for event_id, group in df.groupby("event_id"):
        first = group.iloc[0]
        subscription_id = first["subscription_id"]
        failure_timestamp = first["failure_timestamp"]
        amount = float(first["amount"])
        classification_bucket = classify(None, first["error_reason"]).bucket

        predicted_probs = dict(zip(group["candidate_type"], group["predicted_recovery_probability"]))
        latent_probs = dict(zip(group["candidate_type"], group["recovery_probability_latent"]))
        realized_recovered = dict(zip(group["candidate_type"], group["recovered_within_14d"]))
        realized_amount = dict(zip(group["candidate_type"], group["amount_recovered"]))

        selections = {
            "no_recovery": NO_ACTION,
            "fixed_retry": fixed_retry_baseline(event_id, subscription_id, failure_timestamp, amount, classification_bucket, 0.0)["selected_candidate_type"],
            "rule_based": rule_based_baseline(event_id, subscription_id, failure_timestamp, amount, classification_bucket, 0.0)["selected_candidate_type"],
        }
        ai_decision = decide_candidate_aware(event_id, subscription_id, failure_timestamp, amount, classification_bucket, predicted_probs)
        oracle_decision = decide_candidate_aware(event_id, subscription_id, failure_timestamp, amount, classification_bucket, latent_probs)
        selections["ai_assisted_policy"] = ai_decision.selected_candidate_type
        selections["oracle_policy"] = oracle_decision.selected_candidate_type

        record = {
            "event_id": event_id,
            "subscription_id": subscription_id,
            "amount": amount,
            "classification_bucket": classification_bucket,
        }
        for policy_name in POLICY_NAMES:
            selected = selections[policy_name]
            record[f"{policy_name}__selected_candidate_type"] = selected
            if selected == NO_ACTION:
                record[f"{policy_name}__realized_recovered"] = False
                record[f"{policy_name}__realized_amount_recovered"] = 0.0
                record[f"{policy_name}__predicted_probability"] = None
                record[f"{policy_name}__latent_probability"] = None
            else:
                record[f"{policy_name}__realized_recovered"] = bool(realized_recovered[selected])
                record[f"{policy_name}__realized_amount_recovered"] = float(realized_amount[selected])
                record[f"{policy_name}__predicted_probability"] = predicted_probs[selected]
                record[f"{policy_name}__latent_probability"] = latent_probs[selected]

        # Ranking quantities: AI-Assisted vs Oracle only (Fixed/Rule-Based/No
        # Recovery are naive baselines, not ranking-based -- see brief section 6).
        ai_selected, oracle_selected = selections["ai_assisted_policy"], selections["oracle_policy"]
        record["top1_match"] = ai_selected == oracle_selected

        valid_types = [ct for ct in CANDIDATE_TYPES if validate_candidate(_row_to_candidate(group[group["candidate_type"] == ct].iloc[0]), failure_timestamp)[0]]
        ranked_by_prediction = sorted(valid_types, key=lambda ct: predicted_probs[ct], reverse=True)
        record["top2_coverage"] = oracle_selected in ranked_by_prediction[:2] if oracle_selected != NO_ACTION else None

        oracle_ev = (latent_probs.get(oracle_selected, 0.0) * amount) if oracle_selected != NO_ACTION else 0.0
        ai_ev = (latent_probs.get(ai_selected, 0.0) * amount) if ai_selected != NO_ACTION else 0.0
        record["regret_expected_value"] = oracle_ev - ai_ev  # >= 0 by construction (oracle maximizes latent value among valid candidates)
        record["oracle_expected_value"] = oracle_ev
        record["ai_expected_value_at_latent_truth"] = ai_ev

        # DIAGNOSTIC (see README "Day 6: a surprising finding"): pooled
        # ROC-AUC measures discrimination across ALL test rows, which mixes
        # between-event (context) and within-event (candidate) variance. It
        # says nothing about whether the model ranks the 5 candidates of a
        # SINGLE event correctly -- that's what top-1 accuracy actually
        # needs. This computes that directly: rank correlation between
        # predicted and latent probability, within one event's 5 candidates.
        if group["candidate_type"].nunique() >= 2 and pd.Series(list(predicted_probs.values())).nunique() >= 2 and pd.Series(list(latent_probs.values())).nunique() >= 2:
            corr, _ = spearmanr(list(predicted_probs.values()), list(latent_probs.values()))
            record["within_event_rank_correlation"] = corr
        else:
            record["within_event_rank_correlation"] = None

        records.append(record)

    return pd.DataFrame(records)


def summarize_ranking(events: pd.DataFrame) -> dict:
    n = len(events)
    return {
        "n_events": n,
        "top1_candidate_selection_accuracy": float(events["top1_match"].mean()),
        "top2_candidate_coverage": float(events["top2_coverage"].dropna().mean()) if events["top2_coverage"].notna().any() else None,
        "avg_regret_expected_value_rs": float(events["regret_expected_value"].mean()),
        "sum_regret_expected_value_rs": float(events["regret_expected_value"].sum()),
        "avg_predicted_probability_of_ai_selected_action": float(events["ai_assisted_policy__predicted_probability"].dropna().mean()),
        "avg_latent_probability_of_ai_selected_action": float(events["ai_assisted_policy__latent_probability"].dropna().mean()),
        "avg_latent_probability_of_oracle_selected_action": float(events["oracle_policy__latent_probability"].dropna().mean()),
        "policy_oracle_gap_probability_scale": float(
            events["oracle_policy__latent_probability"].dropna().mean() - events["ai_assisted_policy__latent_probability"].dropna().mean()
        ),
        "mean_within_event_rank_correlation": (
            float(events["within_event_rank_correlation"].dropna().mean())
            if events["within_event_rank_correlation"].notna().any() else None
        ),
    }


def summarize_money_metrics(events: pd.DataFrame) -> dict:
    n_events = len(events)
    total_eligible_amount = float(events["amount"].sum())

    summary = {}
    for policy_name in POLICY_NAMES:
        selected_col = f"{policy_name}__selected_candidate_type"
        recovered_col = f"{policy_name}__realized_recovered"
        amount_col = f"{policy_name}__realized_amount_recovered"

        total_recovered = float(events[amount_col].sum())
        recovery_rate = float(events[recovered_col].mean())
        n_actions = int((events[selected_col] != NO_ACTION).sum())
        n_unnecessary = int(((events[selected_col] != NO_ACTION) & (~events[recovered_col])).sum())

        summary[policy_name] = {
            "n_events": n_events,
            "n_actions_taken": n_actions,
            "total_eligible_payment_amount_rs": total_eligible_amount,
            "total_amount_recovered_rs": total_recovered,
            "recovery_rate": recovery_rate,
            "average_rs_recovered_per_failed_payment": total_recovered / n_events if n_events else 0.0,
            "number_of_unnecessary_interventions": n_unnecessary,
        }

    fixed_rate = summary["fixed_retry"]["recovery_rate"]
    fixed_amount = summary["fixed_retry"]["total_amount_recovered_rs"]
    no_recovery_amount = summary["no_recovery"]["total_amount_recovered_rs"]
    oracle_amount = summary["oracle_policy"]["total_amount_recovered_rs"]
    for policy_name in POLICY_NAMES:
        summary[policy_name]["recovery_lift_vs_fixed_retry_pp"] = round((summary[policy_name]["recovery_rate"] - fixed_rate) * 100, 4)
        summary[policy_name]["incremental_rs_vs_fixed_retry"] = summary[policy_name]["total_amount_recovered_rs"] - fixed_amount
        summary[policy_name]["incremental_rs_vs_no_recovery"] = summary[policy_name]["total_amount_recovered_rs"] - no_recovery_amount
        summary[policy_name]["realized_policy_regret_vs_oracle_rs"] = oracle_amount - summary[policy_name]["total_amount_recovered_rs"]

    return summary


def summarize_candidate_distributions(events: pd.DataFrame) -> dict:
    return {
        policy_name: events[f"{policy_name}__selected_candidate_type"].value_counts().to_dict()
        for policy_name in POLICY_NAMES
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        artifacts = load_all_candidate_artifacts()
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Day-6 candidate-aware model artifacts not found: {exc}\n"
            "Run ./venv/bin/python model/train_candidate_model.py first."
        ) from exc

    _train, _val, test_df = load_candidate_splits()
    metrics_report, predicted_probabilities = compute_candidate_model_metrics(test_df, artifacts)

    with open(REPORTS_DIR / "candidate_model_metrics.json", "w") as f:
        json.dump(metrics_report, f, indent=2, default=str)

    guardrail_mismatches = _check_guardrail_consistency(test_df)

    events = evaluate_events(test_df, predicted_probabilities)
    events.to_csv(REPORTS_DIR / "counterfactual_policy_decisions_test_set.csv", index=False)

    ranking_summary = summarize_ranking(events)
    money_summary = summarize_money_metrics(events)
    candidate_distributions = summarize_candidate_distributions(events)

    report = {
        "label": "SYNTHETIC COUNTERFACTUAL EVALUATION -- does not measure real Razorpay recovery performance",
        "guardrail_regeneration_consistency_check": {
            "n_events_with_validity_mismatch": guardrail_mismatches,
            "note": "decide_candidate_aware() regenerates candidates internally; this confirms that never disagrees with the actual test-set candidate_datetime on validity.",
        },
        "candidate_model_metrics": {k: {kk: vv for kk, vv in v.items() if kk != "calibration_curve"} for k, v in metrics_report.items()},
        "ranking_quality": ranking_summary,
        "money_metrics_by_policy": money_summary,
        "selected_candidate_distribution_by_policy": candidate_distributions,
    }
    with open(REPORTS_DIR / "counterfactual_policy_evaluation.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("=== SYNTHETIC COUNTERFACTUAL EVALUATION (not real Razorpay performance) ===")
    print(f"Test candidate rows: {len(test_df)} | events: {ranking_summary['n_events']}")
    print(f"Guardrail regeneration consistency: {guardrail_mismatches} mismatched events (should be 0)")
    print()
    print("Candidate-aware model metrics (sigmoid-calibrated CatBoost):")
    m = metrics_report["catboost_calibrated_sigmoid"]
    print(f"  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  LogLoss={m['log_loss']:.4f}  Brier={m['brier_score']:.4f}  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}")
    print()
    print("Ranking quality:")
    for k, v in ranking_summary.items():
        print(f"  {k}: {v}")
    print()
    print("Money metrics by policy (Synthetic counterfactual evaluation):")
    for policy_name in POLICY_NAMES:
        s = money_summary[policy_name]
        print(
            f"  {policy_name:20s} recovered=Rs{s['total_amount_recovered_rs']:>10.2f}  rate={s['recovery_rate']:.4f}  "
            f"lift_vs_fixed={s['recovery_lift_vs_fixed_retry_pp']:+.2f}pp  regret_vs_oracle=Rs{s['realized_policy_regret_vs_oracle_rs']:>8.2f}  "
            f"unnecessary={s['number_of_unnecessary_interventions']}"
        )
    print()
    print("Selected candidate distribution by policy:")
    for policy_name, dist in candidate_distributions.items():
        print(f"  {policy_name}: {dist}")
    print()
    print(f"Reports written to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
