"""
Offline policy comparison.

    ./venv/bin/python evaluation/evaluate_policy.py

Compares four approaches on data/processed/test.csv -- the same held-out
split evaluation/evaluate_models.py uses, never touched by training or
calibration:

  1. No Recovery      (policy/baselines.py::no_recovery_baseline)
  2. Fixed Retry       (policy/baselines.py::fixed_retry_baseline)
  3. Rule-Based Retry  (policy/baselines.py::rule_based_baseline)
  4. AI-Assisted Policy (policy/recovery_policy.py::decide, pure -- no DB writes)

IMPORTANT -- what this script does NOT claim (see README §16
"counterfactual-evaluation limitation" and policy/scoring.py's module
docstring for the full reasoning):

The synthetic dataset records exactly ONE observed outcome
(`recovered_within_14d`) per failure event -- the outcome of whatever
actually happened, not what would have happened under each of the 5
candidate retry times. There is no genuine per-candidate counterfactual
label in this dataset. So this script NEVER reports an "actual recovery
rate" broken out by candidate_type or by policy -- doing so would silently
imply we ran each policy against reality and observed its outcome, which we
did not.

What it DOES report, kept in clearly separate sections:

  (A) PREDICTED expected recovery value per approach -- the model/heuristic
      score each approach's chosen action, deterministic and reproducible,
      not causal.
  (B) The OBSERVED historical recovery rate/value on this test set as a
      single aggregate reference point (what actually happened, under
      whatever single retry path each row's data-generation actually took)
      -- never sliced by candidate_type or by which policy "would have"
      selected what.
"""
from __future__ import annotations

import json

import pandas as pd

from classification.rules import classify
from model.preprocessing import PROJECT_ROOT, load_processed_splits, select_features_and_target
from policy.baselines import NO_ACTION, fixed_retry_baseline, no_recovery_baseline, rule_based_baseline
from policy.recovery_policy import decide
from policy.retry_candidates import CANDIDATE_TYPES
from policy.scoring import ScoringModelUnavailable, load_calibrated_model, predict_base_recovery_probability

REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"

APPROACHES = {
    "no_recovery": lambda row, base_prob: no_recovery_baseline(row.event_id, row.subscription_id),
    "fixed_retry": lambda row, base_prob: fixed_retry_baseline(
        row.event_id, row.subscription_id, row.failure_timestamp, row.amount, row.classification_bucket, base_prob
    ),
    "rule_based": lambda row, base_prob: rule_based_baseline(
        row.event_id, row.subscription_id, row.failure_timestamp, row.amount, row.classification_bucket, base_prob
    ),
    "ai_assisted_policy": lambda row, base_prob: _decide_as_dict(row, base_prob),
}


def _decide_as_dict(row, base_prob: float) -> dict:
    result = decide(
        event_id=row.event_id,
        subscription_id=row.subscription_id,
        failure_timestamp=row.failure_timestamp,
        amount=row.amount,
        classification_bucket=row.classification_bucket,
        base_probability=base_prob,
        attempts_so_far=0,      # offline comparison: one decision per row, no prior-attempt history to consult
        already_decided=False,
    )
    return {
        "selected_candidate_type": result.selected_candidate_type,
        "predicted_recovery_probability": result.predicted_recovery_probability,
        "expected_recovery_value": result.expected_recovery_value,
        "expected_incremental_value": result.expected_incremental_value,
        "decision_reason": result.decision_reason,
    }


def build_scored_events(test_df: pd.DataFrame, model, imputer) -> pd.DataFrame:
    """One row per test-set failure event: classification bucket, calibrated
    base_probability, and every approach's decision on that event."""
    X_test, _y = select_features_and_target(test_df)
    base_probabilities = predict_base_recovery_probability(X_test, model, imputer)

    df = test_df.copy()
    df["failure_timestamp"] = pd.to_datetime(df["failure_timestamp"])
    df["classification_bucket"] = df["error_reason"].apply(lambda r: classify(None, r).bucket)
    df["base_probability"] = base_probabilities.values

    records = []
    for row in df.itertuples(index=False):
        base_prob = row.base_probability
        record = {
            "event_id": row.event_id,
            "subscription_id": row.subscription_id,
            "amount": row.amount,
            "classification_bucket": row.classification_bucket,
            "base_probability": base_prob,
            "observed_recovered_within_14d": bool(row.recovered_within_14d),
        }
        for approach_name, fn in APPROACHES.items():
            decision = fn(row, base_prob)
            record[f"{approach_name}__selected_candidate_type"] = decision["selected_candidate_type"]
            record[f"{approach_name}__predicted_recovery_probability"] = decision["predicted_recovery_probability"]
            record[f"{approach_name}__expected_recovery_value"] = decision["expected_recovery_value"]
            record[f"{approach_name}__expected_incremental_value"] = decision["expected_incremental_value"]
        records.append(record)

    return pd.DataFrame(records)


def summarize_predicted_expected_value(scored: pd.DataFrame) -> dict:
    """Section (A): PREDICTED expected value per approach. Never observed outcomes."""
    summary = {}
    for approach_name in APPROACHES:
        selected_col = f"{approach_name}__selected_candidate_type"
        value_col = f"{approach_name}__expected_recovery_value"
        incremental_col = f"{approach_name}__expected_incremental_value"

        n_events = len(scored)
        n_no_action = int((scored[selected_col] == NO_ACTION).sum())
        n_actions = n_events - n_no_action
        candidate_distribution = (
            scored.loc[scored[selected_col] != NO_ACTION, selected_col].value_counts().to_dict()
        )

        summary[approach_name] = {
            "n_events": n_events,
            "n_no_action": n_no_action,
            "n_actions_taken": n_actions,
            "selected_candidate_type_distribution": candidate_distribution,
            "sum_predicted_expected_recovery_value": float(scored[value_col].fillna(0.0).sum()),
            "sum_predicted_expected_incremental_value": float(scored[incremental_col].fillna(0.0).sum()),
            "mean_predicted_recovery_probability_on_acted_events": (
                float(scored.loc[scored[selected_col] != NO_ACTION, f"{approach_name}__predicted_recovery_probability"].mean())
                if n_actions > 0
                else None
            ),
        }
    return summary


def summarize_observed_reference(scored: pd.DataFrame) -> dict:
    """Section (B): the single OBSERVED historical outcome per event, as an
    aggregate reference point only -- never attributed to any candidate_type
    or any approach's chosen action. See module docstring."""
    return {
        "n_events": len(scored),
        "observed_recovery_rate": float(scored["observed_recovered_within_14d"].mean()),
        "observed_recovered_amount_estimate": float(
            (scored["observed_recovered_within_14d"].astype(float) * scored["amount"]).sum()
        ),
        "total_amount_at_risk": float(scored["amount"].sum()),
        "note": (
            "This is what actually happened in the synthetic data generation for each event's "
            "single realized outcome. It is NOT broken out by candidate_type or by policy, and "
            "must not be read as 'the AI policy would have achieved this rate' -- see module docstring."
        ),
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        model, imputer = load_calibrated_model()
    except ScoringModelUnavailable as exc:
        raise SystemExit(f"{exc}\nRun ./venv/bin/python model/train.py first.") from exc

    _train, _val, test_df = load_processed_splits()
    scored = build_scored_events(test_df, model, imputer)
    scored.to_csv(REPORTS_DIR / "policy_decisions_test_set.csv", index=False)

    predicted_summary = summarize_predicted_expected_value(scored)
    observed_reference = summarize_observed_reference(scored)

    report = {
        "predicted_expected_value_by_approach": predicted_summary,
        "observed_historical_reference_aggregate_only": observed_reference,
        "limitation": (
            "The synthetic dataset provides one observed outcome per failure event, not per "
            "candidate retry time, so no causal or counterfactual recovery-lift claim is made "
            "for candidate timing or for any approach's selected action. Only PREDICTED expected "
            "recovery value (model probability x amount) is compared across approaches. See "
            "README §16 'counterfactual-evaluation limitation'."
        ),
    }
    with open(REPORTS_DIR / "policy_evaluation.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Test rows: {len(scored)}")
    print()
    print("(A) PREDICTED expected value by approach -- model/heuristic estimate, not observed outcome:")
    for approach_name, s in predicted_summary.items():
        mean_prob = f"{s['mean_predicted_recovery_probability_on_acted_events']:.4f}" if s["mean_predicted_recovery_probability_on_acted_events"] is not None else "n/a"
        print(
            f"  {approach_name:20s} actions={s['n_actions_taken']:3d}/{s['n_events']:<3d} "
            f"sum_expected_recovery_value={s['sum_predicted_expected_recovery_value']:>10.2f} "
            f"mean_predicted_prob={mean_prob}"
        )
        print(f"    candidate distribution: {s['selected_candidate_type_distribution']}")
    print()
    print("(B) OBSERVED historical reference (aggregate only, NOT per-candidate, NOT causal):")
    print(f"  observed_recovery_rate = {observed_reference['observed_recovery_rate']:.4f}")
    print(f"  {observed_reference['note']}")
    print()
    print(f"Reports written to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
