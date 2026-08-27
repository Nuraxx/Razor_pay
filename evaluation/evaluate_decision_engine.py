"""
policy-v3 decision-engine evaluation.

    ./venv/bin/python evaluation/evaluate_decision_engine.py

"SYNTHETIC COUNTERFACTUAL EVALUATION" -- same disclaimer as the earlier model evaluations: every
number here comes from data/raw/counterfactual_outcomes.csv, a
hand-designed simulation. It does not measure real Razorpay recovery
performance.

Two phases, strictly ordered (brief section 10/11 -- never tune on test):

  1. THRESHOLD SELECTION on VALIDATION ONLY. Searches the brief's predefined
     set {0,10,25,50,100,150,200,250} (Rs of decision margin), scored by
     total LATENT net value selected on validation. The winning threshold
     is printed and is the one already frozen into
     policy/decision_engine.py::DEFAULT_ABSTENTION_THRESHOLD_RS -- this
     script's search REPRODUCES that choice, it does not change it.
  2. TEST evaluation, run once, comparing 5 policies (brief section 10):
     Fixed Retry, Rule-Based, Model B without abstention, policy-v3 (Model
     B + costs + abstention + fallback), Oracle. Reports LATENT ECONOMIC,
     REALIZED COUNTERFACTUAL, and OPERATIONAL metrics as three explicitly
     separate sections.
"""
from __future__ import annotations

import json
import math

import pandas as pd

from classification.rules import classify
from model.latent_target_preprocessing import LATENT_VALUE_COLUMN, PROJECT_ROOT, build_candidate_level_dataset_with_latent_targets
from model.candidate_preprocessing import split_candidate_dataset
from model.train_latent_target_model import load_latent_target_model
from policy.baselines import fixed_retry_baseline, rule_based_baseline
from policy.costs import DEFAULT_COSTS
from policy.decision_engine import DEFAULT_ABSTENTION_THRESHOLD_RS, NO_ACTION, SOURCE_FALLBACK, decide_engine
from policy.guardrails import validate_candidate
from policy.retry_candidates import Candidate

REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"
THRESHOLD_CANDIDATES = [0, 10, 25, 50, 100, 150, 200, 250]
POLICY_NAMES = ["fixed_retry", "rule_based", "model_b_no_abstention", "policy_v3", "oracle_policy"]

EVENT_FEATURE_KEYS = [
    "day_of_month", "days_to_nearest_payday_window", "prior_if_failure_count", "prior_if_self_resolved_rate",
    "tenure_days", "plan_tier", "primary_instrument", "city_tier", "bank_network_conditions",
    "issuing_bank_downtime_flag", "network_latency_bucket", "is_month_end_settlement_rush",
]


def _event_context(first_row: pd.Series) -> dict:
    return {k: first_row[k] for k in EVENT_FEATURE_KEYS}


def _row_to_candidate(row: pd.Series) -> Candidate:
    return Candidate(
        candidate_type=row["candidate_type"], candidate_datetime=row["candidate_datetime"], hours_from_failure=row["hours_from_failure"],
        candidate_day_of_month=int(row["candidate_day_of_month"]), candidate_day_of_week=row["candidate_day_of_week"],
        candidate_is_payday_aligned=bool(row["candidate_is_payday_aligned"]), candidate_is_month_end_aligned=bool(row["candidate_is_month_end_aligned"]),
        candidate_days_to_payday=int(row["candidate_days_to_payday"]),
    )


def _latent_value_lookup(group: pd.DataFrame) -> dict[str, float]:
    return dict(zip(group["candidate_type"], group[LATENT_VALUE_COLUMN]))


def _run_decision_engine_for_all_events(df: pd.DataFrame, model: dict, abstention_threshold: float) -> pd.DataFrame:
    """Runs decide_engine() once per event, returns a DataFrame of results
    keyed by event_id, including the selected candidate's LATENT value
    (looked up directly, never re-predicted) for scoring."""
    records = []
    for event_id, group in df.groupby("event_id"):
        first = group.iloc[0]
        subscription_id, failure_timestamp, amount = first["subscription_id"], first["failure_timestamp"], float(first["amount"])
        classification_bucket = classify(None, first["error_reason"]).bucket
        decision = decide_engine(
            event_id, subscription_id, failure_timestamp, amount, classification_bucket, _event_context(first),
            costs=DEFAULT_COSTS, abstention_threshold=abstention_threshold, model=model,
        )
        latent_by_type = _latent_value_lookup(group)
        records.append(
            {
                "event_id": event_id, "amount": amount,
                "selected_candidate_type": decision.selected_candidate_type,
                "decision_source": decision.decision_source,
                "decision_margin": decision.decision_margin,
                "latent_value_selected": latent_by_type.get(decision.selected_candidate_type, 0.0) if decision.selected_candidate_type != NO_ACTION else 0.0,
            }
        )
    return pd.DataFrame(records)


def select_abstention_threshold_on_validation(val_df: pd.DataFrame, model: dict) -> tuple[float, dict]:
    """VALIDATION-ONLY search (brief section 11) -- test is never touched
    here. Scores each candidate threshold by total LATENT net value
    (economic ground truth, legitimate for offline threshold tuning --
    same reasoning as Model B's own evaluation) selected across all
    validation events; picks the argmax."""
    results = {}
    for threshold in THRESHOLD_CANDIDATES:
        run = _run_decision_engine_for_all_events(val_df, model, float(threshold))
        results[threshold] = {"total_latent_value_selected_rs": float(run["latent_value_selected"].sum()), "n_actions": int((run["selected_candidate_type"] != NO_ACTION).sum())}
    best_threshold = max(results, key=lambda t: results[t]["total_latent_value_selected_rs"])
    return float(best_threshold), results


# ---------------------------------------------------------------------------
# Test-set evaluation across all 5 policies
# ---------------------------------------------------------------------------

def evaluate_events(test_df: pd.DataFrame, model: dict, frozen_threshold: float) -> pd.DataFrame:
    records = []
    for event_id, group in test_df.groupby("event_id"):
        first = group.iloc[0]
        subscription_id, failure_timestamp, amount = first["subscription_id"], first["failure_timestamp"], float(first["amount"])
        classification_bucket = classify(None, first["error_reason"]).bucket
        realized_recovered = dict(zip(group["candidate_type"], group["recovered_within_14d"]))
        realized_amount = dict(zip(group["candidate_type"], group["amount_recovered"]))
        latent_value = _latent_value_lookup(group)
        valid_mask = {row["candidate_type"]: validate_candidate(_row_to_candidate(row), failure_timestamp)[0] for _, row in group.iterrows()}

        record = {"event_id": event_id, "subscription_id": subscription_id, "amount": amount}

        fixed_sel = fixed_retry_baseline(event_id, subscription_id, failure_timestamp, amount, classification_bucket, 0.0)["selected_candidate_type"]
        rule_sel = rule_based_baseline(event_id, subscription_id, failure_timestamp, amount, classification_bucket, 0.0)["selected_candidate_type"]
        no_abstention_decision = decide_engine(event_id, subscription_id, failure_timestamp, amount, classification_bucket, _event_context(first), costs=DEFAULT_COSTS, abstention_threshold=-math.inf, model=model)
        original_fallback_decision = decide_engine(event_id, subscription_id, failure_timestamp, amount, classification_bucket, _event_context(first), costs=DEFAULT_COSTS, abstention_threshold=frozen_threshold, model=model)
        oracle_sel = max((ct for ct, valid in valid_mask.items() if valid), key=lambda ct: latent_value[ct], default=NO_ACTION)

        selections = {
            "fixed_retry": fixed_sel, "rule_based": rule_sel,
            "model_b_no_abstention": no_abstention_decision.selected_candidate_type,
            "policy_v3": original_fallback_decision.selected_candidate_type,
            "oracle_policy": oracle_sel,
        }
        for policy_name in POLICY_NAMES:
            selected = selections[policy_name]
            record[f"{policy_name}__selected_candidate_type"] = selected
            record[f"{policy_name}__realized_recovered"] = bool(realized_recovered.get(selected, False)) if selected != NO_ACTION else False
            record[f"{policy_name}__realized_amount_recovered"] = float(realized_amount.get(selected, 0.0)) if selected != NO_ACTION else 0.0
            record[f"{policy_name}__latent_value_selected"] = float(latent_value.get(selected, 0.0)) if selected != NO_ACTION else 0.0

        record["policy_v3__decision_source"] = original_fallback_decision.decision_source
        record["policy_v3__decision_margin"] = original_fallback_decision.decision_margin
        record["policy_v3__is_no_action"] = original_fallback_decision.selected_candidate_type == NO_ACTION
        record["policy_v3__is_fallback"] = original_fallback_decision.decision_source == SOURCE_FALLBACK

        records.append(record)

    return pd.DataFrame(records)


def summarize_latent_economic(events: pd.DataFrame) -> dict:
    n = len(events)
    summary = {}
    for name in POLICY_NAMES:
        total = float(events[f"{name}__latent_value_selected"].sum())
        summary[name] = {"total_latent_value_rs": total, "avg_latent_value_per_event_rs": total / n if n else 0.0}
    fixed_total, oracle_total = summary["fixed_retry"]["total_latent_value_rs"], summary["oracle_policy"]["total_latent_value_rs"]
    for name in POLICY_NAMES:
        summary[name]["improvement_vs_fixed_retry_rs"] = summary[name]["total_latent_value_rs"] - fixed_total
        summary[name]["regret_vs_oracle_rs"] = oracle_total - summary[name]["total_latent_value_rs"]
    return summary


def summarize_realized(events: pd.DataFrame) -> dict:
    n = len(events)
    summary = {}
    for name in POLICY_NAMES:
        total = float(events[f"{name}__realized_amount_recovered"].sum())
        summary[name] = {"total_recovered_rs": total, "recovery_rate": float(events[f"{name}__realized_recovered"].mean())}
    fixed_total = summary["fixed_retry"]["total_recovered_rs"]
    for name in POLICY_NAMES:
        summary[name]["incremental_rs_vs_fixed_retry"] = summary[name]["total_recovered_rs"] - fixed_total
    return summary


def summarize_operational(events: pd.DataFrame) -> dict:
    n = len(events)
    n_actions = int((events["policy_v3__selected_candidate_type"] != NO_ACTION).sum())
    n_no_action = n - n_actions
    n_fallback = int(events["policy_v3__is_fallback"].sum())
    margins = events["policy_v3__decision_margin"].dropna()
    return {
        "n_events": n,
        "actions_selected": n_actions,
        "no_action_count": n_no_action,
        "fallback_count": n_fallback,
        "abstention_count": n_fallback,  # every fallback in this dataset is margin-triggered (model always available) -- see report note
        "average_decision_margin_rs": float(margins.mean()) if len(margins) else None,
        "pct_decisions_using_fallback": round(100 * n_fallback / n, 2) if n else 0.0,
        "decision_source_distribution": events["policy_v3__decision_source"].value_counts().to_dict(),
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_latent_target_model("value")
    df = build_candidate_level_dataset_with_latent_targets()
    train_df, val_df, test_df = split_candidate_dataset(df)
    del train_df  # unused here -- decide_engine is not fit on data, only Model B was (already trained)

    print("=== Phase 1: threshold selection on VALIDATION ONLY ===")
    chosen_threshold, search_results = select_abstention_threshold_on_validation(val_df, model)
    for t, r in search_results.items():
        marker = " <-- selected" if t == chosen_threshold else ""
        print(f"  threshold=Rs{t:<5} total_latent_value=Rs{r['total_latent_value_selected_rs']:>10.2f} n_actions={r['n_actions']}{marker}")
    frozen_matches = chosen_threshold == DEFAULT_ABSTENTION_THRESHOLD_RS
    print(f"  Frozen threshold in policy/decision_engine.py: Rs{DEFAULT_ABSTENTION_THRESHOLD_RS:.2f} (matches this search: {frozen_matches})")
    print()

    print("=== Phase 2: TEST evaluation (run once, frozen threshold) ===")
    events = evaluate_events(test_df, model, frozen_threshold=DEFAULT_ABSTENTION_THRESHOLD_RS)
    events.to_csv(REPORTS_DIR / "decision_engine_test_set.csv", index=False)

    latent_summary = summarize_latent_economic(events)
    realized_summary = summarize_realized(events)
    operational_summary = summarize_operational(events)

    report = {
        "label": "SYNTHETIC COUNTERFACTUAL EVALUATION -- does not measure real Razorpay recovery performance.",
        "threshold_selection_validation_only": {"candidates": search_results, "chosen_threshold_rs": chosen_threshold},
        "latent_economic": latent_summary,
        "realized_counterfactual": realized_summary,
        "operational": operational_summary,
    }
    with open(REPORTS_DIR / "decision_engine_evaluation.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Test events: {len(events)}")
    print()
    print("LATENT ECONOMIC (synthetic ground truth):")
    for name in POLICY_NAMES:
        s = latent_summary[name]
        print(f"  {name:28s} total=Rs{s['total_latent_value_rs']:>10.2f} avg/event=Rs{s['avg_latent_value_per_event_rs']:>7.2f} vs_fixed=Rs{s['improvement_vs_fixed_retry_rs']:>+9.2f} regret_vs_oracle=Rs{s['regret_vs_oracle_rs']:>8.2f}")
    print()
    print("REALIZED COUNTERFACTUAL (stochastic sampled outcome):")
    for name in POLICY_NAMES:
        s = realized_summary[name]
        print(f"  {name:28s} recovered=Rs{s['total_recovered_rs']:>10.2f} rate={s['recovery_rate']:.4f} incremental=Rs{s['incremental_rs_vs_fixed_retry']:>+9.2f}")
    print()
    print("OPERATIONAL (policy-v3 decision engine only):")
    for k, v in operational_summary.items():
        print(f"  {k}: {v}")
    print()
    print(f"Reports written to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
