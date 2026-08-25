"""
Day-10 evaluation: validation-only search over (margin_threshold,
fallback_mode, fallback_advantage_threshold), then a single frozen test-set
run comparing 6 policies (Fixed Retry, Rule-Based, Model B alone, Day-9,
Day-10, Oracle).

"SYNTHETIC COUNTERFACTUAL EVALUATION" -- every number below comes from
data/raw/counterfactual_outcomes.csv, a hand-designed simulation. It does
not measure real Razorpay recovery performance.

    ./venv/bin/python evaluation/evaluate_decision_engine_v4.py

Two phases, strictly ordered (brief section 2/6 -- never tune on test):

  1. CONFIGURATION SELECTION on VALIDATION ONLY (brief section 2). Searches
     margin_threshold in {0,5,10,15,20,25,50,75,100} crossed with all 4
     fallback_mode values; for KEEP_MODEL_UNLESS_RULE_HAS_CLEAR_ADVANTAGE
     additionally searches fallback_advantage_threshold over the same set
     (brief section 3 gives no explicit candidate set for this knob, so the
     same 9-value set is reused for consistency -- documented here, not
     silently assumed). Scored by total LATENT net value selected on
     validation. Ties broken by (a) fewer fallbacks -- prefer trusting
     Model B when the numbers are otherwise equal, then (b) fewer
     no-actions, both deterministic and documented, not test-informed.
  2. TEST evaluation, run once, comparing 6 policies (brief section 4/6):
     Fixed Retry, Rule-Based, Model B alone (no abstention), Day-9 original
     fallback (frozen Rs10 threshold, kept exactly as shipped -- brief
     section 4 "Do NOT delete Day-9 results"), Day-10 improved fallback
     (frozen config from phase 1), Oracle.
"""
from __future__ import annotations

import json

import pandas as pd

from classification.rules import classify
from evaluation.statistics import bootstrap_delta_ci, mcnemar_test
from model.candidate_preprocessing import split_candidate_dataset
from model.latent_target_preprocessing import LATENT_VALUE_COLUMN, PROJECT_ROOT, build_candidate_level_dataset_with_latent_targets
from model.train_latent_target_model import load_latent_target_model
from policy.baselines import fixed_retry_baseline, rule_based_baseline
from policy.costs import DEFAULT_COSTS, cost_for_candidate
from policy.decision_engine import DEFAULT_ABSTENTION_THRESHOLD_RS, NO_ACTION, decide_engine
from policy.decision_engine_v4 import (
    DEFAULT_FALLBACK_ADVANTAGE_THRESHOLD_RS,
    DEFAULT_FALLBACK_MODE,
    DEFAULT_MARGIN_THRESHOLD_RS,
    FALLBACK_MODE_KEEP_UNLESS_CLEAR,
    FALLBACK_MODES,
    decide_engine_v4,
)
from policy.economics import compute_recovery_economics
from policy.guardrails import validate_candidate
from policy.retry_candidates import Candidate

REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"
MARGIN_THRESHOLD_CANDIDATES = [0, 5, 10, 15, 20, 25, 50, 75, 100]
FALLBACK_ADVANTAGE_CANDIDATES = [0, 5, 10, 15, 20, 25, 50, 75, 100]  # reused from the same set -- see module docstring

POLICY_NAMES = ["fixed_retry", "rule_based", "day8_model_b_alone", "day9_original_fallback", "day10_improved_fallback", "oracle_policy"]

# The single headline comparison the specification and dashboard care about
# most (brief: "the agent vs. Razorpay's own baseline" / "clearing all three
# baselines, not just the easiest one"). Both the McNemar test and the
# bootstrap CI below are computed for this one pair -- see
# evaluation/statistics.py's module docstring for why McNemar is restricted
# to the binary outcome only.
DEPLOYED_POLICY_NAME = "day10_improved_fallback"
HEADLINE_BASELINE_NAME = "fixed_retry"
BOOTSTRAP_N_RESAMPLES = 10000
BOOTSTRAP_SEED = 42
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

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


def _run_v4_for_all_events(df: pd.DataFrame, model: dict, margin_threshold: float, fallback_mode: str, fallback_advantage_threshold: float) -> pd.DataFrame:
    records = []
    for event_id, group in df.groupby("event_id"):
        first = group.iloc[0]
        subscription_id, failure_timestamp, amount = first["subscription_id"], first["failure_timestamp"], float(first["amount"])
        classification_bucket = classify(None, first["error_reason"]).bucket
        decision = decide_engine_v4(
            event_id, subscription_id, failure_timestamp, amount, classification_bucket, _event_context(first),
            costs=DEFAULT_COSTS, margin_threshold=margin_threshold, fallback_mode=fallback_mode,
            fallback_advantage_threshold=fallback_advantage_threshold, model=model,
        )
        latent_by_type = _latent_value_lookup(group)
        records.append(
            {
                "event_id": event_id,
                "selected_candidate_type": decision.selected_candidate_type,
                "decision_source": decision.decision_source,
                "latent_value_selected": latent_by_type.get(decision.selected_candidate_type, 0.0) if decision.selected_candidate_type != NO_ACTION else 0.0,
            }
        )
    return pd.DataFrame(records)


def select_day10_configuration_on_validation(val_df: pd.DataFrame, model: dict) -> tuple[dict, dict]:
    """VALIDATION-ONLY search (brief section 2) -- test is never touched
    here. Returns (chosen_config, all_results). chosen_config is a dict with
    keys margin_threshold / fallback_mode / fallback_advantage_threshold."""
    results = {}
    for margin_threshold in MARGIN_THRESHOLD_CANDIDATES:
        for fallback_mode in FALLBACK_MODES:
            advantage_values = FALLBACK_ADVANTAGE_CANDIDATES if fallback_mode == FALLBACK_MODE_KEEP_UNLESS_CLEAR else [0.0]
            for fallback_advantage_threshold in advantage_values:
                run = _run_v4_for_all_events(val_df, model, float(margin_threshold), fallback_mode, float(fallback_advantage_threshold))
                key = (float(margin_threshold), fallback_mode, float(fallback_advantage_threshold))
                n_fallback = int((run["decision_source"] == "rule_based_fallback").sum())
                n_no_action = int((run["selected_candidate_type"] == NO_ACTION).sum())
                results[key] = {
                    "margin_threshold": float(margin_threshold), "fallback_mode": fallback_mode,
                    "fallback_advantage_threshold": float(fallback_advantage_threshold),
                    "total_latent_value_selected_rs": float(run["latent_value_selected"].sum()),
                    "avg_latent_value_per_event_rs": float(run["latent_value_selected"].mean()),
                    "n_fallback": n_fallback, "n_no_action": n_no_action,
                }

    # Primary: total latent value (desc). Ties: fewer fallbacks, then fewer no-actions (documented, deterministic).
    best_key = max(results, key=lambda k: (results[k]["total_latent_value_selected_rs"], -results[k]["n_fallback"], -results[k]["n_no_action"]))
    chosen = {k: results[best_key][k] for k in ("margin_threshold", "fallback_mode", "fallback_advantage_threshold")}
    return chosen, results


# ---------------------------------------------------------------------------
# Test-set evaluation across all 6 policies
# ---------------------------------------------------------------------------

def evaluate_events_v4(test_df: pd.DataFrame, model: dict, day10_config: dict) -> pd.DataFrame:
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
        model_alone_decision = decide_engine(event_id, subscription_id, failure_timestamp, amount, classification_bucket, _event_context(first), costs=DEFAULT_COSTS, abstention_threshold=float("-inf"), model=model)
        day9_decision = decide_engine(event_id, subscription_id, failure_timestamp, amount, classification_bucket, _event_context(first), costs=DEFAULT_COSTS, abstention_threshold=DEFAULT_ABSTENTION_THRESHOLD_RS, model=model)
        day10_decision = decide_engine_v4(
            event_id, subscription_id, failure_timestamp, amount, classification_bucket, _event_context(first), costs=DEFAULT_COSTS,
            margin_threshold=day10_config["margin_threshold"], fallback_mode=day10_config["fallback_mode"],
            fallback_advantage_threshold=day10_config["fallback_advantage_threshold"], model=model,
        )
        oracle_sel = max((ct for ct, valid in valid_mask.items() if valid), key=lambda ct: latent_value[ct], default=NO_ACTION)

        selections = {
            "fixed_retry": fixed_sel, "rule_based": rule_sel,
            "day8_model_b_alone": model_alone_decision.selected_candidate_type,
            "day9_original_fallback": day9_decision.selected_candidate_type,
            "day10_improved_fallback": day10_decision.selected_candidate_type,
            "oracle_policy": oracle_sel,
        }
        for policy_name in POLICY_NAMES:
            selected = selections[policy_name]
            record[f"{policy_name}__selected_candidate_type"] = selected
            record[f"{policy_name}__realized_recovered"] = bool(realized_recovered.get(selected, False)) if selected != NO_ACTION else False
            record[f"{policy_name}__realized_amount_recovered"] = float(realized_amount.get(selected, 0.0)) if selected != NO_ACTION else 0.0
            record[f"{policy_name}__latent_value_selected"] = float(latent_value.get(selected, 0.0)) if selected != NO_ACTION else 0.0

        # Decision-trace fields (brief section 7)
        model_best_type = model_alone_decision.selected_candidate_type
        model_best_score = next((s for s in day10_decision.candidate_scores if s.candidate_type == model_best_type and s.valid), None)
        rule_score = next((s for s in day10_decision.candidate_scores if s.candidate_type == rule_sel and s.valid), None)
        record["trace__model_b_best_candidate"] = model_best_type
        record["trace__model_b_predicted_value"] = model_best_score.predicted_recovery_value if model_best_score else None
        record["trace__model_b_net_value"] = model_best_score.expected_net_value if model_best_score else None
        record["trace__rule_candidate"] = rule_sel
        record["trace__rule_predicted_value"] = rule_score.predicted_recovery_value if rule_score else None
        record["trace__rule_net_value"] = rule_score.expected_net_value if rule_score else None
        record["trace__net_value_difference"] = (rule_score.expected_net_value - model_best_score.expected_net_value) if (rule_score and model_best_score) else None
        record["trace__day9_margin"] = day9_decision.decision_margin
        record["trace__day10_margin"] = day10_decision.decision_margin
        record["trace__day10_margin_threshold"] = day10_decision.margin_threshold_used
        record["trace__day10_fallback_advantage_threshold"] = day10_decision.fallback_advantage_threshold
        record["trace__day9_decision_source"] = day9_decision.decision_source
        record["trace__day10_decision_source"] = day10_decision.decision_source
        record["trace__oracle_pick"] = oracle_sel
        record["trace__latent_value_final_selection"] = float(latent_value.get(day10_decision.selected_candidate_type, 0.0)) if day10_decision.selected_candidate_type != NO_ACTION else 0.0
        record["trace__latent_value_model_b_selection"] = float(latent_value.get(model_best_type, 0.0)) if model_best_type != NO_ACTION else 0.0

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


def summarize_operational(events: pd.DataFrame, day10_config: dict) -> dict:
    n = len(events)
    n_actions = int((events["day10_improved_fallback__selected_candidate_type"] != NO_ACTION).sum())
    n_no_action = n - n_actions
    n_fallback = int((events["trace__day10_decision_source"] == "rule_based_fallback").sum())
    n_model_direct = int((events["trace__day10_decision_source"] == "day8_model_b").sum())
    margins = events["trace__day10_margin"].dropna()
    advantages = events["trace__net_value_difference"].dropna()
    n_changed_by_fallback = int((events["day10_improved_fallback__selected_candidate_type"] != events["day8_model_b_alone__selected_candidate_type"]).sum())
    return {
        "config": day10_config,
        "n_events": n,
        "model_b_direct_selections": n_model_direct,
        "fallback_count": n_fallback,
        "no_action_count": n_no_action,
        "fallback_percentage": round(100 * n_fallback / n, 2) if n else 0.0,
        "average_decision_margin_rs": float(margins.mean()) if len(margins) else None,
        "average_fallback_advantage_rs": float(advantages.mean()) if len(advantages) else None,
        "n_decisions_changed_by_fallback_vs_model_b_alone": n_changed_by_fallback,
        "decision_source_distribution": events["trace__day10_decision_source"].value_counts().to_dict(),
    }


def summarize_statistical_tests(
    events: pd.DataFrame,
    *,
    deployed_policy: str = DEPLOYED_POLICY_NAME,
    baseline_policy: str = HEADLINE_BASELINE_NAME,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
) -> dict:
    """
    Statistical significance for the SAME headline comparison
    (`realized_summary`'s `incremental_rs_vs_fixed_retry`) already reported
    elsewhere in this script -- no new evaluation population, no new outcome
    definition. `events` is the exact per-event test-set DataFrame
    `evaluate_events_v4` already built; the two columns used below
    (`{policy}__realized_recovered`, `{policy}__realized_amount_recovered`)
    are the SAME paired outcome columns `summarize_realized` already reads.

    McNemar's test uses ONLY the paired binary `realized_recovered` column
    (never the continuous ₹ amounts -- see evaluation/statistics.py). The
    bootstrap CI uses the continuous `realized_amount_recovered` column.
    Both remain part of the SYNTHETIC COUNTERFACTUAL EVALUATION this whole
    report is (see `report["label"]`) and do not establish real-world
    production superiority.
    """
    mcnemar_result = mcnemar_test(
        events[f"{deployed_policy}__realized_recovered"].tolist(),
        events[f"{baseline_policy}__realized_recovered"].tolist(),
        policy_a=deployed_policy, policy_b=baseline_policy, exact=True,
    )
    bootstrap_result = bootstrap_delta_ci(
        events[f"{deployed_policy}__realized_amount_recovered"].to_numpy(),
        events[f"{baseline_policy}__realized_amount_recovered"].to_numpy(),
        policy_a=deployed_policy, policy_b=baseline_policy, metric_name="realized_rs_recovered",
        n_resamples=n_resamples, seed=seed, confidence_level=confidence_level,
    )
    return {
        "population": {
            "held_out_split": "test",
            "n_events": len(events),
            "source": "evaluation/evaluate_decision_engine_v4.py::evaluate_events_v4 -- the SAME test-set events and outcome columns the rest of this report's realized_counterfactual section uses",
        },
        "mcnemar": mcnemar_result.to_dict(),
        "bootstrap_ci": bootstrap_result.to_dict(),
        "interpretation_note": (
            "Both tests quantify statistical uncertainty WITHIN this SYNTHETIC COUNTERFACTUAL "
            "EVALUATION's held-out test split (n={n}) -- they do not establish, and must never be cited "
            "as establishing, real-world Razorpay production superiority. McNemar's test is restricted to "
            "the paired binary realized_recovered outcome (never applied to the continuous ₹ amounts, "
            "which is what the bootstrap CI is for instead)."
        ).format(n=len(events)),
    }


def summarize_economics(events: pd.DataFrame, realized_summary: dict) -> dict:
    """
    Adds the specification-required GMV/fee/net split (policy/economics.py)
    on top of the EXISTING `realized_summary["...']["total_recovered_rs"]`
    figure -- never recomputes recovered GMV, never blends it with the fee
    take. `intervention_cost` is summed from the EXISTING
    `policy/costs.py::cost_for_candidate` over exactly the real
    (non-NO_ACTION) actions each policy actually selected in this same test
    run -- the same cost model policy/decision_engine.py already uses at
    decision time, applied here to the REALIZED per-policy selections for a
    report-level total, not a new cost model.
    """
    economics = {}
    for name in POLICY_NAMES:
        total_intervention_cost = float(
            sum(cost_for_candidate(t, DEFAULT_COSTS) for t in events[f"{name}__selected_candidate_type"] if t != NO_ACTION)
        )
        recovered_gmv = realized_summary[name]["total_recovered_rs"]
        economics[name] = compute_recovery_economics(recovered_gmv, total_intervention_cost).to_dict()
    return economics


def print_decision_trace(events: pd.DataFrame, n: int = 12) -> None:
    print(f"=== Decision trace (first {n} test events) ===")
    cols = [
        "event_id", "trace__model_b_best_candidate", "trace__model_b_predicted_value", "trace__rule_candidate",
        "trace__rule_predicted_value", "trace__net_value_difference", "trace__day10_margin_threshold",
        "day10_improved_fallback__selected_candidate_type", "trace__day10_decision_source", "trace__oracle_pick",
        "trace__latent_value_final_selection", "trace__latent_value_model_b_selection",
    ]
    for _, row in events[cols].head(n).iterrows():
        print(
            f"  event={row['event_id']:<6} model_b_best={row['trace__model_b_best_candidate']:<20} "
            f"model_b_value=Rs{row['trace__model_b_predicted_value']:<8.2f} rule_pick={row['trace__rule_candidate']:<20} "
            f"rule_value=Rs{row['trace__rule_predicted_value']:<8.2f} diff=Rs{row['trace__net_value_difference']:<8.2f} "
            f"threshold=Rs{row['trace__day10_margin_threshold']:<6.2f} final={row['day10_improved_fallback__selected_candidate_type']:<20} "
            f"source={row['trace__day10_decision_source']:<20} oracle={row['trace__oracle_pick']:<20} "
            f"latent_final=Rs{row['trace__latent_value_final_selection']:<8.2f} latent_model_b=Rs{row['trace__latent_value_model_b_selection']:.2f}"
        )
    print()


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_latent_target_model("value")
    df = build_candidate_level_dataset_with_latent_targets()
    train_df, val_df, test_df = split_candidate_dataset(df)
    del train_df

    print("=== Phase 1: Day-10 configuration search on VALIDATION ONLY ===")
    chosen_config, search_results = select_day10_configuration_on_validation(val_df, model)
    ranked = sorted(search_results.values(), key=lambda r: r["total_latent_value_selected_rs"], reverse=True)
    for r in ranked[:15]:
        is_chosen = r["margin_threshold"] == chosen_config["margin_threshold"] and r["fallback_mode"] == chosen_config["fallback_mode"] and r["fallback_advantage_threshold"] == chosen_config["fallback_advantage_threshold"]
        marker = " <-- selected" if is_chosen else ""
        print(f"  margin=Rs{r['margin_threshold']:<6} mode={r['fallback_mode']:<40} adv=Rs{r['fallback_advantage_threshold']:<6} total_latent=Rs{r['total_latent_value_selected_rs']:>10.2f} n_fallback={r['n_fallback']:<3} n_no_action={r['n_no_action']}{marker}")
    print(f"  (showing top 15 of {len(search_results)} configurations searched)")
    print(f"  CHOSEN: {chosen_config}")
    frozen_matches = (
        chosen_config["margin_threshold"] == DEFAULT_MARGIN_THRESHOLD_RS
        and chosen_config["fallback_mode"] == DEFAULT_FALLBACK_MODE
        and chosen_config["fallback_advantage_threshold"] == DEFAULT_FALLBACK_ADVANTAGE_THRESHOLD_RS
    )
    print(f"  Frozen config in policy/decision_engine_v4.py matches this search: {frozen_matches}")
    print()

    print("=== Phase 2: TEST evaluation (run once, frozen config) ===")
    events = evaluate_events_v4(test_df, model, chosen_config)
    events.to_csv(REPORTS_DIR / "decision_engine_v4_test_set.csv", index=False)

    latent_summary = summarize_latent_economic(events)
    realized_summary = summarize_realized(events)
    operational_summary = summarize_operational(events, chosen_config)
    statistical_tests = summarize_statistical_tests(events)
    economics_summary = summarize_economics(events, realized_summary)

    report = {
        "label": "SYNTHETIC COUNTERFACTUAL EVALUATION -- does not measure real Razorpay recovery performance.",
        "day10_configuration_selection_validation_only": {"chosen": chosen_config, "top_15_by_latent_value": ranked[:15], "n_configurations_searched": len(search_results)},
        "latent_economic": latent_summary,
        "realized_counterfactual": realized_summary,
        "operational": operational_summary,
        "statistical_tests": statistical_tests,
        "economics": economics_summary,
    }
    with open(REPORTS_DIR / "decision_engine_v4_evaluation.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Test events: {len(events)}")
    print()
    print("LATENT ECONOMIC (synthetic ground truth):")
    for name in POLICY_NAMES:
        s = latent_summary[name]
        print(f"  {name:26s} total=Rs{s['total_latent_value_rs']:>10.2f} avg/event=Rs{s['avg_latent_value_per_event_rs']:>7.2f} vs_fixed=Rs{s['improvement_vs_fixed_retry_rs']:>+9.2f} regret_vs_oracle=Rs{s['regret_vs_oracle_rs']:>8.2f}")
    print()
    print("REALIZED COUNTERFACTUAL (stochastic sampled outcome -- n=60, not claimed statistically significant):")
    for name in POLICY_NAMES:
        s = realized_summary[name]
        print(f"  {name:26s} recovered=Rs{s['total_recovered_rs']:>10.2f} rate={s['recovery_rate']:.4f} incremental=Rs{s['incremental_rs_vs_fixed_retry']:>+9.2f}")
    print()
    print("OPERATIONAL (Day-10 decision engine only):")
    for k, v in operational_summary.items():
        print(f"  {k}: {v}")
    print()
    print(f"STATISTICAL TESTS ({DEPLOYED_POLICY_NAME} vs {HEADLINE_BASELINE_NAME}, synthetic held-out test set, n={len(events)}):")
    mc = statistical_tests["mcnemar"]
    print(f"  McNemar ({mc['method']}): b={mc['only_a_recovered']} c={mc['only_b_recovered']} p_value={mc['p_value']:.4f}")
    bs = statistical_tests["bootstrap_ci"]
    print(f"  Bootstrap CI ({bs['confidence_level']:.0%}, {bs['n_resamples']} resamples, seed={bs['seed']}): delta_rs={bs['point_estimate']:+.2f} [{bs['lower_bound']:+.2f}, {bs['upper_bound']:+.2f}]")
    print()
    print("ECONOMICS (GMV / intervention cost / Razorpay fee take / net -- see policy/economics.py):")
    for name in POLICY_NAMES:
        e = economics_summary[name]
        print(f"  {name:26s} gmv=Rs{e['recovered_gmv']:>9.2f} intervention_cost=Rs{e['intervention_cost']:>7.2f} razorpay_fee_take=Rs{e['razorpay_fee_take']:>8.2f} net=Rs{e['net_recovery_value']:>9.2f}")
    print()
    print_decision_trace(events, n=12)
    print(f"Reports written to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
