"""
Day-10 section 1: reproduce and diagnose Day-9's fallback behaviour, on
BOTH validation and test splits, before writing a single line of new policy
logic.

"SYNTHETIC COUNTERFACTUAL EVALUATION" -- same disclaimer as every prior
day: every number here comes from data/raw/counterfactual_outcomes.csv, a
hand-designed simulation, not real Razorpay recovery performance.

    ./venv/bin/python evaluation/diagnose_day9_fallback.py

Reuses evaluation/evaluate_decision_engine.py::evaluate_events (Day 9,
unmodified) to generate per-event records for BOTH splits -- that function
already computes, per event: Model B's own pick (`day8_model_b_no_abstention`),
Rule-Based's pick (`rule_based`), Day-9's actual pick (`day9_decision_engine`),
Oracle's pick, each one's latent value, plus `day9__is_fallback` /
`day9__decision_margin`. Nothing here re-implements that logic; it only
aggregates it into the specific diagnostics brief section 1 asks for.
"""
from __future__ import annotations

import json

import pandas as pd

from evaluation.evaluate_decision_engine import evaluate_events
from model.candidate_preprocessing import split_candidate_dataset
from model.latent_target_preprocessing import PROJECT_ROOT, build_candidate_level_dataset_with_latent_targets
from model.train_latent_target_model import load_latent_target_model
from policy.decision_engine import DEFAULT_ABSTENTION_THRESHOLD_RS

REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"


def diagnose_split(events: pd.DataFrame, split_name: str) -> dict:
    n = len(events)
    is_fallback = events["day9__is_fallback"]
    is_no_action = events["day9_decision_engine__selected_candidate_type"] == "NO_ACTION"
    is_model_direct = (~is_fallback) & (~is_no_action)

    latent_model_alone = float(events["day8_model_b_no_abstention__latent_value_selected"].sum())
    latent_day9 = float(events["day9_decision_engine__latent_value_selected"].sum())
    latent_oracle = float(events["oracle_policy__latent_value_selected"].sum())

    # Value lost/gained SPECIFICALLY because of fallback: restricted to the
    # events where Day-9 actually fell back, compare what Day-9 selected
    # against what Model B alone would have selected for that same event.
    fb = events[is_fallback]
    latent_model_on_fallback_events = float(fb["day8_model_b_no_abstention__latent_value_selected"].sum())
    latent_day9_on_fallback_events = float(fb["day9_decision_engine__latent_value_selected"].sum())
    value_lost_to_fallback = latent_model_on_fallback_events - latent_day9_on_fallback_events

    margins = events["day9__decision_margin"].dropna()

    fallback_rows = []
    for _, row in fb.iterrows():
        model_pick = row["day8_model_b_no_abstention__selected_candidate_type"]
        rule_pick = row["rule_based__selected_candidate_type"]
        oracle_pick = row["oracle_policy__selected_candidate_type"]
        fallback_rows.append(
            {
                "event_id": row["event_id"],
                "model_b_pick": model_pick,
                "model_b_pick_matches_final": row["day9_decision_engine__selected_candidate_type"] == model_pick,
                "rule_pick": rule_pick,
                "rule_pick_matches_final": row["day9_decision_engine__selected_candidate_type"] == rule_pick,
                "oracle_pick": oracle_pick,
                "final_pick": row["day9_decision_engine__selected_candidate_type"],
                "latent_value_model_b_pick": float(row["day8_model_b_no_abstention__latent_value_selected"]),
                "latent_value_rule_pick": float(row["day9_decision_engine__latent_value_selected"]),
                "latent_value_oracle_pick": float(row["oracle_policy__latent_value_selected"]),
                "model_was_actually_right": model_pick == oracle_pick,
                "rule_was_actually_right": rule_pick == oracle_pick,
            }
        )

    n_fallback_events_where_model_was_oracle = sum(1 for r in fallback_rows if r["model_was_actually_right"])
    n_fallback_events_where_rule_was_oracle = sum(1 for r in fallback_rows if r["rule_was_actually_right"])

    verdict = "HURTS" if value_lost_to_fallback > 0 else ("HELPS" if value_lost_to_fallback < 0 else "NEUTRAL")

    return {
        "split": split_name,
        "n_events": n,
        "n_model_direct": int(is_model_direct.sum()),
        "n_fallback": int(is_fallback.sum()),
        "n_no_action": int(is_no_action.sum()),
        "latent_value_model_b_alone_rs": latent_model_alone,
        "latent_value_after_day9_fallback_rs": latent_day9,
        "latent_value_oracle_rs": latent_oracle,
        "latent_value_lost_to_fallback_rs": value_lost_to_fallback,
        "verdict_fallback_helps_or_hurts": verdict,
        "decision_margin_distribution": {
            "count": int(margins.count()), "mean": float(margins.mean()) if len(margins) else None,
            "std": float(margins.std()) if len(margins) else None, "min": float(margins.min()) if len(margins) else None,
            "p25": float(margins.quantile(0.25)) if len(margins) else None, "median": float(margins.median()) if len(margins) else None,
            "p75": float(margins.quantile(0.75)) if len(margins) else None, "max": float(margins.max()) if len(margins) else None,
        },
        "on_fallback_events__n_where_model_pick_was_oracle_pick": n_fallback_events_where_model_was_oracle,
        "on_fallback_events__n_where_rule_pick_was_oracle_pick": n_fallback_events_where_rule_was_oracle,
        "on_fallback_events__n_total": len(fallback_rows),
        "fallback_event_detail": fallback_rows,
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_latent_target_model("value")
    df = build_candidate_level_dataset_with_latent_targets()
    train_df, val_df, test_df = split_candidate_dataset(df)
    del train_df

    print("=== Day-10 section 1: diagnosing Day-9 fallback (SYNTHETIC COUNTERFACTUAL EVALUATION) ===")
    print(f"(reproducing Day-9's own frozen threshold Rs{DEFAULT_ABSTENTION_THRESHOLD_RS:.2f} on both splits)")
    print()

    report = {}
    for split_name, split_df in [("validation", val_df), ("test", test_df)]:
        events = evaluate_events(split_df, model, frozen_threshold=DEFAULT_ABSTENTION_THRESHOLD_RS)
        diagnosis = diagnose_split(events, split_name)
        report[split_name] = diagnosis

        print(f"--- {split_name.upper()} (n={diagnosis['n_events']}) ---")
        print(f"  model-direct: {diagnosis['n_model_direct']}  fallback: {diagnosis['n_fallback']}  no_action: {diagnosis['n_no_action']}")
        print(f"  latent value -- Model B alone: Rs{diagnosis['latent_value_model_b_alone_rs']:.2f}  after Day-9 fallback: Rs{diagnosis['latent_value_after_day9_fallback_rs']:.2f}  Oracle: Rs{diagnosis['latent_value_oracle_rs']:.2f}")
        print(f"  latent value lost SPECIFICALLY to fallback: Rs{diagnosis['latent_value_lost_to_fallback_rs']:.2f}  -> fallback {diagnosis['verdict_fallback_helps_or_hurts']}")
        md = diagnosis["decision_margin_distribution"]
        print(f"  decision margin distribution: mean=Rs{md['mean']:.2f} median=Rs{md['median']:.2f} p25=Rs{md['p25']:.2f} p75=Rs{md['p75']:.2f} min=Rs{md['min']:.2f} max=Rs{md['max']:.2f}")
        print(f"  on fallback events: Model B's own pick was Oracle's pick {diagnosis['on_fallback_events__n_where_model_pick_was_oracle_pick']}/{diagnosis['on_fallback_events__n_total']}; Rule's pick was Oracle's pick {diagnosis['on_fallback_events__n_where_rule_pick_was_oracle_pick']}/{diagnosis['on_fallback_events__n_total']}")
        print()

    with open(REPORTS_DIR / "day9_fallback_diagnosis.json", "w") as f:
        json.dump({"label": "SYNTHETIC COUNTERFACTUAL EVALUATION -- does not measure real Razorpay recovery performance.", **report}, f, indent=2, default=str)
    print(f"Report written to {REPORTS_DIR / 'day9_fallback_diagnosis.json'}")


if __name__ == "__main__":
    main()
