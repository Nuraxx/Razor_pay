"""
Held-out evaluation of the unified ML model's candidate-selection policy
against baselines, per domain and overall.

IMPORTANT HONESTY NOTE: every number this script prints comes from the
SYNTHETIC dataset model/unified_model.py::_make_training_data generates --
a simulated (event, candidate) -> recovered outcome, not a real Razorpay
recovery. "recovered_amount" below is "amount if the simulated outcome for
the SELECTED candidate was recovered=True, else 0" -- it is a backtest
against a simulated label, not authoritative payment confirmation. Nothing
here should be read as a claim about real merchant revenue.

Evaluation unit: one row per TEST-split ENTITY (not per candidate) -- for
each entity, each policy (ML / baseline) picks exactly one candidate from
that domain's valid set, and we look up the SAME entity's simulated outcome
for that specific candidate. TRAIN/VALIDATION data is never touched here
(only the TEST-split entities produced by the same
model.unified_model._entity_level_split the training run itself used).

Baselines:
  - "first_candidate": always picks the first candidate in
    CANDIDATE_SPACE[event_type] (a naive fixed default -- no ranking at all).
  - "random_candidate": picks uniformly at random among valid candidates
    (seeded, reproducible) -- a lower bar than any real policy should clear.
  - "rule_baseline": the actual deterministic rule module this domain's
    live policy dispatcher falls back to when ML is unavailable
    (policy/checkout_rules.py, policy/receivables_rules.py) -- computed only
    for the 2 domains whose rule function's inputs are fully present in the
    flat per-entity synthetic schema (cart/inactivity/payment_method for
    checkout; days_overdue/dispute/promise flags for receivables).
    mandate_failed's and promise_to_pay_broken's rule modules are STATEFUL
    (they need a sequence position / prior-attempt count the flat synthetic
    schema doesn't generate one-to-one) and payment_failed's rule module
    needs a specific, officially-documented Razorpay error_reason string
    the synthetic `failure_reason` values only partially overlap with --
    forcing defaulted/guessed inputs into those 3 would produce a
    misleading rule-baseline number, so they're left out rather than faked.

"Oracle" metrics (top-1 accuracy, NDCG, regret) use `_true_probability` --
the DETERMINISTIC, feature-driven mean probability the generator computed
BEFORE the small per-draw idiosyncratic noise (see model/unified_model.py's
_make_training_data docstring) -- i.e. the best any feature-based model
could theoretically achieve, since that per-draw noise is by construction
unpredictable from any feature. This column is never given to the model as
a feature; it is read here purely for evaluation.

Usage:
    ./venv/bin/python -m evaluation.evaluate_unified_model
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from model.unified_model import (
    CANDIDATE_SPACE,
    SUPPORTED_EVENT_TYPES,
    _entity_level_split,
    _make_training_data,
    load_unified_model,
    score_event_candidates,
)

REPORT_PATH_JSON = "model/reports/unified_model_evaluation_report.json"

INTERVENTION_COST_RS = 5.0  # matches policy/costs.py's existing retry_cost convention


def _row_to_event(row: pd.Series) -> dict:
    return row.drop(labels=["candidate_type", "recovered", "entity_key", "_true_probability"]).to_dict()


def _pick_first_candidate(event_type: str) -> str:
    return CANDIDATE_SPACE[event_type][0]


def _pick_random_candidate(event_type: str, rng: np.random.Generator) -> str:
    return rng.choice(CANDIDATE_SPACE[event_type])


def _pick_rule_baseline_candidate(event_type: str, entity_row: pd.Series) -> str | None:
    """The REAL deterministic rule module each domain's live dispatcher uses
    -- see module docstring for why only these 2 domains are well-posed
    against the flat synthetic schema."""
    if event_type == "checkout_abandoned":
        from policy.checkout_rules import decide_checkout_recovery

        result = decide_checkout_recovery(
            cart_amount=float(entity_row["amount"]),
            inactivity_minutes=float(entity_row["checkout_age_minutes"]),
            previous_outreach_count=0,
            payment_method=entity_row["payment_method"],
        )
        return result.candidate_type
    if event_type == "receivable_overdue":
        from policy.receivables_rules import decide_receivable_action

        result = decide_receivable_action(
            days_overdue=int(entity_row["days_overdue"]), is_disputed=False, has_active_promise=False,
        )
        return result.candidate_type
    return None


def _true_prob_for(domain_df: pd.DataFrame, entity_key: str, candidate: str) -> float | None:
    row = domain_df[(domain_df["entity_key"] == entity_key) & (domain_df["candidate_type"] == candidate)]
    return float(row.iloc[0]["_true_probability"]) if not row.empty else None


def _ndcg(true_relevance_in_predicted_order: list[float]) -> float:
    """Standard NDCG: DCG of the true relevance values, in the order the
    model predicted, normalized by the IDEAL (true-probability-sorted) DCG."""
    dcg = sum(rel / np.log2(pos + 2) for pos, rel in enumerate(true_relevance_in_predicted_order))
    idcg = sum(rel / np.log2(pos + 2) for pos, rel in enumerate(sorted(true_relevance_in_predicted_order, reverse=True)))
    return float(dcg / idcg) if idcg > 0 else 1.0


def _evaluate_domain(domain_df: pd.DataFrame, event_type: str, model: dict, rng: np.random.Generator) -> dict:
    entities = domain_df.drop_duplicates(subset=["entity_key"])

    policy_names = ["unified_ml", "first_candidate", "random_candidate"]
    has_rule_baseline = event_type in ("checkout_abandoned", "receivable_overdue")
    if has_rule_baseline:
        policy_names.append("rule_baseline")
    policies: dict[str, list[dict]] = {name: [] for name in policy_names}

    top1_hits = 0
    ndcg_values: list[float] = []
    regret_prob_values: list[float] = []
    regret_value_rs: list[float] = []
    n_multi_candidate_entities = 0

    for _, entity_row in entities.iterrows():
        event = _row_to_event(entity_row)
        amount = float(entity_row["amount"])
        entity_key = entity_row["entity_key"]

        scores = score_event_candidates(event, model)
        if not scores:
            continue
        ml_candidate = max(scores, key=lambda s: s["predicted_recovery_value"])["candidate_type"]

        candidates_in_domain = CANDIDATE_SPACE[event_type]
        true_probs = {c: _true_prob_for(domain_df, entity_key, c) for c in candidates_in_domain}
        if all(v is not None for v in true_probs.values()) and len(candidates_in_domain) > 1:
            n_multi_candidate_entities += 1
            true_best_candidate = max(true_probs, key=true_probs.get)
            top1_hits += int(ml_candidate == true_best_candidate)

            predicted_order = [s["candidate_type"] for s in sorted(scores, key=lambda s: s["predicted_recovery_probability"], reverse=True)]
            ndcg_values.append(_ndcg([true_probs[c] for c in predicted_order]))

            oracle_prob = true_probs[true_best_candidate]
            selected_prob = true_probs[ml_candidate]
            regret_prob_values.append(oracle_prob - selected_prob)
            regret_value_rs.append((oracle_prob - selected_prob) * amount)

        first_candidate = _pick_first_candidate(event_type)
        random_candidate = _pick_random_candidate(event_type, rng)
        rule_candidate = _pick_rule_baseline_candidate(event_type, entity_row) if has_rule_baseline else None

        for policy_name, candidate in (
            ("unified_ml", ml_candidate), ("first_candidate", first_candidate),
            ("random_candidate", random_candidate), ("rule_baseline", rule_candidate),
        ):
            if candidate is None or policy_name not in policies:
                continue
            outcome_row = domain_df[(domain_df["entity_key"] == entity_key) & (domain_df["candidate_type"] == candidate)]
            if outcome_row.empty:
                continue
            recovered = bool(outcome_row.iloc[0]["recovered"])
            policies[policy_name].append({
                "candidate": candidate, "recovered": recovered, "amount": amount,
                "recovered_amount": amount if recovered else 0.0, "cost": INTERVENTION_COST_RS,
            })

    results: dict[str, dict | None] = {}
    for policy_name, rows in policies.items():
        if not rows:
            results[policy_name] = None
            continue
        n = len(rows)
        recovered_amount = sum(r["recovered_amount"] for r in rows)
        total_amount = sum(r["amount"] for r in rows)
        total_cost = sum(r["cost"] for r in rows)
        n_recovered = sum(1 for r in rows if r["recovered"])
        results[policy_name] = {
            "n_entities": n,
            "recovery_rate": n_recovered / n,
            "recovered_amount_rs": recovered_amount,
            "at_risk_amount_rs": total_amount,
            "cost_rs": total_cost,
            "net_value_rs": recovered_amount - total_cost,
            "cost_per_recovery_rs": (total_cost / n_recovered) if n_recovered else None,
            "customer_contact_rate": 1.0,  # every policy here always contacts (no NO_ACTION candidate in CANDIDATE_SPACE)
        }

    ml = results.get("unified_ml")
    for baseline_name in ("first_candidate", "rule_baseline"):
        baseline = results.get(baseline_name)
        if ml and baseline:
            ml[f"incremental_recovered_amount_rs_vs_{baseline_name}"] = ml["recovered_amount_rs"] - baseline["recovered_amount_rs"]
            ml[f"recovery_lift_vs_{baseline_name}"] = (
                (ml["recovery_rate"] - baseline["recovery_rate"]) / baseline["recovery_rate"] if baseline["recovery_rate"] > 0 else None
            )

    ranking_quality = {
        "note": "Only meaningful for domains with >1 candidate -- payment_failed has exactly 1, so it's trivially excluded here.",
        "n_entities_evaluated": n_multi_candidate_entities,
        "top1_accuracy_vs_oracle": (top1_hits / n_multi_candidate_entities) if n_multi_candidate_entities else None,
        "mean_ndcg": float(np.mean(ndcg_values)) if ndcg_values else None,
        "mean_regret_probability": float(np.mean(regret_prob_values)) if regret_prob_values else None,
        "mean_regret_value_rs": float(np.mean(regret_value_rs)) if regret_value_rs else None,
        "total_regret_value_rs": float(np.sum(regret_value_rs)) if regret_value_rs else None,
    }

    return {"outcomes": results, "ranking_quality": ranking_quality}


def run_evaluation() -> dict:
    model = load_unified_model()
    full_dataset = _make_training_data()
    _train_df, _val_df, test_df = _entity_level_split(full_dataset)

    rng = np.random.default_rng(99)
    per_domain = {}
    for event_type in SUPPORTED_EVENT_TYPES:
        domain_test_df = test_df[test_df["event_type"] == event_type]
        per_domain[event_type] = _evaluate_domain(domain_test_df, event_type, model, rng)

    overall = {"unified_ml": {"n_entities": 0, "recovered_amount_rs": 0.0, "at_risk_amount_rs": 0.0, "cost_rs": 0.0},
               "first_candidate": {"n_entities": 0, "recovered_amount_rs": 0.0, "at_risk_amount_rs": 0.0, "cost_rs": 0.0}}
    all_regret_rs = []
    for domain_results in per_domain.values():
        outcomes = domain_results["outcomes"]
        for policy_name in ("unified_ml", "first_candidate"):
            d = outcomes.get(policy_name)
            if not d:
                continue
            overall[policy_name]["n_entities"] += d["n_entities"]
            overall[policy_name]["recovered_amount_rs"] += d["recovered_amount_rs"]
            overall[policy_name]["at_risk_amount_rs"] += d["at_risk_amount_rs"]
            overall[policy_name]["cost_rs"] += d["cost_rs"]
        if domain_results["ranking_quality"]["total_regret_value_rs"] is not None:
            all_regret_rs.append(domain_results["ranking_quality"]["total_regret_value_rs"])
    for policy_name, agg in overall.items():
        agg["recovery_rate"] = agg["recovered_amount_rs"] / agg["at_risk_amount_rs"] if agg["at_risk_amount_rs"] else None
        agg["net_value_rs"] = agg["recovered_amount_rs"] - agg["cost_rs"]
    overall["total_regret_value_rs_vs_oracle"] = float(sum(all_regret_rs))

    report = {
        "note": (
            "SYNTHETIC/SIMULATED evaluation on the unified model's own held-out TEST "
            "split -- not real Razorpay recovery data, not authoritative payment "
            "confirmation. See module docstring."
        ),
        "test_rows": int(len(test_df)),
        "test_entities": int(test_df["entity_key"].nunique()),
        "per_domain": per_domain,
        "overall": overall,
    }
    return report


def main() -> None:
    report = run_evaluation()
    print(json.dumps(report, indent=2))
    with open(REPORT_PATH_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWritten to {REPORT_PATH_JSON}")


if __name__ == "__main__":
    main()
