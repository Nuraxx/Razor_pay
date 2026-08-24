"""
Day-6 synthetic counterfactual layer.

Day 3/5's dataset records exactly ONE observed outcome per failure event --
whatever actually happened -- not what would have happened under each of
the 5 candidate retry times. That meant Day 5 could score candidates but
could never honestly claim one candidate *causes* higher recovery than
another (see policy/scoring.py's module docstring and README "Day 5").

This module fixes that by generating a SEPARATE, ADDITIONAL outcome for
every (failure event, candidate) pair -- 5 simulated counterfactual outcomes
per event, one per candidate_type. It does not touch or regenerate
data/raw/{subscriptions,failure_events,retry_candidates,recovery_outcomes}.csv
(Day 3's original single-outcome mechanism, still used by Day 4/5) -- it
reuses that exact same generation (same seed -> byte-identical subscriptions/
failure_events/retry_candidates) and layers a new, independent random stream
on top to produce data/raw/counterfactual_outcomes.csv.

THIS DATA IS SYNTHETIC, same disclaimer as Day 3 (see data/README.md). It is
a hand-designed probabilistic model, not derived from real Razorpay
transactions.

Run:
    ./venv/bin/python data/generate_counterfactual_dataset.py

Reproducibility: one additional numpy Generator, seeded at
`seed + COUNTERFACTUAL_SEED_OFFSET` (default 42 + 5000), independent of the
generator `generate_synthetic_dataset.generate_dataset()` uses internally --
regenerating Day 3's tables is unaffected, and regenerating this layer with
the same seed reproduces byte-identical output.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data.generate_synthetic_dataset import (
    ARCHETYPE_BASE_LOGIT,
    BANK_CONDITION_PENALTY,
    DEFAULT_N_SUBSCRIPTIONS,
    DEFAULT_SEED,
    LATENCY_PENALTY,
    MAX_AMOUNT,
    MIN_AMOUNT,
    NOISE_STD,
    RECOVERED_VIA_PROBS,
    RECOVERY_SPEED_SCALE_DAYS,
    RETRY_CANDIDATE_TYPES,
    days_to_nearest_payday_window,
    generate_dataset,
)

COUNTERFACTUAL_SEED_OFFSET = 5000
RECOVERY_HORIZON_DAYS = 14  # same "recovered_within_14d" definition as Day 3/4/5

# ---------------------------------------------------------------------------
# Candidate-timing causal mechanism, hidden-archetype-dependent (generation
# only -- see module docstring in data/generate_synthetic_dataset.py for why
# this is legitimate at generation time but must never reach the model).
#
# Sized to satisfy the Day-6 brief's qualitative requirements directly:
#   - cash_strapped_cyclical: strong payday-alignment sensitivity
#   - reliable: timing matters comparatively little (funds are usually there)
#   - chronic_struggler: limited timing effect (mostly noise-driven)
#   - quiet_canceller: timing barely moves an already-low probability
# ---------------------------------------------------------------------------

CF_PAYDAY_SENSITIVITY = {
    "reliable": 0.2,
    "cash_strapped_cyclical": 1.8,
    "chronic_struggler": 0.5,
    "quiet_canceller": 0.1,
}
CF_MONTH_END_SENSITIVITY = {
    "reliable": 0.05,
    "cash_strapped_cyclical": 0.5,
    "chronic_struggler": 0.2,
    "quiet_canceller": 0.0,
}
# Retrying within the same hour rarely gives insufficient-funds time to
# resolve -- a penalty, sized larger for archetypes whose recovery
# fundamentally depends on funds becoming available (cash_strapped_cyclical)
# and near-zero for archetypes whose recovery doesn't depend on timing at all.
CF_IMMEDIATE_PENALTY = {
    "reliable": -0.1,
    "cash_strapped_cyclical": -0.9,
    "chronic_struggler": -0.35,
    "quiet_canceller": -0.05,
}
# Mild bonus for more elapsed time even off a payday window (some chance
# funds show up incidentally, or the customer self-serves eventually).
CF_ELAPSED_TIME_BONUS = {
    "reliable": 0.1,
    "cash_strapped_cyclical": 0.3,
    "chronic_struggler": 0.2,
    "quiet_canceller": 0.05,
}


def _candidate_timing_logit_term(archetype: str, candidate_type: str, candidate_days_to_payday: int, candidate_is_month_end_aligned: bool, hours_from_failure: float) -> float:
    """The ONLY thing that differs between the 5 counterfactual outcomes for
    the same failure event -- every other logit term below is shared across
    an event's 5 candidates because it describes the same underlying
    failure/subscription context."""
    proximity_to_payday = max(0.0, 1.0 - min(candidate_days_to_payday, 14) / 14)
    term = CF_PAYDAY_SENSITIVITY[archetype] * proximity_to_payday
    term += CF_MONTH_END_SENSITIVITY[archetype] * (1.0 if candidate_is_month_end_aligned else 0.0)
    if candidate_type == "immediate":
        term += CF_IMMEDIATE_PENALTY[archetype]
    elapsed_fraction = min(max(hours_from_failure, 0.0) / 24.0, RECOVERY_HORIZON_DAYS) / RECOVERY_HORIZON_DAYS
    term += CF_ELAPSED_TIME_BONUS[archetype] * elapsed_fraction
    return term


def _counterfactual_recovery_logit(
    rng: np.random.Generator,
    archetype: str,
    candidate_type: str,
    candidate_days_to_payday: int,
    candidate_is_month_end_aligned: bool,
    hours_from_failure: float,
    prior_self_resolved_rate: float,
    amount: float,
    tenure_at_failure: int,
    downtime_flag: bool,
    bank_condition: str,
    month_end_rush: bool,
    latency_bucket: str,
) -> float:
    """Same shared-context terms as generate_synthetic_dataset._recovery_logit
    (archetype base rate, prior history, amount, exogenous conditions,
    tenure), PLUS the candidate-timing term above that Day 3's single-outcome
    mechanism never had a reason to include."""
    logit = ARCHETYPE_BASE_LOGIT[archetype]
    logit += _candidate_timing_logit_term(archetype, candidate_type, candidate_days_to_payday, candidate_is_month_end_aligned, hours_from_failure)
    if not np.isnan(prior_self_resolved_rate):
        logit += 1.0 * prior_self_resolved_rate
    logit += -0.8 * (amount - MIN_AMOUNT) / (MAX_AMOUNT - MIN_AMOUNT)
    logit += -0.5 if downtime_flag else 0.0
    logit += BANK_CONDITION_PENALTY[bank_condition]
    logit += -0.3 if month_end_rush else 0.0
    logit += LATENCY_PENALTY[latency_bucket]
    logit += 0.3 * min(tenure_at_failure, 365) / 365
    logit += rng.normal(0, NOISE_STD)
    return logit


def generate_counterfactual_outcomes(
    rng: np.random.Generator,
    subscriptions: pd.DataFrame,
    failure_events: pd.DataFrame,
    retry_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (failure event, candidate_type) -- 5 per event, in
    RETRY_CANDIDATE_TYPES order, matching retry_candidates.csv's own row
    order so counterfactual_id and retry_candidate_id line up 1:1."""
    archetype_by_sub = subscriptions.set_index("subscription_id")["archetype"].to_dict()
    failures_by_event = failure_events.set_index("event_id")

    rows: list[dict] = []
    counter = 0
    for cand in retry_candidates.itertuples(index=False):
        event_id = cand.event_id
        fe = failures_by_event.loc[event_id]
        archetype = archetype_by_sub[cand.subscription_id]

        failure_ts = fe["failure_timestamp"]
        candidate_dt = cand.candidate_datetime
        candidate_days_to_payday = days_to_nearest_payday_window(candidate_dt)
        remaining_days = (failure_ts + timedelta(days=RECOVERY_HORIZON_DAYS) - candidate_dt).total_seconds() / 86400

        logit = _counterfactual_recovery_logit(
            rng,
            archetype=archetype,
            candidate_type=cand.candidate_type,
            candidate_days_to_payday=candidate_days_to_payday,
            candidate_is_month_end_aligned=bool(cand.is_month_end_aligned),
            hours_from_failure=cand.offset_hours_from_failure,
            prior_self_resolved_rate=fe["prior_if_self_resolved_rate"],
            amount=fe["amount"],
            tenure_at_failure=fe["tenure_days"],
            downtime_flag=bool(fe["issuing_bank_downtime_flag"]),
            bank_condition=fe["bank_network_conditions"],
            month_end_rush=bool(fe["is_month_end_settlement_rush"]),
            latency_bucket=fe["network_latency_bucket"],
        )
        p_recover = float(np.clip(1.0 / (1.0 + np.exp(-logit)), 0.02, 0.98))

        # Physical constraint, not a modeling choice: a candidate whose own
        # retry time already lands at/after the 14-day recovery horizon
        # cannot possibly produce a "recovered_within_14d" outcome, no
        # matter how high its latent probability -- there's no time left for
        # even an instantaneous recovery. See tests/test_counterfactual_dataset.py.
        if remaining_days <= 0.05:
            recovered = False
        else:
            recovered = bool(rng.random() < p_recover)

        if recovered:
            scale = RECOVERY_SPEED_SCALE_DAYS[archetype]
            days_to_recover = float(np.clip(rng.exponential(scale), 0.05, remaining_days))
            recovered_at = (candidate_dt + timedelta(days=days_to_recover)).replace(second=0, microsecond=0)
            via_probs = RECOVERED_VIA_PROBS[archetype]
            recovered_via = str(rng.choice(list(via_probs.keys()), p=list(via_probs.values())))
            fraction = 1.0 if rng.random() < 0.9 else float(rng.uniform(0.85, 0.99))
            amount_recovered = round(fe["amount"] * fraction, 2)
        else:
            recovered_at = pd.NaT
            recovered_via = "none"
            amount_recovered = 0.0

        rows.append(
            {
                "counterfactual_id": f"cfo_SYN{counter:06d}",
                "event_id": event_id,
                "subscription_id": cand.subscription_id,
                "candidate_type": cand.candidate_type,
                "candidate_datetime": candidate_dt,
                "recovery_probability_latent": round(p_recover, 6),
                "recovered_within_14d": recovered,
                "recovered_at": recovered_at,
                "recovered_via": recovered_via,
                "amount_recovered": amount_recovered,
            }
        )
        counter += 1

    return pd.DataFrame(rows)


def write_counterfactual_outcomes(df: pd.DataFrame, output_dir: Path) -> None:
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_dir / "counterfactual_outcomes.csv", index=False)


# ---------------------------------------------------------------------------
# Sanity checks (Day-6 brief section 10) -- archetype is used here for
# reporting only, exactly as Day 3's own summarize_dataset() does; it is
# never written into a model-facing column.
# ---------------------------------------------------------------------------

def summarize_counterfactual_outcomes(cf: pd.DataFrame, subscriptions: pd.DataFrame) -> dict:
    archetype_by_sub = subscriptions.set_index("subscription_id")["archetype"].to_dict()
    cf = cf.copy()
    cf["archetype"] = cf["subscription_id"].map(archetype_by_sub)

    recovery_rate_by_candidate = cf.groupby("candidate_type")["recovered_within_14d"].mean().round(4).to_dict()
    avg_latent_by_candidate = cf.groupby("candidate_type")["recovery_probability_latent"].mean().round(4).to_dict()
    recovery_rate_by_archetype_candidate = (
        cf.groupby(["archetype", "candidate_type"])["recovered_within_14d"].mean().round(4).unstack().to_dict()
    )

    oracle_idx = cf.groupby("event_id")["recovery_probability_latent"].idxmax()
    oracle_candidates = cf.loc[oracle_idx, "candidate_type"]
    oracle_distribution = oracle_candidates.value_counts().to_dict()
    max_oracle_share = max(oracle_distribution.values()) / len(oracle_candidates) if len(oracle_candidates) else 0.0

    return {
        "n_events": cf["event_id"].nunique(),
        "n_counterfactual_rows": len(cf),
        "recovery_rate_by_candidate_type": recovery_rate_by_candidate,
        "avg_latent_recovery_probability_by_candidate_type": avg_latent_by_candidate,
        "recovery_rate_by_archetype_x_candidate_type": recovery_rate_by_archetype_candidate,
        "oracle_candidate_distribution": oracle_distribution,
        "max_single_candidate_oracle_share": round(max_oracle_share, 4),
        "no_candidate_dominates": bool(max_oracle_share < 0.90),
    }


def print_summary(summary: dict) -> None:
    print("=== Counterfactual dataset summary ===")
    print(f"n_events: {summary['n_events']} | n_counterfactual_rows: {summary['n_counterfactual_rows']}")
    print(f"recovery_rate_by_candidate_type: {summary['recovery_rate_by_candidate_type']}")
    print(f"avg_latent_recovery_probability_by_candidate_type: {summary['avg_latent_recovery_probability_by_candidate_type']}")
    print("recovery_rate_by_archetype_x_candidate_type (generation-only, not a model feature):")
    for archetype, per_candidate in summary["recovery_rate_by_archetype_x_candidate_type"].items():
        print(f"  {archetype}: {per_candidate}")
    print(f"oracle_candidate_distribution: {summary['oracle_candidate_distribution']}")
    print(f"max_single_candidate_oracle_share: {summary['max_single_candidate_oracle_share']} (no single candidate should dominate: {summary['no_candidate_dominates']})")


def validate_counterfactual_outcomes(cf: pd.DataFrame, failure_events: pd.DataFrame) -> list[str]:
    issues: list[str] = []

    if cf["counterfactual_id"].duplicated().any():
        issues.append("duplicate counterfactual_id")

    counts_per_event = cf.groupby("event_id").size()
    if not (counts_per_event == 5).all():
        issues.append("not every event_id has exactly 5 counterfactual outcomes")

    if not set(cf["event_id"]).issubset(set(failure_events["event_id"])):
        issues.append("counterfactual_outcomes references an event_id not present in failure_events")

    for candidate_type_set in [set(cf["candidate_type"].unique())]:
        if candidate_type_set != set(RETRY_CANDIDATE_TYPES):
            issues.append(f"unexpected candidate_type set: {candidate_type_set}")

    joined = cf.merge(failure_events[["event_id", "failure_timestamp"]], on="event_id", how="left")
    if (joined["candidate_datetime"] <= joined["failure_timestamp"]).any():
        issues.append("a counterfactual candidate_datetime is not after its failure_timestamp")

    if not cf["recovery_probability_latent"].between(0.0, 1.0).all():
        issues.append("recovery_probability_latent outside [0, 1]")

    if (cf["amount_recovered"] < 0).any():
        issues.append("amount_recovered is negative for at least one row")

    amount_joined = cf.merge(failure_events[["event_id", "amount"]], on="event_id", how="left")
    if (amount_joined["amount_recovered"] > amount_joined["amount"]).any():
        issues.append("amount_recovered exceeds the original amount for at least one row")

    mismatched_null = cf["recovered_within_14d"] != cf["recovered_at"].notna()
    if mismatched_null.any():
        issues.append("recovered_at nullness does not match recovered_within_14d")

    recovered_rows = joined[cf["recovered_within_14d"]]
    if (recovered_rows["recovered_at"] <= recovered_rows["candidate_datetime"]).any():
        issues.append("recovered_at is not after candidate_datetime for a recovered counterfactual row")
    too_late = recovered_rows["recovered_at"] > recovered_rows["failure_timestamp"] + pd.Timedelta(days=RECOVERY_HORIZON_DAYS)
    if too_late.any():
        issues.append("recovered_at falls beyond failure_timestamp + 14 days for a recovered counterfactual row")

    # A candidate whose own datetime is already beyond the recovery horizon
    # can never be recorded as recovered -- see the physical constraint in
    # generate_counterfactual_outcomes().
    beyond_horizon = joined["candidate_datetime"] > joined["failure_timestamp"] + pd.Timedelta(days=RECOVERY_HORIZON_DAYS)
    if (beyond_horizon & cf["recovered_within_14d"]).any():
        issues.append("a candidate beyond the 14-day horizon is recorded as recovered_within_14d")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Day-6 synthetic counterfactual outcomes layer.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-subscriptions", type=int, default=DEFAULT_N_SUBSCRIPTIONS)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    dfs = generate_dataset(seed=args.seed, n_subscriptions=args.n_subscriptions)
    cf_rng = np.random.default_rng(args.seed + COUNTERFACTUAL_SEED_OFFSET)
    cf = generate_counterfactual_outcomes(cf_rng, dfs["subscriptions"], dfs["failure_events"], dfs["retry_candidates"])
    write_counterfactual_outcomes(cf, args.output_dir)

    print(f"Wrote {args.output_dir / 'raw' / 'counterfactual_outcomes.csv'}")
    print()
    print_summary(summarize_counterfactual_outcomes(cf, dfs["subscriptions"]))
    print()

    issues = validate_counterfactual_outcomes(cf, dfs["failure_events"])
    if issues:
        print(f"=== VALIDATION: FAILED ({len(issues)} issue(s)) ===")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)
    print("=== VALIDATION: PASSED (all checks green) ===")


if __name__ == "__main__":
    main()
