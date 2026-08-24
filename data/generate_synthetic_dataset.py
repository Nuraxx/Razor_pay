"""
Day-3 synthetic dataset generator.

Razorpay Subscriptions -> insufficient_fund failures -> retry candidates ->
probabilistic recovery outcome. Produces the raw event tables plus a
leakage-safe train/validation/test split, ready for Day 4/5 model work.

THIS DATA IS SYNTHETIC. It is generated from a hand-designed probabilistic
model (see _recovery_logit below), not from real Razorpay transactions. It
must never be presented as evidence of real customer behavior -- see
data/README.md.

Run:
    ./venv/bin/python data/generate_synthetic_dataset.py

Reproducibility: every random draw goes through a single numpy Generator
seeded once at the top of generate_dataset(). Same seed + same
n_subscriptions => byte-identical output, every time, regardless of when or
where it's run (all dates are computed relative to the fixed EXTRACTION_DATE
constant below, never real wall-clock time).
"""
from __future__ import annotations

import argparse
import calendar
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42
DEFAULT_N_SUBSCRIPTIONS = 200

# Fixed reference date the whole dataset is generated relative to. Using a
# constant (never datetime.now()) is what makes generation reproducible
# regardless of what day it's actually run.
EXTRACTION_DATE = datetime(2026, 6, 30)
SIGNUP_LOOKBACK_DAYS = (30, 730)  # subscriptions signed up 1 month - 2 years before EXTRACTION_DATE

ARCHETYPES = ["reliable", "cash_strapped_cyclical", "chronic_struggler", "quiet_canceller"]
ARCHETYPE_PROPORTIONS = [0.50, 0.25, 0.15, 0.10]

PLAN_TIERS = {
    "mobile": (79, 199),
    "mid": (299, 499),
    "premium": (649, 999),
    "saas": (1000, 5000),
}
PLAN_TIER_WEIGHTS = [0.35, 0.35, 0.20, 0.10]  # mobile, mid, premium, saas
MIN_AMOUNT = min(lo for lo, _ in PLAN_TIERS.values())
MAX_AMOUNT = max(hi for _, hi in PLAN_TIERS.values())

CITY_TIERS = ["tier_1", "tier_2", "tier_3"]
CITY_TIER_WEIGHTS = [0.45, 0.35, 0.20]

INSTRUMENTS = ["credit_card", "debit_card", "upi_autopay", "netbanking"]
INSTRUMENT_WEIGHTS = [0.35, 0.30, 0.25, 0.10]

# Distractor features -- sanity-check-only, deliberately uncorrelated with
# everything else in the generator (requirement: must NOT meaningfully
# predict recovery).
APP_VERSIONS = ["4.2.0", "4.3.1", "4.4.0", "5.0.0", "5.1.2"]
DEVICE_BUILDS = ["build_A1", "build_B7", "build_C3", "build_D9", "build_E2"]
UI_THEMES = ["light", "dark", "system"]

BANK_CONDITIONS = ["good", "degraded", "poor"]
BANK_CONDITION_WEIGHTS = [0.70, 0.20, 0.10]
BANK_CONDITION_PENALTY = {"good": 0.0, "degraded": -0.2, "poor": -0.5}

LATENCY_BUCKETS = ["low", "medium", "high"]
LATENCY_WEIGHTS = [0.60, 0.30, 0.10]
LATENCY_PENALTY = {"low": 0.0, "medium": -0.1, "high": -0.3}

# Hidden, generation-only mechanism. Never exposed as a model feature.
ARCHETYPE_BASE_LOGIT = {
    "reliable": 2.2,
    "cash_strapped_cyclical": 0.6,
    "chronic_struggler": -0.6,
    "quiet_canceller": -2.0,
}
PAYDAY_SENSITIVITY = {
    "reliable": 0.3,
    "cash_strapped_cyclical": 1.4,
    "chronic_struggler": 0.7,
    "quiet_canceller": 0.1,
}
RECOVERY_SPEED_SCALE_DAYS = {  # exponential-distribution scale; smaller = recovers faster
    "reliable": 1.5,
    "cash_strapped_cyclical": 4.0,
    "chronic_struggler": 6.0,
    "quiet_canceller": 9.0,
}
RECOVERED_VIA_PROBS = {
    "reliable": {"auto_retry": 0.7, "customer_self_serve": 0.3},
    "cash_strapped_cyclical": {"auto_retry": 0.8, "customer_self_serve": 0.2},
    "chronic_struggler": {"auto_retry": 0.5, "customer_self_serve": 0.5},
    "quiet_canceller": {"auto_retry": 0.1, "customer_self_serve": 0.9},
}
NOISE_STD = 0.9  # dominant lever controlling how learnable the label is

RETRY_CANDIDATE_TYPES = ["immediate", "plus_1_day_morning", "payday_window", "plus_3_days", "month_end_window"]

RAW_TABLES = ("subscriptions", "failure_events", "retry_candidates", "recovery_outcomes")


# ---------------------------------------------------------------------------
# Calendar helpers (payday-window / month-end logic, used by both the
# probabilistic label mechanism and retry-candidate generation)
# ---------------------------------------------------------------------------

def _payday_window_dates(year: int, month: int) -> list[datetime]:
    """First few days + last few days of the month -- a common salary-cycle proxy."""
    last_day = calendar.monthrange(year, month)[1]
    days = sorted({1, 2, 3, max(1, last_day - 2), max(1, last_day - 1), last_day})
    return [datetime(year, month, d) for d in days]


def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    total = (year * 12 + (month - 1)) + delta
    return total // 12, total % 12 + 1


def days_to_nearest_payday_window(dt: datetime) -> int:
    candidates: list[datetime] = []
    for delta in (-1, 0, 1):
        y, m = _shift_month(dt.year, dt.month, delta)
        candidates.extend(_payday_window_dates(y, m))
    return min(abs((dt.date() - c.date()).days) for c in candidates)


def next_payday_window_after(dt: datetime) -> datetime:
    for delta in (0, 1, 2):
        y, m = _shift_month(dt.year, dt.month, delta)
        for wd in _payday_window_dates(y, m):
            if wd.date() > dt.date():
                return wd.replace(hour=10, minute=0)
    return dt + timedelta(days=30)  # unreachable in practice


def next_month_end_after(dt: datetime) -> datetime:
    y, m = dt.year, dt.month
    last_day = calendar.monthrange(y, m)[1]
    month_end = datetime(y, m, last_day)
    if month_end.date() > dt.date():
        return month_end.replace(hour=18, minute=0)
    y2, m2 = _shift_month(y, m, 1)
    last_day2 = calendar.monthrange(y2, m2)[1]
    return datetime(y2, m2, last_day2, 18, 0)


# ---------------------------------------------------------------------------
# Subscriptions
# ---------------------------------------------------------------------------

def generate_subscriptions(rng: np.random.Generator, n: int) -> pd.DataFrame:
    archetypes = rng.choice(ARCHETYPES, size=n, p=ARCHETYPE_PROPORTIONS)
    plan_tiers = rng.choice(list(PLAN_TIERS.keys()), size=n, p=PLAN_TIER_WEIGHTS)
    city_tiers = rng.choice(CITY_TIERS, size=n, p=CITY_TIER_WEIGHTS)
    instruments = rng.choice(INSTRUMENTS, size=n, p=INSTRUMENT_WEIGHTS)

    rows = []
    for i in range(n):
        plan_tier = str(plan_tiers[i])
        lo, hi = PLAN_TIERS[plan_tier]
        monthly_amount = int(rng.integers(lo, hi + 1))
        signup_offset = int(rng.integers(*SIGNUP_LOOKBACK_DAYS))
        signup_date = EXTRACTION_DATE - timedelta(days=signup_offset)
        tenure_days = (EXTRACTION_DATE - signup_date).days

        rows.append(
            {
                "subscription_id": f"sub_SYN{i:05d}",
                "plan_tier": plan_tier,
                "monthly_amount": monthly_amount,
                "signup_date": signup_date,
                "primary_instrument": str(instruments[i]),
                "city_tier": str(city_tiers[i]),
                "tenure_days": tenure_days,  # as of EXTRACTION_DATE -- NOT usable as an event-level feature, see README
                "archetype": str(archetypes[i]),  # hidden generation-only field -- never a model feature
            }
        )

    df = pd.DataFrame(rows)

    # Split BY subscription_id, before any event rows exist, so a subscription
    # and everything derived from it lands in exactly one split.
    n_rows = len(df)
    perm = rng.permutation(n_rows)
    n_train = int(round(n_rows * 0.6))
    n_val = int(round(n_rows * 0.2))
    split = np.empty(n_rows, dtype=object)
    split[perm[:n_train]] = "train"
    split[perm[n_train : n_train + n_val]] = "validation"
    split[perm[n_train + n_val :]] = "test"
    df["split"] = split

    return df


# ---------------------------------------------------------------------------
# Failure events + recovery outcomes (generated together, per subscription,
# in chronological order -- this is what makes prior_if_* causally correct)
# ---------------------------------------------------------------------------

def _sample_n_failures(rng: np.random.Generator, archetype: str) -> int:
    if archetype == "reliable":
        return int(rng.choice([1, 2], p=[0.9, 0.1]))
    if archetype == "cash_strapped_cyclical":
        return int(rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25]))
    if archetype == "chronic_struggler":
        return int(rng.choice([2, 3, 4], p=[0.3, 0.4, 0.3]))
    if archetype == "quiet_canceller":
        return 1
    raise ValueError(f"unknown archetype: {archetype}")


def _sample_gap_days(rng: np.random.Generator, archetype: str) -> float:
    if archetype == "cash_strapped_cyclical":
        return float(np.clip(rng.normal(30, 5), 15, 60))
    if archetype == "chronic_struggler":
        return float(rng.uniform(10, 45))
    return float(rng.uniform(60, 200))  # reliable's rare 2nd failure


def _recovery_logit(
    rng: np.random.Generator,
    archetype: str,
    days_to_payday: int,
    prior_self_resolved_rate: float,
    amount: float,
    tenure_at_failure: int,
    downtime_flag: bool,
    bank_condition: str,
    month_end_rush: bool,
    latency_bucket: str,
) -> float:
    logit = ARCHETYPE_BASE_LOGIT[archetype]
    logit += PAYDAY_SENSITIVITY[archetype] * (1 - min(days_to_payday, 14) / 14)
    if not np.isnan(prior_self_resolved_rate):
        logit += 1.0 * prior_self_resolved_rate
    logit += -0.8 * (amount - MIN_AMOUNT) / (MAX_AMOUNT - MIN_AMOUNT)
    logit += -0.5 if downtime_flag else 0.0
    logit += BANK_CONDITION_PENALTY[bank_condition]
    logit += -0.3 if month_end_rush else 0.0
    logit += LATENCY_PENALTY[latency_bucket]
    logit += 0.3 * min(tenure_at_failure, 365) / 365
    logit += rng.normal(0, NOISE_STD)  # moderate noise -- keeps this learnable, not solvable
    return logit


def generate_failure_events_and_outcomes(
    rng: np.random.Generator, subscriptions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    failure_rows: list[dict] = []
    outcome_rows: list[dict] = []
    event_counter = 0

    for sub in subscriptions.itertuples(index=False):
        archetype = sub.archetype
        signup_date: datetime = sub.signup_date
        tenure_days_at_extraction: int = sub.tenure_days
        amount = float(sub.monthly_amount)

        n_failures = _sample_n_failures(rng, archetype)
        prior_self_resolved: list[bool] = []

        first_offset_hi = max(15, tenure_days_at_extraction - 10)
        t = signup_date + timedelta(days=int(rng.integers(10, first_offset_hi + 1)))

        for k in range(n_failures):
            if k > 0:
                t = t + timedelta(days=_sample_gap_days(rng, archetype))
            t = t.replace(hour=int(rng.integers(0, 24)), minute=int(rng.integers(0, 60)), second=0, microsecond=0)

            # Natural censoring: don't generate failures unrealistically far
            # past the dataset's reference point.
            if t > EXTRACTION_DATE + timedelta(days=60):
                break

            failure_ts = t
            prior_count = len(prior_self_resolved)
            prior_self_resolved_rate = (
                float(np.mean(prior_self_resolved)) if prior_count > 0 else float("nan")
            )
            tenure_at_failure = (failure_ts - signup_date).days
            day_of_month = failure_ts.day
            payday_dist = days_to_nearest_payday_window(failure_ts)

            # Exogenous conditions -- drawn independently of archetype.
            bank_condition = str(rng.choice(BANK_CONDITIONS, p=BANK_CONDITION_WEIGHTS))
            downtime_flag = bool(rng.random() < 0.05)
            latency_bucket = str(rng.choice(LATENCY_BUCKETS, p=LATENCY_WEIGHTS))
            rush_p = 0.35 if day_of_month in (29, 30, 31, 1, 2) else 0.05
            month_end_rush = bool(rng.random() < rush_p)

            event_id = f"evt_SYN{event_counter:06d}"
            event_counter += 1

            failure_rows.append(
                {
                    "event_id": event_id,
                    "subscription_id": sub.subscription_id,
                    "failure_timestamp": failure_ts,
                    "day_of_month": day_of_month,
                    "days_to_nearest_payday_window": payday_dist,
                    "error_reason": "insufficient_fund",
                    "amount": amount,
                    "prior_if_failure_count": prior_count,
                    "prior_if_self_resolved_rate": prior_self_resolved_rate,
                    "tenure_days": tenure_at_failure,
                    "bank_network_conditions": bank_condition,
                    "issuing_bank_downtime_flag": downtime_flag,
                    "network_latency_bucket": latency_bucket,
                    "is_month_end_settlement_rush": month_end_rush,
                    "app_version": str(rng.choice(APP_VERSIONS)),
                    "device_build": str(rng.choice(DEVICE_BUILDS)),
                    "ui_theme": str(rng.choice(UI_THEMES)),
                }
            )

            logit = _recovery_logit(
                rng,
                archetype,
                payday_dist,
                prior_self_resolved_rate,
                amount,
                tenure_at_failure,
                downtime_flag,
                bank_condition,
                month_end_rush,
                latency_bucket,
            )
            p_recover = float(np.clip(1.0 / (1.0 + np.exp(-logit)), 0.02, 0.98))
            recovered = bool(rng.random() < p_recover)

            if recovered:
                scale = RECOVERY_SPEED_SCALE_DAYS[archetype]
                days_to_recover = float(np.clip(rng.exponential(scale), 0.05, 14.0))
                recovered_at = (failure_ts + timedelta(days=days_to_recover)).replace(second=0, microsecond=0)
                via_probs = RECOVERED_VIA_PROBS[archetype]
                recovered_via = str(rng.choice(list(via_probs.keys()), p=list(via_probs.values())))
                self_resolved = recovered_via == "customer_self_serve"
                fraction = 1.0 if rng.random() < 0.9 else float(rng.uniform(0.85, 0.99))
                final_amount_recovered = round(amount * fraction, 2)
            else:
                recovered_at = pd.NaT
                recovered_via = "none"
                self_resolved = False
                final_amount_recovered = 0.0

            outcome_rows.append(
                {
                    "event_id": event_id,
                    "subscription_id": sub.subscription_id,
                    "recovered_within_14d": recovered,
                    "recovered_at": recovered_at,
                    "recovered_via": recovered_via,
                    "final_amount_recovered": final_amount_recovered,
                }
            )

            prior_self_resolved.append(self_resolved)

    return pd.DataFrame(failure_rows), pd.DataFrame(outcome_rows)


# ---------------------------------------------------------------------------
# Retry candidates -- descriptive only, no outcome/label (retry-time
# selection is explicitly out of scope for Day 3).
# ---------------------------------------------------------------------------

def generate_retry_candidates(rng: np.random.Generator, failure_events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    counter = 0

    for ev in failure_events.itertuples(index=False):
        failure_ts: datetime = ev.failure_timestamp

        candidate_times = {
            "immediate": failure_ts + timedelta(minutes=int(rng.integers(5, 120))),
            "plus_1_day_morning": (failure_ts + timedelta(days=1)).replace(
                hour=int(rng.integers(8, 12)), minute=int(rng.integers(0, 60))
            ),
            "payday_window": next_payday_window_after(failure_ts),
            "plus_3_days": (failure_ts + timedelta(days=3)).replace(
                hour=int(rng.integers(9, 20)), minute=int(rng.integers(0, 60))
            ),
            "month_end_window": next_month_end_after(failure_ts),
        }

        for candidate_type in RETRY_CANDIDATE_TYPES:
            candidate_dt = candidate_times[candidate_type]
            rows.append(
                {
                    "retry_candidate_id": f"rtc_SYN{counter:06d}",
                    "event_id": ev.event_id,
                    "subscription_id": ev.subscription_id,
                    "candidate_type": candidate_type,
                    "candidate_datetime": candidate_dt,
                    "offset_hours_from_failure": round((candidate_dt - failure_ts).total_seconds() / 3600, 2),
                    "day_of_week": candidate_dt.strftime("%A"),
                    "is_payday_aligned": days_to_nearest_payday_window(candidate_dt) <= 1,
                    "is_month_end_aligned": (next_month_end_after(candidate_dt - timedelta(days=1)).date() - candidate_dt.date()).days
                    <= 1,
                }
            )
            counter += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Processed model-ready splits (archetype + split columns dropped here)
# ---------------------------------------------------------------------------

PROCESSED_FEATURE_COLUMNS = [
    "event_id",
    "subscription_id",
    "failure_timestamp",
    "day_of_month",
    "days_to_nearest_payday_window",
    "error_reason",
    "amount",
    "prior_if_failure_count",
    "prior_if_self_resolved_rate",
    "tenure_days",
    "plan_tier",
    "monthly_amount",
    "primary_instrument",
    "city_tier",
    "signup_date",
    "bank_network_conditions",
    "issuing_bank_downtime_flag",
    "network_latency_bucket",
    "is_month_end_settlement_rush",
    "app_version",
    "device_build",
    "ui_theme",
    "recovered_within_14d",
    "recovered_at",
    "recovered_via",
    "final_amount_recovered",
]


def build_processed_splits(
    subscriptions: pd.DataFrame, failure_events: pd.DataFrame, recovery_outcomes: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    # NOTE: subscriptions.tenure_days is "as of EXTRACTION_DATE" and is
    # deliberately NOT joined in here -- it would leak the future (a
    # subscription that's still around at EXTRACTION_DATE implies it didn't
    # churn). Only failure_events.tenure_days ("as of this failure") is used.
    sub_cols = subscriptions[
        ["subscription_id", "plan_tier", "monthly_amount", "primary_instrument", "city_tier", "signup_date", "split"]
    ]

    merged = failure_events.merge(recovery_outcomes, on=["event_id", "subscription_id"], how="left").merge(
        sub_cols, on="subscription_id", how="left"
    )

    splits: dict[str, pd.DataFrame] = {}
    for split_name, out_name in (("train", "train"), ("validation", "validation"), ("test", "test")):
        subset = merged[merged["split"] == split_name][PROCESSED_FEATURE_COLUMNS].reset_index(drop=True)
        splits[out_name] = subset
    return splits


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_dataset(seed: int = DEFAULT_SEED, n_subscriptions: int = DEFAULT_N_SUBSCRIPTIONS) -> dict[str, pd.DataFrame]:
    """Pure in-memory generation -- no file I/O. Deterministic for a given (seed, n_subscriptions)."""
    rng = np.random.default_rng(seed)

    subscriptions = generate_subscriptions(rng, n_subscriptions)
    failure_events, recovery_outcomes = generate_failure_events_and_outcomes(rng, subscriptions)
    retry_candidates = generate_retry_candidates(rng, failure_events)
    processed = build_processed_splits(subscriptions, failure_events, recovery_outcomes)

    return {
        "subscriptions": subscriptions,
        "failure_events": failure_events,
        "retry_candidates": retry_candidates,
        "recovery_outcomes": recovery_outcomes,
        "train": processed["train"],
        "validation": processed["validation"],
        "test": processed["test"],
    }


def write_dataset(dfs: dict[str, pd.DataFrame], output_dir: Path) -> None:
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    dfs["subscriptions"].to_csv(raw_dir / "subscriptions.csv", index=False)
    dfs["failure_events"].to_csv(raw_dir / "failure_events.csv", index=False)
    dfs["retry_candidates"].to_csv(raw_dir / "retry_candidates.csv", index=False)
    dfs["recovery_outcomes"].to_csv(raw_dir / "recovery_outcomes.csv", index=False)

    dfs["train"].to_csv(processed_dir / "train.csv", index=False)
    dfs["validation"].to_csv(processed_dir / "validation.csv", index=False)
    dfs["test"].to_csv(processed_dir / "test.csv", index=False)


# ---------------------------------------------------------------------------
# Summary (requirement 16)
# ---------------------------------------------------------------------------

def summarize_dataset(dfs: dict[str, pd.DataFrame]) -> dict:
    subscriptions = dfs["subscriptions"]
    failure_events = dfs["failure_events"]
    retry_candidates = dfs["retry_candidates"]
    recovery_outcomes = dfs["recovery_outcomes"]

    n_recovered = int(recovery_outcomes["recovered_within_14d"].sum())
    n_events = len(failure_events)
    amount_failed = float(failure_events["amount"].sum())
    amount_recovered = float(recovery_outcomes["final_amount_recovered"].sum())

    missing_values = {}
    for name in RAW_TABLES:
        counts = dfs[name].isna().sum()
        missing_values[name] = {col: int(c) for col, c in counts.items() if c > 0}

    duplicate_ids = {
        "subscription_id_in_subscriptions": int(subscriptions["subscription_id"].duplicated().sum()),
        "event_id_in_failure_events": int(failure_events["event_id"].duplicated().sum()),
        "event_id_in_recovery_outcomes": int(recovery_outcomes["event_id"].duplicated().sum()),
        "retry_candidate_id": int(retry_candidates["retry_candidate_id"].duplicated().sum()),
    }

    return {
        "n_subscriptions": len(subscriptions),
        "n_failure_events": n_events,
        "n_recovered_events": n_recovered,
        "recovery_rate": round(n_recovered / n_events, 4) if n_events else None,
        "amount_failed": round(amount_failed, 2),
        "amount_recovered": round(amount_recovered, 2),
        "distribution_by_archetype_internal": subscriptions["archetype"].value_counts().to_dict(),
        "distribution_by_plan_tier": subscriptions["plan_tier"].value_counts().to_dict(),
        "distribution_by_city_tier": subscriptions["city_tier"].value_counts().to_dict(),
        "distribution_by_candidate_retry_time": retry_candidates["candidate_type"].value_counts().to_dict(),
        "class_balance": recovery_outcomes["recovered_within_14d"].value_counts(normalize=True).round(4).to_dict(),
        "missing_values_by_table": missing_values,
        "duplicate_ids": duplicate_ids,
        "split_counts": {
            "train": len(dfs["train"]["subscription_id"].unique()),
            "validation": len(dfs["validation"]["subscription_id"].unique()),
            "test": len(dfs["test"]["subscription_id"].unique()),
        },
    }


def print_summary(summary: dict) -> None:
    print("=== Dataset summary ===")
    for key in (
        "n_subscriptions",
        "n_failure_events",
        "n_recovered_events",
        "recovery_rate",
        "amount_failed",
        "amount_recovered",
    ):
        print(f"{key}: {summary[key]}")
    print(f"distribution_by_archetype_internal (generation-only, not a model feature): {summary['distribution_by_archetype_internal']}")
    print(f"distribution_by_plan_tier: {summary['distribution_by_plan_tier']}")
    print(f"distribution_by_city_tier: {summary['distribution_by_city_tier']}")
    print(f"distribution_by_candidate_retry_time: {summary['distribution_by_candidate_retry_time']}")
    print(f"class_balance (recovered_within_14d): {summary['class_balance']}")
    print(f"split subscription counts: {summary['split_counts']}")
    print(f"duplicate_ids: {summary['duplicate_ids']}")
    print(f"missing_values_by_table (non-zero columns only): {summary['missing_values_by_table']}")


# ---------------------------------------------------------------------------
# Validation (requirement 17)
# ---------------------------------------------------------------------------

def validate_dataset(dfs: dict[str, pd.DataFrame]) -> list[str]:
    issues: list[str] = []

    subscriptions = dfs["subscriptions"]
    failure_events = dfs["failure_events"]
    retry_candidates = dfs["retry_candidates"]
    recovery_outcomes = dfs["recovery_outcomes"]
    train, validation, test = dfs["train"], dfs["validation"], dfs["test"]

    # No subscription in more than one split / zero overlap between splits.
    train_ids = set(train["subscription_id"])
    val_ids = set(validation["subscription_id"])
    test_ids = set(test["subscription_id"])
    if train_ids & val_ids:
        issues.append("train and validation share subscription_ids")
    if train_ids & test_ids:
        issues.append("train and test share subscription_ids")
    if val_ids & test_ids:
        issues.append("validation and test share subscription_ids")

    # No duplicate IDs.
    if subscriptions["subscription_id"].duplicated().any():
        issues.append("duplicate subscription_id in subscriptions.csv")
    if failure_events["event_id"].duplicated().any():
        issues.append("duplicate event_id in failure_events.csv")
    if retry_candidates["retry_candidate_id"].duplicated().any():
        issues.append("duplicate retry_candidate_id in retry_candidates.csv")
    if recovery_outcomes["event_id"].duplicated().any():
        issues.append("duplicate event_id in recovery_outcomes.csv")

    # No missing required (non-nullable) fields.
    required = {
        "subscriptions": ["subscription_id", "plan_tier", "monthly_amount", "signup_date", "primary_instrument", "city_tier", "tenure_days"],
        "failure_events": ["event_id", "subscription_id", "failure_timestamp", "day_of_month", "days_to_nearest_payday_window", "error_reason", "amount", "prior_if_failure_count", "tenure_days"],
        "retry_candidates": ["retry_candidate_id", "event_id", "subscription_id", "candidate_type", "candidate_datetime"],
        "recovery_outcomes": ["event_id", "subscription_id", "recovered_within_14d", "recovered_via", "final_amount_recovered"],
    }
    for table_name, cols in required.items():
        df = dfs[table_name]
        for col in cols:
            if df[col].isna().any():
                issues.append(f"{table_name}.{col} has missing values in required field")

    # error_reason always insufficient_fund.
    if not (failure_events["error_reason"] == "insufficient_fund").all():
        issues.append("failure_events.error_reason contains a value other than insufficient_fund")

    # Recovery outcomes / retry candidates reference valid event_ids.
    valid_event_ids = set(failure_events["event_id"])
    if not set(recovery_outcomes["event_id"]).issubset(valid_event_ids):
        issues.append("recovery_outcomes references an event_id not present in failure_events")
    if not set(retry_candidates["event_id"]).issubset(valid_event_ids):
        issues.append("retry_candidates references an event_id not present in failure_events")

    # Distractor features present.
    for col in ("app_version", "device_build", "ui_theme"):
        if col not in failure_events.columns:
            issues.append(f"distractor feature {col} missing from failure_events.csv")

    # Hidden archetype excluded from processed/model feature set.
    for split_name, df in (("train", train), ("validation", validation), ("test", test)):
        if "archetype" in df.columns:
            issues.append(f"archetype leaked into processed/{split_name}.csv")

    # Recovered amount never exceeds the failed amount.
    joined = recovery_outcomes.merge(failure_events[["event_id", "amount"]], on="event_id", how="left")
    if (joined["final_amount_recovered"] > joined["amount"]).any():
        issues.append("final_amount_recovered exceeds amount for at least one event")

    # Recovery timestamps valid: present iff recovered, and within [failure_timestamp, +14d].
    outcome_joined = recovery_outcomes.merge(
        failure_events[["event_id", "failure_timestamp"]], on="event_id", how="left"
    )
    mismatched_null = outcome_joined["recovered_within_14d"] != outcome_joined["recovered_at"].notna()
    if mismatched_null.any():
        issues.append("recovered_at nullness does not match recovered_within_14d")
    recovered_rows = outcome_joined[outcome_joined["recovered_within_14d"]]
    too_early = recovered_rows["recovered_at"] <= recovered_rows["failure_timestamp"]
    too_late = recovered_rows["recovered_at"] > recovered_rows["failure_timestamp"] + pd.Timedelta(days=14)
    if too_early.any() or too_late.any():
        issues.append("recovered_at falls outside (failure_timestamp, failure_timestamp + 14 days]")

    # Candidate retry times valid: known type, strictly after failure_timestamp.
    candidates_joined = retry_candidates.merge(
        failure_events[["event_id", "failure_timestamp"]], on="event_id", how="left"
    )
    if not candidates_joined["candidate_type"].isin(RETRY_CANDIDATE_TYPES).all():
        issues.append("retry_candidates contains an unknown candidate_type")
    if (candidates_joined["candidate_datetime"] <= candidates_joined["failure_timestamp"]).any():
        issues.append("a retry candidate_datetime is not after its failure_timestamp")

    return issues


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Day-3 synthetic recovery dataset.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-subscriptions", type=int, default=DEFAULT_N_SUBSCRIPTIONS)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    dfs = generate_dataset(seed=args.seed, n_subscriptions=args.n_subscriptions)
    write_dataset(dfs, args.output_dir)

    print(f"Wrote raw tables to {args.output_dir / 'raw'} and processed splits to {args.output_dir / 'processed'}")
    print()
    print_summary(summarize_dataset(dfs))
    print()

    issues = validate_dataset(dfs)
    if issues:
        print(f"=== VALIDATION: FAILED ({len(issues)} issue(s)) ===")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)
    print("=== VALIDATION: PASSED (all checks green) ===")


if __name__ == "__main__":
    main()
