"""
Counterfactual dataset generator tests.

Uses a smaller n_subscriptions than the real data/raw output for speed, same
convention as tests/test_dataset_generation.py.
"""
import numpy as np
import pandas as pd
import pytest

from data.generate_counterfactual_dataset import (
    COUNTERFACTUAL_SEED_OFFSET,
    generate_counterfactual_outcomes,
    summarize_counterfactual_outcomes,
    validate_counterfactual_outcomes,
)
from data.generate_synthetic_dataset import RETRY_CANDIDATE_TYPES, generate_dataset

TEST_SEED = 42
TEST_N = 80


@pytest.fixture(scope="module")
def base_dataset() -> dict[str, pd.DataFrame]:
    return generate_dataset(seed=TEST_SEED, n_subscriptions=TEST_N)


@pytest.fixture(scope="module")
def counterfactual(base_dataset) -> pd.DataFrame:
    rng = np.random.default_rng(TEST_SEED + COUNTERFACTUAL_SEED_OFFSET)
    return generate_counterfactual_outcomes(rng, base_dataset["subscriptions"], base_dataset["failure_events"], base_dataset["retry_candidates"])


def test_validation_reports_no_issues(counterfactual, base_dataset):
    issues = validate_counterfactual_outcomes(counterfactual, base_dataset["failure_events"])
    assert issues == []


def test_exactly_five_candidates_per_event(counterfactual, base_dataset):
    counts = counterfactual.groupby("event_id").size()
    assert len(counts) == len(base_dataset["failure_events"])
    assert (counts == 5).all()


def test_candidate_ids_are_unique(counterfactual):
    assert not counterfactual["counterfactual_id"].duplicated().any()


def test_all_five_candidate_types_present_per_event(counterfactual):
    per_event_types = counterfactual.groupby("event_id")["candidate_type"].apply(set)
    assert (per_event_types == set(RETRY_CANDIDATE_TYPES)).all()


def test_counterfactual_outcomes_reference_valid_events(counterfactual, base_dataset):
    assert set(counterfactual["event_id"]).issubset(set(base_dataset["failure_events"]["event_id"]))


def test_counterfactual_outcomes_occur_after_failure(counterfactual, base_dataset):
    joined = counterfactual.merge(base_dataset["failure_events"][["event_id", "failure_timestamp"]], on="event_id")
    assert (joined["candidate_datetime"] > joined["failure_timestamp"]).all()


def test_recovered_at_after_failure_when_recovered(counterfactual, base_dataset):
    joined = counterfactual.merge(base_dataset["failure_events"][["event_id", "failure_timestamp"]], on="event_id")
    recovered = joined[joined["recovered_within_14d"]]
    assert (recovered["recovered_at"] > recovered["failure_timestamp"]).all()
    assert (recovered["recovered_at"] <= recovered["failure_timestamp"] + pd.Timedelta(days=14)).all()


def test_latent_probability_in_unit_interval(counterfactual):
    assert counterfactual["recovery_probability_latent"].between(0.0, 1.0).all()


def test_recovered_amount_never_exceeds_original_amount(counterfactual, base_dataset):
    joined = counterfactual.merge(base_dataset["failure_events"][["event_id", "amount"]], on="event_id")
    assert (joined["amount_recovered"] <= joined["amount"]).all()
    assert (joined["amount_recovered"] >= 0).all()


def test_candidate_beyond_horizon_can_never_be_recorded_recovered(counterfactual, base_dataset):
    joined = counterfactual.merge(base_dataset["failure_events"][["event_id", "failure_timestamp"]], on="event_id")
    beyond_horizon = joined["candidate_datetime"] > joined["failure_timestamp"] + pd.Timedelta(days=14)
    assert not (beyond_horizon & joined["recovered_within_14d"]).any()


def test_no_single_candidate_type_dominates_the_oracle_selection(counterfactual, base_dataset):
    summary = summarize_counterfactual_outcomes(counterfactual, base_dataset["subscriptions"])
    assert summary["max_single_candidate_oracle_share"] < 0.90
    assert len(summary["oracle_candidate_distribution"]) >= 3  # several distinct types actually win, not just one or two


def test_reproducibility_same_seed_produces_identical_counterfactual_data(base_dataset):
    rng1 = np.random.default_rng(TEST_SEED + COUNTERFACTUAL_SEED_OFFSET)
    rng2 = np.random.default_rng(TEST_SEED + COUNTERFACTUAL_SEED_OFFSET)
    first = generate_counterfactual_outcomes(rng1, base_dataset["subscriptions"], base_dataset["failure_events"], base_dataset["retry_candidates"])
    second = generate_counterfactual_outcomes(rng2, base_dataset["subscriptions"], base_dataset["failure_events"], base_dataset["retry_candidates"])
    assert first.equals(second)


def test_candidate_timing_affects_latent_outcomes(counterfactual, base_dataset):
    """The core causal requirement: candidate_type must not be inert --
    different candidates for the SAME event must get different latent
    probabilities at least some of the time (not a constant offset, and not
    always the same winner)."""
    varies_within_event = counterfactual.groupby("event_id")["recovery_probability_latent"].nunique()
    assert (varies_within_event > 1).mean() > 0.95  # virtually every event's 5 candidates differ

    winners = counterfactual.loc[counterfactual.groupby("event_id")["recovery_probability_latent"].idxmax(), "candidate_type"]
    assert winners.nunique() >= 3  # at least 3 distinct candidate types win somewhere in this smaller test dataset


def test_payday_sensitive_archetype_shows_larger_candidate_spread_than_insensitive_archetype(counterfactual, base_dataset):
    """cash_strapped_cyclical (high CF_PAYDAY_SENSITIVITY) should show a
    materially larger latent-probability spread across its 5 candidates,
    on average, than quiet_canceller (near-zero sensitivity) -- confirms
    the archetype x timing interaction the brief requires, without
    hardcoding exact probability values (noise-dependent)."""
    archetype_by_sub = base_dataset["subscriptions"].set_index("subscription_id")["archetype"].to_dict()
    cf = counterfactual.copy()
    cf["archetype"] = cf["subscription_id"].map(archetype_by_sub)

    spread_by_event = cf.groupby(["event_id", "archetype"])["recovery_probability_latent"].agg(lambda s: s.max() - s.min())
    spread_by_archetype = spread_by_event.reset_index().groupby("archetype")["recovery_probability_latent"].mean()

    assert spread_by_archetype["cash_strapped_cyclical"] > spread_by_archetype["quiet_canceller"]
