"""
Synthetic dataset generator tests.

Uses a smaller n_subscriptions than the real data/raw output for speed --
the generation mechanism doesn't depend on scale, so a smaller run exercises
the same code paths.
"""
import pandas as pd
import pytest

from data.generate_synthetic_dataset import (
    RETRY_CANDIDATE_TYPES,
    generate_dataset,
    validate_dataset,
    write_dataset,
)

TEST_SEED = 42
TEST_N = 80


@pytest.fixture(scope="module")
def dataset() -> dict[str, pd.DataFrame]:
    return generate_dataset(seed=TEST_SEED, n_subscriptions=TEST_N)


def test_validate_dataset_reports_no_issues(dataset):
    issues = validate_dataset(dataset)
    assert issues == []


def test_reproducibility_same_seed_produces_identical_data():
    first = generate_dataset(seed=TEST_SEED, n_subscriptions=TEST_N)
    second = generate_dataset(seed=TEST_SEED, n_subscriptions=TEST_N)
    for table in ("subscriptions", "failure_events", "retry_candidates", "recovery_outcomes", "train", "validation", "test"):
        assert first[table].equals(second[table]), f"{table} differs between two runs with the same seed"


def test_different_seed_produces_different_data():
    first = generate_dataset(seed=TEST_SEED, n_subscriptions=TEST_N)
    second = generate_dataset(seed=TEST_SEED + 1, n_subscriptions=TEST_N)
    assert not first["subscriptions"].equals(second["subscriptions"])


def test_subscription_count_in_expected_range():
    # Real generation targets ~150-250; this just confirms n_subscriptions is respected.
    d = generate_dataset(seed=TEST_SEED, n_subscriptions=200)
    assert len(d["subscriptions"]) == 200


def test_split_ratios_approximately_60_20_20(dataset):
    n = len(dataset["subscriptions"])
    train_n = dataset["subscriptions"]["split"].eq("train").sum()
    val_n = dataset["subscriptions"]["split"].eq("validation").sum()
    test_n = dataset["subscriptions"]["split"].eq("test").sum()

    assert train_n + val_n + test_n == n
    assert abs(train_n / n - 0.60) < 0.05
    assert abs(val_n / n - 0.20) < 0.05
    assert abs(test_n / n - 0.20) < 0.05


def test_zero_subscription_overlap_between_splits(dataset):
    train_ids = set(dataset["train"]["subscription_id"])
    val_ids = set(dataset["validation"]["subscription_id"])
    test_ids = set(dataset["test"]["subscription_id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_a_subscriptions_full_event_history_stays_in_one_split(dataset):
    """A subscription with multiple failures must have ALL of them in the same split."""
    sub_to_split = dict(zip(dataset["subscriptions"]["subscription_id"], dataset["subscriptions"]["split"]))
    events_with_split = dataset["failure_events"].assign(
        split=dataset["failure_events"]["subscription_id"].map(sub_to_split)
    )
    for sub_id, group in events_with_split.groupby("subscription_id"):
        assert group["split"].nunique() == 1


def test_valid_foreign_key_relationships(dataset):
    valid_event_ids = set(dataset["failure_events"]["event_id"])
    assert set(dataset["recovery_outcomes"]["event_id"]).issubset(valid_event_ids)
    assert set(dataset["retry_candidates"]["event_id"]).issubset(valid_event_ids)

    valid_subscription_ids = set(dataset["subscriptions"]["subscription_id"])
    assert set(dataset["failure_events"]["subscription_id"]).issubset(valid_subscription_ids)


def test_no_duplicate_subscription_ids(dataset):
    assert not dataset["subscriptions"]["subscription_id"].duplicated().any()


def test_no_duplicate_event_ids(dataset):
    assert not dataset["failure_events"]["event_id"].duplicated().any()
    assert not dataset["recovery_outcomes"]["event_id"].duplicated().any()


def test_no_duplicate_retry_candidate_ids(dataset):
    assert not dataset["retry_candidates"]["retry_candidate_id"].duplicated().any()


def test_no_missing_required_values(dataset):
    required = {
        "subscriptions": ["subscription_id", "plan_tier", "monthly_amount", "signup_date", "primary_instrument", "city_tier", "tenure_days"],
        "failure_events": ["event_id", "subscription_id", "failure_timestamp", "error_reason", "amount", "prior_if_failure_count"],
        "recovery_outcomes": ["event_id", "subscription_id", "recovered_within_14d", "recovered_via", "final_amount_recovered"],
        "retry_candidates": ["retry_candidate_id", "event_id", "candidate_type", "candidate_datetime"],
    }
    for table, cols in required.items():
        for col in cols:
            assert not dataset[table][col].isna().any(), f"{table}.{col} has unexpected missing values"


def test_prior_self_resolved_rate_is_missing_only_when_no_prior_failures(dataset):
    fe = dataset["failure_events"]
    first_failures = fe[fe["prior_if_failure_count"] == 0]
    later_failures = fe[fe["prior_if_failure_count"] > 0]
    assert first_failures["prior_if_self_resolved_rate"].isna().all()
    assert not later_failures["prior_if_self_resolved_rate"].isna().any()


def test_error_reason_always_insufficient_fund(dataset):
    assert (dataset["failure_events"]["error_reason"] == "insufficient_fund").all()


def test_hidden_archetype_excluded_from_processed_datasets(dataset):
    for split in ("train", "validation", "test"):
        assert "archetype" not in dataset[split].columns
        assert "split" not in dataset[split].columns
    # But it IS present in the raw, internal-only subscriptions table.
    assert "archetype" in dataset["subscriptions"].columns


def test_distractor_features_present_and_not_used_as_split_key(dataset):
    for col in ("app_version", "device_build", "ui_theme"):
        assert col in dataset["failure_events"].columns
        assert col in dataset["train"].columns


def test_distractor_features_are_statistically_independent_of_archetype(dataset):
    """Evaluation-compliance audit: the specification's whole reason for
    including distractor features is to check that a trained model's
    feature importances correctly ignore them (Section 10, item 4) -- that
    property can only hold if the DATA itself never gives a distractor a
    real, exploitable association with the hidden archetype (which would
    make it a backdoor archetype proxy) or with the recovery label. This
    tests the GENERATION methodology directly (a chi-square independence
    test on the actual generated data), which is what's fixable if wrong --
    a small-sample trained model's own feature-importance ranking is a
    separate, expected small-n phenomenon this test does not (and cannot)
    control for."""
    from scipy.stats import chi2_contingency

    events = dataset["failure_events"].merge(dataset["subscriptions"][["subscription_id", "archetype"]], on="subscription_id")
    labels = dataset["recovery_outcomes"].set_index("event_id")["recovered_within_14d"]
    events = events.merge(labels.rename("recovered_within_14d"), on="event_id")

    for col in ("app_version", "device_build", "ui_theme"):
        contingency = pd.crosstab(events[col], events["archetype"])
        _, p_value, _, _ = chi2_contingency(contingency)
        assert p_value > 0.01, f"{col} shows a statistically significant association with archetype (p={p_value:.4f}) -- possible backdoor archetype leak"

        contingency_label = pd.crosstab(events[col], events["recovered_within_14d"])
        _, p_value_label, _, _ = chi2_contingency(contingency_label)
        assert p_value_label > 0.01, f"{col} shows a statistically significant association with the recovery label (p={p_value_label:.4f}) -- distractor should have no causal role"


def test_recovered_amount_never_exceeds_transaction_amount(dataset):
    joined = dataset["recovery_outcomes"].merge(dataset["failure_events"][["event_id", "amount"]], on="event_id")
    assert (joined["final_amount_recovered"] <= joined["amount"]).all()
    assert (joined["final_amount_recovered"] >= 0).all()


def test_recovery_timestamps_valid(dataset):
    joined = dataset["recovery_outcomes"].merge(
        dataset["failure_events"][["event_id", "failure_timestamp"]], on="event_id"
    )
    # recovered_at present iff recovered_within_14d True.
    assert (joined["recovered_within_14d"] == joined["recovered_at"].notna()).all()

    recovered = joined[joined["recovered_within_14d"]]
    assert (recovered["recovered_at"] > recovered["failure_timestamp"]).all()
    assert (recovered["recovered_at"] <= recovered["failure_timestamp"] + pd.Timedelta(days=14)).all()

    not_recovered_via = dataset["recovery_outcomes"].loc[~dataset["recovery_outcomes"]["recovered_within_14d"], "recovered_via"]
    assert (not_recovered_via == "none").all()


def test_candidate_retry_times_valid(dataset):
    joined = dataset["retry_candidates"].merge(
        dataset["failure_events"][["event_id", "failure_timestamp"]], on="event_id"
    )
    assert joined["candidate_type"].isin(RETRY_CANDIDATE_TYPES).all()
    assert (joined["candidate_datetime"] > joined["failure_timestamp"]).all()
    # Every event has exactly one of each candidate type.
    counts = dataset["retry_candidates"].groupby("event_id")["candidate_type"].apply(lambda s: sorted(s) == sorted(RETRY_CANDIDATE_TYPES))
    assert counts.all()


def test_archetype_proportions_are_approximately_as_specified():
    d = generate_dataset(seed=TEST_SEED, n_subscriptions=200)
    counts = d["subscriptions"]["archetype"].value_counts(normalize=True)
    assert abs(counts["reliable"] - 0.50) < 0.08
    assert abs(counts["cash_strapped_cyclical"] - 0.25) < 0.08
    assert abs(counts["chronic_struggler"] - 0.15) < 0.08
    assert abs(counts["quiet_canceller"] - 0.10) < 0.08


def test_class_balance_is_not_degenerate(dataset):
    """The label must not collapse to (near-)always-one-value -- that would
    make the learning problem trivial/undefined, contrary to this generator's goal."""
    rate = dataset["recovery_outcomes"]["recovered_within_14d"].mean()
    assert 0.05 < rate < 0.95


def test_cli_writes_expected_files(tmp_path):
    d = generate_dataset(seed=TEST_SEED, n_subscriptions=TEST_N)
    write_dataset(d, tmp_path)

    for fname in ("subscriptions.csv", "failure_events.csv", "retry_candidates.csv", "recovery_outcomes.csv"):
        assert (tmp_path / "raw" / fname).exists()
    for fname in ("train.csv", "validation.csv", "test.csv"):
        assert (tmp_path / "processed" / fname).exists()

    reloaded = pd.read_csv(tmp_path / "raw" / "subscriptions.csv")
    assert len(reloaded) == TEST_N
