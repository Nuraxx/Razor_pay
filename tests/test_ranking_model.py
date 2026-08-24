"""
Day-7 pairwise ranking model tests.

Uses a small synthetic dataset generated fresh (not the committed
data/raw/*.csv) so these tests are fast and self-contained -- same pattern
as tests/test_candidate_model.py.
"""
import numpy as np
import pandas as pd
import pytest

from data.generate_counterfactual_dataset import COUNTERFACTUAL_SEED_OFFSET, generate_counterfactual_outcomes
from data.generate_synthetic_dataset import generate_dataset
from model.candidate_preprocessing import build_candidate_level_dataset_from_tables, split_candidate_dataset
from model.ranking_preprocessing import EXCLUDED_COLUMNS, FEATURE_COLUMNS, select_features_and_target
from model.train_ranking_model import build_column_transformer, build_pairwise_dataset, fit_pipeline, predict_ranking_scores, score_candidates_for_event

TEST_SEED = 42
TEST_N = 120


@pytest.fixture(scope="module")
def base_dataset() -> dict[str, pd.DataFrame]:
    return generate_dataset(seed=TEST_SEED, n_subscriptions=TEST_N)


@pytest.fixture(scope="module")
def candidate_dataset(base_dataset) -> pd.DataFrame:
    rng = np.random.default_rng(TEST_SEED + COUNTERFACTUAL_SEED_OFFSET)
    counterfactual = generate_counterfactual_outcomes(rng, base_dataset["subscriptions"], base_dataset["failure_events"], base_dataset["retry_candidates"])
    return build_candidate_level_dataset_from_tables(counterfactual, base_dataset["retry_candidates"], base_dataset["failure_events"], base_dataset["subscriptions"])


@pytest.fixture(scope="module")
def candidate_splits(candidate_dataset):
    return split_candidate_dataset(candidate_dataset)


@pytest.fixture(scope="module")
def fitted(candidate_splits) -> dict:
    train_df, val_df, _test_df = candidate_splits
    return fit_pipeline(train_df, val_df)


# ---------------------------------------------------------------------------
# Ranking groups / no leakage
# ---------------------------------------------------------------------------

def test_ranking_groups_contain_exactly_five_candidates(candidate_dataset):
    counts = candidate_dataset.groupby("event_id").size()
    assert (counts == 5).all()


def test_same_event_never_crosses_splits(candidate_dataset):
    by_sub = candidate_dataset.groupby("subscription_id")["split"].nunique()
    assert (by_sub == 1).all()


def test_ranking_feature_list_excludes_distractors_and_hidden_fields():
    leaky_or_distractor = {"app_version", "device_build", "ui_theme", "archetype", "recovery_probability_latent", "recovered_at", "recovered_via", "amount_recovered", "recovered_within_14d"}
    assert leaky_or_distractor.isdisjoint(set(FEATURE_COLUMNS))
    # every one of them is either the target itself or explicitly documented as excluded, never silently dropped
    for col in leaky_or_distractor - {"recovered_within_14d"}:
        assert col in EXCLUDED_COLUMNS, f"{col} is neither a feature nor documented in EXCLUDED_COLUMNS"


def test_no_label_leakage_in_pairwise_features(candidate_splits):
    train_df, _val, _test = candidate_splits
    X, y = select_features_and_target(train_df)
    assert "recovered_within_14d" not in X.columns
    assert "recovery_probability_latent" not in X.columns
    assert y.name == "recovered_within_14d"


# ---------------------------------------------------------------------------
# Pairwise dataset construction
# ---------------------------------------------------------------------------

def test_build_pairwise_dataset_only_pairs_within_the_same_event():
    Z = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1, 0, 1, 0])
    event_ids = np.array(["e1", "e1", "e2", "e2"])

    pair_X, pair_y = build_pairwise_dataset(Z, y, event_ids)

    # 1 informative pair per event (1 pos vs 1 neg), x2 directions each = 4 rows total
    assert len(pair_y) == 4
    diffs = {row[0] for row in pair_X}
    # e1: Z[0](pos)-Z[1](neg) = -1.0, and its reverse +1.0
    assert {-1.0, 1.0}.issubset(diffs)
    # e2: Z[2](pos)-Z[3](neg) = -1.0 too (same value, different event) -- but a
    # CROSS-event pair like Z[0]-Z[2] (=-2.0) or Z[0]-Z[3] (=-3.0) must never appear
    assert -2.0 not in diffs and -3.0 not in diffs and 2.0 not in diffs and 3.0 not in diffs


def test_build_pairwise_dataset_includes_both_directions_symmetrically():
    Z = np.array([[5.0], [1.0]])
    y = np.array([1, 0])
    event_ids = np.array(["e1", "e1"])

    pair_X, pair_y = build_pairwise_dataset(Z, y, event_ids)

    assert len(pair_y) == 2
    assert set(pair_y.tolist()) == {0, 1}
    winner_row = pair_X[pair_y == 1][0]
    loser_row = pair_X[pair_y == 0][0]
    np.testing.assert_array_almost_equal(winner_row, -loser_row)


def test_build_pairwise_dataset_produces_no_pairs_for_uniform_labels():
    Z = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1, 1, 1])  # all recovered -- no informative comparison
    event_ids = np.array(["e1", "e1", "e1"])

    pair_X, pair_y = build_pairwise_dataset(Z, y, event_ids)
    assert len(pair_y) == 0


def test_build_pairwise_dataset_correct_pair_count_multiple_pos_and_neg():
    # event with 2 positive, 3 negative candidates -> 2*3 = 6 informative pairs x2 directions = 12
    Z = np.arange(5).reshape(5, 1).astype(float)
    y = np.array([1, 1, 0, 0, 0])
    event_ids = np.array(["e1"] * 5)

    _pair_X, pair_y = build_pairwise_dataset(Z, y, event_ids)
    assert len(pair_y) == 12


# ---------------------------------------------------------------------------
# Model trains, scores, deterministic
# ---------------------------------------------------------------------------

def test_ranking_model_trains(fitted):
    assert fitted["model"] is not None
    assert fitted["model_config"]["n_pairwise_train_examples"] > 0


def test_ranking_score_produced_for_every_candidate(fitted, candidate_splits):
    _train, _val, test_df = candidate_splits
    scores = predict_ranking_scores(test_df, fitted)
    assert len(scores) == len(test_df)
    assert scores.notna().all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_round_robin_scores_are_bounded_and_shaped_correctly(fitted):
    transformer = fitted["transformer"]
    dummy = pd.DataFrame(
        {**{c: [0.0] * 5 for c in ["day_of_month", "days_to_nearest_payday_window", "amount", "prior_if_failure_count", "prior_if_self_resolved_rate", "tenure_days", "hours_from_failure", "candidate_day_of_month", "candidate_days_to_payday"]},
         **{c: ["x"] * 5 for c in ["plan_tier", "primary_instrument", "city_tier", "bank_network_conditions", "network_latency_bucket", "candidate_type", "candidate_day_of_week"]},
         **{c: [0] * 5 for c in ["issuing_bank_downtime_flag", "is_month_end_settlement_rush", "candidate_is_payday_aligned", "candidate_is_month_end_aligned", "prior_if_self_resolved_rate_missing"]}}
    )
    Z = transformer.transform(dummy)
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    scores = score_candidates_for_event(Z, fitted["model"])
    assert scores.shape == (5,)
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_score_candidates_for_event_single_candidate_returns_neutral_score(fitted):
    # A single-row group has no opponent to compare against -- must not
    # crash, and returns a neutral 0.5 rather than claiming merit either way.
    n_features = len(fitted["model"].feature_names_)
    single_candidate = np.zeros((1, n_features))
    scores = score_candidates_for_event(single_candidate, fitted["model"])
    assert scores.shape == (1,)
    assert scores[0] == pytest.approx(0.5)


def test_predictions_are_deterministic(fitted, candidate_splits):
    _train, _val, test_df = candidate_splits
    scores_a = predict_ranking_scores(test_df, fitted)
    scores_b = predict_ranking_scores(test_df, fitted)
    pd.testing.assert_series_equal(scores_a, scores_b)


def test_reproducibility_same_inputs_produce_identical_model(candidate_splits):
    train_df, val_df, test_df = candidate_splits
    fitted_a = fit_pipeline(train_df, val_df)
    fitted_b = fit_pipeline(train_df, val_df)
    scores_a = predict_ranking_scores(test_df, fitted_a)
    scores_b = predict_ranking_scores(test_df, fitted_b)
    pd.testing.assert_series_equal(scores_a, scores_b)


def test_column_transformer_output_is_purely_numeric(candidate_splits):
    train_df, _val, _test = candidate_splits
    X, _y = select_features_and_target(train_df)
    from model.candidate_preprocessing import PriorSelfResolvedImputer

    imputer = PriorSelfResolvedImputer().fit(X)
    X_imp = imputer.transform(X)
    transformer = build_column_transformer()
    Z = transformer.fit_transform(X_imp)
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    assert np.issubdtype(Z.dtype, np.floating)
