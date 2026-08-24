"""
Day-7 ranking-metric correctness tests -- hand-computed toy examples, no
model or dataset involved (evaluation/ranking_metrics.py is pure).
"""
import numpy as np
import pytest

from evaluation.ranking_metrics import (
    ndcg_at_5,
    pairwise_concordant_pairs,
    reciprocal_rank,
    regret,
    top_k_accuracy,
    within_event_rank,
)

TYPES = np.array(["immediate", "plus_1_day_morning", "payday_window", "plus_3_days", "month_end_window"])
TRUTH = np.array([0.2, 0.5, 0.9, 0.6, 0.3])  # payday_window (index 2) is truly best
PERFECT_SCORES = TRUTH.copy()  # ranks candidates in exactly the true order
WORST_SCORES = np.array([0.9, 0.2, 0.1, 0.3, 0.5])  # ranks the true-worst candidate (immediate) first, true-best (payday_window) last


def test_top1_accuracy_true_when_best_ranked_first():
    assert top_k_accuracy(PERFECT_SCORES, TRUTH, TYPES, k=1) is True


def test_top1_accuracy_false_when_best_not_ranked_first():
    assert top_k_accuracy(WORST_SCORES, TRUTH, TYPES, k=1) is False


def test_top2_accuracy_true_when_best_within_top_two():
    scores = np.array([0.9, 0.5, 0.8, 0.3, 0.2])  # true-best (payday_window) ranked 2nd
    assert top_k_accuracy(scores, TRUTH, TYPES, k=1) is False
    assert top_k_accuracy(scores, TRUTH, TYPES, k=2) is True


def test_reciprocal_rank_is_one_when_best_ranked_first():
    assert reciprocal_rank(PERFECT_SCORES, TRUTH, TYPES) == pytest.approx(1.0)


def test_reciprocal_rank_is_one_fifth_when_best_ranked_last():
    assert reciprocal_rank(WORST_SCORES, TRUTH, TYPES) == pytest.approx(1.0 / 5)


def test_within_event_rank_matches_reciprocal_rank_inverse():
    assert within_event_rank(PERFECT_SCORES, TRUTH, TYPES) == 1
    assert within_event_rank(WORST_SCORES, TRUTH, TYPES) == 5


def test_ndcg_at_5_is_one_for_perfect_ranking():
    assert ndcg_at_5(PERFECT_SCORES, TRUTH) == pytest.approx(1.0)


def test_ndcg_at_5_is_less_than_one_for_imperfect_ranking():
    assert ndcg_at_5(WORST_SCORES, TRUTH) < 1.0


def test_pairwise_concordant_pairs_all_concordant_for_perfect_scores():
    concordant, total = pairwise_concordant_pairs(PERFECT_SCORES, TRUTH)
    assert total == 10  # C(5,2), all pairs have distinct truth values here
    assert concordant == total


def test_pairwise_concordant_pairs_partially_discordant_for_worst_scores():
    concordant, total = pairwise_concordant_pairs(WORST_SCORES, TRUTH)
    assert 0 < concordant < total


def test_pairwise_concordant_pairs_ignores_tied_truth_values():
    truth_with_tie = np.array([0.5, 0.5, 0.9, 0.6, 0.3])
    _concordant, total = pairwise_concordant_pairs(PERFECT_SCORES, truth_with_tie)
    assert total == 9  # one fewer pair: the tied (0,1) pair contributes nothing


def test_regret_is_zero_for_perfect_scores():
    assert regret(PERFECT_SCORES, TRUTH, TYPES, amount=1000.0) == pytest.approx(0.0)


def test_regret_is_positive_and_matches_formula_for_worst_scores():
    amount = 1000.0
    r = regret(WORST_SCORES, TRUTH, TYPES, amount)
    expected = (TRUTH.max() - TRUTH[np.argmax(WORST_SCORES)]) * amount
    assert r == pytest.approx(expected)
    assert r > 0


def test_regret_respects_valid_mask():
    amount = 1000.0
    # payday_window (truly best, index 2) is invalid -- oracle and policy must both be restricted to valid candidates
    valid_mask = np.array([True, True, False, True, True])
    r = regret(PERFECT_SCORES, TRUTH, TYPES, amount, valid_mask=valid_mask)
    # among valid candidates, plus_3_days (index 3, truth=0.6) is now the best; perfect scores still rank it best among valid ones
    assert r == pytest.approx(0.0)
