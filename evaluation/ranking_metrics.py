"""
Day-7 ranking metrics -- pure functions, no I/O, so each is independently
testable (tests/test_ranking_metrics.py) against hand-computed toy examples.

Every metric here takes ONE event's candidates at a time (a "ranking
group") as parallel arrays: `scores` (a policy's predicted preference,
higher = better), `truth` (ground truth preference to rank against --
typically `recovery_probability_latent`, the Oracle's own signal, for
apples-to-apples continuity with Day 6's Oracle definition), and
`candidate_types` (labels, for readability / top-k membership checks).
`evaluate_ranking_policy.py` aggregates these across all test events.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import ndcg_score


def _best_candidate(truth: np.ndarray, candidate_types: np.ndarray) -> str:
    return candidate_types[int(np.argmax(truth))]


def _predicted_rank_of(candidate: str, scores: np.ndarray, candidate_types: np.ndarray) -> int:
    """1-indexed rank of `candidate` in descending-score order (rank 1 = highest score)."""
    order = np.argsort(-scores)
    ranked_types = candidate_types[order]
    return int(np.where(ranked_types == candidate)[0][0]) + 1


def top_k_accuracy(scores: np.ndarray, truth: np.ndarray, candidate_types: np.ndarray, k: int) -> bool:
    """Is the true-best candidate (argmax truth) among the top-k predicted-score candidates?"""
    best = _best_candidate(truth, candidate_types)
    rank = _predicted_rank_of(best, scores, candidate_types)
    return rank <= k


def reciprocal_rank(scores: np.ndarray, truth: np.ndarray, candidate_types: np.ndarray) -> float:
    """1 / (predicted rank of the true-best candidate). 1.0 if ranked first, 0.2 if ranked 5th of 5."""
    best = _best_candidate(truth, candidate_types)
    rank = _predicted_rank_of(best, scores, candidate_types)
    return 1.0 / rank


def ndcg_at_5(scores: np.ndarray, truth: np.ndarray) -> float:
    """Standard NDCG@5 (sklearn.metrics.ndcg_score) using `truth` (continuous,
    e.g. latent probability) as the relevance grade -- well-defined for
    continuous relevance, not just discrete 0/1/2 grades."""
    return float(ndcg_score(np.asarray(truth).reshape(1, -1), np.asarray(scores).reshape(1, -1), k=5))


def within_event_rank(scores: np.ndarray, truth: np.ndarray, candidate_types: np.ndarray) -> int:
    """Predicted rank (1=best) of the true-best candidate -- same quantity
    reciprocal_rank inverts; reported directly for "mean within-event rank"."""
    best = _best_candidate(truth, candidate_types)
    return _predicted_rank_of(best, scores, candidate_types)


def pairwise_concordant_pairs(scores: np.ndarray, truth: np.ndarray) -> tuple[int, int]:
    """(n_concordant, n_total) among pairs with truth_i != truth_j: a pair is
    concordant if the predicted score ordering agrees with the truth
    ordering. Aggregate many events' (concordant, total) tuples for a global
    pairwise accuracy = sum(concordant) / sum(total)."""
    n = len(scores)
    concordant = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if truth[i] == truth[j]:
                continue
            total += 1
            truth_prefers_i = truth[i] > truth[j]
            score_prefers_i = scores[i] > scores[j]
            if truth_prefers_i == score_prefers_i:
                concordant += 1
    return concordant, total


def regret(scores: np.ndarray, truth: np.ndarray, candidate_types: np.ndarray, amount: float, valid_mask: np.ndarray | None = None) -> float:
    """oracle_expected_value - policy_expected_value, both using `truth` as
    the probability (same definition as Day 6). `valid_mask` restricts both
    the oracle's and the policy's choice to guardrail-valid candidates (all
    True if not supplied -- callers should supply it in production use)."""
    if valid_mask is None:
        valid_mask = np.ones(len(scores), dtype=bool)
    if not valid_mask.any():
        return 0.0
    oracle_idx = np.where(valid_mask, truth, -np.inf).argmax()
    policy_idx = np.where(valid_mask, scores, -np.inf).argmax()
    return float((truth[oracle_idx] - truth[policy_idx]) * amount)
