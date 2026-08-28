"""
FIX pass tests: evaluation/statistics.py -- McNemar's test (paired binary
outcome only) and the bootstrap CI (paired continuous ₹ delta), both pure
functions operating on caller-supplied paired sequences, no DB/no I/O.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.statistics import BootstrapCIResult, McNemarResult, bootstrap_delta_ci, mcnemar_test

# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------


class TestMcNemarTest:
    def test_known_toy_contingency_table(self):
        # Classic textbook example: a=1 (both), b=10 (A only), c=3 (B only), d=6 (neither) -- 20 pairs total.
        outcomes_a = [True] * 1 + [True] * 10 + [False] * 3 + [False] * 6
        outcomes_b = [True] * 1 + [False] * 10 + [True] * 3 + [False] * 6
        result = mcnemar_test(outcomes_a, outcomes_b, policy_a="A", policy_b="B")
        assert result.both_recovered == 1
        assert result.only_a_recovered == 10
        assert result.only_b_recovered == 3
        assert result.neither_recovered == 6
        assert result.n_paired_events == 20
        assert result.method == "exact_binomial"
        assert result.exact is True
        # exact binomial two-sided p-value for b=10, c=3 (n=13, k=min=3)
        assert result.p_value == pytest.approx(0.09229, abs=1e-4)

    def test_b_zero_edge_case(self):
        # b=0, c=5 -- policy_b strictly better on every discordant pair.
        outcomes_a = [False] * 5 + [True] * 2
        outcomes_b = [True] * 5 + [True] * 2
        result = mcnemar_test(outcomes_a, outcomes_b, policy_a="A", policy_b="B")
        assert result.only_a_recovered == 0
        assert result.only_b_recovered == 5
        assert result.method == "exact_binomial"
        assert 0.0 < result.p_value <= 1.0

    def test_c_zero_edge_case(self):
        outcomes_a = [True] * 5 + [True] * 2
        outcomes_b = [False] * 5 + [True] * 2
        result = mcnemar_test(outcomes_a, outcomes_b, policy_a="A", policy_b="B")
        assert result.only_a_recovered == 5
        assert result.only_b_recovered == 0
        assert 0.0 < result.p_value <= 1.0

    def test_balanced_b_and_c_discordant_pairs(self):
        """HARDENING PASS: b == c (nonzero, evenly split discordant pairs) --
        the exact binomial test's mode is at k=n/2, so an evenly-balanced
        split is the LEAST extreme possible outcome and always yields
        p_value == 1.0 exactly (verified independently against
        scipy.stats.binomtest(6, n=12, p=0.5) directly, not merely asserted
        by construction)."""
        outcomes_a = [True] * 6 + [False] * 6 + [True] * 3 + [False] * 3
        outcomes_b = [False] * 6 + [True] * 6 + [True] * 3 + [False] * 3
        result = mcnemar_test(outcomes_a, outcomes_b, policy_a="A", policy_b="B")
        assert result.only_a_recovered == 6
        assert result.only_b_recovered == 6
        assert result.method == "exact_binomial"
        assert result.p_value == pytest.approx(1.0)

    def test_both_b_and_c_zero_no_discordant_pairs(self):
        # Two policies that agree on every single event -- no discordant pairs at all.
        outcomes_a = [True, False, True, True, False]
        outcomes_b = [True, False, True, True, False]
        result = mcnemar_test(outcomes_a, outcomes_b, policy_a="A", policy_b="B")
        assert result.only_a_recovered == 0
        assert result.only_b_recovered == 0
        assert result.method == "no_discordant_pairs"
        assert result.p_value == 1.0
        assert result.statistic is None

    def test_reproducibility_same_inputs_same_result(self):
        outcomes_a = [True, False, True, False, True, True, False]
        outcomes_b = [False, False, True, True, True, False, False]
        r1 = mcnemar_test(outcomes_a, outcomes_b, policy_a="A", policy_b="B")
        r2 = mcnemar_test(outcomes_a, outcomes_b, policy_a="A", policy_b="B")
        assert r1 == r2

    def test_policy_labels_are_reported_correctly(self):
        result = mcnemar_test([True, False], [False, True], policy_a="improved_fallback_policy", policy_b="fixed_retry")
        assert result.policy_a == "improved_fallback_policy"
        assert result.policy_b == "fixed_retry"

    def test_swapping_a_and_b_swaps_b_and_c(self):
        outcomes_a = [True, True, False, False, True]
        outcomes_b = [False, True, True, False, False]
        forward = mcnemar_test(outcomes_a, outcomes_b, policy_a="A", policy_b="B")
        backward = mcnemar_test(outcomes_b, outcomes_a, policy_a="B", policy_b="A")
        assert forward.only_a_recovered == backward.only_b_recovered
        assert forward.only_b_recovered == backward.only_a_recovered
        # p-value is symmetric in b/c (McNemar doesn't care which side is which)
        assert forward.p_value == pytest.approx(backward.p_value)

    def test_chi_square_approximation_available_and_differs_in_method(self):
        outcomes_a = [True] * 10 + [False] * 15
        outcomes_b = [False] * 10 + [True] * 15
        result = mcnemar_test(outcomes_a, outcomes_b, policy_a="A", policy_b="B", exact=False)
        assert result.method == "chi_square_continuity_corrected"
        assert result.exact is False
        assert result.statistic is not None
        assert result.statistic >= 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            mcnemar_test([True, False], [True], policy_a="A", policy_b="B")

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            mcnemar_test([], [], policy_a="A", policy_b="B")

    def test_result_is_a_mcnemar_result_with_to_dict(self):
        result = mcnemar_test([True, False, True], [False, False, True], policy_a="A", policy_b="B")
        assert isinstance(result, McNemarResult)
        d = result.to_dict()
        for key in ("policy_a", "policy_b", "n_paired_events", "both_recovered", "only_a_recovered", "only_b_recovered", "neither_recovered", "statistic", "p_value", "method", "exact"):
            assert key in d


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------


class TestBootstrapDeltaCI:
    def test_deterministic_seed_reproduces_identical_bounds(self):
        rng = np.random.default_rng(0)
        values_a = rng.normal(100, 20, size=50)
        values_b = rng.normal(90, 20, size=50)
        r1 = bootstrap_delta_ci(values_a, values_b, policy_a="A", policy_b="B", metric_name="rs", seed=7, n_resamples=500)
        r2 = bootstrap_delta_ci(values_a, values_b, policy_a="A", policy_b="B", metric_name="rs", seed=7, n_resamples=500)
        assert r1 == r2

    def test_different_seed_can_produce_different_bounds(self):
        rng = np.random.default_rng(0)
        values_a = rng.normal(100, 20, size=50)
        values_b = rng.normal(90, 20, size=50)
        r1 = bootstrap_delta_ci(values_a, values_b, policy_a="A", policy_b="B", metric_name="rs", seed=1, n_resamples=500)
        r2 = bootstrap_delta_ci(values_a, values_b, policy_a="A", policy_b="B", metric_name="rs", seed=2, n_resamples=500)
        assert (r1.lower_bound, r1.upper_bound) != (r2.lower_bound, r2.upper_bound)

    def test_resampling_is_paired_not_independent(self):
        """HARDENING PASS: explicit regression guard for "resample EVENT
        INDICES, not the two value arrays independently"
        (`bootstrap_delta_ci`'s own docstring contract). 8 DISTINCT-valued
        events whose per-event delta (values_a[i] - values_b[i]) is FIXED at
        exactly +5. Correct PAIRED resampling always draws the same index
        for both arrays, so values_a[idx]-values_b[idx] == 5 for every draw
        regardless of which events get picked -- the bootstrap distribution
        must therefore collapse to a ZERO-WIDTH interval at exactly 5*8=40,
        even though the underlying values themselves vary a lot (0..75).
        If the implementation were changed to resample values_a and
        values_b with two INDEPENDENT index draws, the mismatched pairs
        pulled from these genuinely different-valued arrays would produce
        real variance across resamples -- lower_bound would no longer equal
        upper_bound -- so this assertion fails under that regression,
        unlike a same-valued-arrays test (e.g. the zero-variance case below)
        where independent resampling could coincidentally still look
        zero-width."""
        values_b = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        values_a = [v + 5.0 for v in values_b]
        result = bootstrap_delta_ci(values_a, values_b, policy_a="A", policy_b="B", metric_name="rs", n_resamples=5000, seed=11)
        assert result.point_estimate == pytest.approx(5.0 * 8)
        assert result.lower_bound == pytest.approx(5.0 * 8)
        assert result.upper_bound == pytest.approx(5.0 * 8)

    def test_toy_synthetic_data_known_behavior(self):
        # values_a is always exactly 10 more than values_b at every paired
        # event -- every possible resample (with replacement) of ANY size
        # must therefore also show a total delta of exactly 10 * n.
        values_b = [50.0, 60.0, 70.0, 80.0, 90.0]
        values_a = [v + 10.0 for v in values_b]
        result = bootstrap_delta_ci(values_a, values_b, policy_a="A", policy_b="B", metric_name="rs", n_resamples=2000, seed=42)
        assert result.point_estimate == pytest.approx(50.0)  # 10 * 5 events
        assert result.lower_bound == pytest.approx(50.0)
        assert result.upper_bound == pytest.approx(50.0)

    def test_zero_variance_case_all_zero_delta(self):
        values = [10.0, 20.0, 30.0, 40.0]
        result = bootstrap_delta_ci(values, values, policy_a="A", policy_b="B", metric_name="rs", n_resamples=1000, seed=1)
        assert result.point_estimate == 0.0
        assert result.lower_bound == 0.0
        assert result.upper_bound == 0.0

    def test_small_sample_single_event_does_not_crash(self):
        result = bootstrap_delta_ci([100.0], [80.0], policy_a="A", policy_b="B", metric_name="rs", n_resamples=200, seed=1)
        assert result.point_estimate == pytest.approx(20.0)
        assert result.n_events == 1
        # Only one event exists -- every resample IS that one event, so the
        # interval collapses to the point estimate.
        assert result.lower_bound == pytest.approx(20.0)
        assert result.upper_bound == pytest.approx(20.0)

    def test_correct_sign_convention_positive_when_a_beats_b(self):
        result = bootstrap_delta_ci([100.0, 200.0], [50.0, 60.0], policy_a="A", policy_b="B", metric_name="rs", seed=1)
        assert result.point_estimate > 0

    def test_correct_sign_convention_negative_when_b_beats_a(self):
        result = bootstrap_delta_ci([50.0, 60.0], [100.0, 200.0], policy_a="A", policy_b="B", metric_name="rs", seed=1)
        assert result.point_estimate < 0

    def test_confidence_level_widens_interval(self):
        rng = np.random.default_rng(3)
        values_a = rng.normal(100, 30, size=60)
        values_b = rng.normal(95, 30, size=60)
        narrow = bootstrap_delta_ci(values_a, values_b, policy_a="A", policy_b="B", metric_name="rs", seed=1, n_resamples=3000, confidence_level=0.80)
        wide = bootstrap_delta_ci(values_a, values_b, policy_a="A", policy_b="B", metric_name="rs", seed=1, n_resamples=3000, confidence_level=0.99)
        assert (wide.upper_bound - wide.lower_bound) >= (narrow.upper_bound - narrow.lower_bound)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            bootstrap_delta_ci([1.0, 2.0], [1.0], policy_a="A", policy_b="B", metric_name="rs")

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            bootstrap_delta_ci([], [], policy_a="A", policy_b="B", metric_name="rs")

    def test_invalid_confidence_level_raises(self):
        with pytest.raises(ValueError):
            bootstrap_delta_ci([1.0], [1.0], policy_a="A", policy_b="B", metric_name="rs", confidence_level=1.5)

    def test_result_metadata_fields(self):
        result = bootstrap_delta_ci([10.0, 20.0], [5.0, 5.0], policy_a="deployed", policy_b="fixed_retry", metric_name="realized_rs_recovered", n_resamples=100, seed=99, confidence_level=0.90)
        assert isinstance(result, BootstrapCIResult)
        assert result.metric == "realized_rs_recovered"
        assert result.policy_a == "deployed"
        assert result.policy_b == "fixed_retry"
        assert result.n_resamples == 100
        assert result.seed == 99
        assert result.confidence_level == 0.90
        assert result.method == "percentile_bootstrap_paired_event_resampling"
        d = result.to_dict()
        for key in ("metric", "policy_a", "policy_b", "point_estimate", "method", "n_resamples", "seed", "confidence_level", "lower_bound", "upper_bound", "n_events"):
            assert key in d


# ---------------------------------------------------------------------------
# Integration with the real evaluation pipeline (correct policy pairing,
# correct held-out population)
# ---------------------------------------------------------------------------


class TestStatisticalTestsIntegration:
    """Exercises evaluation/evaluate_decision_engine_v4.py::summarize_statistical_tests
    against a small, hand-built `events` DataFrame shaped exactly like the
    real one -- verifies the headline comparison is deployed policy vs Fixed
    Retry, and that only the two required paired columns are read (no new
    population, no new outcome definition)."""

    def test_summarize_statistical_tests_uses_deployed_vs_fixed_retry(self):
        import pandas as pd

        from evaluation.evaluate_decision_engine_v4 import DEPLOYED_POLICY_NAME, HEADLINE_BASELINE_NAME, summarize_statistical_tests

        events = pd.DataFrame(
            {
                "fixed_retry__realized_recovered": [True, True, False, True, False],
                "fixed_retry__realized_amount_recovered": [100.0, 200.0, 0.0, 150.0, 0.0],
                "improved_fallback_policy__realized_recovered": [True, False, False, True, True],
                "improved_fallback_policy__realized_amount_recovered": [100.0, 0.0, 0.0, 150.0, 90.0],
                # Evaluation-compliance audit fix: summarize_statistical_tests
                # now also computes additional_comparisons vs rule_based and
                # vs no_recovery (specification Section 7 -- must check all
                # three baselines), so a realistically-shaped `events` needs
                # these two policies' columns too, not just the headline pair.
                "rule_based__realized_recovered": [True, False, False, True, False],
                "rule_based__realized_amount_recovered": [100.0, 0.0, 0.0, 150.0, 0.0],
                "no_recovery__realized_recovered": [False, False, False, False, False],
                "no_recovery__realized_amount_recovered": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        result = summarize_statistical_tests(events, n_resamples=200, seed=1)
        assert result["mcnemar"]["policy_a"] == DEPLOYED_POLICY_NAME
        assert result["mcnemar"]["policy_b"] == HEADLINE_BASELINE_NAME
        assert result["bootstrap_ci"]["policy_a"] == DEPLOYED_POLICY_NAME
        assert result["bootstrap_ci"]["policy_b"] == HEADLINE_BASELINE_NAME
        assert result["population"]["n_events"] == 5
        assert result["population"]["held_out_split"] == "test"
        # b: deployed-only recovered (event 5), c: fixed_retry-only recovered (event 2)
        assert result["mcnemar"]["only_a_recovered"] == 1
        assert result["mcnemar"]["only_b_recovered"] == 1

    def test_summarize_statistical_tests_includes_rule_based_and_no_recovery_comparisons(self):
        import pandas as pd

        from evaluation.evaluate_decision_engine_v4 import summarize_statistical_tests

        events = pd.DataFrame(
            {
                "fixed_retry__realized_recovered": [True, True, False, True, False],
                "fixed_retry__realized_amount_recovered": [100.0, 200.0, 0.0, 150.0, 0.0],
                "improved_fallback_policy__realized_recovered": [True, False, False, True, True],
                "improved_fallback_policy__realized_amount_recovered": [100.0, 0.0, 0.0, 150.0, 90.0],
                "rule_based__realized_recovered": [True, False, False, True, False],
                "rule_based__realized_amount_recovered": [100.0, 0.0, 0.0, 150.0, 0.0],
                "no_recovery__realized_recovered": [False, False, False, False, False],
                "no_recovery__realized_amount_recovered": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        result = summarize_statistical_tests(events, n_resamples=200, seed=1)
        assert "rule_based" in result["additional_comparisons"]
        assert "no_recovery" in result["additional_comparisons"]
        for baseline in ("rule_based", "no_recovery"):
            comparison = result["additional_comparisons"][baseline]
            assert comparison["mcnemar"]["policy_b"] == baseline
            assert comparison["bootstrap_ci"]["policy_b"] == baseline
        # deployed recovered 3/5, no_recovery recovers 0/5 by construction --
        # every deployed recovery is a discordant pair in the agent's favor.
        no_recovery_mcnemar = result["additional_comparisons"]["no_recovery"]["mcnemar"]
        assert no_recovery_mcnemar["only_a_recovered"] == 3
        assert no_recovery_mcnemar["only_b_recovered"] == 0
