"""
FIX pass tests: policy/economics.py -- separates recovered GMV,
intervention cost, and Razorpay's fee take into distinct, never-blended
fields (per the original specification's "report both raw merchant GMV and
Razorpay's own fee take ... as two separate numbers").
"""
from __future__ import annotations

import pytest

from policy.economics import BASE_FEE_RATE, EFFECTIVE_FEE_RATE, GST_RATE, RecoveryEconomics, compute_recovery_economics


class TestFeeCalculation:
    def test_fee_rate_matches_specification_assumption(self):
        # Specification: "roughly 2% + 18% GST" -- base fee 2%, GST 18% ON the fee.
        assert BASE_FEE_RATE == pytest.approx(0.02)
        assert GST_RATE == pytest.approx(0.18)
        assert EFFECTIVE_FEE_RATE == pytest.approx(0.02 * 1.18)
        assert EFFECTIVE_FEE_RATE == pytest.approx(0.0236)

    def test_fee_take_is_gross_including_gst(self):
        econ = compute_recovery_economics(recovered_gmv=10000.0, intervention_cost=0.0)
        base_fee = 10000.0 * BASE_FEE_RATE
        gst_on_fee = base_fee * GST_RATE
        assert econ.razorpay_fee_take == pytest.approx(base_fee + gst_on_fee)
        assert econ.razorpay_fee_take == pytest.approx(236.0)

    def test_fee_scales_linearly_with_gmv(self):
        econ_1x = compute_recovery_economics(recovered_gmv=1000.0, intervention_cost=0.0)
        econ_2x = compute_recovery_economics(recovered_gmv=2000.0, intervention_cost=0.0)
        assert econ_2x.razorpay_fee_take == pytest.approx(econ_1x.razorpay_fee_take * 2)


class TestFieldSeparation:
    def test_recovered_gmv_never_includes_fee_or_cost(self):
        econ = compute_recovery_economics(recovered_gmv=5000.0, intervention_cost=250.0)
        # recovered_gmv is passed through unchanged (rounded only) -- never
        # reduced by fee or intervention cost.
        assert econ.recovered_gmv == pytest.approx(5000.0)

    def test_intervention_cost_kept_separate_from_fee(self):
        econ = compute_recovery_economics(recovered_gmv=5000.0, intervention_cost=250.0)
        assert econ.intervention_cost == pytest.approx(250.0)
        # intervention_cost must not itself be scaled by the fee rate.
        assert econ.intervention_cost != pytest.approx(250.0 * (1 + EFFECTIVE_FEE_RATE))

    def test_four_fields_are_all_independently_present(self):
        econ = compute_recovery_economics(recovered_gmv=1000.0, intervention_cost=50.0)
        d = econ.to_dict()
        assert set(d.keys()) == {"recovered_gmv", "intervention_cost", "razorpay_fee_take", "net_recovery_value"}


class TestNetValueArithmetic:
    def test_net_recovery_value_subtracts_both_cost_and_fee(self):
        econ = compute_recovery_economics(recovered_gmv=1000.0, intervention_cost=50.0)
        expected_fee = round(1000.0 * EFFECTIVE_FEE_RATE, 2)
        expected_net = round(1000.0 - 50.0 - expected_fee, 2)
        assert econ.net_recovery_value == pytest.approx(expected_net)

    def test_net_can_go_negative_when_costs_exceed_gmv(self):
        econ = compute_recovery_economics(recovered_gmv=10.0, intervention_cost=100.0)
        assert econ.net_recovery_value < 0

    def test_zero_recovery(self):
        econ = compute_recovery_economics(recovered_gmv=0.0, intervention_cost=0.0)
        assert econ.recovered_gmv == 0.0
        assert econ.razorpay_fee_take == 0.0
        assert econ.net_recovery_value == 0.0

    def test_zero_recovery_with_nonzero_intervention_cost(self):
        # A real scenario: retries were attempted (cost incurred) but nothing recovered.
        econ = compute_recovery_economics(recovered_gmv=0.0, intervention_cost=25.0)
        assert econ.razorpay_fee_take == 0.0
        assert econ.net_recovery_value == pytest.approx(-25.0)

    def test_large_values_no_overflow_or_precision_blowup(self):
        econ = compute_recovery_economics(recovered_gmv=50_000_000.0, intervention_cost=100_000.0)
        assert econ.recovered_gmv == pytest.approx(50_000_000.0)
        assert econ.razorpay_fee_take == pytest.approx(50_000_000.0 * EFFECTIVE_FEE_RATE)
        assert econ.net_recovery_value == pytest.approx(50_000_000.0 - 100_000.0 - 50_000_000.0 * EFFECTIVE_FEE_RATE)


class TestRoundingBehavior:
    def test_all_fields_rounded_to_two_decimal_places(self):
        econ = compute_recovery_economics(recovered_gmv=1000.333333, intervention_cost=5.0001)
        for value in (econ.recovered_gmv, econ.intervention_cost, econ.razorpay_fee_take, econ.net_recovery_value):
            assert round(value, 2) == value

    def test_result_type_is_recovery_economics(self):
        econ = compute_recovery_economics(recovered_gmv=100.0, intervention_cost=5.0)
        assert isinstance(econ, RecoveryEconomics)


class TestEvaluationIntegration:
    """Exercises evaluation/evaluate_decision_engine_v4.py::summarize_economics
    against a small, hand-built `events` DataFrame shaped exactly like the
    real one -- verifies intervention cost is summed only over real
    (non-NO_ACTION) selections and GMV is read from the EXISTING
    realized_summary, never recomputed."""

    def test_summarize_economics_reads_existing_realized_totals(self):
        import pandas as pd

        from evaluation.evaluate_decision_engine_v4 import POLICY_NAMES, summarize_economics
        from policy.decision_engine import NO_ACTION

        events = pd.DataFrame(
            {f"{name}__selected_candidate_type": (["plus_1_day_morning"] * 2 + [NO_ACTION]) for name in POLICY_NAMES}
        )
        # BASELINE-FIDELITY FIX: fixed_retry's intervention_cost is now
        # driven by n_attempts (1 attempt per recovered event here, 0 for
        # the NO_ACTION event), and rule_based's by n_contacts (0 here, so
        # its cost matches the other policies' plain per-action retry_cost).
        events["fixed_retry__n_attempts"] = [1, 1, 0]
        # MULTI-ATTEMPT PERSISTENCE: day10_improved_fallback is costed the
        # same n_attempts-based way as fixed_retry -- see summarize_economics.
        events["day10_improved_fallback__n_attempts"] = [1, 1, 0]
        events["rule_based__n_contacts"] = [0, 0, 0]
        realized_summary = {name: {"total_recovered_rs": 100.0, "recovery_rate": 2 / 3} for name in POLICY_NAMES}

        economics = summarize_economics(events, realized_summary)
        for name in POLICY_NAMES:
            assert economics[name]["recovered_gmv"] == pytest.approx(100.0)
            # 2 real actions * Rs5 retry_cost (DEFAULT_COSTS), the third is NO_ACTION -- no cost.
            assert economics[name]["intervention_cost"] == pytest.approx(10.0)
