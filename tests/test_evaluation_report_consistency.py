"""
DOCUMENTATION / EVALUATION RECONCILIATION regression test.

Guards against the exact drift this project once had: `evaluate_decision_engine_v4.py`'s
Phase 2 TEST-set evaluation must always be computed against the SAME
`deployed_config` that `policy/decision_engine_v4.py`'s own frozen defaults
specify -- the config the live orchestrator (`recovery/orchestrator.py`,
via `decide_for_failure_event_engine_v4`'s default parameters) actually
uses -- never whatever a fresh re-run of `select_validation_configuration()`'s
validation-only search happens to prefer on the current dataset.

History (see evaluate_decision_engine_v4.py::main()'s `deployed_config` /
`validation_search_argmax` split and README §16d): after the synthetic
dataset was rescaled from 200 to 1,500 subscriptions, that search's raw
argmax silently diverged from the frozen, safety-guaranteed default
(FALLBACK_MODE_KEEP_UNLESS_CLEAR at margin_threshold=0.0, which
policy/decision_engine_v4.py's own STRUCTURAL FINDING proves can never
select a worse candidate than Model B's own best pick), and the persisted
report reflected the wrong one -- a real gap between what README's prose
described and what the documented reproduction command actually produced.

These tests read the ALREADY-GENERATED
evaluation/reports/decision_engine_v4_evaluation.json artifact (never
generate it themselves -- a full run takes several minutes) and skip
gracefully if it doesn't exist yet in this environment, matching this
project's existing convention for artifact-dependent tests
(evaluation/reports/ is gitignored, present only after a local run of
`python -m evaluation.evaluate_decision_engine_v4`).

Deliberately does NOT assert specific ₹ amounts, recovery rates, or p-values
-- those are legitimate to change on a genuine data/model regeneration.
What must never change is the INTERNAL CONSISTENCY relationships below.
"""
from __future__ import annotations

import json

import pytest

from evaluation.evaluate_decision_engine_v4 import REPORTS_DIR
from policy.decision_engine_v4 import (
    DEFAULT_FALLBACK_ADVANTAGE_THRESHOLD_RS,
    DEFAULT_FALLBACK_MODE,
    DEFAULT_MARGIN_THRESHOLD_RS,
    FALLBACK_MODE_KEEP_IF_BETTER,
    FALLBACK_MODE_KEEP_UNLESS_CLEAR,
)

REPORT_PATH = REPORTS_DIR / "decision_engine_v4_evaluation.json"


@pytest.fixture
def report() -> dict:
    if not REPORT_PATH.exists():
        pytest.skip(f"{REPORT_PATH} not generated in this environment -- run `python -m evaluation.evaluate_decision_engine_v4` first")
    with open(REPORT_PATH) as f:
        return json.load(f)


class TestDeployedConfigConsistency:
    def test_deployed_config_matches_the_frozen_source_default(self, report):
        # The whole point of the reconciliation fix: the config the report
        # was actually EVALUATED with must be byte-identical to the frozen
        # default policy/decision_engine_v4.py documents and the live
        # orchestrator uses -- never a silently different, freshly re-derived
        # validation-search argmax.
        deployed = report["validation_configuration_selection"]["deployed_config"]
        assert deployed["margin_threshold"] == DEFAULT_MARGIN_THRESHOLD_RS
        assert deployed["fallback_mode"] == DEFAULT_FALLBACK_MODE
        assert deployed["fallback_advantage_threshold"] == DEFAULT_FALLBACK_ADVANTAGE_THRESHOLD_RS

    def test_operational_config_matches_deployed_config(self, report):
        # operational.config is a second, independently-serialized copy of
        # the same config (see summarize_operational) -- must never drift
        # from validation_configuration_selection.deployed_config.
        assert report["operational"]["config"] == report["validation_configuration_selection"]["deployed_config"]

    def test_stage_decomposition_uses_the_same_deployed_config(self, report):
        assert report["stage_decomposition"]["deployed_config"] == report["validation_configuration_selection"]["deployed_config"]


class TestOperationalCountsAreInternallyConsistent:
    def test_decision_source_counts_sum_to_n_events(self, report):
        op = report["operational"]
        assert op["model_b_direct_selections"] + op["fallback_count"] + op["no_action_count"] == op["n_events"]

    def test_fallback_percentage_matches_fallback_count(self, report):
        op = report["operational"]
        expected = round(100 * op["fallback_count"] / op["n_events"], 2) if op["n_events"] else 0.0
        assert op["fallback_percentage"] == pytest.approx(expected, abs=0.01)

    def test_statistical_population_matches_operational_population(self, report):
        # statistical_tests and operational must describe the SAME held-out
        # events -- a divergence here would mean the p-value/CI reported
        # doesn't actually correspond to the deployed-config decisions above it.
        assert report["statistical_tests"]["population"]["n_events"] == report["operational"]["n_events"]

    def test_safe_fallback_mode_structurally_never_falls_back(self, report):
        # STRUCTURAL FINDING (policy/decision_engine_v4.py): at
        # margin_threshold=0.0, FALLBACK_MODE_KEEP_IF_BETTER and
        # FALLBACK_MODE_KEEP_UNLESS_CLEAR are mathematically guaranteed to
        # never select a candidate other than Model B's own best pick --
        # Rule-Based's candidate is always one of the same set Model B
        # already scored, so it can never have a strictly higher net value
        # than Model B's own top pick. This is a property of the mechanism,
        # not of any particular dataset -- if the deployed config is one of
        # these two safe modes at margin=0, fallback_count MUST be exactly 0
        # regardless of which model/dataset produced this report.
        deployed = report["operational"]["config"]
        if deployed["margin_threshold"] == 0.0 and deployed["fallback_mode"] in (FALLBACK_MODE_KEEP_IF_BETTER, FALLBACK_MODE_KEEP_UNLESS_CLEAR):
            assert report["operational"]["fallback_count"] == 0
            assert report["operational"]["model_b_direct_selections"] == report["operational"]["n_events"]


class TestValidationConfigurationReconciliationConsistency:
    """VALIDATION-CONFIGURATION RECONCILIATION: guards the bootstrap-CI
    robustness gate (evaluate_decision_engine_v4.py::decide_deployed_config)
    that decides whether a fresh validation_search_argmax may override
    policy/decision_engine_v4.py's structural_safety_baseline -- reading only
    the already-persisted artifact, matching this file's existing convention.
    """

    def test_deployed_config_is_either_the_argmax_or_the_baseline(self, report):
        sel = report["validation_configuration_selection"]
        assert sel["deployed_config"] in (sel["validation_search_argmax"], sel["structural_safety_baseline"])

    def test_robustness_ci_present_iff_argmax_differs_from_baseline(self, report):
        sel = report["validation_configuration_selection"]
        if sel["validation_search_argmax"] == sel["structural_safety_baseline"]:
            assert sel["robustness_ci_validation_only"] is None
            assert sel["decision_reason"] == "validation_search_argmax_matches_structural_safety_baseline"
        else:
            assert sel["robustness_ci_validation_only"] is not None

    def test_decision_reason_matches_which_config_was_actually_deployed(self, report):
        # The reason string and the actual deployed_config must never
        # disagree -- this is the exact "README says one configuration while
        # main() generates another" class of bug this whole reconciliation
        # pass exists to prevent, applied one level deeper (to the SELECTION
        # decision itself, not just its downstream numbers).
        sel = report["validation_configuration_selection"]
        if sel["decision_reason"] == "validation_search_argmax_not_bootstrap_robust_retaining_structural_safety_baseline":
            assert sel["deployed_config"] == sel["structural_safety_baseline"]
        elif sel["decision_reason"] == "validation_search_argmax_is_bootstrap_robust_improvement_over_structural_safety_baseline":
            assert sel["deployed_config"] == sel["validation_search_argmax"]
        elif sel["decision_reason"] == "validation_search_argmax_matches_structural_safety_baseline":
            assert sel["deployed_config"] == sel["validation_search_argmax"] == sel["structural_safety_baseline"]

    def test_robustness_ci_not_robust_implies_lower_bound_not_positive(self, report):
        sel = report["validation_configuration_selection"]
        rc = sel["robustness_ci_validation_only"]
        if rc is not None and sel["decision_reason"] == "validation_search_argmax_not_bootstrap_robust_retaining_structural_safety_baseline":
            assert rc["lower_bound"] <= 0

    def test_structural_safety_baseline_matches_frozen_source_defaults(self, report):
        baseline = report["validation_configuration_selection"]["structural_safety_baseline"]
        assert baseline["margin_threshold"] == DEFAULT_MARGIN_THRESHOLD_RS
        assert baseline["fallback_mode"] == DEFAULT_FALLBACK_MODE
        assert baseline["fallback_advantage_threshold"] == DEFAULT_FALLBACK_ADVANTAGE_THRESHOLD_RS
