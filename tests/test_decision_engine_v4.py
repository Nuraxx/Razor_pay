"""
Day-10 tests: fallback-advantage arithmetic, validation-only configuration
search, the four fallback modes, edge cases (brief section 8), guardrails,
idempotency, audit logging, config/version propagation, and determinism.

Reuses the exact fixture pattern from tests/test_decision_engine.py (Day 9):
a small synthetic dataset + a freshly-fit Day-8 Model B for realistic
end-to-end tests, plus hand-crafted fake model objects for deterministic
failure-mode testing.
"""
import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from app.models import AuditLog, PolicyDecision
from data.generate_counterfactual_dataset import COUNTERFACTUAL_SEED_OFFSET, generate_counterfactual_outcomes
from data.generate_synthetic_dataset import generate_dataset
from model.candidate_preprocessing import build_candidate_level_dataset_from_tables, split_candidate_dataset
from model.latent_target_preprocessing import add_latent_targets
from model.train_latent_target_model import fit_pipeline_for_target
from policy.costs import DEFAULT_COSTS, InterventionCosts, cost_for_candidate
from policy.decision_engine import NO_ACTION, SOURCE_FALLBACK, SOURCE_MODEL, SOURCE_NO_ACTION
from policy.decision_engine_v4 import (
    DEFAULT_FALLBACK_ADVANTAGE_THRESHOLD_RS,
    DEFAULT_FALLBACK_MODE,
    DEFAULT_MARGIN_THRESHOLD_RS,
    FALLBACK_MODE_ALWAYS,
    FALLBACK_MODE_KEEP_IF_BETTER,
    FALLBACK_MODE_KEEP_UNLESS_CLEAR,
    FALLBACK_MODE_NO_ACTION,
    FALLBACK_MODES,
    POLICY_VERSION_V4,
    _rule_has_any_advantage,
    _rule_has_clear_advantage,
    build_retry_schedule_from_decision,
    decide_engine_v4,
    decide_for_failure_event_engine_v4,
    fallback_advantage,
)
from policy.guardrails import MAX_RETRY_ATTEMPTS
from policy.retry_candidates import CANDIDATE_TYPES

TEST_SEED = 42
TEST_N = 120
FAILURE_TS = datetime(2026, 2, 24, 10, 0, 0)  # all 5 candidates valid (established Day 7/9)

FAILURE_CONTEXT = {
    "day_of_month": 24, "days_to_nearest_payday_window": 6, "prior_if_failure_count": 0,
    "prior_if_self_resolved_rate": float("nan"), "tenure_days": 200, "plan_tier": "mid",
    "primary_instrument": "upi_autopay", "city_tier": "tier_1", "bank_network_conditions": "good",
    "issuing_bank_downtime_flag": False, "network_latency_bucket": "low", "is_month_end_settlement_rush": False,
}


class _PassthroughImputer:
    def transform(self, X):
        return X


class _FakeModel:
    def __init__(self, values=None, raises=None):
        self._values = values
        self._raises = raises

    def predict(self, X):
        if self._raises:
            raise self._raises
        return np.array(self._values[: len(X)]) if self._values is not None else np.zeros(len(X))


def _fake_model_dict(values=None, raises=None) -> dict:
    return {"imputer": _PassthroughImputer(), "catboost_model": _FakeModel(values=values, raises=raises)}


@pytest.fixture(scope="module")
def base_dataset():
    return generate_dataset(seed=TEST_SEED, n_subscriptions=TEST_N)


@pytest.fixture(scope="module")
def latent_splits(base_dataset):
    rng = np.random.default_rng(TEST_SEED + COUNTERFACTUAL_SEED_OFFSET)
    counterfactual = generate_counterfactual_outcomes(rng, base_dataset["subscriptions"], base_dataset["failure_events"], base_dataset["retry_candidates"])
    joined = build_candidate_level_dataset_from_tables(counterfactual, base_dataset["retry_candidates"], base_dataset["failure_events"], base_dataset["subscriptions"])
    df = add_latent_targets(joined)
    return split_candidate_dataset(df)


@pytest.fixture(scope="module")
def real_model(latent_splits) -> dict:
    train_df, val_df, _test_df = latent_splits
    fitted = fit_pipeline_for_target(train_df, val_df, "value")
    return {"imputer": fitted["imputer"], "catboost_model": fitted["catboost_model"]}


# ---------------------------------------------------------------------------
# Fallback-advantage arithmetic (brief section 3, section 9 "fallback
# advantage calculation")
# ---------------------------------------------------------------------------

def test_fallback_advantage_is_rule_minus_model():
    assert fallback_advantage(rule_candidate_value=120.0, model_best_value=100.0) == pytest.approx(20.0)
    assert fallback_advantage(rule_candidate_value=80.0, model_best_value=100.0) == pytest.approx(-20.0)
    assert fallback_advantage(rule_candidate_value=100.0, model_best_value=100.0) == pytest.approx(0.0)


def test_rule_has_any_advantage_gate():
    assert _rule_has_any_advantage(101.0, 100.0) is True
    assert _rule_has_any_advantage(100.0, 100.0) is False  # tie is not an advantage
    assert _rule_has_any_advantage(99.0, 100.0) is False


def test_rule_has_clear_advantage_gate_respects_threshold():
    # model retained when rule advantage is too small
    assert _rule_has_clear_advantage(rule_candidate_value=110.0, model_best_value=100.0, fallback_advantage_threshold=25.0) is False
    # fallback when rule advantage is sufficiently large
    assert _rule_has_clear_advantage(rule_candidate_value=130.0, model_best_value=100.0, fallback_advantage_threshold=25.0) is True
    # exactly at the threshold is NOT a clear advantage (strict >)
    assert _rule_has_clear_advantage(rule_candidate_value=125.0, model_best_value=100.0, fallback_advantage_threshold=25.0) is False


# ---------------------------------------------------------------------------
# Structural finding regression test: with a real Model-B-scores-every-
# candidate flow, Rule-Based's own candidate can never beat Model B's own
# global best -- so KEEP_IF_BETTER / KEEP_UNLESS_CLEAR must always retain
# the model in that flow. This is the documented Day-10 diagnosis finding;
# pin it so a future change can't silently reintroduce a different, buggier
# behaviour without a test noticing.
# ---------------------------------------------------------------------------

def test_evidence_based_modes_never_beat_models_own_global_best(real_model, latent_splits):
    _train, val_df, _test_df = latent_splits
    from evaluation.evaluate_decision_engine_v4 import _event_context
    from classification.rules import classify

    for event_id, group in val_df.groupby("event_id"):
        first = group.iloc[0]
        d = decide_engine_v4(
            event_id, first["subscription_id"], first["failure_timestamp"], float(first["amount"]),
            classify(None, first["error_reason"]).bucket, _event_context(first),
            margin_threshold=1e9, fallback_mode=FALLBACK_MODE_KEEP_IF_BETTER, model=real_model,
        )
        valid_scores = [s.expected_net_value for s in d.candidate_scores if s.valid]
        if not valid_scores:
            continue
        assert max(valid_scores) == pytest.approx(max(valid_scores))  # sanity
        # decision_source must be day8_model_b (never fallback) since the gate can't fire
        assert d.decision_source in (SOURCE_MODEL, SOURCE_NO_ACTION)


# ---------------------------------------------------------------------------
# Fallback modes (brief section 2B / 8 edge cases)
# ---------------------------------------------------------------------------

def test_model_slightly_better_than_rule_all_modes_keep_model_or_gate_off():
    # tiny margin, model's best beats rule's own Model-B value by a small amount
    values = [100.0, 90.0, 10.0, 5.0, 1.0]  # immediate best, plus_1_day_morning close behind
    for mode in FALLBACK_MODES:
        d = decide_engine_v4(1, "sub_1", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, margin_threshold=25.0, fallback_mode=mode, fallback_advantage_threshold=5.0, model=_fake_model_dict(values))
        if mode in (FALLBACK_MODE_KEEP_IF_BETTER, FALLBACK_MODE_KEEP_UNLESS_CLEAR):
            assert d.decision_source == SOURCE_MODEL  # rule can never have "advantage" here (see structural finding)
        assert d.margin_threshold_used == 25.0
        assert d.fallback_strategy == mode


def test_always_mode_falls_back_when_margin_ambiguous():
    values = [100.0, 99.0, 98.0, 97.0, 96.0]
    d = decide_engine_v4(2, "sub_2", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, margin_threshold=25.0, fallback_mode=FALLBACK_MODE_ALWAYS, model=_fake_model_dict(values))
    assert d.decision_source == SOURCE_FALLBACK
    assert "ambiguous_margin" in d.decision_reason


def test_no_action_mode_abstains_entirely_when_margin_ambiguous():
    values = [100.0, 99.0, 98.0, 97.0, 96.0]
    d = decide_engine_v4(3, "sub_3", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, margin_threshold=25.0, fallback_mode=FALLBACK_MODE_NO_ACTION, model=_fake_model_dict(values))
    assert d.selected_candidate_type == NO_ACTION
    assert d.decision_source == SOURCE_NO_ACTION
    assert "ambiguous_margin" in d.decision_reason


def test_keep_if_better_and_keep_unless_clear_retain_model_when_margin_ambiguous():
    values = [100.0, 99.0, 98.0, 97.0, 96.0]
    for mode in (FALLBACK_MODE_KEEP_IF_BETTER, FALLBACK_MODE_KEEP_UNLESS_CLEAR):
        d = decide_engine_v4(4, "sub_4", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, margin_threshold=25.0, fallback_mode=mode, model=_fake_model_dict(values))
        assert d.decision_source == SOURCE_MODEL
        assert d.selected_candidate_type == "immediate"  # model's own best is retained


def test_rule_clearly_better_than_model_is_structurally_impossible_via_full_flow():
    """Documents (rather than merely asserting) the Day-10 structural
    finding: since Rule-Based always picks from the same candidate pool
    Model B scores in full, "rule clearly better than model" can only be
    exercised at the pure fallback_advantage() function level (see the
    tests above), never through decide_engine_v4's real integrated flow."""
    values = [10.0, 200.0, 5.0, 1.0, 1.0]  # plus_1_day_morning (a type rule can pick) is Model B's actual best here
    d = decide_engine_v4(5, "sub_5", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, margin_threshold=1e9, fallback_mode=FALLBACK_MODE_KEEP_UNLESS_CLEAR, fallback_advantage_threshold=0.0, model=_fake_model_dict(values))
    assert d.decision_source == SOURCE_MODEL
    assert d.selected_candidate_type == "plus_1_day_morning"


# ---------------------------------------------------------------------------
# Edge cases (brief section 8): model unavailable, malformed, all invalid, no positive net value
# ---------------------------------------------------------------------------

def test_model_unavailable_falls_back_regardless_of_mode(monkeypatch):
    import policy.decision_engine_v4 as v4

    def _raise(model):
        from policy.decision_engine import ModelUnavailableError

        raise ModelUnavailableError("no artifact")

    monkeypatch.setattr(v4, "_load_model_safely", _raise)
    for mode in FALLBACK_MODES:
        d = decide_engine_v4(6, "sub_6", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, fallback_mode=mode, model=None)
        assert d.decision_source == SOURCE_FALLBACK
        assert "model_unavailable" in d.decision_reason
        assert d.predicted_recovery_value is None


def test_malformed_prediction_falls_back():
    d = decide_engine_v4(7, "sub_7", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, model=_fake_model_dict([float("nan")] * 5))
    assert d.decision_source == SOURCE_FALLBACK
    assert "invalid_model_output" in d.decision_reason


def test_all_candidates_invalid_is_no_action(monkeypatch):
    import policy.decision_engine_v4 as v4

    monkeypatch.setattr(v4, "generate_candidates", lambda ts: [])
    d = decide_engine_v4(8, "sub_8", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, model=_fake_model_dict([100.0] * 5))
    assert d.selected_candidate_type == NO_ACTION
    assert "blocked_no_valid_candidates" in d.decision_reason


def test_no_positive_net_value_is_no_action_even_when_confident():
    costs = InterventionCosts(retry_cost=1000.0)
    values = [50.0, 40.0, 30.0, 20.0, 10.0]  # huge margin, but all net-negative
    d = decide_engine_v4(9, "sub_9", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, costs=costs, margin_threshold=1.0, model=_fake_model_dict(values))
    assert d.selected_candidate_type == NO_ACTION
    assert "no_positive_net_value" in d.decision_reason


def test_no_positive_net_value_on_ambiguous_keep_model_path_is_no_action():
    costs = InterventionCosts(retry_cost=1000.0)
    values = [50.0, 49.0, 48.0, 47.0, 46.0]  # tiny margin AND all net-negative
    d = decide_engine_v4(10, "sub_10", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, costs=costs, margin_threshold=25.0, fallback_mode=FALLBACK_MODE_KEEP_UNLESS_CLEAR, model=_fake_model_dict(values))
    assert d.selected_candidate_type == NO_ACTION


# ---------------------------------------------------------------------------
# Guardrails (unchanged from Day 5-9, verified still enforced in v4)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bucket", ["hard_decline", "customer_cancelled", "unmapped"])
def test_non_retryable_classification_blocks_action(bucket):
    d = decide_engine_v4(20, "sub_20", FAILURE_TS, 1000.0, bucket, FAILURE_CONTEXT, model=_fake_model_dict([100.0] * 5))
    assert d.selected_candidate_type == NO_ACTION
    assert d.decision_source == SOURCE_NO_ACTION


def test_max_retry_attempts_blocks_action():
    d = decide_engine_v4(21, "sub_21", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, attempts_so_far=MAX_RETRY_ATTEMPTS, model=_fake_model_dict([100.0] * 5))
    assert d.selected_candidate_type == NO_ACTION
    assert "max_retry_attempts" in d.decision_reason


def test_already_decided_is_no_action():
    d = decide_engine_v4(22, "sub_22", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, already_decided=True, model=_fake_model_dict([100.0] * 5))
    assert d.selected_candidate_type == NO_ACTION
    assert "duplicate_decision_skipped" in d.decision_reason


def test_candidate_after_horizon_never_selected():
    far_ts = datetime(2026, 3, 5, 14, 0, 0)
    d = decide_engine_v4(23, "sub_23", far_ts, 1000.0, "retryable_soft", FAILURE_CONTEXT, margin_threshold=1e9, model=_fake_model_dict([1.0, 1.0, 500.0, 1.0, 500.0]))
    assert d.selected_candidate_type not in ("payday_window", "month_end_window") or d.selected_candidate_type == NO_ACTION


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic_output():
    values = [100.0, 90.0, 80.0, 70.0, 60.0]
    kwargs = dict(event_id=30, subscription_id="sub_30", failure_timestamp=FAILURE_TS, amount=1000.0, classification_bucket="retryable_soft", failure_context=FAILURE_CONTEXT, model=_fake_model_dict(values))
    d1 = decide_engine_v4(**kwargs)
    d2 = decide_engine_v4(**kwargs)
    assert d1 == d2


def test_deterministic_across_all_fallback_modes():
    values = [100.0, 99.0, 98.0, 97.0, 96.0]
    for mode in FALLBACK_MODES:
        kwargs = dict(event_id=31, subscription_id="sub_31", failure_timestamp=FAILURE_TS, amount=1000.0, classification_bucket="retryable_soft", failure_context=FAILURE_CONTEXT, margin_threshold=25.0, fallback_mode=mode, model=_fake_model_dict(values))
        d1 = decide_engine_v4(**kwargs)
        d2 = decide_engine_v4(**kwargs)
        assert d1 == d2


# ---------------------------------------------------------------------------
# Config / version propagation (brief section 10)
# ---------------------------------------------------------------------------

def test_policy_version_is_v4():
    d = decide_engine_v4(40, "sub_40", FAILURE_TS, 1000.0, "hard_decline", FAILURE_CONTEXT)
    assert d.policy_version == POLICY_VERSION_V4


def test_config_fields_propagate_on_every_decision_type():
    for kwargs_extra, model in [
        ({}, None),  # guardrail no_action path (hard_decline)
        ({}, _fake_model_dict([100.0] * 5)),  # model-direct path
    ]:
        bucket = "hard_decline" if model is None else "retryable_soft"
        d = decide_engine_v4(41, "sub_41", FAILURE_TS, 1000.0, bucket, FAILURE_CONTEXT, margin_threshold=42.0, fallback_advantage_threshold=7.0, fallback_mode=FALLBACK_MODE_ALWAYS, model=model, **kwargs_extra)
        assert d.margin_threshold_used == 42.0
        assert d.fallback_advantage_threshold == 7.0
        assert d.fallback_strategy == FALLBACK_MODE_ALWAYS


def test_default_config_values_are_the_validation_selected_ones():
    # Pins the frozen defaults so an accidental edit is caught by CI, not
    # silently shipped. ECONOMIC CORRECTION (final pre-submission audit):
    # these changed from (margin=5.0, ALWAYS) after re-running the
    # validation-only search with REALIZED Rs recovered (not latent value
    # alone) as the primary selection metric -- see
    # policy/decision_engine_v4.py's ECONOMIC-CORRECTION FINDING and
    # evaluation/reports/decision_engine_v4_evaluation.json.
    assert DEFAULT_MARGIN_THRESHOLD_RS == 0.0
    assert DEFAULT_FALLBACK_MODE == FALLBACK_MODE_KEEP_UNLESS_CLEAR
    assert DEFAULT_FALLBACK_ADVANTAGE_THRESHOLD_RS == 0.0


def test_default_config_never_blindly_swaps_away_from_models_own_best_pick():
    # Regression test for the exact economic bug this correction fixed:
    # ALWAYS_FALLBACK_WHEN_BELOW_MARGIN would unconditionally discard Model
    # B's own top pick whenever its top-2 margin was small, without ever
    # checking whether the substitute was actually any good -- this cost
    # Rs1280.95 on the held-out TEST set alone. The corrected default must
    # never select a candidate the model itself did not choose as best.
    model = _fake_model_dict([100.0, 99.0, 98.0, 97.0, 96.0])  # deliberately tiny margin
    decision = decide_engine_v4(
        70001, "sub_v4_no_blind_swap", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT,
        margin_threshold=DEFAULT_MARGIN_THRESHOLD_RS, fallback_mode=DEFAULT_FALLBACK_MODE,
        fallback_advantage_threshold=DEFAULT_FALLBACK_ADVANTAGE_THRESHOLD_RS, model=model,
    )
    assert decision.decision_source == SOURCE_MODEL
    assert decision.selected_candidate_type == CANDIDATE_TYPES[0]  # the highest-scored candidate in the fake model


# ---------------------------------------------------------------------------
# MULTI-ATTEMPT PERSISTENCE (final pre-submission audit): build_retry_schedule_from_decision
# ---------------------------------------------------------------------------

def test_build_retry_schedule_ranks_remaining_by_net_value():
    # immediate=100, plus_1_day_morning=90, payday_window=80, plus_3_days=70,
    # month_end_window=60 -- CANDIDATE_TYPES order, so net-value ranking and
    # list order coincide here (deliberately, for a readable assertion).
    values = [100.0, 90.0, 80.0, 70.0, 60.0]
    d = decide_engine_v4(60001, "sub_v4_schedule_1", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, margin_threshold=1e9, model=_fake_model_dict(values))
    assert d.selected_candidate_type == "immediate"
    types, datetimes = build_retry_schedule_from_decision(d)
    assert types == ["immediate", "plus_1_day_morning", "payday_window"]  # capped at MAX_RETRY_ATTEMPTS=3
    assert len(datetimes) == len(types) == MAX_RETRY_ATTEMPTS
    assert types[0] == d.selected_candidate_type
    assert datetimes[0] == d.selected_candidate_datetime


def test_build_retry_schedule_returns_empty_for_no_action():
    d = decide_engine_v4(60002, "sub_v4_schedule_2", FAILURE_TS, 1000.0, "hard_decline", FAILURE_CONTEXT, model=_fake_model_dict([100.0] * 5))
    assert d.selected_candidate_type == NO_ACTION
    types, datetimes = build_retry_schedule_from_decision(d)
    assert types == []
    assert datetimes == []


def test_build_retry_schedule_never_exceeds_max_retry_attempts():
    values = [50.0, 49.0, 48.0, 47.0, 46.0]
    d = decide_engine_v4(60003, "sub_v4_schedule_3", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, margin_threshold=1e9, model=_fake_model_dict(values))
    types, datetimes = build_retry_schedule_from_decision(d)
    assert len(types) <= MAX_RETRY_ATTEMPTS
    assert len(set(types)) == len(types)  # no candidate type repeated


def test_build_retry_schedule_first_slot_matches_selected_candidate_under_fallback():
    # Ambiguous margin, ALWAYS mode -- forces a rule_based_fallback decision
    # (a real historical failure mode, see the ECONOMIC-CORRECTION FINDING);
    # slot 1 of the schedule must still be exactly whatever was actually
    # selected, regardless of which tier decided it.
    values = [100.0, 99.0, 98.0, 97.0, 96.0]
    d = decide_engine_v4(60004, "sub_v4_schedule_4", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, margin_threshold=25.0, fallback_mode=FALLBACK_MODE_ALWAYS, model=_fake_model_dict(values))
    assert d.decision_source == SOURCE_FALLBACK
    types, datetimes = build_retry_schedule_from_decision(d)
    assert types[0] == d.selected_candidate_type
    assert datetimes[0] == d.selected_candidate_datetime


def test_build_retry_schedule_degrades_to_single_attempt_when_model_unavailable(monkeypatch):
    # When Model B itself is unavailable, decision.candidate_scores holds no
    # scored candidates to rank (see _rule_based_only) -- the schedule must
    # safely degrade to just the one fallback-selected attempt, never crash.
    import policy.decision_engine_v4 as v4

    def _raise(model):
        from policy.decision_engine import ModelUnavailableError

        raise ModelUnavailableError("no artifact")

    monkeypatch.setattr(v4, "_load_model_safely", _raise)
    d = decide_engine_v4(60005, "sub_v4_schedule_5", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, model=None)
    assert d.decision_source == SOURCE_FALLBACK
    types, datetimes = build_retry_schedule_from_decision(d)
    assert types == [d.selected_candidate_type]
    assert datetimes == [d.selected_candidate_datetime]


# ---------------------------------------------------------------------------
# DB-backed: audit logging + idempotency
# ---------------------------------------------------------------------------

def test_decide_for_failure_event_engine_v4_creates_decision_and_audit_rows(test_db_session):
    db = test_db_session()
    row, created = decide_for_failure_event_engine_v4(
        db, event_id=50001, subscription_id="sub_V4AuditTest", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", failure_context=FAILURE_CONTEXT,
        model=_fake_model_dict([100.0, 90.0, 80.0, 70.0, 60.0]),
    )
    assert created is True
    assert isinstance(row, PolicyDecision)
    assert row.policy_version == POLICY_VERSION_V4
    assert row.decision_source == SOURCE_MODEL
    assert row.margin_threshold_used == DEFAULT_MARGIN_THRESHOLD_RS
    assert row.fallback_strategy == DEFAULT_FALLBACK_MODE

    audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 50001).all()
    assert len(audit_rows) == 1
    assert audit_rows[0].actor == "policy"
    assert "fallback_strategy=" in audit_rows[0].reason
    assert "margin_threshold=" in audit_rows[0].reason
    db.close()


def test_decide_for_failure_event_engine_v4_persists_retry_schedule(test_db_session):
    # MULTI-ATTEMPT PERSISTENCE (final pre-submission audit): the DB-aware
    # wrapper must populate retry_schedule_json/retry_schedule_datetimes_json
    # from build_retry_schedule_from_decision -- purely additive, so
    # selected_candidate_type/decision_source above are unaffected (already
    # asserted by the sibling test above using the same fake model values).
    import json

    db = test_db_session()
    row, created = decide_for_failure_event_engine_v4(
        db, event_id=50010, subscription_id="sub_V4RetrySchedule", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", failure_context=FAILURE_CONTEXT,
        model=_fake_model_dict([100.0, 90.0, 80.0, 70.0, 60.0]),
    )
    assert created is True
    schedule_types = json.loads(row.retry_schedule_json)
    schedule_datetimes = json.loads(row.retry_schedule_datetimes_json)
    assert schedule_types[0] == row.selected_candidate_type
    assert len(schedule_types) == len(schedule_datetimes) == MAX_RETRY_ATTEMPTS
    assert row.retry_schedule_next_index == 1
    db.close()


def test_decide_for_failure_event_engine_v4_no_action_leaves_retry_schedule_null(test_db_session):
    db = test_db_session()
    row, created = decide_for_failure_event_engine_v4(
        db, event_id=50011, subscription_id="sub_V4RetryScheduleNoAction", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="hard_decline", failure_context=FAILURE_CONTEXT,
        model=_fake_model_dict([100.0] * 5),
    )
    assert created is True
    assert row.selected_candidate_type == NO_ACTION
    assert row.retry_schedule_json is None
    assert row.retry_schedule_datetimes_json is None
    db.close()


def test_decide_for_failure_event_engine_v4_is_idempotent(test_db_session):
    db = test_db_session()
    first, first_created = decide_for_failure_event_engine_v4(
        db, event_id=50002, subscription_id="sub_V4Idempotent", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", failure_context=FAILURE_CONTEXT,
        model=_fake_model_dict([100.0, 90.0, 80.0, 70.0, 60.0]),
    )
    second, second_created = decide_for_failure_event_engine_v4(
        db, event_id=50002, subscription_id="sub_V4Idempotent", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", failure_context=FAILURE_CONTEXT,
        model=_fake_model_dict([100.0, 90.0, 80.0, 70.0, 60.0]),
    )
    assert first_created is True
    assert second_created is False
    assert first.id == second.id
    assert db.query(PolicyDecision).filter(PolicyDecision.event_id == 50002).count() == 1

    audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 50002).order_by(AuditLog.id).all()
    assert len(audit_rows) == 2
    assert audit_rows[1].action == "policy_decision_skipped_duplicate"
    db.close()


def test_max_retry_attempts_enforced_across_v4_db_calls(test_db_session):
    db = test_db_session()
    for i in range(MAX_RETRY_ATTEMPTS):
        decide_for_failure_event_engine_v4(
            db, event_id=500 + i, subscription_id="sub_v4_maxattempts",
            failure_timestamp=FAILURE_TS + timedelta(days=i * 20), amount=1000.0,
            classification_bucket="retryable_soft", failure_context=FAILURE_CONTEXT,
            model=_fake_model_dict([100.0, 90.0, 80.0, 70.0, 60.0]),
        )
    blocked_row, created = decide_for_failure_event_engine_v4(
        db, event_id=600, subscription_id="sub_v4_maxattempts", failure_timestamp=FAILURE_TS + timedelta(days=100),
        amount=1000.0, classification_bucket="retryable_soft", failure_context=FAILURE_CONTEXT,
        model=_fake_model_dict([100.0, 90.0, 80.0, 70.0, 60.0]),
    )
    assert created is True
    assert blocked_row.selected_candidate_type == NO_ACTION
    assert "max_retry_attempts" in blocked_row.decision_reason
    db.close()


# ---------------------------------------------------------------------------
# Validation-only configuration search
# ---------------------------------------------------------------------------

def test_configuration_search_uses_only_validation_data(latent_splits, real_model):
    from evaluation.evaluate_decision_engine_v4 import MARGIN_THRESHOLD_CANDIDATES, select_day10_configuration_on_validation

    _train, val_df, test_df = latent_splits
    chosen, results = select_day10_configuration_on_validation(val_df, real_model)
    assert chosen["margin_threshold"] in MARGIN_THRESHOLD_CANDIDATES
    assert chosen["fallback_mode"] in FALLBACK_MODES
    assert len(results) > 0
    # structural check: the function signature only accepts val_df, never test_df
    import inspect

    params = list(inspect.signature(select_day10_configuration_on_validation).parameters)
    assert "test_df" not in params and "test" not in params


def test_configuration_search_selects_best_by_total_realized_value(latent_splits, real_model):
    # ECONOMIC CORRECTION: the primary selection key is now REALIZED Rs
    # recovered on validation (what the search's docstring calls out as the
    # fix), not total_latent_value_selected_rs alone -- see
    # policy/decision_engine_v4.py's ECONOMIC-CORRECTION FINDING. Both
    # metrics are still reported for every configuration; only realized is
    # asserted to be optimal here, since latent and realized can legitimately
    # disagree (that disagreement is exactly what this correction fixed).
    from evaluation.evaluate_decision_engine_v4 import select_day10_configuration_on_validation

    _train, val_df, _test_df = latent_splits
    chosen, results = select_day10_configuration_on_validation(val_df, real_model)
    chosen_key = next(k for k, r in results.items() if r["margin_threshold"] == chosen["margin_threshold"] and r["fallback_mode"] == chosen["fallback_mode"] and r["fallback_advantage_threshold"] == chosen["fallback_advantage_threshold"])
    chosen_value = results[chosen_key]["total_realized_value_selected_rs"]
    assert all(chosen_value >= r["total_realized_value_selected_rs"] for r in results.values())
    assert all("total_realized_value_selected_rs" in r and "total_latent_value_selected_rs" in r for r in results.values())


# ---------------------------------------------------------------------------
# APPLES-TO-APPLES FIX (final pre-submission audit, third pass): Oracle must
# also get multi-attempt scoring once Fixed Retry and the deployed policy do.
# ---------------------------------------------------------------------------

def test_oracle_realized_value_is_never_below_the_deployed_policy(latent_splits, real_model):
    # Regression test for the exact bug this fix corrected: with Oracle
    # scored as a single-attempt pick while Fixed Retry and the deployed
    # policy both get up to 3 scheduled attempts, a multi-attempt policy
    # could beat "Oracle" purely from having more chances at a stochastic
    # outcome -- not a real upper-bound violation, just a broken comparison.
    # Oracle must be AT LEAST as good as every other policy on realized Rs,
    # by construction, once it is scored with the SAME multi-attempt
    # machinery (see evaluate_events_v4's oracle_schedule).
    from evaluation.evaluate_decision_engine_v4 import evaluate_events_v4, select_day10_configuration_on_validation

    _train, val_df, test_df = latent_splits
    chosen_config, _results = select_day10_configuration_on_validation(val_df, real_model)
    events = evaluate_events_v4(test_df, real_model, chosen_config)

    oracle_total = events["oracle_policy__realized_amount_recovered"].sum()
    for policy_name in ("fixed_retry", "rule_based", "day8_model_b_alone", "day9_original_fallback", "day10_improved_fallback"):
        assert oracle_total >= events[f"{policy_name}__realized_amount_recovered"].sum(), (
            f"oracle_policy scored below {policy_name} -- Oracle is no longer a valid upper bound"
        )


def test_oracle_schedule_never_exceeds_max_retry_attempts(latent_splits, real_model):
    from evaluation.evaluate_decision_engine_v4 import evaluate_events_v4, select_day10_configuration_on_validation
    from policy.guardrails import MAX_RETRY_ATTEMPTS

    _train, val_df, test_df = latent_splits
    chosen_config, _results = select_day10_configuration_on_validation(val_df, real_model)
    events = evaluate_events_v4(test_df, real_model, chosen_config)
    assert (events["oracle_policy__n_attempts"] <= MAX_RETRY_ATTEMPTS).all()
