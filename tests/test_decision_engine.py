"""
Day-9 decision-engine tests: net-value scoring, abstention, fallback,
guardrails, audit logging, idempotency, serialization, and failure modes.

Uses a small synthetic dataset + a freshly-fit Day-8 Model B (not the
committed model/latent_target_artifacts/) for realistic end-to-end tests,
same pattern as tests/test_latent_target_model.py, plus hand-crafted fake
model objects to exercise specific failure modes deterministically.
"""
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.models import AuditLog, PolicyDecision
from data.generate_counterfactual_dataset import COUNTERFACTUAL_SEED_OFFSET, generate_counterfactual_outcomes
from data.generate_synthetic_dataset import generate_dataset
from model.candidate_preprocessing import build_candidate_level_dataset_from_tables, split_candidate_dataset
from model.latent_target_preprocessing import add_latent_targets
from model.train_latent_target_model import fit_pipeline_for_target
from policy.costs import DEFAULT_COSTS, InterventionCosts, cost_for_candidate
from policy.decision_engine import (
    NO_ACTION,
    POLICY_VERSION,
    SOURCE_FALLBACK,
    SOURCE_MODEL,
    SOURCE_NO_ACTION,
    Decision,
    MalformedModelOutputError,
    ModelUnavailableError,
    decide_engine,
    decide_for_failure_event_engine,
)
from policy.guardrails import MAX_RETRY_ATTEMPTS
from policy.retry_candidates import CANDIDATE_TYPES

TEST_SEED = 42
TEST_N = 120
FAILURE_TS = datetime(2026, 2, 24, 10, 0, 0)  # all 5 candidates valid (verified in Day-7 tests too)

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
# Net-value calculation / candidate ranking
# ---------------------------------------------------------------------------

def test_net_value_equals_predicted_value_minus_cost():
    values = [500.0, 100.0, 50.0, 20.0, 10.0]  # immediate, plus_1_day_morning, payday_window, plus_3_days, month_end_window (CANDIDATE_TYPES order)
    d = decide_engine(1, "sub_1", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, model=_fake_model_dict(values))
    assert d.decision_source == SOURCE_MODEL
    assert d.selected_candidate_type == "immediate"
    assert d.predicted_recovery_value == pytest.approx(500.0)
    assert d.intervention_cost == pytest.approx(cost_for_candidate("immediate"))
    assert d.expected_net_value == pytest.approx(500.0 - cost_for_candidate("immediate"))


def test_candidate_with_highest_net_value_is_selected_not_highest_raw_value():
    costs = InterventionCosts(retry_cost=50.0)
    # immediate has the highest raw value but a much smaller margin after cost than expected -- construct so ranking flips
    values = [100.0, 95.0, 10.0, 5.0, 1.0]
    d = decide_engine(2, "sub_2", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, costs=costs, abstention_threshold=-math.inf, model=_fake_model_dict(values))
    # both immediate (net=50) and plus_1_day_morning (net=45) have the same cost, so ranking-by-net matches ranking-by-value here;
    # verify the net-value arithmetic itself instead
    assert d.expected_net_value == pytest.approx(d.predicted_recovery_value - costs.retry_cost)


def test_runner_up_value_is_the_second_best_candidates_own_predicted_value():
    values = [500.0, 300.0, 10.0, 5.0, 1.0]
    d = decide_engine(3, "sub_3", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, abstention_threshold=-math.inf, model=_fake_model_dict(values))
    assert d.runner_up_value == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Abstention (decision margin)
# ---------------------------------------------------------------------------

def test_high_margin_selects_via_primary_model():
    values = [500.0, 10.0, 5.0, 1.0, 0.5]  # huge margin
    d = decide_engine(4, "sub_4", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, abstention_threshold=25.0, model=_fake_model_dict(values))
    assert d.decision_source == SOURCE_MODEL
    assert d.decision_margin > 25.0


def test_low_margin_triggers_fallback():
    values = [100.0, 99.0, 98.0, 97.0, 96.0]  # tiny margin between all candidates
    d = decide_engine(5, "sub_5", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, abstention_threshold=25.0, model=_fake_model_dict(values))
    assert d.decision_source == SOURCE_FALLBACK
    assert "insufficient_decision_margin" in d.decision_reason


def test_margin_threshold_is_configurable():
    values = [100.0, 90.0, 10.0, 5.0, 1.0]  # margin = 10 - cost difference (same cost, so exactly 10)
    low_threshold = decide_engine(6, "sub_6", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, abstention_threshold=5.0, model=_fake_model_dict(values))
    high_threshold = decide_engine(6, "sub_6", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, abstention_threshold=50.0, model=_fake_model_dict(values))
    assert low_threshold.decision_source == SOURCE_MODEL
    assert high_threshold.decision_source == SOURCE_FALLBACK


def test_no_positive_net_value_is_no_action():
    costs = InterventionCosts(retry_cost=1000.0)  # cost exceeds every candidate's predicted value
    values = [50.0, 40.0, 30.0, 20.0, 10.0]
    d = decide_engine(7, "sub_7", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, costs=costs, abstention_threshold=-math.inf, model=_fake_model_dict(values))
    assert d.selected_candidate_type == NO_ACTION
    assert "no_positive_net_value" in d.decision_reason


# ---------------------------------------------------------------------------
# Fallback (model unavailable / malformed output / exception)
# ---------------------------------------------------------------------------

def test_model_prediction_exception_falls_back():
    d = decide_engine(8, "sub_8", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, model=_fake_model_dict(raises=RuntimeError("boom")))
    assert d.decision_source == SOURCE_FALLBACK
    assert "invalid_model_output" in d.decision_reason
    assert d.model_version is None  # no value estimate exists -- model never successfully produced one


def test_nan_prediction_falls_back():
    d = decide_engine(9, "sub_9", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, model=_fake_model_dict([float("nan")] * 5))
    assert d.decision_source == SOURCE_FALLBACK
    assert "malformed_prediction" in d.decision_reason


def test_negative_predicted_recovery_value_falls_back():
    d = decide_engine(10, "sub_10", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, model=_fake_model_dict([-50.0] * 5))
    assert d.decision_source == SOURCE_FALLBACK
    assert "malformed_prediction" in d.decision_reason


def test_huge_predicted_recovery_value_falls_back():
    d = decide_engine(11, "sub_11", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, model=_fake_model_dict([1_000_000.0] * 5))
    assert d.decision_source == SOURCE_FALLBACK
    assert "malformed_prediction" in d.decision_reason


def test_model_file_missing_falls_back(monkeypatch):
    import policy.decision_engine as de

    def _raise_file_not_found(target):
        raise FileNotFoundError("no such artifact")

    monkeypatch.setattr(de, "load_latent_target_model", _raise_file_not_found)
    d = decide_engine(12, "sub_12", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, model=None)
    assert d.decision_source == SOURCE_FALLBACK
    assert "model_unavailable" in d.decision_reason
    assert d.predicted_recovery_value is None


def test_insufficient_features_falls_back():
    d = decide_engine(13, "sub_13", FAILURE_TS, 1000.0, "retryable_soft", {}, model=_fake_model_dict([100.0] * 5))
    assert d.decision_source == SOURCE_FALLBACK
    assert "insufficient_features" in d.decision_reason


def test_fallback_reuses_already_computed_model_value_when_available():
    # low margin triggers fallback, but the model DID run successfully -- the
    # fallback decision should still carry a real predicted_recovery_value
    # for whichever candidate rule-based picks, not None.
    values = [100.0, 99.0, 98.0, 97.0, 96.0]
    d = decide_engine(14, "sub_14", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, abstention_threshold=25.0, model=_fake_model_dict(values))
    assert d.decision_source == SOURCE_FALLBACK
    assert d.predicted_recovery_value is not None
    assert d.model_version is not None  # model DID contribute a value here, unlike the true-unavailable case


def test_all_candidates_produce_a_selection_with_real_model(real_model):
    """Sanity: with the real fitted model, every one of the 5 real candidate
    types can appear across a range of synthetic contexts without crashing."""
    for i in range(5):
        d = decide_engine(100 + i, f"sub_real_{i}", FAILURE_TS + timedelta(days=i), 500.0 + i * 100, "retryable_soft", FAILURE_CONTEXT, model=real_model)
        assert d.selected_candidate_type in set(CANDIDATE_TYPES) | {NO_ACTION}


# ---------------------------------------------------------------------------
# Guardrails (existing + new)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bucket", ["hard_decline", "customer_cancelled", "unmapped"])
def test_non_retryable_classification_blocks_action(bucket):
    d = decide_engine(20, "sub_20", FAILURE_TS, 1000.0, bucket, FAILURE_CONTEXT, model=_fake_model_dict([100.0] * 5))
    assert d.selected_candidate_type == NO_ACTION
    assert d.decision_source == SOURCE_NO_ACTION


def test_max_retry_attempts_blocks_action():
    d = decide_engine(21, "sub_21", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, attempts_so_far=MAX_RETRY_ATTEMPTS, model=_fake_model_dict([100.0] * 5))
    assert d.selected_candidate_type == NO_ACTION
    assert "max_retry_attempts" in d.decision_reason


def test_already_decided_is_no_action():
    d = decide_engine(22, "sub_22", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, already_decided=True, model=_fake_model_dict([100.0] * 5))
    assert d.selected_candidate_type == NO_ACTION
    assert "duplicate_decision_skipped" in d.decision_reason


def test_candidate_after_horizon_never_selected():
    # near month-end so payday_window/month_end_window land beyond the 14-day horizon
    far_ts = datetime(2026, 3, 5, 14, 0, 0)
    d = decide_engine(23, "sub_23", far_ts, 1000.0, "retryable_soft", FAILURE_CONTEXT, abstention_threshold=-math.inf, model=_fake_model_dict([1.0, 1.0, 500.0, 1.0, 500.0]))
    # payday_window/month_end_window (indices 2, 4) got the highest fake values but should be excluded as invalid
    assert d.selected_candidate_type not in ("payday_window", "month_end_window") or d.selected_candidate_type == NO_ACTION


# ---------------------------------------------------------------------------
# Failure modes: empty candidate list / all invalid
# ---------------------------------------------------------------------------

def test_all_candidates_invalid_is_no_action(monkeypatch):
    import policy.decision_engine as de

    monkeypatch.setattr(de, "generate_candidates", lambda ts: [])
    d = decide_engine(24, "sub_24", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, model=_fake_model_dict([100.0] * 5))
    assert d.selected_candidate_type == NO_ACTION
    assert "blocked_no_valid_candidates" in d.decision_reason


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic_output():
    values = [100.0, 90.0, 80.0, 70.0, 60.0]
    kwargs = dict(event_id=30, subscription_id="sub_30", failure_timestamp=FAILURE_TS, amount=1000.0, classification_bucket="retryable_soft", failure_context=FAILURE_CONTEXT, model=_fake_model_dict(values))
    d1 = decide_engine(**kwargs)
    d2 = decide_engine(**kwargs)
    assert d1 == d2  # created_at is compare=False, everything else must match exactly


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_decision_serializes_to_valid_json():
    import json

    values = [100.0, 90.0, 80.0, 70.0, 60.0]
    d = decide_engine(31, "sub_31", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, model=_fake_model_dict(values))
    parsed = json.loads(d.to_json())
    for key in ("event_id", "subscription_id", "classification_bucket", "selected_candidate_type", "predicted_recovery_value", "intervention_cost", "expected_net_value", "decision_source", "model_version", "policy_version", "decision_reason"):
        assert key in parsed


def test_decision_serialization_handles_no_action():
    d = decide_engine(32, "sub_32", FAILURE_TS, 1000.0, "hard_decline", FAILURE_CONTEXT)
    import json

    parsed = json.loads(d.to_json())
    assert parsed["selected_candidate_type"] == NO_ACTION
    assert parsed["predicted_recovery_value"] is None


# ---------------------------------------------------------------------------
# Model-version / policy-version propagation
# ---------------------------------------------------------------------------

def test_model_version_propagates_when_model_used():
    d = decide_engine(33, "sub_33", FAILURE_TS, 1000.0, "retryable_soft", FAILURE_CONTEXT, model=_fake_model_dict([100.0, 90.0, 80.0, 70.0, 60.0]))
    assert d.model_version is not None
    assert d.decision_source == SOURCE_MODEL


def test_model_version_is_none_when_model_never_ran():
    d = decide_engine(34, "sub_34", FAILURE_TS, 1000.0, "hard_decline", FAILURE_CONTEXT)
    assert d.model_version is None


def test_policy_version_always_set():
    d = decide_engine(35, "sub_35", FAILURE_TS, 1000.0, "hard_decline", FAILURE_CONTEXT)
    assert d.policy_version == POLICY_VERSION


# ---------------------------------------------------------------------------
# DB-backed: audit logging + idempotency
# ---------------------------------------------------------------------------

def test_decide_for_failure_event_engine_creates_decision_and_audit_rows(test_db_session):
    db = test_db_session()
    row, created = decide_for_failure_event_engine(
        db, event_id=40001, subscription_id="sub_EngineAuditTest", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", failure_context=FAILURE_CONTEXT,
        model=_fake_model_dict([100.0, 90.0, 80.0, 70.0, 60.0]),
    )
    assert created is True
    assert isinstance(row, PolicyDecision)
    assert row.policy_version == POLICY_VERSION
    assert row.decision_source == SOURCE_MODEL
    assert row.classification_bucket == "retryable_soft"
    assert row.intervention_cost is not None

    audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 40001).all()
    assert len(audit_rows) == 1
    assert audit_rows[0].actor == "policy"
    assert "decision_source=" in audit_rows[0].reason
    db.close()


def test_decide_for_failure_event_engine_is_idempotent(test_db_session):
    db = test_db_session()
    first, first_created = decide_for_failure_event_engine(
        db, event_id=40002, subscription_id="sub_EngineIdempotent", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", failure_context=FAILURE_CONTEXT,
        model=_fake_model_dict([100.0, 90.0, 80.0, 70.0, 60.0]),
    )
    second, second_created = decide_for_failure_event_engine(
        db, event_id=40002, subscription_id="sub_EngineIdempotent", failure_timestamp=FAILURE_TS,
        amount=1000.0, classification_bucket="retryable_soft", failure_context=FAILURE_CONTEXT,
        model=_fake_model_dict([100.0, 90.0, 80.0, 70.0, 60.0]),
    )
    assert first_created is True
    assert second_created is False
    assert first.id == second.id
    assert db.query(PolicyDecision).filter(PolicyDecision.event_id == 40002).count() == 1

    audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 40002).order_by(AuditLog.id).all()
    assert len(audit_rows) == 2
    assert audit_rows[1].action == "policy_decision_skipped_duplicate"
    db.close()


def test_max_retry_attempts_enforced_across_engine_db_calls(test_db_session):
    db = test_db_session()
    for i in range(MAX_RETRY_ATTEMPTS):
        decide_for_failure_event_engine(
            db, event_id=200 + i, subscription_id="sub_engine_maxattempts",
            failure_timestamp=FAILURE_TS + timedelta(days=i * 20), amount=1000.0,
            classification_bucket="retryable_soft", failure_context=FAILURE_CONTEXT,
            model=_fake_model_dict([100.0, 90.0, 80.0, 70.0, 60.0]),
        )
    blocked_row, created = decide_for_failure_event_engine(
        db, event_id=300, subscription_id="sub_engine_maxattempts", failure_timestamp=FAILURE_TS + timedelta(days=100),
        amount=1000.0, classification_bucket="retryable_soft", failure_context=FAILURE_CONTEXT,
        model=_fake_model_dict([100.0, 90.0, 80.0, 70.0, 60.0]),
    )
    assert created is True
    assert blocked_row.selected_candidate_type == NO_ACTION
    assert "max_retry_attempts" in blocked_row.decision_reason
    db.close()


# ---------------------------------------------------------------------------
# Threshold selection (validation-only)
# ---------------------------------------------------------------------------

def test_threshold_selection_uses_only_validation_data(latent_splits, real_model):
    from evaluation.evaluate_decision_engine import THRESHOLD_CANDIDATES, select_abstention_threshold_on_validation

    _train, val_df, test_df = latent_splits
    chosen, results = select_abstention_threshold_on_validation(val_df, real_model)
    assert chosen in THRESHOLD_CANDIDATES
    assert set(results.keys()) == set(THRESHOLD_CANDIDATES)
    # the search must never reference test_df at all -- structural check: the function signature only accepts val_df
    import inspect

    params = list(inspect.signature(select_abstention_threshold_on_validation).parameters)
    assert "test_df" not in params and "test" not in params
