"""
Tests for model/unified_model.py -- the single ML model shared across all 5
revenue-risk domains. Covers: feature construction, all five event types,
domain-valid candidate generation, no target leakage, entity-split
integrity, and the artifact-unavailable fallback (Phase 16/17 of the
unified-ML generalization work).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier

from model.unified_model import (
    CANDIDATE_SPACE,
    FEATURE_COLUMNS,
    SUPPORTED_EVENT_TYPES,
    UnifiedModelUnavailable,
    _entity_level_split,
    _make_training_data,
    build_unified_feature_vector,
    generate_valid_candidates,
    get_live_unified_model,
    load_unified_model,
    normalize_event,
    reset_live_unified_model_cache,
    score_event_candidates,
    select_best_candidate,
    train_unified_model,
)

EVENTS_BY_DOMAIN = {
    "payment_failed": {"event_type": "payment_failed_no_subscription", "amount": 1.0, "failure_reason": "insufficient_fund"},
    "checkout_abandoned": {"event_type": "checkout_abandoned", "amount": 999.0, "cart_value": 999.0, "checkout_age_minutes": 120},
    "mandate_failed": {"event_type": "mandate_failed", "amount": 1200.0, "mandate_attempt_number": 1},
    "receivable_overdue": {"event_type": "receivable_overdue", "amount": 45000.0, "days_overdue": 40, "invoice_amount": 45000.0},
    "promise_to_pay_broken": {"event_type": "promise_to_pay_broken", "amount": 800.0, "promise_age_days": 5, "promise_confidence": 0.6},
}


@pytest.fixture(autouse=True)
def _restore_live_model_cache():
    """The live-model cache is process-global -- reset it before AND after
    every test in this module so tests never leak state into each other or
    into tests outside this file."""
    reset_live_unified_model_cache()
    yield
    reset_live_unified_model_cache()


class TestEventTypeAliasing:
    def test_payment_failed_no_subscription_aliases_to_payment_failed_for_ml(self):
        normalized = normalize_event({"event_type": "payment_failed_no_subscription", "amount": 1.0})
        assert normalized["event_type"] == "payment_failed"

    def test_aliased_event_type_gets_real_candidates_not_an_empty_list(self):
        # Before the alias fix, this returned [] -- the exact bug that made
        # a real Payment Link failure (subscription_id=NULL) silently never
        # reach the unified model.
        candidates = generate_valid_candidates("payment_failed_no_subscription")
        assert candidates == CANDIDATE_SPACE["payment_failed"]
        assert candidates != []


class TestFeatureConstruction:
    def test_feature_vector_has_explicit_event_type_column(self):
        frame = build_unified_feature_vector({"event_type": "checkout_abandoned", "amount": 500.0}, candidate_type="reminder")
        assert "event_type" in frame.columns
        assert frame.iloc[0]["event_type"] == "checkout_abandoned"

    def test_missing_domain_fields_are_nan_not_a_meaningful_zero(self):
        # checkout_abandoned has no invoice_amount/days_overdue -- those must
        # be NaN (a deliberate "not applicable" marker), never silently 0.0
        # (which would be indistinguishable from a real zero-value invoice).
        frame = build_unified_feature_vector({"event_type": "checkout_abandoned", "amount": 500.0})
        assert np.isnan(frame.iloc[0]["invoice_amount"])
        assert np.isnan(frame.iloc[0]["days_overdue"])
        assert np.isnan(frame.iloc[0]["promise_confidence"])

    def test_feature_columns_never_include_the_label_or_entity_key(self):
        # No target/outcome leakage: the label ("recovered") and the
        # split-only bookkeeping column ("entity_key") must never be part of
        # the feature schema the model is fit/scored on.
        assert "recovered" not in FEATURE_COLUMNS
        assert "entity_key" not in FEATURE_COLUMNS
        assert "recovery_probability" not in FEATURE_COLUMNS
        assert "recovery_value" not in FEATURE_COLUMNS


class TestCandidateGeneration:
    @pytest.mark.parametrize("event_type", SUPPORTED_EVENT_TYPES)
    def test_every_supported_domain_has_valid_candidates(self, event_type):
        candidates = generate_valid_candidates(event_type)
        assert len(candidates) >= 1
        assert candidates == CANDIDATE_SPACE[event_type]

    def test_unsupported_event_type_returns_no_candidates(self):
        assert generate_valid_candidates("not_a_real_domain") == []

    def test_candidates_are_never_the_eligibility_gate_values(self):
        # NO_ACTION/"wait"/"human_handoff" mean "nothing eligible" or "must
        # escalate to a human" -- those stay rule-authoritative
        # (policy/revenue_recovery_policy.py's _ML_SKIP_CANDIDATES gate), so
        # ML must never be offered them as something to recommend.
        for event_type in SUPPORTED_EVENT_TYPES:
            candidates = set(CANDIDATE_SPACE[event_type])
            assert "NO_ACTION" not in candidates
            assert "wait" not in candidates
            assert "human_handoff" not in candidates


class TestUnifiedDatasetAndSplit:
    def test_dataset_contains_all_five_domains(self):
        df = _make_training_data()
        assert set(df["event_type"].unique()) == set(SUPPORTED_EVENT_TYPES)

    def test_label_is_not_deterministic_from_a_single_feature(self):
        # Two rows with the identical amount/candidate must NOT always share
        # the same label -- proves genuine sampling noise, not a
        # deterministic function of one input feature.
        df = _make_training_data()
        grouped = df.groupby(["event_type", "candidate_type", "amount"])["recovered"].nunique()
        assert (grouped > 1).any(), "expected at least one (event_type, candidate, amount) group with a mixed label"

    def test_entity_level_split_has_no_overlap_and_covers_all_domains(self):
        df = _make_training_data()
        train_df, val_df, test_df = _entity_level_split(df)

        train_keys, val_keys, test_keys = set(train_df["entity_key"]), set(val_df["entity_key"]), set(test_df["entity_key"])
        assert not (train_keys & val_keys)
        assert not (train_keys & test_keys)
        assert not (val_keys & test_keys)

        for split_df, name in ((train_df, "train"), (val_df, "validation"), (test_df, "test")):
            assert set(split_df["event_type"].unique()) == set(SUPPORTED_EVENT_TYPES), f"{name} split is missing a domain"
            assert len(split_df) > 0

    def test_split_sizes_are_non_trivial(self):
        df = _make_training_data()
        train_df, val_df, test_df = _entity_level_split(df)
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(df)
        assert len(val_df) > 0 and len(test_df) > 0


class TestTrainingProducesARealModel:
    def test_train_unified_model_fits_a_real_catboost_classifier(self, tmp_path, monkeypatch):
        import model.unified_model as um

        artifact_path = tmp_path / "unified_model_test.joblib"
        monkeypatch.setattr(um, "UNIFIED_MODEL_PATH", artifact_path)
        monkeypatch.setattr(um, "TRAINING_REPORT_PATH", tmp_path / "report.json")

        fitted = train_unified_model()

        assert isinstance(fitted["model"], CatBoostClassifier)
        assert fitted["model_version"] == "unified_catboost_v1"
        assert artifact_path.exists()

        # Test metrics are computed (for reporting) but never used to fit.
        assert fitted["metrics"]["test"]["n"] > 0
        assert fitted["dataset_stats"]["train_rows"] > 0
        assert fitted["dataset_stats"]["validation_rows"] > 0
        assert fitted["dataset_stats"]["test_rows"] > 0

    def test_loaded_artifact_is_independently_usable_for_all_five_domains(self, tmp_path, monkeypatch):
        import model.unified_model as um

        artifact_path = tmp_path / "unified_model_test.joblib"
        monkeypatch.setattr(um, "UNIFIED_MODEL_PATH", artifact_path)
        monkeypatch.setattr(um, "TRAINING_REPORT_PATH", tmp_path / "report.json")
        train_unified_model()

        # Independent load -- a fresh call, not the object train_unified_model returned.
        model = load_unified_model()
        assert model["model_version"] == "unified_catboost_v1"
        assert isinstance(model["model"], CatBoostClassifier)

        for domain, event in EVENTS_BY_DOMAIN.items():
            scores = score_event_candidates(event, model)
            assert len(scores) >= 1, f"no candidate scores for {domain}"
            for s in scores:
                assert 0.0 <= s["predicted_recovery_probability"] <= 1.0
                assert s["model_version"] == "unified_catboost_v1"
            best = select_best_candidate(event, model)
            assert best["candidate_type"] in CANDIDATE_SPACE[normalize_event(event)["event_type"]]


class TestArtifactUnavailableFallback:
    def test_load_unified_model_raises_when_artifact_missing(self, tmp_path, monkeypatch):
        import model.unified_model as um

        monkeypatch.setattr(um, "UNIFIED_MODEL_PATH", tmp_path / "does_not_exist.joblib")
        with pytest.raises(UnifiedModelUnavailable):
            load_unified_model()

    def test_get_live_unified_model_returns_none_without_raising_when_artifact_missing(self, tmp_path, monkeypatch):
        import model.unified_model as um

        monkeypatch.setattr(um, "UNIFIED_MODEL_PATH", tmp_path / "does_not_exist.joblib")
        reset_live_unified_model_cache()
        result = get_live_unified_model()  # must not raise -- caller falls back to rule-based deciders
        assert result is None

    def test_get_live_unified_model_caches_across_calls(self):
        first = get_live_unified_model()
        second = get_live_unified_model()
        assert first is second  # same object -- loaded/deserialized once, not per call

    def test_load_unified_model_raises_cleanly_on_corrupt_artifact(self, tmp_path, monkeypatch):
        # Regression test: a corrupt/truncated artifact used to raise a raw
        # joblib/pickle exception (e.g. ValueError) instead of
        # UnifiedModelUnavailable -- get_live_unified_model()'s
        # `except UnifiedModelUnavailable` would NOT catch that, so the
        # live app would crash at startup or on the first webhook request
        # instead of degrading to the rule-based fallback.
        import model.unified_model as um

        corrupt_path = tmp_path / "corrupt_unified_model.joblib"
        corrupt_path.write_text("this is not a valid joblib/pickle file")
        monkeypatch.setattr(um, "UNIFIED_MODEL_PATH", corrupt_path)
        with pytest.raises(UnifiedModelUnavailable):
            load_unified_model()

    def test_get_live_unified_model_falls_back_cleanly_on_corrupt_artifact(self, tmp_path, monkeypatch):
        import model.unified_model as um

        corrupt_path = tmp_path / "corrupt_unified_model.joblib"
        corrupt_path.write_text("this is not a valid joblib/pickle file")
        monkeypatch.setattr(um, "UNIFIED_MODEL_PATH", corrupt_path)
        reset_live_unified_model_cache()
        result = get_live_unified_model()  # must not raise
        assert result is None

    def test_load_unified_model_rejects_wrong_shaped_artifact(self, tmp_path, monkeypatch):
        # A file that deserializes fine but isn't actually a unified-model
        # dict (e.g. a stale artifact from a different model) must also be
        # treated as unavailable, not used as-is (which would crash later,
        # deep inside score_event_candidates, with a confusing KeyError).
        import joblib

        import model.unified_model as um

        wrong_shape_path = tmp_path / "wrong_shape.joblib"
        joblib.dump({"not": "a real unified model artifact"}, wrong_shape_path)
        monkeypatch.setattr(um, "UNIFIED_MODEL_PATH", wrong_shape_path)
        with pytest.raises(UnifiedModelUnavailable):
            load_unified_model()
