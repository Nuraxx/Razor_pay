"""
Track-03 tests: the new "Revenue at Risk" dashboard page and its
get_live_revenue_* / get_live_recovery_* / get_live_customer_recovery_queue_df
query functions. Mirrors tests/test_ui.py's exact patterns:
AppTest.from_file for page rendering, a seeded in-memory engine
(monkeypatching ui.data.get_live_session) for the query-layer tests --
NEVER the real data/recovery_agent.db.
"""
from datetime import datetime
from pathlib import Path

import pytest

import ui.data as data

APP_PATH = str(Path(__file__).resolve().parent.parent / "ui" / "app.py")

NOW = datetime(2026, 8, 25, 10, 0, 0)


@pytest.fixture
def live_db_session_factory(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    data.Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(data, "get_live_session", lambda: factory())
    return factory


_seed_counter = [0]


def _seed_revenue_risk_case(
    factory, *, event_type="checkout_abandoned", customer_ref="cust_1", amount=500.0, candidate_type="reminder",
    decision_source="rule_checkout_abandoned", model_version=None, predicted_recovery_probability=None, decision_reason="test seed",
):
    from app.models import PolicyDecision, RecoveryOutcome, RevenueRiskEvent
    from policy.policy_decision_store import REVENUE_DOMAIN_EVENT_ID_OFFSET

    _seed_counter[0] += 1
    db = factory()
    rre = RevenueRiskEvent(
        idempotency_key=f"{event_type}:{customer_ref}:{_seed_counter[0]}", event_type=event_type, external_id=customer_ref,
        customer_ref=customer_ref, amount=amount, occurred_at=NOW, received_at=NOW, reason="test_seed", status="OPEN",
    )
    db.add(rre)
    db.flush()
    db.add(PolicyDecision(
        event_id=rre.id + REVENUE_DOMAIN_EVENT_ID_OFFSET, subscription_id=customer_ref, selected_candidate_type=candidate_type,
        selected_candidate_datetime=NOW, policy_version="v1", decision_reason=decision_reason, decision_source=decision_source,
        classification_bucket="test_bucket", model_version=model_version, predicted_recovery_probability=predicted_recovery_probability,
    ))
    db.add(RecoveryOutcome(
        event_id=rre.id, event_type=event_type, at_risk_amount=amount, recovery_status="PENDING", confirmed_by="unconfirmed_pending",
    ))
    db.commit()
    rre_id = rre.id
    db.close()
    return rre_id


class TestQueryLayerEmptyDB:
    def test_get_live_revenue_risk_events_df_empty(self, live_db_session_factory):
        assert data.get_live_revenue_risk_events_df().empty

    def test_get_live_revenue_recovery_queue_df_empty(self, live_db_session_factory):
        assert data.get_live_revenue_recovery_queue_df().empty

    def test_get_live_revenue_at_risk_kpis_empty(self, live_db_session_factory):
        kpis = data.get_live_revenue_at_risk_kpis()
        assert kpis["total_revenue_risk_events"] == 0
        assert kpis["total_at_risk_amount"] == 0
        assert kpis["pending_cases"] == 0
        assert kpis["demo_synthetic_recovered_cases"] == 0

    def test_get_live_recovery_timeline_df_empty(self, live_db_session_factory):
        assert data.get_live_recovery_timeline_df().empty

    def test_get_live_recovery_outcomes_df_empty(self, live_db_session_factory):
        assert data.get_live_recovery_outcomes_df().empty

    def test_get_live_revenue_by_intervention_df_empty(self, live_db_session_factory):
        assert data.get_live_revenue_by_intervention_df().empty

    def test_get_live_customer_recovery_queue_df_empty(self, live_db_session_factory):
        assert data.get_live_customer_recovery_queue_df().empty


class TestQueryLayerSeededDB:
    def test_get_live_revenue_risk_events_df_returns_seeded_row(self, live_db_session_factory):
        _seed_revenue_risk_case(live_db_session_factory)
        df = data.get_live_revenue_risk_events_df()
        assert len(df) == 1
        assert df.iloc[0]["event_type"] == "checkout_abandoned"

    def test_get_live_revenue_recovery_queue_df_joins_correctly(self, live_db_session_factory):
        _seed_revenue_risk_case(live_db_session_factory)
        df = data.get_live_revenue_recovery_queue_df()
        assert len(df) == 1
        assert df.iloc[0]["selected_candidate_type"] == "reminder"
        assert df.iloc[0]["customer_ref"] == "cust_1"

    def test_recovery_queue_never_leaks_payment_failed_rows(self, live_db_session_factory):
        """The critical correctness case: a policy_decisions row from the
        EXISTING payment_failed path (decision_source="rule_based_fallback")
        must never appear in the revenue-risk queue, even if its event_id
        numerically collides with a revenue_risk_events.id."""
        from app.models import FailureEvent, PolicyDecision, RawEvent

        db = live_db_session_factory()
        raw = RawEvent(razorpay_event_id="evt_x", event_type="payment.failed", subscription_id="sub_x", signature_verified=True, raw_payload="{}")
        db.add(raw)
        db.flush()
        failure = FailureEvent(raw_event_id=raw.id, classification_bucket="retryable_soft", classification_confidence=1.0, rule_version="v1")
        db.add(failure)
        db.flush()
        # Deliberately give this payment_failed PolicyDecision the SAME id as
        # a soon-to-be-created revenue_risk_events row, to prove the exact
        # decision_source filter (not a "rule_%" wildcard) is what keeps them apart.
        db.add(PolicyDecision(
            event_id=failure.id, subscription_id="sub_x", selected_candidate_type="plus_1_day_morning",
            policy_version="v4", decision_reason="real payment fallback", decision_source="rule_based_fallback",
            classification_bucket="retryable_soft",
        ))
        db.commit()
        db.close()

        _seed_revenue_risk_case(live_db_session_factory, customer_ref="cust_2")

        df = data.get_live_revenue_recovery_queue_df()
        assert len(df) == 1  # only the real checkout_abandoned row, never the payment_failed one
        assert df.iloc[0]["customer_ref"] == "cust_2"

    def test_get_live_revenue_at_risk_kpis_counts_seeded_case(self, live_db_session_factory):
        _seed_revenue_risk_case(live_db_session_factory, amount=750.0)
        kpis = data.get_live_revenue_at_risk_kpis()
        assert kpis["total_revenue_risk_events"] == 1
        assert kpis["total_at_risk_amount"] == 750.0
        assert kpis["pending_cases"] == 1
        assert kpis["by_event_type"] == {"checkout_abandoned": 1}

    def test_get_live_recovery_outcomes_df_returns_seeded_row(self, live_db_session_factory):
        _seed_revenue_risk_case(live_db_session_factory)
        df = data.get_live_recovery_outcomes_df()
        assert len(df) == 1
        assert df.iloc[0]["recovery_status"] == "PENDING"
        assert df.iloc[0]["recovered_amount"] is None

    def test_dashboard_shows_recovered_only_after_authoritative_confirmation(self, live_db_session_factory):
        # Closed-loop hardening: the dashboard must show PENDING until (and
        # only until) a real payment.captured confirmation runs -- never
        # "recovered" merely because an action was scheduled.
        from app.models import FailureEvent, PolicyDecision, RawEvent, RecoveryOutcome
        from recovery.payment_reconciliation import confirm_payment_recovery

        db = live_db_session_factory()
        raw = RawEvent(razorpay_event_id="evt_ui_close_loop", event_type="payment.failed", payment_id="pay_ui_fail",
                        subscription_id="sub_ui_close_loop", amount=100000, signature_verified=True, raw_payload="{}")
        db.add(raw)
        db.flush()
        failure = FailureEvent(raw_event_id=raw.id, classification_bucket="retryable_soft", classification_confidence=1.0, rule_version="v1")
        db.add(failure)
        db.flush()
        db.add(PolicyDecision(
            event_id=failure.id, subscription_id="sub_ui_close_loop", selected_candidate_type="payday_window",
            policy_version="v4", decision_reason="seed", decision_source="rule_based_fallback", classification_bucket="retryable_soft",
        ))
        outcome = RecoveryOutcome(event_id=failure.id, event_type="payment_failed", at_risk_amount=1000.0, recovery_status="PENDING", confirmed_by="unconfirmed_pending")
        db.add(outcome)
        db.commit()
        failure_id = failure.id

        before = data.get_live_recovery_outcomes_df()
        row_before = before[before["event_id"] == failure_id].iloc[0]
        assert row_before["recovery_status"] == "PENDING"
        assert row_before["recovered_amount"] is None

        captured = RawEvent(razorpay_event_id="evt_ui_close_loop_captured", event_type="payment.captured", payment_id="pay_ui_retry_success",
                             subscription_id="sub_ui_close_loop", amount=100000, signature_verified=True, raw_payload="{}")
        db.add(captured)
        db.flush()
        confirm_payment_recovery(db, captured)
        db.close()

        after = data.get_live_recovery_outcomes_df()
        row_after = after[after["event_id"] == failure_id].iloc[0]
        assert row_after["recovery_status"] == "RECOVERED"
        assert row_after["recovered_amount"] == 1000.0
        assert row_after["confirmed_by"] == "webhook_confirmed"
        assert row_after["confirmed_payment_id"] == "pay_ui_retry_success"

    def test_get_live_revenue_by_intervention_df_groups_correctly(self, live_db_session_factory):
        _seed_revenue_risk_case(live_db_session_factory, customer_ref="a", candidate_type="reminder")
        _seed_revenue_risk_case(live_db_session_factory, customer_ref="b", candidate_type="reminder")
        _seed_revenue_risk_case(live_db_session_factory, customer_ref="c", candidate_type="escalation", event_type="receivable_overdue", decision_source="rule_receivable_overdue")
        df = data.get_live_revenue_by_intervention_df()
        reminder_row = df[df["intervention"] == "reminder"].iloc[0]
        assert reminder_row["case_count"] == 2

    def test_get_live_customer_recovery_queue_df_dedupes_by_customer(self, live_db_session_factory):
        _seed_revenue_risk_case(live_db_session_factory, customer_ref="dup_cust")
        _seed_revenue_risk_case(live_db_session_factory, customer_ref="dup_cust")
        df = data.get_live_customer_recovery_queue_df()
        assert len(df) == 1  # one row per customer, most recent only

    def test_ml_sourced_decisions_appear_in_the_recovery_queue(self, live_db_session_factory):
        # Regression test: REVENUE_DOMAIN_DECISION_SOURCES used to omit
        # "ml_unified_v1" entirely, which would silently make every real
        # ML-driven decision invisible on this dashboard once the unified
        # model was actually wired into the live path.
        _seed_revenue_risk_case(
            live_db_session_factory, customer_ref="cust_ml", candidate_type="retry_checkout",
            decision_source="ml_unified_v1", model_version="unified_catboost_v1", predicted_recovery_probability=0.42,
        )
        df = data.get_live_revenue_recovery_queue_df()
        assert len(df) == 1
        assert df.iloc[0]["decision_source"] == "ml_unified_v1"
        assert df.iloc[0]["model_version"] == "unified_catboost_v1"
        assert df.iloc[0]["predicted_recovery_probability"] == 0.42


class TestRevenuePipelineSnapshot:
    def test_returns_none_on_empty_db(self, live_db_session_factory):
        assert data.get_live_revenue_pipeline_snapshot() is None

    def test_reflects_the_latest_ml_sourced_decision(self, live_db_session_factory):
        _seed_revenue_risk_case(live_db_session_factory, customer_ref="cust_old", decision_source="rule_checkout_abandoned")
        _seed_revenue_risk_case(
            live_db_session_factory, customer_ref="cust_new", event_type="receivable_overdue", candidate_type="friendly_reminder",
            decision_source="ml_unified_v1", model_version="unified_catboost_v1", predicted_recovery_probability=0.77,
            decision_reason="unified_model_score=0.770; candidate=friendly_reminder | rule_baseline_candidate=escalation",
        )
        snapshot = data.get_live_revenue_pipeline_snapshot()
        assert snapshot is not None
        assert snapshot["event_type"] == "receivable_overdue"
        assert snapshot["is_ml_sourced"] is True
        assert snapshot["ml_status"] == "USED"
        assert snapshot["model_version"] == "unified_catboost_v1"
        assert snapshot["predicted_recovery_probability"] == 0.77
        assert snapshot["selected_candidate_type"] == "friendly_reminder"
        assert snapshot["rule_baseline_candidate"] == "escalation"

    def test_rule_sourced_decision_is_not_marked_ml_sourced(self, live_db_session_factory):
        _seed_revenue_risk_case(live_db_session_factory, decision_source="rule_checkout_abandoned")
        snapshot = data.get_live_revenue_pipeline_snapshot()
        assert snapshot["is_ml_sourced"] is False
        assert snapshot["ml_status"] == "FALLBACK"
        assert snapshot["model_version"] is None

    def test_ml_consulted_but_overridden_is_distinct_from_fallback(self, live_db_session_factory):
        # The critical dashboard distinction: ML ran and scored a real
        # recommendation, but the rule-based eligibility gate overrode it --
        # this must never be indistinguishable from "ML never ran at all".
        _seed_revenue_risk_case(
            live_db_session_factory, event_type="receivable_overdue", candidate_type="human_handoff",
            decision_source="rule_receivable_overdue", model_version="unified_catboost_v1", predicted_recovery_probability=0.63,
            decision_reason="disputed_requires_human_review | ml_consulted=True ml_recommendation=friendly_reminder ml_score=0.630 ml_model=unified_catboost_v1 | policy_overrides_ml_due_to_eligibility",
        )
        snapshot = data.get_live_revenue_pipeline_snapshot()
        assert snapshot["ml_status"] == "CONSULTED_OVERRIDDEN"
        assert snapshot["is_ml_sourced"] is False  # policy made the final call, not ML
        assert snapshot["ml_recommendation"] == "friendly_reminder"
        assert snapshot["selected_candidate_type"] == "human_handoff"
        assert snapshot["predicted_recovery_probability"] == 0.63


class TestRevenueAtRiskPageRenders:
    def test_page_renders_without_exception(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_PATH, default_timeout=120)
        at.run()
        at.sidebar.radio[0].set_value("Revenue at Risk").run()
        assert not at.exception, f"Revenue at Risk raised: {[e.value for e in at.exception]}"

    def test_existing_pages_still_render_after_adding_the_new_one(self):
        from streamlit.testing.v1 import AppTest

        for page in ["Overview", "Recovery Queue", "Payment Events", "Analytics", "Communications", "Audit Log", "System / Demo"]:
            at = AppTest.from_file(APP_PATH, default_timeout=120)
            at.run()
            at.sidebar.radio[0].set_value(page).run()
            assert not at.exception, f"{page} raised: {[e.value for e in at.exception]}"

    def test_intervention_section_heading_is_labeled_at_risk_not_recovered(self):
        # Full-system audit finding: this section's own table only ever
        # shows summed at-risk GMV per intervention type (RecoveryOutcome.
        # recovered_amount is never populated for live data) -- the header
        # used to read "Revenue Recovered by Intervention", overclaiming
        # relative to what the data underneath it actually is. Pinned here
        # so this specific honesty regression can't silently reappear.
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_PATH, default_timeout=120)
        at.run()
        at.sidebar.radio[0].set_value("Revenue at Risk").run()
        headings = [md.value for md in at.markdown]
        assert any("Revenue At Risk by Intervention" in h for h in headings)
        assert not any("Revenue Recovered by Intervention" in h for h in headings)

    def test_demo_generator_button_works_via_ui(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_PATH, default_timeout=120)
        at.run()
        at.sidebar.radio[0].set_value("System / Demo").run()
        buttons = [b for b in at.button if "Generate demo revenue-risk events" in b.label]
        assert len(buttons) == 1
        buttons[0].click().run()
        assert not at.exception, f"raised: {[e.value for e in at.exception]}"
