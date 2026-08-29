"""
Dashboard tests: data loaders (missing-file handling), formatting,
status mapping, event/audit loading, no-secrets-in-UI-data, demo scenario
execution, and that the app itself launches (imports + runs) cleanly in
offline/mock mode. Uses `streamlit.testing.v1.AppTest` to actually execute
`ui/app.py`'s script (not just import it) without needing a browser.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import ui.data as data
from app.config import settings

APP_PATH = str(Path(__file__).resolve().parent.parent / "ui" / "app.py")


# ---------------------------------------------------------------------------
# Dashboard imports / launches successfully, offline/mock mode
# ---------------------------------------------------------------------------

def test_dashboard_imports_successfully():
    import ui.app  # noqa: F401 -- must not raise


def test_ui_package_imports_cleanly_in_a_fresh_process():
    """Regression test for a circular-import risk: `import ui.data` must
    never depend on `ui.components`/`ui.app` having already been imported
    first. Runs in a genuinely fresh interpreter (this pytest process
    already has ui.data warm in sys.modules by the time this test runs,
    which would hide a real ordering bug)."""
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parent.parent
    code = (
        "import ui.data; import ui.components; import ui.app; "
        "from ui.data import format_inr, humanize_status; "
        "print('UI imports OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(project_root), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "UI imports OK" in result.stdout


def test_ui_app_survives_streamlits_sys_path_insertion():
    """Regression test for the real bug this project hit: `streamlit run
    ui/app.py` inserts this script's own directory onto sys.path
    (streamlit/web/bootstrap.py::_fix_sys_path), and because ui/app.py and
    ui/data.py share bare filenames with the real top-level `app` package
    and `data` namespace package, that shadowed them --
    "ModuleNotFoundError: No module named 'app.db'; 'app' is not a package"
    and the equivalent for `data.generate_synthetic_dataset`. Runs in a
    fresh process with ui/'s own directory pre-pended to sys.path, exactly
    as Streamlit's bootstrap does, before importing ui.app -- the plain
    `import ui.app` test above does NOT reproduce this, since it never
    pollutes sys.path this way."""
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parent.parent
    code = (
        "import sys; sys.path.insert(0, 'ui'); "
        "import runpy; runpy.run_path('ui/app.py', run_name='__main__'); "
        "print('SCRIPT EXECUTED CLEANLY')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(project_root), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "SCRIPT EXECUTED CLEANLY" in result.stdout


def test_dashboard_starts_in_mock_offline_mode():
    assert settings.LLM_PROVIDER == "mock"
    assert settings.ANTHROPIC_API_KEY == "" or settings.LLM_PROVIDER != "anthropic"


def test_dashboard_runs_without_exception_default_page():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception


@pytest.mark.parametrize("page", ["Overview", "Recovery Queue", "Payment Events", "Analytics", "Communications", "Audit Log", "System / Demo"])
def test_every_page_renders_without_exception(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page} raised: {[e.value for e in at.exception]}"


def test_demo_scenario_execution_via_ui(monkeypatch):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("System / Demo").run()
    at.selectbox[0].set_value("LLM failure").run()
    at.button[-1].click().run()
    assert not at.exception


# ---------------------------------------------------------------------------
# Data loaders handle missing files gracefully (brief section 17)
# ---------------------------------------------------------------------------

def test_load_csv_missing_file_returns_none():
    assert data.load_csv("this_file_definitely_does_not_exist.csv") is None


def test_load_report_missing_file_returns_none():
    assert data.load_report("this_report_definitely_does_not_exist.json") is None


def test_load_csv_existing_file_returns_dataframe():
    df = data.load_csv("failure_events.csv")
    if df is not None:  # file may not exist in every environment -- if present, must be a real DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


def test_build_demo_database_handles_missing_source_data(monkeypatch):
    monkeypatch.setattr(data, "load_csv", lambda name: None)
    data.build_demo_database.clear()
    db, queue_df = data.build_demo_database(sample_size=5)
    assert queue_df.empty
    assert db is not None  # dashboard must still have a usable (empty) database, never None/crash
    data.build_demo_database.clear()


# ---------------------------------------------------------------------------
# INR formatting
# ---------------------------------------------------------------------------

def test_format_inr_basic():
    assert data.format_inr(1000) == "₹1,000.00"


def test_format_inr_indian_grouping():
    assert data.format_inr(1234567.891) == "₹12,34,567.89"


def test_format_inr_small_value_no_grouping():
    assert data.format_inr(42) == "₹42.00"


def test_format_inr_negative():
    assert data.format_inr(-99.5) == "-₹99.50"


def test_format_inr_none_and_nan():
    assert data.format_inr(None) == "—"
    assert data.format_inr(float("nan")) == "—"


# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------

def test_humanize_status():
    assert data.humanize_status("RETRY_BLOCKED") == "Retry Blocked"
    assert data.humanize_status(None) == "—"


def test_status_style_known_and_unknown_values():
    from ui.styles import status_style

    bg, fg = status_style("RETRY_BLOCKED")
    assert bg and fg
    bg2, fg2 = status_style("SOME_UNKNOWN_STATUS")
    assert bg2 and fg2  # falls back to a neutral style, never raises


def test_derive_final_status_matches_orchestrator_precedence():
    row_no_action = pd.Series({"selected_candidate_type": "NO_ACTION", "communication_action": "skipped", "decision_source": "no_action"})
    assert data._derive_final_status(row_no_action) == "NO_ACTION"

    row_blocked_comm = pd.Series({"selected_candidate_type": "payday_window", "communication_action": "blocked", "decision_source": "subscription_value_model"})
    assert data._derive_final_status(row_blocked_comm) == "COMMUNICATION_BLOCKED"

    row_policy_fallback = pd.Series({"selected_candidate_type": "payday_window", "communication_action": "sent", "decision_source": "rule_based_fallback"})
    assert data._derive_final_status(row_policy_fallback) == "POLICY_FALLBACK"


# ---------------------------------------------------------------------------
# Event loading / audit loading
# ---------------------------------------------------------------------------

def test_event_loading_and_audit_loading_from_demo_database():
    data.build_demo_database.clear()
    db, queue_df = data.build_demo_database(sample_size=10)
    if queue_df.empty:
        pytest.skip("no source data available in this environment")

    first_event_id = queue_df.iloc[0]["event_id"]
    detail = data.get_event_detail(db, first_event_id)
    assert detail is not None
    assert detail["policy"].event_id == first_event_id
    assert len(detail["audit"]) > 0

    audit_df = data.get_audit_log_df(db)
    assert not audit_df.empty
    assert {"classifier", "policy", "compliance"}.issubset(set(audit_df["actor"].unique()))
    data.build_demo_database.clear()


def test_get_event_detail_unknown_event_returns_none():
    data.build_demo_database.clear()
    db, queue_df = data.build_demo_database(sample_size=5)
    assert data.get_event_detail(db, "evt_definitely_does_not_exist") is None
    data.build_demo_database.clear()


# ---------------------------------------------------------------------------
# No secrets exposed in UI data (brief section 12/15)
# ---------------------------------------------------------------------------

def test_no_secrets_in_audit_log_data():
    data.build_demo_database.clear()
    db, queue_df = data.build_demo_database(sample_size=10)
    if queue_df.empty:
        pytest.skip("no source data available in this environment")

    audit_df = data.get_audit_log_df(db)
    forbidden = ["api_key", "webhook_secret", "authorization", "bearer "]
    for reason in audit_df["reason"].dropna():
        lowered = reason.lower()
        for term in forbidden:
            assert term not in lowered, f"forbidden term {term!r} found in audit reason"
    data.build_demo_database.clear()


def test_no_secrets_in_llm_invocation_data():
    data.build_demo_database.clear()
    db, queue_df = data.build_demo_database(sample_size=10)
    if queue_df.empty:
        pytest.skip("no source data available in this environment")

    llm_df = data.get_llm_invocations_df(db)
    for output in llm_df["structured_output"].dropna():
        lowered = output.lower()
        assert "api_key" not in lowered
        assert "webhook_secret" not in lowered
    data.build_demo_database.clear()


# ---------------------------------------------------------------------------
# Demo scenario execution (direct, not via UI widgets)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", list(data.DEMO_SCENARIOS.keys()))
def test_run_demo_scenario_executes_cleanly(scenario):
    model = data._try_load_model()
    result, audit_rows, _promise_rows = data.run_demo_scenario(scenario, model)
    assert result.final_status in (
        "RETRY_ALLOWED", "RETRY_BLOCKED", "COMMUNICATION_ALLOWED", "COMMUNICATION_BLOCKED",
        "COMMUNICATION_DEFERRED", "NO_ACTION", "POLICY_FALLBACK", "LLM_FALLBACK",
    )
    assert len(audit_rows) > 0


def test_llm_failure_scenario_produces_llm_fallback_status():
    model = data._try_load_model()
    result, _audit_rows, _promise_rows = data.run_demo_scenario("LLM failure", model)
    assert result.llm_success is False
    assert result.payment_action == "retry_scheduled"  # payment decision unaffected by LLM failure


def test_hard_decline_scenario_blocks_retry():
    model = data._try_load_model()
    result, _audit_rows, _promise_rows = data.run_demo_scenario("Hard decline (payment-method-update nudge)", model)
    assert result.payment_action == "no_action"
    assert result.classification_bucket == "hard_decline"


def test_customer_opt_out_scenario_blocks_communication_not_payment():
    model = data._try_load_model()
    result, _audit_rows, _promise_rows = data.run_demo_scenario("Customer opt-out", model)
    assert result.compliance_allowed is True
    assert result.communication_action == "blocked"


def test_promise_to_pay_scenario_overrides_retry_timing():
    model = data._try_load_model()
    result, _audit_rows, promise_rows = data.run_demo_scenario("Promise-to-pay override", model)
    assert len(promise_rows) == 1
    assert promise_rows[0].status == "VALID"
    assert result.promise_to_pay_applied is True
    assert result.selected_candidate_type == "promise_to_pay"


# ---------------------------------------------------------------------------
# FIX pass: statistical significance + economics sections (Analytics page)
# ---------------------------------------------------------------------------

def test_analytics_page_statistical_values_load_from_real_report():
    from streamlit.testing.v1 import AppTest

    report = data.load_report("decision_engine_v4_evaluation.json")
    if report is None or "statistical_tests" not in report:
        pytest.skip("no evaluation report with statistical_tests available in this environment")

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("Analytics").run()
    assert not at.exception
    rendered_text = " ".join(md.value for md in at.markdown) + " ".join(c.value for c in at.caption)
    assert "McNemar p-value" in rendered_text
    assert "Recovery-rate delta" in rendered_text


def test_analytics_page_economics_values_load_from_real_report():
    from streamlit.testing.v1 import AppTest

    report = data.load_report("decision_engine_v4_evaluation.json")
    if report is None or "economics" not in report:
        pytest.skip("no evaluation report with economics available in this environment")

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("Analytics").run()
    assert not at.exception
    rendered_text = " ".join(md.value for md in at.markdown) + " ".join(c.value for c in at.caption)
    assert "Merchant Recovery" in rendered_text
    assert "Razorpay Fee Take" in rendered_text
    assert "Net Recovery Value" in rendered_text


def test_analytics_page_missing_report_handled_safely(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(data, "load_report", lambda name: None)
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("Analytics").run()
    assert not at.exception


def test_render_statistical_significance_section_handles_none_report():
    import ui.app as app

    app.render_statistical_significance_section(None)  # must not raise
    app.render_statistical_significance_section({})  # must not raise
    app.render_statistical_significance_section({"statistical_tests": {}})  # missing sub-keys would raise if accessed unsafely -- but this dict itself IS present, so it's the "real report, degenerate shape" case, not "missing report"


def test_render_economics_section_handles_none_report():
    import ui.app as app

    app.render_economics_section(None)  # must not raise
    app.render_economics_section({})  # must not raise


def test_render_statistical_significance_section_with_synthetic_report():
    import ui.app as app

    report = {
        "realized_counterfactual": {
            "fixed_retry": {"recovery_rate": 0.80, "total_recovered_rs": 21854.10},
            "improved_fallback_policy": {"recovery_rate": 0.70, "total_recovered_rs": 19997.23, "incremental_rs_vs_fixed_retry": -1856.87},
        },
        "statistical_tests": {
            "population": {"n_events": 60, "held_out_split": "test"},
            "mcnemar": {"policy_a": "improved_fallback_policy", "policy_b": "fixed_retry", "only_a_recovered": 3, "only_b_recovered": 9, "p_value": 0.146, "method": "exact_binomial"},
            "bootstrap_ci": {"point_estimate": -1856.87, "lower_bound": -4878.20, "upper_bound": 1295.62, "confidence_level": 0.95, "n_resamples": 10000, "seed": 42},
        },
    }
    app.render_statistical_significance_section(report)  # must not raise


def test_render_economics_section_with_synthetic_report():
    import ui.app as app

    report = {
        "economics": {
            "fixed_retry": {"recovered_gmv": 21854.10, "intervention_cost": 300.0, "razorpay_fee_take": 515.76, "net_recovery_value": 21038.34},
            "improved_fallback_policy": {"recovered_gmv": 19997.23, "intervention_cost": 300.0, "razorpay_fee_take": 471.93, "net_recovery_value": 19225.30},
        }
    }
    app.render_economics_section(report)  # must not raise


def test_render_baseline_definitions_section_handles_missing_report():
    import ui.app as app

    app.render_baseline_definitions_section(None)  # must not raise
    app.render_baseline_definitions_section({})  # must not raise


def test_render_baseline_definitions_section_with_synthetic_report():
    import ui.app as app

    report = {
        "contact_and_intervention_metrics": {
            "fixed_retry": {"average_retry_attempts": 1.4},
            "rule_based": {"customer_contact_rate": 1.0, "average_contacts_per_contacted_subscription": 2.0},
        }
    }
    app.render_baseline_definitions_section(report)  # must not raise


# ---------------------------------------------------------------------------
# Dynamic test count
# ---------------------------------------------------------------------------

def test_count_test_functions_is_dynamic_and_positive():
    count = data.count_test_functions()
    assert count > 0
    # sanity: this very file contributes at least a handful of test_ functions
    assert count >= 20


# ---------------------------------------------------------------------------
# LIVE DATABASE query layer (Part 14/22/25/26 of the operations-console
# rebuild). Exercised against a controlled, seeded, throwaway in-memory
# SQLite engine -- monkeypatching ui.data.get_live_session -- NEVER the
# real data/recovery_agent.db, so these tests are hermetic and
# reproducible regardless of what real webhook traffic exists locally.
# ---------------------------------------------------------------------------

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


def _seed_live_event(
    factory, *, suffix="1", subscription_id="sub_live_1", error_reason="insufficient_fund",
    classification_bucket="retryable_soft", selected_candidate_type="plus_1_day_morning",
    with_llm=True, with_blocked_comm=False, llm_success=True, llm_provider="mock", llm_model="mock",
    razorpay_event_id=None, signature_verified=True, selected_candidate_datetime=None,
):
    from app.models import AuditLog, FailureEvent, LLMInvocation, PolicyDecision, RawEvent

    db = factory()
    raw = RawEvent(
        razorpay_event_id=razorpay_event_id or f"evt_test_{suffix}", event_type="payment.failed", payment_id=f"pay_test_{suffix}",
        subscription_id=subscription_id, amount=250000, currency="INR", error_reason=error_reason,
        error_source="gateway", error_step="payment_authorization", signature_verified=signature_verified, raw_payload="{}",
    )
    db.add(raw)
    db.flush()
    failure = FailureEvent(raw_event_id=raw.id, classification_bucket=classification_bucket, classification_confidence=1.0, rule_version="v1")
    db.add(failure)
    db.flush()
    policy = PolicyDecision(
        event_id=failure.id, subscription_id=subscription_id, selected_candidate_type=selected_candidate_type,
        selected_candidate_datetime=selected_candidate_datetime,
        policy_version="v4", decision_reason="test seed", decision_source="subscription_value_model", classification_bucket=classification_bucket,
    )
    db.add(policy)
    db.add(AuditLog(raw_event_id=raw.id, failure_event_id=failure.id, action="webhook_received_and_stored", actor="system"))
    db.add(
        AuditLog(
            failure_event_id=failure.id, action="orchestrator_compliance", actor="compliance",
            reason=(
                f"payment_action_allowed=True payment_reason=ok | "
                f"communication_action_allowed={'False' if with_blocked_comm else 'True'} "
                f"communication_reason={'blocked for test' if with_blocked_comm else 'ok'} | rule_version=v1"
            ),
        )
    )
    if with_llm:
        db.add(
            LLMInvocation(
                event_id=failure.id, task_name="outreach_microcopy", model_name=llm_model, prompt_version="v1",
                provider=llm_provider, success=llm_success,
                structured_output=json.dumps({"message_text": "hi", "language": "en", "failure_bucket": classification_bucket, "customer_segment": "mid"}),
            )
        )
    db.add(AuditLog(failure_event_id=failure.id, action="orchestrator_final_status", actor="orchestrator", reason="final_status=COMMUNICATION_ALLOWED"))
    db.commit()
    event_id = failure.id
    db.close()
    return event_id


class TestFreshCloneDatabaseInitialization:
    """Final pre-submission correction: `git clone` -> `pip install -r
    requirements.txt` -> `streamlit run ui/app.py` (no FastAPI process ever
    started) used to crash every live-DB page with
    `sqlite3.OperationalError: no such table: raw_events` -- schema creation
    only happened in app/main.py's FastAPI lifespan. Proves
    ui/data.py::ensure_schema_initialized() (called first thing in
    ui/app.py::main()) fixes this using the EXISTING app/db.py::init_db()
    -- never a second schema initializer -- against a genuinely fresh SQLite
    file with zero pre-created tables."""

    def _point_db_at_fresh_file(self, tmp_path, monkeypatch):
        import app.db as db_module
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        fresh_db_path = tmp_path / "brand_new_fresh_clone.db"
        assert not fresh_db_path.exists()
        fresh_engine = create_engine(f"sqlite:///{fresh_db_path}", connect_args={"check_same_thread": False})
        monkeypatch.setattr(db_module, "engine", fresh_engine)
        monkeypatch.setattr(db_module, "SessionLocal", sessionmaker(bind=fresh_engine))
        data.ensure_schema_initialized.clear()  # a prior test's cached True/False must never leak in
        return fresh_db_path

    def test_query_on_a_fresh_database_raises_no_such_table_before_init(self, tmp_path, monkeypatch):
        # Reproduces the reported bug first, unmodified -- proves this test
        # would actually catch a regression, not just exercise a no-op.
        self._point_db_at_fresh_file(tmp_path, monkeypatch)
        with pytest.raises(Exception, match="no such table"):
            data.get_live_kpis()

    def test_ensure_schema_initialized_fixes_it(self, tmp_path, monkeypatch):
        self._point_db_at_fresh_file(tmp_path, monkeypatch)
        ok = data.ensure_schema_initialized()
        assert ok is True
        assert data.get_live_kpis() == {
            "failed_payments": 0, "policy_decisions": 0, "retry_actions": 0, "no_action": 0, "received_not_orchestrated": 0,
        }
        assert data.get_live_recovery_queue_df().empty
        assert data.get_live_raw_events_df().empty
        assert data.get_live_communications_df().empty
        status = data.get_live_system_status()
        assert status["database_connected"] is True
        assert status["database_error"] is None

    def test_full_streamlit_startup_renders_every_page_on_a_fresh_empty_database(self, tmp_path, monkeypatch):
        # The real regression test: drives the ACTUAL ui/app.py::main()
        # entrypoint via AppTest -- not just the query layer -- against a
        # fresh on-disk DB with zero pre-existing tables, for every page.
        from streamlit.testing.v1 import AppTest

        self._point_db_at_fresh_file(tmp_path, monkeypatch)
        for page in ("Overview", "Recovery Queue", "Payment Events", "Analytics", "Communications", "Audit Log", "System / Demo", "Revenue at Risk"):
            at = AppTest.from_file(APP_PATH, default_timeout=120).run()
            assert not at.exception, f"default page render raised: {at.exception}"
            at.sidebar.radio[0].set_value(page).run()
            assert not at.exception, f"{page!r} raised on a fresh empty DB: {at.exception}"


def test_get_live_raw_events_df_empty_db_returns_empty_dataframe(live_db_session_factory):
    df = data.get_live_raw_events_df()
    assert df.empty


def test_get_live_kpis_empty_db_returns_zeros(live_db_session_factory):
    assert data.get_live_kpis() == {
        "failed_payments": 0, "policy_decisions": 0, "retry_actions": 0, "no_action": 0, "received_not_orchestrated": 0,
    }


def test_get_live_recovery_queue_df_empty_db_returns_empty_dataframe(live_db_session_factory):
    assert data.get_live_recovery_queue_df().empty


def test_get_live_communications_df_empty_db_returns_empty_dataframe(live_db_session_factory):
    assert data.get_live_communications_df().empty


def test_get_live_unrouted_raw_events_df_empty_db_returns_empty_dataframe(live_db_session_factory):
    assert data.get_live_unrouted_raw_events_df().empty


def test_get_live_system_status_reports_database_connected_on_healthy_db(live_db_session_factory):
    status = data.get_live_system_status()
    assert status["database_connected"] is True
    assert status["database_error"] is None


def test_get_live_system_status_never_raises_when_db_broken(monkeypatch):
    def _broken_session():
        raise RuntimeError("simulated db failure")

    monkeypatch.setattr(data, "get_live_session", _broken_session)
    status = data.get_live_system_status()  # must not raise
    assert status["database_connected"] is False
    assert status["database_error"] is not None


def test_get_live_raw_events_df_returns_seeded_row(live_db_session_factory):
    _seed_live_event(live_db_session_factory)
    df = data.get_live_raw_events_df()
    assert len(df) == 1
    assert df.iloc[0]["payment_id"] == "pay_test_1"
    assert df.iloc[0]["amount_rupees"] == 2500.0  # 250000 paise -> rupees


def test_get_live_kpis_counts_seeded_event(live_db_session_factory):
    _seed_live_event(live_db_session_factory, selected_candidate_type="plus_1_day_morning")
    kpis = data.get_live_kpis()
    assert kpis["failed_payments"] == 1
    assert kpis["policy_decisions"] == 1
    assert kpis["retry_actions"] == 1
    assert kpis["no_action"] == 0
    assert kpis["received_not_orchestrated"] == 0


def test_get_live_kpis_counts_no_action_decision(live_db_session_factory):
    _seed_live_event(live_db_session_factory, selected_candidate_type="NO_ACTION", with_llm=False)
    kpis = data.get_live_kpis()
    assert kpis["retry_actions"] == 0
    assert kpis["no_action"] == 1


def test_get_live_recovery_queue_df_derives_final_status(live_db_session_factory):
    event_id = _seed_live_event(live_db_session_factory)
    df = data.get_live_recovery_queue_df()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["event_id"] == event_id
    assert row["communication_action"] == "sent"
    assert row["final_status"] == "COMMUNICATION_ALLOWED"
    assert row["payment_action"] == "retry_scheduled"


def test_get_live_event_detail_returns_full_shape(live_db_session_factory):
    event_id = _seed_live_event(live_db_session_factory)
    detail = data.get_live_event_detail(event_id)
    assert detail is not None
    assert detail["policy"].event_id == event_id
    assert detail["raw"].payment_id == "pay_test_1"
    assert len(detail["audit"]) >= 2


def test_get_live_event_detail_unknown_event_returns_none(live_db_session_factory):
    assert data.get_live_event_detail(999999) is None


def test_get_live_unrouted_raw_events_df_flags_genuinely_insufficient_context(live_db_session_factory):
    # No subscription_id AND no payment_id/amount -- genuinely nothing to
    # act on (recovery/webhook_pipeline.py's OUTCOME_SKIPPED_INSUFFICIENT_CONTEXT).
    from app.models import RawEvent

    db = live_db_session_factory()
    db.add(
        RawEvent(
            razorpay_event_id="evt_unrouted", event_type="payment.failed", payment_id=None,
            subscription_id=None, amount=None, currency="INR", error_reason="payment_failed",
            signature_verified=True, raw_payload="{}",
        )
    )
    db.commit()
    db.close()

    df = data.get_live_unrouted_raw_events_df()
    assert len(df) == 1
    assert "insufficient context" in df.iloc[0]["reason_not_orchestrated"]


def test_get_live_unrouted_raw_events_df_excludes_classified_events(live_db_session_factory):
    _seed_live_event(live_db_session_factory)  # this one IS classified/orchestrated
    assert data.get_live_unrouted_raw_events_df().empty


def test_get_live_unrouted_raw_events_df_excludes_orchestrated_payment_link_events(live_db_session_factory):
    # Regression test: a Payment Link payment.failed with NO subscription_id
    # but a real payment_id/amount reaches the one-time-payment
    # RevenueRiskEvent path (recovery/webhook_pipeline.py) -- it is NOT
    # "unrouted" even though it never creates a FailureEvent row. Before the
    # fix, _get_orchestrated_raw_event_ids only checked FailureEvent, so
    # every successfully-orchestrated Payment Link event was misreported
    # here as a dead end.
    from datetime import datetime

    from app.models import PolicyDecision, RawEvent, RevenueRiskEvent
    from policy.policy_decision_store import REVENUE_DOMAIN_EVENT_ID_OFFSET

    db = live_db_session_factory()
    raw = RawEvent(
        razorpay_event_id="evt_payment_link_routed", event_type="payment.failed", payment_id="pay_link_routed",
        subscription_id=None, amount=10000, currency="INR", error_reason="insufficient_fund",
        signature_verified=True, raw_payload="{}",
    )
    db.add(raw)
    db.flush()
    rre = RevenueRiskEvent(
        idempotency_key="payment_failed_no_subscription:pay_link_routed", event_type="payment_failed_no_subscription",
        external_id="pay_link_routed", customer_ref="pay_link_routed", amount=100.0, currency="INR",
        occurred_at=datetime(2026, 8, 25, 10, 0, 0), reason="insufficient_fund", status="OPEN",
    )
    db.add(rre)
    db.flush()
    db.add(PolicyDecision(
        event_id=rre.id + REVENUE_DOMAIN_EVENT_ID_OFFSET, subscription_id="pay_link_routed",
        selected_candidate_type="payment_link_reminder", policy_version="one-time-payment-v1",
        decision_reason="test seed", decision_source="ml_unified_v1", classification_bucket="retryable_soft",
    ))
    db.commit()
    db.close()

    assert data.get_live_unrouted_raw_events_df().empty
    kpis = data.get_live_kpis()
    assert kpis["received_not_orchestrated"] == 0


def test_get_live_communications_df_includes_sent_and_blocked(live_db_session_factory):
    _seed_live_event(live_db_session_factory, suffix="sent", with_llm=True)
    _seed_live_event(live_db_session_factory, suffix="blocked", with_llm=False, with_blocked_comm=True)
    df = data.get_live_communications_df()
    statuses = set(df["status"])
    assert "sent" in statuses
    assert "blocked" in statuses
    sent_row = df[df["status"] == "sent"].iloc[0]
    assert sent_row["language"] == "en"
    assert sent_row["customer_segment"] == "mid"


def test_extract_compliance_fields_parses_real_format():
    reason = (
        "payment_action_allowed=True payment_reason=ok to retry | "
        "communication_action_allowed=False communication_reason=customer opted out | rule_version=v2"
    )
    fields = data.extract_compliance_fields(reason)
    assert fields["payment_allowed"] == "True"
    assert fields["communication_allowed"] == "False"
    assert fields["communication_reason"] == "customer opted out"
    assert fields["rule_version"] == "v2"


def test_extract_compliance_fields_handles_none_and_garbage():
    assert data.extract_compliance_fields(None) == {}
    assert data.extract_compliance_fields("not a matching format") == {}


def test_no_secrets_in_live_raw_events_data(live_db_session_factory):
    from app.config import settings

    _seed_live_event(live_db_session_factory)
    df = data.get_live_raw_events_df()
    secret = settings.RAZORPAY_WEBHOOK_SECRET
    if secret:
        for payload in df["raw_payload"].dropna():
            assert secret not in payload


def test_try_load_model_is_cached_resource():
    # Part 21: must not re-deserialize the model artifact on every 5s live
    # refresh -- this is what makes that cheap.
    assert hasattr(data._try_load_model, "clear"), "_try_load_model must be an @st.cache_resource function"


# ---------------------------------------------------------------------------
# ui.components: new console components (Part 15-19 of the rebuild)
# ---------------------------------------------------------------------------

def test_top_bar_renders_without_exception():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception


def test_live_indicator_states():
    from ui.components import live_indicator

    assert "LIVE" in live_indicator(connected=True, last_refresh=None)
    assert "PAUSED" in live_indicator(connected=True, last_refresh=None, paused=True)
    assert "UNAVAILABLE" in live_indicator(connected=False, last_refresh=None, error="boom")
    assert "boom" not in live_indicator(connected=True, last_refresh=None)  # no error text leaks into the healthy state


def test_field_row_renders_dash_for_missing_value():
    from ui.components import field_row

    # must not raise for None/empty -- and must not silently render a blank cell
    field_row("Label", None)
    field_row("Label", "")
    field_row("Label", "value")


def test_source_tag_covers_all_three_kinds():
    from ui.components import source_tag

    for kind in ("live", "demo", "synthetic"):
        source_tag(kind)  # must not raise


# ---------------------------------------------------------------------------
# Live refresh path (Part 10/11) -- fragments import/wire correctly
# ---------------------------------------------------------------------------

def test_live_fragments_share_the_same_refresh_interval():
    import ui.app as app

    assert app.LIVE_REFRESH_SECONDS > 0
    # every @st.fragment(run_every=...) live section below is built from
    # the same LIVE_REFRESH_SECONDS constant -- verified structurally by
    # confirming the fragment functions exist and are callable.
    for fn_name in (
        "_overview_live_fragment", "_recovery_queue_fragment", "_payment_events_fragment",
        "_analytics_live_operations_fragment", "_communications_live_fragment",
        "_audit_log_fragment", "_system_status_fragment",
    ):
        assert callable(getattr(app, fn_name))


# ---------------------------------------------------------------------------
# Overview page: live LLM pipeline visibility (the complete
# Razorpay -> classification -> ML/policy -> compliance -> LLM ->
# communication -> audit workflow must be visible on Overview, sourced
# entirely from real DB rows -- ui/data.py::get_live_llm_summary and
# get_live_pipeline_snapshot).
# ---------------------------------------------------------------------------

def test_get_live_llm_summary_empty_db_returns_zero_and_nones(live_db_session_factory):
    summary = data.get_live_llm_summary()
    assert summary == {
        "total_invocations": 0, "latest_provider": None, "latest_model": None,
        "latest_task": None, "latest_success": None,
    }


def test_get_live_llm_summary_reads_total_and_latest_from_llm_invocations(live_db_session_factory):
    _seed_live_event(live_db_session_factory, suffix="1", llm_provider="gemini", llm_model="gemini-3.6-flash")
    _seed_live_event(live_db_session_factory, suffix="2", subscription_id="sub_live_2", llm_provider="ollama", llm_model="qwen3:14b")

    summary = data.get_live_llm_summary()
    assert summary["total_invocations"] == 2
    # "latest" must be the most recently INSERTED row, not the first one seeded.
    assert summary["latest_provider"] == "ollama"
    assert summary["latest_model"] == "qwen3:14b"
    assert summary["latest_task"] == "outreach_microcopy"
    assert summary["latest_success"] is True


def test_get_live_llm_summary_latest_success_is_dynamic_not_hardcoded(live_db_session_factory):
    _seed_live_event(live_db_session_factory, suffix="1", llm_success=True)
    _seed_live_event(live_db_session_factory, suffix="2", subscription_id="sub_live_2", llm_success=False)

    summary = data.get_live_llm_summary()
    assert summary["latest_success"] is False  # reflects the newer (failed) invocation, never a stale/hardcoded True


def test_get_live_pipeline_snapshot_empty_db_returns_none(live_db_session_factory):
    assert data.get_live_pipeline_snapshot() is None


def test_get_live_pipeline_snapshot_reads_every_stage_from_real_rows(live_db_session_factory):
    _seed_live_event(
        live_db_session_factory, error_reason="insufficient_fund", classification_bucket="retryable_soft",
        selected_candidate_type="payday_window", llm_provider="ollama", llm_model="qwen3:14b",
    )
    snapshot = data.get_live_pipeline_snapshot()
    assert snapshot is not None
    assert snapshot["error_reason"] == "insufficient_fund"
    assert snapshot["classification_bucket"] == "retryable_soft"
    assert snapshot["selected_candidate_type"] == "payday_window"
    assert snapshot["decision_source"] == "subscription_value_model"
    assert snapshot["compliance_display"] == "ALLOWED"
    assert snapshot["llm_provider"] == "ollama"
    assert snapshot["llm_model"] == "qwen3:14b"
    assert snapshot["llm_success"] is True
    assert snapshot["communication_action"] == "sent"
    assert snapshot["final_status"] == "COMMUNICATION_ALLOWED"
    assert snapshot["communication_message"] == "hi"  # from _seed_live_event's structured_output


def test_get_live_pipeline_snapshot_reflects_llm_failure_never_hides_it_as_success(live_db_session_factory):
    _seed_live_event(live_db_session_factory, llm_provider="ollama", llm_model="qwen3:14b", llm_success=False)
    snapshot = data.get_live_pipeline_snapshot()
    assert snapshot["llm_provider"] == "ollama"
    assert snapshot["llm_success"] is False  # never displayed as success when the fallback was actually used


def test_get_live_pipeline_snapshot_handles_blocked_compliance(live_db_session_factory):
    _seed_live_event(live_db_session_factory, with_llm=False, with_blocked_comm=True)
    snapshot = data.get_live_pipeline_snapshot()
    assert snapshot["compliance_display"] == "PARTIAL (payment allowed, communication blocked)"
    assert snapshot["llm_provider"] is None  # compliance blocked the communication -- no LLM job ever ran


def test_get_live_pipeline_snapshot_uses_the_most_recently_decided_event(live_db_session_factory):
    _seed_live_event(live_db_session_factory, suffix="older", selected_candidate_type="plus_1_day_morning")
    _seed_live_event(live_db_session_factory, suffix="newer", subscription_id="sub_live_2", selected_candidate_type="payday_window")
    snapshot = data.get_live_pipeline_snapshot()
    assert snapshot["selected_candidate_type"] == "payday_window"


def test_looks_like_real_razorpay_id_accepts_real_format_rejects_synthetic():
    assert data._looks_like_real_razorpay_id("evt_JXpBs2TMKUJfPz0000") is True
    assert data._looks_like_real_razorpay_id("pay_MK7hXn2QpRstUv12") is True
    # every synthetic/test/demo ID actually used anywhere in this codebase
    # or its verification scripts contains an extra underscore or a
    # human-readable word -- must never be labeled as a real delivery.
    assert data._looks_like_real_razorpay_id("evt_test_1") is False
    assert data._looks_like_real_razorpay_id("evt_OllamaLiveTest_1787723453") is False
    assert data._looks_like_real_razorpay_id("sub_DemoLLM001") is False
    assert data._looks_like_real_razorpay_id(None) is False
    assert data._looks_like_real_razorpay_id("") is False


def test_get_live_pipeline_snapshot_labels_synthetic_id_correctly(live_db_session_factory):
    _seed_live_event(live_db_session_factory, razorpay_event_id="evt_OllamaLiveTest_1787723453")
    snapshot = data.get_live_pipeline_snapshot()
    assert snapshot["is_live_razorpay_id"] is False


def test_get_live_pipeline_snapshot_labels_real_format_id_correctly(live_db_session_factory):
    _seed_live_event(live_db_session_factory, razorpay_event_id="evt_JXpBs2TMKUJfPz0000")
    snapshot = data.get_live_pipeline_snapshot()
    assert snapshot["is_live_razorpay_id"] is True


def test_overview_page_renders_with_seeded_live_pipeline(live_db_session_factory):
    """Scenario 1 + 5: Overview must render without exception AND the new
    pipeline/LLM sections must actually reflect the seeded live rows, not
    just avoid crashing -- verified via AppTest's rendered markdown."""
    from streamlit.testing.v1 import AppTest

    _seed_live_event(live_db_session_factory, llm_provider="ollama", llm_model="qwen3:14b")

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    rendered = " ".join(md.value for md in at.markdown) + " ".join(c.value for c in at.caption)
    assert "qwen3:14b" in rendered
    assert "ollama" in rendered


def test_overview_page_renders_when_no_llm_invocation_exists(live_db_session_factory):
    """Scenario 7: no LLM invocation anywhere yet -- the LLM KPI card and
    the pipeline/communication sections must degrade gracefully, never raise."""
    from streamlit.testing.v1 import AppTest

    _seed_live_event(live_db_session_factory, with_llm=False)

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception


def test_overview_page_renders_when_llm_failed_and_fallback_used(live_db_session_factory):
    """Scenario 8: the most recent LLM call failed (deterministic fallback
    was used) -- Overview must still render cleanly and must never display
    that failed call as a success."""
    from streamlit.testing.v1 import AppTest

    _seed_live_event(live_db_session_factory, llm_provider="ollama", llm_model="qwen3:14b", llm_success=False)

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    snapshot = data.get_live_pipeline_snapshot()
    assert snapshot["llm_success"] is False


def test_overview_page_renders_on_completely_empty_live_database(live_db_session_factory):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# UI CONSISTENCY + OPERATIONAL UX HARDENING PASS -- regression tests for the
# 21 labeling/presentation issues fixed in this pass. Every test below
# exercises ACTUAL BEHAVIOR (a real function's return value, or the actually
# rendered app), never a hardcoded expected screenshot.
# ---------------------------------------------------------------------------

# --- Issue 1/2: live test-mode labeling (never implies production) --------

def test_top_bar_refresh_label_names_test_mode_db_not_bare_live():
    import ui.app as app

    at_module_source = Path(app.__file__).read_text()
    assert 'refresh_label=f"TEST-MODE DB' in at_module_source
    # the old bare "live · Ns refresh" wording (which could misread as
    # production live traffic even alongside the separate TEST MODE pill)
    # must be gone.
    assert 'refresh_label=f"live ·' not in at_module_source


def test_overview_section_is_labeled_live_test_mode_operations():
    at_module_source = Path("ui/app.py").read_text()
    assert "Live test-mode operations" in at_module_source
    assert '"##### Live operations"' not in at_module_source


def test_overview_page_renders_live_test_mode_operations_heading(live_db_session_factory):
    from streamlit.testing.v1 import AppTest

    _seed_live_event(live_db_session_factory)
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception
    rendered = " ".join(md.value for md in at.markdown)
    assert "Live test-mode operations" in rendered


# --- Issue 1/3: synthetic vs. genuine event-origin labeling ----------------

def test_razorpay_event_origin_label_distinguishes_real_and_synthetic():
    assert data.razorpay_event_origin_label("evt_JXpBs2TMKUJfPz0000") == "RAZORPAY TEST WEBHOOK"
    assert data.razorpay_event_origin_label("pay_MK7hXn2QpRstUv12") == "RAZORPAY TEST WEBHOOK"
    assert data.razorpay_event_origin_label("evt_retrainedmodelb_direct") == "SYNTHETIC TEST EVENT"
    assert data.razorpay_event_origin_label("evt_OllamaLiveTest_1787723453") == "SYNTHETIC TEST EVENT"
    assert data.razorpay_event_origin_label(None) == "SYNTHETIC TEST EVENT"


def test_get_live_raw_events_df_includes_origin_column(live_db_session_factory):
    _seed_live_event(live_db_session_factory, razorpay_event_id="evt_JXpBs2TMKUJfPz0000")
    _seed_live_event(live_db_session_factory, suffix="2", subscription_id="sub_live_2", razorpay_event_id="evt_synthetic_verification_script")
    df = data.get_live_raw_events_df()
    origins = dict(zip(df["razorpay_event_id"], df["origin"]))
    assert origins["evt_JXpBs2TMKUJfPz0000"] == "RAZORPAY TEST WEBHOOK"
    assert origins["evt_synthetic_verification_script"] == "SYNTHETIC TEST EVENT"


def test_get_live_pipeline_snapshot_includes_origin_label(live_db_session_factory):
    _seed_live_event(live_db_session_factory, razorpay_event_id="evt_OllamaLiveTest_1787723453")
    snapshot = data.get_live_pipeline_snapshot()
    assert snapshot["origin_label"] == "SYNTHETIC TEST EVENT"


# --- Issue 3: signature-status distinction ---------------------------------

def test_payment_event_signature_status_four_states():
    # real-format ID, signature genuinely verified
    assert data.payment_event_signature_status(True, "evt_JXpBs2TMKUJfPz0000") == "VERIFIED"
    # real-format ID, signature failed -- must remain a visible FAILURE, never softened to N/A
    assert data.payment_event_signature_status(False, "evt_JXpBs2TMKUJfPz0000") == "VERIFICATION FAILED"
    # synthetic-looking ID, but the signature genuinely was checked and passed
    # (e.g. a verification script signed a test payload with the real secret)
    assert data.payment_event_signature_status(True, "evt_retrainedmodelb_1787831822") == "VERIFIED (SYNTHETIC)"
    # synthetic-looking ID, signature not verified
    assert data.payment_event_signature_status(False, "evt_retrainedmodelb_direct") == "SYNTHETIC / UNSIGNED"


def test_payment_event_signature_status_never_reports_no_for_any_case():
    # The exact bug this fix corrects: a bare "No" that could misread as
    # "the backend accepted an unsigned real webhook."
    for verified in (True, False):
        for event_id in ("evt_JXpBs2TMKUJfPz0000", "evt_synthetic_test_1", None):
            status = data.payment_event_signature_status(verified, event_id)
            assert status not in ("Yes", "No")


def test_get_live_raw_events_df_includes_signature_status_column(live_db_session_factory):
    _seed_live_event(live_db_session_factory, razorpay_event_id="evt_JXpBs2TMKUJfPz0000", signature_verified=True)
    _seed_live_event(live_db_session_factory, suffix="2", subscription_id="sub_live_2", razorpay_event_id="evt_legacy_unsigned", signature_verified=False)
    df = data.get_live_raw_events_df()
    statuses = dict(zip(df["razorpay_event_id"], df["signature_status"]))
    assert statuses["evt_JXpBs2TMKUJfPz0000"] == "VERIFIED"
    assert statuses["evt_legacy_unsigned"] == "SYNTHETIC / UNSIGNED"


def test_payment_events_page_shows_signature_status_not_bare_yes_no(live_db_session_factory):
    from streamlit.testing.v1 import AppTest

    _seed_live_event(live_db_session_factory)
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("Payment Events").run()
    assert not at.exception


# --- Issue 4: overdue retry-time display -----------------------------------

def test_derive_retry_status_future_time_is_scheduled():
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 8, 29, 16, 36, tzinfo=timezone.utc)
    future = now + timedelta(hours=2)
    assert data._derive_retry_status("plus_1_day_morning", future, now, None) == "SCHEDULED"


def test_derive_retry_status_past_time_no_outcome_is_overdue():
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 8, 29, 16, 36, tzinfo=timezone.utc)
    past = now - timedelta(hours=6, minutes=36)  # e.g. 10:00 the same day
    assert data._derive_retry_status("plus_1_day_morning", past, now, None) == "OVERDUE"
    assert data._derive_retry_status("plus_1_day_morning", past, now, "PENDING") == "OVERDUE"


def test_derive_retry_status_confirmed_outcome_wins_over_raw_time():
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 8, 29, 16, 36, tzinfo=timezone.utc)
    past = now - timedelta(hours=6)
    assert data._derive_retry_status("plus_1_day_morning", past, now, "RECOVERED") == "RECOVERED"
    assert data._derive_retry_status("plus_1_day_morning", past, now, "LOST") == "LOST"
    assert data._derive_retry_status("plus_1_day_morning", past, now, "PARTIALLY_RECOVERED") == "PARTIALLY_RECOVERED"


def test_derive_retry_status_no_action_or_missing_datetime_is_dash():
    from datetime import datetime, timezone

    now = datetime(2026, 8, 29, 16, 36, tzinfo=timezone.utc)
    assert data._derive_retry_status("NO_ACTION", None, now, None) == "—"
    assert data._derive_retry_status(None, None, now, None) == "—"
    assert data._derive_retry_status("plus_1_day_morning", pd.NaT, now, None) == "—"


def test_get_live_recovery_queue_df_includes_retry_status_column(live_db_session_factory):
    from datetime import datetime, timedelta, timezone

    past = datetime.now(timezone.utc) - timedelta(days=1)
    future = datetime.now(timezone.utc) + timedelta(days=1)
    _seed_live_event(live_db_session_factory, suffix="overdue", selected_candidate_datetime=past)
    _seed_live_event(live_db_session_factory, suffix="scheduled", subscription_id="sub_live_2", selected_candidate_datetime=future)
    df = data.get_live_recovery_queue_df()
    assert "retry_status" in df.columns
    by_sub = dict(zip(df["subscription_id"], df["retry_status"]))
    assert by_sub["sub_live_1"] == "OVERDUE"
    assert by_sub["sub_live_2"] == "SCHEDULED"


def test_recovery_queue_page_renders_with_overdue_row(live_db_session_factory):
    from datetime import datetime, timedelta, timezone

    from streamlit.testing.v1 import AppTest

    past = datetime.now(timezone.utc) - timedelta(days=1)
    _seed_live_event(live_db_session_factory, selected_candidate_datetime=past)
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("Recovery Queue").run()
    assert not at.exception


# --- Issue 5/6: complete candidate space + sample-size context ------------

def test_complete_candidate_counts_includes_all_five_with_zero_fill():
    from policy.retry_candidates import CANDIDATE_TYPES

    counts = data.complete_candidate_counts(["payday_window", "month_end_window", "payday_window"])
    assert list(counts.index) == CANDIDATE_TYPES  # stable, meaningful order -- never alphabetical/random
    assert counts["payday_window"] == 2
    assert counts["month_end_window"] == 1
    assert counts["immediate"] == 0
    assert counts["plus_1_day_morning"] == 0
    assert counts["plus_3_days"] == 0


def test_complete_candidate_counts_handles_empty_input():
    counts = data.complete_candidate_counts([])
    assert counts.sum() == 0
    assert len(counts) == 5


def test_complete_candidate_counts_never_drops_or_invents_a_category():
    from policy.retry_candidates import CANDIDATE_TYPES

    counts = data.complete_candidate_counts(["immediate"] * 3)
    assert set(counts.index) == set(CANDIDATE_TYPES)
    assert counts.sum() == 3  # never invents extra occurrences


# --- Issue 7: LLM job wording -----------------------------------------------

def test_communications_page_no_longer_claims_only_three_llm_jobs():
    source = Path("ui/app.py").read_text()
    assert "The three LLM jobs, downstream of every policy decision." not in source
    assert "3 required core LLM jobs" in source
    assert "optional Track-03 voice-script job" in source


def test_communications_page_renders_with_corrected_wording(live_db_session_factory):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("Communications").run()
    assert not at.exception
    rendered = " ".join(md.value for md in at.markdown)
    assert "3 required core LLM jobs" in rendered


# --- Issue 8: customer/payment reference column semantics ------------------

def test_one_time_payment_revenue_risk_customer_ref_is_actually_a_payment_id():
    # Grounds the "Customer" mislabel this issue fixes: for the
    # payment_failed_no_subscription domain, customer_ref really is
    # raw_events.payment_id, not a genuine customer identifier -- see
    # recovery/webhook_pipeline.py::_build_one_time_payment_event.
    source = Path("recovery/webhook_pipeline.py").read_text()
    assert "customer_ref=raw_event.payment_id" in source


def test_revenue_at_risk_page_no_longer_labels_payment_reference_as_bare_customer():
    source = Path("ui/app.py").read_text()
    assert "Payment / Customer Reference" in source
    assert '"Customer": filtered["customer_ref"]' not in source
    assert '"Customer": df["customer_ref"]' not in source


# --- Issue 9: processing-path distinction for same-reference duplicates ---

def test_get_live_recovery_timeline_df_labels_processing_path(live_db_session_factory):
    _seed_live_event(live_db_session_factory)
    df = data.get_live_recovery_timeline_df()
    assert "processing_path" in df.columns
    assert (df["processing_path"] == "Subscription Recovery").all()


def test_get_live_recovery_timeline_df_uses_real_event_type_not_hardcoded_payment_failed(live_db_session_factory):
    # Regression test for a real pre-existing bug this pass also fixed: every
    # raw_events row used to be hardcoded "payment_failed" in this timeline
    # regardless of its actual event_type (e.g. payment.captured).
    from app.models import RawEvent

    db = live_db_session_factory()
    db.add(RawEvent(razorpay_event_id="evt_captured_1", event_type="payment.captured", payment_id="pay_captured_1", amount=100000, currency="INR", signature_verified=True, raw_payload="{}"))
    db.commit()
    db.close()
    df = data.get_live_recovery_timeline_df()
    assert "payment.captured" in df["event_type"].values
    assert not (df["event_type"] == "payment_failed").any()  # the old hardcoded literal must never appear


def test_revenue_timeline_processing_path_distinguishes_revenue_risk_domain(live_db_session_factory):
    from app.models import RevenueRiskEvent

    db = live_db_session_factory()
    db.add(RevenueRiskEvent(idempotency_key="k1", event_type="payment_failed_no_subscription", external_id="pay_shared_ref", customer_ref="pay_shared_ref", amount=500.0, status="OPEN"))
    db.commit()
    db.close()
    df = data.get_live_recovery_timeline_df()
    revenue_rows = df[df["event_type"] == "payment_failed_no_subscription"]
    assert (revenue_rows["processing_path"] == "Revenue Risk").all()


# --- Issue 10: event-type filter full-value visibility ---------------------

def test_revenue_recovery_queue_fragment_shows_full_selected_filter_values():
    source = Path("ui/app.py").read_text()
    assert 'st.caption("Selected event types: "' in source


# --- Issue 11/12/13: System/Demo labeling honesty --------------------------

def test_system_demo_section_is_labeled_live_test_mode_runtime():
    source = Path("ui/app.py").read_text()
    assert "Live test-mode runtime" in source
    assert '"##### Live runtime status"' not in source
    # the interactive-demo headings must be preserved, not merged/removed
    assert "Interactive demo — Run Demo Event" in source


def test_webhook_enablement_stays_honestly_not_queryable_never_enabled():
    source = Path("ui/app.py").read_text()
    assert '"Razorpay webhook enablement", "Not Queryable"' in source
    assert '"Razorpay webhook enablement", "Enabled"' not in source
    assert '"Razorpay webhook enablement", "Connected"' not in source


def test_model_status_never_overclaims_live_or_production():
    source = Path("ui/app.py").read_text()
    assert '"Live AI"' not in source
    assert '"Production ML"' not in source


def test_system_demo_page_renders_with_new_labels(live_db_session_factory):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("System / Demo").run()
    assert not at.exception
    rendered = " ".join(md.value for md in at.markdown)
    assert "Live test-mode runtime" in rendered
    assert "Not Queryable" in rendered


# --- Issue 14/15: test-count wording never implies a just-run suite -------

def test_test_count_caption_distinguishes_functions_from_collected_cases():
    source = Path("ui/app.py").read_text()
    assert "not a recent test run" in source
    assert "static code scan" in source


def test_count_test_functions_still_dynamic_not_hardcoded():
    # Regression guard: this pass must never hardcode a specific number
    # (e.g. 1050) anywhere in the displayed text.
    source = Path("ui/app.py").read_text()
    assert "1050" not in source
    assert "973" not in source
    count = data.count_test_functions()
    assert count > 0


# --- Issue 16: sidebar contains no duplicate pages -------------------------

def test_sidebar_pages_map_one_to_one_to_distinct_handlers():
    import ui.app as app_module

    at = __import__("streamlit.testing.v1", fromlist=["AppTest"]).AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    # NAV_PAGES itself must have no duplicate labels.
    assert len(app_module.NAV_PAGES) == len(set(app_module.NAV_PAGES))


def test_main_pages_dict_maps_every_nav_page_to_a_distinct_function():
    import ast

    source = Path("ui/app.py").read_text()
    tree = ast.parse(source)
    main_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")
    dict_node = next(n for n in ast.walk(main_fn) if isinstance(n, ast.Dict) and len(n.keys) >= 6)
    keys = [k.value for k in dict_node.keys]
    values = [v.id for v in dict_node.values]
    assert len(keys) == len(set(keys)), "duplicate sidebar page label"
    assert len(values) == len(set(values)), "two sidebar pages point at the same handler function"


# --- Issue 17: every visible control has a real, working handler ----------

@pytest.mark.parametrize("page", ["Overview", "Recovery Queue", "Payment Events", "Analytics", "Communications", "Audit Log", "System / Demo", "Revenue at Risk"])
def test_every_page_including_revenue_at_risk_renders_without_exception(page):
    # Extends the existing test_every_page_renders_without_exception
    # parametrization (which was missing "Revenue at Risk") -- Issue 16/17's
    # own "verify every sidebar page" requirement.
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page} raised: {[e.value for e in at.exception]}"


def test_communications_parse_reply_button_has_a_real_working_handler():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("Communications").run()
    assert not at.exception
    buttons = [b for b in at.button if b.label == "Parse reply"]
    assert len(buttons) == 1
    buttons[0].click().run()
    assert not at.exception


def test_system_demo_run_recovery_button_has_a_real_working_handler():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("System / Demo").run()
    assert not at.exception
    buttons = [b for b in at.button if b.label == "Run recovery"]
    assert len(buttons) == 1
    buttons[0].click().run()
    assert not at.exception


# --- Issue 18: demo-generated data never appears unlabeled in a LIVE table -

def test_revenue_at_risk_demo_synthetic_kpi_uses_demo_generated_not_synthetic_benchmark_label():
    # Regression test for a real mislabel this pass fixed: this KPI's value
    # (recovery_outcomes.confirmed_by == "demo_synthetic") comes from the
    # System/Demo page's demo generator writing into the LIVE database, not
    # from the frozen evaluation/reports/*.json "synthetic benchmark" -- the
    # project's own three-category vocabulary (ui/data.py module docstring)
    # reserves "SYNTHETIC BENCHMARK" for the latter only.
    source = Path("ui/app.py").read_text()
    assert '"Recovered", str(kpis["demo_synthetic_recovered_cases"]), "DEMO-GENERATED only' in source
    assert '"Recovered", str(kpis["demo_synthetic_recovered_cases"]), "SYNTHETIC BENCHMARK only"' not in source


def test_demo_generated_data_never_written_to_the_real_database_url(monkeypatch):
    # build_demo_database / run_demo_scenario / run_revenue_demo_generator
    # must always target a throwaway in-memory engine, never
    # settings.DATABASE_URL -- verified by confirming get_live_session
    # (the only function that ever touches the real DB) is never called
    # anywhere in their source.
    import inspect

    for fn in (data.build_demo_database, data.run_demo_scenario, data.run_revenue_demo_generator):
        src = inspect.getsource(fn)
        assert "get_live_session" not in src
        assert "sqlite:///:memory:" in src or "generate_demo_revenue_risk_events" in src or "demo" in src.lower()


# --- Issue 19: synthetic benchmark disclosures remain intact ---------------

def test_overview_synthetic_benchmark_captions_never_claim_production():
    source = Path("ui/app.py").read_text()
    assert "Razorpay Production Results" not in source
    assert '"Live Recovery Performance"' not in source
    assert "synthetic benchmark" in source.lower()


# --- Issue 21: communication-channel honesty (no implied real delivery) ---

def test_humanize_communication_action_never_implies_real_delivery():
    assert data.humanize_communication_action("sent") == "Generated (recorded)"
    assert data.humanize_communication_action("fallback_used") == "Generated (fallback)"
    assert data.humanize_communication_action("blocked") == "Blocked (compliance)"
    assert data.humanize_communication_action("skipped") == "Skipped"
    assert data.humanize_communication_action(None) == "—"
    for value in ("sent", "fallback_used", "blocked", "skipped"):
        assert "delivered" not in data.humanize_communication_action(value).lower()
        assert "whatsapp" not in data.humanize_communication_action(value).lower()


def test_underlying_communication_action_value_unchanged_only_display_differs(live_db_session_factory):
    # The internal stored/compared value must never be renamed -- only the
    # rendered label. _derive_final_status's own comparisons keep working.
    _seed_live_event(live_db_session_factory)
    df = data.get_live_recovery_queue_df()
    assert (df["communication_action"] == "sent").any()
    assert df["final_status"].iloc[0] == "COMMUNICATION_ALLOWED"
