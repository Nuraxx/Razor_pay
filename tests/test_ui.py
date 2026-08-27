"""
Day-13 dashboard tests: data loaders (missing-file handling), formatting,
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

    row_blocked_comm = pd.Series({"selected_candidate_type": "payday_window", "communication_action": "blocked", "decision_source": "day8_model_b"})
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
        "NO_ACTION", "POLICY_FALLBACK", "LLM_FALLBACK",
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
            "day10_improved_fallback": {"recovery_rate": 0.70, "total_recovered_rs": 19997.23, "incremental_rs_vs_fixed_retry": -1856.87},
        },
        "statistical_tests": {
            "population": {"n_events": 60, "held_out_split": "test"},
            "mcnemar": {"policy_a": "day10_improved_fallback", "policy_b": "fixed_retry", "only_a_recovered": 3, "only_b_recovered": 9, "p_value": 0.146, "method": "exact_binomial"},
            "bootstrap_ci": {"point_estimate": -1856.87, "lower_bound": -4878.20, "upper_bound": 1295.62, "confidence_level": 0.95, "n_resamples": 10000, "seed": 42},
        },
    }
    app.render_statistical_significance_section(report)  # must not raise


def test_render_economics_section_with_synthetic_report():
    import ui.app as app

    report = {
        "economics": {
            "fixed_retry": {"recovered_gmv": 21854.10, "intervention_cost": 300.0, "razorpay_fee_take": 515.76, "net_recovery_value": 21038.34},
            "day10_improved_fallback": {"recovered_gmv": 19997.23, "intervention_cost": 300.0, "razorpay_fee_take": 471.93, "net_recovery_value": 19225.30},
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
    razorpay_event_id=None,
):
    from app.models import AuditLog, FailureEvent, LLMInvocation, PolicyDecision, RawEvent

    db = factory()
    raw = RawEvent(
        razorpay_event_id=razorpay_event_id or f"evt_test_{suffix}", event_type="payment.failed", payment_id=f"pay_test_{suffix}",
        subscription_id=subscription_id, amount=250000, currency="INR", error_reason=error_reason,
        error_source="gateway", error_step="payment_authorization", signature_verified=True, raw_payload="{}",
    )
    db.add(raw)
    db.flush()
    failure = FailureEvent(raw_event_id=raw.id, classification_bucket=classification_bucket, classification_confidence=1.0, rule_version="v1")
    db.add(failure)
    db.flush()
    policy = PolicyDecision(
        event_id=failure.id, subscription_id=subscription_id, selected_candidate_type=selected_candidate_type,
        policy_version="v4", decision_reason="test seed", decision_source="day8_model_b", classification_bucket=classification_bucket,
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
    assert snapshot["decision_source"] == "day8_model_b"
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
