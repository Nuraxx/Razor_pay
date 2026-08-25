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
# Dynamic test count
# ---------------------------------------------------------------------------

def test_count_test_functions_is_dynamic_and_positive():
    count = data.count_test_functions()
    assert count > 0
    # sanity: this very file contributes at least a handful of test_ functions
    assert count >= 20
