"""
Day-14 dashboard entry point -- "Adaptive Recovery: AI-assisted payment
recovery" operations console.

    ./venv/bin/streamlit run ui/app.py

The UI sits entirely ON TOP of the existing system: it never re-implements
classification, scoring, compliance, or LLM logic -- every page either (a)
reads a frozen SYNTHETIC BENCHMARK evaluation report Days 6-10 already
wrote, (b) queries the REAL, webhook-backed SQLite database
(settings.DATABASE_URL) read-only for LIVE operational data, or (c) calls
recovery/orchestrator.py::orchestrate_recovery (Day 12, unmodified) against
a throwaway in-memory DB to produce DEMO-GENERATED data for the interactive
demo. See ui/data.py's module docstring for the full three-way distinction,
which is never blurred anywhere below -- every section on this page carries
an explicit LIVE / DEMO-GENERATED / SYNTHETIC BENCHMARK tag.

No ML model is trained here, Day-8 Model B is used exactly as trained,
Day-10 policy / Day-12 compliance logic is unmodified, and the three Day-11
LLM jobs are unmodified. No live Razorpay payment or real customer message
is ever sent from this UI -- it only ever READS the database a real
webhook delivery (via app/main.py, unmodified) may have already written.
"""
from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run ui/app.py` inserts this script's own directory (ui/) into
# sys.path (streamlit/web/bootstrap.py::_fix_sys_path), so files under ui/
# shadow real top-level packages that share their bare filename: ui/app.py
# vs. the app/ package (`from app.db import Base` raises "'app' is not a
# package"), and ui/data.py vs. the data/ namespace package (data/ has no
# __init__.py, so a *later* sys.path entry's regular module always wins
# over its namespace portion -- `from data.generate_synthetic_dataset
# import ...` inside model/candidate_preprocessing.py resolves to ui/data.py
# instead). Nothing here needs ui/ importable by bare top-level names, so
# dropping it from sys.path (and making sure the real project root is
# present) fixes this at the source -- must run first, before any
# project-internal import below.
_UI_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = str(_UI_DIR.parent)


def _is_ui_dir(path_entry: str) -> bool:
    try:
        return bool(path_entry) and Path(path_entry).resolve() == _UI_DIR
    except OSError:
        return False


sys.path[:] = [p for p in sys.path if not _is_ui_dir(p)]
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import html
import json
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui.data as data
from ui.components import (
    empty_state,
    field_group,
    field_row,
    inject_css,
    kpi_card,
    live_indicator,
    mono,
    money,
    render_status_badge,
    section_header,
    sidebar_brand,
    sidebar_status_block,
    source_tag,
    timeline_step,
    top_bar,
)

st.set_page_config(page_title="Adaptive Recovery", page_icon="💳", layout="wide")

LIVE_REFRESH_SECONDS = 5  # Part 10/11/31: smallest reasonable live-fragment boundary

NAV_PAGES = ["Overview", "Recovery Queue", "Payment Events", "Analytics", "Communications", "Audit Log", "System / Demo", "Revenue at Risk"]
NAV_ICONS = {
    "Overview": "", "Recovery Queue": "", "Payment Events": "", "Analytics": "",
    "Communications": "", "Audit Log": "", "System / Demo": "", "Revenue at Risk": "",
}
POLICY_LABELS = {
    "fixed_retry": "Fixed Retry", "rule_based": "Rule-Based", "day8_model_b_alone": "Day-8 Model B",
    "day9_original_fallback": "Day-9 Policy", "day10_improved_fallback": "Day-10 Policy", "oracle_policy": "Oracle",
}
# The one headline baseline comparison the statistical-significance and
# economics sections focus on (matches
# evaluation/evaluate_decision_engine_v4.py's DEPLOYED_POLICY_NAME / HEADLINE_BASELINE_NAME).
DEPLOYED_POLICY_KEY = "day10_improved_fallback"
BASELINE_POLICY_KEY = "fixed_retry"


# ---------------------------------------------------------------------------
# Live-fragment bookkeeping (Part 11): every live section shares this
# tiny helper so its refresh pill reflects a REAL just-performed query,
# never an assumed-fresh state.
# ---------------------------------------------------------------------------

def _run_live(section_key: str, fn, *args, **kwargs):
    """Runs fn, records success/failure + timestamp in session_state for
    _render_live_pill, and NEVER lets a live-query exception escape into
    the page (Part 22: a failed live query must show a compact status, not
    crash the dashboard)."""
    now = datetime.now()
    try:
        result = fn(*args, **kwargs)
        st.session_state[f"_live_{section_key}"] = {"ok": True, "at": now, "error": None}
        return result
    except Exception as exc:  # noqa: BLE001 -- a live UI query must never propagate into a crashed page
        st.session_state[f"_live_{section_key}"] = {"ok": False, "at": now, "error": str(exc)}
        return None


def _render_live_pill(section_key: str) -> None:
    state = st.session_state.get(f"_live_{section_key}", {"ok": True, "at": None, "error": None})
    st.markdown(
        live_indicator(connected=state["ok"], last_refresh=state["at"], error=state["error"]),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Shared: policy comparison charts (SYNTHETIC BENCHMARK only -- used by
# Overview + Analytics). UNCHANGED from the prior evaluation-fidelity pass:
# reads the frozen report, never recomputes a number.
# ---------------------------------------------------------------------------

def render_policy_comparison_charts(report: dict) -> None:
    latent = report.get("latent_economic", {})
    realized = report.get("realized_counterfactual", {})
    policies = [p for p in POLICY_LABELS if p in latent]
    labels = [POLICY_LABELS[p] for p in policies]

    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.markdown("**Latent expected value — synthetic benchmark**")
        values = [latent[p]["total_latent_value_rs"] for p in policies]
        fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#2F4CDD", text=[f"₹{v:,.0f}" for v in values], textposition="outside"))
        fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white", yaxis_title="₹ (latent, test set)")
        st.plotly_chart(fig, width='stretch')
    with col2:
        st.markdown("**Realized counterfactual recovery — synthetic benchmark**")
        values = [realized[p]["total_recovered_rs"] for p in policies]
        fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#1D9A6C", text=[f"₹{v:,.0f}" for v in values], textposition="outside"))
        fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white", yaxis_title="₹ (realized, test set)")
        st.plotly_chart(fig, width='stretch')

    st.caption(
        "**Latent expected value** is the synthetic simulation's own ground-truth expectation for the candidate each policy "
        "selected. **Realized counterfactual recovery** is a single stochastic sampled outcome under that same simulation. "
        "They are deliberately shown separately, never combined into one number — see README \"Day 9\" for why conflating "
        "them would overstate confidence. Neither reflects real Razorpay production performance."
    )


def render_statistical_significance_section(report: dict) -> None:
    """Shows the McNemar's-test / bootstrap-CI results
    `evaluate_decision_engine_v4.py::summarize_statistical_tests` already
    computed and wrote into the report -- never recomputed here, and never
    hidden if the result is negative. UNCHANGED evaluation methodology."""
    stats = (report or {}).get("statistical_tests") or {}
    mc = stats.get("mcnemar")
    bs = stats.get("bootstrap_ci")
    if not mc or not bs:
        empty_state("No statistical test results available — run evaluation/evaluate_decision_engine_v4.py to generate them.")
        return

    realized = report.get("realized_counterfactual", {})
    deployed = realized.get(DEPLOYED_POLICY_KEY, {})
    baseline = realized.get(BASELINE_POLICY_KEY, {})

    rate_delta_pp = (deployed.get("recovery_rate", 0.0) - baseline.get("recovery_rate", 0.0)) * 100
    rs_delta = deployed.get("incremental_rs_vs_fixed_retry", bs["point_estimate"])

    cols = st.columns(4, gap="small")
    with cols[0]:
        kpi_card("Recovery-rate delta", f"{rate_delta_pp:+.1f} pp", f"{POLICY_LABELS.get(DEPLOYED_POLICY_KEY, DEPLOYED_POLICY_KEY)} vs Fixed Retry")
    with cols[1]:
        kpi_card("₹ recovered delta", money(rs_delta), f"{bs['confidence_level']:.0%} CI [{money(bs['lower_bound'])}, {money(bs['upper_bound'])}]")
    with cols[2]:
        kpi_card("McNemar p-value", f"{mc['p_value']:.4f}", f"b={mc['only_a_recovered']} c={mc['only_b_recovered']} ({mc['method']})")
    with cols[3]:
        kpi_card("Test-set size", str(stats.get("population", {}).get("n_events", "—")), "held-out synthetic events")

    significant = mc["p_value"] < 0.05
    ci_excludes_zero = bs["lower_bound"] > 0 or bs["upper_bound"] < 0
    direction = "an improvement over" if rs_delta > 0 else "a regression versus" if rs_delta < 0 else "no difference from"
    verdict = "statistically significant" if significant else "NOT statistically significant"
    st.caption(
        f"The deployed policy shows {direction} Fixed Retry on this held-out synthetic test set "
        f"(₹{rs_delta:+,.2f}), and this difference is **{verdict} at p<0.05** (McNemar {mc['method']}, "
        f"p={mc['p_value']:.4f}; the bootstrap {bs['confidence_level']:.0%} CI "
        f"{'excludes' if ci_excludes_zero else 'includes'} zero). This is a SYNTHETIC COUNTERFACTUAL "
        "EVALUATION — these figures quantify uncertainty within this held-out test split only and do not "
        "establish real-world Razorpay production superiority."
    )


def render_economics_section(report: dict) -> None:
    """Merchant-recovered GMV, this project's own intervention cost, and
    Razorpay's disclosed fee take, kept as separate fields per the
    specification -- never blended into one number (policy/economics.py).
    UNCHANGED evaluation methodology."""
    if not report or "economics" not in report:
        empty_state("No economics data available — run evaluation/evaluate_decision_engine_v4.py to generate it.")
        return

    econ = report["economics"]
    deployed = econ.get(DEPLOYED_POLICY_KEY, {})
    baseline = econ.get(BASELINE_POLICY_KEY, {})

    st.caption(f"Deployed policy ({POLICY_LABELS.get(DEPLOYED_POLICY_KEY, DEPLOYED_POLICY_KEY)}), synthetic held-out test set:")
    cols = st.columns(4, gap="small")
    with cols[0]:
        kpi_card("Merchant Recovery", money(deployed.get("recovered_gmv")), "recovered GMV")
    with cols[1]:
        kpi_card("Intervention Cost", money(deployed.get("intervention_cost")), "policy/costs.py")
    with cols[2]:
        kpi_card("Razorpay Fee Take", money(deployed.get("razorpay_fee_take")), "~2% + 18% GST, gross")
    with cols[3]:
        kpi_card("Net Recovery Value", money(deployed.get("net_recovery_value")), "GMV − cost − fee")

    st.caption(
        f"For comparison, Fixed Retry: recovered GMV {money(baseline.get('recovered_gmv'))}, "
        f"intervention cost {money(baseline.get('intervention_cost'))}, Razorpay fee take "
        f"{money(baseline.get('razorpay_fee_take'))}, net recovery value {money(baseline.get('net_recovery_value'))}. "
        "Merchant-recovered GMV and Razorpay's own fee take are two separate numbers, never combined into one "
        "blended metric. Fee take uses the specification's disclosed ~2% + 18% GST domestic card rate, applied "
        "uniformly (Razorpay's own public materials are inconsistent on UPI's exact rate — see README). Figures "
        "are part of the same SYNTHETIC COUNTERFACTUAL EVALUATION as the rest of this page."
    )


def render_baseline_definitions_section(report: dict) -> None:
    """Makes the Fixed Retry / Rule-Based baseline definitions visible, with
    real numbers from the report -- never fabricated. UNCHANGED."""
    b1, b2 = st.columns(2, gap="small")
    with b1:
        st.markdown("**Fixed Retry** — silent auto-retry, same channel, no communication:")
        st.markdown("`T+1` → `T+2` → `T+3`, then gives up")
        attempts = ((report or {}).get("contact_and_intervention_metrics") or {}).get(BASELINE_POLICY_KEY, {}).get("average_retry_attempts")
        if attempts is not None:
            st.caption(f"Average {attempts:.2f} attempts/event on this held-out test set (1 if recovered at T+1, up to 3 otherwise).")
    with b2:
        st.markdown("**Rule-Based** — hand-coded, deterministic:")
        st.markdown("payday-window retry + WhatsApp nudge + follow-up (+3 days), then stop")
        rb_contact = ((report or {}).get("contact_and_intervention_metrics") or {}).get("rule_based", {})
        if rb_contact:
            st.caption(f"Contact rate {rb_contact.get('customer_contact_rate', 0):.0%}, {rb_contact.get('average_contacts_per_contacted_subscription', 0):.1f} contacts/contacted subscription on this held-out test set.")


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

def _render_latest_recovery_pipeline(snapshot: dict | None) -> None:
    """Compact, top-to-bottom rendering of the full live pipeline for the
    single most recent orchestrated event: Razorpay -> payment.failed ->
    classification -> ML/policy -> compliance -> LLM -> communication ->
    final status. Every value comes straight from `snapshot` (built by
    ui/data.py::get_live_pipeline_snapshot off real DB rows) -- no stage
    here is ever invented when the underlying data is missing; field_row
    already renders "—" for None."""
    st.markdown("###### Latest recovery event")
    if snapshot is None:
        empty_state(
            "No live event has reached classification/policy yet. Once a real Razorpay payment.failed "
            "webhook with a subscription_id is orchestrated, its full pipeline appears here automatically."
        )
        return

    if snapshot["is_live_razorpay_id"]:
        id_label = mono(snapshot["razorpay_event_id"])
    else:
        id_label = f'{mono(snapshot["razorpay_event_id"])} <span class="ar-tag ar-tag-demo">SYNTHETIC / TEST ID</span>'
    st.markdown(
        f'<div class="ar-field-row"><span class="ar-field-label">Razorpay event ID</span>'
        f'<span class="ar-field-value">{id_label}</span></div>',
        unsafe_allow_html=True,
    )

    field_row("Received", data.format_ts(snapshot["received_at"]))
    field_row("Stage: payment.failed", f'{snapshot["error_reason"] or "—"} ({money(snapshot["amount_rupees"])})')
    field_row("Stage: classification", snapshot["classification_bucket"])
    field_row("Stage: ML / policy decision", snapshot["selected_candidate_type"])
    field_row("Decision source", snapshot["decision_source"])
    field_row("Stage: compliance", snapshot["compliance_display"])

    if snapshot["llm_provider"] is None:
        field_row("Stage: LLM", "—")
    else:
        status = "SUCCESS" if snapshot["llm_success"] else "FAILED"
        field_row("Stage: LLM", f'{snapshot["llm_provider"]} / {snapshot["llm_model"]} / {status}')
        if not snapshot["llm_success"]:
            field_row("Fallback", "deterministic (LLM failure never changes the recovery decision)")

    field_row("Stage: communication", snapshot["communication_action"])
    field_row("Final status", snapshot["final_status"])


def _render_latest_communication(snapshot: dict | None) -> None:
    """Compact "latest communication" summary -- same snapshot as the
    pipeline above (zero extra queries), never the model's hidden reasoning
    (llm_invocations.structured_output only ever contains the validated
    schema fields -- see llm/client.py's OllamaLLMClient, which never reads
    message.thinking in the first place)."""
    st.markdown("###### Latest communication")
    if snapshot is None or snapshot["llm_provider"] is None:
        empty_state("No LLM communication has been generated yet for a live event.")
        return

    status = "Success" if snapshot["llm_success"] else "Failed (deterministic fallback used)"
    message = snapshot["communication_message"] or "—"
    if len(message) > 240:
        message = message[:237] + "..."
    field_row("Provider", str(snapshot["llm_provider"]).capitalize())
    field_row("Model", snapshot["llm_model"])
    field_row("Task", snapshot["llm_task"])
    field_row("Status", status)
    field_row("Message", message)


def _render_latest_revenue_pipeline(snapshot: dict | None) -> None:
    """Same idea as _render_latest_recovery_pipeline above, but for the most
    recent event across the other 5 unified-ML domains (checkout_abandoned/
    mandate_failed/receivable_overdue/promise_to_pay_broken/Payment-Link).
    Distinguishes 3 ML states -- USED (ML's candidate is the final one),
    CONSULTED_OVERRIDDEN (ML ran and produced a real score, but the
    rule-based eligibility/human-review gate is what actually won), and
    FALLBACK (ML never ran, e.g. artifact unavailable) -- see
    ui/data.py::get_live_revenue_pipeline_snapshot. The middle state is
    the one that used to be silently indistinguishable from "not
    consulted"; it never is here."""
    st.markdown("###### Latest revenue-risk event")
    if snapshot is None:
        empty_state(
            "No live checkout/mandate/receivable/promise/Payment-Link event has been orchestrated yet."
        )
        return

    field_row("Event type", snapshot["event_type"])
    field_row("External ID", mono(snapshot["external_id"]) if snapshot["external_id"] else "—")
    field_row("Received", data.format_ts(snapshot["received_at"]))
    field_row("Stage: classification", snapshot["classification_bucket"])

    prob = snapshot["predicted_recovery_probability"]
    prob_str = f"{prob:.1%}" if prob is not None else "—"
    ml_status = snapshot["ml_status"]
    if ml_status == "USED":
        field_row("Stage: unified ML model", f'{snapshot["model_version"]} · ML USED · P(recovery)={prob_str}')
        field_row("ML-selected candidate", snapshot["selected_candidate_type"])
        if snapshot["rule_baseline_candidate"]:
            field_row("Rule baseline (for comparison)", snapshot["rule_baseline_candidate"])
    elif ml_status == "CONSULTED_OVERRIDDEN":
        field_row("Stage: unified ML model", f'{snapshot["model_version"]} · ML CONSULTED, overridden by policy · P(recovery)={prob_str}')
        field_row("ML recommendation (not used)", snapshot["ml_recommendation"] or "—")
        field_row("Policy-selected candidate", snapshot["selected_candidate_type"])
    else:
        field_row("Stage: unified ML model", "ML FALLBACK (artifact unavailable at decision time)")
        field_row("Policy-selected candidate", snapshot["selected_candidate_type"])
    field_row("Decision source", snapshot["decision_source"])
    field_row("Stage: compliance", snapshot["compliance_display"] or "—")

    if snapshot["llm_provider"] is None:
        field_row("Stage: LLM", "—")
    else:
        status = "SUCCESS" if snapshot["llm_success"] else "FAILED"
        field_row("Stage: LLM", f'{snapshot["llm_provider"]} / {snapshot["llm_model"]} / {snapshot["llm_task"]} / {status}')

    field_row("Final status", snapshot["final_status"] or "—")


@st.fragment(run_every=f"{LIVE_REFRESH_SECONDS}s")
def _overview_live_fragment() -> None:
    source_tag("live")
    _render_live_pill("overview")
    kpis = _run_live("overview", data.get_live_kpis) or {}
    llm_summary = _run_live("overview_llm", data.get_live_llm_summary) or {}

    cols = st.columns(6, gap="small")
    with cols[0]:
        kpi_card("Failed payments", str(kpis.get("failed_payments", 0)), "raw_events, all-time")
    with cols[1]:
        kpi_card("Policy decisions", str(kpis.get("policy_decisions", 0)), "All domains")
    with cols[2]:
        kpi_card("Retry actions", str(kpis.get("retry_actions", 0)), "selected candidate ≠ NO_ACTION")
    with cols[3]:
        kpi_card("No action", str(kpis.get("no_action", 0)), "policy chose NO_ACTION")
    with cols[4]:
        kpi_card("Received, not orchestrated", str(kpis.get("received_not_orchestrated", 0)), "e.g. no subscription_id")
    with cols[5]:
        total_llm = llm_summary.get("total_invocations", 0)
        if llm_summary.get("latest_provider") is None:
            llm_sub = "No LLM calls yet"
        else:
            status = "SUCCESS" if llm_summary.get("latest_success") else "FAILED"
            llm_sub = f'Latest: {llm_summary["latest_provider"]} / {llm_summary["latest_model"]} · {llm_summary["latest_task"]} · {status}'
        kpi_card("LLM calls", str(total_llm), llm_sub)

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    pipeline_snapshot = _run_live("overview_pipeline", data.get_live_pipeline_snapshot)
    _render_latest_recovery_pipeline(pipeline_snapshot)

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    revenue_pipeline_snapshot = _run_live("overview_revenue_pipeline", data.get_live_revenue_pipeline_snapshot)
    _render_latest_revenue_pipeline(revenue_pipeline_snapshot)

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    st.markdown("###### Recent payment events")
    events_df = data.get_live_raw_events_df(limit=25)
    if events_df.empty:
        empty_state("No live webhook events received yet. Send a test Razorpay payment.failed webhook to see it appear here automatically.")
        return

    table = pd.DataFrame(
        {
            "Time": events_df["received_at"].map(data.format_ts),
            "Payment ID": events_df["payment_id"],
            "Event": events_df["event_type"],
            "Amount": events_df["amount_rupees"].map(money),
            "Failure reason": events_df["error_reason"].fillna("—"),
            "Subscription": events_df["subscription_id"].fillna("—"),
        }
    )
    st.dataframe(table, width='stretch', height=300, hide_index=True)
    st.caption(
        "Recovery Queue shows the classification/policy/compliance outcome for events that had a "
        "subscription_id and could be orchestrated (Recovery Queue page)."
    )

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    _render_latest_communication(pipeline_snapshot)


def page_overview() -> None:
    section_header("Recovery Overview", "Live payment-recovery operations and synthetic benchmark performance, kept strictly separate.")

    st.markdown("##### Live operations")
    _overview_live_fragment()

    st.markdown("<div style='height:0.45rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Synthetic benchmark — money recovery by policy")
    source_tag("synthetic")
    report = data.load_report("decision_engine_v4_evaluation.json")
    if report:
        render_policy_comparison_charts(report)
    else:
        empty_state("No evaluation report available — run evaluation/evaluate_decision_engine_v4.py to generate one.")


# ---------------------------------------------------------------------------
# Page: Recovery Queue -- LIVE database
# ---------------------------------------------------------------------------

def render_live_event_detail(event_id) -> None:
    detail = data.get_live_event_detail(event_id)
    if detail is None:
        empty_state("No detail available for this event.")
        return

    policy_row, failure_row, raw_row = detail["policy"], detail["failure"], detail["raw"]
    audit_rows, llm_rows = detail["audit"], detail["llm"]

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    st.markdown(f"#### Event detail — {mono(event_id)}", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="small")
    with col1:
        field_group("Event")
        field_row("Payment ID", raw_row.payment_id if raw_row else None, mono=True)
        field_row("Order ID", raw_row.order_id if raw_row else None, mono=True)
        field_row("Subscription ID", policy_row.subscription_id, mono=True)
        field_row("Amount", money((raw_row.amount or 0) / 100.0) if raw_row else None)
        field_row("Currency", raw_row.currency if raw_row else None)
        field_row("Received", data.format_ts(raw_row.received_at) if raw_row else None)

        field_group("Failure")
        field_row("Error code", raw_row.error_code if raw_row else None)
        field_row("Reason", raw_row.error_reason if raw_row else None)
        field_row("Source", raw_row.error_source if raw_row else None)
        field_row("Step", raw_row.error_step if raw_row else None)
        field_row("Description", raw_row.error_description if raw_row else None)

        field_group("Classification")
        field_row("Bucket", failure_row.classification_bucket if failure_row else None)
        field_row("Confidence", f"{failure_row.classification_confidence:.2f}" if failure_row and failure_row.classification_confidence is not None else None)
        field_row("Rule version", failure_row.rule_version if failure_row else None)

    with col2:
        field_group("Policy")
        field_row("Candidate", policy_row.selected_candidate_type)
        field_row("Candidate time", data.format_ts(policy_row.selected_candidate_datetime))
        field_row("Decision source", policy_row.decision_source)
        field_row("Policy version", policy_row.policy_version)
        field_row("Decision reason", policy_row.decision_reason)

        compliance_audit = next((a for a in reversed(audit_rows) if a.action == "orchestrator_compliance"), None)
        compliance_fields = data.extract_compliance_fields(compliance_audit.reason if compliance_audit else None)
        field_group("Compliance")
        field_row("Payment allowed", compliance_fields.get("payment_allowed"))
        field_row("Communication allowed", compliance_fields.get("communication_allowed"))
        field_row("Rule version", compliance_fields.get("rule_version"))
        field_row("Reason", compliance_fields.get("payment_reason") or compliance_fields.get("communication_reason"))

        comm = llm_rows[-1] if llm_rows else None
        field_group("Actions")
        field_row("Payment action", "no_action" if policy_row.selected_candidate_type == "NO_ACTION" else "retry_scheduled")
        field_row("Communication action", ("sent" if comm.success else "fallback_used") if comm else "—")
        field_row("LLM success", ("Yes" if comm.success else "No") if comm else "—")

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    field_group("Audit trail")
    if not audit_rows:
        empty_state("No audit rows for this event.")
    for i, row in enumerate(audit_rows, start=1):
        timeline_step(i, f"{row.actor} — {row.action}", html.escape(row.reason or "—"))


@st.fragment(run_every=f"{LIVE_REFRESH_SECONDS}s")
def _recovery_queue_fragment() -> None:
    source_tag("live")
    _render_live_pill("recovery_queue")
    df = _run_live("recovery_queue", data.get_live_recovery_queue_df, limit=200)

    if df is None or df.empty:
        empty_state(
            "No live recovery decisions yet. Events without a subscription_id (e.g. a generic Payment Link "
            "failure) are never orchestrated into a policy decision — see \"Received, not orchestrated\" below."
        )
    else:
        table = pd.DataFrame(
            {
                "Event ID": df["event_id"],
                "Payment ID": df["payment_id"],
                "Subscription": df["subscription_id"].fillna("—"),
                "Amount": df["amount_rupees"].map(money),
                "Failure reason": df["error_reason"].fillna("—"),
                "Classification": df["classification_bucket"],
                "Recommended retry": df["selected_candidate_type"],
                "Retry time": df["selected_candidate_datetime"].map(data.format_ts),
                "Communication": df["communication_action"],
                "Final status": df["final_status"],
                "Updated": df["decided_at"].map(data.format_ts),
            }
        )
        st.caption(f"{len(table)} live decisions.")
        event = st.dataframe(
            table, width='stretch', height=380, hide_index=True,
            on_select="rerun", selection_mode="single-row", key="live_recovery_queue_table",
        )
        selected_rows = event.selection.rows if event and event.selection else []
        if selected_rows:
            render_live_event_detail(table.iloc[selected_rows[0]]["Event ID"])
        else:
            empty_state("Select a row above to see its full decision detail.")

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    unrouted = _run_live("recovery_queue_unrouted", data.get_live_unrouted_raw_events_df, limit=25)
    if unrouted is not None and not unrouted.empty:
        st.markdown("###### Received, not orchestrated")
        st.caption("Stored webhook deliveries that never reached classification/policy at all — with the real reason why.")
        table2 = pd.DataFrame(
            {
                "Raw event ID": unrouted["id"],
                "Received": unrouted["received_at"].map(data.format_ts),
                "Payment ID": unrouted["payment_id"],
                "Amount": unrouted["amount_rupees"].map(money),
                "Error reason": unrouted["error_reason"].fillna("—"),
                "Reason not orchestrated": unrouted["reason_not_orchestrated"],
            }
        )
        st.dataframe(table2, width='stretch', hide_index=True)


def page_recovery_queue() -> None:
    section_header("Recovery Queue", "Every failure event the recovery pipeline has decided on, read live from the database.")
    _recovery_queue_fragment()


# ---------------------------------------------------------------------------
# Page: Payment Events -- LIVE raw_events explorer
# ---------------------------------------------------------------------------

@st.fragment(run_every=f"{LIVE_REFRESH_SECONDS}s")
def _payment_events_fragment() -> None:
    source_tag("live")
    _render_live_pill("payment_events")
    df = _run_live("payment_events", data.get_live_raw_events_df, limit=300)

    if df is None or df.empty:
        empty_state("No webhook deliveries stored yet.")
        return

    with st.expander("Filters", expanded=False):
        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            event_types = sorted(df["event_type"].dropna().unique().tolist())
            chosen_types = st.multiselect("Event type", event_types, default=event_types, key="pe_filter_event_type")
        with c2:
            reasons = sorted(df["error_reason"].dropna().unique().tolist())
            chosen_reasons = st.multiselect("Failure reason", reasons, default=reasons, key="pe_filter_reason")
        with c3:
            payment_id_filter = st.text_input("Payment ID contains", key="pe_filter_payment_id")

    filtered = df[df["event_type"].isin(chosen_types)]
    if chosen_reasons:
        filtered = filtered[filtered["error_reason"].isin(chosen_reasons) | filtered["error_reason"].isna()]
    if payment_id_filter:
        filtered = filtered[filtered["payment_id"].fillna("").str.contains(payment_id_filter, case=False)]

    table = pd.DataFrame(
        {
            "Timestamp": filtered["received_at"].map(data.format_ts),
            "Razorpay Event ID": filtered["razorpay_event_id"],
            "Payment ID": filtered["payment_id"],
            "Order ID": filtered["order_id"],
            "Subscription ID": filtered["subscription_id"].fillna("—"),
            "Amount": filtered["amount_rupees"].map(money),
            "Currency": filtered["currency"],
            "Error Code": filtered["error_code"].fillna("—"),
            "Error Reason": filtered["error_reason"].fillna("—"),
            "Signature Verified": filtered["signature_verified"].map(lambda v: "Yes" if v else "No"),
        }
    )
    st.caption(f"Showing {len(table)} of {len(df)} stored raw events.")
    event = st.dataframe(
        table, width='stretch', height=420, hide_index=True,
        on_select="rerun", selection_mode="single-row", key="live_raw_events_table",
    )
    selected_rows = event.selection.rows if event and event.selection else []
    if not selected_rows:
        return

    selected_id = filtered.iloc[selected_rows[0]]["id"]
    row = filtered[filtered["id"] == selected_id].iloc[0]
    st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
    st.markdown("###### Parsed event detail")
    d1, d2, d3 = st.columns(3, gap="small")
    with d1:
        field_group("Event")
        field_row("Payment ID", row["payment_id"], mono=True)
        field_row("Order ID", row["order_id"], mono=True)
        field_row("Amount", money(row["amount_rupees"]))
    with d2:
        field_group("Failure")
        field_row("Error code", row["error_code"])
        field_row("Reason", row["error_reason"])
        field_row("Source", row["error_source"])
        field_row("Step", row["error_step"])
    with d3:
        field_group("Delivery")
        field_row("Received", data.format_ts(row["received_at"]))
        field_row("Signature verified", "Yes" if row["signature_verified"] else "No")
        field_row("Subscription ID", row["subscription_id"], mono=True)

    with st.expander("Raw webhook payload (as stored — never includes the webhook secret or API keys)"):
        try:
            st.code(json.dumps(json.loads(row["raw_payload"] or "{}"), indent=2), language="json")
        except ValueError:
            st.code(row["raw_payload"] or "—")


def page_payment_events() -> None:
    section_header("Payment Events", "The real raw_events table this project's FastAPI webhook handler writes to.")
    _payment_events_fragment()


# ---------------------------------------------------------------------------
# Page: Analytics -- Live Operations tab + Model / Policy Evaluation tab
# ---------------------------------------------------------------------------

@st.fragment(run_every=f"{LIVE_REFRESH_SECONDS}s")
def _analytics_live_operations_fragment() -> None:
    source_tag("live")
    _render_live_pill("analytics_live")
    kpis = _run_live("analytics_live", data.get_live_kpis) or {}
    df = data.get_live_recovery_queue_df(limit=1000)

    cols = st.columns(4, gap="small")
    with cols[0]:
        kpi_card("Failure volume", str(kpis.get("failed_payments", 0)), "raw_events, all-time")
    with cols[1]:
        kpi_card("Retry rate", f"{(kpis.get('retry_actions', 0) / kpis['policy_decisions'] * 100):.0f}%" if kpis.get("policy_decisions") else "—", "of orchestrated decisions")
    with cols[2]:
        kpi_card("Blocked / no-action rate", f"{(kpis.get('no_action', 0) / kpis['policy_decisions'] * 100):.0f}%" if kpis.get("policy_decisions") else "—", "of orchestrated decisions")
    with cols[3]:
        n_sent = int((df["communication_action"] == "sent").sum()) if not df.empty else 0
        n_blocked = int((df["communication_action"] == "blocked").sum()) if not df.empty else 0
        kpi_card("Communication outcomes", f"{n_sent} sent / {n_blocked} blocked", "live LLMInvocation + compliance rows")

    if df.empty:
        empty_state("No live recovery decisions yet to summarize.")
        return
    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.markdown("###### Candidate selection")
        counts = df["selected_candidate_type"].value_counts()
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color="#2F4CDD"))
        fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, width='stretch')
    with c2:
        st.markdown("###### Classification distribution")
        counts = df["classification_bucket"].value_counts()
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55, marker_colors=["#1D9A6C", "#C23A2E", "#B0740B", "#5B6172"]))
        fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')


def page_analytics() -> None:
    section_header("Analytics", "Live operational analytics and offline model/policy evaluation, kept in separate tabs — never mixed.")

    tab_live, tab_eval = st.tabs(["Live Operations", "Model / Policy Evaluation"])

    with tab_live:
        _analytics_live_operations_fragment()

    with tab_eval:
        report = data.load_report("decision_engine_v4_evaluation.json")
        db, queue_df = data.build_demo_database()

        st.markdown("##### 1–2. Recovery value & recovery rate by policy")
        source_tag("synthetic")
        if report:
            render_policy_comparison_charts(report)
        else:
            empty_state("No evaluation report available.")

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        st.markdown("##### Baseline definitions")
        source_tag("synthetic")
        render_baseline_definitions_section(report)

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        st.markdown("##### Statistical significance — Fixed Retry vs Deployed Policy")
        source_tag("synthetic")
        render_statistical_significance_section(report)

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        st.markdown("##### Recovery economics")
        source_tag("synthetic")
        render_economics_section(report)

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="small")
        with c1:
            st.markdown("##### 3. Candidate selection distribution (demo sample)")
            source_tag("demo")
            if not queue_df.empty:
                counts = queue_df["selected_candidate_type"].value_counts()
                fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color="#2F4CDD"))
                fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig, width='stretch')
            else:
                empty_state("No demo data available.")
        with c2:
            st.markdown("##### 4. Failure classification distribution (demo sample)")
            source_tag("demo")
            if not queue_df.empty:
                counts = queue_df["classification_bucket"].value_counts()
                fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55, marker_colors=["#1D9A6C", "#C23A2E", "#B0740B", "#5B6172"]))
                fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, width='stretch')
            else:
                empty_state("No demo data available.")


# ---------------------------------------------------------------------------
# Page: Communications -- LIVE database + the 3 interactive LLM job demos
# ---------------------------------------------------------------------------

@st.fragment(run_every=f"{LIVE_REFRESH_SECONDS}s")
def _communications_live_fragment() -> None:
    source_tag("live")
    _render_live_pill("communications")
    df = _run_live("communications", data.get_live_communications_df, limit=200)

    if df is None or df.empty:
        empty_state("No communications recorded yet.")
        return

    table = pd.DataFrame(
        {
            "Time": df["created_at"].map(data.format_ts),
            "Event": df["event_id"],
            "Task": df["task_name"],
            "Language": df["language"].fillna("—"),
            "Customer Segment": df["customer_segment"].fillna("—"),
            "Retry Window": df["retry_window"].fillna("—"),
            "Provider": df["provider"].fillna("—"),
            "Status": df["status"],
            "Detail": df["message_text"].fillna("—").map(lambda t: (t[:100] + "…") if len(t) > 100 else t),
        }
    )
    st.caption(f"{len(table)} communication events (sent / fallback / blocked).")
    st.dataframe(table, width='stretch', height=380, hide_index=True)


def page_communications() -> None:
    section_header("Communications", "The three LLM jobs, downstream of every policy decision.")

    st.markdown("##### Outreach activity")
    _communications_live_fragment()

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Promise-to-Pay")
    source_tag("demo")
    _active_provider = data.get_live_system_status().get("llm_active_provider") or "—"
    st.caption(f"This job parses an INBOUND customer reply — try one below (calls llm/service.py::parse_promise_to_pay directly, via the currently configured LLM provider: {_active_provider.upper()}).")
    reply_text = st.text_input("Customer reply", value="I'll pay Friday when salary comes, via UPI")
    if st.button("Parse reply"):
        from datetime import date

        from llm.service import parse_promise_to_pay

        result = parse_promise_to_pay(customer_reply_text=reply_text, today=date(2026, 8, 24))
        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            st.metric("Parsed date", result.structured_result.get("date") or "—")
        with c2:
            st.metric("Confidence", f"{result.structured_result.get('confidence', 0):.2f}")
        with c3:
            st.metric("Channel", result.structured_result.get("channel", "—"))
        render_status_badge("SUCCESS" if result.success else "FALLBACK_USED")

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Batch Explanation")
    source_tag("synthetic")
    report = data.load_report("decision_engine_v4_evaluation.json")
    if report is None:
        empty_state("No evaluation report available to explain.")
    else:
        from llm.service import generate_batch_explanation

        explanation = generate_batch_explanation(report_summary=report)
        st.markdown(f'<div class="ar-card">{html.escape(explanation.structured_result["explanation_text"])}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Revenue at Risk -- LIVE database (Track-03: checkout_abandoned /
# mandate_failed / receivable_overdue / promise_to_pay_broken -- the SAME
# live-refresh mechanism as every other page below, nothing new introduced.)
# ---------------------------------------------------------------------------

@st.fragment(run_every=f"{LIVE_REFRESH_SECONDS}s")
def _revenue_at_risk_overview_fragment() -> None:
    source_tag("live")
    _render_live_pill("revenue_overview")
    kpis = _run_live("revenue_overview", data.get_live_revenue_at_risk_kpis)
    if kpis is None:
        empty_state("Live revenue-risk KPIs unavailable.")
        return
    cols = st.columns(4, gap="small")
    with cols[0]:
        kpi_card("Revenue at risk", money(kpis["total_at_risk_amount"]), f"{kpis['total_revenue_risk_events']} cases")
    with cols[1]:
        kpi_card("Pending", str(kpis["pending_cases"]), "awaiting outcome")
    with cols[2]:
        kpi_card("No action", str(kpis["no_action_cases"]), "not eligible / capped")
    with cols[3]:
        kpi_card("Recovered", str(kpis["demo_synthetic_recovered_cases"]), "SYNTHETIC BENCHMARK only")


@st.fragment(run_every=f"{LIVE_REFRESH_SECONDS}s")
def _revenue_recovery_queue_fragment() -> None:
    source_tag("live")
    _render_live_pill("revenue_queue")
    df = _run_live("revenue_queue", data.get_live_revenue_recovery_queue_df, limit=200)
    if df is None or df.empty:
        empty_state("No revenue-risk recovery cases yet.")
        return
    event_types = sorted(df["event_type"].unique().tolist())
    selected = st.multiselect("Filter by event type", event_types, default=event_types, key="revenue_queue_filter")
    filtered = df[df["event_type"].isin(selected)] if selected else df
    table = pd.DataFrame({
        "Event": filtered["event_id"], "Type": filtered["event_type"], "Customer": filtered["customer_ref"],
        "Amount": filtered["amount"].map(money), "Candidate": filtered["selected_candidate_type"],
        "Scheduled": filtered["selected_candidate_datetime"].map(data.format_ts),
        "Decided": filtered["decided_at"].map(data.format_ts),
    })
    st.dataframe(table, width='stretch', height=340, hide_index=True)


@st.fragment(run_every=f"{LIVE_REFRESH_SECONDS}s")
def _revenue_timeline_fragment() -> None:
    source_tag("live")
    _render_live_pill("revenue_timeline")
    df = _run_live("revenue_timeline", data.get_live_recovery_timeline_df, limit=100)
    if df is None or df.empty:
        empty_state("No recovery timeline events yet.")
        return
    table = pd.DataFrame({
        "Time": df["timestamp"].map(data.format_ts), "Event Type": df["event_type"],
        "Reference": df["reference"], "Amount": df["amount"].map(money),
    })
    st.dataframe(table, width='stretch', height=300, hide_index=True)


@st.fragment(run_every=f"{LIVE_REFRESH_SECONDS}s")
def _revenue_outcomes_fragment() -> None:
    source_tag("live")
    _render_live_pill("revenue_outcomes")
    df = _run_live("revenue_outcomes", data.get_live_recovery_outcomes_df, limit=200)
    if df is None or df.empty:
        empty_state("No recovery outcomes recorded yet.")
        return
    table = pd.DataFrame({
        "Event": df["event_id"], "Type": df["event_type"], "At risk": df["at_risk_amount"].map(money),
        "Status": df["recovery_status"], "Recovered amount": df["recovered_amount"].map(money),
        "Confirmed by": df["confirmed_by"], "Confirmed payment id": df["confirmed_payment_id"].fillna("—"),
    })
    st.dataframe(table, width='stretch', height=280, hide_index=True)


@st.fragment(run_every=f"{LIVE_REFRESH_SECONDS}s")
def _revenue_by_intervention_fragment() -> None:
    source_tag("live")
    _render_live_pill("revenue_by_intervention")
    df = _run_live("revenue_by_intervention", data.get_live_revenue_by_intervention_df)
    if df is None or df.empty:
        empty_state("No intervention data yet.")
        return
    table = pd.DataFrame({
        "Intervention": df["intervention"], "Cases": df["case_count"],
        "Total at-risk amount": df["total_at_risk_amount"].map(money),
    })
    st.dataframe(table, width='stretch', height=260, hide_index=True)


@st.fragment(run_every=f"{LIVE_REFRESH_SECONDS}s")
def _customer_recovery_queue_fragment() -> None:
    source_tag("live")
    _render_live_pill("customer_queue")
    df = _run_live("customer_queue", data.get_live_customer_recovery_queue_df, limit=200)
    if df is None or df.empty:
        empty_state("No customer recovery cases yet.")
        return
    table = pd.DataFrame({
        "Customer": df["customer_ref"], "Latest event type": df["event_type"], "Amount": df["amount"].map(money),
        "Status": df["status"], "Received": df["received_at"].map(data.format_ts),
    })
    st.dataframe(table, width='stretch', height=300, hide_index=True)


def page_revenue_at_risk() -> None:
    section_header("Revenue at Risk", "Track-03: checkout drop-off, subscriptions, mandates, receivables, and promise-to-pay — one recovery engine.")

    st.markdown("##### Revenue-at-Risk Overview")
    _revenue_at_risk_overview_fragment()

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Recovery Queue")
    _revenue_recovery_queue_fragment()

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Recovery Timeline")
    _revenue_timeline_fragment()

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Recovery Outcomes")
    _revenue_outcomes_fragment()

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Revenue At Risk by Intervention")
    _revenue_by_intervention_fragment()

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Customer Recovery Queue")
    _customer_recovery_queue_fragment()


# ---------------------------------------------------------------------------
# Page: Audit Log -- LIVE database
# ---------------------------------------------------------------------------

@st.fragment(run_every=f"{LIVE_REFRESH_SECONDS}s")
def _audit_log_fragment() -> None:
    source_tag("live")
    _render_live_pill("audit_log")
    db = data.get_live_session()
    try:
        audit_df = data.get_audit_log_df(db)
    finally:
        db.close()

    if audit_df.empty:
        empty_state("No audit records yet.")
        return

    c1, c2 = st.columns(2, gap="small")
    with c1:
        actor = st.selectbox("Filter by actor", ["All"] + sorted(audit_df["actor"].unique().tolist()), key="audit_actor_filter")
    with c2:
        status = st.selectbox("Filter by status", ["All", "ok", "attention"], key="audit_status_filter")

    filtered = audit_df.copy()
    if actor != "All":
        filtered = filtered[filtered["actor"] == actor]
    if status != "All":
        filtered = filtered[filtered["status"] == status]

    display = pd.DataFrame(
        {
            "Timestamp": filtered["timestamp"].map(data.format_ts),
            "Event": filtered["event_id"],
            "Actor": filtered["actor"],
            "Action": filtered["action"],
            "Reason": filtered["reason"].map(lambda r: (r or "")[:180] + ("…" if r and len(r) > 180 else "")),
        }
    )
    st.caption(f"{len(display)} audit rows.")
    st.dataframe(display, width='stretch', height=460, hide_index=True)


def page_audit_log() -> None:
    section_header("Audit Log", "Every stage of every recovery decision, actor-labeled and timestamped, read live from the database.")
    _audit_log_fragment()


# ---------------------------------------------------------------------------
# Page: System / Demo
# ---------------------------------------------------------------------------

@st.fragment(run_every=f"{LIVE_REFRESH_SECONDS}s")
def _system_status_fragment() -> None:
    source_tag("live")
    _render_live_pill("system_status")
    status = _run_live("system_status", data.get_live_system_status) or {}

    cols = st.columns(4, gap="small")
    with cols[0]:
        kpi_card("FastAPI", "Connected" if status.get("fastapi_connected") else "Unavailable", "GET /health")
    with cols[1]:
        kpi_card("Database", "Connected" if status.get("database_connected") else "Unavailable", status.get("database_error") or "settings.DATABASE_URL")
    with cols[2]:
        kpi_card("Webhook secret", "Configured" if status.get("webhook_secret_configured") else "Missing", "RAZORPAY_WEBHOOK_SECRET (value never shown)")
    with cols[3]:
        kpi_card("Razorpay webhook enablement", "Unknown", "Not queryable from this backend — no Razorpay account API call is implemented")

    cols2 = st.columns(5, gap="small")
    with cols2[0]:
        active_provider = status.get("llm_active_provider") or status.get("llm_provider", "—")
        active_model = status.get("llm_active_model") or "—"
        kpi_card("LLM provider (active)", active_provider.upper(), f"{active_model} — get_llm_client()")
    with cols2[1]:
        kpi_card("Subscription model", "Loaded" if status.get("model_loaded") else "Missing", "Day-8 latent-value model artifact")
    with cols2[2]:
        kpi_card("Unified ML model", "Loaded" if status.get("unified_model_loaded") else "Missing", "model/unified_model.py artifact")
    with cols2[3]:
        kpi_card("Last event received", data.format_ts(status.get("last_event_received")), "MAX(raw_events.received_at)")
    with cols2[4]:
        kpi_card("Last successful processing", data.format_ts(status.get("last_successful_processing")), "latest orchestrator_final_status audit row")

    st.caption(f"Live refresh: enabled, every {LIVE_REFRESH_SECONDS} seconds (st.fragment(run_every=\"{LIVE_REFRESH_SECONDS}s\")).")


def page_system_demo() -> None:
    section_header("System / Demo", "Actual runtime state, plus an interactive recovery-flow runner over demo-generated data.")

    from model.unified_model import MODEL_VERSION as UNIFIED_MODEL_VERSION
    from policy.compliance import COMPLIANCE_RULE_VERSION
    from policy.compliance_v2 import COMPLIANCE_V2_RULE_VERSION
    from policy.decision_engine import MODEL_VERSION
    from policy.decision_engine_v4 import POLICY_VERSION_V4
    from policy.revenue_recovery_policy import UNIFIED_ML_POLICY_VERSION

    status = data.get_live_system_status()

    st.markdown("##### Live runtime status")
    _system_status_fragment()

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Component versions")
    st.caption("Two decision paths run live side by side: subscription-linked failures go through the Day-8/policy-v4 path; every other domain goes through the unified ML model.")
    source_tag("live")

    st.markdown("**Razorpay environment & LLM**")
    cols0 = st.columns(2, gap="small")
    with cols0[0]:
        kpi_card("Razorpay environment", status.get("environment", "—").upper(), "settings.RAZORPAY_ENV")
    with cols0[1]:
        active_provider = status.get("llm_active_provider") or status.get("llm_provider", "—")
        active_model = status.get("llm_active_model") or "—"
        kpi_card("LLM provider (active)", active_provider.upper(), f"{active_model} — settings.LLM_PROVIDER={status.get('llm_provider', '—')}")

    st.markdown("**Subscription-linked path** (`payment_failed` / `subscription_payment_failed`)")
    cols1 = st.columns(3, gap="small")
    with cols1[0]:
        kpi_card("Policy", POLICY_VERSION_V4, "policy/decision_engine_v4.py")
    with cols1[1]:
        kpi_card("Compliance", COMPLIANCE_RULE_VERSION, "policy/compliance.py")
    with cols1[2]:
        kpi_card("Model", "Loaded" if status.get("model_loaded") else "Missing", MODEL_VERSION)

    st.markdown("**Revenue-risk path** (checkout / mandate / receivable / promise-broken / payment-link)")
    cols2 = st.columns(3, gap="small")
    with cols2[0]:
        kpi_card("Policy", UNIFIED_ML_POLICY_VERSION, "policy/revenue_recovery_policy.py — ML candidate, policy-gated")
    with cols2[1]:
        kpi_card("Compliance", COMPLIANCE_V2_RULE_VERSION, "policy/compliance_v2.py")
    with cols2[2]:
        kpi_card("Model", "Loaded" if status.get("unified_model_loaded") else "Missing", UNIFIED_MODEL_VERSION)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Test suite")
    n_functions = data.count_test_functions()
    st.markdown(f'<div class="ar-card">{n_functions} test functions defined across tests/*.py (dynamically counted; some are parametrized into more individual cases at collection time). Use the button below for a live count.</div>', unsafe_allow_html=True)
    if st.button("Run full test suite now"):
        with st.spinner("Running pytest tests/ -q ..."):
            code, tail = data.run_full_test_suite()
        render_status_badge("SUCCESS" if code == 0 else "FAILURE")
        st.code(tail)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Interactive demo — Run Demo Event")
    source_tag("demo")
    _demo_provider = status.get("llm_active_provider") or "—"
    st.caption(f"Uses recovery/orchestrator.py directly, over a throwaway in-memory database. No live Razorpay actions are ever attempted — but the LLM call is real, via the currently configured provider ({_demo_provider.upper()}).")
    scenario = st.selectbox("Scenario", list(data.DEMO_SCENARIOS.keys()))
    if st.button("Run recovery"):
        model = data._try_load_model()
        with st.spinner("Running orchestrator..."):
            result, audit_rows, promise_rows = data.run_demo_scenario(scenario, model)

        steps = [
            ("Classification", result.classification_bucket),
            ("Policy", f"{result.selected_candidate_type} via {result.decision_source}"),
            ("Compliance", "Allowed" if result.compliance_allowed else f"Blocked — {result.compliance_reason}"),
            ("Communication", f"{result.communication_action}" + (f" ({result.llm_task_name})" if result.llm_task_name else "")),
            ("Final result", result.final_status),
        ]
        cols = st.columns(5, gap="small")
        for col, (title, body) in zip(cols, steps):
            with col:
                st.markdown(f"**{title}**")
                st.write(body)
        st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
        render_status_badge(result.final_status)

        if promise_rows:
            st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
            st.markdown("##### Promise-to-pay")
            promise = promise_rows[-1]
            p1, p2, p3, p4 = st.columns(4, gap="small")
            with p1:
                st.markdown("**Promised date**")
                st.write(promise.promised_date.date().isoformat() if promise.promised_date else "—")
            with p2:
                st.markdown("**Confidence**")
                st.write(f"{promise.confidence:.2f}")
            with p3:
                st.markdown("**Channel**")
                st.write(promise.channel)
            with p4:
                st.markdown("**Status**")
                render_status_badge(promise.status)
            st.markdown(
                f"**Override applied:** {'Yes' if promise.override_applied else 'No'}  \n"
                f"**Original candidate:** `{result.original_candidate_type}`"
                + (f" @ {result.original_candidate_datetime}" if result.original_candidate_datetime else "")
                + f"  \n**Final candidate:** `{result.selected_candidate_type}`"
                + (f" @ {result.selected_candidate_datetime}" if result.selected_candidate_datetime else "")
            )

        if result.llm_task_name:
            message_text = data.extract_llm_message_from_audit_rows(audit_rows)
            if message_text:
                st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
                st.markdown("**Generated / fallback message**")
                st.markdown(f'<div class="ar-card">{html.escape(message_text)}</div>', unsafe_allow_html=True)

        with st.expander("Audit trail for this run"):
            for row in audit_rows:
                st.markdown(f"`{row.actor}` — {row.action}")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Interactive demo — Generate one Track-03 event of each new kind")
    source_tag("demo")
    st.caption(
        "Uses recovery/revenue_orchestrator.py directly (plus the existing orchestrator for the payment/subscription "
        "kinds), over a throwaway in-memory database — never the real Razorpay-webhook-backed DB. No Razorpay actions "
        f"are ever attempted, but LLM calls are real, via the currently configured provider ({_demo_provider.upper()})."
    )
    if st.button("Generate demo revenue-risk events"):
        model = data._try_load_model()
        with st.spinner("Running the recovery engine over 7 synthetic events..."):
            results = data.run_revenue_demo_generator(model=model)

        for kind, result in results.items():
            with st.expander(kind.replace("_", " ").title()):
                if result is None:
                    st.write("— (no result)")
                elif hasattr(result, "final_status"):
                    st.write(f"final_status: `{result.final_status}`")
                    st.write(f"selected_candidate_type: `{result.selected_candidate_type}`")
                elif hasattr(result, "status"):
                    st.write(f"status: `{result.status}`")
                elif hasattr(result, "lifecycle_status"):
                    st.write(f"lifecycle_status: `{result.lifecycle_status}`")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    inject_css()

    # Fresh-clone correction: must run before ANY live-DB query below --
    # see ui/data.py::ensure_schema_initialized's docstring.
    data.ensure_schema_initialized()

    status = data.get_live_system_status()
    top_bar(
        test_mode=status.get("environment") == "test",
        live_ok=status.get("database_connected", False),
        refresh_label=f"live · {LIVE_REFRESH_SECONDS}s refresh",
    )

    sidebar_brand()
    page = st.sidebar.radio(
        "Navigate", NAV_PAGES, label_visibility="collapsed",
        format_func=lambda p: f"{NAV_ICONS[p]}  {p}",
    )
    sidebar_status_block(status)

    pages = {
        "Overview": page_overview,
        "Recovery Queue": page_recovery_queue,
        "Payment Events": page_payment_events,
        "Analytics": page_analytics,
        "Communications": page_communications,
        "Audit Log": page_audit_log,
        "System / Demo": page_system_demo,
        "Revenue at Risk": page_revenue_at_risk,
    }
    pages[page]()


if __name__ == "__main__":
    main()
