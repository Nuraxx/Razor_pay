"""
Day-13 dashboard entry point -- "Adaptive Recovery: AI-assisted payment recovery".

    ./venv/bin/streamlit run ui/app.py

The UI sits entirely ON TOP of the existing system: it never re-implements
classification, scoring, compliance, or LLM logic -- every page either (a)
reads a frozen SYNTHETIC BENCHMARK evaluation report Days 6-10 already
wrote, or (b) calls recovery/orchestrator.py::orchestrate_recovery (Day 12,
unmodified) to produce OPERATIONAL DEMO DATA. See ui/data.py's module
docstring for that distinction, which is never blurred anywhere below.

No ML model is trained here, Day-8 Model B is used exactly as trained,
Day-10 policy / Day-12 compliance logic is unmodified, and the three Day-11
LLM jobs are unmodified. No live Razorpay payment or real customer message
is ever sent -- this entire app runs offline against the mock LLM provider
and a throwaway in-memory database.
"""
from __future__ import annotations

import html
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui.data as data
from ui.components import (
    empty_state,
    inject_css,
    kpi_card,
    money,
    render_status_badge,
    section_header,
    sidebar_brand,
    source_tag,
    status_badge,
    timeline_step,
)

st.set_page_config(page_title="Adaptive Recovery", page_icon="💳", layout="wide")

NAV_PAGES = ["Overview", "Recovery Queue", "Payment Events", "Analytics", "Communications", "Audit Log", "System / Demo"]
NAV_ICONS = {
    "Overview": "📊", "Recovery Queue": "📋", "Payment Events": "💳", "Analytics": "📈",
    "Communications": "✉️", "Audit Log": "🧾", "System / Demo": "⚙️",
}
POLICY_LABELS = {
    "fixed_retry": "Fixed Retry", "rule_based": "Rule-Based", "day8_model_b_alone": "Day-8 Model B",
    "day9_original_fallback": "Day-9 Policy", "day10_improved_fallback": "Day-10 Policy", "oracle_policy": "Oracle",
}
# FIX pass: the one headline baseline comparison the statistical-significance
# and economics sections below focus on (matches
# evaluation/evaluate_decision_engine_v4.py's DEPLOYED_POLICY_NAME / HEADLINE_BASELINE_NAME).
DEPLOYED_POLICY_KEY = "day10_improved_fallback"
BASELINE_POLICY_KEY = "fixed_retry"


# ---------------------------------------------------------------------------
# Shared: policy comparison charts (used by Overview + Analytics)
# ---------------------------------------------------------------------------

def render_policy_comparison_charts(report: dict) -> None:
    latent = report.get("latent_economic", {})
    realized = report.get("realized_counterfactual", {})
    policies = [p for p in POLICY_LABELS if p in latent]
    labels = [POLICY_LABELS[p] for p in policies]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Latent expected value — synthetic benchmark**")
        values = [latent[p]["total_latent_value_rs"] for p in policies]
        fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#2F4CDD", text=[f"₹{v:,.0f}" for v in values], textposition="outside"))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white", yaxis_title="₹ (latent, test set)")
        st.plotly_chart(fig, width='stretch')
    with col2:
        st.markdown("**Realized counterfactual recovery — synthetic benchmark**")
        values = [realized[p]["total_recovered_rs"] for p in policies]
        fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#1D9A6C", text=[f"₹{v:,.0f}" for v in values], textposition="outside"))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white", yaxis_title="₹ (realized, test set)")
        st.plotly_chart(fig, width='stretch')

    st.caption(
        "**Latent expected value** is the synthetic simulation's own ground-truth expectation for the candidate each policy "
        "selected. **Realized counterfactual recovery** is a single stochastic sampled outcome under that same simulation. "
        "They are deliberately shown separately, never combined into one number — see README \"Day 9\" for why conflating "
        "them would overstate confidence. Neither reflects real Razorpay production performance."
    )


# ---------------------------------------------------------------------------
# Statistical significance + economics (FIX pass) -- used by Analytics
# ---------------------------------------------------------------------------

def render_statistical_significance_section(report: dict) -> None:
    """Shows the McNemar's-test / bootstrap-CI results
    `evaluate_decision_engine_v4.py::summarize_statistical_tests` already
    computed and wrote into the report -- never recomputed here, and never
    hidden if the result is negative (deployed policy currently loses to
    Fixed Retry on realized ₹ -- see README §16/§19)."""
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

    cols = st.columns(4)
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
    specification -- never blended into one number (policy/economics.py)."""
    if not report or "economics" not in report:
        empty_state("No economics data available — run evaluation/evaluate_decision_engine_v4.py to generate it.")
        return

    econ = report["economics"]
    deployed = econ.get(DEPLOYED_POLICY_KEY, {})
    baseline = econ.get(BASELINE_POLICY_KEY, {})

    st.caption(f"Deployed policy ({POLICY_LABELS.get(DEPLOYED_POLICY_KEY, DEPLOYED_POLICY_KEY)}), synthetic held-out test set:")
    cols = st.columns(4)
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


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

def page_overview() -> None:
    section_header("Recovery Overview", "Monitor failed payments, recovery decisions and expected recovered value.")

    report = data.load_report("decision_engine_v4_evaluation.json")
    db, queue_df = data.build_demo_database()

    st.markdown("##### Failed payments & recovery outcomes")
    row1 = st.columns(3)
    row2 = st.columns(3)

    with row1[0]:
        source_tag("operational")
        kpi_card("Failed payments", str(len(queue_df)) if not queue_df.empty else "0", "Events processed by the live orchestrator (demo sample)")
    with row1[1]:
        source_tag("synthetic")
        if report:
            rate = report["realized_counterfactual"]["day10_improved_fallback"]["recovery_rate"]
            kpi_card("Recovery rate", f"{rate * 100:.1f}%", "Day-10 policy, held-out test set")
        else:
            kpi_card("Recovery rate", "—", "No evaluation report available")
    with row1[2]:
        source_tag("synthetic")
        if report:
            val = report["latent_economic"]["day10_improved_fallback"]["total_latent_value_rs"]
            kpi_card("Expected recovery value", money(val), "Latent, Day-10 policy, test set")
        else:
            kpi_card("Expected recovery value", "—", "No evaluation report available")

    with row2[0]:
        source_tag("synthetic")
        if report:
            val = report["realized_counterfactual"]["day10_improved_fallback"]["total_recovered_rs"]
            kpi_card("Amount recovered", money(val), "Realized, Day-10 policy, test set")
        else:
            kpi_card("Amount recovered", "—", "No evaluation report available")
    with row2[1]:
        source_tag("operational")
        n_retry = int((queue_df["payment_action"] == "retry_scheduled").sum()) if not queue_df.empty else 0
        kpi_card("Retry actions selected", str(n_retry), "Live orchestrator run (demo sample)")
    with row2[2]:
        source_tag("operational")
        n_blocked = int((queue_df["payment_action"] == "no_action").sum()) if not queue_df.empty else 0
        kpi_card("NO_ACTION / blocked events", str(n_blocked), "Live orchestrator run (demo sample)")

    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Money recovery by policy")
    if report:
        render_policy_comparison_charts(report)
    else:
        empty_state("No evaluation report available — run evaluation/evaluate_decision_engine_v4.py to generate one.")


# ---------------------------------------------------------------------------
# Page: Recovery Queue (+ inline event detail / candidate visualization)
# ---------------------------------------------------------------------------

def page_recovery_queue() -> None:
    section_header("Recovery Queue", "Every failure event the orchestrator has processed, and the action selected for it.")
    source_tag("operational")

    db, queue_df = data.build_demo_database()
    if queue_df.empty:
        empty_state("No events available — data/raw/failure_events.csv or subscriptions.csv could not be loaded.")
        return

    display_df = queue_df.copy()
    display_df["Amount"] = display_df["event_id"].map(lambda _: None)
    events_raw = data.load_csv("failure_events.csv")
    amount_by_event = events_raw.set_index("event_id")["amount"].to_dict() if events_raw is not None else {}
    reason_by_event = events_raw.set_index("event_id")["error_reason"].to_dict() if events_raw is not None else {}

    table = pd.DataFrame({
        "Event ID": display_df["event_id"],
        "Subscription": display_df["subscription_id"],
        "Amount": display_df["event_id"].map(lambda e: money(amount_by_event.get(e))),
        "Failure reason": display_df["event_id"].map(lambda e: reason_by_event.get(e, "—")),
        "Classification": display_df["classification_bucket"],
        "Selected retry": display_df["selected_candidate_type"],
        "Expected recovery ₹": display_df["expected_recovery_value"].map(money),
        "Communication": display_df["communication_action"],
        "Promise-to-pay": display_df["promise_applied"].map(lambda v: "Applied" if v else "—"),
        "Final status": display_df["final_status"],
    })

    st.caption(f"{len(table)} events in this demo run.")
    event = st.dataframe(
        table, width='stretch', height=420, hide_index=True,
        on_select="rerun", selection_mode="single-row", key="recovery_queue_table",
    )

    selected_rows = event.selection.rows if event and event.selection else []
    if not selected_rows:
        empty_state("Select a row above to see its full decision timeline and candidate scoring.")
        return

    selected_event_id = table.iloc[selected_rows[0]]["Event ID"]
    render_event_detail(db, selected_event_id)


def render_event_detail(db, event_id) -> None:
    detail = data.get_event_detail(db, event_id)
    if detail is None:
        empty_state("No detail available for this event.")
        return

    policy_row = detail["policy"]
    llm_rows = detail["llm"]
    promise_rows = detail["promises"]
    active_promise = next((p for p in reversed(promise_rows) if p.status == "VALID" or p.override_applied), promise_rows[-1] if promise_rows else None)

    message_text = None
    if llm_rows:
        try:
            message_text = json.loads(llm_rows[-1].structured_output or "{}").get("message_text")
        except ValueError:
            message_text = None

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    st.markdown(f"#### Event detail — {html.escape(str(event_id))}")

    col1, col2 = st.columns([2, 1])
    with col1:
        steps = [
            ("Failure received", f"subscription={policy_row.subscription_id}, event_id={event_id}"),
            ("Failure classification", f"<code>{html.escape(policy_row.classification_bucket or '—')}</code>"),
            ("Policy decision", f"selected <code>{html.escape(policy_row.selected_candidate_type)}</code> via <code>{html.escape(policy_row.decision_source or '—')}</code> (policy_version={policy_row.policy_version})"),
            ("Candidate retry scoring", html.escape(policy_row.decision_reason or "—")),
            ("Compliance check", f"intervention_cost={money(policy_row.intervention_cost)}, decision_margin={policy_row.decision_margin}"),
            ("Payment action", "retry_scheduled" if policy_row.selected_candidate_type != "NO_ACTION" else "no_action"),
            (
                "Communication action",
                (f"{llm_rows[-1].task_name}: {'sent' if llm_rows[-1].success else 'fallback_used'}" if llm_rows else "skipped")
                + (f'<br><span class="ar-subtext">{html.escape(message_text)}</span>' if message_text else ""),
            ),
            ("Final result", "see badge →"),
        ]
        for i, (title, body) in enumerate(steps, start=1):
            timeline_step(i, title, body)

        if active_promise is not None:
            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            st.markdown("##### Promise-to-pay")
            pp1, pp2, pp3, pp4 = st.columns(4)
            with pp1:
                st.markdown("**Promised date**")
                st.write(active_promise.promised_date.date().isoformat() if active_promise.promised_date else "—")
            with pp2:
                st.markdown("**Confidence**")
                st.write(f"{active_promise.confidence:.2f}")
            with pp3:
                st.markdown("**Channel**")
                st.write(active_promise.channel)
            with pp4:
                st.markdown("**Status**")
                render_status_badge(active_promise.status)
            st.markdown(
                f"**Override applied:** {'Yes' if active_promise.override_applied else 'No'}"
                + (f" ({active_promise.override_outcome})" if active_promise.override_outcome else "")
                + f"  \n**Original candidate:** `{policy_row.selected_candidate_type}`"
                + f"  \n**Final candidate:** `{'promise_to_pay' if active_promise.override_applied else policy_row.selected_candidate_type}`"
            )
    with col2:
        st.markdown("**Classification**")
        st.write(policy_row.classification_bucket)
        st.markdown("**Policy**")
        st.write(policy_row.selected_candidate_type)
        st.markdown("**Expected value**")
        st.write(money(policy_row.expected_recovery_value))
        st.markdown("**Compliance**")
        render_status_badge("ALLOWED" if policy_row.selected_candidate_type != "NO_ACTION" else "BLOCKED")
        st.markdown("**Communication**")
        if llm_rows:
            render_status_badge("SENT" if llm_rows[-1].success else "FALLBACK_USED")
        else:
            render_status_badge("SKIPPED")

    render_candidate_visualization(event_id)


def render_candidate_visualization(event_id) -> None:
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Retry-candidate scoring")
    source_tag("synthetic")

    events = data.load_csv("failure_events.csv")
    subs = data.load_csv("subscriptions.csv")
    if events is None or subs is None:
        empty_state("Raw event/subscription data not available for candidate scoring.")
        return
    row = events[events["event_id"] == event_id]
    if row.empty:
        empty_state("This event isn't present in data/raw/failure_events.csv (may be an interactive demo event).")
        return
    row = row.iloc[0]
    sub_row = subs.set_index("subscription_id").loc[row["subscription_id"]]
    model = data._try_load_model()
    breakdown = data.get_candidate_breakdown(row, sub_row, model)

    cand_df = pd.DataFrame(breakdown["candidates"])
    cand_df["Candidate"] = cand_df["candidate_type"]
    cand_df["Predicted recovery ₹"] = cand_df["predicted_recovery_value"].map(money)
    cand_df["Intervention cost ₹"] = cand_df["intervention_cost"].map(money)
    cand_df["Expected net ₹"] = cand_df["expected_net_value"].map(money)
    cand_df["Valid"] = cand_df["valid"].map(lambda v: "Yes" if v else "No")
    cand_df["Selected"] = cand_df["is_selected"].map(lambda v: "✓" if v else "")

    st.dataframe(
        cand_df[["Candidate", "Predicted recovery ₹", "Intervention cost ₹", "Expected net ₹", "Valid", "Selected"]],
        width='stretch', hide_index=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Model selected:** `{breakdown['selected']}`")
    with c2:
        st.markdown(f"**Rule-based selected:** `{breakdown['rule_pick']}`")
    with c3:
        st.markdown(f"**Oracle selected:** `{breakdown['oracle_pick'] or '—'}`")


# ---------------------------------------------------------------------------
# Page: Payment Events (raw generated dataset browser)
# ---------------------------------------------------------------------------

def page_payment_events() -> None:
    section_header("Payment Events", "The raw generated failure-event dataset this project's synthetic simulation produced.")
    source_tag("synthetic")

    events = data.load_csv("failure_events.csv")
    subs = data.load_csv("subscriptions.csv")
    if events is None:
        empty_state("data/raw/failure_events.csv not found.")
        return

    merged = events.merge(subs, on="subscription_id", how="left") if subs is not None else events
    with st.expander("Filters", expanded=False):
        reasons = sorted(merged["error_reason"].dropna().unique().tolist())
        chosen_reasons = st.multiselect("Failure reason", reasons, default=reasons)
        if "plan_tier" in merged.columns:
            tiers = sorted(merged["plan_tier"].dropna().unique().tolist())
            chosen_tiers = st.multiselect("Plan tier", tiers, default=tiers)
            merged = merged[merged["plan_tier"].isin(chosen_tiers)]
    merged = merged[merged["error_reason"].isin(chosen_reasons)]

    show_cols = [c for c in ["event_id", "subscription_id", "failure_timestamp", "amount", "error_reason", "plan_tier", "city_tier", "primary_instrument"] if c in merged.columns]
    display = merged[show_cols].head(300).copy()
    if "amount" in display.columns:
        display["amount"] = display["amount"].map(money)
    st.caption(f"Showing {len(display)} of {len(merged)} matching events (max 300 rows).")
    st.dataframe(display, width='stretch', height=500, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Analytics
# ---------------------------------------------------------------------------

def page_analytics() -> None:
    section_header("Analytics", "Recovery performance, decision distribution, and system health.")

    report = data.load_report("decision_engine_v4_evaluation.json")
    db, queue_df = data.build_demo_database()

    st.markdown("##### 1–2. Recovery value & recovery rate by policy")
    source_tag("synthetic")
    if report:
        render_policy_comparison_charts(report)
    else:
        empty_state("No evaluation report available.")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Statistical significance — Fixed Retry vs Deployed Policy")
    source_tag("synthetic")
    render_statistical_significance_section(report)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Recovery economics")
    source_tag("synthetic")
    render_economics_section(report)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 3. Candidate selection distribution")
        source_tag("operational")
        if not queue_df.empty:
            counts = queue_df["selected_candidate_type"].value_counts()
            fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color="#2F4CDD"))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, width='stretch')
        else:
            empty_state("No demo data available.")
    with c2:
        st.markdown("##### 4. Failure classification distribution")
        source_tag("operational")
        if not queue_df.empty:
            counts = queue_df["classification_bucket"].value_counts()
            fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55, marker_colors=["#1D9A6C", "#C23A2E", "#B0740B", "#5B6172"]))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width='stretch')
        else:
            empty_state("No demo data available.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### 5. Compliance outcomes")
        source_tag("operational")
        if not queue_df.empty:
            counts = queue_df["payment_action"].value_counts()
            fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color="#1D9A6C"))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, width='stretch')
        else:
            empty_state("No demo data available.")
    with c4:
        st.markdown("##### 6. LLM success / fallback rate")
        source_tag("operational")
        llm_df = data.get_llm_invocations_df(db)
        if not llm_df.empty:
            counts = llm_df["success"].map({True: "Success", False: "Fallback used"}).value_counts()
            fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55, marker_colors=["#1D9A6C", "#B0740B"]))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width='stretch')
        else:
            empty_state("No LLM invocations recorded yet.")

    st.markdown("##### 7. Audit / event volume by actor")
    source_tag("operational")
    audit_df = data.get_audit_log_df(db)
    if not audit_df.empty:
        counts = audit_df["actor"].value_counts()
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color="#2F4CDD"))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, width='stretch')
    else:
        empty_state("No audit records yet.")


# ---------------------------------------------------------------------------
# Page: Communications
# ---------------------------------------------------------------------------

def page_communications() -> None:
    section_header("Communications", "The three Day-11 LLM jobs, downstream of every policy decision.")

    db, queue_df = data.build_demo_database()

    st.markdown("##### Outreach Microcopy")
    source_tag("operational")
    llm_df = data.get_llm_invocations_df(db, task_name="outreach_microcopy")
    if llm_df.empty:
        empty_state("No outreach microcopy generated in this demo run.")
    else:
        import json as _json

        records = []
        for _, r in llm_df.iterrows():
            try:
                structured = _json.loads(r["structured_output"] or "{}")
            except ValueError:
                structured = {}
            records.append({
                "Event": r["event_id"], "Language": structured.get("language", "—"),
                "Message": structured.get("message_text", "—"), "Provider": r["provider"],
                "Outcome": "Sent" if r["success"] else "Fallback used",
            })
        out_df = pd.DataFrame.from_records(records)
        for lang, label in [("en", "English"), ("hi", "Hindi"), ("hinglish", "Hinglish")]:
            subset = out_df[out_df["Language"] == lang]
            if subset.empty:
                continue
            st.markdown(f"**{label}**")
            st.dataframe(subset[["Event", "Message", "Provider", "Outcome"]].head(5), width='stretch', hide_index=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Promise-to-Pay")
    source_tag("operational")
    st.caption("This job parses an INBOUND customer reply — try one below (calls llm/service.py::parse_promise_to_pay directly, mock provider).")
    reply_text = st.text_input("Customer reply", value="I'll pay Friday when salary comes, via UPI")
    if st.button("Parse reply"):
        from datetime import date

        from llm.service import parse_promise_to_pay

        result = parse_promise_to_pay(customer_reply_text=reply_text, today=date(2026, 8, 24))
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Parsed date", result.structured_result.get("date") or "—")
        with c2:
            st.metric("Confidence", f"{result.structured_result.get('confidence', 0):.2f}")
        with c3:
            st.metric("Channel", result.structured_result.get("channel", "—"))
        render_status_badge("SUCCESS" if result.success else "FALLBACK_USED")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
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
# Page: Audit Log
# ---------------------------------------------------------------------------

def page_audit_log() -> None:
    section_header("Audit Log", "Every stage of every recovery decision, actor-labeled and timestamped.")
    source_tag("operational")

    db, queue_df = data.build_demo_database()
    audit_df = data.get_audit_log_df(db)
    if audit_df.empty:
        empty_state("No audit records yet.")
        return

    c1, c2 = st.columns(2)
    with c1:
        actor = st.selectbox("Filter by actor", ["All"] + sorted(audit_df["actor"].unique().tolist()))
    with c2:
        status = st.selectbox("Filter by status", ["All", "ok", "attention"])

    filtered = audit_df.copy()
    if actor != "All":
        filtered = filtered[filtered["actor"] == actor]
    if status != "All":
        filtered = filtered[filtered["status"] == status]

    display = pd.DataFrame({
        "Timestamp": filtered["timestamp"].map(data.format_ts),
        "Event": filtered["event_id"],
        "Actor": filtered["actor"],
        "Action": filtered["action"],
        "Status": filtered["status"],
        "Reason": filtered["reason"].map(lambda r: (r or "")[:180] + ("…" if r and len(r) > 180 else "")),
    })
    st.caption(f"{len(display)} audit rows.")
    st.dataframe(display, width='stretch', height=520, hide_index=True)


# ---------------------------------------------------------------------------
# Page: System / Demo
# ---------------------------------------------------------------------------

def page_system_demo() -> None:
    section_header("System / Demo", "Environment, versions, and an interactive recovery-flow runner.")

    from llm.client import MOCK_MODEL_NAME
    from policy.compliance import COMPLIANCE_RULE_VERSION
    from policy.decision_engine import MODEL_VERSION
    from policy.decision_engine_v4 import POLICY_VERSION_V4

    cols = st.columns(5)
    with cols[0]:
        kpi_card("Environment", "Offline Demo", "No live Razorpay/LLM network calls")
    with cols[1]:
        kpi_card("LLM", "Mock", MOCK_MODEL_NAME)
    with cols[2]:
        kpi_card("Policy", POLICY_VERSION_V4, "Day-10 decision engine")
    with cols[3]:
        kpi_card("Compliance", COMPLIANCE_RULE_VERSION, "Day-12 gate")
    with cols[4]:
        kpi_card("Model", "Day-8 Model B", MODEL_VERSION)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Test suite")
    n_functions = data.count_test_functions()
    st.markdown(f'<div class="ar-card">{n_functions} test functions defined across tests/*.py (dynamically counted; some are parametrized into more individual cases at collection time — the last full pytest run reported 432 passing). Use the button below for a live count.</div>', unsafe_allow_html=True)
    if st.button("Run full test suite now"):
        with st.spinner("Running pytest tests/ -q ..."):
            code, tail = data.run_full_test_suite()
        render_status_badge("SUCCESS" if code == 0 else "FAILURE")
        st.code(tail)

    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Interactive demo — Run Demo Event")
    st.caption("Uses recovery/orchestrator.py directly. No live Razorpay actions are ever attempted.")
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
        cols = st.columns(5)
        for col, (title, body) in zip(cols, steps):
            with col:
                st.markdown(f"**{title}**")
                st.write(body)
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        render_status_badge(result.final_status)

        if promise_rows:
            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
            st.markdown("##### Promise-to-pay")
            promise = promise_rows[-1]
            p1, p2, p3, p4 = st.columns(4)
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
                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                st.markdown("**Generated / fallback message**")
                st.markdown(f'<div class="ar-card">{html.escape(message_text)}</div>', unsafe_allow_html=True)

        with st.expander("Audit trail for this run"):
            for row in audit_rows:
                st.markdown(f"`{row.actor}` — {row.action}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    inject_css()
    sidebar_brand()
    page = st.sidebar.radio(
        "Navigate", NAV_PAGES, label_visibility="collapsed",
        format_func=lambda p: f"{NAV_ICONS[p]}  {p}",
    )

    pages = {
        "Overview": page_overview,
        "Recovery Queue": page_recovery_queue,
        "Payment Events": page_payment_events,
        "Analytics": page_analytics,
        "Communications": page_communications,
        "Audit Log": page_audit_log,
        "System / Demo": page_system_demo,
    }
    pages[page]()


if __name__ == "__main__":
    main()
