"""
Day-13/14 reusable Streamlit render helpers -- every page composes these
rather than hand-rolling markup, so spacing/typography/badges stay
consistent across the whole console.
"""
from __future__ import annotations

import html

import streamlit as st

from ui.styles import DANGER as DANGER_COLOR
from ui.styles import NEUTRAL as NEUTRAL_COLOR
from ui.styles import SUCCESS as SUCCESS_COLOR
from ui.styles import inject_base_css, status_style
from ui.utils import format_inr, format_ts, humanize_status


def inject_css() -> None:
    st.markdown(inject_base_css(), unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.sidebar.markdown(
        '<div class="ar-brand-title">Adaptive Recovery</div>'
        '<div class="ar-brand-subtitle">AI-assisted payment recovery</div>'
        '<div style="height:0.7rem"></div>',
        unsafe_allow_html=True,
    )


_SOURCE_TAG_LABELS = {
    "live": ("ar-tag-live", "Live database"),
    "demo": ("ar-tag-demo", "Demo-generated"),
    "synthetic": ("ar-tag-synthetic", "Synthetic benchmark"),
}


def source_tag(kind: str) -> None:
    """kind: 'live' (real settings.DATABASE_URL rows), 'demo'
    (orchestrator re-run over sample CSV rows into a throwaway in-memory
    DB), or 'synthetic' (evaluation/reports/*.json + data/raw/*.csv) --
    the never-blurred data-source distinction this console requires on
    every section that could otherwise be ambiguous."""
    css_class, label = _SOURCE_TAG_LABELS.get(kind, _SOURCE_TAG_LABELS["demo"])
    st.markdown(f'<span class="ar-tag {css_class}">{label}</span>', unsafe_allow_html=True)


def top_bar(*, test_mode: bool, live_ok: bool, refresh_label: str) -> None:
    """Part 16: dark top bar -- brand, section label, Test Mode pill, a
    live/refresh pill, and a settings icon. Sits as an ordinary block
    beneath Streamlit's own native header (see ui/styles.py's
    .block-container comment for why that header is restyled around,
    never hidden/replaced)."""
    mode_class = "ar-topbar-pill" if test_mode else "ar-topbar-pill ar-topbar-pill-muted"
    mode_label = "TEST MODE" if test_mode else "LIVE MODE"
    live_class = "ar-live-dot ar-live-dot-live" if live_ok else "ar-live-dot ar-live-dot-error"
    st.markdown(
        f"""<div class="ar-topbar">
            <div class="ar-topbar-left">
                <span class="ar-topbar-brand">Adaptive Recovery</span>
                <span class="ar-topbar-section">Operations</span>
            </div>
            <div class="ar-topbar-right">
                <span class="{mode_class}">{mode_label}</span>
                <span class="ar-topbar-pill"><span class="{live_class}"></span> {html.escape(refresh_label)}</span>
                <span class="ar-topbar-pill ar-topbar-pill-muted">&#9881;</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def live_indicator(*, connected: bool, last_refresh, error: str | None = None, paused: bool = False) -> str:
    """Returns the HTML for a compact live-state pill -- Part 11: LIVE /
    PAUSED / LIVE DATA UNAVAILABLE, never silently showing stale data as
    current. Caller decides where to place it (st.markdown(..., unsafe_allow_html=True))."""
    if paused:
        return '<span class="ar-topbar-pill ar-topbar-pill-muted"><span class="ar-live-dot ar-live-dot-paused"></span> PAUSED</span>'
    if not connected:
        detail = f" &mdash; {html.escape(error)}" if error else ""
        return f'<span class="ar-topbar-pill"><span class="ar-live-dot ar-live-dot-error"></span> LIVE DATA UNAVAILABLE{detail}</span>'
    when = format_ts(last_refresh) if last_refresh else "just now"
    return f'<span class="ar-topbar-pill"><span class="ar-live-dot ar-live-dot-live"></span> LIVE &bull; updated {html.escape(when)}</span>'


def sidebar_status_block(status: dict) -> None:
    """Part 2/17: bottom-of-sidebar Test Mode / DB / webhook / live
    connection status -- every row reflects an actual check
    (ui/data.py::get_live_system_status), never an assumed "healthy"."""
    def _row(label: str, text: str, *, color: str) -> str:
        return f'<div class="ar-sidebar-status-row"><span>{html.escape(label)}</span><span style="color:{color};font-weight:600;">{html.escape(text)}</span></div>'

    def _bool_row(label: str, ok: bool, ok_text: str, bad_text: str) -> str:
        return _row(label, ok_text if ok else bad_text, color=SUCCESS_COLOR if ok else DANGER_COLOR)

    rows = [
        _row("Environment", status.get("environment", "—").upper(), color=NEUTRAL_COLOR),
        _bool_row("Database", bool(status.get("database_connected")), "Connected", "Unavailable"),
        _bool_row("FastAPI", bool(status.get("fastapi_connected")), "Connected", "Unavailable"),
        _bool_row("Webhook secret", bool(status.get("webhook_secret_configured")), "Configured", "Missing"),
        _row("LLM provider", status.get("llm_provider", "—").upper(), color=NEUTRAL_COLOR),
    ]
    st.sidebar.markdown(f'<div class="ar-sidebar-status">{"".join(rows)}</div>', unsafe_allow_html=True)


def field_group(title: str) -> None:
    """Part 19 detail-panel section header, e.g. "EVENT", "FAILURE", "POLICY"."""
    st.markdown(f'<div class="ar-field-group-title">{html.escape(title)}</div>', unsafe_allow_html=True)


def field_row(label: str, value, *, mono: bool = False) -> None:
    """One label/value row inside a field_group -- renders '—' for
    None/empty rather than an empty cell (never a blank that could be
    mistaken for a loading state)."""
    display = "—" if value is None or value == "" else str(value)
    value_class = "ar-field-value ar-mono" if mono else "ar-field-value"
    st.markdown(
        f'<div class="ar-field-row"><span class="ar-field-label">{html.escape(label)}</span>'
        f'<span class="{value_class}">{html.escape(display)}</span></div>',
        unsafe_allow_html=True,
    )


def mono(value) -> str:
    """Inline monospace span for an ID displayed inside other markdown (Part 18)."""
    display = "—" if value is None or value == "" else str(value)
    return f'<span class="ar-mono">{html.escape(display)}</span>'


def status_badge(status: str) -> str:
    bg, fg = status_style(status)
    label = html.escape(humanize_status(status))
    return f'<span class="ar-badge" style="background-color:{bg};color:{fg};">{label}</span>'


def render_status_badge(status: str) -> None:
    st.markdown(status_badge(status), unsafe_allow_html=True)


def section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="ar-section-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="ar-subtext">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str | None = None) -> None:
    sub_html = f'<div class="ar-kpi-sub">{html.escape(sub)}</div>' if sub else ""
    st.markdown(
        f"""<div class="ar-card">
            <div class="ar-kpi-label">{html.escape(label)}</div>
            <div class="ar-kpi-value">{html.escape(value)}</div>
            {sub_html}
        </div>""",
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple[str, str, str | None]], columns: int = 3) -> None:
    """items: list of (label, value, sub)."""
    cols = st.columns(columns)
    for i, (label, value, sub) in enumerate(items):
        with cols[i % columns]:
            kpi_card(label, value, sub)


def empty_state(message: str) -> None:
    st.markdown(f'<div class="ar-empty">{html.escape(message)}</div>', unsafe_allow_html=True)


def timeline_step(step_no: int, title: str, body: str) -> None:
    st.markdown(
        f"""<div class="ar-timeline-step">
            <div class="ar-timeline-title">{step_no}. {html.escape(title)}</div>
            <div class="ar-timeline-body">{body}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def money(value) -> str:
    return format_inr(value)


def boolean_badge_html(value: bool | None) -> str:
    if value is None:
        return status_badge("SKIPPED")
    return status_badge("TRUE" if value else "FALSE")
