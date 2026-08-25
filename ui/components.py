"""
Day-13 reusable Streamlit render helpers -- every page composes these
rather than hand-rolling markup, so spacing/typography/badges stay
consistent across the whole dashboard (brief section 15).
"""
from __future__ import annotations

import html

import streamlit as st

from ui.styles import inject_base_css, status_style
from ui.utils import format_inr, humanize_status


def inject_css() -> None:
    st.markdown(inject_base_css(), unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.sidebar.markdown(
        '<div class="ar-brand-title">Adaptive Recovery</div>'
        '<div class="ar-brand-subtitle">AI-assisted payment recovery</div>'
        '<div style="height:0.9rem"></div>',
        unsafe_allow_html=True,
    )


def source_tag(kind: str) -> None:
    """kind: 'synthetic' or 'operational' -- brief section 5's required distinction."""
    if kind == "synthetic":
        st.markdown('<span class="ar-tag ar-tag-synthetic">Synthetic benchmark</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="ar-tag ar-tag-operational">Operational demo data</span>', unsafe_allow_html=True)


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
