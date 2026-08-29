"""
Design system: color tokens, CSS injection, and status-badge styling
for the "Adaptive Recovery" operations console.

Visual direction (Part 15/30 of the console rebuild): a Razorpay-style
payment-operations console -- dark top bar, near-white app background,
white bordered cards with restrained shadow, a blue primary accent, dense
enterprise tables, compact status badges, monospace IDs. An ORIGINAL design
inspired by that visual language, not a copy of any specific product's
layout, logo, or proprietary asset -- no Razorpay branding/wordmark/logo is
reproduced anywhere; this project's own name ("Adaptive Recovery") is used
throughout instead.
"""
from __future__ import annotations

# -- Color tokens -------------------------------------------------------
BG = "#F5F6F8"
CARD_BG = "#FFFFFF"
BORDER = "#E4E7EC"
BORDER_STRONG = "#D3D7E0"
TEXT_PRIMARY = "#161B33"      # dark charcoal/navy
TEXT_SECONDARY = "#5B6172"
TEXT_MUTED = "#8A90A2"
ACCENT = "#2F4CDD"            # restrained indigo-blue brand accent
ACCENT_SOFT = "#EEF1FE"
ACCENT_DARK = "#1E37B0"
SUCCESS = "#1D9A6C"
SUCCESS_SOFT = "#E6F6EF"
WARNING = "#B0740B"
WARNING_SOFT = "#FCF3DF"
DANGER = "#C23A2E"
DANGER_SOFT = "#FBEAE8"
INFO = "#2F4CDD"
INFO_SOFT = "#EEF1FE"
NEUTRAL = "#5B6172"
NEUTRAL_SOFT = "#EEF0F3"

# Dark top-bar tokens -- deliberately a different, darker palette from the
# rest of the app (Part 16/30: "dark top navigation bar" against an
# otherwise near-white console, matching the reference screenshot's
# contrast, not this project's own card palette).
TOPBAR_BG = "#12162B"
TOPBAR_BG_SOFT = "#1B2140"
TOPBAR_TEXT = "#F3F4F8"
TOPBAR_MUTED = "#9AA0B8"
TOPBAR_BORDER = "#262C4A"

# final_status / badge label -> (background, foreground) -- see recovery/schemas.py for the 7 canonical values.
STATUS_STYLES: dict[str, tuple[str, str]] = {
    "RETRY_ALLOWED": (SUCCESS_SOFT, SUCCESS),
    "RETRY_SCHEDULED": (SUCCESS_SOFT, SUCCESS),
    "COMMUNICATION_ALLOWED": (SUCCESS_SOFT, SUCCESS),
    "SENT": (SUCCESS_SOFT, SUCCESS),
    "NO_ACTION": (NEUTRAL_SOFT, NEUTRAL),
    "SKIPPED": (NEUTRAL_SOFT, NEUTRAL),
    "RETRY_BLOCKED": (DANGER_SOFT, DANGER),
    "BLOCKED": (DANGER_SOFT, DANGER),
    "COMMUNICATION_BLOCKED": (WARNING_SOFT, WARNING),
    "LLM_FALLBACK": (WARNING_SOFT, WARNING),
    "FALLBACK_USED": (WARNING_SOFT, WARNING),
    "POLICY_FALLBACK": (INFO_SOFT, INFO),
    "ALLOWED": (SUCCESS_SOFT, SUCCESS),
    "TRUE": (SUCCESS_SOFT, SUCCESS),
    "FALSE": (DANGER_SOFT, DANGER),
    "CONNECTED": (SUCCESS_SOFT, SUCCESS),
    "ENABLED": (SUCCESS_SOFT, SUCCESS),
    "CONFIGURED": (SUCCESS_SOFT, SUCCESS),
    "LOADED": (SUCCESS_SOFT, SUCCESS),
    "UNAVAILABLE": (DANGER_SOFT, DANGER),
    "ERROR": (DANGER_SOFT, DANGER),
    "MISSING": (DANGER_SOFT, DANGER),
    "UNKNOWN": (NEUTRAL_SOFT, NEUTRAL),
    "NOT ORCHESTRATED": (WARNING_SOFT, WARNING),
    # UI consistency pass: webhook signature status (Issue 3) and derived
    # retry-schedule status (Issue 4) -- see ui/data.py::payment_event_signature_status
    # / _derive_retry_status for how these are computed.
    "VERIFIED": (SUCCESS_SOFT, SUCCESS),
    "VERIFICATION FAILED": (DANGER_SOFT, DANGER),
    "VERIFIED (SYNTHETIC)": (INFO_SOFT, INFO),
    "SYNTHETIC / UNSIGNED": (NEUTRAL_SOFT, NEUTRAL),
    "SCHEDULED": (SUCCESS_SOFT, SUCCESS),
    "OVERDUE": (WARNING_SOFT, WARNING),
    "RECOVERED": (SUCCESS_SOFT, SUCCESS),
    "PARTIALLY_RECOVERED": (WARNING_SOFT, WARNING),
    "LOST": (DANGER_SOFT, DANGER),
    "NOT QUERYABLE": (NEUTRAL_SOFT, NEUTRAL),
}


def status_style(status: str) -> tuple[str, str]:
    return STATUS_STYLES.get((status or "").upper(), (NEUTRAL_SOFT, NEUTRAL))


def inject_base_css() -> str:
    return f"""
<style>
    .stApp {{
        background-color: {BG};
    }}

    /* Shrink Streamlit's own default top padding so our custom top bar
       (ui/components.py::top_bar) sits flush beneath its native header
       rather than leaving a large empty gap -- we do not hide/replace
       Streamlit's native header (fragile across versions, and it carries
       the sidebar-collapse control); we sit our own bar below it. */
    .block-container {{
        padding-top: 0.7rem;
        padding-bottom: 1.4rem;
        max-width: 100%;
    }}
    /* Denser vertical rhythm between stacked blocks (Part 5: remove
       excessive empty space) -- Streamlit's own default gap between
       elements is generous by design; this tightens it app-wide. */
    div[data-testid="stVerticalBlock"] {{
        gap: 0.5rem;
    }}

    section[data-testid="stSidebar"] {{
        background-color: {CARD_BG};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] * {{
        color: {TEXT_PRIMARY};
    }}
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Helvetica, Arial, sans-serif;
        color: {TEXT_PRIMARY};
    }}
    h1, h2, h3, h4 {{
        color: {TEXT_PRIMARY};
        letter-spacing: -0.01em;
    }}
    p, span, div {{
        color: {TEXT_PRIMARY};
    }}
    .ar-subtext {{
        color: {TEXT_SECONDARY};
        font-size: 0.82rem;
        margin-top: -0.2rem;
        margin-bottom: 0.5rem;
    }}
    .ar-brand-title {{
        font-size: 0.96rem;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        margin-bottom: 0;
        line-height: 1.2;
    }}
    .ar-brand-subtitle {{
        font-size: 0.66rem;
        color: {TEXT_MUTED};
        margin-top: 0;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }}

    /* Compact sidebar (Part 6) */
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.1rem;
        padding-left: 0.9rem;
        padding-right: 0.9rem;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label p {{
        font-size: 0.86rem;
    }}

    /* -- Top bar (Part 16) -- dark, sits directly beneath Streamlit's own
       native (slim, mostly transparent) header. Built as an ordinary
       block-level element at the top of the page body, not `position:
       fixed`, to avoid overlapping page content at narrow viewport widths
       (Part 20: 1280x720 desktop-first, but must not visually break). */
    .ar-topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: {TOPBAR_BG};
        border: 1px solid {TOPBAR_BORDER};
        border-radius: 8px;
        padding: 0.42rem 0.85rem;
        margin-bottom: 0.7rem;
        flex-wrap: wrap;
        gap: 0.5rem;
    }}
    .ar-topbar-left {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .ar-topbar-brand {{
        color: {TOPBAR_TEXT};
        font-weight: 700;
        font-size: 0.88rem;
        letter-spacing: -0.01em;
    }}
    .ar-topbar-section {{
        color: {TOPBAR_MUTED};
        font-size: 0.74rem;
        border-left: 1px solid {TOPBAR_BORDER};
        padding-left: 0.5rem;
        margin-left: 0.05rem;
    }}
    .ar-topbar-right {{
        display: flex;
        align-items: center;
        gap: 0.4rem;
        flex-wrap: wrap;
    }}
    .ar-topbar-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        font-size: 0.66rem;
        font-weight: 600;
        padding: 0.16rem 0.5rem;
        border-radius: 999px;
        background-color: {TOPBAR_BG_SOFT};
        color: {TOPBAR_TEXT};
        border: 1px solid {TOPBAR_BORDER};
        white-space: nowrap;
    }}
    .ar-topbar-pill-muted {{ color: {TOPBAR_MUTED}; }}

    /* Live / paused / unavailable indicator dot + label */
    .ar-live-dot {{
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: {SUCCESS};
    }}
    .ar-live-dot-live {{
        animation: ar-pulse 1.6s ease-in-out infinite;
    }}
    .ar-live-dot-paused {{ background-color: {TEXT_MUTED}; }}
    .ar-live-dot-error {{ background-color: {DANGER}; }}
    @keyframes ar-pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(29, 154, 108, 0.55); }}
        70% {{ box-shadow: 0 0 0 5px rgba(29, 154, 108, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(29, 154, 108, 0); }}
    }}

    /* Cards -- Part 3: reduced height and visual weight vs. the prior pass */
    .ar-card {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        box-shadow: none;
    }}
    .ar-kpi-label {{
        font-size: 0.64rem;
        font-weight: 600;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.15rem;
    }}
    .ar-kpi-value {{
        font-size: 1.12rem;
        font-weight: 600;
        color: {TEXT_PRIMARY};
        line-height: 1.2;
    }}
    .ar-kpi-sub {{
        font-size: 0.7rem;
        color: {TEXT_SECONDARY};
        margin-top: 0.1rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    /* Source tags: LIVE / DEMO-GENERATED / SYNTHETIC BENCHMARK -- Part 1's
       required, never-blurred distinction. */
    .ar-tag {{
        display: inline-block;
        font-size: 0.6rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        margin-bottom: 0.2rem;
    }}
    .ar-tag-live {{ background-color: {SUCCESS_SOFT}; color: {SUCCESS}; }}
    .ar-tag-demo {{ background-color: {INFO_SOFT}; color: {INFO}; }}
    .ar-tag-synthetic {{ background-color: {WARNING_SOFT}; color: {WARNING}; }}

    /* Status badges */
    .ar-badge {{
        display: inline-block;
        font-size: 0.66rem;
        font-weight: 600;
        padding: 0.12rem 0.45rem;
        border-radius: 4px;
        white-space: nowrap;
    }}

    /* Monospace IDs (Part 18) */
    .ar-mono {{
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 0.82rem;
        color: {TEXT_SECONDARY};
    }}

    /* Section header -- Part 8: smaller, more professional (was 1.18rem/700) */
    .ar-section-title {{
        font-size: 0.98rem;
        font-weight: 600;
        color: {TEXT_PRIMARY};
        margin-bottom: 0.05rem;
        letter-spacing: -0.005em;
    }}
    /* Sub-headings (##### / ######) throughout the app -- Streamlit renders
       these as h5/h6; toned down from Streamlit's own default weight/size
       so they read as compact enterprise labels, not page titles. */
    h5, h6 {{
        font-size: 0.86rem !important;
        font-weight: 600 !important;
        color: {TEXT_SECONDARY} !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-top: 0.3rem !important;
        margin-bottom: 0.2rem !important;
    }}

    /* Detail-panel field group (Part 19) */
    .ar-field-group-title {{
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: {TEXT_MUTED};
        margin: 0.6rem 0 0.25rem 0;
        padding-bottom: 0.2rem;
        border-bottom: 1px solid {BORDER};
    }}
    .ar-field-row {{
        display: flex;
        justify-content: space-between;
        gap: 0.6rem;
        padding: 0.15rem 0;
        font-size: 0.82rem;
        border-bottom: 1px solid {BORDER};
    }}
    .ar-field-row:last-child {{ border-bottom: none; }}
    .ar-field-label {{ color: {TEXT_SECONDARY}; }}
    .ar-field-value {{ color: {TEXT_PRIMARY}; font-weight: 500; text-align: right; }}

    /* Timeline (audit trail) */
    .ar-timeline-step {{
        border-left: 2px solid {BORDER};
        padding: 0.1rem 0 0.55rem 0.85rem;
        margin-left: 0.3rem;
        position: relative;
    }}
    .ar-timeline-step::before {{
        content: "";
        position: absolute;
        left: -5.5px;
        top: 0.22rem;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background-color: {ACCENT};
        border: 2px solid {CARD_BG};
    }}
    .ar-timeline-title {{
        font-weight: 600;
        font-size: 0.82rem;
        color: {TEXT_PRIMARY};
    }}
    .ar-timeline-body {{
        font-size: 0.78rem;
        color: {TEXT_SECONDARY};
        margin-top: 0.1rem;
    }}

    /* Empty state */
    .ar-empty {{
        border: 1px dashed {BORDER};
        border-radius: 6px;
        padding: 0.9rem;
        text-align: center;
        color: {TEXT_MUTED};
        background-color: {CARD_BG};
        font-size: 0.8rem;
    }}

    div[data-testid="stMetric"] {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 0.55rem 0.8rem;
    }}

    /* Enterprise table (Part 1/2/18): thin neutral border, compact rows,
       subtle radius -- row hover and the light/dark grid theme itself come
       from .streamlit/config.toml's [theme] base="light" (a canvas-rendered
       grid follows Streamlit's theme resolution, not arbitrary CSS). */
    .stDataFrame, div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 6px;
    }}
    div[data-testid="stDataFrame"] div[role="gridcell"],
    div[data-testid="stDataFrame"] div[role="columnheader"] {{
        font-size: 0.8rem;
    }}

    /* Sidebar nav: subtle highlighted background on the active radio item
       (Part 17) -- restyles st.sidebar.radio's own labels rather than
       swapping the widget, so it stays compatible with AppTest-driven
       navigation in tests/test_ui.py. */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        border-radius: 5px;
        padding: 0.22rem 0.45rem;
        margin-bottom: 0.02rem;
        transition: background-color 0.1s ease;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] {{
        gap: 0.05rem;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
        background-color: {NEUTRAL_SOFT};
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
        background-color: {ACCENT_SOFT};
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{
        color: {ACCENT_DARK};
        font-weight: 600;
    }}

    /* Sidebar bottom status block (Part 2/17) */
    .ar-sidebar-status {{
        border-top: 1px solid {BORDER};
        margin-top: 0.7rem;
        padding-top: 0.5rem;
    }}
    .ar-sidebar-status-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.7rem;
        padding: 0.1rem 0;
        color: {TEXT_SECONDARY};
    }}
</style>
"""
