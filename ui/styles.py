"""
Day-13 design system: color tokens, CSS injection, and status-badge styling
for the "Adaptive Recovery" dashboard.

Visual direction (brief section 1/16): a clean, light fintech SaaS
dashboard in the spirit of a merchant payments console -- white cards,
charcoal/navy text, a restrained blue accent, subtle borders, compact
tables -- an ORIGINAL design inspired by that visual language, not a copy
of any specific product's layout, logo, or proprietary asset.
"""
from __future__ import annotations

# -- Color tokens -------------------------------------------------------
BG = "#F5F6F8"
CARD_BG = "#FFFFFF"
BORDER = "#E4E7EC"
TEXT_PRIMARY = "#161B33"      # dark charcoal/navy
TEXT_SECONDARY = "#5B6172"
TEXT_MUTED = "#8A90A2"
ACCENT = "#2F4CDD"            # restrained indigo-blue brand accent
ACCENT_SOFT = "#EEF1FE"
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
}


def status_style(status: str) -> tuple[str, str]:
    return STATUS_STYLES.get((status or "").upper(), (NEUTRAL_SOFT, NEUTRAL))


def inject_base_css() -> str:
    return f"""
<style>
    .stApp {{
        background-color: {BG};
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
        font-size: 0.94rem;
        margin-top: -0.4rem;
        margin-bottom: 1.1rem;
    }}
    .ar-brand-title {{
        font-size: 1.28rem;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        margin-bottom: 0;
        line-height: 1.2;
    }}
    .ar-brand-subtitle {{
        font-size: 0.78rem;
        color: {TEXT_MUTED};
        margin-top: 0;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }}

    /* Cards */
    .ar-card {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 1.05rem 1.2rem;
        box-shadow: 0 1px 2px rgba(22, 27, 51, 0.04);
    }}
    .ar-kpi-label {{
        font-size: 0.76rem;
        font-weight: 600;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
    }}
    .ar-kpi-value {{
        font-size: 1.65rem;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        line-height: 1.15;
    }}
    .ar-kpi-sub {{
        font-size: 0.8rem;
        color: {TEXT_SECONDARY};
        margin-top: 0.3rem;
    }}

    /* Source tags: Synthetic benchmark vs Operational demo data */
    .ar-tag {{
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        padding: 0.16rem 0.5rem;
        border-radius: 999px;
        margin-bottom: 0.4rem;
    }}
    .ar-tag-synthetic {{ background-color: {WARNING_SOFT}; color: {WARNING}; }}
    .ar-tag-operational {{ background-color: {INFO_SOFT}; color: {INFO}; }}

    /* Status badges */
    .ar-badge {{
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        white-space: nowrap;
    }}

    /* Section header */
    .ar-section-title {{
        font-size: 1.35rem;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        margin-bottom: 0.1rem;
    }}

    /* Timeline */
    .ar-timeline-step {{
        border-left: 2px solid {BORDER};
        padding: 0.15rem 0 0.9rem 1rem;
        margin-left: 0.4rem;
        position: relative;
    }}
    .ar-timeline-step::before {{
        content: "";
        position: absolute;
        left: -6.5px;
        top: 0.28rem;
        width: 11px;
        height: 11px;
        border-radius: 50%;
        background-color: {ACCENT};
        border: 2px solid {CARD_BG};
    }}
    .ar-timeline-title {{
        font-weight: 700;
        font-size: 0.92rem;
        color: {TEXT_PRIMARY};
    }}
    .ar-timeline-body {{
        font-size: 0.85rem;
        color: {TEXT_SECONDARY};
        margin-top: 0.15rem;
    }}

    /* Empty state */
    .ar-empty {{
        border: 1px dashed {BORDER};
        border-radius: 10px;
        padding: 1.6rem;
        text-align: center;
        color: {TEXT_MUTED};
        background-color: {CARD_BG};
        font-size: 0.88rem;
    }}

    div[data-testid="stMetric"] {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }}

    .stDataFrame {{
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}
</style>
"""
