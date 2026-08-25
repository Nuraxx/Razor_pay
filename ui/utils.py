"""
Low-level UI formatting helpers.

Deliberately has ZERO imports from the rest of this project (no app.*,
policy.*, recovery.*, llm.*, model.* -- only pandas) so that ui.components
can depend on it without ever needing ui.data (and its much heavier
SQLAlchemy / orchestrator / model import chain) to finish initializing
first. See ui/__init__.py for the resulting one-directional import graph:

    ui.app -> ui.components -> ui.data -> application/data services
                             \\_> ui.utils <_/
"""
from __future__ import annotations

import pandas as pd


def format_inr(amount: float | None) -> str:
    """Indian-style comma grouping (e.g. 12,34,567.89), Rs-prefixed."""
    if amount is None or (isinstance(amount, float) and pd.isna(amount)):
        return "—"
    negative = amount < 0
    amount = abs(round(float(amount), 2))
    whole, _, frac = f"{amount:.2f}".partition(".")
    if len(whole) > 3:
        last3 = whole[-3:]
        rest = whole[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        whole = ",".join(groups + [last3])
    sign = "-" if negative else ""
    return f"{sign}₹{whole}.{frac}"


def format_ts(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return str(value)
    return ts.strftime("%d %b %Y, %H:%M")


def humanize_status(value: str | None) -> str:
    if not value:
        return "—"
    return str(value).replace("_", " ").title()
