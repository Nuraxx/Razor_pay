"""
Deterministic candidate retry-time generation.

Reuses the exact payday/month-end calendar logic the synthetic dataset
generator uses (data/generate_synthetic_dataset.py) so a candidate's meaning
("payday window", "month end window") is identical whether it came from the
synthetic dataset or a live policy decision.

Unlike the dataset generator -- which adds small random jitter to candidate
offsets for dataset diversity -- candidate times here use FIXED offsets.
A policy decision must be reproducible: the same failure_timestamp must
always generate the same candidates, every time (see
tests/test_policy.py::test_policy_is_deterministic).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from data.generate_synthetic_dataset import (
    days_to_nearest_payday_window,
    next_month_end_after,
    next_payday_window_after,
)

CANDIDATE_TYPES = ["immediate", "plus_1_day_morning", "payday_window", "plus_3_days", "month_end_window"]


@dataclass(frozen=True)
class Candidate:
    candidate_type: str
    candidate_datetime: datetime

    # Candidate-action features (section 2) -- everything about WHEN we'd
    # retry, as opposed to WHO/WHAT failed (those are failure-time features,
    # already defined in model/preprocessing.py and reused unchanged here).
    hours_from_failure: float
    candidate_day_of_month: int
    candidate_day_of_week: str
    candidate_is_payday_aligned: bool
    candidate_is_month_end_aligned: bool
    candidate_days_to_payday: int


def generate_candidates(failure_timestamp: datetime) -> list[Candidate]:
    """Always returns the same 5 candidates, in CANDIDATE_TYPES order, for a given failure_timestamp."""
    raw_times = {
        "immediate": failure_timestamp + timedelta(hours=1),
        "plus_1_day_morning": (failure_timestamp + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0),
        "payday_window": next_payday_window_after(failure_timestamp),
        "plus_3_days": (failure_timestamp + timedelta(days=3)).replace(hour=12, minute=0, second=0, microsecond=0),
        "month_end_window": next_month_end_after(failure_timestamp),
    }

    candidates = []
    for candidate_type in CANDIDATE_TYPES:
        dt = raw_times[candidate_type]
        is_month_end_aligned = (next_month_end_after(dt - timedelta(days=1)).date() - dt.date()).days <= 1
        candidates.append(
            Candidate(
                candidate_type=candidate_type,
                candidate_datetime=dt,
                hours_from_failure=round((dt - failure_timestamp).total_seconds() / 3600, 2),
                candidate_day_of_month=dt.day,
                candidate_day_of_week=dt.strftime("%A"),
                candidate_is_payday_aligned=days_to_nearest_payday_window(dt) <= 1,
                candidate_is_month_end_aligned=is_month_end_aligned,
                candidate_days_to_payday=days_to_nearest_payday_window(dt),
            )
        )
    return candidates
