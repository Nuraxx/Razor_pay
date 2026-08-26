"""
Track-03: bounded, auditable mandate retry sequencer. Pure, versioned,
step-driven -- this is a controlled RECOVERY PLANNER only (brief: "do NOT
actually execute live financial actions"); it decides the NEXT step and WHEN,
never executes a retry itself.

The step sequence is fixed and never revisited out of order:
    attempt_1 -> wait -> attempt_2 -> alternate_window -> communication ->
    final_attempt -> escalation
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

RULE_VERSION = "mandate-v1"

STEP_SEQUENCE = ["attempt_1", "wait", "attempt_2", "alternate_window", "communication", "final_attempt", "escalation"]
ATTEMPT_STEPS = frozenset({"attempt_1", "attempt_2", "final_attempt"})  # steps that count against attempt_count/max_attempts

# Deterministic scheduling offset for each step, from the moment it's
# planned. attempt_1 is 1 hour, not 0 -- mirrors policy/retry_candidates.py's
# own "immediate" candidate convention (failure_timestamp + 1h, never
# literally simultaneous): policy/compliance_v2.py requires a candidate
# strictly AFTER the event timestamp.
STEP_WAIT_HOURS = {
    "attempt_1": 1, "wait": 6, "attempt_2": 6, "alternate_window": 24,
    "communication": 24, "final_attempt": 48, "escalation": 48,
}

SEQUENCE_PLANNED = "PLANNED"
SEQUENCE_IN_PROGRESS = "IN_PROGRESS"
SEQUENCE_ESCALATED = "ESCALATED"
SEQUENCE_COMPLETED = "COMPLETED"
SEQUENCE_ABORTED = "ABORTED"


@dataclass(frozen=True)
class MandateStepDecision:
    sequence_status: str
    current_step: str
    next_action_type: str | None  # None once the sequence is terminal -- nothing left to schedule
    next_action_at: datetime | None
    retry_reason: str
    terminal_reason: str | None
    rule_version: str = RULE_VERSION

    @property
    def is_terminal(self) -> bool:
        return self.next_action_type is None


def plan_mandate_retry_sequence(
    *, current_step: str | None, attempt_count: int, max_attempts: int, now: datetime,
    prior_terminal_failure: bool = False, compliance_blocked: bool = False,
) -> MandateStepDecision:
    """Advances exactly ONE step per call. `current_step=None` means "no
    sequence exists yet" -- starts at attempt_1. Prevents retry storms and
    retries-after-terminal-failure/compliance-block by construction: both
    conditions immediately abort to a terminal (next_action_type=None)
    result regardless of where the sequence currently is."""
    if prior_terminal_failure:
        return MandateStepDecision(
            SEQUENCE_ABORTED, current_step or "attempt_1", None, None,
            "terminal_failure_reported_upstream_no_further_retries", "prior_terminal_failure",
        )
    if compliance_blocked:
        return MandateStepDecision(
            SEQUENCE_ABORTED, current_step or "attempt_1", None, None,
            "compliance_blocked_further_retries", "compliance_block",
        )

    if current_step is None:
        return MandateStepDecision(
            SEQUENCE_IN_PROGRESS, "attempt_1", "attempt_1", now + timedelta(hours=STEP_WAIT_HOURS["attempt_1"]),
            "initial_mandate_failure_retry", None,
        )
    if current_step not in STEP_SEQUENCE:
        raise ValueError(f"unknown current_step: {current_step!r}")
    if current_step == "escalation":
        # Terminal already reached -- a second call for the same sequence
        # must never re-plan or re-escalate (duplicate-action prevention).
        return MandateStepDecision(SEQUENCE_ESCALATED, "escalation", None, None, "already_escalated", "escalated")

    if attempt_count >= max_attempts:
        return MandateStepDecision(
            SEQUENCE_ESCALATED, "escalation", "escalation", now,
            f"max_attempts_reached: {attempt_count} >= {max_attempts}", "max_attempts_reached",
        )

    idx = STEP_SEQUENCE.index(current_step)
    next_idx = idx + 1
    if next_idx >= len(STEP_SEQUENCE):  # pragma: no cover -- unreachable while "escalation" is STEP_SEQUENCE's last element (caught above); kept as a defensive guard if the sequence is ever extended
        return MandateStepDecision(SEQUENCE_COMPLETED, current_step, None, None, "sequence_exhausted", "sequence_exhausted")

    next_step = STEP_SEQUENCE[next_idx]
    next_status = SEQUENCE_ESCALATED if next_step == "escalation" else SEQUENCE_IN_PROGRESS
    next_action_at = now + timedelta(hours=STEP_WAIT_HOURS[next_step])
    return MandateStepDecision(next_status, next_step, next_step, next_action_at, f"advanced_from_{current_step}", None)
