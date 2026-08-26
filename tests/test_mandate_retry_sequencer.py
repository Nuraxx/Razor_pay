"""
Track-03 tests: policy/mandate_rules.py -- pure step-sequencer, no DB.
Orchestration-level idempotency/compliance-block/LLM-failure coverage lives
in tests/test_revenue_orchestrator.py; this file is the rule module alone.
"""
from datetime import datetime, timedelta

import pytest

from policy.mandate_rules import (
    ATTEMPT_STEPS,
    RULE_VERSION,
    SEQUENCE_ABORTED,
    SEQUENCE_COMPLETED,
    SEQUENCE_ESCALATED,
    SEQUENCE_IN_PROGRESS,
    STEP_SEQUENCE,
    plan_mandate_retry_sequence,
)

NOW = datetime(2026, 8, 25, 10, 0, 0)


class TestStepProgression:
    def test_first_call_starts_at_attempt_1(self):
        result = plan_mandate_retry_sequence(current_step=None, attempt_count=0, max_attempts=3, now=NOW)
        assert result.current_step == "attempt_1"
        assert result.sequence_status == SEQUENCE_IN_PROGRESS
        assert result.next_action_type == "attempt_1"
        assert result.next_action_at == NOW + timedelta(hours=1)  # STEP_WAIT_HOURS["attempt_1"] -- never literally simultaneous with `now`

    def test_advances_through_full_sequence_in_order(self):
        # Each call reports the step whose action should now be taken
        # (next_action_type == current_step, e.g. "do attempt_1 now") --
        # exactly len(STEP_SEQUENCE) calls walk through all 7 actionable
        # steps; a subsequent call is what actually observes termination.
        step = None
        seen = []
        attempt_count = 0
        for _ in range(len(STEP_SEQUENCE)):
            result = plan_mandate_retry_sequence(current_step=step, attempt_count=attempt_count, max_attempts=99, now=NOW)
            seen.append(result.current_step)
            if result.current_step in ATTEMPT_STEPS:
                attempt_count += 1
            step = result.current_step
        assert seen == STEP_SEQUENCE  # attempt_1..escalation, in exact order, no step skipped or revisited

        final = plan_mandate_retry_sequence(current_step=step, attempt_count=attempt_count, max_attempts=99, now=NOW)
        assert final.is_terminal

    def test_never_revisits_a_completed_step(self):
        result = plan_mandate_retry_sequence(current_step="attempt_1", attempt_count=1, max_attempts=99, now=NOW)
        assert result.current_step == "wait"  # advances forward, never back to attempt_1

    def test_escalation_is_terminal_and_idempotent(self):
        result = plan_mandate_retry_sequence(current_step="escalation", attempt_count=5, max_attempts=99, now=NOW)
        assert result.is_terminal
        assert result.sequence_status == SEQUENCE_ESCALATED
        assert result.terminal_reason == "escalated"

    def test_unknown_step_raises(self):
        with pytest.raises(ValueError):
            plan_mandate_retry_sequence(current_step="not_a_real_step", attempt_count=0, max_attempts=3, now=NOW)


class TestMaxAttemptsAndBounding:
    def test_max_attempts_forces_escalation(self):
        result = plan_mandate_retry_sequence(current_step="attempt_1", attempt_count=3, max_attempts=3, now=NOW)
        assert result.current_step == "escalation"
        assert result.sequence_status == SEQUENCE_ESCALATED
        assert result.terminal_reason == "max_attempts_reached"

    def test_scheduling_is_deterministic(self):
        result = plan_mandate_retry_sequence(current_step="attempt_1", attempt_count=1, max_attempts=99, now=NOW)
        assert result.next_action_at == NOW + timedelta(hours=6)  # STEP_WAIT_HOURS["wait"]


class TestNoRetryStorms:
    def test_prior_terminal_failure_aborts_regardless_of_current_step(self):
        result = plan_mandate_retry_sequence(current_step="attempt_1", attempt_count=1, max_attempts=99, now=NOW, prior_terminal_failure=True)
        assert result.is_terminal
        assert result.sequence_status == SEQUENCE_ABORTED
        assert result.terminal_reason == "prior_terminal_failure"

    def test_compliance_blocked_aborts_regardless_of_current_step(self):
        result = plan_mandate_retry_sequence(current_step="wait", attempt_count=1, max_attempts=99, now=NOW, compliance_blocked=True)
        assert result.is_terminal
        assert result.sequence_status == SEQUENCE_ABORTED
        assert result.terminal_reason == "compliance_block"

    def test_terminal_failure_outranks_max_attempts_and_step_progression(self):
        result = plan_mandate_retry_sequence(current_step="escalation", attempt_count=0, max_attempts=99, now=NOW, prior_terminal_failure=True)
        assert result.terminal_reason == "prior_terminal_failure"

    def test_sequence_exhaustion_after_final_step_is_completed_not_reopened(self):
        result = plan_mandate_retry_sequence(current_step="escalation", attempt_count=0, max_attempts=99, now=NOW)
        # escalation is the last step -- already covered as terminal above, but
        # explicitly confirm a second identical call yields the same terminal state.
        second = plan_mandate_retry_sequence(current_step="escalation", attempt_count=0, max_attempts=99, now=NOW)
        assert result == second


class TestVersioning:
    def test_rule_version_recorded(self):
        result = plan_mandate_retry_sequence(current_step=None, attempt_count=0, max_attempts=3, now=NOW)
        assert result.rule_version == RULE_VERSION


class TestCandidateAlwaysStrictlyAfterNow:
    """policy/compliance_v2.py::_candidate_time_is_valid rejects any candidate
    with selected_candidate_datetime <= occurred_at ("candidate_not_after_event").
    This class pins the producing side of that invariant: plan_mandate_retry_sequence
    must never itself compute a next_action_at that is equal to or earlier than
    `now`, for any step in the sequence -- not just attempt_1 (which was the
    site of a real bug: STEP_WAIT_HOURS["attempt_1"] was originally 0, scheduling
    the first retry AT the failure instant rather than after it)."""

    def test_initial_call_schedules_strictly_after_now(self):
        result = plan_mandate_retry_sequence(current_step=None, attempt_count=0, max_attempts=3, now=NOW)
        assert result.next_action_at is not None
        assert result.next_action_at > NOW
        assert not (result.next_action_at <= NOW)  # explicit: equality is also forbidden, not just "earlier"

    def test_every_step_in_the_sequence_schedules_strictly_after_now(self):
        step = None
        attempt_count = 0
        for _ in range(len(STEP_SEQUENCE)):
            result = plan_mandate_retry_sequence(current_step=step, attempt_count=attempt_count, max_attempts=99, now=NOW)
            if result.next_action_at is not None:  # terminal steps (e.g. escalation reached via max_attempts) carry no schedule
                assert result.next_action_at > NOW, f"step {result.current_step!r} scheduled next_action_at <= now"
            if result.current_step in ATTEMPT_STEPS:
                attempt_count += 1
            step = result.current_step

    def test_max_attempts_forced_escalation_is_scheduled_AT_now_not_after(self):
        # KNOWN, DELIBERATE EXCEPTION to this class's own invariant -- pinned
        # here rather than hidden. When attempt_count >= max_attempts is
        # discovered on a step other than the initial None branch, the
        # sequence force-escalates with next_action_at == now exactly (not
        # now + offset), because "escalate immediately" is the intended
        # product behavior, not a retry that needs a wait window.
        #
        # This DOES mean plan_mandate_retry_sequence can hand back a
        # candidate that fails the strict "> occurred_at" test -- if this
        # scenario is ever reached live (a caller reports an already-exhausted
        # mandate on its very first call for that mandate, so
        # attempts_so_far computed from PolicyDecision history is still below
        # MAX_RETRY_ATTEMPTS and doesn't block it first), compliance's
        # candidate_not_after_event check will BLOCK this escalation rather
        # than allow it -- see
        # TestMandateCandidateTimingInvariant::test_equal_or_earlier_candidate_is_blocked_not_allowed
        # in tests/test_compliance_v2.py, which proves nothing with this
        # shape is ever ALLOWED. It is flagged to the user as a known rough
        # edge (misleading block reason, escalation never actually fires
        # automatically in that specific scenario) rather than silently
        # patched, since fixing the *scheduling* is a product-behavior change
        # outside what was asked here.
        result = plan_mandate_retry_sequence(current_step="attempt_1", attempt_count=3, max_attempts=3, now=NOW)
        assert result.next_action_at == NOW  # exact equality, not "> NOW" -- this is the one place that's true
