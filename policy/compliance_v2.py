"""
Track-03 generalized compliance gate: ALLOWED / BLOCKED / HUMAN_REVIEW.

`policy/compliance.py` (Day-12, `evaluate_compliance`) stays byte-for-byte
UNCHANGED -- this module is strictly additive. For the existing
payment_failed / subscription_payment_failed event types, `evaluate_compliance_v2`
below delegates to the original `evaluate_compliance` unmodified and maps its
two booleans onto ALLOWED/BLOCKED -- it never invents a different verdict for
those event types, and never returns HUMAN_REVIEW for them (that state only
exists for the new domains, e.g. a B2B "disputed" receivable, that need a
third option `evaluate_compliance` was never designed to express).
`tests/test_compliance_v2.py::TestDelegationIsByteIdentical` proves the
delegated path produces the exact same reason strings `evaluate_compliance`
does for every case already covered in `tests/test_compliance.py` -- this is
a superset, not a reimplementation.

For the new revenue-risk domains (checkout_abandoned, mandate_failed,
receivable_overdue, promise_to_pay_broken), the same ordered-checks style is
used (first failure wins), with one addition: `requires_human_review` is
checked before the terminal BLOCKED checks on each gate independently, so a
domain rule module (e.g. policy/receivables_rules.py flagging a "disputed"
invoice) can route to HUMAN_REVIEW instead of a flat allow/deny. Gemini/LLM
never sets this flag and never sees it -- only the deterministic domain rule
modules do (recovery/revenue_orchestrator.py builds this context from their
output, never from an LLM result).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from policy.compliance import COMPLIANCE_RULE_VERSION, ComplianceContext, evaluate_compliance
from policy.contact_hours import ContactHoursConfig, default_contact_hours_config, is_within_contact_hours, next_contact_hours_start
from policy.decision_engine import NO_ACTION
from policy.guardrails import MAX_CANDIDATE_HORIZON_DAYS, MAX_RETRY_ATTEMPTS

COMPLIANCE_V2_RULE_VERSION = "compliance-v2"

# payment_failed / subscription_payment_failed keep using RawEvent/FailureEvent
# and recovery/orchestrator.py::orchestrate_recovery exactly as before; this
# set is what routes evaluate_compliance_v2 to the untouched legacy gate
# rather than the new domain-rules path.
PAYMENT_FAILED_EVENT_TYPES = frozenset({"payment_failed", "subscription_payment_failed"})

GateVerdict = Literal["ALLOWED", "BLOCKED", "HUMAN_REVIEW"]


@dataclass(frozen=True)
class GeneralizedComplianceContext:
    event_type: str
    classification_bucket: str
    selected_candidate_type: str
    selected_candidate_datetime: datetime | None
    occurred_at: datetime
    attempts_so_far: int
    payment_already_decided: bool = False
    communication_already_sent: bool = False
    customer_opted_out: bool = False
    consent_for_communication: bool = True
    required_fields_present: bool = True
    # Set ONLY by a deterministic domain rule module (never by the LLM) --
    # e.g. policy/receivables_rules.py flags a "disputed" invoice bucket.
    requires_human_review: bool = False
    human_review_reason: str | None = None


@dataclass(frozen=True)
class GeneralizedComplianceResult:
    payment_verdict: GateVerdict
    payment_reason: str
    communication_verdict: GateVerdict
    communication_reason: str
    rule_version: str
    # DEFER, DON'T TERMINATE -- see policy/compliance.py::ComplianceResult's
    # own field of the same name; identical semantics here.
    communication_deferred_until: datetime | None = None

    @property
    def payment_action_allowed(self) -> bool:
        return self.payment_verdict == "ALLOWED"

    @property
    def communication_action_allowed(self) -> bool:
        return self.communication_verdict == "ALLOWED"

    def to_dict(self) -> dict:
        return {
            "payment_verdict": self.payment_verdict,
            "payment_reason": self.payment_reason,
            "communication_verdict": self.communication_verdict,
            "communication_reason": self.communication_reason,
            "rule_version": self.rule_version,
            "payment_action_allowed": self.payment_action_allowed,
            "communication_action_allowed": self.communication_action_allowed,
            "communication_deferred_until": self.communication_deferred_until.isoformat() if self.communication_deferred_until else None,
        }


def _to_legacy_context(context: GeneralizedComplianceContext) -> ComplianceContext:
    return ComplianceContext(
        classification_bucket=context.classification_bucket,
        selected_candidate_type=context.selected_candidate_type,
        selected_candidate_datetime=context.selected_candidate_datetime,
        failure_timestamp=context.occurred_at,
        attempts_so_far=context.attempts_so_far,
        payment_already_decided=context.payment_already_decided,
        communication_already_sent=context.communication_already_sent,
        customer_opted_out=context.customer_opted_out,
        consent_for_communication=context.consent_for_communication,
        required_fields_present=context.required_fields_present,
    )


def _candidate_time_is_valid(candidate_datetime: datetime, occurred_at: datetime) -> tuple[bool, str | None]:
    """Same two checks as policy/compliance.py::_candidate_time_is_valid,
    generalized to `occurred_at` instead of `failure_timestamp` -- reuses the
    same MAX_CANDIDATE_HORIZON_DAYS threshold, no second literal."""
    if candidate_datetime <= occurred_at:
        return False, "candidate_not_after_event"
    if candidate_datetime > occurred_at + timedelta(days=MAX_CANDIDATE_HORIZON_DAYS):
        return False, f"candidate_beyond_{MAX_CANDIDATE_HORIZON_DAYS}_day_recovery_horizon"
    return True, None


def _evaluate_new_domain(context: GeneralizedComplianceContext, hours_config: ContactHoursConfig) -> GeneralizedComplianceResult:
    # --- "payment" gate -- i.e. the primary recovery action for this domain ---
    if not context.required_fields_present:
        payment_verdict, payment_reason = "BLOCKED", "required_fields_missing"
    elif context.payment_already_decided:
        payment_verdict, payment_reason = "BLOCKED", "duplicate_payment_action_blocked"
    elif context.requires_human_review:
        payment_verdict, payment_reason = "HUMAN_REVIEW", context.human_review_reason or "flagged_for_human_review"
    elif context.selected_candidate_type == NO_ACTION:
        payment_verdict, payment_reason = "BLOCKED", "policy_selected_no_action"
    elif context.attempts_so_far >= MAX_RETRY_ATTEMPTS:
        payment_verdict, payment_reason = "BLOCKED", f"max_retry_attempts_reached: {context.attempts_so_far} >= {MAX_RETRY_ATTEMPTS}"
    elif context.selected_candidate_datetime is None:
        payment_verdict, payment_reason = "BLOCKED", "missing_candidate_datetime"
    else:
        candidate_valid, invalid_reason = _candidate_time_is_valid(context.selected_candidate_datetime, context.occurred_at)
        if not candidate_valid:
            payment_verdict, payment_reason = "BLOCKED", f"invalid_candidate_time: {invalid_reason}"
        else:
            payment_verdict, payment_reason = "ALLOWED", "payment_action_allowed: all compliance checks passed"

    # --- communication gate (independent) --------------------------------
    is_cancelled = context.classification_bucket == "customer_cancelled"
    opted_out = context.customer_opted_out or is_cancelled
    comm_within_hours, comm_hours_reason = (
        is_within_contact_hours(context.selected_candidate_datetime, hours_config)
        if context.selected_candidate_datetime is not None
        else (True, "no_candidate_datetime_to_check")
    )

    if not context.required_fields_present:
        comm_verdict, comm_reason = "BLOCKED", "required_fields_missing"
    elif context.communication_already_sent:
        comm_verdict, comm_reason = "BLOCKED", "duplicate_communication_action_blocked"
    elif opted_out:
        comm_verdict, comm_reason = "BLOCKED", "customer_opted_out_or_cancelled: outreach blocked"
    elif context.requires_human_review:
        comm_verdict, comm_reason = "HUMAN_REVIEW", context.human_review_reason or "flagged_for_human_review"
    elif not context.consent_for_communication:
        comm_verdict, comm_reason = "BLOCKED", "consent_for_communication_missing"
    elif not comm_within_hours:
        comm_verdict, comm_reason = "BLOCKED", comm_hours_reason
    else:
        comm_verdict, comm_reason = "ALLOWED", "communication_action_allowed: all compliance checks passed"

    deferred_until = (
        next_contact_hours_start(context.selected_candidate_datetime, hours_config)
        if (comm_verdict == "BLOCKED" and not comm_within_hours and context.selected_candidate_datetime is not None)
        else None
    )

    return GeneralizedComplianceResult(
        payment_verdict=payment_verdict, payment_reason=payment_reason,
        communication_verdict=comm_verdict, communication_reason=comm_reason,
        rule_version=COMPLIANCE_V2_RULE_VERSION,
        communication_deferred_until=deferred_until,
    )


def evaluate_compliance_v2(context: GeneralizedComplianceContext, contact_hours_config: ContactHoursConfig | None = None) -> GeneralizedComplianceResult:
    """Single generalized compliance entry point (brief: "extend the
    existing compliance gate to all new event types"). Routes payment_failed
    / subscription_payment_failed to the exact, unmodified `evaluate_compliance`
    -- never a reimplementation -- and every other event_type to the new
    domain-rules path above, which additionally supports HUMAN_REVIEW.

    `contact_hours_config` defaults to app/config.py::settings when omitted
    (same injectable-for-testing pattern as evaluate_compliance) and is
    forwarded to the legacy gate too, so payment_failed / subscription_payment_failed
    events get the identical contact-hours behavior as every other domain."""
    hours_config = contact_hours_config or default_contact_hours_config()
    if context.event_type in PAYMENT_FAILED_EVENT_TYPES:
        legacy_result = evaluate_compliance(_to_legacy_context(context), hours_config)
        return GeneralizedComplianceResult(
            payment_verdict="ALLOWED" if legacy_result.payment_action_allowed else "BLOCKED",
            payment_reason=legacy_result.payment_reason,
            communication_verdict="ALLOWED" if legacy_result.communication_action_allowed else "BLOCKED",
            communication_reason=legacy_result.communication_reason,
            rule_version=legacy_result.rule_version,
            communication_deferred_until=legacy_result.communication_deferred_until,
        )
    return _evaluate_new_domain(context, hours_config)
