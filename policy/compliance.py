"""
Deterministic compliance gate.

Sits BETWEEN policy selection (policy/decision_engine_v4.py, policy-v4 --
completely unmodified here) and execution (payment retry / LLM
communication). No ML, no LLM, no randomness -- same "deterministic,
explainable" design as policy/guardrails.py, which this module
reuses rather than reimplements wherever the same check applies.

IMPORTANT WORDING (required verbatim by the brief, also in README §18):
Compliance checks in this prototype are deterministic project guardrails.
They are not presented as a complete legal/regulatory compliance
implementation. Every rule below is either (a) a guardrail this project
already enforces elsewhere (classification gate, max attempts, candidate
horizon, idempotency -- reused, not reinvented) or (b) a new but equally
un-invented, clearly-labeled PROJECT guardrail (opt-out/cancellation
respect, consent-for-communication, required-fields presence). Nothing here
claims to satisfy DPDP/TRAI/RBI or any other real regulatory regime.

WHY A SEPARATE GATE, GIVEN POLICY ALREADY ENFORCES SOME OF THIS: policy's
own guardrails (policy/guardrails.py) protect POLICY's own decision-making
-- they stop `decide_engine_v4` from ever selecting a bad candidate in the
first place. Compliance is a second, independent checkpoint that
re-validates a decision ALREADY MADE before it is allowed to actually
execute (payment retry) or communicate (LLM outreach) -- the standard
separation between "what should we do" (policy) and "are we allowed to
actually do it, right now" (compliance) in any decision system with a
downstream execution step. It also carries checks policy has no visibility
into at all (opt-out, consent, required-field completeness at the
orchestration layer).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from policy.contact_hours import ContactHoursConfig, default_contact_hours_config, is_within_contact_hours, next_contact_hours_start
from policy.decision_engine import NO_ACTION
from policy.guardrails import MAX_CANDIDATE_HORIZON_DAYS, MAX_RETRY_ATTEMPTS, is_classification_allowed

COMPLIANCE_RULE_VERSION = "compliance-v1"


@dataclass(frozen=True)
class ComplianceContext:
    """Everything the compliance gate needs, already computed upstream by
    the orchestrator (recovery/orchestrator.py) -- this module never
    queries a database or calls policy/classification itself, keeping it a
    pure, easily-testable function of its inputs."""

    classification_bucket: str
    selected_candidate_type: str  # "NO_ACTION" or one of policy/retry_candidates.py's CANDIDATE_TYPES
    selected_candidate_datetime: datetime | None
    failure_timestamp: datetime
    attempts_so_far: int
    payment_already_decided: bool = False  # a policy_decisions row for this event_id already existed (policy-layer idempotency)
    communication_already_sent: bool = False  # an llm_invocations row for this event_id + task already existed (LLM-layer idempotency, applied at the compliance layer)
    customer_opted_out: bool = False  # PROJECT guardrail: an explicit opt-out signal (not derived from classification alone -- see evaluate_compliance)
    consent_for_communication: bool = True  # PROJECT guardrail placeholder: no real consent-tracking system exists in this project yet; defaults to True (not gated) unless the caller has an actual reason to set it False. Documented, not invented as a legal requirement.
    required_fields_present: bool = True  # event_id / subscription_id / amount / etc. are all non-null


@dataclass(frozen=True)
class ComplianceResult:
    """Structured, explainable result (brief section 2). `allowed`/`reason`
    are the minimal `{allowed, reason, rule_version}` shape the brief asks
    for, aliased to the PAYMENT gate specifically (the primary gated
    action); `payment_action_allowed` / `communication_action_allowed` are
    the two independently-computed gates (brief section 3)."""

    payment_action_allowed: bool
    payment_reason: str
    communication_action_allowed: bool
    communication_reason: str
    rule_version: str
    # DEFER, DON'T TERMINATE (final pre-submission audit): set ONLY when
    # communication was blocked SPECIFICALLY by contact-hours -- never for
    # an opt-out/consent/duplicate block, which re-trying later can never
    # fix. None in every other case, including when communication is simply
    # allowed. See policy/contact_hours.py::next_contact_hours_start and
    # recovery/retry_sweep.py (which fires the deferred communication once
    # this time arrives).
    communication_deferred_until: datetime | None = None

    @property
    def allowed(self) -> bool:
        return self.payment_action_allowed

    @property
    def reason(self) -> str:
        return self.payment_reason

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "rule_version": self.rule_version,
            "payment_action_allowed": self.payment_action_allowed,
            "payment_reason": self.payment_reason,
            "communication_action_allowed": self.communication_action_allowed,
            "communication_reason": self.communication_reason,
            "communication_deferred_until": self.communication_deferred_until.isoformat() if self.communication_deferred_until else None,
        }


def _candidate_time_is_valid(candidate_datetime: datetime, failure_timestamp: datetime) -> tuple[bool, str | None]:
    """Re-derives policy/guardrails.py::validate_candidate's two checks
    directly against a (type, datetime) pair rather than a full Candidate
    object (which this layer doesn't have -- it only sees policy's already-
    selected output). Reuses MAX_CANDIDATE_HORIZON_DAYS as the single
    source of truth for the threshold rather than a second literal."""
    if candidate_datetime <= failure_timestamp:
        return False, "candidate_not_after_failure"
    if candidate_datetime > failure_timestamp + timedelta(days=MAX_CANDIDATE_HORIZON_DAYS):
        return False, f"candidate_beyond_{MAX_CANDIDATE_HORIZON_DAYS}_day_recovery_horizon"
    return True, None


def evaluate_compliance(context: ComplianceContext, contact_hours_config: ContactHoursConfig | None = None) -> ComplianceResult:
    """Evaluates every rule in brief section 2's minimum list. Payment and
    communication are gated independently (brief section 3) -- a rule that
    blocks one does not automatically block the other unless the rule
    itself is about both (required_fields_present, duplicate checks).

    `contact_hours_config` defaults to app/config.py::settings (via
    policy/contact_hours.py::default_contact_hours_config) when omitted --
    injectable here purely for deterministic testing, same pattern as
    `model` / `llm_client` elsewhere in this codebase. Checks
    `context.selected_candidate_datetime` (the SCHEDULED action's own time),
    never the current process clock. Applied to the COMMUNICATION gate only
    -- "contact hours" (TRAI's own term for its commercial-communication
    window) governs when it is acceptable to reach out to a customer
    (SMS/WhatsApp/call); a backend payment-retry API call does not itself
    contact anyone and is not scoped by that concept, so the payment gate is
    intentionally left unchanged."""
    hours_config = contact_hours_config or default_contact_hours_config()

    # customer_cancelled is treated as an automatic opt-out signal -- this
    # is DERIVED from the project's own existing classification vocabulary
    # (classification/rules.py's CUSTOMER_CANCELLED bucket), not an
    # invented new concept. `customer_opted_out` additionally allows an
    # explicit opt-out independent of classification (brief section 3's
    # own example: a retryable_soft customer who separately opted out).
    is_cancelled = context.classification_bucket == "customer_cancelled"
    opted_out = context.customer_opted_out or is_cancelled

    # --- Payment-action gate --------------------------------------------
    if not context.required_fields_present:
        payment_allowed, payment_reason = False, "required_fields_missing"
    elif context.payment_already_decided:
        payment_allowed, payment_reason = False, "duplicate_payment_action_blocked"
    elif not is_classification_allowed(context.classification_bucket):
        payment_allowed, payment_reason = False, f"classification_not_retryable_soft: bucket={context.classification_bucket!r}"
    elif context.selected_candidate_type == NO_ACTION:
        payment_allowed, payment_reason = False, "policy_selected_no_action"
    elif context.attempts_so_far >= MAX_RETRY_ATTEMPTS:
        payment_allowed, payment_reason = False, f"max_retry_attempts_reached: {context.attempts_so_far} >= {MAX_RETRY_ATTEMPTS}"
    elif context.selected_candidate_datetime is None:
        payment_allowed, payment_reason = False, "missing_candidate_datetime"
    else:
        candidate_valid, invalid_reason = _candidate_time_is_valid(context.selected_candidate_datetime, context.failure_timestamp)
        if not candidate_valid:
            payment_allowed, payment_reason = False, f"invalid_candidate_time: {invalid_reason}"
        else:
            payment_allowed, payment_reason = True, "payment_action_allowed: all compliance checks passed"

    # --- Communication-action gate (independent of the above) -----------
    comm_within_hours, comm_hours_reason = (
        is_within_contact_hours(context.selected_candidate_datetime, hours_config)
        if context.selected_candidate_datetime is not None
        else (True, "no_candidate_datetime_to_check")
    )
    if not context.required_fields_present:
        comm_allowed, comm_reason = False, "required_fields_missing"
    elif context.communication_already_sent:
        comm_allowed, comm_reason = False, "duplicate_communication_action_blocked"
    elif opted_out:
        comm_allowed, comm_reason = False, "customer_opted_out_or_cancelled: outreach blocked"
    elif not context.consent_for_communication:
        comm_allowed, comm_reason = False, "consent_for_communication_missing"
    elif not comm_within_hours:
        comm_allowed, comm_reason = False, comm_hours_reason
    else:
        comm_allowed, comm_reason = True, "communication_action_allowed: all compliance checks passed"

    # DEFER, DON'T TERMINATE: only a pure contact-hours block gets a
    # deferred-until time -- an opt-out/consent/duplicate block is not a
    # timing problem, so there is nothing a later retry could fix.
    deferred_until = (
        next_contact_hours_start(context.selected_candidate_datetime, hours_config)
        if (not comm_allowed and not comm_within_hours and context.selected_candidate_datetime is not None)
        else None
    )

    return ComplianceResult(
        payment_action_allowed=payment_allowed,
        payment_reason=payment_reason,
        communication_action_allowed=comm_allowed,
        communication_reason=comm_reason,
        rule_version=COMPLIANCE_RULE_VERSION,
        communication_deferred_until=deferred_until,
    )
