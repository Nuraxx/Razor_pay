"""
Day-12 structured final result -- `RecoveryExecutionResult`, the single
object `recovery/orchestrator.py::orchestrate_recovery` returns.

`final_status` precedence (deterministic, evaluated in this fixed order --
see recovery/orchestrator.py for where it's actually computed):

    1. RETRY_BLOCKED        -- policy selected a REAL candidate (not
                               NO_ACTION), but compliance blocked the
                               payment action (max attempts, duplicate,
                               required fields, or an out-of-horizon time --
                               including a promise-to-pay override that
                               compliance rejected; see below). Outranks
                               POLICY_FALLBACK below: whether a BLOCKED
                               recommendation happened to be fallback-sourced
                               is strictly less important than the fact
                               that it is blocked.
    2. COMMUNICATION_BLOCKED-- communication was requested/applicable, but
                               compliance blocked it specifically (opt-out,
                               cancellation, consent, required fields,
                               duplicate) -- regardless of the payment
                               outcome. This is what a customer_cancelled
                               event reports; it is also what a hard_decline
                               event reports if the customer separately
                               opted out of messaging.
    3. NO_ACTION             -- payment is NO_ACTION AND no communication
                               was attempted either (customer_cancelled /
                               unmapped, or a hard_decline event where
                               communication was never requested).
    4. POLICY_FALLBACK      -- payment allowed and proceeding, but the
                               underlying decision came from Day-9/10's
                               rule-based fallback tier
                               (decision_source == "rule_based_fallback"),
                               not the primary model. Surfaced ahead of the
                               communication outcome because "this used the
                               fallback path" is a policy-confidence flag,
                               independent of what happened with messaging.
    5. LLM_FALLBACK         -- communication was attempted but the LLM call
                               itself failed and a deterministic fallback
                               message was used. Reachable for a REAL retry
                               AND for a hard_decline payment-method-update
                               nudge -- `payment_action` distinguishes which.
    6. COMMUNICATION_ALLOWED-- communication was attempted and the LLM call
                               succeeded. Same dual reachability as above:
                               `payment_action` says whether a retry was
                               also scheduled, or this was a hard_decline
                               nudge with `payment_action="no_action"`.
    7. RETRY_ALLOWED        -- payment allowed via the primary model;
                               communication was not requested (or not
                               applicable) this call.

**Promise-to-pay override** (recovery/promise_service.py +
recovery/orchestrator.py): when a VALID promise exists for this event and
policy did not select NO_ACTION, the orchestrator tries compliance against
the promise's own date FIRST. If compliance accepts it, the promise's
timing becomes the effective candidate everywhere above (payment action,
communication's retry-window description, `selected_candidate_type` /
`selected_candidate_datetime` on this result) -- reported via
`decision_source` staying whatever the underlying policy tier was, plus the
`promise_to_pay_applied` / `promise_to_pay_id` fields below, so the
original model choice is never hidden. If compliance rejects the promise's
timing (e.g. it falls outside the 14-day recovery horizon), the ORIGINAL,
already-valid model/policy candidate is used instead -- the promise never
causes RETRY_BLOCKED by itself; only compliance rejecting the ORIGINAL
candidate (for an unrelated reason, e.g. max attempts) does.

Each status is mutually exclusive by construction; the full detail
(`payment_action`, `communication_action`, `llm_success`, `compliance_*`,
`promise_to_pay_applied`) is still carried separately so no information is
actually lost by having a single coarse status field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

FinalStatus = Literal[
    "RETRY_ALLOWED",
    "RETRY_BLOCKED",
    "COMMUNICATION_ALLOWED",
    "COMMUNICATION_BLOCKED",
    "NO_ACTION",
    "POLICY_FALLBACK",
    "LLM_FALLBACK",
]

# PAYMENT ACTION (brief section 3) -- exactly one of these per orchestration call.
PaymentAction = Literal["retry_scheduled", "blocked", "no_action"]

# COMMUNICATION ACTION (brief section 3) -- exactly one of these per orchestration call.
CommunicationAction = Literal["sent", "fallback_used", "blocked", "skipped"]


@dataclass(frozen=True)
class RecoveryExecutionResult:
    event_id: int
    subscription_id: str
    classification_bucket: str
    policy_version: str
    selected_candidate_type: str
    selected_candidate_datetime: datetime | None
    compliance_allowed: bool  # brief's exact field name -- aliases ComplianceResult.payment_action_allowed
    compliance_reason: str
    payment_action: PaymentAction
    communication_action: CommunicationAction
    llm_task_name: str | None
    llm_success: bool | None
    final_status: FinalStatus
    created_at: datetime | None = field(default=None, compare=False)

    # Extra detail beyond the brief's minimum field list -- always present,
    # never required for the schema's own contract, but useful for the
    # demo/audit trail without inventing a second result type. compare=False
    # so equality (used by tests) is about the DECISION, not incidental detail.
    classification_confidence: float | None = field(default=None, compare=False)
    decision_source: str | None = field(default=None, compare=False)
    decision_reason: str | None = field(default=None, compare=False)
    communication_reason: str | None = field(default=None, compare=False)

    # -- Promise-to-pay (see module docstring). Never hides the original
    # model/policy choice: `original_candidate_type` / `original_candidate_datetime`
    # are always the policy-selected values, regardless of whether a promise
    # override was applied on top of them.
    promise_to_pay_applied: bool = field(default=False, compare=False)
    promise_to_pay_id: int | None = field(default=None, compare=False)
    original_candidate_type: str | None = field(default=None, compare=False)
    original_candidate_datetime: datetime | None = field(default=None, compare=False)

    def to_dict(self) -> dict:
        def _serialize(value):
            return value.isoformat() if isinstance(value, datetime) else value

        return {k: _serialize(v) for k, v in self.__dict__.items()}

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict())
