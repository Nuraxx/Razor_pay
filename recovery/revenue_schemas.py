"""
Track-03: input/result dataclasses for recovery/revenue_orchestrator.py.
Deliberately NOT reusing recovery/schemas.py's RecoveryEventInput /
RecoveryExecutionResult -- different field semantics (customer_ref not
subscription_id, occurred_at not failure_timestamp, primary_action not
payment_action, a 3-way payment_verdict/communication_verdict instead of a
2-way compliance_allowed bool) that would otherwise force overloading the
existing, tested dataclasses. recovery/schemas.py itself is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

FinalStatusV2 = Literal[
    "RETRY_BLOCKED", "COMMUNICATION_BLOCKED", "HUMAN_REVIEW", "NO_ACTION",
    "POLICY_FALLBACK", "LLM_FALLBACK", "COMMUNICATION_ALLOWED", "RETRY_ALLOWED",
]


@dataclass(frozen=True)
class RevenueRiskEventInput:
    """Everything orchestrate_revenue_event needs about one revenue-risk
    event. Same "plain dataclass of individually-named fields, no raw dict,
    no hidden synthetic field possible" pattern as
    recovery/orchestrator.py::RecoveryEventInput."""

    event_type: str  # "checkout_abandoned" | "mandate_failed" | "receivable_overdue" | "promise_to_pay_broken"
    event_id: int  # RevenueRiskEvent.id
    customer_ref: str
    occurred_at: datetime
    amount: float
    currency: str = "INR"
    domain_context: dict | None = None
    customer_segment: str = "unknown"
    language: str = "en"
    channel: str = "text"  # "text" | "voice"
    request_communication: bool = True
    customer_opted_out: bool = False
    consent_for_communication: bool = True
    required_fields_present: bool = True


@dataclass(frozen=True)
class RevenueRecoveryResult:
    event_id: int
    event_type: str
    customer_ref: str
    classification_bucket: str
    policy_version: str
    selected_candidate_type: str
    selected_candidate_datetime: datetime | None
    payment_verdict: str  # "ALLOWED" | "BLOCKED" | "HUMAN_REVIEW"
    payment_reason: str
    communication_verdict: str
    communication_reason: str
    primary_action: str  # domain-neutral name -- "action_scheduled" | "blocked" | "no_action" | "human_review"
    communication_action: str  # "sent" | "fallback_used" | "blocked" | "skipped"
    llm_task_name: str | None
    llm_success: bool | None
    final_status: FinalStatusV2
    created_at: datetime | None = field(default=None, compare=False)

    decision_source: str | None = field(default=None, compare=False)
    decision_reason: str | None = field(default=None, compare=False)
    communication_reason_detail: str | None = field(default=None, compare=False)
    voice_call_result: object | None = field(default=None, compare=False)  # recovery.voice.VoiceCallResult, when channel="voice"

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id, "event_type": self.event_type, "customer_ref": self.customer_ref,
            "classification_bucket": self.classification_bucket, "policy_version": self.policy_version,
            "selected_candidate_type": self.selected_candidate_type,
            "selected_candidate_datetime": self.selected_candidate_datetime.isoformat() if self.selected_candidate_datetime else None,
            "payment_verdict": self.payment_verdict, "payment_reason": self.payment_reason,
            "communication_verdict": self.communication_verdict, "communication_reason": self.communication_reason,
            "primary_action": self.primary_action, "communication_action": self.communication_action,
            "llm_task_name": self.llm_task_name, "llm_success": self.llm_success, "final_status": self.final_status,
        }
