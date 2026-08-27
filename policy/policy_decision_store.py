"""
Track-03: shared persistence for the new domains' policy decisions.

Deliberately a NEW function, not a refactor of
policy/decision_engine_v4.py::decide_for_failure_event_engine_v4's own
persistence block -- duplicating the idempotency-check/row-write/audit-row
pattern here keeps that tested function's code shape completely untouched
(same "policy-v4 doesn't touch policy-v3" precedent this codebase already sets).

ID NAMESPACING (critical): app.models.PolicyDecision.event_id carries a
database-level UNIQUE constraint. It was designed under the assumption that
"event_id" always means FailureEvent.id (one decision per failure event) --
but RevenueRiskEvent.id is a SEPARATE, independently-autoincrementing
sequence that also starts at 1. Writing a revenue-risk decision's raw
RevenueRiskEvent.id straight into this column would collide with an
existing payment_failed PolicyDecision row almost immediately in real use
(both sequences grow in lockstep from 1) and raise a hard IntegrityError --
not a silent bad join, a crash. REVENUE_DOMAIN_EVENT_ID_OFFSET pushes every
revenue-risk domain's stored event_id into a numeric range payment_failed's
FailureEvent.id will never reach, so the two id-spaces can safely share one
physical column. Every place that reads/writes/joins on
PolicyDecision.event_id for the new domains MUST apply this SAME offset
consistently -- see ui/data.py's revenue-risk queries, which add it to the
join condition rather than to RevenueRiskEvent.id itself (that table's own
id stays the real, un-offset value everywhere else)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.logging_config import log
from app.models import AuditLog, PolicyDecision
from policy.decision_engine import NO_ACTION

REVENUE_DOMAIN_EVENT_ID_OFFSET = 1_000_000_000


@dataclass(frozen=True)
class DomainDecision:
    """The one shape every new-domain rule module's output gets translated
    into before being persisted -- keeps persist_policy_decision() a single,
    uniform function regardless of which domain produced the decision."""

    event_id: int
    subscription_id: str  # holds customer_ref (cart customer / mandate subscription / receivable account) -- policy_decisions.subscription_id column reused, not renamed
    classification_bucket: str  # domain-specific bucket/state string (e.g. "overdue_high", "ABANDONED")
    selected_candidate_type: str  # domain-specific candidate string, or NO_ACTION
    selected_candidate_datetime: datetime | None
    policy_version: str
    decision_reason: str
    decision_source: str  # e.g. "rule_checkout_abandoned"
    requires_human_review: bool = False
    human_review_reason: str | None = None
    predicted_recovery_probability: float | None = None
    expected_recovery_value: float | None = None
    expected_incremental_value: float | None = None
    model_version: str | None = None


def persist_policy_decision(db, decision: DomainDecision) -> tuple[PolicyDecision, bool]:
    """Idempotent by event_id -- same convention as decide_for_failure_event_engine_v4.
    Returns (row, created). The RETURNED row's .event_id carries the offset
    value (see module docstring) -- callers needing the real RevenueRiskEvent.id
    already have it via the DomainDecision/RevenueRiskEventInput they built
    themselves; nothing in this codebase reads it back off the PolicyDecision row."""
    stored_event_id = decision.event_id + REVENUE_DOMAIN_EVENT_ID_OFFSET
    existing = db.query(PolicyDecision).filter(PolicyDecision.event_id == stored_event_id).first()
    if existing is not None:
        db.add(
            AuditLog(
                failure_event_id=stored_event_id,
                action="policy_decision_skipped_duplicate",
                reason=f"event_id={decision.event_id} already has policy_decisions.id={existing.id} (selected={existing.selected_candidate_type}); not re-decided.",
                actor="revenue_policy",
            )
        )
        db.commit()
        log.info("Skipped duplicate revenue-policy decision for event_id=%s (already policy_decisions.id=%s)", decision.event_id, existing.id)
        return existing, False

    decision_row = PolicyDecision(
        event_id=stored_event_id,
        subscription_id=decision.subscription_id,
        selected_candidate_type=decision.selected_candidate_type,
        selected_candidate_datetime=decision.selected_candidate_datetime,
        predicted_recovery_probability=decision.predicted_recovery_probability,
        expected_recovery_value=decision.expected_recovery_value,
        expected_incremental_value=decision.expected_incremental_value,
        policy_version=decision.policy_version,
        decision_reason=decision.decision_reason,
        classification_bucket=decision.classification_bucket,
        decision_source=decision.decision_source,
        model_version=decision.model_version,
    )
    db.add(decision_row)
    db.flush()

    action = "policy_no_action" if decision.selected_candidate_type == NO_ACTION else "policy_decision_made"
    db.add(
        AuditLog(
            failure_event_id=stored_event_id,
            action=action,
            reason=(
                f"{decision.decision_reason} | decision_source={decision.decision_source} | "
                f"policy_version={decision.policy_version} | requires_human_review={decision.requires_human_review} | "
                f"human_review_reason={decision.human_review_reason}"
            ),
            actor="revenue_policy",
        )
    )
    db.commit()

    log.info(
        "Revenue-policy decision for event_id=%s: %s via %s (policy_decisions.id=%s)",
        decision.event_id, decision.selected_candidate_type, decision.decision_source, decision_row.id,
    )
    return decision_row, True
