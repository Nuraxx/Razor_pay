"""
Track-03 end-to-end revenue-risk orchestrator -- the single place that wires
together the new domains' modules, mirroring recovery/orchestrator.py's own
sequencing but for checkout_abandoned / mandate_failed / receivable_overdue /
promise_to_pay_broken:

    revenue_risk_event
        -> policy decision   (policy/revenue_recovery_policy.py -- domain rule
                               module combines classification + candidate
                               selection in one step for these domains)
        -> compliance gate   (policy/compliance_v2.py -- ALLOWED/BLOCKED/HUMAN_REVIEW)
        -> primary action    (recorded, never actually executed -- no live action)
        -> communication     (llm/service.py Job 1 text, or Job 4 voice script +
                               recovery/voice.py::VoiceRecoveryProvider -- ONLY
                               if compliance allows it)
        -> outcome           (app.models.RecoveryOutcome -- ALWAYS "PENDING"/
                               "unconfirmed_pending" here; this backend never
                               confirms a real payment succeeded)
        -> audit trail       (actor values: revenue_policy/revenue_compliance/
                               revenue_orchestrator/voice)

Zero shared mutable code with recovery/orchestrator.py::orchestrate_recovery
-- that function, recovery/schemas.py, and recovery/webhook_pipeline.py are
untouched by this module. payment_failed / subscription_payment_failed never
reach this orchestrator (see recovery/orchestrator.py for their path).

THE LLM CANNOT AFFECT THE PRIMARY ACTION: policy_row is fully computed and
persisted BEFORE any LLM/voice code runs here, exactly like
recovery/orchestrator.py's own invariant. An LLM/voice failure only ever
changes `communication_action` / `llm_success` / (at most) `final_status`'s
communication-related values -- never `primary_action`, `selected_candidate_type`,
or `payment_verdict`.

ID NAMESPACING: app.models.AuditLog.failure_event_id and
app.models.LLMInvocation.event_id are BOTH also written by the existing,
untouched payment_failed path (recovery/orchestrator.py) using
FailureEvent.id values -- neither column has a unique constraint, so unlike
policy_decisions.event_id (see policy/policy_decision_store.py's own
docstring) a collision doesn't crash, but it DOES silently corrupt the
communication_already_sent idempotency check: a brand-new revenue-risk event
can spuriously look like it "already sent" communication because an
unrelated old payment_failed event happened to reuse the same small
event_id. Every write/read of these two columns below uses `stored_event_id`
(event.event_id + REVENUE_DOMAIN_EVENT_ID_OFFSET), never the raw
event.event_id, for exactly this reason -- confirmed via a real live-DB
run where this bug reproduced before the fix (revenue_risk_events.id=1
collided with a pre-existing llm_invocations.event_id=1 row from the
unrelated payment_failed demo data). app.models.RecoveryOutcome.event_id
does NOT need this treatment -- this module is the only writer of that
table, and its own lookup already disambiguates on (event_id, event_type)
together.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.logging_config import log
from app.models import AuditLog, LLMInvocation, PolicyDecision, RecoveryOutcome
from llm.client import LLMClient
from llm.service import generate_outreach_microcopy_and_log, generate_voice_script_and_log
from policy.compliance_v2 import GeneralizedComplianceContext, evaluate_compliance_v2
from policy.decision_engine import NO_ACTION
from policy.policy_decision_store import REVENUE_DOMAIN_EVENT_ID_OFFSET
from policy.revenue_recovery_policy import decide_for_revenue_risk_event
from recovery.revenue_schemas import RevenueRecoveryResult, RevenueRiskEventInput
from recovery.voice import MockVoiceProvider, VoiceRecoveryProvider

# Candidates that exist but represent "nothing to communicate about yet"
# (checkout's CHECKOUT_STALLED "wait" candidate) -- treated like NO_ACTION
# for communication purposes only; the primary action is still recorded as
# a real, non-blocked "wait" outcome, not folded into NO_ACTION's own status.
_NO_COMMUNICATION_YET_CANDIDATES = frozenset({NO_ACTION, "wait"})

_WINDOW_DESCRIPTIONS = {
    "reminder": "shortly", "payment_link_reminder": "with a payment link shortly",
    "retry_checkout": "shortly", "alternate_payment_method": "shortly",
    "attempt_1": "shortly", "attempt_2": "soon", "final_attempt": "as a final attempt",
    "communication": "soon", "escalation": "as an escalation",
    "friendly_reminder": "shortly", "payment_request": "shortly",
    "promise_to_pay_request": "shortly", "human_handoff": "via a human follow-up",
    "urgent_reminder": "urgently", "final_notice": "as a final notice",
}


def _describe_window(candidate_type: str) -> str | None:
    return _WINDOW_DESCRIPTIONS.get(candidate_type)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def orchestrate_revenue_event(
    db: Session,
    event: RevenueRiskEventInput,
    *,
    model: dict | None = None,
    llm_client: LLMClient | None = None,
    voice_provider: VoiceRecoveryProvider | None = None,
) -> RevenueRecoveryResult:
    voice_provider = voice_provider or MockVoiceProvider()
    stored_event_id = event.event_id + REVENUE_DOMAIN_EVENT_ID_OFFSET  # see module docstring's ID NAMESPACING note

    # --- 1. Policy decision (domain rule module, idempotent + self-auditing) ---
    policy_row, policy_created, requires_human_review, human_review_reason = decide_for_revenue_risk_event(
        db, event_type=event.event_type, event_id=event.event_id, customer_ref=event.customer_ref,
        occurred_at=event.occurred_at, amount=event.amount, domain_context=event.domain_context or {}, model=model,
    )

    # --- 2. Compliance gate ------------------------------------------------
    task_name = "voice_script_generation" if event.channel == "voice" else "outreach_microcopy"
    attempts_so_far = (
        db.query(PolicyDecision)
        .filter(PolicyDecision.subscription_id == event.customer_ref, PolicyDecision.selected_candidate_type != NO_ACTION, PolicyDecision.id != policy_row.id)
        .count()
    )
    communication_already_sent = (
        db.query(LLMInvocation)
        .filter(LLMInvocation.event_id == stored_event_id, LLMInvocation.task_name == task_name)
        .count()
        > 0
    )
    compliance = evaluate_compliance_v2(GeneralizedComplianceContext(
        event_type=event.event_type,
        classification_bucket=policy_row.classification_bucket or "",
        selected_candidate_type=policy_row.selected_candidate_type,
        selected_candidate_datetime=policy_row.selected_candidate_datetime,
        occurred_at=event.occurred_at,
        attempts_so_far=attempts_so_far,
        payment_already_decided=not policy_created,
        communication_already_sent=communication_already_sent,
        customer_opted_out=event.customer_opted_out,
        consent_for_communication=event.consent_for_communication,
        required_fields_present=event.required_fields_present,
        requires_human_review=requires_human_review,
        human_review_reason=human_review_reason,
    ))
    db.add(
        AuditLog(
            failure_event_id=stored_event_id,
            action="revenue_orchestrator_compliance",
            reason=(
                f"payment_verdict={compliance.payment_verdict} payment_reason={compliance.payment_reason} | "
                f"communication_verdict={compliance.communication_verdict} communication_reason={compliance.communication_reason} | "
                f"rule_version={compliance.rule_version}"
            ),
            actor="revenue_compliance",
        )
    )
    db.commit()

    # --- 3. Primary action (recorded only -- no live action ever executed) ---
    # "wait" (checkout's CHECKOUT_STALLED candidate) is treated as
    # no_action-equivalent here regardless of its compliance verdict --
    # nothing is actually being scheduled yet, it's just "check back later".
    if policy_row.selected_candidate_type in _NO_COMMUNICATION_YET_CANDIDATES:
        primary_action = "no_action"
    elif compliance.payment_verdict == "HUMAN_REVIEW":
        primary_action = "human_review"
    elif compliance.payment_verdict == "ALLOWED":
        primary_action = "action_scheduled"
    else:
        primary_action = "blocked"

    # --- 4. Communication (Job 1 text, or Job 4 voice -- ONLY if allowed) ---
    llm_task_name: str | None = None
    llm_success: bool | None = None
    voice_call_result = None
    communication_eligible = policy_row.selected_candidate_type not in _NO_COMMUNICATION_YET_CANDIDATES

    if not event.request_communication or not communication_eligible:
        communication_action = "skipped"
    elif compliance.communication_verdict == "BLOCKED":
        communication_action = "blocked"
    elif compliance.communication_verdict == "HUMAN_REVIEW":
        # Held pending a human, never auto-sent -- HUMAN_REVIEW is not ALLOWED.
        communication_action = "skipped"
    else:
        will_retry = compliance.payment_verdict == "ALLOWED"
        retry_window_description = _describe_window(policy_row.selected_candidate_type)
        if event.channel == "voice":
            llm_result, _invocation = generate_voice_script_and_log(
                db, event_id=stored_event_id, failure_bucket=policy_row.classification_bucket or "",
                customer_segment=event.customer_segment, language=event.language, will_retry=will_retry,
                retry_window_description=retry_window_description, amount_rupees=event.amount, client=llm_client,
            )
            voice_call_result = voice_provider.place_call(llm_result.structured_result.get("script_text", ""), event.customer_ref)
            db.add(
                AuditLog(
                    failure_event_id=stored_event_id, action="voice_call_attempted",
                    reason=f"provider={voice_call_result.provider_name} attempted={voice_call_result.attempted} connected={voice_call_result.connected} audio_available={voice_call_result.audio_available}",
                    actor="voice",
                )
            )
            db.commit()
        else:
            llm_result, _invocation = generate_outreach_microcopy_and_log(
                db, event_id=stored_event_id, failure_bucket=policy_row.classification_bucket or "",
                customer_segment=event.customer_segment, language=event.language, will_retry=will_retry,
                retry_window_description=retry_window_description, amount_rupees=event.amount, client=llm_client,
            )
        llm_task_name = llm_result.task_name
        llm_success = llm_result.success
        communication_action = "sent" if llm_result.success else "fallback_used"
        if not llm_result.success:
            log.warning("Revenue-risk LLM communication failed for event_id=%s (%s) -- primary action unaffected, deterministic fallback used", event.event_id, llm_result.error_type)

    # --- 5. Final status (deterministic precedence, superset of ------------
    # --- recovery/schemas.py's table with one new HUMAN_REVIEW state) ------
    primary_action_blocked = (
        policy_row.selected_candidate_type not in _NO_COMMUNICATION_YET_CANDIDATES
        and compliance.payment_verdict == "BLOCKED"
    )
    if primary_action_blocked:
        final_status = "RETRY_BLOCKED"
    elif communication_action == "blocked":
        final_status = "COMMUNICATION_BLOCKED"
    elif compliance.payment_verdict == "HUMAN_REVIEW" or compliance.communication_verdict == "HUMAN_REVIEW":
        final_status = "HUMAN_REVIEW"
    elif policy_row.selected_candidate_type in _NO_COMMUNICATION_YET_CANDIDATES and communication_action == "skipped":
        final_status = "NO_ACTION"
    elif communication_action == "fallback_used":
        final_status = "LLM_FALLBACK"
    elif communication_action == "sent":
        final_status = "COMMUNICATION_ALLOWED"
    else:
        final_status = "RETRY_ALLOWED"

    result = RevenueRecoveryResult(
        event_id=event.event_id, event_type=event.event_type, customer_ref=event.customer_ref,
        classification_bucket=policy_row.classification_bucket or "", policy_version=policy_row.policy_version,
        selected_candidate_type=policy_row.selected_candidate_type, selected_candidate_datetime=policy_row.selected_candidate_datetime,
        payment_verdict=compliance.payment_verdict, payment_reason=compliance.payment_reason,
        communication_verdict=compliance.communication_verdict, communication_reason=compliance.communication_reason,
        primary_action=primary_action, communication_action=communication_action,
        llm_task_name=llm_task_name, llm_success=llm_success, final_status=final_status,
        created_at=_utcnow(), decision_source=policy_row.decision_source, decision_reason=policy_row.decision_reason,
        voice_call_result=voice_call_result,
    )

    # --- 6. Generalized outcome (ALWAYS PENDING/unconfirmed_pending here -- ---
    # --- this backend never confirms a real payment/action succeeded; only ---
    # --- recovery/demo_generator.py may ever mark RECOVERED/demo_synthetic) ---
    existing_outcome = (
        db.query(RecoveryOutcome)
        .filter(RecoveryOutcome.event_id == event.event_id, RecoveryOutcome.event_type == event.event_type)
        .first()
    )
    if existing_outcome is None:
        recovery_status = "NO_ACTION" if final_status == "NO_ACTION" else "PENDING"
        db.add(
            RecoveryOutcome(
                event_id=event.event_id, event_type=event.event_type, at_risk_amount=event.amount,
                recovered_amount=None, retained_amount=None, lost_amount=None,
                recovery_status=recovery_status, confirmed_by="unconfirmed_pending",
            )
        )

    db.add(
        AuditLog(
            failure_event_id=stored_event_id,
            action="revenue_orchestrator_final_status",
            reason=(
                f"final_status={final_status} primary_action={primary_action} communication_action={communication_action} "
                f"llm_task_name={llm_task_name} llm_success={llm_success} requires_human_review={requires_human_review}"
            ),
            actor="revenue_orchestrator",
        )
    )
    db.commit()

    log.info("Revenue-risk orchestration complete for event_id=%s (event_type=%s): final_status=%s", event.event_id, event.event_type, final_status)
    return result
