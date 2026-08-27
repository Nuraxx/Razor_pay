"""
Day-12 end-to-end recovery orchestrator -- the single place that wires
together every previous day's module in the order the brief specifies:

    failure_event
        -> classification            (classification/rules.py, Day 2, reused as-is)
        -> policy decision           (policy/decision_engine_v4.py, Day 10, reused as-is)
        -> promise-to-pay override   (recovery/promise_service.py, new -- see below)
        -> compliance gate           (policy/compliance.py, Day 12, reused as-is)
        -> payment action            (recorded, never actually executed -- no live Razorpay calls)
        -> LLM communication         (llm/service.py, Day 11, reused as-is -- ONLY if compliance allows it)
        -> audit trail               (app.models.AuditLog, actor values: classifier/policy/promise/compliance/llm/orchestrator)

This module contains NO decision logic of its own -- it does not classify,
does not score candidates, does not decide compliance, does not validate a
promise, and does not write prompts. Its only job is sequencing calls to
the modules that already do those things and assembling their outputs into
one `recovery.schemas.RecoveryExecutionResult`. See that module's docstring
for the `final_status` precedence rules.

THE LLM CANNOT AFFECT THE PAYMENT DECISION: `policy_row` (the Day-10
decision) is fully computed and persisted BEFORE any LLM code runs, and
nothing below ever reads an LLM result back into a policy/compliance field.
An LLM failure only changes `communication_action` / `llm_success` /
(at most) `final_status`'s communication-related values -- never
`payment_action`, `selected_candidate_type`, or `compliance_allowed`.

PROMISE-TO-PAY CANNOT BYPASS COMPLIANCE EITHER: a promise only ever
supplies an alternative CANDIDATE DATETIME for the SAME event's already-
selected action. That candidate is run through the exact same
`policy.compliance.evaluate_compliance` every model/rule-based candidate
already goes through -- max attempts, duplicate-decision, required fields,
and the 14-day recovery horizon all still apply, unmodified. If compliance
rejects the promise's timing, the orchestrator falls back to the original,
already-valid policy candidate rather than blocking the payment outright --
see `_apply_promise_override` below.

HARD DECLINES DO NOT GET RETRIED, BUT MAY STILL GET A NUDGE: policy already
guarantees `selected_candidate_type == NO_ACTION` for any non-`retryable_soft`
bucket (guardrails, unchanged). For `hard_decline` specifically, the
specification still calls for a "please update your payment method"
communication -- this orchestrator now attempts communication in that one
case too, reusing `generate_outreach_microcopy_and_log` with `will_retry=False`
(a code path that already existed in llm/service.py's fallback text and
prompt design, just never reached from here). `customer_cancelled` and
`unmapped` never get this treatment: compliance's own opt-out-on-cancellation
rule blocks the former, and there is nothing truthful to say for the
latter (an unmapped reason is, by design, never guessed at).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.logging_config import log
from app.models import AuditLog, LLMInvocation, PolicyDecision, PromiseToPay, RecoveryOutcome
from classification.rules import HARD_DECLINE, classify
from llm.client import LLMClient
from llm.service import generate_outreach_microcopy_and_log
from policy.compliance import ComplianceContext, ComplianceResult, evaluate_compliance
from policy.decision_engine import NO_ACTION, SOURCE_FALLBACK
from policy.decision_engine_v4 import decide_for_failure_event_engine_v4
from recovery.promise_service import get_active_promise
from recovery.schemas import RecoveryExecutionResult

# Used only for a promise-driven candidate -- deliberately NOT added to
# policy/retry_candidates.py::CANDIDATE_TYPES (that list is the model's own
# fixed set of calendar-computed candidates it scores; a promise is a
# fundamentally different, customer-stated kind of candidate that nothing
# in policy/ ever needs to generate, cost, or rank).
PROMISE_TO_PAY_CANDIDATE_TYPE = "promise_to_pay"

_WINDOW_DESCRIPTIONS = {
    "immediate": "within the hour",
    "plus_1_day_morning": "tomorrow morning",
    "payday_window": "around your next payday",
    "plus_3_days": "in a few days",
    "month_end_window": "around month-end",
    PROMISE_TO_PAY_CANDIDATE_TYPE: "on the date you told us",
}


def describe_retry_window(candidate_type: str) -> str | None:
    """Plain-language phrase for a candidate_type, passed to the LLM layer
    INSTEAD OF a raw datetime (llm/prompts.py's Job-1 prompt explicitly
    takes a pre-computed description, never a timestamp, to keep exact
    scheduling logic out of the LLM's hands)."""
    return _WINDOW_DESCRIPTIONS.get(candidate_type)


@dataclass(frozen=True)
class RecoveryEventInput:
    """Everything the orchestrator needs about one failure event. Deliberately
    a plain dataclass of individually-named fields (same pattern as
    llm/service.py's job functions) -- no raw dict, no hidden synthetic
    fields possible (archetype / recovery_probability_latent / etc. have no
    field here to occupy)."""

    event_id: int
    subscription_id: str
    failure_timestamp: datetime
    amount: float
    error_code: str | None
    error_reason: str | None
    error_source: str | None = None
    error_step: str | None = None
    failure_context: dict | None = None  # the 12 Model-B feature keys -- see model/latent_target_preprocessing.py::FEATURE_COLUMNS
    customer_segment: str = "unknown"
    language: str = "en"
    request_communication: bool = True
    customer_opted_out: bool = False
    consent_for_communication: bool = True
    required_fields_present: bool = True


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _apply_promise_override(
    db: Session,
    *,
    event_id: int,
    failure_timestamp: datetime,
    policy_row: PolicyDecision,
    compliance_context_kwargs: dict,
) -> tuple[str, datetime | None, ComplianceResult, PromiseToPay | None, bool]:
    """
    Returns (effective_candidate_type, effective_candidate_datetime,
    compliance_result, active_promise, promise_applied).

    If no VALID promise exists for this event, or policy itself already
    selected NO_ACTION (nothing for a promise to override), this is a
    single compliance evaluation against the policy candidate, unchanged
    from pre-promise behavior.

    If a VALID promise exists, compliance is evaluated TWICE: once against
    the promise's own date (a trial), and -- if that trial is rejected --
    once against the original policy candidate. `evaluate_compliance` is a
    pure function (no DB, no side effects), so this costs nothing and keeps
    the promise from ever being able to force an otherwise-invalid payment
    action through; it can only ever RETIME an already-valid one.
    """
    original_type = policy_row.selected_candidate_type
    original_datetime = policy_row.selected_candidate_datetime

    if original_type == NO_ACTION:
        compliance = evaluate_compliance(ComplianceContext(
            selected_candidate_type=original_type, selected_candidate_datetime=original_datetime,
            failure_timestamp=failure_timestamp, **compliance_context_kwargs,
        ))
        return original_type, original_datetime, compliance, None, False

    active_promise = get_active_promise(db, event_id)
    if active_promise is None:
        compliance = evaluate_compliance(ComplianceContext(
            selected_candidate_type=original_type, selected_candidate_datetime=original_datetime,
            failure_timestamp=failure_timestamp, **compliance_context_kwargs,
        ))
        return original_type, original_datetime, compliance, None, False

    trial_compliance = evaluate_compliance(ComplianceContext(
        selected_candidate_type=PROMISE_TO_PAY_CANDIDATE_TYPE, selected_candidate_datetime=active_promise.promised_date,
        failure_timestamp=failure_timestamp, **compliance_context_kwargs,
    ))
    if trial_compliance.payment_action_allowed:
        return PROMISE_TO_PAY_CANDIDATE_TYPE, active_promise.promised_date, trial_compliance, active_promise, True

    # Promise's own timing didn't clear compliance (e.g. outside the 14-day
    # recovery horizon) -- fall back to the original, already-valid policy
    # candidate rather than blocking the payment outright.
    fallback_compliance = evaluate_compliance(ComplianceContext(
        selected_candidate_type=original_type, selected_candidate_datetime=original_datetime,
        failure_timestamp=failure_timestamp, **compliance_context_kwargs,
    ))
    return original_type, original_datetime, fallback_compliance, active_promise, False


def orchestrate_recovery(
    db: Session,
    event: RecoveryEventInput,
    *,
    model: dict | None = None,
    llm_client: LLMClient | None = None,
) -> RecoveryExecutionResult:
    """The single entry point (brief section 4). `model` / `llm_client` are
    injectable purely for testing/offline use (same pattern as
    policy/decision_engine_v4.py and llm/service.py already establish) --
    production code can omit both and get the real Day-8 Model B / the
    configured LLM_PROVIDER."""

    # --- 1. Classification (Day 2, reused as-is) ------------------------
    classification = classify(event.error_code, event.error_reason, event.error_source, event.error_step)
    db.add(
        AuditLog(
            failure_event_id=event.event_id,
            action="orchestrator_classification",
            reason=f"bucket={classification.bucket} confidence={classification.confidence} rule_version={classification.rule_version} reason={classification.reason}",
            actor="classifier",
        )
    )
    db.commit()

    # --- 2. Policy decision (Day 10, reused as-is; already idempotent & ---
    # --- self-auditing -- see policy/decision_engine_v4.py::decide_for_failure_event_engine_v4) ---
    policy_row, policy_created = decide_for_failure_event_engine_v4(
        db, event_id=event.event_id, subscription_id=event.subscription_id, failure_timestamp=event.failure_timestamp,
        amount=event.amount, classification_bucket=classification.bucket, failure_context=event.failure_context or {},
        model=model,
    )

    # --- 3. Promise-to-pay override + compliance gate ---------------------
    attempts_so_far = (
        db.query(PolicyDecision)
        .filter(PolicyDecision.subscription_id == event.subscription_id, PolicyDecision.selected_candidate_type != NO_ACTION, PolicyDecision.id != policy_row.id)
        .count()
    )
    communication_already_sent = (
        db.query(LLMInvocation)
        .filter(LLMInvocation.event_id == event.event_id, LLMInvocation.task_name == "outreach_microcopy")
        .count()
        > 0
    )
    compliance_context_kwargs = dict(
        classification_bucket=classification.bucket,
        attempts_so_far=attempts_so_far,
        payment_already_decided=not policy_created,
        communication_already_sent=communication_already_sent,
        customer_opted_out=event.customer_opted_out,
        consent_for_communication=event.consent_for_communication,
        required_fields_present=event.required_fields_present,
    )
    effective_candidate_type, effective_candidate_datetime, compliance, active_promise, promise_applied = _apply_promise_override(
        db, event_id=event.event_id, failure_timestamp=event.failure_timestamp,
        policy_row=policy_row, compliance_context_kwargs=compliance_context_kwargs,
    )

    if active_promise is not None:
        promise_outcome = "accepted" if promise_applied else f"rejected_by_compliance: {compliance.payment_reason}"
        active_promise.override_applied = promise_applied
        active_promise.override_outcome = promise_outcome
        active_promise.updated_at = _utcnow()
        db.add(
            AuditLog(
                failure_event_id=event.event_id,
                action="promise_override_applied" if promise_applied else "promise_override_not_applied",
                reason=(
                    f"promises_to_pay.id={active_promise.id} | "
                    f"policy_candidate={policy_row.selected_candidate_type}@{policy_row.selected_candidate_datetime} | "
                    f"promise_candidate={PROMISE_TO_PAY_CANDIDATE_TYPE}@{active_promise.promised_date} | "
                    f"final_candidate={effective_candidate_type}@{effective_candidate_datetime} | "
                    f"outcome={promise_outcome} | policy_version={policy_row.policy_version} | compliance_rule_version={compliance.rule_version}"
                ),
                actor="promise",
            )
        )

    db.add(
        AuditLog(
            failure_event_id=event.event_id,
            action="orchestrator_compliance",
            reason=(
                f"payment_action_allowed={compliance.payment_action_allowed} payment_reason={compliance.payment_reason} | "
                f"communication_action_allowed={compliance.communication_action_allowed} communication_reason={compliance.communication_reason} | "
                f"rule_version={compliance.rule_version}"
            ),
            actor="compliance",
        )
    )
    db.commit()

    # --- 4. Payment action (recorded only -- no live Razorpay call, ------
    # --- brief explicitly forbids live integration this day) -------------
    if effective_candidate_type == NO_ACTION:
        payment_action = "no_action"
    elif compliance.payment_action_allowed:
        payment_action = "retry_scheduled"
    else:
        payment_action = "blocked"

    # --- 5. LLM communication (Day 11, reused as-is) -- ONLY when either ---
    # --- a real retry is proceeding, OR this is a hard_decline event (the ---
    # --- specification's payment-method-update nudge -- customer_cancelled ---
    # --- and unmapped never reach here: compliance's opt-out-on-cancellation ---
    # --- rule blocks the former, and there is nothing truthful to tell the ---
    # --- customer for the latter). This is the ONLY place llm/service.py is ---
    # --- called; nothing above depends on its result. -----------------------
    llm_task_name: str | None = None
    llm_success: bool | None = None
    communication_reason: str | None = None
    communication_eligible = effective_candidate_type != NO_ACTION or classification.bucket == HARD_DECLINE

    if not event.request_communication:
        communication_action = "skipped"
    elif not communication_eligible:
        communication_action = "skipped"
    elif not compliance.communication_action_allowed:
        # DEFER, DON'T TERMINATE (final pre-submission audit): a contact-
        # hours-only block gets "deferred" (recovery/retry_sweep.py fires it
        # once the window opens) instead of a permanent "blocked" -- every
        # other block reason (opt-out, consent, duplicate) is not a timing
        # problem and stays "blocked".
        if compliance.communication_deferred_until is not None:
            communication_action = "deferred"
        else:
            communication_action = "blocked"
        communication_reason = compliance.communication_reason
    else:
        will_retry = compliance.payment_action_allowed
        llm_result, _invocation = generate_outreach_microcopy_and_log(
            db, event_id=event.event_id, failure_bucket=classification.bucket, customer_segment=event.customer_segment,
            language=event.language, will_retry=will_retry, retry_window_description=describe_retry_window(effective_candidate_type),
            amount_rupees=event.amount, client=llm_client,
        )
        llm_task_name = llm_result.task_name
        llm_success = llm_result.success
        communication_action = "sent" if llm_result.success else "fallback_used"
        if not llm_result.success:
            log.warning("LLM communication failed for event_id=%s (%s) -- payment decision unaffected, deterministic fallback used", event.event_id, llm_result.error_type)

    # --- 6. Final status (deterministic precedence -- see recovery/schemas.py) ---
    payment_was_blocked = effective_candidate_type != NO_ACTION and not compliance.payment_action_allowed
    if payment_was_blocked:
        final_status = "RETRY_BLOCKED"
    elif communication_action == "blocked":
        final_status = "COMMUNICATION_BLOCKED"
    elif communication_action == "deferred":
        final_status = "COMMUNICATION_DEFERRED"
    elif effective_candidate_type == NO_ACTION and communication_action == "skipped":
        final_status = "NO_ACTION"
    elif policy_row.decision_source == SOURCE_FALLBACK:
        final_status = "POLICY_FALLBACK"
    elif communication_action == "fallback_used":
        final_status = "LLM_FALLBACK"
    elif communication_action == "sent":
        final_status = "COMMUNICATION_ALLOWED"
    else:
        final_status = "RETRY_ALLOWED"

    result = RecoveryExecutionResult(
        event_id=event.event_id,
        subscription_id=event.subscription_id,
        classification_bucket=classification.bucket,
        policy_version=policy_row.policy_version,
        selected_candidate_type=effective_candidate_type,
        selected_candidate_datetime=effective_candidate_datetime,
        compliance_allowed=compliance.payment_action_allowed,
        compliance_reason=compliance.payment_reason,
        payment_action=payment_action,
        communication_action=communication_action,
        llm_task_name=llm_task_name,
        llm_success=llm_success,
        final_status=final_status,
        created_at=_utcnow(),
        classification_confidence=classification.confidence,
        decision_source=policy_row.decision_source,
        decision_reason=policy_row.decision_reason,
        communication_reason=communication_reason,
        promise_to_pay_applied=promise_applied,
        promise_to_pay_id=active_promise.id if active_promise is not None else None,
        original_candidate_type=policy_row.selected_candidate_type,
        original_candidate_datetime=policy_row.selected_candidate_datetime,
        communication_deferred_until=compliance.communication_deferred_until if communication_action == "deferred" else None,
    )

    # DEFER, DON'T TERMINATE (final pre-submission audit): persist the
    # deferred-until time on the SAME policy_decisions row recovery/retry_sweep.py
    # already reads for multi-attempt persistence -- one sweep pass handles
    # both concerns. Only ever set once (communication_action can only be
    # "deferred" the FIRST time a given event is orchestrated -- a second
    # call for the same event_id short-circuits via `existing is not None` in
    # decide_for_failure_event_engine_v4, so this never overwrites an
    # already-firing deferred communication with a stale new one).
    if communication_action == "deferred":
        policy_row.communication_deferred_until = compliance.communication_deferred_until
        db.add(policy_row)

    # --- 6b. Generalized outcome tracking (Track-03, additive) -------------
    # app.models.RecoveryOutcome's own docstring: "the generalized revenue-
    # outcome model, shared by every domain (payment_failed included)" --
    # this backend never confirms a real Razorpay payment (payment_action is
    # recorded only, never executed), so this row stays honestly PENDING/
    # None/unconfirmed_pending for every live event, matching the exact same
    # binding rule recovery/revenue_orchestrator.py follows for the 4 newer
    # domains -- including its existence check, since orchestrate_recovery
    # can be invoked more than once for the same event_id (e.g. a manual
    # reprocess) and every other write in this function is already
    # idempotent; a bare insert here would silently duplicate outcome rows.
    existing_outcome = (
        db.query(RecoveryOutcome)
        .filter(RecoveryOutcome.event_id == event.event_id, RecoveryOutcome.event_type == "payment_failed")
        .first()
    )
    if existing_outcome is None:
        db.add(
            RecoveryOutcome(
                event_id=event.event_id,
                event_type="payment_failed",
                at_risk_amount=event.amount,
                recovered_amount=None,
                retained_amount=None,
                lost_amount=None,
                recovery_status="NO_ACTION" if final_status == "NO_ACTION" else "PENDING",
                confirmed_by="unconfirmed_pending",
            )
        )

    db.add(
        AuditLog(
            failure_event_id=event.event_id,
            action="orchestrator_final_status",
            reason=(
                f"final_status={final_status} payment_action={payment_action} communication_action={communication_action} "
                f"llm_task_name={llm_task_name} llm_success={llm_success} promise_to_pay_applied={promise_applied}"
            ),
            actor="orchestrator",
        )
    )
    db.commit()

    log.info("Orchestration complete for event_id=%s: final_status=%s", event.event_id, final_status)
    return result
