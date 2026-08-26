"""
Track-03: the ONE unified policy-selection framework (brief: "Do NOT create
six independent policy engines"). `decide_for_revenue_risk_event` is the
single entry point every event type goes through.

For payment_failed / subscription_payment_failed, it calls
policy/decision_engine_v4.py::decide_for_failure_event_engine_v4 EXACTLY as
recovery/orchestrator.py already does today -- same function, same
arguments, completely unmodified. Existing policy-v4 behavior for payment
failures is therefore provably stable (see
tests/test_revenue_recovery_policy.py's delegation test).

For the new domains, it dispatches (plain dict, mirrors
classification/rules.py's style) to a small pure per-domain rule module,
translates that module's output into the shared `DomainDecision` shape, and
persists it via policy/policy_decision_store.py::persist_policy_decision.
"""
from __future__ import annotations

import dataclasses
from datetime import timedelta

from app.models import PolicyDecision
from model.unified_model import score_event_candidates
from policy.checkout_rules import decide_checkout_recovery
from policy.decision_engine import NO_ACTION
from policy.decision_engine_v4 import decide_for_failure_event_engine_v4
from policy.guardrails import MAX_RETRY_ATTEMPTS
from policy.mandate_rules import plan_mandate_retry_sequence
from policy.one_time_payment_rules import decide_one_time_payment_recovery
from policy.policy_decision_store import DomainDecision, persist_policy_decision
from policy.promise_broken_rules import decide_promise_broken_action
from policy.receivables_rules import decide_receivable_action

# payment_failed / subscription_payment_failed keep using RawEvent/FailureEvent
# and recovery/orchestrator.py::orchestrate_recovery exactly as before.
PAYMENT_FAILED_EVENT_TYPES = frozenset({"payment_failed", "subscription_payment_failed"})

# A generic "act soon" placeholder for domains whose rule module doesn't
# compute its own schedule (checkout, receivables -- these are near-term
# communications, not calendar-scored retries the way payment_failed's
# candidates are). Chosen well within policy/guardrails.py::MAX_CANDIDATE_HORIZON_DAYS
# so it always clears compliance's horizon check.
_DEFAULT_ACTION_DELAY = timedelta(hours=1)

# Candidates that mean "nothing eligible to recommend an intervention for
# right now" -- NO_ACTION (opted-out/unmapped/terminal/guardrail-blocked)
# and checkout's "wait" (CHECKOUT_STALLED; revenue_orchestrator.py already
# treats it as no_action-equivalent for communication purposes). When the
# rule decider returns either, ML is never consulted -- see
# decide_for_revenue_risk_event below.
_ML_SKIP_CANDIDATES = frozenset({NO_ACTION, "wait"})


def _decide_checkout(*, event_id, customer_ref, occurred_at, amount, domain_context) -> DomainDecision:
    result = decide_checkout_recovery(
        cart_amount=domain_context.get("cart_amount", amount),
        inactivity_minutes=domain_context.get("inactivity_minutes", 0.0),
        previous_outreach_count=domain_context.get("previous_outreach_count", 0),
        payment_method=domain_context.get("payment_method"),
    )
    # "wait" gets a real datetime too (even though nothing is actually
    # communicated for it -- recovery/revenue_orchestrator.py treats it as
    # no_action-equivalent) so compliance_v2 evaluates it as a normal,
    # structurally-valid candidate (ALLOWED) rather than spuriously BLOCKED
    # for "missing_candidate_datetime".
    has_action = result.candidate_type != NO_ACTION
    return DomainDecision(
        event_id=event_id, subscription_id=customer_ref, classification_bucket=result.state,
        selected_candidate_type=result.candidate_type,
        selected_candidate_datetime=(occurred_at + _DEFAULT_ACTION_DELAY) if has_action else None,
        policy_version=result.rule_version, decision_reason=result.eligibility_reason,
        decision_source="rule_checkout_abandoned",
    )


def _decide_mandate(*, event_id, customer_ref, occurred_at, amount, domain_context) -> DomainDecision:
    result = plan_mandate_retry_sequence(
        current_step=domain_context.get("current_step"),
        attempt_count=domain_context.get("attempt_count", 0),
        max_attempts=domain_context.get("max_attempts", 3),
        now=occurred_at,
        prior_terminal_failure=domain_context.get("prior_terminal_failure", False),
        compliance_blocked=domain_context.get("compliance_blocked", False),
    )
    return DomainDecision(
        event_id=event_id, subscription_id=customer_ref, classification_bucket=result.sequence_status,
        selected_candidate_type=result.next_action_type or NO_ACTION,
        selected_candidate_datetime=result.next_action_at,
        policy_version=result.rule_version, decision_reason=result.retry_reason,
        decision_source="rule_mandate_failed",
    )


def _decide_receivable(*, event_id, customer_ref, occurred_at, amount, domain_context) -> DomainDecision:
    result = decide_receivable_action(
        days_overdue=domain_context.get("days_overdue", 0),
        is_disputed=domain_context.get("is_disputed", False),
        has_active_promise=domain_context.get("has_active_promise", False),
    )
    has_action = result.candidate_type != NO_ACTION
    return DomainDecision(
        event_id=event_id, subscription_id=customer_ref, classification_bucket=result.bucket,
        selected_candidate_type=result.candidate_type,
        selected_candidate_datetime=(occurred_at + _DEFAULT_ACTION_DELAY) if has_action else None,
        policy_version=result.rule_version, decision_reason=result.reason,
        decision_source="rule_receivable_overdue",
        requires_human_review=result.requires_human_review,
        human_review_reason=(f"escalation_level={result.escalation_level}: {result.reason}" if result.requires_human_review else None),
    )


def _decide_one_time_payment(*, event_id, customer_ref, occurred_at, amount, domain_context) -> DomainDecision:
    """A payment.failed event with no subscription_id but with enough
    authoritative context (payment_id + amount) to act on -- e.g. a
    Razorpay Payment Link failure. See recovery/webhook_pipeline.py for the
    eligibility gate that runs before this domain is ever reached, and
    policy/one_time_payment_rules.py for why this reuses the exact same
    deterministic classifier every subscription payment failure uses."""
    result = decide_one_time_payment_recovery(
        error_code=domain_context.get("error_code"), error_reason=domain_context.get("error_reason"),
        error_source=domain_context.get("error_source"), error_step=domain_context.get("error_step"),
    )
    has_action = result.candidate_type != NO_ACTION
    return DomainDecision(
        event_id=event_id, subscription_id=customer_ref, classification_bucket=result.classification_bucket,
        selected_candidate_type=result.candidate_type,
        selected_candidate_datetime=(occurred_at + _DEFAULT_ACTION_DELAY) if has_action else None,
        policy_version=result.rule_version,
        decision_reason=f"{result.classification_reason} | {result.eligibility_reason}",
        decision_source="rule_one_time_payment_failed",
    )


def _decide_promise_broken(*, event_id, customer_ref, occurred_at, amount, domain_context) -> DomainDecision:
    result = decide_promise_broken_action(
        attempts_so_far=domain_context.get("attempts_so_far", 0),
        cumulative_payment_attempts=domain_context.get("cumulative_payment_attempts", 0),
        max_payment_attempts=domain_context.get("max_payment_attempts", MAX_RETRY_ATTEMPTS),
    )
    has_action = result.candidate_type != NO_ACTION
    return DomainDecision(
        event_id=event_id, subscription_id=customer_ref, classification_bucket="promise_broken",
        selected_candidate_type=result.candidate_type,
        selected_candidate_datetime=(occurred_at + _DEFAULT_ACTION_DELAY) if has_action else None,
        policy_version=result.rule_version, decision_reason=result.reason,
        decision_source="rule_promise_to_pay_broken",
    )


def _decide_unified_ml(*, event_type, event_id, customer_ref, occurred_at, amount, domain_context, model) -> DomainDecision | None:
    """Attempt a unified, event-agnostic ML recommendation when a model is
    supplied. This is intentionally additive: if the model is missing or the
    inference fails, the existing rule-based decision still remains the
    default path."""
    if model is None:
        return None
    try:
        candidate_scores = score_event_candidates({
            "event_type": event_type,
            "amount": amount,
            "currency": domain_context.get("currency", "INR"),
            "failure_reason": domain_context.get("failure_reason", "unknown"),
            "failure_code": domain_context.get("failure_code", "unknown"),
            "payment_method": domain_context.get("payment_method", "unknown"),
            "attempt_count": domain_context.get("attempt_count", 0),
            "prior_failure_count": domain_context.get("prior_failure_count", 0),
            "prior_recovery_rate": domain_context.get("prior_recovery_rate", 0.0),
            "customer_tenure": domain_context.get("customer_tenure", 0.0),
            "customer_segment": domain_context.get("customer_segment", "unknown"),
            "days_to_payday": domain_context.get("days_to_payday", 30.0),
            "days_since_last_activity": domain_context.get("days_since_last_activity", 7.0),
            **{k: domain_context.get(k) for k in (
                "subscription_age_days", "days_to_subscription_renewal", "checkout_age_minutes",
                "cart_value", "mandate_attempt_number", "days_overdue", "invoice_amount",
                "invoice_age_days", "promise_age_days", "promise_confidence"
            ) if k in domain_context},
        }, model=model)
    except Exception:
        return None
    if not candidate_scores:
        return None
    best = max(candidate_scores, key=lambda row: row["predicted_recovery_value"])
    return DomainDecision(
        event_id=event_id,
        subscription_id=customer_ref,
        classification_bucket=(domain_context.get("classification_bucket") or event_type),
        selected_candidate_type=best["candidate_type"],
        selected_candidate_datetime=(occurred_at + _DEFAULT_ACTION_DELAY),
        policy_version="unified-ml-v1",
        decision_reason=f"unified_model_score={best['predicted_recovery_probability']:.3f}; candidate={best['candidate_type']}",
        decision_source="ml_unified_v1",
        predicted_recovery_probability=best["predicted_recovery_probability"],
        expected_recovery_value=best["predicted_recovery_value"],
        expected_incremental_value=best["expected_incremental_value"],
        model_version=best.get("model_version", "unified_catboost_v1"),
    )


# Exact decision_source values the 4 new domains ever write -- deliberately
# NOT matched by a "rule_%" wildcard: the existing payment_failed fallback
# source is literally "rule_based_fallback" (policy/decision_engine.py::SOURCE_FALLBACK),
# which would collide with a loose prefix match. ui/data.py's revenue-risk
# queries filter on this exact set before joining PolicyDecision.event_id to
# RevenueRiskEvent.id -- those two columns share no real FK, so an imprecise
# filter could join an unrelated payment_failed row to a same-numbered
# RevenueRiskEvent by coincidence.
REVENUE_DOMAIN_DECISION_SOURCES = frozenset({
    "rule_checkout_abandoned", "rule_mandate_failed", "rule_receivable_overdue", "rule_promise_to_pay_broken",
    "rule_one_time_payment_failed",
    # The unified ML model's decision_source (see _decide_unified_ml above)
    # -- written for any of the 5 domains once a trained model is live.
    # Without this, a real ML-sourced decision would silently vanish from
    # every ui/data.py revenue-risk dashboard query (get_live_revenue_recovery_queue_df,
    # get_live_revenue_at_risk_kpis, etc.), which filter on this exact set.
    "ml_unified_v1",
})

_DOMAIN_DECIDERS = {
    "checkout_abandoned": _decide_checkout,
    "mandate_failed": _decide_mandate,
    "receivable_overdue": _decide_receivable,
    "promise_to_pay_broken": _decide_promise_broken,
    "payment_failed_no_subscription": _decide_one_time_payment,
}


def decide_for_revenue_risk_event(
    db, *, event_type: str, event_id: int, customer_ref: str, occurred_at, amount: float,
    domain_context: dict, classification_bucket: str | None = None, model: dict | None = None,
) -> tuple[PolicyDecision, bool, bool, str | None]:
    """Returns (row, created, requires_human_review, human_review_reason).
    The trailing two fields are always (False, None) for payment_failed/
    subscription_payment_failed -- decide_for_failure_event_engine_v4 never
    produces a human-review signal; only the new domains' rule modules do
    (e.g. a disputed receivable)."""
    if event_type in PAYMENT_FAILED_EVENT_TYPES:
        if classification_bucket is None:
            raise ValueError(f"classification_bucket is required for event_type={event_type!r}")
        row, created = decide_for_failure_event_engine_v4(
            db, event_id=event_id, subscription_id=customer_ref, failure_timestamp=occurred_at,
            amount=amount, classification_bucket=classification_bucket, failure_context=domain_context or {}, model=model,
        )
        return row, created, False, None

    decider = _DOMAIN_DECIDERS.get(event_type)
    if decider is None:
        raise ValueError(f"unknown event_type: {event_type!r}")

    # The rule-based decider is ALWAYS computed -- it is the authoritative
    # ELIGIBILITY gate (opted-out/unmapped/disputed/terminal/not-yet-
    # actionable), a policy/compliance concern ML never gets to decide.
    #
    # "should ML evaluate this event" and "should policy act on ML's
    # recommendation" are two DIFFERENT questions -- ML is therefore ALSO
    # always attempted (when a model is supplied), independently of what
    # the rule decider says. This keeps ML's recommendation visible in the
    # audit trail even when policy overrides it, rather than silently
    # skipping ML inference and reporting a bare rule decision as if ML had
    # never run at all.
    rule_decision = decider(event_id=event_id, customer_ref=customer_ref, occurred_at=occurred_at, amount=amount, domain_context=domain_context or {})
    ml_decision = _decide_unified_ml(
        event_type=event_type,
        event_id=event_id,
        customer_ref=customer_ref,
        occurred_at=occurred_at,
        amount=amount,
        domain_context=domain_context or {},
        model=model,
    )

    if rule_decision.selected_candidate_type in _ML_SKIP_CANDIDATES or rule_decision.requires_human_review:
        # Policy/eligibility is authoritative here -- ML's recommendation
        # (if it ran) is recorded for audit but never used as the final
        # candidate; it never gets a chance to fabricate an action where
        # the rule decider says there should be none, and it never gets to
        # override a human-review escalation. This is the Phase-13/14
        # policy/compliance boundary: ML recommends, policy decides.
        if ml_decision is not None:
            final_decision = dataclasses.replace(
                rule_decision,
                decision_reason=(
                    f"{rule_decision.decision_reason} | ml_consulted=True "
                    f"ml_recommendation={ml_decision.selected_candidate_type} "
                    f"ml_score={ml_decision.predicted_recovery_probability:.3f} "
                    f"ml_model={ml_decision.model_version} | policy_overrides_ml_due_to_eligibility"
                ),
                predicted_recovery_probability=ml_decision.predicted_recovery_probability,
                expected_recovery_value=ml_decision.expected_recovery_value,
                expected_incremental_value=ml_decision.expected_incremental_value,
                model_version=ml_decision.model_version,
            )
        else:
            final_decision = dataclasses.replace(rule_decision, decision_reason=f"{rule_decision.decision_reason} | ml_consulted=False")
        row, created = persist_policy_decision(db, final_decision)
        return row, created, final_decision.requires_human_review, final_decision.human_review_reason

    if ml_decision is not None:
        # Phase-13 policy boundary (part 2): even once ML IS consulted, the
        # audit trail still records what the rule decider's own candidate
        # was for the same event, so the two are always distinguishable
        # after the fact -- this is deliberately NOT a silent override.
        ml_decision = dataclasses.replace(
            ml_decision,
            # classification is a domain FACT (computed by the same
            # deterministic classifier every payment_failed path uses), not
            # an ML output -- ML only ever affects candidate selection here,
            # so the ML-sourced decision carries the SAME classification the
            # rule decider already computed, never event_type as a stand-in.
            classification_bucket=rule_decision.classification_bucket,
            decision_reason=f"{ml_decision.decision_reason} | rule_baseline_candidate={rule_decision.selected_candidate_type}",
        )
        row, created = persist_policy_decision(db, ml_decision)
        return row, created, ml_decision.requires_human_review, ml_decision.human_review_reason

    row, created = persist_policy_decision(db, rule_decision)
    return row, created, rule_decision.requires_human_review, rule_decision.human_review_reason
