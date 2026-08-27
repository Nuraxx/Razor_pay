"""
AI-assisted recovery policy.

Failed payment -> candidate retry times -> per-candidate scoring (calibrated
probability + heuristic adjustment, see policy/scoring.py)
-> hard guardrails (policy/guardrails.py) -> highest-scoring allowed
candidate, or NO_ACTION.

`decide()` is PURE: given a failure event's context, its classification
bucket, and however much guardrail state the caller already knows (prior
attempt count, whether a decision already exists), it always returns the
same DecisionResult for the same inputs -- deterministic, per the brief.
All persistence (querying policy_decisions history for that state, writing
policy_decisions + audit_log rows) lives in `decide_for_failure_event`, the
DB-aware wrapper -- mirroring the classification/rules.py (pure) +
classification/service.py (DB-aware) split.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.logging_config import log
from app.models import AuditLog, PolicyDecision
from policy.baselines import rule_based_baseline
from policy.guardrails import MAX_RETRY_ATTEMPTS, is_classification_allowed, validate_candidate
from policy.retry_candidates import generate_candidates
from policy.scoring import score_candidate, score_candidate_with_model_probability

POLICY_VERSION = "policy-v1"
POLICY_VERSION_CANDIDATE_AWARE = "policy-v2"  # see decide_candidate_aware() below
NO_ACTION = "NO_ACTION"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class DecisionResult:
    event_id: object
    subscription_id: str
    selected_candidate_type: str
    selected_candidate_datetime: datetime | None
    predicted_recovery_probability: float | None
    expected_recovery_value: float | None
    expected_incremental_value: float | None
    baseline_action: str
    policy_version: str
    decision_reason: str
    candidate_scores: list[dict] = field(default_factory=list, repr=False, compare=False)


def _no_action(event_id, subscription_id: str, baseline_action: str, reason: str, candidate_scores: list[dict] | None = None) -> DecisionResult:
    return DecisionResult(
        event_id=event_id,
        subscription_id=subscription_id,
        selected_candidate_type=NO_ACTION,
        selected_candidate_datetime=None,
        predicted_recovery_probability=None,
        expected_recovery_value=None,
        expected_incremental_value=None,
        baseline_action=baseline_action,
        policy_version=POLICY_VERSION,
        decision_reason=reason,
        candidate_scores=candidate_scores or [],
    )


def decide(
    event_id,
    subscription_id: str,
    failure_timestamp: datetime,
    amount: float,
    classification_bucket: str,
    base_probability: float,
    attempts_so_far: int = 0,
    already_decided: bool = False,
    intervention_cost: float = 0.0,
) -> DecisionResult:
    """
    Pure decision function -- see module docstring. `attempts_so_far` and
    `already_decided` are guardrail STATE the caller (decide_for_failure_event)
    is responsible for computing from persisted history; this function never
    queries anything itself.
    """
    baseline_action = rule_based_baseline(
        event_id, subscription_id, failure_timestamp, amount, classification_bucket, base_probability
    )["selected_candidate_type"]

    if already_decided:
        return _no_action(
            event_id, subscription_id, baseline_action,
            "duplicate_decision_skipped: a policy decision already exists for this event_id",
        )

    if not is_classification_allowed(classification_bucket):
        return _no_action(
            event_id, subscription_id, baseline_action,
            f"blocked_by_classification: bucket={classification_bucket!r} is not retryable_soft "
            "(hard_decline / customer_cancelled / unmapped never get a retry action)",
        )

    if attempts_so_far >= MAX_RETRY_ATTEMPTS:
        return _no_action(
            event_id, subscription_id, baseline_action,
            f"blocked_max_retry_attempts: {attempts_so_far} prior attempts >= limit of {MAX_RETRY_ATTEMPTS}",
        )

    candidates = generate_candidates(failure_timestamp)
    scored = []
    for candidate in candidates:
        is_valid, invalid_reason = validate_candidate(candidate, failure_timestamp)
        entry = score_candidate(base_probability, candidate, amount, intervention_cost)
        entry["valid"] = is_valid
        entry["invalid_reason"] = invalid_reason
        scored.append(entry)

    allowed = [s for s in scored if s["valid"]]
    if not allowed:
        return _no_action(
            event_id, subscription_id, baseline_action,
            "blocked_no_valid_candidates: every candidate failed validation",
            candidate_scores=scored,
        )

    best = max(allowed, key=lambda s: s["expected_incremental_value"])
    return DecisionResult(
        event_id=event_id,
        subscription_id=subscription_id,
        selected_candidate_type=best["candidate_type"],
        selected_candidate_datetime=best["candidate_datetime"],
        predicted_recovery_probability=best["predicted_recovery_probability"],
        expected_recovery_value=best["expected_recovery_value"],
        expected_incremental_value=best["expected_incremental_value"],
        baseline_action=baseline_action,
        policy_version=POLICY_VERSION,
        decision_reason=(
            f"selected {best['candidate_type']} at {best['candidate_datetime'].isoformat()} "
            f"(predicted_recovery_probability={best['predicted_recovery_probability']:.4f}, "
            f"expected_incremental_value={best['expected_incremental_value']:.2f})"
        ),
        candidate_scores=scored,
    )


def _serialize_candidate_scores(candidate_scores: list[dict]) -> str:
    def default(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"not JSON serializable: {type(obj)}")

    return json.dumps(candidate_scores, default=default)


def decide_for_failure_event(
    db: Session,
    event_id: int,
    subscription_id: str,
    failure_timestamp: datetime,
    amount: float,
    classification_bucket: str,
    base_probability: float,
    intervention_cost: float = 0.0,
) -> tuple[PolicyDecision | DecisionResult, bool]:
    """
    DB-aware wrapper around `decide()`. Queries policy_decisions for
    idempotency (has this event_id already been decided?) and for the
    subscription's prior attempt count (max-retry-attempts guardrail), then
    persists the result as a policy_decisions row plus an audit_log row.

    Returns (result, created) where created is False for a duplicate call --
    matching classification/service.py::classify_raw_event's convention.
    """
    existing = db.query(PolicyDecision).filter(PolicyDecision.event_id == event_id).first()
    if existing is not None:
        db.add(
            AuditLog(
                failure_event_id=event_id,
                action="policy_decision_skipped_duplicate",
                reason=(
                    f"event_id={event_id} already has policy_decisions.id={existing.id} "
                    f"(selected={existing.selected_candidate_type}); not re-decided."
                ),
                actor="policy",
            )
        )
        db.commit()
        log.info("Skipped duplicate policy decision for event_id=%s (already policy_decisions.id=%s)", event_id, existing.id)
        return existing, False

    prior_attempts = (
        db.query(PolicyDecision)
        .filter(
            PolicyDecision.subscription_id == subscription_id,
            PolicyDecision.selected_candidate_type != NO_ACTION,
        )
        .count()
    )

    result = decide(
        event_id=event_id,
        subscription_id=subscription_id,
        failure_timestamp=failure_timestamp,
        amount=amount,
        classification_bucket=classification_bucket,
        base_probability=base_probability,
        attempts_so_far=prior_attempts,
        already_decided=False,
        intervention_cost=intervention_cost,
    )

    decision_row = PolicyDecision(
        event_id=event_id,
        subscription_id=subscription_id,
        selected_candidate_type=result.selected_candidate_type,
        selected_candidate_datetime=result.selected_candidate_datetime,
        predicted_recovery_probability=result.predicted_recovery_probability,
        expected_recovery_value=result.expected_recovery_value,
        expected_incremental_value=result.expected_incremental_value,
        baseline_action=result.baseline_action,
        policy_version=result.policy_version,
        decision_reason=result.decision_reason,
    )
    db.add(decision_row)
    db.flush()  # populate decision_row.id before referencing it in the audit log

    action = "policy_no_action" if result.selected_candidate_type == NO_ACTION else "policy_decision_made"
    db.add(
        AuditLog(
            failure_event_id=event_id,
            action=action,
            reason=(
                f"{result.decision_reason} | candidate_scores="
                f"{_serialize_candidate_scores(result.candidate_scores)}"
            ),
            actor="policy",
        )
    )
    db.commit()

    log.info(
        "Policy decision for event_id=%s: %s (policy_decisions.id=%s)",
        event_id, result.selected_candidate_type, decision_row.id,
    )
    return decision_row, True


# ---------------------------------------------------------------------------
# Candidate-aware policy.
#
# Identical guardrails, selection rule, determinism, and audit/idempotency
# behavior to decide() / decide_for_failure_event() above -- the ONLY
# difference is where each candidate's predicted_recovery_probability comes
# from: a `candidate_probabilities` dict supplied by the caller (populated
# from model/train_candidate_model.py's genuinely candidate-aware model, via
# policy/scoring.py::predict_candidate_aware_recovery_probability), instead
# of one shared failure-time base_probability + a heuristic adjustment.
#
# Kept as separate functions rather than added parameters to decide() /
# decide_for_failure_event() so the heuristic policy's tested behavior is
# untouched byte-for-byte; the near-duplication below is deliberate, not an
# oversight.
# ---------------------------------------------------------------------------

def decide_candidate_aware(
    event_id,
    subscription_id: str,
    failure_timestamp: datetime,
    amount: float,
    classification_bucket: str,
    candidate_probabilities: dict[str, float],
    attempts_so_far: int = 0,
    already_decided: bool = False,
    intervention_cost: float = 0.0,
) -> DecisionResult:
    """Pure decision function -- see module docstring. `candidate_probabilities`
    must have one entry per policy.retry_candidates.CANDIDATE_TYPES key,
    already conditioned on that specific candidate (no further adjustment is
    applied here)."""
    baseline_action = rule_based_baseline(
        event_id, subscription_id, failure_timestamp, amount, classification_bucket,
        candidate_probabilities.get("plus_1_day_morning", 0.0),  # baseline only needs a probability for its own scoring, not the winner's
    )["selected_candidate_type"]

    if already_decided:
        return _no_action(
            event_id, subscription_id, baseline_action,
            "duplicate_decision_skipped: a policy decision already exists for this event_id",
        )

    if not is_classification_allowed(classification_bucket):
        return _no_action(
            event_id, subscription_id, baseline_action,
            f"blocked_by_classification: bucket={classification_bucket!r} is not retryable_soft "
            "(hard_decline / customer_cancelled / unmapped never get a retry action)",
        )

    if attempts_so_far >= MAX_RETRY_ATTEMPTS:
        return _no_action(
            event_id, subscription_id, baseline_action,
            f"blocked_max_retry_attempts: {attempts_so_far} prior attempts >= limit of {MAX_RETRY_ATTEMPTS}",
        )

    candidates = generate_candidates(failure_timestamp)
    scored = []
    for candidate in candidates:
        is_valid, invalid_reason = validate_candidate(candidate, failure_timestamp)
        prob = candidate_probabilities[candidate.candidate_type]
        entry = score_candidate_with_model_probability(prob, candidate, amount, intervention_cost)
        entry["valid"] = is_valid
        entry["invalid_reason"] = invalid_reason
        scored.append(entry)

    allowed = [s for s in scored if s["valid"]]
    if not allowed:
        return _no_action(
            event_id, subscription_id, baseline_action,
            "blocked_no_valid_candidates: every candidate failed validation",
            candidate_scores=scored,
        )

    best = max(allowed, key=lambda s: s["expected_incremental_value"])
    return DecisionResult(
        event_id=event_id,
        subscription_id=subscription_id,
        selected_candidate_type=best["candidate_type"],
        selected_candidate_datetime=best["candidate_datetime"],
        predicted_recovery_probability=best["predicted_recovery_probability"],
        expected_recovery_value=best["expected_recovery_value"],
        expected_incremental_value=best["expected_incremental_value"],
        baseline_action=baseline_action,
        policy_version=POLICY_VERSION_CANDIDATE_AWARE,
        decision_reason=(
            f"selected {best['candidate_type']} at {best['candidate_datetime'].isoformat()} "
            f"(predicted_recovery_probability={best['predicted_recovery_probability']:.4f}, "
            f"expected_incremental_value={best['expected_incremental_value']:.2f}, "
            f"candidate-aware model)"
        ),
        candidate_scores=scored,
    )


def decide_for_failure_event_candidate_aware(
    db: Session,
    event_id: int,
    subscription_id: str,
    failure_timestamp: datetime,
    amount: float,
    classification_bucket: str,
    candidate_probabilities: dict[str, float],
    intervention_cost: float = 0.0,
) -> tuple[PolicyDecision | DecisionResult, bool]:
    """DB-aware wrapper around `decide_candidate_aware()` -- same idempotency
    (by event_id, shared with decide_for_failure_event: an event_id decided
    once by either policy version can never be decided again) and the same
    max-retry-attempts accounting (prior non-NO_ACTION policy_decisions rows
    for the subscription, regardless of which policy_version made them --
    it's a real-world attempt count, not a per-model one)."""
    existing = db.query(PolicyDecision).filter(PolicyDecision.event_id == event_id).first()
    if existing is not None:
        db.add(
            AuditLog(
                failure_event_id=event_id,
                action="policy_decision_skipped_duplicate",
                reason=(
                    f"event_id={event_id} already has policy_decisions.id={existing.id} "
                    f"(selected={existing.selected_candidate_type}); not re-decided."
                ),
                actor="policy",
            )
        )
        db.commit()
        log.info("Skipped duplicate policy decision for event_id=%s (already policy_decisions.id=%s)", event_id, existing.id)
        return existing, False

    prior_attempts = (
        db.query(PolicyDecision)
        .filter(
            PolicyDecision.subscription_id == subscription_id,
            PolicyDecision.selected_candidate_type != NO_ACTION,
        )
        .count()
    )

    result = decide_candidate_aware(
        event_id=event_id,
        subscription_id=subscription_id,
        failure_timestamp=failure_timestamp,
        amount=amount,
        classification_bucket=classification_bucket,
        candidate_probabilities=candidate_probabilities,
        attempts_so_far=prior_attempts,
        already_decided=False,
        intervention_cost=intervention_cost,
    )

    decision_row = PolicyDecision(
        event_id=event_id,
        subscription_id=subscription_id,
        selected_candidate_type=result.selected_candidate_type,
        selected_candidate_datetime=result.selected_candidate_datetime,
        predicted_recovery_probability=result.predicted_recovery_probability,
        expected_recovery_value=result.expected_recovery_value,
        expected_incremental_value=result.expected_incremental_value,
        baseline_action=result.baseline_action,
        policy_version=result.policy_version,
        decision_reason=result.decision_reason,
    )
    db.add(decision_row)
    db.flush()

    action = "policy_no_action" if result.selected_candidate_type == NO_ACTION else "policy_decision_made"
    db.add(
        AuditLog(
            failure_event_id=event_id,
            action=action,
            reason=(
                f"{result.decision_reason} | candidate_scores="
                f"{_serialize_candidate_scores(result.candidate_scores)}"
            ),
            actor="policy",
        )
    )
    db.commit()

    log.info(
        "Candidate-aware policy decision for event_id=%s: %s (policy_decisions.id=%s)",
        event_id, result.selected_candidate_type, decision_row.id,
    )
    return decision_row, True
