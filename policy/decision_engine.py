"""
policy-v3 production-shaped recovery decision engine.

Per the brief: NO new ML model. Model B's (predicts
`expected_recovery_value_latent` directly, in Rs -- see
model/latent_target_preprocessing.py) is used AS-IS, value-native (no
probability conversion -- unlike the candidate-aware/ranking/latent-target models' reuse of
`policy/recovery_policy.py::decide_candidate_aware`, which expects a
probability and multiplies by amount internally). This module is new
because that interface doesn't fit what policy-v3 needs: cost-aware net-value
scoring, deterministic abstention, and a rule-based fallback chain, none of
which `decide_candidate_aware` was built for.

Flow (brief section 1):

    candidates -> predicted_recovery_value (Model B) -> intervention_cost
    -> expected_net_value -> decision margin / abstention -> guardrails
    -> selected action OR NO_ACTION

FALLBACK CHAIN (brief section 6) -- three tiers, in order:

    1. PRIMARY: Model B, if available, error-free, well-formed output,
       and confident (decision margin >= threshold).
    2. FALLBACK: Rule-Based Retry (policy/baselines.py's existing,
       deterministic payday-proximity heuristic), tried whenever the
       primary tier is unavailable/erroring/malformed/under-confident.
    3. NO_ACTION: the final safety net, when even the fallback can't
       produce a valid, positive-net-value candidate.

Every decision records WHICH tier actually decided it (`decision_source`)
-- never silent. "Model confidence" here means only the deterministic
decision-margin abstention rule below; it is NOT calibrated probabilistic
uncertainty (see README "policy-v3").

All PREVIOUS guardrails (policy/guardrails.py, unchanged, reused directly)
still apply and are checked FIRST, before any model call: `retryable_soft`
only, no duplicate decision, max retry attempts, candidate must be valid
(after failure, within the 14-day horizon).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy.orm import Session

from app.logging_config import log
from app.models import AuditLog, PolicyDecision
from model.latent_target_preprocessing import FEATURE_COLUMNS, prepare_for_catboost
from model.train_latent_target_model import load_latent_target_model
from policy.baselines import rule_based_baseline
from policy.costs import DEFAULT_COSTS, InterventionCosts, cost_for_candidate
from policy.guardrails import MAX_RETRY_ATTEMPTS, is_classification_allowed, validate_candidate
from policy.retry_candidates import Candidate, generate_candidates

POLICY_VERSION = "policy-v3"
MODEL_VERSION = "subscription_value_model_catboost_regressor_v1"
NO_ACTION = "NO_ACTION"

SOURCE_MODEL = "subscription_value_model"
SOURCE_FALLBACK = "rule_based_fallback"
SOURCE_NO_ACTION = "no_action"

# Chosen via validation-only search over {0,10,25,50,100,150,200,250} -- see
# evaluation/evaluate_decision_engine.py::select_abstention_threshold_on_validation
# and README "policy-v3: threshold selection". Frozen here; never touched again
# after being set from that search (never re-tuned against test results).
# Search result on the 59 validation events (total latent Rs selected):
#   Rs0->18258.62  Rs10->18590.40 (best)  Rs25..250->17997.99 (flat, all tied)
DEFAULT_ABSTENTION_THRESHOLD_RS = 10.0

# A predicted recovery value more than this many multiples of the original
# amount is treated as malformed model output (a real recovery value can
# never legitimately exceed amount by construction of the synthetic
# target -- see model/latent_target_preprocessing.py -- so a prediction
# that does is a red flag, not a plausible high estimate). Deliberately
# generous (2x) so this only catches genuinely broken output, not
# legitimate model error/variance.
MAX_SANE_VALUE_MULTIPLE_OF_AMOUNT = 2.0

EVENT_FEATURE_KEYS = [
    "day_of_month",
    "days_to_nearest_payday_window",
    "prior_if_failure_count",
    "prior_if_self_resolved_rate",
    "tenure_days",
    "plan_tier",
    "primary_instrument",
    "city_tier",
    "bank_network_conditions",
    "issuing_bank_downtime_flag",
    "network_latency_bucket",
    "is_month_end_settlement_rush",
]  # `amount` is passed separately (also a guardrail/cost input, not just a feature)


class ModelUnavailableError(Exception):
    """Raised (and caught internally) when Model B's artifact can't be loaded."""


class MalformedModelOutputError(Exception):
    """Raised (and caught internally) when Model B's prediction is NaN, infinite, negative, or implausibly huge."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class CandidateScore:
    candidate_type: str
    candidate_datetime: datetime
    valid: bool
    invalid_reason: str | None
    predicted_recovery_value: float | None
    intervention_cost: float
    expected_net_value: float | None


@dataclass(frozen=True)
class Decision:
    """Structured, JSON-serializable decision record (brief section 7)."""

    event_id: object
    subscription_id: str
    classification_bucket: str
    selected_candidate_type: str
    selected_candidate_datetime: datetime | None
    predicted_recovery_value: float | None
    intervention_cost: float | None
    expected_net_value: float | None
    runner_up_value: float | None
    decision_margin: float | None
    decision_source: str
    model_version: str | None
    policy_version: str
    decision_reason: str
    created_at: datetime | None = field(default=None, compare=False)  # set only by the DB-aware wrapper -- see module docstring
    candidate_scores: list[CandidateScore] = field(default_factory=list, repr=False, compare=False)
    # -- Added policy-v4 (policy/decision_engine_v4.py). Always None for every
    # policy-v3/policy-v3 decision produced by this module -- v3's decide_engine()
    # below never sets them, so v3 semantics are unchanged bit-for-bit. Only
    # policy-v4 (decide_engine_v4) populates them, to record which config was
    # actually in effect for a decision (see app/models.py::PolicyDecision).
    margin_threshold_used: float | None = None
    fallback_advantage_threshold: float | None = None
    fallback_strategy: str | None = None

    def to_dict(self) -> dict:
        def _serialize(value):
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, CandidateScore):
                return {**value.__dict__, "candidate_datetime": value.candidate_datetime.isoformat()}
            return value

        return {k: _serialize(v) if not isinstance(v, list) else [_serialize(item) for item in v] for k, v in self.__dict__.items()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def _no_action(event_id, subscription_id: str, classification_bucket: str, reason: str, candidate_scores: list[CandidateScore] | None = None) -> Decision:
    return Decision(
        event_id=event_id,
        subscription_id=subscription_id,
        classification_bucket=classification_bucket,
        selected_candidate_type=NO_ACTION,
        selected_candidate_datetime=None,
        predicted_recovery_value=None,
        intervention_cost=None,
        expected_net_value=None,
        runner_up_value=None,
        decision_margin=None,
        decision_source=SOURCE_NO_ACTION,
        model_version=None,
        policy_version=POLICY_VERSION,
        decision_reason=reason,
        candidate_scores=candidate_scores or [],
    )


def _build_feature_row(candidate: Candidate, amount: float, failure_context: dict) -> dict:
    return {
        "day_of_month": failure_context["day_of_month"],
        "days_to_nearest_payday_window": failure_context["days_to_nearest_payday_window"],
        "amount": amount,
        "prior_if_failure_count": failure_context["prior_if_failure_count"],
        "prior_if_self_resolved_rate": failure_context["prior_if_self_resolved_rate"],
        "tenure_days": failure_context["tenure_days"],
        "hours_from_failure": candidate.hours_from_failure,
        "candidate_day_of_month": candidate.candidate_day_of_month,
        "candidate_days_to_payday": candidate.candidate_days_to_payday,
        "plan_tier": failure_context["plan_tier"],
        "primary_instrument": failure_context["primary_instrument"],
        "city_tier": failure_context["city_tier"],
        "bank_network_conditions": failure_context["bank_network_conditions"],
        "network_latency_bucket": failure_context["network_latency_bucket"],
        "candidate_type": candidate.candidate_type,
        "candidate_day_of_week": candidate.candidate_day_of_week,
        "issuing_bank_downtime_flag": int(bool(failure_context["issuing_bank_downtime_flag"])),
        "is_month_end_settlement_rush": int(bool(failure_context["is_month_end_settlement_rush"])),
        "candidate_is_payday_aligned": int(candidate.candidate_is_payday_aligned),
        "candidate_is_month_end_aligned": int(candidate.candidate_is_month_end_aligned),
    }


def _predict_recovery_values(candidates: list[Candidate], amount: float, failure_context: dict, model: dict) -> dict[str, float]:
    """Raises ModelUnavailableError / MalformedModelOutputError; never
    returns a partially-invalid result -- if ANY candidate's prediction is
    malformed, the whole batch is treated as untrustworthy (fail closed)."""
    missing = [k for k in EVENT_FEATURE_KEYS if k not in failure_context]
    if missing:
        raise MalformedModelOutputError(f"insufficient_features: missing {missing}")

    rows = [_build_feature_row(c, amount, failure_context) for c in candidates]
    X = pd.DataFrame(rows)[FEATURE_COLUMNS]

    try:
        X_imp = model["imputer"].transform(X)
        X_cb = prepare_for_catboost(X_imp)
        preds = model["catboost_model"].predict(X_cb)
    except Exception as exc:  # noqa: BLE001 -- any prediction-time failure funnels into the fallback chain, not a crash
        raise MalformedModelOutputError(f"prediction_exception: {exc}") from exc

    values = {}
    for candidate, pred in zip(candidates, preds):
        pred = float(pred)
        if not math.isfinite(pred) or pred < -1e-6 or pred > amount * MAX_SANE_VALUE_MULTIPLE_OF_AMOUNT:
            raise MalformedModelOutputError(f"malformed_prediction for {candidate.candidate_type}: {pred} (amount={amount})")
        values[candidate.candidate_type] = max(0.0, pred)  # clip tiny negative float noise, already validated above

    return values


def _load_model_safely(model: dict | None) -> dict:
    if model is not None:
        return model
    try:
        return load_latent_target_model("value")
    except Exception as exc:  # noqa: BLE001 -- file missing, corrupt artifact, anything -- all mean "model unavailable"
        raise ModelUnavailableError(str(exc)) from exc


def decide_engine(
    event_id,
    subscription_id: str,
    failure_timestamp: datetime,
    amount: float,
    classification_bucket: str,
    failure_context: dict,
    attempts_so_far: int = 0,
    already_decided: bool = False,
    costs: InterventionCosts = DEFAULT_COSTS,
    abstention_threshold: float = DEFAULT_ABSTENTION_THRESHOLD_RS,
    model: dict | None = None,
) -> Decision:
    """
    Pure decision function (deterministic given identical inputs) -- see
    module docstring for the full flow. `model`, if supplied, is a
    pre-loaded {"imputer":..., "catboost_model":...} dict (avoids reloading
    from disk per call in batch evaluation, and lets tests inject a broken
    model to exercise failure modes); if None, loads Model B from
    disk, tolerating its absence via the fallback chain.
    """
    if already_decided:
        return _no_action(event_id, subscription_id, classification_bucket, "duplicate_decision_skipped: a policy decision already exists for this event_id")

    if not is_classification_allowed(classification_bucket):
        return _no_action(
            event_id, subscription_id, classification_bucket,
            f"blocked_by_classification: bucket={classification_bucket!r} is not retryable_soft "
            "(hard_decline / customer_cancelled / unmapped never get a retry action)",
        )

    if attempts_so_far >= MAX_RETRY_ATTEMPTS:
        return _no_action(event_id, subscription_id, classification_bucket, f"blocked_max_retry_attempts: {attempts_so_far} prior attempts >= limit of {MAX_RETRY_ATTEMPTS}")

    candidates = generate_candidates(failure_timestamp)
    valid_candidates = []
    invalid_scores = []
    for c in candidates:
        is_valid, invalid_reason = validate_candidate(c, failure_timestamp)
        if is_valid:
            valid_candidates.append(c)
        else:
            invalid_scores.append(CandidateScore(c.candidate_type, c.candidate_datetime, False, invalid_reason, None, cost_for_candidate(c.candidate_type, costs), None))

    if not valid_candidates:
        return _no_action(event_id, subscription_id, classification_bucket, "blocked_no_valid_candidates: every candidate failed validation", candidate_scores=invalid_scores)

    # --- Tier 1: PRIMARY -- Model B ---------------------------------
    model_values: dict[str, float] | None = None
    primary_failure_reason: str | None = None
    try:
        loaded_model = _load_model_safely(model)
        model_values = _predict_recovery_values(valid_candidates, amount, failure_context, loaded_model)
    except ModelUnavailableError as exc:
        primary_failure_reason = f"model_unavailable: {exc}"
    except MalformedModelOutputError as exc:
        primary_failure_reason = f"invalid_model_output: {exc}"

    scored: list[CandidateScore] = list(invalid_scores)

    if model_values is not None:
        for c in valid_candidates:
            recovery_value = model_values[c.candidate_type]
            cost = cost_for_candidate(c.candidate_type, costs)
            scored.append(CandidateScore(c.candidate_type, c.candidate_datetime, True, None, recovery_value, cost, recovery_value - cost))

        ranked = sorted([s for s in scored if s.valid], key=lambda s: s.expected_net_value, reverse=True)
        best = ranked[0]
        margin = (best.expected_net_value - ranked[1].expected_net_value) if len(ranked) > 1 else None

        if margin is not None and margin < abstention_threshold:
            fallback_decision = _try_rule_based_fallback(
                event_id, subscription_id, failure_timestamp, amount, classification_bucket, valid_candidates, costs, scored,
                reason_prefix=f"insufficient_decision_margin ({margin:.2f} < {abstention_threshold:.2f}) -- fell back to rule-based",
                model_version=MODEL_VERSION,
            )
            return fallback_decision

        if best.expected_net_value <= 0:
            return _no_action(event_id, subscription_id, classification_bucket, f"no_positive_net_value: best candidate {best.candidate_type} net={best.expected_net_value:.2f}", candidate_scores=scored)

        runner_up_value = ranked[1].predicted_recovery_value if len(ranked) > 1 else None
        return Decision(
            event_id=event_id,
            subscription_id=subscription_id,
            classification_bucket=classification_bucket,
            selected_candidate_type=best.candidate_type,
            selected_candidate_datetime=best.candidate_datetime,
            predicted_recovery_value=best.predicted_recovery_value,
            intervention_cost=best.intervention_cost,
            expected_net_value=best.expected_net_value,
            runner_up_value=runner_up_value,
            decision_margin=margin,
            decision_source=SOURCE_MODEL,
            model_version=MODEL_VERSION,
            policy_version=POLICY_VERSION,
            decision_reason=(
                f"selected {best.candidate_type} at {best.candidate_datetime.isoformat()} via {SOURCE_MODEL} "
                f"(predicted_recovery_value={best.predicted_recovery_value:.2f}, cost={best.intervention_cost:.2f}, "
                f"net={best.expected_net_value:.2f}, margin={margin if margin is not None else 'n/a'})"
            ),
            candidate_scores=scored,
        )

    # --- Model unavailable/errored entirely -- go straight to fallback ---
    return _try_rule_based_fallback(
        event_id, subscription_id, failure_timestamp, amount, classification_bucket, valid_candidates, costs, scored,
        reason_prefix=primary_failure_reason or "primary_model_failed",
        model_version=None,
    )


def _try_rule_based_fallback(
    event_id, subscription_id: str, failure_timestamp: datetime, amount: float, classification_bucket: str,
    valid_candidates: list[Candidate], costs: InterventionCosts, scored_so_far: list[CandidateScore],
    reason_prefix: str, model_version: str | None,
) -> Decision:
    """Tier 2. If Model B's values are already known for these candidates
    (low-margin path), we reuse them to fill in the fallback decision's
    numeric fields and still enforce the positive-net-value guardrail. If
    Model B genuinely failed (unavailable/errored), no value is known --
    the rule-based candidate is trusted on its own merits, subject only to
    the guardrails already passed above."""
    valid_types = {c.candidate_type: c for c in valid_candidates}
    fallback_type = rule_based_baseline(event_id, subscription_id, failure_timestamp, amount, classification_bucket, 0.0)["selected_candidate_type"]

    if fallback_type not in valid_types:
        return _no_action(event_id, subscription_id, classification_bucket, f"{reason_prefix}; fallback candidate {fallback_type!r} also invalid", candidate_scores=scored_so_far)

    known = next((s for s in scored_so_far if s.candidate_type == fallback_type and s.valid and s.predicted_recovery_value is not None), None)
    fallback_candidate = valid_types[fallback_type]
    cost = cost_for_candidate(fallback_type, costs)

    if known is not None:
        if known.expected_net_value <= 0:
            return _no_action(event_id, subscription_id, classification_bucket, f"{reason_prefix}; fallback candidate {fallback_type!r} also has non-positive net value ({known.expected_net_value:.2f})", candidate_scores=scored_so_far)
        predicted_value, net_value = known.predicted_recovery_value, known.expected_net_value
    else:
        predicted_value, net_value = None, None  # model genuinely unavailable -- no value estimate exists

    return Decision(
        event_id=event_id,
        subscription_id=subscription_id,
        classification_bucket=classification_bucket,
        selected_candidate_type=fallback_type,
        selected_candidate_datetime=fallback_candidate.candidate_datetime,
        predicted_recovery_value=predicted_value,
        intervention_cost=cost,
        expected_net_value=net_value,
        runner_up_value=None,
        decision_margin=None,
        decision_source=SOURCE_FALLBACK,
        model_version=model_version,
        policy_version=POLICY_VERSION,
        decision_reason=f"{reason_prefix}; selected {fallback_type} via {SOURCE_FALLBACK}",
        candidate_scores=scored_so_far,
    )


def _serialize_candidate_scores(candidate_scores: list[CandidateScore]) -> str:
    def default(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, CandidateScore):
            return obj.__dict__
        raise TypeError(f"not JSON serializable: {type(obj)}")

    return json.dumps(candidate_scores, default=default)


def decide_for_failure_event_engine(
    db: Session,
    event_id: int,
    subscription_id: str,
    failure_timestamp: datetime,
    amount: float,
    classification_bucket: str,
    failure_context: dict,
    costs: InterventionCosts = DEFAULT_COSTS,
    abstention_threshold: float = DEFAULT_ABSTENTION_THRESHOLD_RS,
    model: dict | None = None,
) -> tuple[PolicyDecision | Decision, bool]:
    """DB-aware wrapper -- same idempotency (by event_id, shared with every
    prior policy_decisions row regardless of which policy_version made it)
    and max-retry-attempts accounting as policy/recovery_policy.py's
    equivalents. Persists a policy_decisions row plus an audit_log row for
    every call, including NO_ACTION, blocked, and duplicate ones."""
    existing = db.query(PolicyDecision).filter(PolicyDecision.event_id == event_id).first()
    if existing is not None:
        db.add(
            AuditLog(
                failure_event_id=event_id,
                action="policy_decision_skipped_duplicate",
                reason=f"event_id={event_id} already has policy_decisions.id={existing.id} (selected={existing.selected_candidate_type}); not re-decided.",
                actor="policy",
            )
        )
        db.commit()
        log.info("Skipped duplicate decision-engine decision for event_id=%s (already policy_decisions.id=%s)", event_id, existing.id)
        return existing, False

    prior_attempts = (
        db.query(PolicyDecision)
        .filter(PolicyDecision.subscription_id == subscription_id, PolicyDecision.selected_candidate_type != NO_ACTION)
        .count()
    )

    result = decide_engine(
        event_id=event_id, subscription_id=subscription_id, failure_timestamp=failure_timestamp, amount=amount,
        classification_bucket=classification_bucket, failure_context=failure_context, attempts_so_far=prior_attempts,
        already_decided=False, costs=costs, abstention_threshold=abstention_threshold, model=model,
    )

    decision_row = PolicyDecision(
        event_id=event_id,
        subscription_id=subscription_id,
        selected_candidate_type=result.selected_candidate_type,
        selected_candidate_datetime=result.selected_candidate_datetime,
        expected_recovery_value=result.predicted_recovery_value,
        expected_incremental_value=result.expected_net_value,
        policy_version=result.policy_version,
        decision_reason=result.decision_reason,
        classification_bucket=result.classification_bucket,
        intervention_cost=result.intervention_cost,
        runner_up_value=result.runner_up_value,
        decision_margin=result.decision_margin,
        decision_source=result.decision_source,
        model_version=result.model_version,
    )
    db.add(decision_row)
    db.flush()

    action = "policy_no_action" if result.selected_candidate_type == NO_ACTION else "policy_decision_made"
    db.add(
        AuditLog(
            failure_event_id=event_id,
            action=action,
            reason=f"{result.decision_reason} | decision_source={result.decision_source} | candidate_scores={_serialize_candidate_scores(result.candidate_scores)}",
            actor="policy",
        )
    )
    db.commit()

    log.info(
        "Decision-engine decision for event_id=%s: %s via %s (policy_decisions.id=%s)",
        event_id, result.selected_candidate_type, result.decision_source, decision_row.id,
    )
    return decision_row, True
