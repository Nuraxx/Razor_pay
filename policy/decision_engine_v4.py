"""
Day-10 improved fallback/abstention decision engine (policy-v4).

DOES NOT retrain or replace Day-8 Model B, and does NOT delete or modify
Day-9 (policy-v3, policy/decision_engine.py -- untouched, still fully
supported, still the value of `PolicyDecision.decision_source == "day8_model_b"`
etc for any Day-9 row). This module only changes the ABSTENTION/FALLBACK
LOGIC that sits on top of Model B's predictions.

WHY (brief section "Day-9 finding" + this module's own diagnosis -- see
evaluation/diagnose_day9_fallback.py and README "Day 10"): Day-9's fallback
question was "is Model B confident in its own #1 vs #2 pick?" (the gap
between Model B's own top-2 net values). That question is blind to whether
Rule-Based's candidate is actually any good -- it triggers a fallback any
time Model B's top candidates are close together, even when ALL of them
(including Rule-Based's pick) are close together because they are all
similarly good, not because Model B is wrong. On the Day-9 test set this
handed 41/60 events to Rule-Based, foregoing ~Rs1,599 of latent value Model B
itself would have picked correctly.

Day 10 asks a different, more relevant question once the margin gate
triggers: "does Rule-Based's candidate look meaningfully BETTER than Model
B's own top pick, ALSO according to Model B's own value predictions?" This
reuses a value Day-9 already computed as a side effect (see Day-9's
`_try_rule_based_fallback`'s `known` lookup) and promotes it into the actual
gating mechanism (brief section 3).

Two independently validation-searched knobs, evaluated together in
evaluation/evaluate_decision_engine_v4.py:

  margin_threshold   -- same Day-9 concept (gap between Model B's own top-2
                         *net* values). Still only gates WHETHER to even
                         consider deviating from Model B's own top pick;
                         it is NOT by itself the fallback decision.
  fallback_mode       -- WHAT to do once that gate triggers (below).
  fallback_advantage_threshold -- only consulted by
                         KEEP_MODEL_UNLESS_RULE_HAS_CLEAR_ADVANTAGE (below);
                         brief section 3's `fallback_advantage_threshold`.

Four fallback modes (brief section 2B):

  ALWAYS_FALLBACK_WHEN_BELOW_MARGIN
      Day-9's original behaviour, reimplemented here for a fair, apples-to-
      apples comparison inside the same evaluation harness: below margin,
      always take Rule-Based's candidate (subject to it being valid and
      having positive net value).
  NO_ACTION_WHEN_BELOW_MARGIN
      Deliberately conservative: below margin, take no action at all rather
      than guess between Model B and Rule-Based.
  KEEP_MODEL_WHEN_BETTER_THAN_RULE
      Below margin, compare Model B's OWN predicted net value for Model B's
      best pick against Model B's OWN predicted net value for Rule-Based's
      pick; keep whichever is higher (Rule-Based only wins by ANY amount).
  KEEP_MODEL_UNLESS_RULE_HAS_CLEAR_ADVANTAGE
      Same comparison as above, but Rule-Based must beat Model B's best pick
      by more than `fallback_advantage_threshold` to win -- this is brief
      section 3's rule, and the one this module's diagnosis expects to
      perform best (a small numeric edge for Rule-Based within noise
      shouldn't flip the decision; a clear edge should).

All previous guardrails (policy/guardrails.py, unchanged) still apply and
are checked FIRST: `retryable_soft` only, no duplicate decision, max retry
attempts, candidate must be valid (after failure, within the 14-day
horizon). The three-tier fail-closed structure is unchanged: Model B
(primary) -> Rule-Based (secondary, now evidence-gated) -> NO_ACTION (final
safety net). Every decision still records exactly which tier decided it, via
the same `decision_source` vocabulary as Day 9.

NOTE ON COSTS: with the current synthetic cost model (policy/costs.py), all
5 candidate types share the same `retry_cost` + `operational_cost`, so
comparing by *net* value (predicted_recovery_value - cost) or by *raw*
predicted value ranks identically today. Net value is used throughout below
so this stays correct if a future day differentiates cost by candidate type.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from app.logging_config import log
from app.models import AuditLog, PolicyDecision
from policy.baselines import rule_based_baseline
from policy.costs import DEFAULT_COSTS, InterventionCosts, cost_for_candidate
from policy.decision_engine import (
    MODEL_VERSION,
    NO_ACTION,
    SOURCE_FALLBACK,
    SOURCE_MODEL,
    SOURCE_NO_ACTION,
    CandidateScore,
    Decision,
    MalformedModelOutputError,
    ModelUnavailableError,
    _load_model_safely,
    _predict_recovery_values,
)
from policy.guardrails import MAX_RETRY_ATTEMPTS, is_classification_allowed, validate_candidate
from policy.retry_candidates import generate_candidates

POLICY_VERSION_V4 = "policy-v4"

FALLBACK_MODE_ALWAYS = "always_fallback_when_below_margin"
FALLBACK_MODE_NO_ACTION = "no_action_when_below_margin"
FALLBACK_MODE_KEEP_IF_BETTER = "keep_model_when_better_than_rule"
FALLBACK_MODE_KEEP_UNLESS_CLEAR = "keep_model_unless_rule_has_clear_advantage"

FALLBACK_MODES = [
    FALLBACK_MODE_ALWAYS,
    FALLBACK_MODE_NO_ACTION,
    FALLBACK_MODE_KEEP_IF_BETTER,
    FALLBACK_MODE_KEEP_UNLESS_CLEAR,
]

FallbackMode = Literal[
    "always_fallback_when_below_margin",
    "no_action_when_below_margin",
    "keep_model_when_better_than_rule",
    "keep_model_unless_rule_has_clear_advantage",
]

# Chosen via validation-only search over 108 (margin_threshold, fallback_mode,
# fallback_advantage_threshold) combinations -- see
# evaluation/evaluate_decision_engine_v4.py::select_day10_configuration_on_validation
# and README "Day 10: configuration selection". Frozen here; never re-tuned
# against test results.
#
# ECONOMIC-CORRECTION FINDING (final pre-submission audit -- see README "Day
# 10: economic correction" and evaluation/reports/decision_engine_v4_evaluation.json):
# the ORIGINAL Day-10 search picked ALWAYS_FALLBACK_WHEN_BELOW_MARGIN at a
# Rs5 margin because it scored highest on total LATENT value
# (Rs18609.15 vs Rs18258.62 for Model-B-alone) -- a smooth, low-variance
# proxy (`expected_recovery_value_latent = recovery_probability_latent *
# amount`), never the actual/realized outcome. Re-running that SAME 108-
# config search on the SAME validation split (n=59), but scoring by total
# REALIZED Rs recovered instead, shows the two metrics DISAGREE: the chosen
# config's realized total is Rs16417.73 -- Rs2105.48 WORSE than Model-B-alone
# (Rs18523.21) on validation itself, not merely on held-out test. Every
# config that ever blind-swaps away from Model B's own top pick (i.e. every
# ALWAYS_FALLBACK_WHEN_BELOW_MARGIN margin > 0) scores worse by realized Rs
# on validation; Model-B-alone (or any config that is mathematically
# equivalent to it -- see structural finding below) is tied for the single
# best realized-value config out of all 108 searched. This reproduced on the
# held-out TEST split too (Model-B-alone Rs21278.18 vs the old default's
# Rs19997.23, a Rs1280.95 loss purely from blind fallback swapping) --
# confirming this is a real property of the mechanism on this population,
# not test-set noise. The default below has been corrected accordingly.
#
# STRUCTURAL FINDING (verified, not a bug -- see README "Day 10: why the
# evidence-based modes never fire"): KEEP_MODEL_WHEN_BETTER_THAN_RULE and
# KEEP_MODEL_UNLESS_RULE_HAS_CLEAR_ADVANTAGE scored EXACTLY Rs18258.62 (=
# zero fallbacks) across all 90 margin x advantage combinations tested for
# them. This is mathematically guaranteed, not coincidental: Rule-Based's
# candidate is always one of the same <=5 valid candidate types Model B
# already scores, so Model B's own top pick (the argmax over that whole set)
# can never have a LOWER net value than Model B's own estimate for
# Rule-Based's candidate -- there is nothing for a "clear advantage" to ever
# detect. Both modes therefore ALWAYS collapse to "day8_model_b_alone"'s
# selections in this architecture, regardless of margin_threshold or
# fallback_advantage_threshold: they can never do WORSE than Model B alone,
# by construction. That safety property is exactly why
# KEEP_MODEL_UNLESS_RULE_HAS_CLEAR_ADVANTAGE is the corrected default below
# -- unlike ALWAYS_FALLBACK_WHEN_BELOW_MARGIN, it is provably incapable of
# selecting a worse candidate than Model B's own best pick, because it
# ignores the model's own valuation of the substitute entirely.
#
# margin_threshold=0.0 (rather than the original Rs5) is not a loss of
# information: decision_margin is ALWAYS computed and recorded in the audit
# trail from Model B's own top-2 net values regardless of margin_threshold
# (see decide_engine_v4 below) -- the threshold only controls whether a
# deviation is ever TAKEN, and every deviation this architecture can take is
# either unsafe (ALWAYS mode) or a structural no-op (the two KEEP_* modes).
# With the safe mode, margin_threshold=0.0 and margin_threshold=100.0 select
# identically; 0.0 is kept as the plainest, most self-explanatory value, and
# is exactly what evaluation/evaluate_decision_engine_v4.py's validation-only
# search (re-run after the economic correction, see finding above) selects.
DEFAULT_MARGIN_THRESHOLD_RS = 0.0
DEFAULT_FALLBACK_MODE: FallbackMode = FALLBACK_MODE_KEEP_UNLESS_CLEAR
DEFAULT_FALLBACK_ADVANTAGE_THRESHOLD_RS = 0.0


def fallback_advantage(rule_candidate_value: float, model_best_value: float) -> float:
    """Brief section 3: `model_advantage = model_best_value - rule_candidate_value`.
    This returns the negation of that (positive means Rule-Based is ahead of
    Model B's own best pick, by this many Rs of net value) since that is the
    more natural sign for the "does rule have an advantage" comparisons
    below. Deliberately a free-standing pure function, independent of
    decide_engine_v4's real candidate-scoring flow, so the exact arithmetic
    is directly unit-testable -- see the structural finding documented above
    DEFAULT_MARGIN_THRESHOLD_RS: in an actual decide_engine_v4() call this
    can never legitimately be positive, because Rule-Based's candidate is
    always drawn from the same set Model B's own best pick was the argmax
    of. It is exercised here as a pure formula so that fact doesn't prevent
    the comparison logic itself from being tested."""
    return rule_candidate_value - model_best_value


def _rule_has_any_advantage(rule_candidate_value: float, model_best_value: float) -> bool:
    """KEEP_MODEL_WHEN_BETTER_THAN_RULE's gate: any positive advantage, however small."""
    return fallback_advantage(rule_candidate_value, model_best_value) > 0


def _rule_has_clear_advantage(rule_candidate_value: float, model_best_value: float, fallback_advantage_threshold: float) -> bool:
    """KEEP_MODEL_UNLESS_RULE_HAS_CLEAR_ADVANTAGE's gate (brief section 3):
    `rule_candidate_value > model_best_value + fallback_advantage_threshold`."""
    return fallback_advantage(rule_candidate_value, model_best_value) > fallback_advantage_threshold


def _no_action_v4(
    event_id, subscription_id: str, classification_bucket: str, reason: str,
    margin_threshold: float, fallback_mode: str, fallback_advantage_threshold: float,
    candidate_scores: list[CandidateScore] | None = None,
) -> Decision:
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
        policy_version=POLICY_VERSION_V4,
        decision_reason=reason,
        candidate_scores=candidate_scores or [],
        margin_threshold_used=margin_threshold,
        fallback_advantage_threshold=fallback_advantage_threshold,
        fallback_strategy=fallback_mode,
    )


def decide_engine_v4(
    event_id,
    subscription_id: str,
    failure_timestamp: datetime,
    amount: float,
    classification_bucket: str,
    failure_context: dict,
    attempts_so_far: int = 0,
    already_decided: bool = False,
    costs: InterventionCosts = DEFAULT_COSTS,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD_RS,
    fallback_mode: FallbackMode = DEFAULT_FALLBACK_MODE,
    fallback_advantage_threshold: float = DEFAULT_FALLBACK_ADVANTAGE_THRESHOLD_RS,
    model: dict | None = None,
) -> Decision:
    """Pure decision function (deterministic given identical inputs) -- see
    module docstring for the full flow. Mirrors policy/decision_engine.py's
    decide_engine() for guardrails and Model-B interaction; differs only in
    what happens once Model B's own top-2 margin is below `margin_threshold`."""

    def na(reason: str, scores: list[CandidateScore] | None = None) -> Decision:
        return _no_action_v4(event_id, subscription_id, classification_bucket, reason, margin_threshold, fallback_mode, fallback_advantage_threshold, scores)

    if already_decided:
        return na("duplicate_decision_skipped: a policy decision already exists for this event_id")

    if not is_classification_allowed(classification_bucket):
        return na(
            f"blocked_by_classification: bucket={classification_bucket!r} is not retryable_soft "
            "(hard_decline / customer_cancelled / unmapped never get a retry action)"
        )

    if attempts_so_far >= MAX_RETRY_ATTEMPTS:
        return na(f"blocked_max_retry_attempts: {attempts_so_far} prior attempts >= limit of {MAX_RETRY_ATTEMPTS}")

    candidates = generate_candidates(failure_timestamp)
    valid_candidates = []
    invalid_scores: list[CandidateScore] = []
    for c in candidates:
        is_valid, invalid_reason = validate_candidate(c, failure_timestamp)
        if is_valid:
            valid_candidates.append(c)
        else:
            invalid_scores.append(CandidateScore(c.candidate_type, c.candidate_datetime, False, invalid_reason, None, cost_for_candidate(c.candidate_type, costs), None))

    if not valid_candidates:
        return na("blocked_no_valid_candidates: every candidate failed validation", invalid_scores)

    # --- Tier 1: PRIMARY -- Day-8 Model B (same interaction code as v3) ---
    try:
        loaded_model = _load_model_safely(model)
        model_values = _predict_recovery_values(valid_candidates, amount, failure_context, loaded_model)
    except ModelUnavailableError as exc:
        return _rule_based_only(
            event_id, subscription_id, failure_timestamp, amount, classification_bucket, valid_candidates, costs,
            invalid_scores, margin_threshold, fallback_mode, fallback_advantage_threshold, reason_prefix=f"model_unavailable: {exc}",
        )
    except MalformedModelOutputError as exc:
        return _rule_based_only(
            event_id, subscription_id, failure_timestamp, amount, classification_bucket, valid_candidates, costs,
            invalid_scores, margin_threshold, fallback_mode, fallback_advantage_threshold, reason_prefix=f"invalid_model_output: {exc}",
        )

    scored: list[CandidateScore] = list(invalid_scores)
    for c in valid_candidates:
        recovery_value = model_values[c.candidate_type]
        cost = cost_for_candidate(c.candidate_type, costs)
        scored.append(CandidateScore(c.candidate_type, c.candidate_datetime, True, None, recovery_value, cost, recovery_value - cost))

    ranked = sorted([s for s in scored if s.valid], key=lambda s: s.expected_net_value, reverse=True)
    best = ranked[0]
    margin = (best.expected_net_value - ranked[1].expected_net_value) if len(ranked) > 1 else None

    def model_decision() -> Decision:
        if best.expected_net_value <= 0:
            return na(f"no_positive_net_value: best candidate {best.candidate_type} net={best.expected_net_value:.2f}", scored)
        runner_up_value = ranked[1].predicted_recovery_value if len(ranked) > 1 else None
        return Decision(
            event_id=event_id, subscription_id=subscription_id, classification_bucket=classification_bucket,
            selected_candidate_type=best.candidate_type, selected_candidate_datetime=best.candidate_datetime,
            predicted_recovery_value=best.predicted_recovery_value, intervention_cost=best.intervention_cost,
            expected_net_value=best.expected_net_value, runner_up_value=runner_up_value, decision_margin=margin,
            decision_source=SOURCE_MODEL, model_version=MODEL_VERSION, policy_version=POLICY_VERSION_V4,
            decision_reason=(
                f"selected {best.candidate_type} at {best.candidate_datetime.isoformat()} via {SOURCE_MODEL} "
                f"(predicted_recovery_value={best.predicted_recovery_value:.2f}, cost={best.intervention_cost:.2f}, "
                f"net={best.expected_net_value:.2f}, margin={margin if margin is not None else 'n/a'} "
                f">= margin_threshold={margin_threshold:.2f})"
            ),
            candidate_scores=scored, margin_threshold_used=margin_threshold,
            fallback_advantage_threshold=fallback_advantage_threshold, fallback_strategy=fallback_mode,
        )

    if margin is None or margin >= margin_threshold:
        return model_decision()

    # --- Margin is ambiguous: Model B's own top-2 picks are too close to ---
    # --- trust blindly. Consult fallback_mode for what to do next. --------
    rule_type = rule_based_baseline(event_id, subscription_id, failure_timestamp, amount, classification_bucket, 0.0)["selected_candidate_type"]
    rule_known = next((s for s in scored if s.candidate_type == rule_type and s.valid), None)

    def take_rule() -> Decision:
        if rule_known is None:
            return na(
                f"ambiguous_margin ({margin:.2f} < {margin_threshold:.2f}) via {fallback_mode}; "
                f"rule-based candidate {rule_type!r} is invalid or unpriced",
                scored,
            )
        if rule_known.expected_net_value <= 0:
            return na(
                f"ambiguous_margin ({margin:.2f} < {margin_threshold:.2f}) via {fallback_mode}; "
                f"rule-based candidate {rule_type!r} has non-positive net value ({rule_known.expected_net_value:.2f})",
                scored,
            )
        return Decision(
            event_id=event_id, subscription_id=subscription_id, classification_bucket=classification_bucket,
            selected_candidate_type=rule_known.candidate_type, selected_candidate_datetime=rule_known.candidate_datetime,
            predicted_recovery_value=rule_known.predicted_recovery_value, intervention_cost=rule_known.intervention_cost,
            expected_net_value=rule_known.expected_net_value, runner_up_value=best.predicted_recovery_value, decision_margin=margin,
            decision_source=SOURCE_FALLBACK, model_version=MODEL_VERSION, policy_version=POLICY_VERSION_V4,
            decision_reason=(
                f"ambiguous_margin ({margin:.2f} < {margin_threshold:.2f}) via {fallback_mode}; selected rule-based "
                f"candidate {rule_known.candidate_type} (net={rule_known.expected_net_value:.2f}) over model's best "
                f"{best.candidate_type} (net={best.expected_net_value:.2f}, advantage={rule_known.expected_net_value - best.expected_net_value:.2f})"
            ),
            candidate_scores=scored, margin_threshold_used=margin_threshold,
            fallback_advantage_threshold=fallback_advantage_threshold, fallback_strategy=fallback_mode,
        )

    if fallback_mode == FALLBACK_MODE_ALWAYS:
        return take_rule()

    if fallback_mode == FALLBACK_MODE_NO_ACTION:
        return na(f"ambiguous_margin ({margin:.2f} < {margin_threshold:.2f}) via {fallback_mode}: abstaining entirely", scored)

    if fallback_mode == FALLBACK_MODE_KEEP_IF_BETTER:
        if rule_known is not None and _rule_has_any_advantage(rule_known.expected_net_value, best.expected_net_value):
            return take_rule()
        return model_decision()

    if fallback_mode == FALLBACK_MODE_KEEP_UNLESS_CLEAR:
        if rule_known is not None and _rule_has_clear_advantage(rule_known.expected_net_value, best.expected_net_value, fallback_advantage_threshold):
            return take_rule()
        return model_decision()

    raise ValueError(f"unknown fallback_mode: {fallback_mode!r}")


def _rule_based_only(
    event_id, subscription_id: str, failure_timestamp: datetime, amount: float, classification_bucket: str,
    valid_candidates, costs: InterventionCosts, invalid_scores: list[CandidateScore],
    margin_threshold: float, fallback_mode: str, fallback_advantage_threshold: float, reason_prefix: str,
) -> Decision:
    """Model B genuinely unavailable/errored (not merely ambiguous) -- Tier
    2, trusting Rule-Based on its own merits since no Model-B value exists
    for any candidate. Identical semantics to Day-9's equivalent path."""
    valid_types = {c.candidate_type: c for c in valid_candidates}
    fallback_type = rule_based_baseline(event_id, subscription_id, failure_timestamp, amount, classification_bucket, 0.0)["selected_candidate_type"]

    if fallback_type not in valid_types:
        return _no_action_v4(
            event_id, subscription_id, classification_bucket, f"{reason_prefix}; fallback candidate {fallback_type!r} also invalid",
            margin_threshold, fallback_mode, fallback_advantage_threshold, invalid_scores,
        )

    fallback_candidate = valid_types[fallback_type]
    cost = cost_for_candidate(fallback_type, costs)
    return Decision(
        event_id=event_id, subscription_id=subscription_id, classification_bucket=classification_bucket,
        selected_candidate_type=fallback_type, selected_candidate_datetime=fallback_candidate.candidate_datetime,
        predicted_recovery_value=None, intervention_cost=cost, expected_net_value=None, runner_up_value=None,
        decision_margin=None, decision_source=SOURCE_FALLBACK, model_version=None, policy_version=POLICY_VERSION_V4,
        decision_reason=f"{reason_prefix}; selected {fallback_type} via {SOURCE_FALLBACK} (no model value available)",
        candidate_scores=invalid_scores, margin_threshold_used=margin_threshold,
        fallback_advantage_threshold=fallback_advantage_threshold, fallback_strategy=fallback_mode,
    )


def decide_for_failure_event_engine_v4(
    db,
    event_id: int,
    subscription_id: str,
    failure_timestamp: datetime,
    amount: float,
    classification_bucket: str,
    failure_context: dict,
    costs: InterventionCosts = DEFAULT_COSTS,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD_RS,
    fallback_mode: FallbackMode = DEFAULT_FALLBACK_MODE,
    fallback_advantage_threshold: float = DEFAULT_FALLBACK_ADVANTAGE_THRESHOLD_RS,
    model: dict | None = None,
) -> tuple[PolicyDecision, bool]:
    """DB-aware wrapper -- same idempotency (by event_id, shared across
    every policy_version) and max-retry-attempts accounting as Day 9's
    equivalent. Persists a policy_decisions row plus an audit_log row for
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
        log.info("Skipped duplicate decision-engine-v4 decision for event_id=%s (already policy_decisions.id=%s)", event_id, existing.id)
        return existing, False

    prior_attempts = (
        db.query(PolicyDecision)
        .filter(PolicyDecision.subscription_id == subscription_id, PolicyDecision.selected_candidate_type != NO_ACTION)
        .count()
    )

    result = decide_engine_v4(
        event_id=event_id, subscription_id=subscription_id, failure_timestamp=failure_timestamp, amount=amount,
        classification_bucket=classification_bucket, failure_context=failure_context, attempts_so_far=prior_attempts,
        already_decided=False, costs=costs, margin_threshold=margin_threshold, fallback_mode=fallback_mode,
        fallback_advantage_threshold=fallback_advantage_threshold, model=model,
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
        margin_threshold_used=result.margin_threshold_used,
        fallback_advantage_threshold=result.fallback_advantage_threshold,
        fallback_strategy=result.fallback_strategy,
    )
    db.add(decision_row)
    db.flush()

    action = "policy_no_action" if result.selected_candidate_type == NO_ACTION else "policy_decision_made"
    db.add(
        AuditLog(
            failure_event_id=event_id,
            action=action,
            reason=(
                f"{result.decision_reason} | decision_source={result.decision_source} | "
                f"policy_version={result.policy_version} | margin_threshold={result.margin_threshold_used} | "
                f"fallback_advantage_threshold={result.fallback_advantage_threshold} | fallback_strategy={result.fallback_strategy}"
            ),
            actor="policy",
        )
    )
    db.commit()

    log.info(
        "Decision-engine-v4 decision for event_id=%s: %s via %s (policy_decisions.id=%s)",
        event_id, result.selected_candidate_type, result.decision_source, decision_row.id,
    )
    return decision_row, True
