"""
Day-1 schema.

raw_events    — fully populated today: every verified, de-duplicated webhook
                delivery is stored here with its complete raw payload plus a
                handful of extracted convenience fields.

failure_events — table created today, left EMPTY today. Day 2's deterministic
                classifier will read from raw_events and write one row here
                per failure, bucketing it (retryable_soft / hard_decline /
                customer_cancelled / unmapped). Columns are nullable now so
                Day 2 only needs to populate them, not migrate the schema.

audit_log     — table created today. Day 1 writes one row per webhook
                received ("stored") so the audit trail concept is real from
                day one; Day 2+ will add richer decision rows (classified,
                retry_scheduled, blocked_by_compliance, etc.) using the same
                table and the same `actor` convention.

policy_decisions — added Day 5. One row per recovery-policy decision (see
                policy/recovery_policy.py) for a failure_events row —
                selected retry candidate (or NO_ACTION), the model+heuristic
                probability behind it, and expected recovery value. Every
                decision (including NO_ACTION, blocked, and duplicate ones)
                also gets an audit_log row via `failure_event_id` below.
                Extended Day 9 (policy/decision_engine.py) with cost/margin/
                fallback-source columns for the production-shaped decision
                engine -- see that module's docstring. Extended again Day 10
                (policy/decision_engine_v4.py) with the margin/advantage
                threshold config and fallback-strategy label actually in
                effect for a given decision -- see that module's docstring.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RawEvent(Base):
    __tablename__ = "raw_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Idempotency key — Razorpay's `x-razorpay-event-id` header, unique per
    # event per Razorpay's own docs. This column has a UNIQUE constraint so a
    # duplicate delivery cannot create a second row even under a race.
    razorpay_event_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)

    event_type: Mapped[str] = mapped_column(String(64), index=True)  # e.g. "payment.failed"

    # Convenience fields extracted from payload.payment.entity / payload.subscription.entity.
    # All nullable: not every event type carries all of these.
    payment_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    subscription_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    amount: Mapped[int | None] = mapped_column(Integer, nullable=True)  # paise, as Razorpay sends it
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)

    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_reason: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)  # e.g. "insufficient_fund"
    error_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error_step: Mapped[str | None] = mapped_column(String(64), nullable=True)

    razorpay_created_at: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )  # unix timestamp from the payload's top-level "created_at", if present

    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    signature_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    raw_payload: Mapped[str] = mapped_column(Text)  # complete raw JSON body, verbatim


class FailureEvent(Base):
    """Schema only as of Day 1 — Day 2 populates this."""
    __tablename__ = "failure_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    raw_event_id: Mapped[int] = mapped_column(Integer, index=True)  # FK to raw_events.id (added Day 2)

    classification_bucket: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )  # "retryable_soft" | "hard_decline" | "customer_cancelled" | "unmapped"
    classification_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    classified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rule_version: Mapped[str | None] = mapped_column(String(16), nullable=True)


class AuditLog(Base):
    """Every decision the system makes — including deciding to do nothing — goes here."""
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    raw_event_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    # Added Day 5: classification/policy rows trace to a failure_events row.
    # Nullable and additive -- existing Day 1/2 rows only ever set raw_event_id.
    failure_event_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)

    action: Mapped[str] = mapped_column(String(64))  # e.g. "webhook_received_and_stored"
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    actor: Mapped[str] = mapped_column(String(32), default="system")  # "system" | "rule" | "model" | "llm" | "policy"

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class PolicyDecision(Base):
    """
    Day 5: one row per recovery-policy decision. See policy/recovery_policy.py
    for the deterministic decision logic that produces these.

    Idempotent by event_id -- a second decide() call for the same event_id
    returns the existing row instead of creating a new one (see
    policy/recovery_policy.py::decide_for_failure_event).
    """
    __tablename__ = "policy_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(Integer, index=True, unique=True)  # FK to failure_events.id (logical)
    subscription_id: Mapped[str] = mapped_column(String(64), index=True)

    # "NO_ACTION" | "immediate" | "plus_1_day_morning" | "payday_window" | "plus_3_days" | "month_end_window"
    selected_candidate_type: Mapped[str] = mapped_column(String(32))
    selected_candidate_datetime: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    predicted_recovery_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_recovery_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_incremental_value: Mapped[float | None] = mapped_column(Float, nullable=True)

    baseline_action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    policy_version: Mapped[str] = mapped_column(String(16))
    decision_reason: Mapped[str] = mapped_column(Text)

    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    # -- Added Day 9 (policy/decision_engine.py). All nullable/additive --
    # existing Day 5-8 rows never populate these. expected_recovery_value
    # already holds Day-9's `predicted_recovery_value`, and
    # expected_incremental_value already holds Day-9's `expected_net_value`
    # (same formula: recovery value - intervention cost) -- no duplicate
    # columns for those two.
    classification_bucket: Mapped[str | None] = mapped_column(String(32), nullable=True)
    intervention_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    runner_up_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision_margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    # "day8_model_b" | "rule_based_fallback" | "no_action"
    decision_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # -- Added Day 10 (policy/decision_engine_v4.py). All nullable/additive --
    # existing Day 5-9 rows never populate these. `decision_margin` above
    # already holds the *observed* margin for a decision; these three record
    # the CONFIG that was in effect (which is not always recoverable from the
    # observed margin alone, since the search now varies more than one knob).
    margin_threshold_used: Mapped[float | None] = mapped_column(Float, nullable=True)
    fallback_advantage_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    # "always_fallback_when_below_margin" | "no_action_when_below_margin" |
    # "keep_model_when_better_than_rule" | "keep_model_unless_rule_has_clear_advantage"
    fallback_strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)


class LLMInvocation(Base):
    """
    Day 11: one row per LLM call (any of the 3 jobs in llm/service.py).
    Every invocation ALSO gets a companion audit_log row (actor="llm",
    reason includes task_name/success/error_type) -- this table exists
    alongside audit_log, not instead of it, so LLM calls are both
    queryable in their own right (this table) and part of the single
    unified narrative log every other day's decisions already use
    (audit_log). Never stores an API key, webhook secret, or raw auth
    header -- only `structured_output` (validated JSON) and metadata.

    `event_id` is set for the two per-event jobs (outreach microcopy,
    promise-to-pay parsing); `batch_id` is set instead for the batch-level
    explanation job. Exactly one of the two is populated per row.
    """
    __tablename__ = "llm_invocations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    batch_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # "outreach_microcopy" | "promise_to_pay_parse" | "batch_explanation"
    task_name: Mapped[str] = mapped_column(String(32), index=True)
    model_name: Mapped[str] = mapped_column(String(64))
    prompt_version: Mapped[str] = mapped_column(String(32))
    provider: Mapped[str] = mapped_column(String(16))  # "mock" | "anthropic"
    success: Mapped[bool] = mapped_column(Boolean)
    structured_output: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON-serialized structured_result, or fallback result if success=False
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class PromiseToPay(Base):
    """
    First-class, persistent promise-to-pay object -- the structured result of
    llm/service.py::parse_promise_to_pay, deterministically validated
    (policy/promise_to_pay.py) and, if valid, capable of overriding the
    policy-selected retry timing for its event (recovery/orchestrator.py).

    Deliberately does NOT store the customer's raw reply text -- only a
    SHA-256 hash of it (`source_text_hash`), used solely to detect an exact
    duplicate reply. This matches the project's existing convention: even
    `llm_invocations.structured_output` never stores raw customer text,
    only the validated structured parse.

    `status` lifecycle (see policy/promise_to_pay.py for the exact rules):
      VALID           -- date parsed, in the future, confidence >= threshold.
                         Eligible to override retry timing.
      LOW_CONFIDENCE  -- parsed cleanly but below the confidence threshold.
      INVALID_DATE    -- no date extracted, or not a parseable ISO date.
      EXPIRED         -- date parsed but not strictly in the future.
      SUPERSEDED      -- was VALID, but a newer distinct reply for the same
                         event_id replaced it as the active promise.
    Only these five states are modeled -- there is no live payment
    execution loop in this project to observe an actual FULFILLED/BROKEN
    outcome against, so those states would be theatrical, not real.

    `override_applied` / `override_outcome` are populated by the
    orchestrator every time it evaluates this promise against compliance --
    not merely whether the promise was VALID, but whether ITS retry timing
    was actually used for a real payment decision.
    """

    __tablename__ = "promises_to_pay"
    __table_args__ = (UniqueConstraint("event_id", "source_text_hash", name="uq_promise_event_text"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(Integer, index=True)  # FK to failure_events.id (logical) -- same convention as PolicyDecision.event_id
    subscription_id: Mapped[str] = mapped_column(String(64), index=True)

    promised_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # None unless status == VALID
    confidence: Mapped[float] = mapped_column(Float)
    channel: Mapped[str] = mapped_column(String(32))

    status: Mapped[str] = mapped_column(String(32), index=True)
    status_reason: Mapped[str] = mapped_column(Text)

    source_text_hash: Mapped[str] = mapped_column(String(64), index=True)  # sha256 hex of the raw reply text -- the raw text itself is never stored
    llm_invocation_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # FK to llm_invocations.id (logical) -- the parse that produced this row

    override_applied: Mapped[bool] = mapped_column(Boolean, default=False)  # this promise's timing was fed into a real orchestration decision
    override_outcome: Mapped[str | None] = mapped_column(String(64), nullable=True)  # "accepted" | "rejected_by_compliance: <reason>" | None (never evaluated yet)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
