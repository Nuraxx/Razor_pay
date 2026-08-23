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
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
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

    action: Mapped[str] = mapped_column(String(64))  # e.g. "webhook_received_and_stored"
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    actor: Mapped[str] = mapped_column(String(32), default="system")  # "system" | "rule" | "model" | "llm"

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
