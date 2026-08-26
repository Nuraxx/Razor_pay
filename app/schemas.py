"""
Track-03: request models for the new revenue-risk API surface
(app/main.py's POST /events/* routes). The Razorpay webhook endpoint
(/webhook/razorpay) parses its own payload manually and is untouched --
these are ordinary Pydantic request bodies for the 4 new, non-Razorpay routes.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CheckoutAbandonedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cart_id: str
    customer_id: str
    cart_amount: float = Field(..., ge=0)
    checkout_started_at: datetime
    last_activity_at: datetime
    payment_method: str | None = None
    consent_for_communication: bool = True
    customer_opted_out: bool = False
    previous_outreach_count: int = Field(0, ge=0)
    language: str = "en"
    idempotency_key: str | None = None


class MandateFailedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mandate_id: str
    subscription_id: str | None = None
    amount: float = Field(..., ge=0)
    occurred_at: datetime
    current_step: str | None = None
    attempt_count: int = Field(0, ge=0)
    max_attempts: int = Field(3, ge=1)
    customer_opted_out: bool = False
    language: str = "en"
    idempotency_key: str | None = None


class ReceivableOverdueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoice_id: str
    customer_account_id: str
    invoice_amount: float = Field(..., ge=0)
    due_date: datetime
    days_overdue: int
    customer_segment: str = "unknown"
    is_disputed: bool = False
    has_active_promise: bool = False
    customer_opted_out: bool = False
    language: str = "en"
    idempotency_key: str | None = None


class PromiseToPayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: int
    subscription_id: str
    customer_reply_text: str = Field(..., min_length=1)
