"""
Track-03: a single demo runner that generates one synthetic event of EACH
kind Track 03 covers, each routed through the REAL recovery engine (never a
raw DB insert) -- brief section 15: "Each generated event must enter the
SAME recovery engine." Defaults to a throwaway in-memory DB (mirrors
ui/data.py::run_demo_scenario's exact pattern) so it never pollutes the real
Razorpay-webhook-backed database unless the caller explicitly passes a live
session.

The 7 kinds (brief section 15):
  1. failed payment              -- recovery/orchestrator.py::orchestrate_recovery (UNCHANGED)
  2. checkout abandonment        -- recovery/revenue_orchestrator.py::orchestrate_revenue_event
  3. subscription failure        -- orchestrate_recovery again, subscription-linked amount/context
  4. mandate failure             -- orchestrate_revenue_event
  5. overdue receivable          -- orchestrate_revenue_event
  6. promise-to-pay              -- recovery/promise_service.py::record_customer_reply (UNCHANGED)
  7. broken promise              -- an already-past-due promise + recovery/promise_lifecycle.py::mark_broken_promises,
                                     which itself opens a revenue_risk_events row routed through orchestrate_revenue_event
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.models import CheckoutSession, MandateRetrySequence, PromiseToPay, Receivable, RevenueRiskEvent
from classification.rules import classify
from llm.client import LLMClient
from model.unified_model import get_live_unified_model
from policy.promise_to_pay import STATUS_VALID
from recovery.orchestrator import RecoveryEventInput, orchestrate_recovery
from recovery.promise_lifecycle import mark_broken_promises
from recovery.promise_service import record_customer_reply
from recovery.revenue_orchestrator import orchestrate_revenue_event
from recovery.revenue_schemas import RevenueRiskEventInput

DEMO_TIMESTAMP = datetime(2026, 8, 25, 9, 0, 0)


def build_demo_database() -> Session:
    """Throwaway in-memory SQLite DB -- same engine construction as
    ui/data.py::run_demo_scenario, plus expire_on_commit=False: unlike that
    function (which only ever returns plain dataclasses / freshly-queried
    rows), this generator returns raw ORM objects (promise, broken_promise)
    straight out of functions that already committed internally -- without
    this, reading their attributes after db.close() below raises
    DetachedInstanceError."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, expire_on_commit=False)()


def generate_demo_revenue_risk_events(
    db: Session | None = None, *, model: dict | None = None, unified_model: dict | None = None, llm_client: LLMClient | None = None,
) -> dict[str, object]:
    """Returns {kind: result} for all 7 kinds. `db=None` (default) builds and
    uses a throwaway in-memory DB -- pass a real session explicitly (e.g.
    ui.data.get_live_session()) only when deliberately targeting the live
    database (brief: "without polluting the real Razorpay webhook history
    unless explicitly triggered").

    NOTE two DIFFERENT model shapes flow through this function: `model` is
    Model B (policy/decision_engine_v4.py's calibrated CatBoost + imputer
    tuple) -- used ONLY by the two orchestrate_recovery() legs below
    (failed_payment, subscription_failure). `unified_model` is the unified
    ML model (model/unified_model.py) -- used ONLY by the orchestrate_revenue_event()
    legs (checkout/mandate/receivable/broken_promise). They are never
    interchangeable; passing one where the other is expected would silently
    fail structurally and fall back to the rule-based/Model B tier (each
    orchestrator's own model-unavailable handling), never crash but also
    never actually exercise the model you meant to demo. Defaults to the
    SAME cached artifact the live app uses (get_live_unified_model()) so a
    demo run is never a second, parallel inference implementation."""
    owns_db = db is None
    db = db or build_demo_database()
    if unified_model is None:
        unified_model = get_live_unified_model()
    results: dict[str, object] = {}

    # --- 1. Failed payment (existing, unmodified orchestrate_recovery) -----
    bucket = classify(None, "insufficient_fund").bucket
    payment_event = RecoveryEventInput(
        event_id=900101, subscription_id="demo_sub_payment_failed", failure_timestamp=DEMO_TIMESTAMP,
        amount=799.0, error_code=None, error_reason="insufficient_fund", customer_segment="mid", language="en",
    )
    results["failed_payment"] = orchestrate_recovery(db, payment_event, model=model, llm_client=llm_client)

    # --- 2. Checkout abandonment --------------------------------------------
    # Query-before-insert (same pattern app/main.py's routes use): the
    # default in-memory DB is always fresh so this never fires there, but
    # this function's own docstring documents passing a caller-owned LIVE
    # session as a supported use -- calling it twice against the SAME live
    # session would otherwise hit the idempotency_key unique constraint as a
    # hard IntegrityError instead of gracefully reusing the existing event.
    rre_checkout = db.query(RevenueRiskEvent).filter(RevenueRiskEvent.idempotency_key == "demo:checkout_abandoned:1").first()
    if rre_checkout is None:
        rre_checkout = RevenueRiskEvent(
            idempotency_key="demo:checkout_abandoned:1", event_type="checkout_abandoned", external_id="demo_cart_1",
            customer_ref="demo_cust_checkout", amount=999.0, occurred_at=DEMO_TIMESTAMP, reason="checkout_inactivity", status="OPEN",
        )
        db.add(rre_checkout)
        db.flush()
        db.add(CheckoutSession(
            revenue_risk_event_id=rre_checkout.id, cart_id="demo_cart_1", customer_id="demo_cust_checkout",
            cart_amount=999.0, checkout_started_at=DEMO_TIMESTAMP - timedelta(hours=2), last_activity_at=DEMO_TIMESTAMP - timedelta(hours=2),
            state="CHECKOUT_STARTED", inactivity_minutes=120.0,
        ))
        db.commit()
    checkout_event = RevenueRiskEventInput(
        event_type="checkout_abandoned", event_id=rre_checkout.id, customer_ref="demo_cust_checkout", occurred_at=DEMO_TIMESTAMP,
        amount=999.0, domain_context={"cart_amount": 999.0, "inactivity_minutes": 120.0, "previous_outreach_count": 0},
    )
    results["checkout_abandoned"] = orchestrate_revenue_event(db, checkout_event, model=unified_model, llm_client=llm_client)

    # --- 3. Subscription failure (existing orchestrate_recovery, subscription-linked) ---
    subscription_event = RecoveryEventInput(
        event_id=900102, subscription_id="demo_sub_subscription_failed", failure_timestamp=DEMO_TIMESTAMP,
        amount=1499.0, error_code=None, error_reason="bank_technical_error", customer_segment="high", language="en",
    )
    results["subscription_failure"] = orchestrate_recovery(db, subscription_event, model=model, llm_client=llm_client)

    # --- 4. Mandate failure -------------------------------------------------
    rre_mandate = db.query(RevenueRiskEvent).filter(RevenueRiskEvent.idempotency_key == "demo:mandate_failed:1").first()
    if rre_mandate is None:
        rre_mandate = RevenueRiskEvent(
            idempotency_key="demo:mandate_failed:1", event_type="mandate_failed", external_id="demo_mandate_1",
            customer_ref="demo_sub_mandate", amount=1200.0, occurred_at=DEMO_TIMESTAMP, reason="mandate_payment_failed", status="OPEN",
        )
        db.add(rre_mandate)
        db.flush()
        db.add(MandateRetrySequence(
            revenue_risk_event_id=rre_mandate.id, mandate_id="demo_mandate_1", subscription_id="demo_sub_mandate",
            sequence_status="PLANNED", current_step="attempt_1", attempt_count=0, max_attempts=3, retry_reason="mandate_payment_failed",
        ))
        db.commit()
    mandate_event = RevenueRiskEventInput(
        event_type="mandate_failed", event_id=rre_mandate.id, customer_ref="demo_sub_mandate", occurred_at=DEMO_TIMESTAMP, amount=1200.0,
    )
    results["mandate_failed"] = orchestrate_revenue_event(db, mandate_event, model=unified_model, llm_client=llm_client)

    # --- 5. Overdue receivable ------------------------------------------------
    rre_receivable = db.query(RevenueRiskEvent).filter(RevenueRiskEvent.idempotency_key == "demo:receivable_overdue:1").first()
    if rre_receivable is None:
        rre_receivable = RevenueRiskEvent(
            idempotency_key="demo:receivable_overdue:1", event_type="receivable_overdue", external_id="demo_invoice_1",
            customer_ref="demo_acct_1", amount=45000.0, occurred_at=DEMO_TIMESTAMP, reason="invoice_overdue", status="OPEN",
        )
        db.add(rre_receivable)
        db.flush()
        db.add(Receivable(
            revenue_risk_event_id=rre_receivable.id, invoice_id="demo_invoice_1", customer_account_id="demo_acct_1",
            invoice_amount=45000.0, due_date=DEMO_TIMESTAMP - timedelta(days=40), days_overdue=40,
            customer_segment="enterprise", escalation_bucket="unclassified", status="OPEN",
        ))
        db.commit()
    receivable_event = RevenueRiskEventInput(
        event_type="receivable_overdue", event_id=rre_receivable.id, customer_ref="demo_acct_1", occurred_at=DEMO_TIMESTAMP,
        amount=45000.0, customer_segment="enterprise", domain_context={"days_overdue": 40},
    )
    results["receivable_overdue"] = orchestrate_revenue_event(db, receivable_event, model=unified_model, llm_client=llm_client)

    # --- 6. Promise-to-pay (existing, unmodified record_customer_reply) -----
    promise, _created = record_customer_reply(
        db, event_id=900101, subscription_id="demo_sub_payment_failed",
        customer_reply_text="I'll pay tomorrow via UPI", today=DEMO_TIMESTAMP.date(), client=llm_client,
    )
    results["promise_to_pay"] = promise

    # --- 7. Broken promise (an already-past-due VALID promise, swept) -------
    broken_promise = db.query(PromiseToPay).filter(PromiseToPay.source_text_hash == "demo_broken_promise_hash").first()
    if broken_promise is None:
        broken_promise = PromiseToPay(
            event_id=900103, subscription_id="demo_sub_broken_promise", promised_date=DEMO_TIMESTAMP - timedelta(days=3),
            confidence=0.9, channel="upi_autopay", status=STATUS_VALID, status_reason="demo_fixture",
            source_text_hash="demo_broken_promise_hash",
        )
        db.add(broken_promise)
        db.flush()
        db.commit()
    broken_outcomes = mark_broken_promises(db, as_of=DEMO_TIMESTAMP)
    if broken_outcomes:
        outcome = broken_outcomes[0]
        rre_broken = db.query(RevenueRiskEvent).filter(RevenueRiskEvent.id == outcome.reevaluation_event_id).first()
        broken_context = json.loads(rre_broken.context_json or "{}")
        broken_event = RevenueRiskEventInput(
            event_type="promise_to_pay_broken", event_id=rre_broken.id, customer_ref=rre_broken.customer_ref,
            occurred_at=rre_broken.occurred_at, amount=broken_context.get("original_amount", 0.0) or 0.0, domain_context=broken_context,
        )
        results["broken_promise"] = orchestrate_revenue_event(db, broken_event, model=unified_model, llm_client=llm_client)
    else:
        results["broken_promise"] = None

    if owns_db:
        db.close()
    return results
