"""
Razorpay Test Mode -> webhook -> zrok -> FastAPI -> HMAC verification
-> idempotency check -> SQLite -> stored structured event -> classification
-> recovery orchestration (policy/compliance/LLM) -> audit trail.

FIX #2 (full-system audit): the webhook handler used to stop at "stored" --
classification and orchestration required a separately-run script. It now
continues automatically into recovery/webhook_pipeline.py::process_raw_event
for every `payment.failed` event, right after the raw event is durably
committed. See that module's docstring for exactly which events qualify and
why, and the try/except below for why a downstream failure can never
un-store an already-verified webhook delivery.
"""
import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Request, Response
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db, init_db
from app.logging_config import log
from app.models import AuditLog, CheckoutSession, MandateRetrySequence, RawEvent, Receivable, RevenueRiskEvent
from app.schemas import CheckoutAbandonedRequest, MandateFailedRequest, PromiseToPayRequest, ReceivableOverdueRequest
from app.webhook_security import is_valid_signature
from model.unified_model import get_live_unified_model
from policy.policy_decision_store import REVENUE_DOMAIN_EVENT_ID_OFFSET
from recovery.promise_service import record_customer_reply
from recovery.revenue_orchestrator import orchestrate_revenue_event
from recovery.revenue_schemas import RevenueRiskEventInput
from recovery.retry_sweep import retry_sweep_background_loop
from recovery.scheduler import promise_sweep_background_loop
from recovery.webhook_pipeline import process_raw_event


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate_webhook_secret_present()
    init_db()
    get_live_unified_model()  # eager load + log at startup, not on the first webhook
    log.info("Startup complete. RAZORPAY_ENV=%s DATABASE_URL=%s", settings.RAZORPAY_ENV, settings.DATABASE_URL)

    # Track-03 hardening: automatic broken-promise detection. Started AFTER
    # startup logging above (never blocks startup itself -- asyncio.create_task
    # schedules it and returns immediately); cancelled cleanly on shutdown
    # below. See recovery/scheduler.py for why this never runs during tests.
    sweep_task: asyncio.Task | None = None
    if settings.ENABLE_PROMISE_SWEEP_SCHEDULER:
        sweep_task = asyncio.create_task(promise_sweep_background_loop(settings.PROMISE_SWEEP_INTERVAL_SECONDS))
        log.info("Promise sweep scheduler started (interval=%ss)", settings.PROMISE_SWEEP_INTERVAL_SECONDS)
    else:
        log.info("Promise sweep scheduler disabled (ENABLE_PROMISE_SWEEP_SCHEDULER=false)")

    # MULTI-ATTEMPT PERSISTENCE (final pre-submission audit): same
    # asyncio-loop pattern as the promise sweep above -- see
    # recovery/retry_sweep.py for what it advances and why.
    retry_sweep_task: asyncio.Task | None = None
    if settings.ENABLE_RETRY_SWEEP_SCHEDULER:
        retry_sweep_task = asyncio.create_task(retry_sweep_background_loop(settings.RETRY_SWEEP_INTERVAL_SECONDS))
        log.info("Retry sweep scheduler started (interval=%ss)", settings.RETRY_SWEEP_INTERVAL_SECONDS)
    else:
        log.info("Retry sweep scheduler disabled (ENABLE_RETRY_SWEEP_SCHEDULER=false)")

    yield

    if sweep_task is not None:
        sweep_task.cancel()
        try:
            await sweep_task
        except asyncio.CancelledError:
            pass
    if retry_sweep_task is not None:
        retry_sweep_task.cancel()
        try:
            await retry_sweep_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Adaptive Payment Recovery Agent", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Plain liveness check — hit this first to confirm the server is up before configuring zrok/webhooks."""
    return {"status": "ok", "env": settings.RAZORPAY_ENV}


@app.post("/webhook/razorpay")
async def razorpay_webhook(request: Request, db: Session = Depends(get_db)) -> Response:
    # 1. Read the RAW body bytes. This must happen before any JSON parsing —
    #    signature verification is computed over these exact bytes.
    raw_body: bytes = await request.body()

    signature = request.headers.get("x-razorpay-signature")
    event_id = request.headers.get("x-razorpay-event-id")

    # 2. Verify signature over the raw body. Never verify against a
    #    re-serialized/parsed version of the JSON.
    if not is_valid_signature(raw_body, signature, settings.RAZORPAY_WEBHOOK_SECRET):
        log.warning("Webhook rejected: invalid or missing signature. event_id=%s", event_id)
        return Response(status_code=400, content="invalid signature")

    # 3. A verified Razorpay webhook always carries x-razorpay-event-id per
    #    Razorpay's own idempotency documentation. Its absence on an
    #    otherwise-valid-signature request is treated as malformed.
    if not event_id:
        log.warning("Webhook rejected: missing x-razorpay-event-id header despite valid signature.")
        return Response(status_code=400, content="missing x-razorpay-event-id")

    # 4. Idempotency check BEFORE parsing/storing anything else. Duplicate
    #    deliveries are expected Razorpay behavior, not an error — Razorpay
    #    still expects a 2xx, or it will keep retrying for up to 24 hours.
    existing = db.query(RawEvent).filter(RawEvent.razorpay_event_id == event_id).first()
    if existing is not None:
        log.info("Duplicate webhook ignored. event_id=%s already stored as raw_events.id=%s", event_id, existing.id)
        return Response(status_code=200, content="duplicate, already processed")

    # 5. Parse JSON only after signature + idempotency checks pass.
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        log.warning("Webhook rejected: valid signature but body is not valid JSON. event_id=%s", event_id)
        return Response(status_code=400, content="malformed json body")

    event_type = payload.get("event", "unknown")
    payment_entity = (payload.get("payload") or {}).get("payment", {}).get("entity", {}) or {}
    subscription_entity = (payload.get("payload") or {}).get("subscription", {}).get("entity", {}) or {}

    raw_event = RawEvent(
        razorpay_event_id=event_id,
        event_type=event_type,
        payment_id=payment_entity.get("id"),
        subscription_id=subscription_entity.get("id"),
        order_id=payment_entity.get("order_id"),
        amount=payment_entity.get("amount"),
        currency=payment_entity.get("currency"),
        error_code=payment_entity.get("error_code"),
        error_description=payment_entity.get("error_description"),
        error_reason=payment_entity.get("error_reason"),
        error_source=payment_entity.get("error_source"),
        error_step=payment_entity.get("error_step"),
        razorpay_created_at=payload.get("created_at"),
        signature_verified=True,
        raw_payload=raw_body.decode("utf-8", errors="replace"),
    )
    db.add(raw_event)
    db.flush()  # populate raw_event.id before referencing it in the audit log

    db.add(
        AuditLog(
            raw_event_id=raw_event.id,
            action="webhook_received_and_stored",
            reason=f"event_type={event_type} error_reason={payment_entity.get('error_reason')}",
            actor="system",
        )
    )
    db.commit()

    log.info(
        "Stored event_type=%s event_id=%s payment_id=%s error_reason=%s (raw_events.id=%s)",
        event_type, event_id, payment_entity.get("id"), payment_entity.get("error_reason"), raw_event.id,
    )

    # 6. Continue automatically into classification + orchestration (FIX #2).
    #    The raw event above is ALREADY committed -- a failure here can
    #    never un-store an already-verified webhook delivery, never
    #    duplicates any business action (process_raw_event's own stages are
    #    each independently idempotent), and never turns this response into
    #    a 4xx/5xx (which would just make Razorpay redeliver a payload whose
    #    storage already succeeded -- redelivery cannot fix an orchestration
    #    bug). The response body honestly reports what actually happened.
    try:
        orchestration_outcome = process_raw_event(db, raw_event, model=get_live_unified_model())
    except Exception:
        db.rollback()
        log.exception("Orchestration failed for raw_events.id=%s after successful storage -- raw event remains stored and reprocessable", raw_event.id)
        db.add(
            AuditLog(
                raw_event_id=raw_event.id,
                action="orchestration_failed_after_storage",
                reason="unhandled exception during classify+orchestrate; raw event remains stored; safe to reprocess via scripts/reprocess_raw_events.py",
                actor="system",
            )
        )
        db.commit()
        orchestration_outcome = "failed"

    return Response(status_code=200, content=f"stored; orchestration={orchestration_outcome}")


# ---------------------------------------------------------------------------
# Track-03: revenue-risk event API -- the non-Razorpay event types. Each
# route: idempotency check -> store RevenueRiskEvent + domain detail row ->
# orchestrate (same never-unstore-on-failure pattern as /webhook/razorpay
# above) -> return a structured JSON result. All 4 routes funnel through the
# SAME recovery/revenue_orchestrator.py::orchestrate_revenue_event -- no
# duplicated orchestration logic.
# ---------------------------------------------------------------------------

def _to_naive(dt: datetime) -> datetime:
    """This project's policy/recovery layers use naive datetimes throughout
    (see recovery/orchestrator.py, policy/compliance.py) -- strip tzinfo from
    any tz-aware request input rather than let a naive/aware comparison
    raise deep inside the orchestrator."""
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def _run_revenue_orchestration_safely(db: Session, event: RevenueRiskEventInput, revenue_risk_event: RevenueRiskEvent) -> dict:
    """Same never-unstore-on-failure contract as razorpay_webhook's own try/
    except above: the RevenueRiskEvent row is already committed by the time
    this runs, so an orchestration failure here can never lose the stored
    event, never duplicates any business action on retry (every stage below
    is independently idempotent), and is reported honestly rather than
    turned into a 4xx/5xx."""
    try:
        result = orchestrate_revenue_event(db, event, model=get_live_unified_model())
        return {"status": "processed", "revenue_risk_event_id": revenue_risk_event.id, "orchestration": result.to_dict()}
    except Exception:
        db.rollback()
        log.exception(
            "Orchestration failed for revenue_risk_events.id=%s after successful storage -- event remains stored and reprocessable",
            revenue_risk_event.id,
        )
        db.add(
            AuditLog(
                failure_event_id=revenue_risk_event.id + REVENUE_DOMAIN_EVENT_ID_OFFSET,
                action="revenue_orchestration_failed_after_storage",
                reason="unhandled exception during revenue-risk orchestration; event remains stored",
                actor="system",
            )
        )
        db.commit()
        return {"status": "stored_orchestration_failed", "revenue_risk_event_id": revenue_risk_event.id}


@app.post("/events/checkout-abandoned")
def checkout_abandoned(body: CheckoutAbandonedRequest, db: Session = Depends(get_db)) -> dict:
    idempotency_key = body.idempotency_key or f"checkout_abandoned:{body.cart_id}"
    existing = db.query(RevenueRiskEvent).filter(RevenueRiskEvent.idempotency_key == idempotency_key).first()
    if existing is not None:
        log.info("Duplicate checkout-abandoned event ignored. idempotency_key=%s already revenue_risk_events.id=%s", idempotency_key, existing.id)
        return {"status": "duplicate", "revenue_risk_event_id": existing.id}

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    last_activity_at = _to_naive(body.last_activity_at)
    inactivity_minutes = max((now - last_activity_at).total_seconds() / 60.0, 0.0)

    revenue_risk_event = RevenueRiskEvent(
        idempotency_key=idempotency_key, event_type="checkout_abandoned", external_id=body.cart_id,
        customer_ref=body.customer_id, amount=body.cart_amount, currency="INR", occurred_at=now,
        reason="checkout_inactivity", context_json=json.dumps({"payment_method": body.payment_method}), status="OPEN",
    )
    db.add(revenue_risk_event)
    db.flush()
    db.add(CheckoutSession(
        revenue_risk_event_id=revenue_risk_event.id, cart_id=body.cart_id, customer_id=body.customer_id,
        cart_amount=body.cart_amount, payment_method=body.payment_method,
        checkout_started_at=_to_naive(body.checkout_started_at), last_activity_at=last_activity_at,
        state="CHECKOUT_STARTED", inactivity_minutes=inactivity_minutes,
        consent_for_communication=body.consent_for_communication, previous_outreach_count=body.previous_outreach_count,
    ))
    db.add(AuditLog(failure_event_id=revenue_risk_event.id + REVENUE_DOMAIN_EVENT_ID_OFFSET, action="revenue_risk_event_received_and_stored", reason=f"event_type=checkout_abandoned cart_id={body.cart_id}", actor="system"))
    db.commit()

    event = RevenueRiskEventInput(
        event_type="checkout_abandoned", event_id=revenue_risk_event.id, customer_ref=body.customer_id, occurred_at=now,
        amount=body.cart_amount, language=body.language, consent_for_communication=body.consent_for_communication,
        customer_opted_out=body.customer_opted_out,
        domain_context={
            "cart_amount": body.cart_amount, "inactivity_minutes": inactivity_minutes,
            "previous_outreach_count": body.previous_outreach_count, "payment_method": body.payment_method,
        },
    )
    return _run_revenue_orchestration_safely(db, event, revenue_risk_event)


@app.post("/events/mandate-failed")
def mandate_failed(body: MandateFailedRequest, db: Session = Depends(get_db)) -> dict:
    idempotency_key = body.idempotency_key or f"mandate_failed:{body.mandate_id}"
    existing = db.query(RevenueRiskEvent).filter(RevenueRiskEvent.idempotency_key == idempotency_key).first()
    if existing is not None:
        log.info("Duplicate mandate-failed event ignored. idempotency_key=%s already revenue_risk_events.id=%s", idempotency_key, existing.id)
        return {"status": "duplicate", "revenue_risk_event_id": existing.id}

    occurred_at = _to_naive(body.occurred_at)
    customer_ref = body.subscription_id or body.mandate_id

    revenue_risk_event = RevenueRiskEvent(
        idempotency_key=idempotency_key, event_type="mandate_failed", external_id=body.mandate_id,
        customer_ref=customer_ref, amount=body.amount, currency="INR", occurred_at=occurred_at,
        reason="mandate_payment_failed",
        context_json=json.dumps({"current_step": body.current_step, "attempt_count": body.attempt_count}), status="OPEN",
    )
    db.add(revenue_risk_event)
    db.flush()
    db.add(MandateRetrySequence(
        revenue_risk_event_id=revenue_risk_event.id, mandate_id=body.mandate_id, subscription_id=body.subscription_id,
        sequence_status="PLANNED", current_step=body.current_step or "attempt_1",
        attempt_count=body.attempt_count, max_attempts=body.max_attempts, retry_reason="mandate_payment_failed",
    ))
    db.add(AuditLog(failure_event_id=revenue_risk_event.id + REVENUE_DOMAIN_EVENT_ID_OFFSET, action="revenue_risk_event_received_and_stored", reason=f"event_type=mandate_failed mandate_id={body.mandate_id}", actor="system"))
    db.commit()

    event = RevenueRiskEventInput(
        event_type="mandate_failed", event_id=revenue_risk_event.id, customer_ref=customer_ref, occurred_at=occurred_at,
        amount=body.amount, language=body.language, customer_opted_out=body.customer_opted_out,
        domain_context={"current_step": body.current_step, "attempt_count": body.attempt_count, "max_attempts": body.max_attempts},
    )
    return _run_revenue_orchestration_safely(db, event, revenue_risk_event)


@app.post("/events/receivable-overdue")
def receivable_overdue(body: ReceivableOverdueRequest, db: Session = Depends(get_db)) -> dict:
    idempotency_key = body.idempotency_key or f"receivable_overdue:{body.invoice_id}"
    existing = db.query(RevenueRiskEvent).filter(RevenueRiskEvent.idempotency_key == idempotency_key).first()
    if existing is not None:
        log.info("Duplicate receivable-overdue event ignored. idempotency_key=%s already revenue_risk_events.id=%s", idempotency_key, existing.id)
        return {"status": "duplicate", "revenue_risk_event_id": existing.id}

    now = datetime.now(timezone.utc).replace(tzinfo=None)

    revenue_risk_event = RevenueRiskEvent(
        idempotency_key=idempotency_key, event_type="receivable_overdue", external_id=body.invoice_id,
        customer_ref=body.customer_account_id, amount=body.invoice_amount, currency="INR", occurred_at=now,
        reason="invoice_overdue",
        context_json=json.dumps({"days_overdue": body.days_overdue, "is_disputed": body.is_disputed}), status="OPEN",
    )
    db.add(revenue_risk_event)
    db.flush()
    db.add(Receivable(
        revenue_risk_event_id=revenue_risk_event.id, invoice_id=body.invoice_id, customer_account_id=body.customer_account_id,
        invoice_amount=body.invoice_amount, due_date=_to_naive(body.due_date), days_overdue=body.days_overdue,
        customer_segment=body.customer_segment, escalation_bucket="unclassified", status="OPEN",
    ))
    db.add(AuditLog(failure_event_id=revenue_risk_event.id + REVENUE_DOMAIN_EVENT_ID_OFFSET, action="revenue_risk_event_received_and_stored", reason=f"event_type=receivable_overdue invoice_id={body.invoice_id}", actor="system"))
    db.commit()

    event = RevenueRiskEventInput(
        event_type="receivable_overdue", event_id=revenue_risk_event.id, customer_ref=body.customer_account_id, occurred_at=now,
        amount=body.invoice_amount, customer_segment=body.customer_segment, language=body.language,
        customer_opted_out=body.customer_opted_out,
        domain_context={"days_overdue": body.days_overdue, "is_disputed": body.is_disputed, "has_active_promise": body.has_active_promise},
    )
    return _run_revenue_orchestration_safely(db, event, revenue_risk_event)


@app.post("/events/promise-to-pay")
def promise_to_pay(body: PromiseToPayRequest, db: Session = Depends(get_db)) -> dict:
    """Thin wrapper directly over recovery/promise_service.py::record_customer_reply
    -- no RevenueRiskEvent involved, since that flow already exists and works
    end to end (LLM communication + compliance). Does NOT duplicate the Razorpay webhook endpoint."""
    promise, created = record_customer_reply(
        db, event_id=body.event_id, subscription_id=body.subscription_id, customer_reply_text=body.customer_reply_text,
    )
    return {"status": "processed" if created else "duplicate", "promise_to_pay_id": promise.id, "promise_status": promise.status}
