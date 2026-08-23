"""
Day 1 scope, exactly:

Razorpay Test Mode -> webhook -> zrok -> FastAPI -> HMAC verification
-> idempotency check -> SQLite -> stored structured event.

No classification, no ML, no LLM, no policy logic, no UI. Those are Day 2+.
"""
import json
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, Response
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db, init_db
from app.logging_config import log
from app.models import AuditLog, RawEvent
from app.webhook_security import is_valid_signature


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate_webhook_secret_present()
    init_db()
    log.info("Startup complete. RAZORPAY_ENV=%s DATABASE_URL=%s", settings.RAZORPAY_ENV, settings.DATABASE_URL)
    yield


app = FastAPI(title="Adaptive Payment Recovery Agent — Day 1", lifespan=lifespan)


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
    return Response(status_code=200, content="stored")
