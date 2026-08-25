"""
Day-12 end-to-end demo CLI, extended in the FIX pass to also demonstrate
FIX #1 (promise-to-pay override) and FIX #2 (webhook -> automatic
orchestration):

    EVENT -> CLASSIFICATION -> POLICY DECISION -> COMPLIANCE -> PAYMENT ACTION
    -> LLM COMMUNICATION -> FINAL RESULT -> AUDIT TRAIL

Runs fully offline: synthetic events, the real Day-8/Day-10 trained model
artifact, deterministic compliance, and the Day-11 mock LLM provider (no
ANTHROPIC_API_KEY needed). No real payment retry or real message send is
ever attempted, and no real Razorpay HTTP call is made anywhere below --
scenario 5's "webhook" is a signed, in-process HTTP request to this
project's own FastAPI app (`fastapi.testclient.TestClient`), not a call to
Razorpay's servers.

Usage (from the project root):

    ./venv/bin/python scripts/run_end_to_end_demo.py

Uses a throwaway in-memory SQLite database per scenario (never touches
data/recovery_agent.db) so it's safe to re-run with no side effects.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import sys
from datetime import datetime
from pathlib import Path

# Documented usage is `./venv/bin/python scripts/run_end_to_end_demo.py` --
# running a script directly (not via `-m`) only puts scripts/ on sys.path,
# not the project root, so the `app`/`llm`/`model`/`recovery` package
# imports below would otherwise fail with ModuleNotFoundError regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.models import AuditLog, PolicyDecision, RawEvent
from llm.client import LLMClient, LLMProviderError
from model.train_latent_target_model import load_latent_target_model
from recovery.orchestrator import RecoveryEventInput, orchestrate_recovery
from recovery.promise_service import record_customer_reply

FAILURE_CONTEXT = {
    "day_of_month": 24, "days_to_nearest_payday_window": 6, "prior_if_failure_count": 0,
    "prior_if_self_resolved_rate": float("nan"), "tenure_days": 200, "plan_tier": "mid",
    "primary_instrument": "upi_autopay", "city_tier": "tier_1", "bank_network_conditions": "good",
    "issuing_bank_downtime_flag": False, "network_latency_bucket": "low", "is_month_end_settlement_rush": False,
}
FAILURE_TS = datetime(2026, 2, 24, 10, 0, 0)


class _AlwaysFailsClient(LLMClient):
    """Injected for scenario 3 to force a real, deterministic LLM failure --
    used only in this demo, never touches a real network."""

    model_name = "demo-broken-client"
    provider_name = "mock"

    def complete(self, system_prompt, user_prompt, *, max_tokens=512):
        raise LLMProviderError("simulated_provider_outage")


def _fresh_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def _print_flow(db, result, event, title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print("EVENT")
    print(f"  event_id={event.event_id} subscription_id={event.subscription_id} error_reason={event.error_reason!r} amount=Rs{event.amount}")
    print("  |")
    print("  v")
    print("CLASSIFICATION")
    print(f"  bucket={result.classification_bucket!r} confidence={result.classification_confidence}")
    print("  |")
    print("  v")
    print("POLICY DECISION (Day-10 policy-v4, unmodified)")
    print(f"  selected_candidate_type={result.original_candidate_type!r} decision_source={result.decision_source!r}")
    print(f"  policy_version={result.policy_version!r}")
    if result.promise_to_pay_id is not None:
        print("  |")
        print("  v")
        print("PROMISE-TO-PAY OVERRIDE (FIX #1 -- recovery/promise_service.py + recovery/orchestrator.py)")
        print(f"  promises_to_pay.id={result.promise_to_pay_id} applied={result.promise_to_pay_applied}")
        print(f"  original_candidate={result.original_candidate_type!r}@{result.original_candidate_datetime}")
        print(f"  final_candidate={result.selected_candidate_type!r}@{result.selected_candidate_datetime}")
    print("  |")
    print("  v")
    print("COMPLIANCE")
    print(f"  compliance_allowed={result.compliance_allowed} reason={result.compliance_reason!r}")
    print("  |")
    print("  v")
    print("PAYMENT ACTION (recorded only -- no live Razorpay call)")
    print(f"  payment_action={result.payment_action!r}")
    print("  |")
    print("  v")
    print("LLM COMMUNICATION (Day-11, mock provider)")
    print(f"  communication_action={result.communication_action!r} llm_task_name={result.llm_task_name!r} llm_success={result.llm_success}")
    print("  |")
    print("  v")
    print("FINAL RESULT")
    print(f"  final_status={result.final_status}")
    print("  |")
    print("  v")
    print("AUDIT TRAIL")
    rows = db.query(AuditLog).filter(AuditLog.failure_event_id == event.event_id).order_by(AuditLog.id).all()
    for row in rows:
        print(f"  actor={row.actor:12s} action={row.action}")


def _print_webhook_flow(response, db) -> None:
    print()
    print("=" * 78)
    print("SCENARIO 5: Webhook ingestion -> automatic orchestration (FIX #2)")
    print("=" * 78)
    print("WEBHOOK POST /webhook/razorpay (HMAC-signed, in-process TestClient -- not a real Razorpay call)")
    print(f"  http_status={response.status_code} body={response.text!r}")
    print("  |")
    print("  v")
    print("app/main.py: verify signature -> idempotency check -> store RawEvent -> commit")
    print("  |")
    print("  v")
    print("recovery/webhook_pipeline.py::process_raw_event -- classification + full orchestration, automatically")
    decision = db.query(PolicyDecision).order_by(PolicyDecision.id.desc()).first()
    print(f"  selected_candidate_type={decision.selected_candidate_type!r} decision_source={decision.decision_source!r}")
    print("  |")
    print("  v")
    print("AUDIT TRAIL")
    raw_event = db.query(RawEvent).order_by(RawEvent.id.desc()).first()
    rows = db.query(AuditLog).filter(AuditLog.raw_event_id == raw_event.id).order_by(AuditLog.id).all()
    rows += db.query(AuditLog).filter(AuditLog.failure_event_id == decision.event_id).order_by(AuditLog.id).all()
    for row in rows:
        print(f"  actor={row.actor:12s} action={row.action}")


def main() -> None:
    model = load_latent_target_model("value")

    # --- Scenario 1: insufficient_fund normal recovery -------------------
    db1 = _fresh_db()
    event1 = RecoveryEventInput(
        event_id=1, subscription_id="sub_demo_success", failure_timestamp=FAILURE_TS, amount=799.0,
        error_code=None, error_reason="insufficient_fund", failure_context=FAILURE_CONTEXT,
        customer_segment="mid", language="en",
    )
    result1 = orchestrate_recovery(db1, event1, model=model)
    _print_flow(db1, result1, event1, "SCENARIO 1: insufficient_fund -- normal recovery flow")

    # --- Scenario 2: hard decline -> payment-method-update communication (FIX #3) ---
    db2 = _fresh_db()
    event2 = RecoveryEventInput(
        event_id=2, subscription_id="sub_demo_hard_decline", failure_timestamp=FAILURE_TS, amount=499.0,
        error_code=None, error_reason="card_expired", failure_context=FAILURE_CONTEXT,
        customer_segment="mobile", language="hinglish",
    )
    result2 = orchestrate_recovery(db2, event2, model=model)
    _print_flow(db2, result2, event2, "SCENARIO 2: hard_decline -- no retry, payment-method-update nudge (FIX #3)")

    # --- Scenario 2b: blocked communication independent of payment ------
    db2b = _fresh_db()
    event2b = RecoveryEventInput(
        event_id=3, subscription_id="sub_demo_optout", failure_timestamp=FAILURE_TS, amount=299.0,
        error_code=None, error_reason="insufficient_fund", failure_context=FAILURE_CONTEXT,
        customer_segment="mid", language="en", customer_opted_out=True,
    )
    result2b = orchestrate_recovery(db2b, event2b, model=model)
    _print_flow(db2b, result2b, event2b, "SCENARIO 2b: Payment allowed, communication blocked (customer opted out)")

    # --- Scenario 3: customer reply -> promise-to-pay -> retry override (FIX #1) ---
    db3 = _fresh_db()
    event3 = RecoveryEventInput(
        event_id=4, subscription_id="sub_demo_promise", failure_timestamp=FAILURE_TS, amount=2999.0,
        error_code=None, error_reason="insufficient_fund", failure_context=FAILURE_CONTEXT,
        customer_segment="mid", language="en",
    )
    print()
    print("=" * 78)
    print("SCENARIO 3: customer reply -> promise-to-pay -> retry override (FIX #1)")
    print("=" * 78)
    print("CUSTOMER REPLY (free text, parsed by llm/service.py::parse_promise_to_pay, mock provider)")
    print('  "I\'ll pay Friday when salary comes"')
    record_customer_reply(
        db3, event_id=event3.event_id, subscription_id=event3.subscription_id,
        customer_reply_text="I'll pay Friday when salary comes", today=FAILURE_TS.date(),
    )
    result3 = orchestrate_recovery(db3, event3, model=model)
    _print_flow(db3, result3, event3, "SCENARIO 3 (continued): orchestration with the promise applied")

    # --- Scenario 4: LLM failure flow ------------------------------------
    # amount=2999 is chosen so the REAL Day-10 model confidently selects via
    # its primary tier (decision_source="day8_model_b", not policy's own
    # fallback) -- isolating the LLM failure as the one thing going wrong in
    # this scenario, so final_status lands on LLM_FALLBACK specifically
    # rather than being masked by POLICY_FALLBACK (both are legitimate,
    # simultaneously-true statuses when they co-occur -- see
    # recovery/schemas.py's precedence docstring -- this amount just avoids
    # that overlap for a clearer demo).
    db4 = _fresh_db()
    event4 = RecoveryEventInput(
        event_id=5, subscription_id="sub_demo_llm_failure", failure_timestamp=FAILURE_TS, amount=2999.0,
        error_code=None, error_reason="insufficient_fund", failure_context=FAILURE_CONTEXT,
        customer_segment="premium", language="en",
    )
    result4 = orchestrate_recovery(db4, event4, model=model, llm_client=_AlwaysFailsClient())
    _print_flow(db4, result4, event4, "SCENARIO 4: LLM failure flow (payment decision unaffected)")

    # --- Scenario 5: webhook ingestion -> automatic orchestration (FIX #2) ---
    # A real, signed HTTP POST to this project's own FastAPI app -- exercises
    # app/main.py's actual request path (HMAC verification, idempotency,
    # storage) end to end into recovery/webhook_pipeline.py, exactly as a
    # live Razorpay delivery would, but entirely in-process (TestClient) and
    # against a throwaway DB override, never data/recovery_agent.db and never
    # a real network call.
    from fastapi.testclient import TestClient

    from app.config import settings
    from app.db import get_db
    from app.main import app as fastapi_app

    webhook_secret = "demo_webhook_secret_for_this_script_only"
    settings.RAZORPAY_WEBHOOK_SECRET = webhook_secret
    db5 = _fresh_db()
    fastapi_app.dependency_overrides[get_db] = lambda: db5
    payload = {
        "entity": "event", "account_id": "acc_DemoAccount", "event": "payment.failed", "contains": ["payment"],
        "payload": {
            "payment": {"entity": {
                "id": "pay_DemoWebhookEvent", "entity": "payment", "amount": 99900, "currency": "INR", "status": "failed",
                "order_id": None, "method": "card", "error_code": "BAD_REQUEST_ERROR",
                "error_description": "Payment failed due to insufficient funds in the customer account.",
                "error_reason": "insufficient_fund", "error_source": "customer", "error_step": "payment_authorization",
            }},
            "subscription": {"entity": {"id": "sub_demo_webhook", "entity": "subscription", "plan_id": "plan_Demo", "status": "active"}},
        },
        "created_at": int(FAILURE_TS.timestamp()),
    }
    body = json.dumps(payload).encode("utf-8")
    signature = hmac.new(webhook_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    client = TestClient(fastapi_app)
    response = client.post(
        "/webhook/razorpay", content=body,
        headers={"Content-Type": "application/json", "x-razorpay-signature": signature, "x-razorpay-event-id": "evt_DemoWebhookEvent"},
    )
    _print_webhook_flow(response, db5)
    fastapi_app.dependency_overrides.clear()

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(json.dumps({
        "scenario_1_success": result1.to_dict(),
        "scenario_2_hard_decline": result2.to_dict(),
        "scenario_2b_communication_blocked": result2b.to_dict(),
        "scenario_3_promise_to_pay": result3.to_dict(),
        "scenario_4_llm_failure": result4.to_dict(),
        "scenario_5_webhook_ingestion": {"http_status": response.status_code, "body": response.text},
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
