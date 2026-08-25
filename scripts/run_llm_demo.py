"""
Day-11 demo CLI: shows the full flow --

    failure_event -> classification -> Day-8 Model B / policy-v4
    -> selected recovery action -> LLM communication tasks
    -> structured outputs -> audit log

end to end, offline, in mock mode by default (no ANTHROPIC_API_KEY needed).

Usage (from the project root):

    ./venv/bin/python scripts/run_llm_demo.py

Uses a throwaway in-memory SQLite database (never touches data/recovery_agent.db)
so it's safe to run repeatedly with no side effects on real project data.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

# Documented usage is `./venv/bin/python scripts/run_llm_demo.py` -- running a
# script directly (not via `-m`) only puts scripts/ on sys.path, not the
# project root, so the `app`/`llm`/`model` package imports below would
# otherwise fail with ModuleNotFoundError regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.db import Base
from app.models import AuditLog, LLMInvocation
from classification.rules import classify
from llm.service import (
    generate_batch_explanation_and_log,
    generate_outreach_microcopy_and_log,
    parse_promise_to_pay_and_log,
)
from model.latent_target_preprocessing import PROJECT_ROOT
from model.train_latent_target_model import load_latent_target_model
from policy.decision_engine_v4 import NO_ACTION, decide_engine_v4

DEMO_EVENT = {
    "event_id": 900001,
    "subscription_id": "sub_DemoLLM001",
    "failure_timestamp": datetime(2026, 8, 24, 9, 0, 0),
    "amount": 799.0,
    "error_reason": "insufficient_fund",
    "customer_segment": "mid",  # plan_tier, reused as "customer segment" per Job 1's spec
    "failure_context": {
        "day_of_month": 24, "days_to_nearest_payday_window": 6, "prior_if_failure_count": 0,
        "prior_if_self_resolved_rate": float("nan"), "tenure_days": 200, "plan_tier": "mid",
        "primary_instrument": "upi_autopay", "city_tier": "tier_1", "bank_network_conditions": "good",
        "issuing_bank_downtime_flag": False, "network_latency_bucket": "low", "is_month_end_settlement_rush": False,
    },
}

_WINDOW_DESCRIPTIONS = {
    "immediate": "within the hour",
    "plus_1_day_morning": "tomorrow morning",
    "payday_window": "around your next payday",
    "plus_3_days": "in a few days",
    "month_end_window": "around month-end",
}


def _demo_db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)()


def _print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    print(f"LLM_PROVIDER = {settings.LLM_PROVIDER!r} (mock = fully offline, no API key needed)")

    _print_header("1. Input recovery event")
    print(json.dumps({k: v for k, v in DEMO_EVENT.items() if k != "failure_context"}, indent=2, default=str))

    _print_header("2. Classification (deterministic, Day 2)")
    classification = classify(None, DEMO_EVENT["error_reason"])
    print(f"classification_bucket = {classification.bucket!r} (confidence={classification.confidence})")

    _print_header("3. Selected policy action (Day-8 Model B + policy-v4, Day 10 -- LLM NOT involved)")
    model = load_latent_target_model("value")
    decision = decide_engine_v4(
        DEMO_EVENT["event_id"], DEMO_EVENT["subscription_id"], DEMO_EVENT["failure_timestamp"], DEMO_EVENT["amount"],
        classification.bucket, DEMO_EVENT["failure_context"], model=model,
    )
    print(f"selected_candidate_type = {decision.selected_candidate_type!r}")
    print(f"decision_source         = {decision.decision_source!r}")
    print(f"policy_version          = {decision.policy_version!r}")
    print(f"predicted_recovery_value= {decision.predicted_recovery_value}")
    print(f"decision_reason         = {decision.decision_reason}")

    will_retry = decision.selected_candidate_type != NO_ACTION
    window_description = _WINDOW_DESCRIPTIONS.get(decision.selected_candidate_type) if will_retry else None

    db = _demo_db_session()

    _print_header("4. LLM Job 1 -- outreach microcopy (per failure bucket x customer segment x language)")
    for language in ("en", "hinglish"):
        result, invocation = generate_outreach_microcopy_and_log(
            db, event_id=DEMO_EVENT["event_id"], failure_bucket=classification.bucket,
            customer_segment=DEMO_EVENT["customer_segment"], language=language, will_retry=will_retry,
            retry_window_description=window_description, amount_rupees=DEMO_EVENT["amount"],
        )
        print(f"  [{language}] provider={result.provider} success={result.success} error_type={result.error_type}")
        print(f"  [{language}] message_text: {result.structured_result['message_text']}")
        print(f"  [{language}] llm_invocations.id={invocation.id}")
        print()

    _print_header("5. LLM Job 2 -- promise-to-pay parsing")
    sample_reply = "I'll pay Friday when salary comes, via UPI"
    result2, invocation2 = parse_promise_to_pay_and_log(
        db, event_id=DEMO_EVENT["event_id"], customer_reply_text=sample_reply, today=date(2026, 8, 24),
    )
    print(f"  customer reply: {sample_reply!r}")
    print(f"  provider={result2.provider} success={result2.success} error_type={result2.error_type}")
    print(f"  structured_result: {result2.structured_result}")
    print(f"  llm_invocations.id={invocation2.id}")

    _print_header("6. LLM Job 2 -- failure-handling example (forced malformed provider)")
    from llm.client import LLMClient

    class _BrokenClient(LLMClient):
        model_name = "broken-demo-client"
        provider_name = "mock"

        def complete(self, system_prompt, user_prompt, *, max_tokens=512):
            return "not valid json at all"

    result2_fail, invocation2_fail = parse_promise_to_pay_and_log(
        db, event_id=DEMO_EVENT["event_id"], customer_reply_text=sample_reply, today=date(2026, 8, 24), client=_BrokenClient(),
    )
    print(f"  (simulated broken provider) success={result2_fail.success} error_type={result2_fail.error_type}")
    print(f"  deterministic fallback used: {result2_fail.structured_result}")
    print(f"  llm_invocations.id={invocation2_fail.id}")

    _print_header("7. LLM Job 3 -- plain-English batch-level explanation")
    report_path = PROJECT_ROOT / "evaluation" / "reports" / "decision_engine_v4_evaluation.json"
    if report_path.exists():
        with open(report_path) as f:
            report_summary = json.load(f)
        print(f"  (using real Day-10 evaluation report: {report_path})")
    else:
        report_summary = {"label": "SYNTHETIC COUNTERFACTUAL EVALUATION -- demo placeholder, no report file found on disk"}
        print("  (no Day-10 report found on disk -- using a placeholder summary)")

    result3, invocation3 = generate_batch_explanation_and_log(
        db, batch_id="day10_test_set_evaluation_demo", report_summary=report_summary,
    )
    print(f"  provider={result3.provider} success={result3.success} error_type={result3.error_type}")
    print(f"  explanation_text: {result3.structured_result['explanation_text']}")
    print(f"  llm_invocations.id={invocation3.id}")

    _print_header("8. Audit records written during this demo run")
    invocations = db.query(LLMInvocation).all()
    print(f"llm_invocations rows: {len(invocations)}")
    for inv in invocations:
        print(f"  id={inv.id} task={inv.task_name} provider={inv.provider} success={inv.success} error_type={inv.error_type}")

    audit_rows = db.query(AuditLog).filter(AuditLog.actor == "llm").all()
    print(f"\naudit_log rows (actor=llm): {len(audit_rows)}")
    for row in audit_rows:
        print(f"  id={row.id} action={row.action}")

    _print_header("9. Confirmation: policy decision is independent of LLM success")
    print("The selected_candidate_type/decision_source printed in section 3 above was")
    print("computed BEFORE any LLM call in this script and is never read back or")
    print("modified by any of the LLM job functions in llm/service.py -- they only")
    print("accept it as a plain data argument (will_retry / retry_window_description).")
    print(f"selected_candidate_type (unchanged) = {decision.selected_candidate_type!r}")

    db.close()


if __name__ == "__main__":
    main()
