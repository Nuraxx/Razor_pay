"""
Day-13 dashboard data layer.

Two distinct data sources, never mixed (brief section 5):

  SYNTHETIC BENCHMARK   -- the frozen offline evaluation reports Days 6-10
                           already computed and wrote to
                           evaluation/reports/*.json -- e.g. "what would
                           Fixed Retry vs. Model B vs. Oracle have scored on
                           the held-out test set." Read-only, never
                           recomputed here.
  OPERATIONAL DEMO DATA -- a live run of recovery/orchestrator.py (Day 12,
                           unmodified) over a sample of REAL generated
                           failure events (data/raw/failure_events.csv +
                           subscriptions.csv), using the real trained Day-8
                           Model B and the Day-11 mock LLM provider. This is
                           what populates the Recovery Queue, Event Detail,
                           Communications, and Audit Log pages.

Every loader here degrades gracefully (brief section 17): a missing
evaluation report or an empty database returns None / an empty DataFrame,
never raises, so the dashboard always launches and renders a clear empty
state instead of crashing.
"""
from __future__ import annotations

import json
import re

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.models import AuditLog, LLMInvocation, PolicyDecision, PromiseToPay
from classification.rules import classify
from llm.client import LLMClient, LLMProviderError
from model.candidate_preprocessing import PROJECT_ROOT
from policy.decision_engine import EVENT_FEATURE_KEYS
from recovery.orchestrator import RecoveryEventInput, orchestrate_recovery
from recovery.promise_service import record_customer_reply

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"

DEMO_SAMPLE_SIZE = 60  # events orchestrated to populate the "operational demo data" pages
DEMO_LANGUAGES = ["en", "hi", "hinglish"]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_inr(amount: float | None) -> str:
    """Indian-style comma grouping (e.g. 12,34,567.89), Rs-prefixed."""
    if amount is None or (isinstance(amount, float) and pd.isna(amount)):
        return "—"
    negative = amount < 0
    amount = abs(round(float(amount), 2))
    whole, _, frac = f"{amount:.2f}".partition(".")
    if len(whole) > 3:
        last3 = whole[-3:]
        rest = whole[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        whole = ",".join(groups + [last3])
    sign = "-" if negative else ""
    return f"{sign}₹{whole}.{frac}"


def format_ts(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return str(value)
    return ts.strftime("%d %b %Y, %H:%M")


def humanize_status(value: str | None) -> str:
    if not value:
        return "—"
    return str(value).replace("_", " ").title()


# ---------------------------------------------------------------------------
# SYNTHETIC BENCHMARK: raw generated data + frozen evaluation reports
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame | None:
    path = RAW_DATA_DIR / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_report(name: str) -> dict | None:
    path = REPORTS_DIR / name
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def list_available_reports() -> list[str]:
    if not REPORTS_DIR.exists():
        return []
    return sorted(p.name for p in REPORTS_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# OPERATIONAL DEMO DATA: live orchestrator run over real generated events
# ---------------------------------------------------------------------------

def _build_failure_context(event_row: pd.Series, sub_row: pd.Series) -> dict:
    merged = {**event_row.to_dict(), **sub_row.to_dict()}
    return {k: merged.get(k) for k in EVENT_FEATURE_KEYS}


@st.cache_resource(show_spinner="Running the recovery orchestrator over sample events...")
def build_demo_database(sample_size: int = DEMO_SAMPLE_SIZE):
    """Runs recovery/orchestrator.py::orchestrate_recovery (Day 12,
    unmodified) over a sample of REAL failure events, using the real
    trained Day-8 Model B if its artifact is present, otherwise falling
    back to policy's own model-unavailable path (still a valid, auditable
    outcome -- never a crash). Cached for the lifetime of the Streamlit
    process (st.cache_resource) so re-navigating pages doesn't re-run it.

    Returns (session, summary_df) -- summary_df is empty if no source data
    exists at all (brief section 17: the dashboard must still launch)."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()

    events = load_csv("failure_events.csv")
    subs = load_csv("subscriptions.csv")
    if events is None or subs is None or events.empty or subs.empty:
        return db, pd.DataFrame()

    model = _try_load_model()

    subs_by_id = subs.set_index("subscription_id")
    sample = events.head(sample_size).reset_index(drop=True)

    for i, row in sample.iterrows():
        sub_id = row["subscription_id"]
        if sub_id not in subs_by_id.index:
            continue
        sub_row = subs_by_id.loc[sub_id]
        failure_context = _build_failure_context(row, sub_row)
        failure_ts = pd.to_datetime(row["failure_timestamp"])

        event = RecoveryEventInput(
            event_id=row["event_id"],
            subscription_id=sub_id,
            failure_timestamp=failure_ts.to_pydatetime(),
            amount=float(row["amount"]),
            error_code=None,
            error_reason=row["error_reason"],
            failure_context=failure_context,
            customer_segment=str(sub_row.get("plan_tier", "unknown")),
            language=DEMO_LANGUAGES[i % len(DEMO_LANGUAGES)],
        )
        try:
            orchestrate_recovery(db, event, model=model)
        except Exception:
            # A single bad row must never take down the whole demo dataset.
            db.rollback()
            continue

    return db, _recovery_queue_df(db)


def _try_load_model():
    try:
        from model.train_latent_target_model import load_latent_target_model

        return load_latent_target_model("value")
    except Exception:
        return None  # orchestrator's own model-unavailable fallback path handles this safely


def _recovery_queue_df(db) -> pd.DataFrame:
    rows = db.query(PolicyDecision).order_by(PolicyDecision.id).all()
    records = []
    for r in rows:
        comm = (
            db.query(LLMInvocation)
            .filter(LLMInvocation.event_id == r.event_id, LLMInvocation.task_name == "outreach_microcopy")
            .order_by(LLMInvocation.id.desc())
            .first()
        )
        promise = (
            db.query(PromiseToPay)
            .filter(PromiseToPay.event_id == r.event_id, PromiseToPay.override_applied.is_(True))
            .order_by(PromiseToPay.id.desc())
            .first()
        )
        # FIX #3: a hard_decline event can have selected_candidate_type ==
        # NO_ACTION AND a real communication invocation (the payment-method-
        # update nudge) -- comm existing is what determines "sent"/
        # "fallback_used" regardless of NO_ACTION. comm being None for a
        # bucket that DOES request communication (hard_decline, or any
        # retryable_soft candidate) means compliance blocked it; comm being
        # None for customer_cancelled/unmapped means it was never attempted
        # at all -- those are the only two buckets compliance's own
        # opt-out-on-cancellation rule / "nothing truthful to say" reasoning
        # ever fully skip communication for.
        if comm is not None:
            communication_action = "sent" if comm.success else "fallback_used"
        elif r.classification_bucket in ("customer_cancelled", "unmapped"):
            communication_action = "skipped"
        else:
            communication_action = "blocked"

        records.append(
            {
                "event_id": r.event_id,
                "subscription_id": r.subscription_id,
                "classification_bucket": r.classification_bucket,
                "selected_candidate_type": r.selected_candidate_type,
                "selected_candidate_datetime": r.selected_candidate_datetime,
                "expected_recovery_value": r.expected_recovery_value,
                "decision_source": r.decision_source,
                "policy_version": r.policy_version,
                "decision_reason": r.decision_reason,
                "communication_action": communication_action,
                "llm_success": comm.success if comm is not None else None,
                "decided_at": r.decided_at,
                "promise_applied": promise is not None,
                "promise_id": promise.id if promise is not None else None,
                "promise_outcome": promise.override_outcome if promise is not None else None,
            }
        )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df["final_status"] = df.apply(_derive_final_status, axis=1)
    df["payment_action"] = df["selected_candidate_type"].apply(lambda t: "no_action" if t == "NO_ACTION" else "retry_scheduled")
    return df


def _derive_final_status(row: pd.Series) -> str:
    """Re-derives the SAME precedence recovery/schemas.py documents, purely
    for display in the queue table (the orchestrator already computed the
    authoritative per-event RecoveryExecutionResult -- this reconstructs
    its final_status from the persisted PolicyDecision/LLMInvocation rows
    for rows fetched back out of the DB, without re-running any decision
    logic). Matches orchestrator.py's own precedence exactly, including
    FIX #3's correction: NO_ACTION no longer short-circuits everything else
    -- a hard_decline event with a real communication result reports
    COMMUNICATION_ALLOWED / LLM_FALLBACK, not NO_ACTION."""
    if row["communication_action"] == "blocked":
        return "COMMUNICATION_BLOCKED"
    if row["selected_candidate_type"] == "NO_ACTION" and row["communication_action"] == "skipped":
        return "NO_ACTION"
    if row["decision_source"] == "rule_based_fallback":
        return "POLICY_FALLBACK"
    if row["communication_action"] == "fallback_used":
        return "LLM_FALLBACK"
    if row["communication_action"] == "sent":
        return "COMMUNICATION_ALLOWED"
    return "RETRY_ALLOWED"


def extract_llm_message_from_audit_rows(audit_rows) -> str | None:
    """Pulls `message_text` back out of the `llm` actor's own audit_log row
    for this run -- that row's `reason` already embeds
    `structured_output=<json>` verbatim (see llm/service.py::_persist), so
    this reads real persisted data rather than re-calling the LLM layer."""
    for row in audit_rows:
        if row.actor != "llm" or "structured_output=" not in (row.reason or ""):
            continue
        payload = row.reason.split("structured_output=", 1)[1]
        try:
            structured = json.loads(payload)
        except ValueError:
            continue
        if "message_text" in structured:
            return structured["message_text"]
    return None


def get_event_detail(db, event_id) -> dict | None:
    policy_row = db.query(PolicyDecision).filter(PolicyDecision.event_id == event_id).first()
    if policy_row is None:
        return None
    audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == event_id).order_by(AuditLog.id).all()
    llm_rows = db.query(LLMInvocation).filter(LLMInvocation.event_id == event_id).order_by(LLMInvocation.id).all()
    promise_rows = db.query(PromiseToPay).filter(PromiseToPay.event_id == event_id).order_by(PromiseToPay.id).all()
    return {"policy": policy_row, "audit": audit_rows, "llm": llm_rows, "promises": promise_rows}


def get_audit_log_df(db, actor: str | None = None, status: str | None = None) -> pd.DataFrame:
    query = db.query(AuditLog)
    if actor:
        query = query.filter(AuditLog.actor == actor)
    rows = query.order_by(AuditLog.id.desc()).all()
    records = [
        {
            "timestamp": r.created_at,
            "event_id": r.failure_event_id if r.failure_event_id is not None else r.raw_event_id,
            "actor": r.actor,
            "action": r.action,
            "reason": r.reason,
        }
        for r in rows
    ]
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df["status"] = df["action"].apply(_status_from_action)
    if status:
        df = df[df["status"] == status]
    return df


def _status_from_action(action: str) -> str:
    action = (action or "").lower()
    if "failed" in action or "blocked" in action or "no_action" in action:
        return "attention"
    return "ok"


def get_llm_invocations_df(db, task_name: str | None = None) -> pd.DataFrame:
    query = db.query(LLMInvocation)
    if task_name:
        query = query.filter(LLMInvocation.task_name == task_name)
    rows = query.order_by(LLMInvocation.id.desc()).all()
    records = [
        {
            "event_id": r.event_id,
            "batch_id": r.batch_id,
            "task_name": r.task_name,
            "provider": r.provider,
            "success": r.success,
            "structured_output": r.structured_output,
            "error_type": r.error_type,
            "created_at": r.created_at,
        }
        for r in rows
    ]
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Dynamic test count (brief section 13: "read dynamically... not
# manually hard-coded"). Counts `def test_*` functions across tests/*.py
# by static scan -- fast and safe on every page load. A "Run full test
# suite now" action (System / Demo page) additionally shells out to the
# REAL pytest run on demand, for a fully live number when the user wants one.
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def count_test_functions() -> int:
    tests_dir = PROJECT_ROOT / "tests"
    if not tests_dir.exists():
        return 0
    pattern = re.compile(r"^\s*def test_[A-Za-z0-9_]*\s*\(", re.MULTILINE)
    count = 0
    for path in tests_dir.glob("test_*.py"):
        try:
            text = path.read_text()
        except OSError:
            continue
        count += len(pattern.findall(text))
    return count


# ---------------------------------------------------------------------------
# Retry-candidate breakdown (brief section 9) -- SYNTHETIC BENCHMARK ONLY.
# Re-runs the pure policy/decision_engine_v4.py::decide_engine_v4 function
# (read-only, no DB write) purely to recover its per-candidate
# CandidateScore list for visualization -- the exact same computation that
# produced the live decision, not a new one.
# ---------------------------------------------------------------------------

def get_candidate_breakdown(event_row: pd.Series, sub_row: pd.Series, model) -> dict | None:
    from policy.baselines import rule_based_baseline
    from policy.decision_engine_v4 import decide_engine_v4

    failure_ts = pd.to_datetime(event_row["failure_timestamp"]).to_pydatetime()
    amount = float(event_row["amount"])
    bucket = classify(None, event_row["error_reason"]).bucket
    failure_context = _build_failure_context(event_row, sub_row)

    decision = decide_engine_v4(event_row["event_id"], event_row["subscription_id"], failure_ts, amount, bucket, failure_context, model=model)
    rule_pick = rule_based_baseline(event_row["event_id"], event_row["subscription_id"], failure_ts, amount, bucket, 0.0)["selected_candidate_type"]

    valid_types = {s.candidate_type for s in decision.candidate_scores if s.valid}
    oracle_pick = _oracle_pick(event_row["event_id"], amount, valid_types)

    candidates = [
        {
            "candidate_type": s.candidate_type,
            "candidate_datetime": s.candidate_datetime,
            "valid": s.valid,
            "predicted_recovery_value": s.predicted_recovery_value,
            "intervention_cost": s.intervention_cost,
            "expected_net_value": s.expected_net_value,
            "is_selected": s.candidate_type == decision.selected_candidate_type,
            "is_rule_pick": s.candidate_type == rule_pick,
            "is_oracle_pick": s.candidate_type == oracle_pick,
        }
        for s in decision.candidate_scores
    ]
    return {
        "candidates": candidates,
        "selected": decision.selected_candidate_type,
        "rule_pick": rule_pick,
        "oracle_pick": oracle_pick,
        "decision_source": decision.decision_source,
        "decision_reason": decision.decision_reason,
    }


def _oracle_pick(event_id: str, amount: float, valid_types: set[str]) -> str | None:
    """Oracle = argmax latent value AMONG VALID candidates only -- matching
    Day 9/10's own oracle definition (evaluation/evaluate_decision_engine_v4.py),
    not merely the argmax over every row in counterfactual_outcomes.csv."""
    outcomes = load_csv("counterfactual_outcomes.csv")
    if outcomes is None or not valid_types:
        return None
    subset = outcomes[(outcomes["event_id"] == event_id) & (outcomes["candidate_type"].isin(valid_types))].copy()
    if subset.empty:
        return None
    subset["latent_value"] = subset["recovery_probability_latent"] * amount
    return subset.loc[subset["latent_value"].idxmax(), "candidate_type"]


# ---------------------------------------------------------------------------
# Interactive demo scenarios (brief section 14) -- reuses
# recovery/orchestrator.py directly, no reimplemented decision logic.
# ---------------------------------------------------------------------------

_DEMO_FAILURE_TS_STR = "2026-02-24 10:00:00"
_DEMO_FAILURE_CONTEXT = {
    "day_of_month": 24, "days_to_nearest_payday_window": 6, "prior_if_failure_count": 0,
    "prior_if_self_resolved_rate": None, "tenure_days": 200, "plan_tier": "mid",
    "primary_instrument": "upi_autopay", "city_tier": "tier_1", "bank_network_conditions": "good",
    "issuing_bank_downtime_flag": False, "network_latency_bucket": "low", "is_month_end_settlement_rush": False,
}


class _AlwaysFailsLLMClient(LLMClient):
    model_name = "demo-simulated-outage"
    provider_name = "mock"

    def complete(self, system_prompt, user_prompt, *, max_tokens=512):
        raise LLMProviderError("simulated_provider_outage")


DEMO_SCENARIOS = {
    "Recoverable insufficient-fund payment": dict(error_reason="insufficient_fund", amount=2999.0, customer_opted_out=False, force_llm_failure=False, customer_reply_text=None),
    "Hard decline (payment-method-update nudge)": dict(error_reason="card_expired", amount=499.0, customer_opted_out=False, force_llm_failure=False, customer_reply_text=None),
    "Customer opt-out": dict(error_reason="insufficient_fund", amount=799.0, customer_opted_out=True, force_llm_failure=False, customer_reply_text=None),
    "LLM failure": dict(error_reason="insufficient_fund", amount=2999.0, customer_opted_out=False, force_llm_failure=True, customer_reply_text=None),
    "Promise-to-pay override": dict(error_reason="insufficient_fund", amount=2999.0, customer_opted_out=False, force_llm_failure=False, customer_reply_text="I'll pay Friday when salary comes"),
}


def run_demo_scenario(scenario_name: str, model):
    import datetime as _dt

    cfg = DEMO_SCENARIOS[scenario_name]
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()

    failure_ts = _dt.datetime.strptime(_DEMO_FAILURE_TS_STR, "%Y-%m-%d %H:%M:%S")
    event = RecoveryEventInput(
        event_id=1, subscription_id="sub_ui_demo", failure_timestamp=failure_ts,
        amount=cfg["amount"], error_code=None, error_reason=cfg["error_reason"], failure_context=_DEMO_FAILURE_CONTEXT,
        customer_segment="mid", language="en", customer_opted_out=cfg["customer_opted_out"],
    )
    # FIX #1: a customer reply must be recorded (parsed, validated, persisted)
    # BEFORE orchestrate_recovery's one and only decision for this event --
    # see recovery/orchestrator.py's module docstring for why the override
    # is evaluated once, at decision time, not as a retroactive re-decision.
    if cfg["customer_reply_text"]:
        record_customer_reply(
            db, event_id=1, subscription_id="sub_ui_demo",
            customer_reply_text=cfg["customer_reply_text"], today=failure_ts.date(),
        )
    llm_client = _AlwaysFailsLLMClient() if cfg["force_llm_failure"] else None
    result = orchestrate_recovery(db, event, model=model, llm_client=llm_client)
    audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == 1).order_by(AuditLog.id).all()
    promise_rows = db.query(PromiseToPay).filter(PromiseToPay.event_id == 1).order_by(PromiseToPay.id).all()
    db.close()
    return result, audit_rows, promise_rows


def run_full_test_suite() -> tuple[int, str]:
    """Shells out to the real pytest run (opt-in, triggered by a button --
    never on automatic page load, since it takes ~40s). Returns
    (returncode, captured tail of output)."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(PROJECT_ROOT / "tests"), "-q"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=300,
    )
    tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-15:])
    return result.returncode, tail
