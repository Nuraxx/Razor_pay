"""
Day-13/14 dashboard data layer.

THREE distinct data sources, never mixed:

  SYNTHETIC BENCHMARK   -- the frozen offline evaluation reports Days 6-10
                           already computed and wrote to
                           evaluation/reports/*.json -- e.g. "what would
                           Fixed Retry vs. Model B vs. Oracle have scored on
                           the held-out test set." Read-only, never
                           recomputed here. STATIC data/raw/*.csv (the
                           synthetic dataset those reports were computed
                           from) is the same category.
  DEMO-GENERATED         -- a live run of recovery/orchestrator.py (Day 12,
                           unmodified) over a sample of REAL generated
                           failure events (data/raw/failure_events.csv +
                           subscriptions.csv), written to a THROWAWAY
                           in-memory SQLite DB, using the real trained
                           Day-8 Model B. The LLM call is NOT mocked --
                           it goes through the same get_llm_client() every
                           live webhook uses, i.e. whatever settings.LLM_PROVIDER
                           is currently configured to (mock/anthropic/gemini/ollama).
                           Only the events/database are throwaway/synthetic;
                           the LLM request itself is real. Powers the
                           "System / Demo" page's interactive scenario
                           runner only -- never labeled "live".
  LIVE DATABASE          -- read-only queries against the REAL,
                           Razorpay-webhook-backed SQLite file at
                           settings.DATABASE_URL (see the "LIVE DATABASE"
                           section below) -- the exact file
                           app/main.py's /webhook/razorpay handler writes
                           to. Powers Overview's live KPIs/recent-events,
                           Recovery Queue, Payment Events, Communications,
                           Audit Log, and the System page's status block.

Every loader here degrades gracefully (brief section 17): a missing
evaluation report or an empty/unreachable database returns None / an empty
DataFrame / an explicit status flag, never raises, so the dashboard always
launches and renders a clear empty state instead of crashing.
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
from app.models import AuditLog, FailureEvent, LLMInvocation, PolicyDecision, PromiseToPay, RawEvent, RevenueRiskEvent
from classification.rules import classify
from llm.client import LLMClient, LLMProviderError, get_llm_client
from model.candidate_preprocessing import PROJECT_ROOT
from model.unified_model import get_live_unified_model
from policy.decision_engine import EVENT_FEATURE_KEYS
from recovery.orchestrator import RecoveryEventInput, orchestrate_recovery
from recovery.promise_service import record_customer_reply
from ui.utils import format_inr, format_ts, humanize_status  # noqa: F401 -- re-exported: `from ui.data import format_inr, humanize_status` and `data.format_ts` stay valid

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"

DEMO_SAMPLE_SIZE = 60  # events orchestrated to populate the "operational demo data" pages
DEMO_LANGUAGES = ["en", "hi", "hinglish"]


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


@st.cache_resource(show_spinner=False)
def _try_load_model():
    """Cached (Part 21: a live 5-second refresh must never repeatedly pay
    model-deserialization cost) -- orchestrate_recovery's own
    model-unavailable fallback path handles a None model safely either way."""
    try:
        from model.train_latent_target_model import load_latent_target_model

        return load_latent_target_model("value")
    except Exception:
        return None


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


# ---------------------------------------------------------------------------
# LIVE DATABASE: read-only queries against the REAL, Razorpay-webhook-backed
# SQLite file at settings.DATABASE_URL (app/db.py's SessionLocal) -- the
# exact file app/main.py's /webhook/razorpay handler writes to. This is NOT
# the same database as "OPERATIONAL DEMO DATA" above (a throwaway :memory:
# DB populated by re-running the orchestrator over sample CSV rows) -- the
# two are never mixed, and no function below ever falls back to demo data.
#
# CACHING (brief section 13): nothing here uses @st.cache_data /
# @st.cache_resource. Each call opens a fresh session, queries, and closes
# it -- the 5-second st.fragment(run_every=...) refresh in ui/app.py is the
# freshness mechanism; caching on top of that would just reintroduce the
# staleness it exists to avoid.
# ---------------------------------------------------------------------------

def get_live_session():
    """A fresh session against the real file-backed DB. Never reused across
    calls -- SQLite reads current committed rows per-connection, which is
    what makes writes from the FastAPI process (a separate OS process)
    visible here."""
    from app.db import SessionLocal

    return SessionLocal()


def get_live_system_status() -> dict:
    """Best-effort, NEVER-raising snapshot of actual runtime state. Every
    field is either a real check performed this call or explicitly
    None/False -- this never reports a component healthy without having
    just verified it (brief: "Do not claim healthy when the code cannot
    actually verify it")."""
    from app.config import settings

    # llm_provider is the raw config value; llm_active_provider/llm_active_model
    # reflect what get_llm_client() will ACTUALLY hand back this call -- these
    # can differ from config if e.g. LLM_PROVIDER=anthropic but no API key is
    # set, in which case get_llm_client() silently falls back to mock. Never
    # let the UI claim a configured provider is active without checking.
    try:
        _live_llm_client = get_llm_client()
        llm_active_provider = _live_llm_client.provider_name
        llm_active_model = _live_llm_client.model_name
    except Exception:  # noqa: BLE001 -- a status probe must never crash the dashboard
        llm_active_provider = None
        llm_active_model = None

    status = {
        "environment": settings.RAZORPAY_ENV,
        "database_connected": False,
        "database_error": None,
        "fastapi_connected": False,
        "fastapi_error": None,
        "webhook_secret_configured": bool(settings.RAZORPAY_WEBHOOK_SECRET),
        "llm_provider": settings.LLM_PROVIDER,
        "llm_active_provider": llm_active_provider,
        "llm_active_model": llm_active_model,
        "model_loaded": _try_load_model() is not None,
        "unified_model_loaded": get_live_unified_model() is not None,
        "last_event_received": None,
        "last_successful_processing": None,
    }

    try:
        db = get_live_session()
        try:
            last_event = db.query(RawEvent).order_by(RawEvent.id.desc()).limit(1).one_or_none()
            status["last_event_received"] = last_event.received_at if last_event else None
            last_final = (
                db.query(AuditLog)
                .filter(AuditLog.action == "orchestrator_final_status")
                .order_by(AuditLog.id.desc())
                .limit(1)
                .one_or_none()
            )
            status["last_successful_processing"] = last_final.created_at if last_final else None
            status["database_connected"] = True
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001 -- a status probe must never crash the dashboard
        status["database_error"] = str(exc)

    try:
        import httpx

        resp = httpx.get(f"http://127.0.0.1:{settings.APP_PORT}/health", timeout=1.5)
        status["fastapi_connected"] = resp.status_code == 200
    except Exception as exc:  # noqa: BLE001 -- FastAPI simply not running is a normal, displayable state
        status["fastapi_error"] = type(exc).__name__

    return status


def _get_orchestrated_raw_event_ids(db) -> set[int]:
    """Raw events that reached SOME real processing path. There are TWO
    (recovery/webhook_pipeline.py::process_raw_event): a subscription-linked
    payment.failed creates a FailureEvent; a no-subscription one-time
    payment.failed with enough authoritative context (payment_id + amount)
    creates a RevenueRiskEvent(event_type="payment_failed_no_subscription")
    instead, matched back to its originating raw_event by payment_id (the
    same idempotency key recovery/webhook_pipeline.py::_build_one_time_payment_event
    uses). Before this fix, only the FailureEvent half was counted here,
    which meant every successfully-orchestrated Payment Link event was
    misreported as "received, not orchestrated" -- a real dashboard bug,
    not a display nuance. A payment.captured event is always attempted by
    confirm_payment_recovery (no eligibility gate), so it's never
    "unrouted" either."""
    classified_ids = {row[0] for row in db.query(FailureEvent.raw_event_id).all()}
    routed_payment_ids = {
        row[0] for row in db.query(RevenueRiskEvent.external_id)
        .filter(RevenueRiskEvent.event_type == "payment_failed_no_subscription").all()
        if row[0] is not None
    }
    if routed_payment_ids:
        classified_ids |= {
            row[0] for row in db.query(RawEvent.id).filter(RawEvent.payment_id.in_(routed_payment_ids)).all()
        }
    classified_ids |= {row[0] for row in db.query(RawEvent.id).filter(RawEvent.event_type == "payment.captured").all()}
    return classified_ids


def get_live_kpis() -> dict:
    """Counts derived ONLY from what's actually persisted. Deliberately has
    NO recovery-rate / recovered-value field of its own -- a scheduled retry
    is not a confirmed recovery, and fabricating that number would violate
    the brief's "never fake live data." A real, authoritative
    payment.captured confirmation signal now exists (see
    recovery/payment_reconciliation.py) and is reflected in
    get_live_recovery_outcomes_df()/the Revenue-at-Risk page's Recovery
    Outcomes section -- this Overview page's own KPI set intentionally
    stays scoped to what it already showed, rather than growing a second,
    overlapping recovery-rate figure here. Recovery rate on THIS page stays
    exclusively a SYNTHETIC BENCHMARK metric (see the Analytics page)."""
    db = get_live_session()
    try:
        total_raw_events = db.query(RawEvent).count()
        # NOTE: this already counts PolicyDecision rows across ALL domains
        # (revenue-risk decisions use the same table, offset-keyed -- see
        # policy/policy_decision_store.py) -- it was never actually
        # "Subscriptions-only scope" despite that label; see ui/app.py's
        # kpi_card call site, now corrected to say so honestly.
        total_decisions = db.query(PolicyDecision).count()
        retry_actions = db.query(PolicyDecision).filter(PolicyDecision.selected_candidate_type != "NO_ACTION").count()
        all_raw_ids = {row[0] for row in db.query(RawEvent.id).all()}
        orchestrated_raw_ids = _get_orchestrated_raw_event_ids(db)
        return {
            "failed_payments": total_raw_events,
            "policy_decisions": total_decisions,
            "retry_actions": retry_actions,
            "no_action": total_decisions - retry_actions,
            "received_not_orchestrated": len(all_raw_ids - orchestrated_raw_ids),
        }
    finally:
        db.close()


def get_live_raw_events_df(limit: int = 200) -> pd.DataFrame:
    db = get_live_session()
    try:
        rows = db.query(RawEvent).order_by(RawEvent.id.desc()).limit(limit).all()
        return pd.DataFrame.from_records(
            [
                {
                    "id": r.id,
                    "received_at": r.received_at,
                    "razorpay_event_id": r.razorpay_event_id,
                    "event_type": r.event_type,
                    "payment_id": r.payment_id,
                    "order_id": r.order_id,
                    "subscription_id": r.subscription_id,
                    "amount_rupees": (r.amount or 0) / 100.0,
                    "currency": r.currency,
                    "error_code": r.error_code,
                    "error_reason": r.error_reason,
                    "error_source": r.error_source,
                    "error_step": r.error_step,
                    "signature_verified": r.signature_verified,
                    "raw_payload": r.raw_payload,
                }
                for r in rows
            ]
        )
    finally:
        db.close()


def get_live_recovery_queue_df(limit: int = 200) -> pd.DataFrame:
    """Reuses _derive_final_status -- the SAME precedence logic the demo
    path above already established (brief: never duplicate decision
    logic). The only real difference from the demo path is that
    amount/error_reason/payment_id come from an actual join to
    raw_events/failure_events instead of a CSV lookup: a live
    PolicyDecision.event_id is a real failure_events.id, not a CSV row
    index the demo path repurposes as one."""
    db = get_live_session()
    try:
        rows = (
            db.query(PolicyDecision, RawEvent)
            .join(FailureEvent, FailureEvent.id == PolicyDecision.event_id)
            .join(RawEvent, RawEvent.id == FailureEvent.raw_event_id)
            .order_by(PolicyDecision.id.desc())
            .limit(limit)
            .all()
        )
        records = []
        for policy_row, raw_row in rows:
            comm = (
                db.query(LLMInvocation)
                .filter(LLMInvocation.event_id == policy_row.event_id, LLMInvocation.task_name == "outreach_microcopy")
                .order_by(LLMInvocation.id.desc())
                .first()
            )
            promise = (
                db.query(PromiseToPay)
                .filter(PromiseToPay.event_id == policy_row.event_id, PromiseToPay.override_applied.is_(True))
                .order_by(PromiseToPay.id.desc())
                .first()
            )
            if comm is not None:
                communication_action = "sent" if comm.success else "fallback_used"
            elif policy_row.classification_bucket in ("customer_cancelled", "unmapped"):
                communication_action = "skipped"
            else:
                communication_action = "blocked"

            record = {
                "event_id": policy_row.event_id,
                "raw_event_id": raw_row.id,
                "payment_id": raw_row.payment_id,
                "subscription_id": policy_row.subscription_id,
                "amount_rupees": (raw_row.amount or 0) / 100.0,
                "error_reason": raw_row.error_reason,
                "classification_bucket": policy_row.classification_bucket,
                "selected_candidate_type": policy_row.selected_candidate_type,
                "selected_candidate_datetime": policy_row.selected_candidate_datetime,
                "decision_source": policy_row.decision_source,
                "policy_version": policy_row.policy_version,
                "decision_reason": policy_row.decision_reason,
                "communication_action": communication_action,
                "llm_success": comm.success if comm is not None else None,
                "decided_at": policy_row.decided_at,
                "promise_applied": promise is not None,
            }
            record["final_status"] = _derive_final_status(pd.Series(record))
            record["payment_action"] = "no_action" if policy_row.selected_candidate_type == "NO_ACTION" else "retry_scheduled"
            records.append(record)
        return pd.DataFrame.from_records(records)
    finally:
        db.close()


def get_live_event_detail(event_id) -> dict | None:
    """Same shape get_event_detail (demo path) already returns, plus the
    raw_events/failure_events rows a live event actually needs (the demo
    path's event_id IS the CSV row, so it never needed this join).

    `event_id` is cast to a plain `int` up front -- a value read back out of
    a pandas DataFrame cell (e.g. a `st.dataframe` row-selection callback, or
    a table built by get_live_recovery_queue_df) is a `numpy.int64`, which
    SQLite's DBAPI adapter does not bind the same way as a native `int` --
    the `==` filter below silently matches zero rows instead of raising,
    so this returned None for every numpy-typed caller until this cast."""
    db = get_live_session()
    try:
        event_id = int(event_id)
        policy_row = db.query(PolicyDecision).filter(PolicyDecision.event_id == event_id).first()
        if policy_row is None:
            return None
        failure_row = db.query(FailureEvent).filter(FailureEvent.id == event_id).first()
        raw_row = db.query(RawEvent).filter(RawEvent.id == failure_row.raw_event_id).first() if failure_row else None
        audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == event_id).order_by(AuditLog.id).all()
        llm_rows = db.query(LLMInvocation).filter(LLMInvocation.event_id == event_id).order_by(LLMInvocation.id).all()
        promise_rows = db.query(PromiseToPay).filter(PromiseToPay.event_id == event_id).order_by(PromiseToPay.id).all()
        return {
            "policy": policy_row, "failure": failure_row, "raw": raw_row,
            "audit": audit_rows, "llm": llm_rows, "promises": promise_rows,
        }
    finally:
        db.close()


def get_live_unrouted_raw_events_df(limit: int = 50) -> pd.DataFrame:
    """Raw events that were stored but genuinely never reached ANY
    processing path -- e.g. a payment.failed with no subscription_id AND
    no payment_id/amount to act on (recovery/webhook_pipeline.py's
    OUTCOME_SKIPPED_INSUFFICIENT_CONTEXT), or an unsupported event_type.
    A no-subscription Payment Link WITH enough context (payment_id +
    amount) is NOT unrouted -- it reaches the one-time-payment
    RevenueRiskEvent path (see _get_orchestrated_raw_event_ids) and is
    correctly excluded from this table. Surfaced explicitly rather than
    silently vanishing from every other live table -- accurately
    representing actual backend state instead of implying a policy
    decision happened when it didn't."""
    db = get_live_session()
    try:
        orchestrated_raw_ids = _get_orchestrated_raw_event_ids(db)
        rows = db.query(RawEvent).order_by(RawEvent.id.desc()).limit(limit * 4).all()
        unrouted = [r for r in rows if r.id not in orchestrated_raw_ids][:limit]

        def _reason(r: RawEvent) -> str:
            if r.event_type not in ("payment.failed", "payment.captured"):
                return "unsupported event_type"
            if not r.subscription_id and (not r.payment_id or r.amount is None):
                return "insufficient context -- no subscription_id and no payment_id/amount to act on"
            return "not yet processed"

        return pd.DataFrame.from_records(
            [
                {
                    "id": r.id,
                    "received_at": r.received_at,
                    "event_type": r.event_type,
                    "payment_id": r.payment_id,
                    "subscription_id": r.subscription_id,
                    "amount_rupees": (r.amount or 0) / 100.0,
                    "error_reason": r.error_reason,
                    "reason_not_orchestrated": _reason(r),
                }
                for r in unrouted
            ]
        )
    finally:
        db.close()


def get_live_communications_df(limit: int = 200) -> pd.DataFrame:
    """LLM invocations from the real DB, PLUS compliance-blocked
    communications -- those never reach llm/service.py at all (see
    recovery/orchestrator.py's communication_action="blocked" branch), so
    without this they'd silently vanish rather than show their real
    compliance reason. Retry-window text is DERIVED (never invented) via
    the same recovery/orchestrator.py::describe_retry_window the
    orchestrator itself used, joined off the correlated PolicyDecision."""
    from recovery.orchestrator import describe_retry_window

    db = get_live_session()
    try:
        llm_rows = (
            db.query(LLMInvocation)
            .filter(LLMInvocation.task_name == "outreach_microcopy")
            .order_by(LLMInvocation.id.desc())
            .limit(limit)
            .all()
        )
        policy_by_event = {p.event_id: p for p in db.query(PolicyDecision).all()}
        records = []
        for r in llm_rows:
            try:
                structured = json.loads(r.structured_output or "{}")
            except ValueError:
                structured = {}
            policy_row = policy_by_event.get(r.event_id)
            records.append(
                {
                    "created_at": r.created_at,
                    "event_id": r.event_id,
                    "task_name": r.task_name,
                    "language": structured.get("language"),
                    "customer_segment": structured.get("customer_segment"),
                    "retry_window": describe_retry_window(policy_row.selected_candidate_type) if policy_row else None,
                    "provider": r.provider,
                    "success": r.success,
                    "status": "sent" if r.success else "fallback_used",
                    "message_text": structured.get("message_text"),
                }
            )

        blocked_audit_rows = (
            db.query(AuditLog)
            .filter(AuditLog.action == "orchestrator_compliance", AuditLog.reason.like("%communication_action_allowed=False%"))
            .order_by(AuditLog.id.desc())
            .limit(limit)
            .all()
        )
        for a in blocked_audit_rows:
            policy_row = policy_by_event.get(a.failure_event_id)
            comm_reason = extract_compliance_fields(a.reason).get("communication_reason")
            records.append(
                {
                    "created_at": a.created_at,
                    "event_id": a.failure_event_id,
                    "task_name": "outreach_microcopy",
                    "language": None,
                    "customer_segment": None,
                    "retry_window": describe_retry_window(policy_row.selected_candidate_type) if policy_row else None,
                    "provider": None,
                    "success": None,
                    "status": "blocked",
                    "message_text": comm_reason,
                }
            )

        df = pd.DataFrame.from_records(records)
        if not df.empty:
            df = df.sort_values("created_at", ascending=False).reset_index(drop=True)
        return df
    finally:
        db.close()


# Razorpay's own documented ID format: `<entity>_<14+ alphanumeric chars>`,
# no further underscores, no human-readable words. This project has no
# synthetic/live flag column on raw_events, so this heuristic is the only
# way to tell a genuine Razorpay Test Mode delivery apart from a
# locally-triggered test/demo webhook call for Overview's live pipeline
# summary. Every synthetic/test ID used anywhere in this codebase or its
# verification scripts (e.g. "evt_OllamaLiveTest_1787723453",
# "sub_DemoLLM001") contains an extra underscore or a human-readable suffix
# and fails this pattern -- conservative in the safe direction: an ID is
# only ever labeled "live" when it plausibly IS Razorpay's own format;
# nothing synthetic is ever mislabeled as real.
_RAZORPAY_ID_RE = re.compile(r"^[a-z]+_[A-Za-z0-9]{14,}$")


def _looks_like_real_razorpay_id(value: str | None) -> bool:
    return bool(value) and bool(_RAZORPAY_ID_RE.match(value))


def get_live_llm_summary() -> dict:
    """Overview's "LLM CALLS" KPI card -- total llm_invocations count plus
    the single most recent invocation's provider/model/task/success, read
    directly off llm_invocations every call. Never hardcoded and never
    inferred from the currently-configured LLM_PROVIDER -- a provider can be
    configured but never yet actually called, or its most recent real call
    could have failed, and only a real row here can say which."""
    db = get_live_session()
    try:
        total = db.query(LLMInvocation).count()
        latest = db.query(LLMInvocation).order_by(LLMInvocation.id.desc()).first()
        return {
            "total_invocations": total,
            "latest_provider": latest.provider if latest else None,
            "latest_model": latest.model_name if latest else None,
            "latest_task": latest.task_name if latest else None,
            "latest_success": latest.success if latest else None,
        }
    finally:
        db.close()


def get_live_pipeline_snapshot() -> dict | None:
    """Overview's "Latest recovery event" pipeline card -- the single most
    recent live PolicyDecision (payment_failed path, the same scope
    get_live_recovery_queue_df already covers for the Recovery Queue page),
    re-using that function plus get_live_event_detail rather than
    duplicating any SQL. Every stage below is read straight off the real
    raw_events / audit_log / llm_invocations rows for that one event --
    never a fabricated or assumed stage. Returns None only when no live
    policy decision exists yet at all (a normal, displayable empty state)."""
    queue_df = get_live_recovery_queue_df(limit=1)
    if queue_df.empty:
        return None
    latest = queue_df.iloc[0]
    event_id = latest["event_id"]

    detail = get_live_event_detail(event_id)
    if detail is None:
        return None
    raw_row, audit_rows, llm_rows = detail["raw"], detail["audit"], detail["llm"]

    compliance_audit = next((a for a in reversed(audit_rows) if a.action == "orchestrator_compliance"), None)
    compliance_fields = extract_compliance_fields(compliance_audit.reason if compliance_audit else None)
    payment_allowed = compliance_fields.get("payment_allowed")
    communication_allowed = compliance_fields.get("communication_allowed")
    if payment_allowed is None or communication_allowed is None:
        compliance_display = None
    elif payment_allowed == "True" and communication_allowed == "True":
        compliance_display = "ALLOWED"
    elif payment_allowed == "False" and communication_allowed == "False":
        compliance_display = "BLOCKED"
    else:
        compliance_display = (
            f"PARTIAL (payment {'allowed' if payment_allowed == 'True' else 'blocked'}, "
            f"communication {'allowed' if communication_allowed == 'True' else 'blocked'})"
        )

    comm = next((r for r in reversed(llm_rows) if r.task_name == "outreach_microcopy"), None)
    message_text = None
    if comm is not None:
        try:
            message_text = json.loads(comm.structured_output or "{}").get("message_text")
        except ValueError:
            message_text = None

    return {
        "event_id": event_id,
        "razorpay_event_id": raw_row.razorpay_event_id if raw_row else None,
        "is_live_razorpay_id": _looks_like_real_razorpay_id(raw_row.razorpay_event_id if raw_row else None),
        "payment_id": raw_row.payment_id if raw_row else None,
        "received_at": raw_row.received_at if raw_row else None,
        "error_reason": None if pd.isna(latest["error_reason"]) else latest["error_reason"],
        "amount_rupees": latest["amount_rupees"],
        "classification_bucket": latest["classification_bucket"],
        "selected_candidate_type": latest["selected_candidate_type"],
        "decision_source": latest["decision_source"],
        "compliance_display": compliance_display,
        "llm_provider": comm.provider if comm is not None else None,
        "llm_model": comm.model_name if comm is not None else None,
        "llm_task": comm.task_name if comm is not None else None,
        "llm_success": comm.success if comm is not None else None,
        "communication_action": latest["communication_action"],
        "communication_message": message_text,
        "final_status": latest["final_status"],
        "decided_at": latest["decided_at"],
    }


# ---------------------------------------------------------------------------
# Track-03: LIVE DATABASE queries for the new revenue-risk domains
# (checkout_abandoned / mandate_failed / receivable_overdue /
# promise_to_pay_broken). Same conventions as the payment_failed queries
# above: get_live_* prefix, fresh session per call, no caching, never falls
# back to demo data.
# ---------------------------------------------------------------------------

def get_live_revenue_risk_events_df(limit: int = 200) -> pd.DataFrame:
    from app.models import RevenueRiskEvent

    db = get_live_session()
    try:
        rows = db.query(RevenueRiskEvent).order_by(RevenueRiskEvent.id.desc()).limit(limit).all()
        return pd.DataFrame.from_records([
            {
                "id": r.id, "event_type": r.event_type, "external_id": r.external_id, "customer_ref": r.customer_ref,
                "amount": r.amount, "currency": r.currency, "occurred_at": r.occurred_at, "received_at": r.received_at,
                "reason": r.reason, "status": r.status,
            }
            for r in rows
        ])
    finally:
        db.close()


def get_live_revenue_recovery_queue_df(limit: int = 200) -> pd.DataFrame:
    """Joins policy_decisions to revenue_risk_events for the 4 new domains
    ONLY -- filtered on the exact REVENUE_DOMAIN_DECISION_SOURCES set before
    joining, never a "rule_%" wildcard (see that constant's own docstring
    for why: policy_decisions.event_id has no real FK, and a loose filter
    could join an unrelated payment_failed row to a same-numbered
    revenue_risk_events row by pure coincidence)."""
    from app.models import PolicyDecision, RevenueRiskEvent
    from policy.policy_decision_store import REVENUE_DOMAIN_EVENT_ID_OFFSET
    from policy.revenue_recovery_policy import REVENUE_DOMAIN_DECISION_SOURCES

    db = get_live_session()
    try:
        rows = (
            db.query(PolicyDecision, RevenueRiskEvent)
            .join(RevenueRiskEvent, PolicyDecision.event_id == RevenueRiskEvent.id + REVENUE_DOMAIN_EVENT_ID_OFFSET)
            .filter(PolicyDecision.decision_source.in_(REVENUE_DOMAIN_DECISION_SOURCES))
            .order_by(PolicyDecision.id.desc())
            .limit(limit)
            .all()
        )
        return pd.DataFrame.from_records([
            {
                "event_id": rre.id, "event_type": rre.event_type, "customer_ref": rre.customer_ref,
                "amount": rre.amount, "selected_candidate_type": p.selected_candidate_type,
                "selected_candidate_datetime": p.selected_candidate_datetime,
                "classification_bucket": p.classification_bucket, "decision_source": p.decision_source,
                "model_version": p.model_version, "predicted_recovery_probability": p.predicted_recovery_probability,
                "decided_at": p.decided_at,
            }
            for p, rre in rows
        ])
    finally:
        db.close()


_REVENUE_COMPLIANCE_REASON_PATTERN = re.compile(
    r"payment_verdict=(?P<payment_verdict>\S+)\s+payment_reason=(?P<payment_reason>.*?)\s*\|\s*"
    r"communication_verdict=(?P<communication_verdict>\S+)\s+communication_reason=(?P<communication_reason>.*?)\s*\|\s*"
    r"rule_version=(?P<rule_version>\S+)"
)


def extract_revenue_compliance_fields(reason: str | None) -> dict:
    """Same convention as extract_compliance_fields above, but tailored to
    recovery/revenue_orchestrator.py's own `actor="revenue_compliance"`
    audit reason format (payment_verdict=.../communication_verdict=...),
    which is a different shape from the payment_failed path's
    payment_action_allowed=... format that function parses."""
    if not reason:
        return {}
    match = _REVENUE_COMPLIANCE_REASON_PATTERN.search(reason)
    return match.groupdict() if match else {}


def get_live_revenue_pipeline_snapshot() -> dict | None:
    """Same idea as get_live_pipeline_snapshot() above, but for the most
    recent event across the OTHER 5 unified-ML domains (checkout_abandoned/
    mandate_failed/receivable_overdue/promise_to_pay_broken, plus the
    one-time-payment/Payment-Link path) -- proves the SAME live event's ML
    score, policy candidate, compliance verdict, and LLM outcome all line up
    end to end, not just the payment_failed-with-subscription path. Every
    value is read straight off the real policy_decisions/audit_log/
    llm_invocations rows for that one event -- never fabricated. Returns
    None only when no live revenue-risk decision exists yet at all."""
    from app.models import AuditLog, LLMInvocation, PolicyDecision, RevenueRiskEvent
    from policy.policy_decision_store import REVENUE_DOMAIN_EVENT_ID_OFFSET
    from policy.revenue_recovery_policy import REVENUE_DOMAIN_DECISION_SOURCES

    db = get_live_session()
    try:
        row = (
            db.query(PolicyDecision, RevenueRiskEvent)
            .join(RevenueRiskEvent, PolicyDecision.event_id == RevenueRiskEvent.id + REVENUE_DOMAIN_EVENT_ID_OFFSET)
            .filter(PolicyDecision.decision_source.in_(REVENUE_DOMAIN_DECISION_SOURCES))
            .order_by(PolicyDecision.id.desc())
            .first()
        )
        if row is None:
            return None
        policy, rre = row
        stored_event_id = rre.id + REVENUE_DOMAIN_EVENT_ID_OFFSET

        audit_rows = db.query(AuditLog).filter(AuditLog.failure_event_id == stored_event_id).order_by(AuditLog.id).all()
        llm_rows = db.query(LLMInvocation).filter(LLMInvocation.event_id == stored_event_id).order_by(LLMInvocation.id).all()

        compliance_audit = next((a for a in reversed(audit_rows) if a.action == "revenue_orchestrator_compliance"), None)
        compliance_fields = extract_revenue_compliance_fields(compliance_audit.reason if compliance_audit else None)
        payment_verdict = compliance_fields.get("payment_verdict")
        communication_verdict = compliance_fields.get("communication_verdict")
        compliance_display = f"{payment_verdict} / {communication_verdict}" if payment_verdict else None

        comm = next((r for r in reversed(llm_rows) if r.task_name in ("outreach_microcopy", "voice_script_generation")), None)
        message_text = None
        if comm is not None:
            try:
                message_text = json.loads(comm.structured_output or "{}").get("message_text") or json.loads(comm.structured_output or "{}").get("script_text")
            except ValueError:
                message_text = None

        final_status_audit = next((a for a in reversed(audit_rows) if a.action == "revenue_orchestrator_final_status"), None)
        final_status = None
        if final_status_audit and "final_status=" in (final_status_audit.reason or ""):
            final_status = final_status_audit.reason.split("final_status=", 1)[1].split(" ", 1)[0]

        def _extract(reason: str | None, key: str) -> str | None:
            if not reason or f"{key}=" not in reason:
                return None
            return reason.split(f"{key}=", 1)[1].split(" ", 1)[0].split("|", 1)[0].strip()

        rule_baseline_candidate = _extract(policy.decision_reason, "rule_baseline_candidate")
        ml_recommendation = _extract(policy.decision_reason, "ml_recommendation")

        # Three distinct states (see policy/revenue_recovery_policy.py's
        # decide_for_revenue_risk_event): ML fully decided (USED), ML ran
        # but the rule-based eligibility/human-review gate overrode it
        # (CONSULTED_OVERRIDDEN -- ML's own score is still on the row), or
        # ML never ran at all, e.g. artifact unavailable (FALLBACK). Never
        # collapse the middle case into "not consulted" -- that's exactly
        # the bug this distinction exists to avoid.
        if policy.decision_source == "ml_unified_v1":
            ml_status = "USED"
        elif policy.predicted_recovery_probability is not None:
            ml_status = "CONSULTED_OVERRIDDEN"
        else:
            ml_status = "FALLBACK"

        return {
            "event_id": rre.id,
            "event_type": rre.event_type,
            "external_id": rre.external_id,
            "customer_ref": rre.customer_ref,
            "amount": rre.amount,
            "occurred_at": rre.occurred_at,
            "received_at": rre.received_at,
            "classification_bucket": policy.classification_bucket,
            "selected_candidate_type": policy.selected_candidate_type,
            "decision_source": policy.decision_source,
            "ml_status": ml_status,
            "is_ml_sourced": policy.decision_source == "ml_unified_v1",  # kept for backward compatibility; prefer ml_status
            "rule_baseline_candidate": rule_baseline_candidate,
            "ml_recommendation": ml_recommendation,
            "model_version": policy.model_version,
            "predicted_recovery_probability": policy.predicted_recovery_probability,
            "expected_recovery_value": policy.expected_recovery_value,
            "compliance_display": compliance_display,
            "llm_provider": comm.provider if comm is not None else None,
            "llm_model": comm.model_name if comm is not None else None,
            "llm_task": comm.task_name if comm is not None else None,
            "llm_success": comm.success if comm is not None else None,
            "communication_message": message_text,
            "final_status": final_status,
            "decided_at": policy.decided_at,
        }
    finally:
        db.close()


def get_live_revenue_at_risk_kpis() -> dict:
    """Same honesty rule as get_live_kpis(): counts only, never a fabricated
    recovery/recovered-amount figure for live data (see
    app.models.RecoveryOutcome's own binding-rule docstring)."""
    from app.models import RecoveryOutcome, RevenueRiskEvent

    from sqlalchemy import func

    db = get_live_session()
    try:
        total_events = db.query(RevenueRiskEvent).count()
        type_counts = dict(db.query(RevenueRiskEvent.event_type, func.count(RevenueRiskEvent.id)).group_by(RevenueRiskEvent.event_type).all())
        outcomes = db.query(RecoveryOutcome).all()
        total_at_risk = sum(o.at_risk_amount for o in outcomes)
        pending = sum(1 for o in outcomes if o.recovery_status == "PENDING")
        no_action = sum(1 for o in outcomes if o.recovery_status == "NO_ACTION")
        # RECOVERED/LOST/PARTIALLY_RECOVERED only ever appear for demo/synthetic
        # rows (confirmed_by="demo_synthetic") -- see RecoveryOutcome's binding rule.
        recovered_synthetic = sum(1 for o in outcomes if o.recovery_status == "RECOVERED" and o.confirmed_by == "demo_synthetic")
        return {
            "total_revenue_risk_events": total_events,
            "by_event_type": type_counts,
            "total_at_risk_amount": total_at_risk,
            "pending_cases": pending,
            "no_action_cases": no_action,
            "demo_synthetic_recovered_cases": recovered_synthetic,
        }
    finally:
        db.close()


def get_live_recovery_timeline_df(limit: int = 200) -> pd.DataFrame:
    """Chronological feed across BOTH payment_failed (raw_events) and the 4
    new domains (revenue_risk_events) -- one unified "Recovery Timeline"."""
    from app.models import RevenueRiskEvent

    db = get_live_session()
    try:
        payment_rows = db.query(RawEvent).order_by(RawEvent.id.desc()).limit(limit).all()
        revenue_rows = db.query(RevenueRiskEvent).order_by(RevenueRiskEvent.id.desc()).limit(limit).all()
        records = [
            {"timestamp": r.received_at, "event_type": "payment_failed", "reference": r.payment_id or r.razorpay_event_id, "amount": (r.amount / 100.0) if r.amount else None}
            for r in payment_rows
        ] + [
            {"timestamp": r.received_at, "event_type": r.event_type, "reference": r.external_id, "amount": r.amount}
            for r in revenue_rows
        ]
        df = pd.DataFrame.from_records(records)
        if not df.empty:
            df = df.sort_values("timestamp", ascending=False).reset_index(drop=True).head(limit)
        return df
    finally:
        db.close()


def get_live_recovery_outcomes_df(limit: int = 200) -> pd.DataFrame:
    from app.models import RecoveryOutcome

    db = get_live_session()
    try:
        rows = db.query(RecoveryOutcome).order_by(RecoveryOutcome.id.desc()).limit(limit).all()
        return pd.DataFrame.from_records([
            {
                "event_id": r.event_id, "event_type": r.event_type, "at_risk_amount": r.at_risk_amount,
                "recovered_amount": r.recovered_amount, "recovery_status": r.recovery_status,
                "confirmed_by": r.confirmed_by, "confirmed_payment_id": r.confirmed_payment_id,
                "observed_at": r.observed_at,
            }
            for r in rows
        ])
    finally:
        db.close()


def get_live_revenue_by_intervention_df() -> pd.DataFrame:
    """Revenue At Risk by Intervention -- grouped by selected_candidate_type
    across the 4 new domains. `recovered_amount` stays None for every live row
    (see RecoveryOutcome's binding rule) -- this reports AT-RISK amount per
    intervention type, honestly, not a fabricated recovered figure. Named to
    match: an earlier "Revenue Recovered by Intervention" label on this same
    data overclaimed relative to what the table actually shows."""
    from app.models import PolicyDecision, RevenueRiskEvent
    from policy.policy_decision_store import REVENUE_DOMAIN_EVENT_ID_OFFSET
    from policy.revenue_recovery_policy import REVENUE_DOMAIN_DECISION_SOURCES

    db = get_live_session()
    try:
        rows = (
            db.query(PolicyDecision, RevenueRiskEvent)
            .join(RevenueRiskEvent, PolicyDecision.event_id == RevenueRiskEvent.id + REVENUE_DOMAIN_EVENT_ID_OFFSET)
            .filter(PolicyDecision.decision_source.in_(REVENUE_DOMAIN_DECISION_SOURCES))
            .all()
        )
        records = [{"intervention": p.selected_candidate_type, "amount": rre.amount or 0.0} for p, rre in rows]
        df = pd.DataFrame.from_records(records)
        if df.empty:
            return df
        return df.groupby("intervention", as_index=False).agg(case_count=("amount", "count"), total_at_risk_amount=("amount", "sum"))
    finally:
        db.close()


def get_live_customer_recovery_queue_df(limit: int = 200) -> pd.DataFrame:
    """One row per customer_ref with their most recent revenue-risk case --
    a customer-centric view of the same recovery queue."""
    from app.models import RevenueRiskEvent

    db = get_live_session()
    try:
        rows = db.query(RevenueRiskEvent).order_by(RevenueRiskEvent.customer_ref, RevenueRiskEvent.id.desc()).limit(limit).all()
        df = pd.DataFrame.from_records([
            {"customer_ref": r.customer_ref, "event_type": r.event_type, "amount": r.amount, "status": r.status, "received_at": r.received_at}
            for r in rows
        ])
        if df.empty:
            return df
        return df.sort_values("received_at", ascending=False).drop_duplicates(subset="customer_ref", keep="first").reset_index(drop=True)
    finally:
        db.close()


def run_revenue_demo_generator(model=None) -> dict:
    """Thin UI wrapper over recovery/demo_generator.py -- parallel to
    run_demo_scenario() above. Defaults to a throwaway in-memory DB (never
    the real DATABASE_URL) unless the caller explicitly opts into targeting
    the live database."""
    from recovery.demo_generator import generate_demo_revenue_risk_events

    return generate_demo_revenue_risk_events(model=model)


_COMPLIANCE_REASON_PATTERN = re.compile(
    r"payment_action_allowed=(?P<payment_allowed>\w+)\s+payment_reason=(?P<payment_reason>.*?)\s*\|\s*"
    r"communication_action_allowed=(?P<communication_allowed>\w+)\s+communication_reason=(?P<communication_reason>.*?)\s*\|\s*"
    r"rule_version=(?P<rule_version>\S+)"
)


def extract_compliance_fields(reason: str | None) -> dict:
    """Best-effort parse of recovery/orchestrator.py's own
    `actor="compliance"` audit_log reason string -- DISPLAY ONLY, never
    feeds any decision. Tailored to that exact f-string format (this
    project's own, unchanged), not a generic parser -- returns {} rather
    than guessing if the format doesn't match."""
    if not reason:
        return {}
    match = _COMPLIANCE_REASON_PATTERN.search(reason)
    return match.groupdict() if match else {}


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

@st.cache_data(show_spinner=False, ttl=60)
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
