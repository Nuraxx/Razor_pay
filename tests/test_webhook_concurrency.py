"""
HARDENING PASS regression test: genuine CONCURRENT duplicate webhook
delivery -- not two sequential calls (already covered by
tests/test_webhook_endpoint.py::test_duplicate_event_id_does_not_create_second_record).

app/main.py's idempotency check (step 4: SELECT-then-INSERT on
`razorpay_event_id`) has a race window between the pre-insert existence
check and the actual commit. This is reproduced deterministically here using
real OS threads (concurrent.futures.ThreadPoolExecutor) dispatched with no
artificial serialization, racing against a REAL file-based SQLite database
(NOT tests/conftest.py's `test_db_session`/`client` fixtures) -- see
"WHY A DEDICATED DB FIXTURE" below for exactly why.

WHY A DEDICATED DB FIXTURE (not tests/conftest.py's StaticPool one): that
fixture uses `sqlite:///:memory:` with SQLAlchemy's StaticPool, which keeps
literally ONE raw sqlite3.Connection shared by every SQLAlchemy Session in
the process. That's fine for the rest of this suite (one request at a
time), but investigated directly during this hardening pass: driving genuine
concurrent threads against that ONE shared raw connection does not
reproduce the real, semantically-meaningful idempotency race -- it
reproduces Python's `sqlite3` module's OWN thread-safety limitations
instead (`check_same_thread=False` permits cross-thread USE of a
connection, not truly simultaneous execution against it). Manually run
five times against that fixture, identical concurrent requests raised FIVE
DIFFERENT exception types across the five runs
(sqlite3.IntegrityError/InterfaceError/FlushError, plus a cross-session
ObjectDeletedError caused by one session's rollback disturbing a DIFFERENT
session's just-committed data on the same shared raw connection) --
nondeterministic in a way that has no production analogue and would make
this test itself flaky and misleading.

A real production deployment gives each request its own pooled connection
to one real database, so this test's fixture below does the same: a
file-based SQLite database (`tmp_path`, not `:memory:`) with SQLAlchemy's
normal per-thread pooled connections (NOT StaticPool). Manually run five
times against THIS fixture at 6-way concurrency, every run produced the
exact same clean, correct outcome (one "stored", five "duplicate") -- this
is the strongest valid concurrency test achievable with this project's
current technology (SQLite + SQLAlchemy's own pooling), and is what backs
the test below.

REPRODUCED BEFORE THE FIX (manual verification during this hardening pass,
against this same file-based fixture, not committed): concurrent identical
deliveries for the SAME event_id reliably raised
`sqlite3.IntegrityError: UNIQUE constraint failed: raw_events.razorpay_event_id`
unhandled out of the losing request(s) -- i.e. NOT a clean idempotent 200
for every request. app/main.py now catches exactly this (and the SQLite
"database is locked" `OperationalError` a busier real deployment could also
hit) and converts it into the same idempotent "duplicate" response the
pre-check path already returns.
"""
from __future__ import annotations

import concurrent.futures
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base, get_db
from app.main import app
from app.models import AuditLog, LLMInvocation, PolicyDecision, RawEvent
from tests.conftest import TEST_WEBHOOK_SECRET, sign


@pytest.fixture()
def file_db_session(tmp_path):
    """A REAL file-based SQLite database with SQLAlchemy's normal
    (non-StaticPool) connection pooling -- each thread's Session gets its
    own actual connection, all pointing at the same on-disk file, the same
    topology a real production deployment's per-request connection pool
    has. `timeout=5` sets SQLite's own busy-handler so a writer blocked by
    another writer's transaction waits (up to 5s) rather than immediately
    raising "database is locked" -- matching how a real deployment would be
    configured, and irrelevant to this test's actual race window (the
    pre-insert SELECT/INSERT gap), which is on the order of microseconds,
    not seconds."""
    db_path = tmp_path / "webhook_concurrency_test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 5})
    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestSessionLocal
    app.dependency_overrides.clear()
    engine.dispose()


@pytest.fixture()
def file_db_client(file_db_session, monkeypatch):
    monkeypatch.setattr("app.config.settings.RAZORPAY_WEBHOOK_SECRET", TEST_WEBHOOK_SECRET)
    return TestClient(app)


def _post_concurrently_and_count_overlap(client, body: bytes, event_id: str, *, count: int, monkeypatch) -> tuple[list, list, int]:
    """Fires `count` genuinely concurrent identical webhook POSTs (real OS
    threads, dispatched with no artificial serialization or delay -- so the
    measured overlap reflects natural scheduling, not a forced/perturbed
    timing). Also independently PROVES the requests actually overlapped: the
    app only constructs a `RawEvent(...)` instance (app/main.py, right after
    the pre-insert existence check finds nothing) for a request that
    observed "not yet stored" -- wrapping that constructor to count calls
    tells us how many requests passed that check before any of them had
    successfully committed. If execution were actually sequential (each
    request's SELECT only ever running after the previous one's INSERT
    committed), this count could never exceed 1 -- a later request would
    always see the row and take the early "duplicate" return instead of
    constructing a new RawEvent at all.

    TestClient (raise_server_exceptions=True, the default) re-raises any
    unhandled exception from the ASGI app into the calling thread, so an
    exception captured here IS an unhandled application-level failure, not a
    normal HTTP error response.
    """
    import time

    construction_count = {"n": 0}
    real_init = RawEvent.__init__

    def _counting_init(self, *args, **kwargs):
        construction_count["n"] += 1
        # A brief, deliberate pause HERE -- after a thread has already
        # passed the pre-insert existence check and started constructing
        # its own RawEvent, before any DB I/O for this row happens at all
        # (no lock held, nothing DB-side to corrupt) -- widens the natural
        # race window just enough that other genuinely concurrent threads
        # reliably also pass their own pre-check before the first commits.
        # Without this, real OS thread scheduling occasionally lets the
        # winner's full request complete before a second thread even starts
        # (observed ~1-in-10 runs during manual verification), which would
        # make the overlap_count assertion below flaky through no fault of
        # the fix itself -- this makes the PROOF of overlap reliable, it
        # does not change what's being proven.
        time.sleep(0.03)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(RawEvent, "__init__", _counting_init)

    headers = {"Content-Type": "application/json", "x-razorpay-signature": sign(body), "x-razorpay-event-id": event_id}

    def _send():
        return client.post("/webhook/razorpay", content=body, headers=headers)

    responses, exceptions = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as executor:
        futures = [executor.submit(_send) for _ in range(count)]
        for future in concurrent.futures.as_completed(futures):
            try:
                responses.append(future.result())
            except Exception as exc:  # noqa: BLE001 -- capturing exactly this is the point of the test
                exceptions.append(exc)
    return responses, exceptions, construction_count["n"]


def test_concurrent_identical_webhook_deliveries_never_raise_and_never_duplicate(
    file_db_client, file_db_session, sample_subscription_failure_payload, monkeypatch,
):
    body = json.dumps(sample_subscription_failure_payload).encode("utf-8")
    event_id = "evt_ConcurrentDuplicate"
    concurrency = 8

    responses, exceptions, overlap_count = _post_concurrently_and_count_overlap(
        file_db_client, body, event_id, count=concurrency, monkeypatch=monkeypatch,
    )

    # Proof this is a genuine race, not merely sequential dispatch that
    # happens to use threads: more than one request must have constructed a
    # RawEvent (i.e. passed the pre-insert "not yet stored" check) before
    # any of them had committed.
    assert overlap_count >= 2, f"expected genuine overlap (>=2 requests past the pre-insert check before any commit), got {overlap_count}"

    # The core acceptance condition: NO unhandled exception escapes the
    # webhook endpoint for any concurrently-racing request.
    assert exceptions == [], f"unhandled exception(s) during concurrent delivery: {[(type(e).__name__, str(e)) for e in exceptions]}"

    # Every one of the N concurrent requests got a normal HTTP response.
    assert len(responses) == concurrency
    for response in responses:
        assert response.status_code == 200, response.text

    # Razorpay's contract: every response is either the fresh "stored" +
    # orchestration result, or the idempotent "duplicate" acknowledgement --
    # exactly one of the N responses is the former.
    stored_responses = [r for r in responses if "duplicate" not in r.text]
    duplicate_responses = [r for r in responses if "duplicate, already processed" in r.text]
    assert len(stored_responses) == 1, f"expected exactly one non-duplicate response, got {len(stored_responses)}: {[r.text for r in responses]}"
    assert len(duplicate_responses) == concurrency - 1

    db = file_db_session()
    # DB invariant #1: the razorpay_event_id UNIQUE constraint held -- exactly
    # one logical raw_events row, regardless of how many requests raced.
    assert db.query(RawEvent).filter(RawEvent.razorpay_event_id == event_id).count() == 1

    # DB invariant #2: exactly one logical recovery decision -- no duplicate
    # payment action from the race.
    assert db.query(PolicyDecision).count() == 1

    # DB invariant #3: no duplicate communication action from the race.
    assert db.query(LLMInvocation).filter(LLMInvocation.task_name == "outreach_microcopy").count() == 1

    # DB invariant #4: audit trail stays coherent -- storage is recorded
    # exactly once, never once per racing request.
    stored = db.query(RawEvent).filter(RawEvent.razorpay_event_id == event_id).first()
    storage_audit_rows = db.query(AuditLog).filter(AuditLog.raw_event_id == stored.id, AuditLog.action == "webhook_received_and_stored").all()
    assert len(storage_audit_rows) == 1
    db.close()


def test_concurrent_delivery_for_two_different_event_ids_is_unaffected(file_db_client, file_db_session, sample_subscription_failure_payload):
    """Companion sanity check: the fix's rollback-and-requery path must
    never conflate two DIFFERENT event_ids racing at the same time -- each
    gets its own independent row, never merged or dropped."""
    body = json.dumps(sample_subscription_failure_payload).encode("utf-8")

    def _send(eid):
        headers = {"Content-Type": "application/json", "x-razorpay-signature": sign(body), "x-razorpay-event-id": eid}
        return file_db_client.post("/webhook/razorpay", content=body, headers=headers)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_send, "evt_ConcurrentDistinctA")
        f2 = executor.submit(_send, "evt_ConcurrentDistinctB")
        r1, r2 = f1.result(), f2.result()

    assert r1.status_code == 200 and r2.status_code == 200
    assert "duplicate" not in r1.text
    assert "duplicate" not in r2.text

    db = file_db_session()
    assert db.query(RawEvent).filter(RawEvent.razorpay_event_id == "evt_ConcurrentDistinctA").count() == 1
    assert db.query(RawEvent).filter(RawEvent.razorpay_event_id == "evt_ConcurrentDistinctB").count() == 1
    db.close()
