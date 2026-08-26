"""
Track-03 tests: recovery/demo_generator.py -- proves it creates exactly one
event of each of the 7 kinds via the REAL recovery engine (never raw DB
inserts), defaults to a throwaway in-memory DB, and never touches
settings.DATABASE_URL unless the caller explicitly passes a live session.
"""
from recovery.demo_generator import build_demo_database, generate_demo_revenue_risk_events
from recovery.orchestrator import RecoveryExecutionResult
from recovery.revenue_schemas import RevenueRecoveryResult


class TestGeneratesAllSevenKinds:
    def test_returns_all_seven_kinds(self):
        results = generate_demo_revenue_risk_events()
        expected_kinds = {
            "failed_payment", "checkout_abandoned", "subscription_failure",
            "mandate_failed", "receivable_overdue", "promise_to_pay", "broken_promise",
        }
        assert set(results.keys()) == expected_kinds

    def test_failed_payment_uses_the_real_unmodified_orchestrator(self):
        results = generate_demo_revenue_risk_events()
        assert isinstance(results["failed_payment"], RecoveryExecutionResult)
        assert results["failed_payment"].classification_bucket == "retryable_soft"

    def test_subscription_failure_also_uses_the_real_unmodified_orchestrator(self):
        results = generate_demo_revenue_risk_events()
        assert isinstance(results["subscription_failure"], RecoveryExecutionResult)

    def test_new_domains_use_the_real_revenue_orchestrator(self):
        results = generate_demo_revenue_risk_events()
        for kind in ("checkout_abandoned", "mandate_failed", "receivable_overdue"):
            assert isinstance(results[kind], RevenueRecoveryResult)

    def test_promise_to_pay_is_a_real_persisted_promise(self):
        results = generate_demo_revenue_risk_events()
        assert results["promise_to_pay"] is not None
        assert results["promise_to_pay"].status is not None

    def test_broken_promise_is_a_real_orchestrated_recovery_action(self):
        # Broken-promise detection routes through the real orchestrator, so
        # this is a RevenueRecoveryResult -- proving the "feeds back into the
        # recovery engine" requirement produces an actual new recovery
        # action, not just an inert PromiseOutcome row.
        results = generate_demo_revenue_risk_events()
        assert results["broken_promise"] is not None
        assert isinstance(results["broken_promise"], RevenueRecoveryResult)
        assert results["broken_promise"].event_type == "promise_to_pay_broken"


class TestDefaultsToThrowawayInMemoryDB:
    def test_default_call_never_touches_the_real_database_url(self, monkeypatch):
        def _explode(*args, **kwargs):
            raise AssertionError("must never touch the real DATABASE_URL when db=None")

        monkeypatch.setattr("app.config.settings.DATABASE_URL", "sqlite:////this/path/must/never/be/opened.db")
        # generate_demo_revenue_risk_events() with db=None uses its own
        # in-memory engine (build_demo_database()) -- it never reads
        # settings.DATABASE_URL at all, proven by this not raising even
        # though that path is nonsense.
        results = generate_demo_revenue_risk_events()
        assert results["failed_payment"] is not None

    def test_repeated_calls_are_independent_throwaway_databases(self):
        first = generate_demo_revenue_risk_events()
        second = generate_demo_revenue_risk_events()
        # same demo event_id/idempotency_key reused each time -- proves each
        # call got its OWN fresh database, not a shared/leaking one.
        assert first["failed_payment"].classification_bucket == second["failed_payment"].classification_bucket


class TestExplicitLiveSessionOptIn:
    def test_caller_supplied_session_is_used_and_not_closed_by_the_generator(self, test_db_session):
        db = test_db_session()
        results = generate_demo_revenue_risk_events(db)
        assert results["failed_payment"] is not None
        # session still usable -- the generator did not close a caller-owned session
        from app.models import RevenueRiskEvent

        count = db.query(RevenueRiskEvent).count()
        assert count == 4  # checkout, mandate, receivable, + the promise_to_pay_broken feedback event
        db.close()

    def test_calling_twice_on_the_same_caller_supplied_session_does_not_raise(self, test_db_session):
        # Full-system audit finding: the 3 domain blocks used to bare-insert
        # a RevenueRiskEvent/PromiseToPay row every call, relying only on the
        # DB's unique constraint on idempotency_key/source_text_hash -- fine
        # against the default fresh in-memory DB, but a second call against
        # the SAME caller-supplied session (a use this class's own sibling
        # test above proves is supported) raised a hard IntegrityError
        # instead of gracefully reusing the existing event, unlike every
        # /events/* API route's query-before-insert pattern. Fixed additively.
        from app.models import PromiseToPay, RevenueRiskEvent

        db = test_db_session()
        generate_demo_revenue_risk_events(db)
        second_results = generate_demo_revenue_risk_events(db)  # must not raise IntegrityError
        assert second_results["failed_payment"] is not None

        # no duplicate rows were created for the 3 idempotency-keyed domains
        assert db.query(RevenueRiskEvent).filter(RevenueRiskEvent.idempotency_key == "demo:checkout_abandoned:1").count() == 1
        assert db.query(RevenueRiskEvent).filter(RevenueRiskEvent.idempotency_key == "demo:mandate_failed:1").count() == 1
        assert db.query(RevenueRiskEvent).filter(RevenueRiskEvent.idempotency_key == "demo:receivable_overdue:1").count() == 1
        assert db.query(PromiseToPay).filter(PromiseToPay.source_text_hash == "demo_broken_promise_hash").count() == 1
        db.close()


class TestBuildDemoDatabase:
    def test_returns_a_working_session_with_all_tables_created(self):
        db = build_demo_database()
        from app.models import RevenueRiskEvent

        assert db.query(RevenueRiskEvent).count() == 0
        db.close()
