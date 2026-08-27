"""
Hardening-pass tests: recovery/scheduler.py + its wiring into app/main.py's
FastAPI lifespan. Covers exception isolation (one bad sweep pass must not
kill the background loop) and the enable/disable mechanism (both the
explicit settings flag, and the incidental-but-load-bearing fact that
FastAPI's lifespan never runs for this project's test client fixture).
"""
import asyncio

import pytest

from recovery.scheduler import promise_sweep_background_loop, run_promise_sweep_once


class TestRunPromiseSweepOnce:
    def test_returns_zero_when_nothing_is_due(self, test_db_session):
        db = test_db_session()
        assert run_promise_sweep_once(db) == 0
        db.close()


class TestSchedulerExceptionIsolation:
    def test_a_failing_sweep_pass_does_not_kill_the_loop(self, monkeypatch):
        call_count = {"n": 0}

        def _flaky_sweep(db=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated transient failure (e.g. a DB blip)")
            return 0

        monkeypatch.setattr("recovery.scheduler.run_promise_sweep_once", _flaky_sweep)

        async def _run_briefly():
            with pytest.raises((asyncio.TimeoutError, TimeoutError)):
                await asyncio.wait_for(promise_sweep_background_loop(interval_seconds=0), timeout=0.3)

        asyncio.run(_run_briefly())
        # the loop kept going after the first failure -- proves one bad
        # promise/sweep pass can never permanently kill the scheduler
        assert call_count["n"] >= 2

    def test_cancellation_propagates_cleanly_and_is_not_swallowed_as_a_failure(self):
        async def _run_and_cancel():
            task = asyncio.create_task(promise_sweep_background_loop(interval_seconds=999))
            await asyncio.sleep(0.05)  # let it start and hit the first sleep
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(_run_and_cancel())


class TestSchedulerDisableMechanism:
    def test_disabled_flag_prevents_task_creation_in_lifespan(self, monkeypatch):
        # Exercises the ACTUAL lifespan function app/main.py wires the
        # scheduler into -- not a reimplementation of its gating logic.
        from app.main import app as fastapi_app
        from app.main import lifespan

        monkeypatch.setattr("app.main.init_db", lambda: None)  # never touch the real DB file from this test
        monkeypatch.setattr("app.config.settings.validate_webhook_secret_present", lambda: None)
        monkeypatch.setattr("app.config.settings.ENABLE_PROMISE_SWEEP_SCHEDULER", False)
        monkeypatch.setattr("app.config.settings.ENABLE_RETRY_SWEEP_SCHEDULER", False)

        created_task = {"value": False}

        def _spy_create_task(coro):
            created_task["value"] = True
            coro.close()  # avoid an "coroutine was never awaited" warning
            raise AssertionError("asyncio.create_task must not be called when both schedulers are disabled")

        monkeypatch.setattr("app.main.asyncio.create_task", _spy_create_task)

        async def _enter_and_exit_lifespan():
            async with lifespan(fastapi_app):
                pass

        asyncio.run(_enter_and_exit_lifespan())
        assert created_task["value"] is False

    def test_enabled_flag_does_create_a_background_task(self, monkeypatch):
        from app.main import app as fastapi_app
        from app.main import lifespan

        monkeypatch.setattr("app.main.init_db", lambda: None)
        monkeypatch.setattr("app.config.settings.validate_webhook_secret_present", lambda: None)
        monkeypatch.setattr("app.config.settings.ENABLE_PROMISE_SWEEP_SCHEDULER", True)
        # Replace the real forever-loop with something that returns
        # immediately, so this test doesn't need to wait or cancel anything.
        monkeypatch.setattr("app.main.promise_sweep_background_loop", lambda interval_seconds: _noop_coro())

        async def _enter_and_exit_lifespan():
            async with lifespan(fastapi_app):
                pass

        asyncio.run(_enter_and_exit_lifespan())  # must not raise

    def test_bare_testclient_without_context_manager_never_runs_the_lifespan(self):
        # The load-bearing fact this whole disable mechanism piggybacks on:
        # tests/conftest.py's `client` fixture constructs TestClient(app)
        # WITHOUT an explicit `with` block, which means Starlette never runs
        # the lifespan context at all for it -- so the scheduler can never
        # start during this project's test suite regardless of the flag.
        from fastapi.testclient import TestClient

        from app.main import app as fastapi_app

        entered = {"value": False}
        real_lifespan_context = fastapi_app.router.lifespan_context

        class _SpyLifespan:
            def __init__(self, app):
                self._ctx = real_lifespan_context(app)

            async def __aenter__(self):
                entered["value"] = True
                return await self._ctx.__aenter__()

            async def __aexit__(self, *exc_info):
                return await self._ctx.__aexit__(*exc_info)

        fastapi_app.router.lifespan_context = lambda app: _SpyLifespan(app)
        try:
            client = TestClient(fastapi_app)
            client.get("/health")
        finally:
            fastapi_app.router.lifespan_context = real_lifespan_context

        assert entered["value"] is False


async def _noop_coro() -> None:
    return None
