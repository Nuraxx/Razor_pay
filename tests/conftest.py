import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base, get_db
from app.main import app

TEST_WEBHOOK_SECRET = "test_webhook_secret_for_pytest_only"


@pytest.fixture(scope="session", autouse=True)
def _unified_model_test_artifact(tmp_path_factory):
    """
    BUG-2 regression fixture (pre-submission audit): a genuinely fresh clone
    has no model/artifacts/unified_model.joblib (gitignored -- produced only
    by running `python -m model.train_unified_model` by hand). Several tests
    exercise the REAL model-loading path (model.unified_model.load_unified_model
    / get_live_unified_model) -- tests/test_revenue_recovery_policy.py's
    TestUnifiedMLPolicyBoundary, and the ml_unified_v1-asserting tests in
    tests/test_webhook_endpoint.py -- and used to require that artifact to
    already exist on disk, making `pytest tests/ -q` fail (or silently give
    different results) on a fresh clone until someone manually trained a
    model first.

    This fixture trains ONE real CatBoost unified model -- via the exact
    same train_unified_model() the production training script calls, no
    mocking of the ML mechanism itself -- into a session-scoped pytest tmp
    directory, and points the module's own path constants there for the
    whole test session. It never writes into model/artifacts/ (see the
    _model_path()/TRAINING_REPORT_PATH mkdir fix in model/unified_model.py
    this fixture depends on) and never touches the real committed-ignored
    artifact if one happens to exist in a developer's working tree.

    Individual tests that need to exercise the ARTIFACT-UNAVAILABLE path
    (tests/test_unified_model.py::TestArtifactUnavailableFallback) already
    monkeypatch UNIFIED_MODEL_PATH themselves at function scope -- pytest's
    monkeypatch correctly overrides this session default for the duration of
    that one test and restores it afterwards, so the two compose correctly.
    """
    import model.unified_model as um

    artifact_dir = tmp_path_factory.mktemp("unified_model_test_artifact")
    original_model_path = um.UNIFIED_MODEL_PATH
    original_report_path = um.TRAINING_REPORT_PATH
    um.UNIFIED_MODEL_PATH = artifact_dir / "unified_model.joblib"
    um.TRAINING_REPORT_PATH = artifact_dir / "unified_model_training_report.json"
    um.train_unified_model()
    um.reset_live_unified_model_cache()

    yield um.UNIFIED_MODEL_PATH

    um.UNIFIED_MODEL_PATH = original_model_path
    um.TRAINING_REPORT_PATH = original_report_path
    um.reset_live_unified_model_cache()


@pytest.fixture(autouse=True)
def _force_mock_llm_provider_by_default(monkeypatch):
    """The suite must stay runnable offline regardless of the developer's
    own local .env (app/config.py's `load_dotenv()` loads the real .env
    unconditionally at import time -- there is no test-specific override).
    Without this, any test that calls generate_outreach_microcopy() /
    parse_promise_to_pay() / generate_batch_explanation() without an
    explicit `client=` argument falls through to get_llm_client(), which
    would make a REAL network call whenever a developer's .env happens to
    have LLM_PROVIDER=anthropic/gemini set (e.g. for real-API testing) --
    producing non-deterministic output that breaks assertions written
    against the deterministic mock provider.

    A test that specifically wants to exercise a real provider's selection
    logic (e.g. TestGeminiProviderSelection) still can: its own
    monkeypatch.setattr(...) call in the test body runs after this autouse
    fixture and simply overwrites the same attribute for that test only.
    """
    monkeypatch.setattr("app.config.settings.LLM_PROVIDER", "mock")
    monkeypatch.setattr("app.config.settings.ANTHROPIC_API_KEY", "")
    monkeypatch.setattr("app.config.settings.GEMINI_API_KEY", "")


@pytest.fixture()
def test_db_session():
    """A fresh in-memory SQLite DB per test, isolated from the real data/*.db file."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # keeps the same in-memory DB across connections in this test
    )
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


@pytest.fixture()
def client(test_db_session, monkeypatch):
    monkeypatch.setattr("app.config.settings.RAZORPAY_WEBHOOK_SECRET", TEST_WEBHOOK_SECRET)
    return TestClient(app)


def sign(body_bytes: bytes, secret: str = TEST_WEBHOOK_SECRET) -> str:
    """Test-side signature computation, deliberately written independently
    of app/webhook_security.py so a bug in the app code can't hide itself
    by also being present in the test."""
    return hmac.new(secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()


@pytest.fixture()
def sample_subscription_failure_payload() -> dict:
    """
    Shape verified against Razorpay's current documentation: top-level
    entity/account_id/event/contains/payload/created_at envelope, with
    payload.payment.entity carrying the error_* fields and payload.subscription.entity
    present for subscription-linked charges. Field names are not invented —
    see error field names at https://razorpay.com/docs/api/errors/ and the
    webhook envelope at https://razorpay.com/docs/webhooks/payloads/payments/
    """
    return {
        "entity": "event",
        "account_id": "acc_TestAccountId000",
        "event": "payment.failed",
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_TestPaymentId001",
                    "entity": "payment",
                    "amount": 29900,
                    "currency": "INR",
                    "status": "failed",
                    "order_id": None,
                    "invoice_id": "inv_TestInvoiceId001",
                    "method": "card",
                    "error_code": "BAD_REQUEST_ERROR",
                    "error_description": "Payment failed due to insufficient funds in the customer account.",
                    "error_reason": "insufficient_fund",
                    "error_source": "customer",
                    "error_step": "payment_authorization",
                    "created_at": 1755840000,
                },
            },
            "subscription": {
                "entity": {
                    "id": "sub_TestSubscriptionId001",
                    "entity": "subscription",
                    "plan_id": "plan_TestPlanId001",
                    "status": "active",
                },
            },
        },
        "created_at": 1755840005,
    }
