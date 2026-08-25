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
