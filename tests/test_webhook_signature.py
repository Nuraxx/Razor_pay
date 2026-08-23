"""
Signature verification tested in isolation from the HTTP layer.
"""
import razorpay

from app.webhook_security import compute_signature, is_valid_signature
from tests.conftest import TEST_WEBHOOK_SECRET, sign


def test_valid_signature_is_accepted():
    body = b'{"event": "payment.failed", "payload": {}}'
    signature = sign(body)
    assert is_valid_signature(body, signature, TEST_WEBHOOK_SECRET) is True


def test_invalid_signature_is_rejected():
    body = b'{"event": "payment.failed", "payload": {}}'
    wrong_signature = sign(b"a completely different body")
    assert is_valid_signature(body, wrong_signature, TEST_WEBHOOK_SECRET) is False


def test_signature_over_tampered_body_is_rejected():
    """The exact attack a signature check exists to prevent: same signature,
    body changed after the fact."""
    original_body = b'{"event": "payment.failed", "amount": 100}'
    tampered_body = b'{"event": "payment.failed", "amount": 999999}'
    signature_for_original = sign(original_body)
    assert is_valid_signature(tampered_body, signature_for_original, TEST_WEBHOOK_SECRET) is False


def test_missing_signature_is_rejected():
    body = b'{"event": "payment.failed"}'
    assert is_valid_signature(body, None, TEST_WEBHOOK_SECRET) is False


def test_empty_secret_is_rejected():
    body = b'{"event": "payment.failed"}'
    signature = sign(body, secret="")
    assert is_valid_signature(body, signature, "") is False


def test_manual_computation_matches_official_razorpay_sdk():
    """
    Cross-check only — this project's actual request path in app/main.py
    never calls the SDK's verifier. This test exists purely to prove the
    manual implementation agrees with Razorpay's own official SDK, which is
    the strongest evidence it's implemented correctly.
    """
    body = b'{"event": "payment.failed", "payload": {"payment": {"entity": {"id": "pay_123"}}}}'
    our_signature = compute_signature(body, TEST_WEBHOOK_SECRET)

    client = razorpay.Client(auth=("dummy_key_id", "dummy_key_secret"))
    # Raises razorpay.errors.SignatureVerificationError if invalid; returns None if valid.
    client.utility.verify_webhook_signature(body.decode("utf-8"), our_signature, TEST_WEBHOOK_SECRET)
    # If we reach this line without an exception, our manually computed
    # signature was accepted by Razorpay's own SDK verifier.
