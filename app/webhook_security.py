"""
Razorpay webhook signature verification — implemented manually and directly,
per the current official scheme (https://razorpay.com/docs/webhooks/validate-test/):

    key                = webhook secret (Dashboard > Webhooks > Secret — NOT the API key secret)
    message            = the RAW webhook request body (exact bytes, not re-parsed/re-serialized JSON)
    expected_signature = hex(HMAC_SHA256(key, message))
    received_signature = the value of the `X-Razorpay-Signature` header

This is deliberately NOT calling razorpay-python's `client.utility.verify_webhook_signature()`
helper for the actual verification path — per project rules, manual verification is the
one thing here that must not be delegated to the SDK, even though the SDK does the same
computation internally. (The SDK's helper is used only as an independent cross-check in
tests/test_webhook_signature.py, never in the request path.)
"""
import hmac
import hashlib


def compute_signature(raw_body: bytes, webhook_secret: str) -> str:
    """Compute the expected hex-digest HMAC-SHA256 signature for a raw request body."""
    return hmac.new(
        key=webhook_secret.encode("utf-8"),
        msg=raw_body,
        digestmod=hashlib.sha256,
    ).hexdigest()


def is_valid_signature(raw_body: bytes, received_signature: str | None, webhook_secret: str) -> bool:
    """
    Constant-time comparison of the received signature against the expected one.

    Returns False (never raises) for any malformed input — a missing/garbage
    signature header is a verification failure, not a crash.
    """
    if not received_signature or not webhook_secret:
        return False

    expected_signature = compute_signature(raw_body, webhook_secret)

    # hmac.compare_digest specifically to avoid a timing side-channel —
    # a plain `==` comparison leaks information about how many leading
    # characters matched via response-time differences.
    return hmac.compare_digest(expected_signature, received_signature)
