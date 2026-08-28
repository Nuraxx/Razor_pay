"""
BUG-1 regression tests (pre-submission audit): the documented local setup

    cp .env.example .env
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

used to fail at startup because RAZORPAY_WEBHOOK_SECRET was empty in
.env.example (app/config.py::validate_webhook_secret_present raises
RuntimeError on an empty secret, and app/main.py's lifespan calls it before
anything else). The fix is a fake, clearly-labeled local-dev placeholder
value in .env.example -- these tests prove BOTH halves of the fix: (A) the
placeholder is non-empty and satisfies startup validation, and (B) the
placeholder is an ordinary HMAC secret like any other -- it does not weaken
real signature verification, disable it, or fall back to the API key secret.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import settings
from app.webhook_security import compute_signature, is_valid_signature

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"
CONFIG_SOURCE_PATH = PROJECT_ROOT / "app" / "config.py"


def _read_env_example_value(key: str) -> str:
    text = ENV_EXAMPLE_PATH.read_text()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1]
    raise AssertionError(f"{key} not found in .env.example")


# ---------------------------------------------------------------------------
# A. .env.example's placeholder satisfies startup, and is clearly fake
# ---------------------------------------------------------------------------

def test_env_example_webhook_secret_placeholder_is_nonempty_and_clearly_fake():
    value = _read_env_example_value("RAZORPAY_WEBHOOK_SECRET")
    assert value != ""
    assert "placeholder" in value.lower()


def test_env_example_placeholder_satisfies_startup_validation(monkeypatch):
    """The exact call app/main.py's lifespan makes before anything else --
    proves `cp .env.example .env` no longer fails startup."""
    placeholder = _read_env_example_value("RAZORPAY_WEBHOOK_SECRET")
    monkeypatch.setattr(settings, "RAZORPAY_WEBHOOK_SECRET", placeholder)
    settings.validate_webhook_secret_present()  # must not raise


def test_empty_webhook_secret_still_fails_startup_validation(monkeypatch):
    """Regression guard for the OTHER failure mode: startup must still
    refuse to come up silently with a genuinely empty secret."""
    monkeypatch.setattr(settings, "RAZORPAY_WEBHOOK_SECRET", "")
    with pytest.raises(RuntimeError, match="RAZORPAY_WEBHOOK_SECRET"):
        settings.validate_webhook_secret_present()


# ---------------------------------------------------------------------------
# B. The placeholder does not weaken real signature verification
# ---------------------------------------------------------------------------

def test_placeholder_secret_does_not_make_arbitrary_signatures_valid():
    placeholder = _read_env_example_value("RAZORPAY_WEBHOOK_SECRET")
    body = b'{"event": "payment.failed", "payload": {}}'
    garbage_signature = "0" * 64  # well-formed hex, not a valid HMAC for this body/secret
    assert is_valid_signature(body, garbage_signature, placeholder) is False


def test_placeholder_secret_rejects_a_signature_computed_with_a_different_secret():
    placeholder = _read_env_example_value("RAZORPAY_WEBHOOK_SECRET")
    body = b'{"event": "payment.failed", "payload": {}}'
    signed_with_wrong_secret = compute_signature(body, "some_other_secret_entirely")
    assert is_valid_signature(body, signed_with_wrong_secret, placeholder) is False


def test_placeholder_secret_still_correctly_accepts_a_genuinely_matching_signature():
    """The placeholder behaves as an ordinary HMAC key -- a signature
    computed correctly against it IS accepted, proving verification is
    still real (not a bypass), just using a fake local-dev key value."""
    placeholder = _read_env_example_value("RAZORPAY_WEBHOOK_SECRET")
    body = b'{"event": "payment.failed", "payload": {}}'
    correct_signature = compute_signature(body, placeholder)
    assert is_valid_signature(body, correct_signature, placeholder) is True


def test_placeholder_secret_rejects_a_tampered_body():
    placeholder = _read_env_example_value("RAZORPAY_WEBHOOK_SECRET")
    original_body = b'{"event": "payment.failed", "amount": 100}'
    tampered_body = b'{"event": "payment.failed", "amount": 999999}'
    signature_for_original = compute_signature(original_body, placeholder)
    assert is_valid_signature(tampered_body, signature_for_original, placeholder) is False


def test_placeholder_secret_rejects_missing_signature():
    placeholder = _read_env_example_value("RAZORPAY_WEBHOOK_SECRET")
    body = b'{"event": "payment.failed"}'
    assert is_valid_signature(body, None, placeholder) is False


def test_config_source_never_falls_back_webhook_secret_to_the_api_key_secret():
    """Static guard against the exact anti-pattern the audit forbids:
    RAZORPAY_WEBHOOK_SECRET's definition must read its own env var only,
    never silently default/fall back to RAZORPAY_KEY_SECRET."""
    source = CONFIG_SOURCE_PATH.read_text()
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("RAZORPAY_WEBHOOK_SECRET:") or stripped.startswith("RAZORPAY_WEBHOOK_SECRET ="):
            assert "RAZORPAY_KEY_SECRET" not in stripped
            return
    pytest.fail("RAZORPAY_WEBHOOK_SECRET assignment not found in app/config.py")


def test_config_source_never_treats_an_empty_secret_as_automatically_valid():
    """Static guard: validate_webhook_secret_present must actually check
    for emptiness, not just exist as a no-op."""
    source = CONFIG_SOURCE_PATH.read_text()
    assert "def validate_webhook_secret_present" in source
    # the guard must reference the secret itself, not an unconditional pass
    marker = "def validate_webhook_secret_present"
    body_start = source.index(marker)
    body = source[body_start:body_start + 400]
    assert "RAZORPAY_WEBHOOK_SECRET" in body
    assert "raise" in body


def test_config_source_reads_webhook_secret_from_its_own_env_var_with_no_hardcoded_default():
    """Static guard: the class attribute must come from
    os.getenv("RAZORPAY_WEBHOOK_SECRET", ...) with an empty-string default --
    never a hardcoded non-empty value baked into source (that would defeat
    the whole point of requiring configuration; only .env.example -- a
    separate, clearly-labeled file -- should carry the fake placeholder)."""
    assert 'os.getenv("RAZORPAY_WEBHOOK_SECRET", "")' in CONFIG_SOURCE_PATH.read_text()
