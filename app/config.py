"""
Centralized configuration. Every credential comes from environment variables
(loaded from a local .env file) — nothing is ever hardcoded in source.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env from the project root if present


class Settings:
    RAZORPAY_KEY_ID: str = os.getenv("RAZORPAY_KEY_ID", "")
    RAZORPAY_KEY_SECRET: str = os.getenv("RAZORPAY_KEY_SECRET", "")

    # IMPORTANT: this is the *webhook* secret from Dashboard > Webhooks,
    # NOT the same value as RAZORPAY_KEY_SECRET. Razorpay signs webhook
    # payloads with this separate secret.
    RAZORPAY_WEBHOOK_SECRET: str = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")

    RAZORPAY_ENV: str = os.getenv("RAZORPAY_ENV", "test")

    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/recovery_agent.db")

    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # LLM-assisted communication layer (llm/). "mock" (default) makes
    # zero network calls and needs no API key; the project runs fully offline
    # with this unset. "anthropic" requires ANTHROPIC_API_KEY; "gemini"
    # requires GEMINI_API_KEY. "ollama" requires no API key -- it talks to a
    # locally-running Ollama server instead (OLLAMA_BASE_URL/OLLAMA_MODEL below).
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    # Local Ollama provider -- no API key, no external network call. Requires
    # an Ollama server already running locally with OLLAMA_MODEL pulled.
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:14b")

    # Track-03 hardening: automatic broken-promise detection (recovery/scheduler.py).
    # A lightweight in-process asyncio loop, not an external scheduler -- see
    # that module's docstring. Disabled by setting this to "false"; the test
    # suite never starts it regardless (FastAPI's lifespan never runs for a
    # bare TestClient(app) used without an explicit `with` block, which is
    # how tests/conftest.py's `client` fixture constructs it), but this flag
    # is kept explicit rather than relying on that fixture-level fact alone.
    ENABLE_PROMISE_SWEEP_SCHEDULER: bool = os.getenv("ENABLE_PROMISE_SWEEP_SCHEDULER", "true").lower() == "true"
    PROMISE_SWEEP_INTERVAL_SECONDS: int = int(os.getenv("PROMISE_SWEEP_INTERVAL_SECONDS", "300"))

    # Final pre-submission correction: contact-hours gate (policy/contact_hours.py),
    # wired into policy/compliance.py + policy/compliance_v2.py. Default window
    # (09:00-21:00 Asia/Kolkata) follows TRAI's own commercial-communication
    # window -- the only India-specific convention this project has any basis
    # to reference (this is a project guardrail, not a claim of TRAI/DPDP/RBI
    # regulatory compliance -- see policy/compliance.py's own disclaimer).
    # RBI's Fair Practices Code (a stricter, lending-specific 08:00-19:00
    # window with real enforcement history) is deliberately NOT the default
    # here: this project is subscription/receivables recovery, not lending,
    # so RBI's lending-specific code has no direct jurisdiction over it, and
    # this codebase makes no claim that it does. TRAI's broader commercial-
    # communication window is the closest applicable, sourced convention.
    CONTACT_HOURS_ENABLED: bool = os.getenv("CONTACT_HOURS_ENABLED", "true").lower() == "true"
    CONTACT_HOURS_TIMEZONE: str = os.getenv("CONTACT_HOURS_TIMEZONE", "Asia/Kolkata")
    CONTACT_HOURS_START: str = os.getenv("CONTACT_HOURS_START", "09:00")
    CONTACT_HOURS_END: str = os.getenv("CONTACT_HOURS_END", "21:00")

    # MULTI-ATTEMPT PERSISTENCE (final pre-submission audit): recovery/retry_sweep.py,
    # same in-process asyncio-loop pattern as ENABLE_PROMISE_SWEEP_SCHEDULER
    # above (not a second scheduler framework -- see that module's docstring).
    # Advances a PolicyDecision's own pre-computed retry_schedule (policy/
    # decision_engine_v4.py::build_retry_schedule_from_decision) one step at a
    # time as each step's scheduled time arrives, stopping early once
    # RecoveryOutcome confirms recovery. Disabled the same way, for the same
    # reason (never starts during tests -- FastAPI's lifespan never runs for a
    # bare TestClient(app) used without an explicit `with` block).
    ENABLE_RETRY_SWEEP_SCHEDULER: bool = os.getenv("ENABLE_RETRY_SWEEP_SCHEDULER", "true").lower() == "true"
    RETRY_SWEEP_INTERVAL_SECONDS: int = int(os.getenv("RETRY_SWEEP_INTERVAL_SECONDS", "300"))

    # Live feature enrichment (recovery/live_feature_enrichment.py, BUG-4
    # pre-submission audit fix): an OPTIONAL, best-effort call to Razorpay's
    # Subscriptions API to fill in `tenure_days` (the one Model B feature
    # genuinely obtainable that way -- see that module's FEATURE_SOURCES
    # classification for why the other 6 remaining features have no real
    # source at all, live or via any API). Defaults to "false" (the safe,
    # offline default) -- the live webhook path already computes its
    # deterministic/webhook-native features and falls back to the
    # rule-based policy tier with this disabled; nothing about correctness
    # depends on turning it on. An enrichment failure of any kind (timeout,
    # network, 4xx/5xx, malformed response) never crashes the webhook --
    # see that module's docstring.
    LIVE_FEATURE_ENRICHMENT_ENABLED: bool = os.getenv("LIVE_FEATURE_ENRICHMENT_ENABLED", "false").lower() == "true"
    LIVE_FEATURE_ENRICHMENT_TIMEOUT_SECONDS: float = float(os.getenv("LIVE_FEATURE_ENRICHMENT_TIMEOUT_SECONDS", "5"))

    def validate_webhook_secret_present(self) -> None:
        if not self.RAZORPAY_WEBHOOK_SECRET:
            raise RuntimeError(
                "RAZORPAY_WEBHOOK_SECRET is not set. Copy .env.example to .env "
                "and fill it in with the secret from Dashboard > Webhooks "
                "(not the API key secret)."
            )


settings = Settings()
