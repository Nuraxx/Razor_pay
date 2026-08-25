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

    # Day 11 -- LLM-assisted communication layer (llm/). "mock" (default) makes
    # zero network calls and needs no API key; the project runs fully offline
    # with this unset. "anthropic" requires ANTHROPIC_API_KEY; "gemini"
    # requires GEMINI_API_KEY.
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    def validate_webhook_secret_present(self) -> None:
        if not self.RAZORPAY_WEBHOOK_SECRET:
            raise RuntimeError(
                "RAZORPAY_WEBHOOK_SECRET is not set. Copy .env.example to .env "
                "and fill it in with the secret from Dashboard > Webhooks "
                "(not the API key secret)."
            )


settings = Settings()
