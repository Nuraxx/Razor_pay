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

    def validate_webhook_secret_present(self) -> None:
        if not self.RAZORPAY_WEBHOOK_SECRET:
            raise RuntimeError(
                "RAZORPAY_WEBHOOK_SECRET is not set. Copy .env.example to .env "
                "and fill it in with the secret from Dashboard > Webhooks "
                "(not the API key secret)."
            )


settings = Settings()
