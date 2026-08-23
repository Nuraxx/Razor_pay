"""
Structured application logging.

Kept deliberately simple for Day 1 (stdlib logging, no external observability
stack) — the audit_log table is the observability surface that actually
matters for this project's judging criteria; this logger is for operational
visibility while developing (e.g. watching signature checks in the terminal).
"""
import logging
import sys

from app.config import settings


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("recovery_agent")
    logger.setLevel(settings.LOG_LEVEL)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


log = configure_logging()
