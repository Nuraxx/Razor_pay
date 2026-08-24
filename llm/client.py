"""
Day-11 LLM provider abstraction.

Deliberately minimal -- no LangChain, no LangGraph, no agent framework, no
autonomous tool use (brief section 2). Two providers, selected by the
`LLM_PROVIDER` env var (`app/config.py::settings.LLM_PROVIDER`):

  "mock"      (default) -- deterministic, offline, makes zero network calls.
               Works with no ANTHROPIC_API_KEY at all. Used by every test in
               this project and by scripts/run_llm_demo.py by default.
  "anthropic" -- thin wrapper over the real `anthropic` SDK. Requires
               ANTHROPIC_API_KEY to be set.

Both implement the same `LLMClient.complete(system_prompt, user_prompt) ->
str` interface, so `llm/service.py` never needs to know which one it's
talking to.

FAILURE HANDLING (brief section 5): every failure mode below -- provider
unavailable, timeout, SDK/API exception, empty response -- is caught here
and re-raised as the single `LLMProviderError` type. `llm/service.py` is
the only place that decides what to do about it (deterministic fallback);
this module's job is only to normalize failures, never to swallow them
silently and never to let a raw SDK exception (which could embed request
details) escape uncaught.

SECRET HANDLING: `LLMProviderError` messages below are built from
`type(exc).__name__` only, NEVER from `str(exc)` -- an SDK exception's
string form can embed request/response details. The API key itself is
never logged, never included in any exception message, and never appears
in `llm/service.py`'s audit records.
"""
from __future__ import annotations

import abc

from app.config import settings
from app.logging_config import log

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
MOCK_MODEL_NAME = "mock-llm-v1"


class LLMProviderError(Exception):
    """Raised for any provider-level failure -- timeout, API/SDK exception,
    empty response, provider unavailable/misconfigured. Never raised past
    llm/service.py; the policy layer never sees this exception at all."""


class LLMClient(abc.ABC):
    model_name: str

    @abc.abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 512) -> str:
        """Returns the raw text response (expected to be JSON, validated by
        the caller). Raises LLMProviderError on any failure -- never returns
        None, never returns an empty string without raising."""


class MockLLMClient(LLMClient):
    """Deterministic, offline mock -- makes NO network calls (brief section
    6). Returns canned, schema-shaped JSON derived deterministically from
    the user_prompt's content, so tests can assert specific behavior without
    depending on real model output. This is the default provider
    (`LLM_PROVIDER=mock` or unset) and the only one that runs in this
    project's test suite and CI."""

    model_name = MOCK_MODEL_NAME

    def complete(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 512) -> str:
        # Deliberately simple, deterministic routing by a marker embedded in
        # the user prompt by llm/prompts.py -- see MOCK_TASK_MARKER usage
        # there. This keeps the mock entirely self-contained in this module
        # (no cross-import of llm/service.py's task logic) while still
        # producing schema-valid, task-appropriate JSON for each of the 3 jobs.
        from llm.prompts import mock_response_for_prompt

        return mock_response_for_prompt(user_prompt)


class AnthropicLLMClient(LLMClient):
    """Thin wrapper over the real `anthropic` SDK. `anthropic` is imported
    lazily inside __init__ (not at module import time) so importing this
    module -- and running the mock-mode test suite -- never requires the
    package to be installed at all."""

    def __init__(self, api_key: str, model: str = DEFAULT_ANTHROPIC_MODEL):
        try:
            import anthropic  # local import -- see class docstring
        except ImportError as exc:  # pragma: no cover -- exercised only if `anthropic` truly isn't installed
            raise LLMProviderError("anthropic_package_not_installed") from exc

        self.model_name = model
        self._client = anthropic.Anthropic(api_key=api_key)
        self._anthropic = anthropic

    def complete(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 512) -> str:
        try:
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as exc:  # noqa: BLE001 -- normalize every SDK/network failure into LLMProviderError
            # NEVER str(exc) here -- see module docstring's SECRET HANDLING note.
            raise LLMProviderError(f"anthropic_api_error:{type(exc).__name__}") from exc

        text_parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        text = "".join(text_parts).strip()
        if not text:
            raise LLMProviderError("anthropic_empty_response")
        return text


def get_llm_client() -> LLMClient:
    """Config-driven provider selection (brief section 6). Never logs or
    returns the API key itself."""
    provider = (settings.LLM_PROVIDER or "mock").strip().lower()

    if provider == "anthropic":
        if not settings.ANTHROPIC_API_KEY:
            log.warning("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set -- falling back to mock provider")
            return MockLLMClient()
        return AnthropicLLMClient(api_key=settings.ANTHROPIC_API_KEY, model=settings.ANTHROPIC_MODEL)

    if provider != "mock":
        log.warning("Unknown LLM_PROVIDER=%r -- defaulting to mock provider", provider)

    return MockLLMClient()
