"""
LLM provider abstraction.

Deliberately minimal -- no LangChain, no LangGraph, no agent framework, no
autonomous tool use (brief section 2). Four providers, selected by the
`LLM_PROVIDER` env var (`app/config.py::settings.LLM_PROVIDER`):

  "mock"      (default) -- deterministic, offline, makes zero network calls.
               Works with no API key at all. Used by every test in this
               project and by scripts/run_llm_demo.py by default.
  "anthropic" -- thin wrapper over the real `anthropic` SDK. Requires
               ANTHROPIC_API_KEY to be set.
  "gemini"    -- thin wrapper over the real `google-genai` SDK (Gemini
               Developer API). Requires GEMINI_API_KEY to be set. Model
               defaults to `gemini-3.6-flash` (GEMINI_MODEL) -- Google's
               originally-documented `gemini-2.5-flash` now returns HTTP 404
               ("no longer available to new users") for newly-created API
               keys; verify what your own key can call via
               `client.models.list()` before assuming any specific model ID.
  "ollama"    -- local inference via a locally-running Ollama server's HTTP
               API (no SDK, no API key -- see OllamaLLMClient below). Model
               defaults to `qwen3:14b` (OLLAMA_MODEL), server defaults to
               `http://localhost:11434` (OLLAMA_BASE_URL).

All four implement the same `LLMClient.complete(system_prompt, user_prompt) ->
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
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_OLLAMA_MODEL = "qwen3:14b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
MOCK_MODEL_NAME = "mock-llm-v1"

# HTTP-level request timeout for the Gemini SDK's underlying httpx client
# (milliseconds -- see google.genai.types.HttpOptions). Best-effort: the
# generic `except Exception` in GeminiLLMClient.complete() below is the real
# safety net for any request that hangs past this or otherwise never returns.
GEMINI_REQUEST_TIMEOUT_MS = 30_000

# Seconds. Local inference on a 14B model can be slow on the first call after
# the server has evicted the model from memory (observed ~50s reload cost
# separate from generation time itself) -- generous on purpose, since a slow
# local response is still strictly better than a false "provider unavailable"
# on a healthy but cold server. The generic `except Exception` in
# OllamaLLMClient.complete() below is the real safety net either way.
OLLAMA_REQUEST_TIMEOUT_SECONDS = 90.0


class LLMProviderError(Exception):
    """Raised for any provider-level failure -- timeout, API/SDK exception,
    empty response, provider unavailable/misconfigured. Never raised past
    llm/service.py; the policy layer never sees this exception at all."""


class LLMClient(abc.ABC):
    model_name: str
    provider_name: str  # "mock" | "anthropic" | "gemini" -- used by llm/service.py to label LLMResult.provider

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
    provider_name = "mock"

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

    provider_name = "anthropic"

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


class GeminiLLMClient(LLMClient):
    """Thin wrapper over the real `google-genai` SDK (Gemini Developer API).
    `google.genai` is imported lazily inside __init__ (not at module import
    time) so importing this module never requires the package to be
    installed at all -- same reasoning as AnthropicLLMClient above.

    Uses `response_mime_type="application/json"` (brief section 6: "use
    structured output ... where supported") rather than a per-task JSON
    schema -- `LLMClient.complete()` is task-agnostic by design (brief:
    "preserve the existing provider abstraction"), and per-task schema
    validation already happens one layer up in llm/service.py::_invoke.
    Coupling this module to llm/schemas.py's per-job schemas would violate
    that existing layering for no benefit that layer doesn't already provide.
    """

    provider_name = "gemini"

    def __init__(self, api_key: str, model: str = DEFAULT_GEMINI_MODEL):
        try:
            from google import genai  # local import -- see class docstring
        except ImportError as exc:  # pragma: no cover -- exercised only if `google-genai` truly isn't installed
            raise LLMProviderError("gemini_package_not_installed") from exc

        self.model_name = model
        self._client = genai.Client(api_key=api_key)
        self._genai = genai

    def complete(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 512) -> str:
        from google.genai import errors, types

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,  # near-deterministic (brief section 6: "keep output deterministic where practical")
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                    http_options=types.HttpOptions(timeout=GEMINI_REQUEST_TIMEOUT_MS),
                    # Gemini 3.x models think by default, and thinking tokens count
                    # against max_output_tokens -- observed consuming ~490 of a 512
                    # budget on this project's short structured-output prompts,
                    # truncating the actual JSON answer (finish_reason=MAX_TOKENS).
                    # thinking_budget=0 is REJECTED (400 INVALID_ARGUMENT) for this
                    # model generation -- thinking cannot be fully disabled, only
                    # bounded. "low" reliably leaves the JSON answer intact within
                    # the existing max_tokens budget (verified: ~110 thinking tokens
                    # instead of ~490, finish_reason=STOP).
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )
            text = (getattr(response, "text", None) or "").strip()
        except errors.APIError as exc:
            # NEVER exc.message here -- it can echo request/response content.
            # See module docstring's SECRET HANDLING note.
            code = getattr(exc, "code", None)
            if code == 429:
                raise LLMProviderError("gemini_rate_limited_or_quota_exhausted") from exc
            if code in (401, 403):
                raise LLMProviderError("gemini_authentication_failed") from exc
            raise LLMProviderError(f"gemini_api_error:{type(exc).__name__}:{code}") from exc
        except Exception as exc:  # noqa: BLE001 -- normalize every other SDK/network/timeout/safety-block failure
            raise LLMProviderError(f"gemini_api_error:{type(exc).__name__}") from exc

        if not text:
            raise LLMProviderError("gemini_empty_response")
        return text


class OllamaLLMClient(LLMClient):
    """Thin wrapper over a locally-running Ollama server's HTTP `/api/chat`
    endpoint -- no SDK, no API key, no external network call (everything
    stays on OLLAMA_BASE_URL, which defaults to localhost). `httpx` is
    already a direct dependency of this project (requirements.txt), so no
    new package is added for this provider.

    Uses `think: false` -- Qwen3 is a reasoning model that otherwise emits
    its chain-of-thought in a separate `message.thinking` field (verified
    directly against a running local server before writing this). Disabling
    it is both faster (observed ~3-4x fewer output tokens for these short
    structured-output prompts) and guarantees no reasoning text is ever
    read, logged, or stored -- `complete()` below only ever looks at
    `message.content`, never `message.thinking`, so there is nothing to
    strip even if a future model ignored the flag.

    Uses `format: "json"` (Ollama's structured-output constraint, brief
    section 6: "force JSON output where Ollama/Qwen3 supports it") --
    schema-level validation against each job's actual pydantic schema still
    happens one layer up in llm/service.py::_invoke, exactly as for every
    other provider.
    """

    provider_name = "ollama"

    def __init__(self, base_url: str = DEFAULT_OLLAMA_BASE_URL, model: str = DEFAULT_OLLAMA_MODEL):
        self.model_name = model
        self._base_url = (base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")

    def complete(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 512) -> str:
        import httpx

        try:
            response = httpx.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "format": "json",
                    "think": False,  # never let reasoning tokens reach message.content -- see class docstring
                    "options": {
                        "temperature": 0.1,  # near-deterministic, same rationale as GeminiLLMClient above
                        "num_predict": max_tokens,
                    },
                },
                timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS,
            )
        except httpx.TimeoutException as exc:
            raise LLMProviderError("ollama_timeout") from exc
        except httpx.ConnectError as exc:
            raise LLMProviderError("ollama_server_unavailable") from exc
        except Exception as exc:  # noqa: BLE001 -- normalize every other network failure
            raise LLMProviderError(f"ollama_request_error:{type(exc).__name__}") from exc

        if response.status_code == 404:
            # Ollama returns 404 for both "unknown route" and "model not
            # found on this server" -- the latter is the realistic case
            # here, since the route itself is fixed by this client.
            raise LLMProviderError("ollama_model_not_found")
        if response.status_code != 200:
            raise LLMProviderError(f"ollama_http_error:{response.status_code}")

        try:
            data = response.json()
        except Exception as exc:  # noqa: BLE001 -- malformed body from the server itself, not the model's JSON output
            raise LLMProviderError(f"ollama_response_parse_error:{type(exc).__name__}") from exc

        text = (data.get("message", {}).get("content") or "").strip()
        if not text:
            raise LLMProviderError("ollama_empty_response")
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

    if provider == "gemini":
        if not settings.GEMINI_API_KEY:
            log.warning("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set -- falling back to mock provider")
            return MockLLMClient()
        return GeminiLLMClient(api_key=settings.GEMINI_API_KEY, model=settings.GEMINI_MODEL)

    if provider == "ollama":
        # No API key needed -- reachability/model-availability failures are
        # caught inside OllamaLLMClient.complete() and turned into the
        # normal deterministic-fallback path by llm/service.py, exactly like
        # any other provider failure. No network probe here at selection time.
        return OllamaLLMClient(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_MODEL)

    if provider != "mock":
        log.warning("Unknown LLM_PROVIDER=%r -- defaulting to mock provider", provider)

    return MockLLMClient()
