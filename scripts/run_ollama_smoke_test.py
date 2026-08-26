"""
Real, local smoke test for the Ollama provider (llm/client.py::OllamaLLMClient).

Makes REAL calls to a locally-running Ollama server -- never to any external
provider. Verifies the server is reachable, the configured model is present,
then calls the EXISTING outreach_microcopy service path (llm/service.py)
exactly as the recovery orchestrator would.

Usage (from the project root):

    ./venv/bin/python scripts/run_ollama_smoke_test.py

Requires Ollama running locally (`ollama serve`) with OLLAMA_MODEL already
pulled (`ollama pull qwen3:14b`). Fails clearly, with no traceback noise, if
either precondition isn't met.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.config import settings
from llm.client import OllamaLLMClient
from llm.service import generate_outreach_microcopy


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def main() -> None:
    base_url = settings.OLLAMA_BASE_URL
    model = settings.OLLAMA_MODEL

    print(f"OLLAMA_BASE_URL = {base_url!r}")
    print(f"OLLAMA_MODEL    = {model!r}")

    print("\nChecking server reachability...")
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=10.0)
    except httpx.ConnectError:
        _fail(f"Could not reach Ollama server at {base_url}. Is `ollama serve` running?")
        return
    except Exception as exc:  # noqa: BLE001
        _fail(f"Unexpected error reaching Ollama server: {type(exc).__name__}")
        return

    if response.status_code != 200:
        _fail(f"Ollama server at {base_url} returned HTTP {response.status_code}.")
        return

    available_models = {m["name"] for m in response.json().get("models", [])}
    print(f"Server reachable. Models available: {sorted(available_models)}")

    if model not in available_models:
        _fail(f"Model {model!r} is not pulled on this Ollama server. Run: ollama pull {model}")
        return

    print(f"Model {model!r} confirmed available.\n")

    print("Calling the EXISTING outreach_microcopy service path (llm/service.py)...")
    client = OllamaLLMClient(base_url=base_url, model=model)
    result = generate_outreach_microcopy(
        failure_bucket="retryable_soft",
        customer_segment="mid",
        language="en",
        will_retry=True,
        retry_window_description="tomorrow morning",
        amount_rupees=499.0,
        client=client,
    )

    print("\n--- Result ---")
    print(f"provider: {result.provider}")
    print(f"model:    {result.model_name}")
    print(f"success:  {result.success}")
    if result.success:
        print(f"generated message: {result.structured_result['message_text']}")
    else:
        print(f"error_type: {result.error_type}")
        print(f"fallback message: {result.structured_result['message_text']}")

    if not result.success:
        _fail("Ollama call did not succeed -- see error_type above.")
        return

    print("\nOK: local Ollama (Qwen3) provider is working end-to-end through the existing service layer.")


if __name__ == "__main__":
    main()
