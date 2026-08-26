"""
Real, local three-job test for the Ollama provider -- runs the EXISTING
service-layer functions (llm/service.py) for all 3 required LLM jobs against
a locally-running Ollama server (Qwen3). No Ollama-specific service
functions are created; this script only supplies an OllamaLLMClient as the
`client=` argument to the same public functions every other provider uses.

Usage (from the project root):

    ./venv/bin/python scripts/run_ollama_three_job_test.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from llm.client import OllamaLLMClient
from llm.service import (
    generate_batch_explanation,
    generate_outreach_microcopy,
    parse_promise_to_pay,
)


def main() -> None:
    client = OllamaLLMClient(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_MODEL)
    all_succeeded = True

    result = generate_outreach_microcopy(
        failure_bucket="retryable_soft", customer_segment="mid", language="en",
        will_retry=True, retry_window_description="tomorrow morning", amount_rupees=499.0,
        client=client,
    )
    print(f"task={result.task_name} provider={result.provider} model={result.model_name} success={result.success}")
    print(f"result={result.structured_result}")
    print()
    all_succeeded &= result.success

    result = parse_promise_to_pay(
        customer_reply_text="I'll pay Friday when salary comes, via UPI",
        today=date(2026, 8, 24),
        client=client,
    )
    print(f"task={result.task_name} provider={result.provider} model={result.model_name} success={result.success}")
    print(f"result={result.structured_result}")
    print()
    all_succeeded &= result.success

    result = generate_batch_explanation(
        report_summary={
            "label": "SYNTHETIC COUNTERFACTUAL EVALUATION -- ollama three-job smoke test",
            "latent_economic": {"fixed_retry": {"total_latent_value_rs": 1000.0}, "oracle_policy": {"total_latent_value_rs": 1200.0}},
        },
        client=client,
    )
    print(f"task={result.task_name} provider={result.provider} model={result.model_name} success={result.success}")
    print(f"result={result.structured_result}")
    print()
    all_succeeded &= result.success

    if not all_succeeded:
        print("FAIL: at least one of the 3 required LLM jobs did not succeed against the local Ollama provider.")
        sys.exit(1)

    print("OK: all 3 required LLM jobs succeeded against the local Ollama (Qwen3) provider.")


if __name__ == "__main__":
    main()
