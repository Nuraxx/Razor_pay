"""
Versioned prompts -- one system/user prompt pair per LLM job, plus
the deterministic mock-provider response router.

Every prompt version below is a frozen string constant. Changing a prompt's
WORDING requires bumping its version constant (e.g. `..._v1` -> `..._v2`);
never edit a shipped version's text in place, so `LLMResult.prompt_version`
recorded in old audit rows stays meaningful.

Shared guardrail language (brief section 4) is injected into every system
prompt via `_SHARED_GUARDRAILS` rather than duplicated three times.
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta

_SHARED_GUARDRAILS = """
Hard rules, no exceptions:
- Do NOT invent payment facts. Use only the facts supplied in the context below.
- Do NOT invent customer information not present in the context.
- Only use the supplied context -- never assume additional facts about the customer, the payment, or the company.
- Never claim a payment succeeded, failed, or was retried unless the supplied data says so explicitly.
- You do NOT decide whether the payment is retryable, when to retry it, or whether compliance allows any action -- those decisions have already been made upstream, deterministically, before you were called. Do not second-guess or restate them as if they were open questions.
- Return ONLY a single structured JSON object matching the schema below. No prose before or after it, no markdown code fences, no explanations outside the JSON.
""".strip()

MOCK_TASK_MARKER_PREFIX = "TASK_MARKER:"  # embedded in every user prompt so llm/client.py's mock provider can route deterministically

# ---------------------------------------------------------------------------
# Job 1: outreach microcopy generation
#   per (failure bucket x customer segment x language preference)
#   -- Hinglish is a PROMPT PARAMETER value of `language`, not separate infra.
# ---------------------------------------------------------------------------

OUTREACH_MICROCOPY_PROMPT_VERSION = "outreach_microcopy_v1"

OUTREACH_MICROCOPY_SYSTEM_PROMPT = f"""
You are the outreach-microcopy writer for a payment-recovery system. Your
only job is to write ONE short piece of customer-facing text (an SMS/
WhatsApp/email-style message, no channel-specific formatting) telling the
customer about a recent payment failure and what happens next, in the
requested language.

{_SHARED_GUARDRAILS}

Allowed values:
- language: one of "en" (English), "hi" (Hindi, Devanagari script), "hinglish" (Hindi written in Latin script, mixed with English, casual tone).
- failure_bucket: echo back the value given to you in the context; do not translate or alter it.
- customer_segment: echo back the value given to you in the context; do not translate or alter it.

Output schema (return exactly this JSON shape):
{{"message_text": "<string, 1-800 chars>", "language": "<en|hi|hinglish>", "failure_bucket": "<string>", "customer_segment": "<string>"}}

Fallback behavior: if you cannot safely write a message from the given
context, still return valid JSON with a short, neutral, non-committal
message_text rather than omitting the field or adding prose outside the JSON.
""".strip()


def outreach_microcopy_user_prompt(
    *, failure_bucket: str, customer_segment: str, language: str,
    will_retry: bool, retry_window_description: str | None, amount_rupees: float,
) -> str:
    """`retry_window_description` is a plain-language phrase already computed
    upstream by the policy layer (e.g. "in the next day" or "around your next
    payday") -- never a raw candidate_datetime, and never invented here."""
    context = {
        "failure_bucket": failure_bucket,
        "customer_segment": customer_segment,
        "language": language,
        "will_retry": will_retry,
        "retry_window_description": retry_window_description,
        "amount_rupees": round(amount_rupees, 2),
    }
    return (
        f"{MOCK_TASK_MARKER_PREFIX}outreach_microcopy\n\n"
        "Write the outreach microcopy for this context (JSON):\n"
        f"{json.dumps(context, sort_keys=True)}"
    )


# ---------------------------------------------------------------------------
# Job 2: promise-to-pay parsing
#   free-text customer reply -> {date, confidence, channel}
# ---------------------------------------------------------------------------

PROMISE_TO_PAY_PROMPT_VERSION = "promise_to_pay_parse_v1"

PROMISE_TO_PAY_SYSTEM_PROMPT = f"""
You are a structured-data extractor. A customer replied to a payment-recovery
message with free text. Extract a promise-to-pay object from it -- nothing more.

{_SHARED_GUARDRAILS}

Allowed values:
- date: an ISO-8601 date "YYYY-MM-DD" if the customer's reply names or clearly implies a specific date, else null. Never guess a date the text does not support. If the text names a weekday ("Friday") or relative day ("tomorrow") without a specific date, resolve it against the "today" date given in the context.
- confidence: a float from 0.0 to 1.0 -- YOUR confidence that this is a genuine, correctly-parsed promise to pay (not the customer's own stated certainty).
- channel: one of "credit_card", "debit_card", "upi_autopay", "netbanking", "unspecified" -- the payment method the customer mentions, if any; "unspecified" if the text does not mention one. Never guess a channel the text does not support.

Output schema (return exactly this JSON shape):
{{"date": "<YYYY-MM-DD or null>", "confidence": <float 0.0-1.0>, "channel": "<credit_card|debit_card|upi_autopay|netbanking|unspecified>"}}

Fallback behavior: if the text contains no discernible promise to pay at
all, return {{"date": null, "confidence": 0.0, "channel": "unspecified"}} --
do not fabricate a date or channel to fill the schema.
""".strip()


def promise_to_pay_user_prompt(*, customer_reply_text: str, today: date) -> str:
    context = {"customer_reply_text": customer_reply_text, "today": today.isoformat()}
    return (
        f"{MOCK_TASK_MARKER_PREFIX}promise_to_pay_parse\n\n"
        "Extract the promise-to-pay object from this context (JSON):\n"
        f"{json.dumps(context, sort_keys=True)}"
    )


# ---------------------------------------------------------------------------
# Job 3: plain-English batch-level explanation for the final report
# ---------------------------------------------------------------------------

BATCH_EXPLANATION_PROMPT_VERSION = "batch_explanation_v1"

BATCH_EXPLANATION_SYSTEM_PROMPT = f"""
You are writing the plain-English summary paragraph for an internal batch
evaluation report. Your audience is a non-technical stakeholder. The report
data given to you is an ALREADY-COMPUTED, ALREADY-LABELED-SYNTHETIC
evaluation summary -- you are explaining numbers that already exist, not
computing or deciding anything.

{_SHARED_GUARDRAILS}
- The data you are given is a SYNTHETIC COUNTERFACTUAL EVALUATION. You MUST
  make this explicit in your explanation (state plainly that the numbers
  come from a synthetic simulation, not real production outcomes) and must
  NEVER describe any figure as real, measured, or production performance.
- Do not recommend a course of action beyond what the report data itself
  already shows; describe the numbers, do not editorialize about policy.

Output schema (return exactly this JSON shape):
{{"explanation_text": "<string, 1-3000 chars>"}}

Fallback behavior: if the report data is insufficient to write a
meaningful explanation, return a short JSON explanation_text saying so
explicitly, rather than fabricating figures.
""".strip()


def batch_explanation_user_prompt(*, report_summary: dict) -> str:
    return (
        f"{MOCK_TASK_MARKER_PREFIX}batch_explanation\n\n"
        "Write the plain-English explanation for this batch report data (JSON):\n"
        f"{json.dumps(report_summary, sort_keys=True, default=str)}"
    )


# ---------------------------------------------------------------------------
# Job 4 (optional, Track-03): voice recovery script generation
#   same facts-only, LLM-generates-copy-only pattern as Job 1, but for a
#   spoken-register script instead of an SMS/WhatsApp text.
# ---------------------------------------------------------------------------

VOICE_SCRIPT_PROMPT_VERSION = "voice_script_generation_v1"

VOICE_SCRIPT_SYSTEM_PROMPT = f"""
You are the voice-call script writer for a payment-recovery system. Your
only job is to write ONE short SPOKEN-REGISTER script (what an automated or
human caller would say out loud) telling the customer about a recent
payment failure and what happens next, in the requested language.

{_SHARED_GUARDRAILS}
- This is a SPOKEN script, not a text message -- write it as natural spoken
  sentences (no bullet points, no emojis, no markdown), suitable for
  text-to-speech playback or a human reading it aloud.
- Never invent a payment status, retry date, refund, discount, settlement,
  or payment link -- use only the facts supplied in the context below,
  exactly like the outreach-microcopy job.

Allowed values:
- language: one of "en", "hi", "hinglish" -- same vocabulary as the outreach-microcopy job.
- failure_bucket / customer_segment: echo back the values given to you in the context; do not translate or alter them.

Output schema (return exactly this JSON shape):
{{"script_text": "<string, 1-1200 chars>", "estimated_duration_seconds": <float 0-180>, "requires_callback_offer": <true|false>, "language": "<en|hi|hinglish>", "failure_bucket": "<string>", "customer_segment": "<string>"}}

Fallback behavior: if you cannot safely write a script from the given
context, still return valid JSON with a short, neutral, non-committal
script_text rather than omitting the field or adding prose outside the JSON.
""".strip()


def voice_script_user_prompt(
    *, failure_bucket: str, customer_segment: str, language: str,
    will_retry: bool, retry_window_description: str | None, amount_rupees: float,
) -> str:
    """Same context shape as outreach_microcopy_user_prompt -- deliberately
    identical inputs, different output register."""
    context = {
        "failure_bucket": failure_bucket,
        "customer_segment": customer_segment,
        "language": language,
        "will_retry": will_retry,
        "retry_window_description": retry_window_description,
        "amount_rupees": round(amount_rupees, 2),
    }
    return (
        f"{MOCK_TASK_MARKER_PREFIX}voice_script_generation\n\n"
        "Write the voice recovery script for this context (JSON):\n"
        f"{json.dumps(context, sort_keys=True)}"
    )


# ---------------------------------------------------------------------------
# Deterministic mock-provider response router (llm/client.py::MockLLMClient)
# ---------------------------------------------------------------------------

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

_MOCK_MICROCOPY = {
    ("en", True): "Your recent payment did not go through. We will automatically retry it {window}.",
    ("en", False): "Your recent payment did not go through. Please check your payment method when you get a chance.",
    ("hi", True): "aapka haal hi ka payment poora nahi ho paaya. hum {window} dobara koshish karenge.",
    ("hi", False): "aapka haal hi ka payment poora nahi ho paaya. kripya apna payment tarika check karein.",
    ("hinglish", True): "Aapka payment abhi complete nahi hua. Hum {window} automatically retry karenge.",
    ("hinglish", False): "Aapka payment abhi complete nahi hua. Please apna payment method check kar lijiye.",
}


def _extract_json_field(user_prompt: str, key: str):
    match = re.search(r'"' + re.escape(key) + r'":\s*("(?:[^"\\]|\\.)*"|null|true|false|[0-9.]+)', user_prompt)
    if not match:
        return None
    raw = match.group(1)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def mock_response_for_prompt(user_prompt: str) -> str:
    """Deterministic, offline, keyword/regex-based routing -- makes NO
    network calls (brief section 6). Same input always produces the same
    output; different input can plausibly produce different (but always
    schema-valid) output, which is what tests/test_llm.py checks for the
    promise-to-pay job's weekday parsing."""
    if f"{MOCK_TASK_MARKER_PREFIX}outreach_microcopy" in user_prompt:
        language = _extract_json_field(user_prompt, "language") or "en"
        will_retry = bool(_extract_json_field(user_prompt, "will_retry"))
        window = _extract_json_field(user_prompt, "retry_window_description") or "soon"
        failure_bucket = _extract_json_field(user_prompt, "failure_bucket") or ""
        customer_segment = _extract_json_field(user_prompt, "customer_segment") or ""
        template = _MOCK_MICROCOPY.get((language, will_retry), _MOCK_MICROCOPY[("en", will_retry)])
        message = template.format(window=window)
        return json.dumps({"message_text": message, "language": language, "failure_bucket": failure_bucket, "customer_segment": customer_segment})

    if f"{MOCK_TASK_MARKER_PREFIX}promise_to_pay_parse" in user_prompt:
        text = (_extract_json_field(user_prompt, "customer_reply_text") or "").lower()
        today_str = _extract_json_field(user_prompt, "today")
        today_date = date.fromisoformat(today_str) if today_str else date.today()

        resolved_date = None
        confidence = 0.0
        if "tomorrow" in text:
            resolved_date = today_date + timedelta(days=1)
            confidence = 0.7
        else:
            for i, wd in enumerate(_WEEKDAYS):
                if wd in text:
                    days_ahead = (i - today_date.weekday()) % 7
                    days_ahead = days_ahead or 7  # "Friday" means the NEXT Friday, not today, even if today is Friday
                    resolved_date = today_date + timedelta(days=days_ahead)
                    confidence = 0.7
                    break
        if resolved_date is not None and any(word in text for word in ("pay", "salary", "will")):
            confidence = 0.85

        channel = "unspecified"
        for candidate in ("upi_autopay", "upi", "netbanking", "net banking", "credit_card", "credit card", "debit_card", "debit card"):
            if candidate.replace("_", " ") in text or candidate in text:
                channel = "upi_autopay" if "upi" in candidate else candidate.replace(" ", "_")
                break

        return json.dumps({"date": resolved_date.isoformat() if resolved_date else None, "confidence": confidence, "channel": channel})

    if f"{MOCK_TASK_MARKER_PREFIX}voice_script_generation" in user_prompt:
        language = _extract_json_field(user_prompt, "language") or "en"
        will_retry = bool(_extract_json_field(user_prompt, "will_retry"))
        window = _extract_json_field(user_prompt, "retry_window_description") or "soon"
        failure_bucket = _extract_json_field(user_prompt, "failure_bucket") or ""
        customer_segment = _extract_json_field(user_prompt, "customer_segment") or ""
        template = _MOCK_MICROCOPY.get((language, will_retry), _MOCK_MICROCOPY[("en", will_retry)])
        script_text = "Hello. " + template.format(window=window) + " Thank you."
        return json.dumps({
            "script_text": script_text, "estimated_duration_seconds": round(len(script_text) / 15.0, 1),
            "requires_callback_offer": not will_retry, "language": language,
            "failure_bucket": failure_bucket, "customer_segment": customer_segment,
        })

    if f"{MOCK_TASK_MARKER_PREFIX}batch_explanation" in user_prompt:
        report = _extract_json_field(user_prompt, "label") or ""
        is_synthetic = "SYNTHETIC" in str(report) or "SYNTHETIC" in user_prompt
        prefix = "This is a SYNTHETIC COUNTERFACTUAL EVALUATION, not a measurement of real production performance. " if is_synthetic else ""
        explanation = prefix + "The batch report summarizes policy comparison results across the evaluated events; see the structured report data for exact figures."
        return json.dumps({"explanation_text": explanation})

    # Unknown/unmarked prompt -- return syntactically valid but empty-ish
    # JSON so a caller's schema validation fails cleanly rather than this
    # function raising (mirrors "empty response" as a real provider failure mode).
    return json.dumps({})
