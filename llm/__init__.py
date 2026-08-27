"""
LLM-assisted communication layer.

Three REQUIRED LLM jobs (project specification, verbatim terminology
preserved throughout this package -- see README §11), plus one
OPTIONAL Track-03 job:

  1. Outreach microcopy generation: per (failure bucket x customer segment x
     language preference), including a Hinglish variant as a PROMPT
     PARAMETER (not new infrastructure) -- see llm/service.py::generate_outreach_microcopy.
  2. Promise-to-pay parsing: free-text customer reply -> structured
     {date, confidence, channel} -- see llm/service.py::parse_promise_to_pay.
  3. Batch-level plain-English explanation for the final report --
     see llm/service.py::generate_batch_explanation.
  4. (Optional, Track-03) Voice recovery script generation: a spoken-register
     script for recovery/voice.py::VoiceRecoveryProvider -- see
     llm/service.py::generate_voice_script.

The LLM never classifies failures, never selects retry timing, never makes
compliance decisions, never decides escalation level, and never overrides
policy -- see policy/decision_engine_v4.py (policy-v4) and
policy/revenue_recovery_policy.py (Track-03) for where those deterministic
decisions are actually made. The LLM is strictly downstream of a decision
that has already been made without it, for every job including the
optional 4th.
"""
