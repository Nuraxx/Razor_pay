"""
Day-11 LLM-assisted communication layer.

Exactly three LLM jobs (project specification, verbatim terminology
preserved throughout this package -- see README "Day 11"):

  1. Outreach microcopy generation: per (failure bucket x customer segment x
     language preference), including a Hinglish variant as a PROMPT
     PARAMETER (not new infrastructure) -- see llm/service.py::generate_outreach_microcopy.
  2. Promise-to-pay parsing: free-text customer reply -> structured
     {date, confidence, channel} -- see llm/service.py::parse_promise_to_pay.
  3. Batch-level plain-English explanation for the final report --
     see llm/service.py::generate_batch_explanation.

These are the ONLY three LLM jobs in this project. The LLM never
classifies failures, never selects retry timing, never makes compliance
decisions, and never overrides policy -- see policy/decision_engine_v4.py
(Day 10) for where those deterministic decisions are actually made. The
LLM is strictly downstream of a policy decision that has already been
made without it.
"""
