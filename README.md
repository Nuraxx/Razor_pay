# Adaptive Payment Recovery Agent

**Razorpay AI Buildathon — Track 3: AI Revenue Recovery**

**Current status:** 1019 tests collected, all passing (0 skipped when
`model/artifacts/` already has a manually-trained calibrated model, as in
this working copy; 3 honest `pytest.skip`s for that specific artifact on a
genuinely fresh clone that hasn't run `python -m model.train` yet — see §8/§15)
(`pytest tests -q`) as of the BUG-1/2/3/4 pre-submission audit fix pass —
this is a point-in-time figure, not a claim this document keeps itself in
sync with; verify with the command above if it matters for your purposes.

An offline-verifiable prototype that replaces Razorpay Subscriptions' blind
fixed-interval retry (retry once/day for 3 days, regardless of *why* a charge
failed) with a per-failure decision: classify the failure reason
deterministically, score candidate retry times with a calibrated model,
enforce a deterministic compliance gate, generate outreach copy and parse
promise-to-pay replies with an LLM (3 required jobs plus one optional
voice-script extension, never a decision-maker), and report recovered-₹
against Razorpay's own real baseline on a held-out synthetic batch.

Scope, per the original project specification
(`~/Downloads/razorpay-track3-project-specification.md`): **Razorpay
Subscriptions only, `insufficient_fund` decline reason only.** That original
scope is still fully intact and untouched — the system was later extended
("Track-03") into a broader revenue-recovery platform covering checkout
abandonment, mandate retry sequencing, B2B receivables, a promise-to-pay
lifecycle with an automatic broken-promise scheduler, Hinglish/voice
outreach, and a real Razorpay `payment.captured` reconciliation loop that
closes a recovery case only on an authoritative payment confirmation — see
[§3a](#3a-track-03-extension-scope) for exactly what that added. Everything
below this line is described as a single integrated system, by what each
part does today, not as a chronological changelog.

---

## 1. Product overview

The system watches for Razorpay Subscriptions `payment.failed` webhooks,
buckets the failure reason with a deterministic rule table, and — for the
`insufficient_fund` (retryable-soft) bucket specifically — picks the best of
five candidate retry windows using a calibrated regression model trained on
a synthetic, archetype-driven dataset. A deterministic compliance gate
(contact caps, opt-out/cancellation respect, required-field checks) sits
between that decision and any simulated payment retry or outreach message.
Three narrowly-scoped LLM jobs generate outreach microcopy, parse a
customer's free-text promise-to-pay reply, and narrate a batch-level report
— strictly downstream of the payment decision, never able to change it.
Every stage writes to a single `audit_log` table. A Streamlit dashboard
visualizes both the frozen synthetic-benchmark evaluation and a live
operational run of the real orchestrator, always clearly labeled which is
which.

**No live Razorpay payment retry and no live customer message is ever sent.**
Payment actions are recorded intents; outreach is logged as a structured
"would have sent" record.

## 2. Problem

Razorpay Subscriptions' documented retry policy re-attempts a failed charge
once a day for three days, identically, regardless of cause. A customer who
hit a momentary insufficient-funds decline is treated the same as one whose
card is dead or who is actively cancelling. Razorpay earns its transaction
fee only on a successful charge, so every unrecovered failure is revenue
neither the merchant nor Razorpay ever collects. See the original
specification (`~/Downloads/razorpay-track3-project-specification.md`,
sections 1–6) for the full evidence base (RBI's 2026 e-mandate framework,
NACH bounce-rate data, Razorpay's own fee-on-success mechanics) behind this
framing.

## 3. Scope

- **In scope:** Razorpay Subscriptions, `insufficient_fund` decline reason
  only. Webhook ingestion is built and tested against real Razorpay Test
  Mode webhook payloads and HMAC signatures.
- **Out of scope, by design, throughout:** UPI Autopay, checkout/one-time
  payments, invoices, any live payment retry call, any live WhatsApp/SMS/
  voice send. Outreach is simulated and logged, never dispatched.
- **Data:** entirely synthetic (archetype-generated, probabilistic labels),
  clearly and consistently labeled as such everywhere it appears in reports
  and the dashboard — see [§16 Evaluation](#16-evaluation) and
  [§19 Known limitations](#19-known-limitations).

## 3a. Track-03 extension scope

Everything in §3 above is the ORIGINAL project spec's scope and remains
completely unchanged and untouched. On top of it, the system was later
extended into a broader revenue-recovery platform, reusing the same
recovery engine, compliance gate, audit trail, and dashboard rather than
building six separate demos:

- **Checkout abandonment**, **mandate retry sequencing**, and **B2B
  receivables** recovery (`policy/checkout_rules.py`, `policy/mandate_rules.py`,
  `policy/receivables_rules.py`), each with its own deterministic rule
  module, dispatched through one unified policy entry point
  (`policy/revenue_recovery_policy.py`) and one unified, generalized
  compliance gate (`policy/compliance_v2.py`, adding a third `HUMAN_REVIEW`
  verdict for cases like a disputed invoice) alongside the original,
  byte-for-byte-unchanged `policy/compliance.py`.
- **Promise-to-pay lifecycle**: a new `PROMISED → FULFILLED/BROKEN/CANCELLED`
  dimension (`recovery/promise_lifecycle.py`) layered on top of the original
  promise *validation* status (`policy/promise_to_pay.py`, unchanged). An
  automatic, in-process background scheduler (`recovery/scheduler.py`, no
  Celery/Redis) periodically detects broken promises and routes them back
  through the same recovery engine — real, tested end to end, but explicitly
  **not production-grade** (a single-process `asyncio` loop, no distributed
  locking).
- **Hinglish outreach** (a `language` prompt parameter, no new
  infrastructure) and an **optional 4th LLM job**, `voice_script_generation`
  (§5), producing a script for a `MockVoiceProvider` — no real telephony
  integration exists; no voice call is ever actually placed.
- **Payment recovery confirmation closed loop**
  (`recovery/payment_reconciliation.py`): a real Razorpay `payment.captured`
  webhook can now reconcile against a PENDING recovery case and mark it
  `RECOVERED`/`PARTIALLY_RECOVERED` with the authoritative confirmed amount
  — the only thing that is ever allowed to do so. A scheduled retry, a sent
  message, an LLM result, or a customer promise never fabricates a
  recovered outcome.

All of the above is additive: the original `payment_failed` webhook →
classify → decide → compliance → recover pipeline (§4 below) is unchanged
and independently tested, and every new domain has its own test file under
`tests/`.

## 3b. Gap analysis vs. Razorpay's own existing recovery products

Razorpay already ships real recovery tooling this project does not attempt
to replace: Payment Links Reminders, the Subscriptions core T+3 retry, UPI
Autopay's Intelligent Retry Engine, Failed Payment Recovery, and — most
relevant here — the 2026 **Agent Studio "Subscription Recovery"** and
"Abandoned Cart Conversion" agents (built on Anthropic's Claude Agent SDK
per Razorpay's own product page). This project is positioned as the
**orchestration/audit layer that could sit in front of** those agents, not
a competing rebuild of them. Four specific gaps motivate that framing,
none of which are met by Razorpay's own published product descriptions as
of this writing:

1. **No cross-surface orchestration** — every Razorpay recovery tool is
   scoped to its own product line; nothing publicly described shares one
   customer-state layer across Payment Links, Subscriptions, UPI Autopay,
   and the newer agents the way this project's single `RecoveryOutcome`/
   audit trail does across all of its own domains.
2. **No promise-to-pay object** — no Razorpay product describes capturing
   a customer's stated commitment ("I'll pay Friday") as a structured,
   dated, trackable record distinct from a fixed retry timer.
3. **No published measurement layer** — Razorpay's own recovery-product
   copy is qualitative ("protect your revenue"); none publish a
   ₹-recovered / recovery-rate / cost-per-recovery methodology of the kind
   in [§16](#16-evaluation).
4. **No disclosed, demoable compliance gate with its own failing-grade
   metric** — this project's compliance gate can visibly *refuse* an
   action (contact-hours, attempt caps, opt-out) and is itself measured via
   an unnecessary-intervention rate, rather than being an internal,
   undisclosed implementation detail.

This is a narrow, precise claim, not a category-level one: Razorpay's own
Agent Studio agents may already address some of this in ways not disclosed
publicly. See the original specification's Section 5 for the full sourcing
behind each gap.

## 3c. Unified ML generalization

On top of §3a's rule-based Track-03 dispatch, a single ML model
(`model/unified_model.py`, `model_version="unified_catboost_v1"`, a real
`catboost.CatBoostClassifier` — not a heuristic, and not the
sklearn-LogisticRegression placeholder an earlier pass had wired in under
that same name) now scores candidate interventions across **all five**
revenue-risk domains from one shared feature schema and one shared training
pipeline:

- **Domains:** `payment_failed` (Payment Link / no-subscription only — a
  genuine subscription `payment_failed` keeps using the original Model B
  above, unmodified), `checkout_abandoned`, `mandate_failed`,
  `receivable_overdue`, `promise_to_pay_broken`.
- **Candidate vocabularies are the SAME strings each domain's existing rule
  module already used** (`policy/checkout_rules.py`,
  `policy/mandate_rules.py`, `policy/receivables_rules.py`,
  `policy/promise_broken_rules.py`, `policy/one_time_payment_rules.py`) —
  e.g. `attempt_1`/`attempt_2`/`final_attempt` for mandates,
  `friendly_reminder`/`payment_request`/`promise_to_pay_request`/`escalation`
  for receivables. An earlier pass had invented a parallel, unrelated
  vocabulary (e.g. `retry_1_day` for Payment Links, which falsely implied an
  automatic retry Razorpay cannot actually perform for a one-time payment);
  that was a real, fixed bug, not a design choice.
- **Policy boundary:** the rule-based decider for each domain is always
  computed first and is the authoritative **eligibility gate**
  (opted-out/unmapped/disputed/terminal/not-yet-actionable). The unified
  model is *also* always consulted (when its artifact is loaded) —
  "should ML evaluate this event" and "should policy act on ML's
  recommendation" are deliberately two different questions
  (`policy/revenue_recovery_policy.py::decide_for_revenue_risk_event`).
  Three distinct, dashboard-visible outcomes result:
  - **ML USED** — the eligibility gate found an action warranted; ML's
    top-scored candidate is the final decision (`decision_source="ml_unified_v1"`).
  - **ML CONSULTED, overridden by policy** — ML ran and produced a real
    score, but the eligibility gate (or a mandatory human-review
    escalation, e.g. a disputed receivable) is what actually won; ML's
    recommendation/score is still recorded in `decision_reason`
    (`ml_consulted=True ml_recommendation=... ml_score=...`) and in the
    `predicted_recovery_probability`/`model_version` columns, never
    silently discarded.
  - **ML FALLBACK** — the artifact genuinely isn't loaded (missing/corrupt);
    the deterministic rule decider decides alone, `ml_consulted=False`.
- **Training:** `./venv/bin/python -m model.train_unified_model` — a
  synthetic dataset (`model/unified_model.py::_make_training_data`, all 5
  domains, a genuine per-entity customer-segment × candidate-urgency
  interaction so the "best" candidate actually varies per entity, not just
  per domain) split 70/15/15 **at the entity level, independently per
  domain** (`_entity_level_split` — no entity ever appears in two splits,
  verified by test), then fit on TRAIN only (validation used for CatBoost
  early stopping; TEST touched only for the metrics reported below and in
  `model/reports/unified_model_training_report.json`).
- **Artifact:** `model/artifacts/unified_model.joblib`, loaded through one
  process-wide cached function, `model/unified_model.py::get_live_unified_model()`
  — the SAME function `app/main.py` (webhook handler + startup), the promise
  sweep (`recovery/scheduler.py`), and the demo generator
  (`recovery/demo_generator.py`) all call; there is no second, parallel
  inference implementation anywhere. A startup log line
  (`Unified ML model loaded: model=unified_catboost_v1 artifact=...`) or a
  fallback warning if the artifact is missing confirms which happened.
- **Held-out evaluation:** `./venv/bin/python -m evaluation.evaluate_unified_model`
  — see the new subsection under [§16](#16-evaluation). Honest headline: a
  **marginal, mixed** lift over a naive fixed-candidate baseline (+0.4%
  overall net value; better in 1 of 4 multi-candidate domains, worse in 2,
  flat in 1) — this is disclosed, not hidden; see
  [§19](#19-known-limitations).

## 4. Architecture

```
Razorpay Test Mode ──(webhook, HMAC-signed)──▶ Webhook Receiver (FastAPI, app/main.py)
                                                        │
                                                        ▼
                                     raw_events store (SQLite, idempotent on x-razorpay-event-id)
                                                        │
                                     recovery/webhook_pipeline.py::process_raw_event (automatic, FIX #2)
                                                        ▼
                                    Deterministic Classifier (classification/rules.py)
                                                        │
                                                        ▼
                                            failure_events (bucketed)
                                                        │
                         ┌──────────────── recovery/orchestrator.py::orchestrate_recovery ───────────────┐
                         │                                                                                │
                         ▼                                                                                ▼
             Model B: candidate-time value regressor              Compliance Gate (policy/compliance.py,
             (policy/decision_engine_v4.py, CatBoost,               deterministic: contact caps, opt-out,
              rule-based fallback tier, NO_ACTION safety net)        required fields)
                         │                                                                                │
                         │            Promise-to-pay override (recovery/promise_service.py, FIX #1):      │
                         │            a customer reply, parsed+validated, may retime (never bypass)        │
                         │            the candidate above before it reaches compliance                    │
                         └──────────────────────────────────┬─────────────────────────────────────────────┘
                                                              ▼
                                        Payment action (recorded intent only — no live Razorpay call)
                                                              │
                                                              ▼
                                LLM layer (llm/service.py — Claude or offline mock): outreach copy
                                  (including a hard_decline payment-method-update nudge, FIX #3),
                                       promise-to-pay reply parsing, batch narration
                                                              │
                                                              ▼
                                        audit_log (every stage, every actor, no secrets)
                                                              │
                                                              ▼
                                   Streamlit dashboard (ui/) — synthetic benchmark + live operational view
```

**Architectural note (updated in the FIX pass):** the live FastAPI webhook
endpoint (`app/main.py`) verifies the signature, checks idempotency, stores
the raw event, and — for a `payment.failed` event carrying a subscription —
now continues automatically into `recovery/webhook_pipeline.py::process_raw_event`
(classification → full orchestration) in the same request, right after the
raw event is durably committed. A downstream failure can never un-store an
already-verified webhook delivery: the response honestly reports
`stored; orchestration=<outcome>`, and `scripts/reprocess_raw_events.py`
remains available to safely re-run classification+orchestration over any
raw event whose automatic pass failed or that was stored before this wiring
existed (every stage is independently idempotent). See
[§17](#17-failure-handling) and [§19](#19-known-limitations) for the exact
boundary cases this was verified against.

**The diagram above is the ORIGINAL subscription-`payment_failed`-only
path.** The Track-03 revenue-risk domains (§3a) — checkout_abandoned,
mandate_failed, receivable_overdue, promise_to_pay_broken, and Payment-Link
`payment_failed_no_subscription` — run through a parallel, equally-real
path that reuses the same webhook receiver, audit log, and dashboard but a
different orchestrator and policy dispatcher:

```
raw_events (same table, same idempotency)
        │
recovery/webhook_pipeline.py::process_raw_event
        │  (subscription_id present -> diagram above, unchanged)
        │  (subscription_id ABSENT but payment_id+amount present ->)
        ▼
RevenueRiskEvent (payment_failed_no_subscription / checkout_abandoned /
                  mandate_failed / receivable_overdue / promise_to_pay_broken)
        │
recovery/revenue_orchestrator.py::orchestrate_revenue_event
        │
policy/revenue_recovery_policy.py::decide_for_revenue_risk_event
        │  rule-based eligibility decider (always) + unified ML (always, §3c)
        ▼
policy/compliance_v2.py (ALLOWED / BLOCKED / HUMAN_REVIEW)
        │
llm/service.py (Ollama/mock/etc — communication only, ONLY if allowed)
        │
audit_log + RecoveryOutcome (PENDING until an authoritative payment.captured)
```

## 4a. Live vs. legacy: which code actually runs

Final pre-submission audit finding: this repository accumulated multiple
generations of model/policy/decision-engine code across its development
history. Every one of those still exists (nothing was deleted without proof
it was obsolete), but only a specific subset is on the LIVE call path a real
webhook actually walks through today. This section is the single source of
truth for that distinction — no reviewer should have to guess between
"v2 / v4 / Model B / unified / latent-target / old classifier / old policy."

**LIVE SUBSCRIPTION PATH** (`payment_failed` / `subscription_payment_failed`
— a subscription-linked charge):
- Entrypoint: `app/main.py`'s `/webhook/razorpay` → `recovery/webhook_pipeline.py::process_raw_event`
  → `recovery/live_feature_enrichment.py::build_live_features` (honestly
  assembles whatever subset of Model B's features a real event actually has
  — see the BUG-4 note below) → `recovery/orchestrator.py::orchestrate_recovery`
- Model: **Model B** (`model/train_latent_target_model.py`, a CatBoost
  regressor predicting `expected_recovery_value_latent` directly), loaded via
  `policy/decision_engine.py::_load_model_safely` — **in practice, Model B is
  never actually invoked for a genuine live webhook today** (see below);
  every real subscription decision is currently made by the rule-based
  fallback tier
- Policy: **`policy-v4`** (`policy/decision_engine_v4.py::decide_for_failure_event_engine_v4`)
  — margin-gated fallback between Model B and `policy/baselines.py`'s
  rule-based candidate; see [§16b](#16b-economic-correction-subscription-decision-policy)
  for the corrected default configuration

  **Why Model B never actually runs live (pre-submission audit, BUG-4 fix):**
  Model B requires all 12 keys in `policy/decision_engine.py::EVENT_FEATURE_KEYS`
  to be present, or its own `_predict_recovery_values` check fails closed to
  the rule-based tier (unmodified — see that module). `recovery/live_feature_enrichment.py`
  is the boundary that honestly assembles a live event's feature vector, and
  classifies every one of the 12 keys by its real, verified source:

  | Feature | Source | Live today? |
  |---|---|---|
  | `day_of_month` | `DERIVED_FROM_AUTHORITATIVE_DATA` (the failure timestamp) | Yes |
  | `days_to_nearest_payday_window` | `DERIVED_FROM_AUTHORITATIVE_DATA` (calendar math — same helper `policy/retry_candidates.py` already uses live) | Yes |
  | `prior_if_failure_count` | `DERIVED_FROM_AUTHORITATIVE_DATA` (this project's own stored `RawEvent` history) | Yes |
  | `primary_instrument` | `WEBHOOK_NATIVE` (`payload.payment.entity.method`) | Yes |
  | `tenure_days` | `RAZORPAY_API_ENRICHED` — optional, best-effort Subscriptions-API call, off by default (`LIVE_FEATURE_ENRICHMENT_ENABLED=false`) | Only if enabled and the call succeeds |
  | `plan_tier`, `city_tier` | `MERCHANT_PROFILE` — would need a merchant-supplied catalog this project has no integration for | No |
  | `prior_if_self_resolved_rate`, `is_month_end_settlement_rush`, `bank_network_conditions`, `issuing_bank_downtime_flag`, `network_latency_bucket` | `UNAVAILABLE` — simulation-only constructs in the synthetic dataset generator, no real-world source exists | No |

  Because at least 5 keys are always `UNAVAILABLE`, a real webhook can never
  produce a complete feature vector, with or without enrichment enabled —
  Model B is architecturally never invoked for genuine live traffic, and the
  rule-based tier is what actually decides every live subscription recovery
  today. This is an intentional, honestly-documented degradation (see the
  full reasoning and every failure-mode's handling in
  `recovery/live_feature_enrichment.py`'s module docstring), not a bug to
  paper over with fabricated feature values. Set
  `LIVE_FEATURE_ENRICHMENT_ENABLED=true` (requires `RAZORPAY_KEY_ID`/
  `RAZORPAY_KEY_SECRET`) to additionally attempt the one feature
  (`tenure_days`) that genuinely is obtainable via Razorpay's API — any
  failure (timeout, network error, HTTP 4xx/5xx, malformed response) degrades
  silently to "not enriched," never crashes or blocks the webhook.
- Compliance: `policy/compliance.py::evaluate_compliance` (`compliance-v1`),
  now including the contact-hours gate ([§18](#18-security--auditability))
- LLM: `llm/service.py` → `llm/client.py::get_llm_client()` — whatever
  `LLM_PROVIDER` is currently configured to (mock/anthropic/gemini/ollama)

**LIVE REVENUE-RISK PATH** (checkout_abandoned / mandate_failed /
receivable_overdue / promise_to_pay_broken / Payment-Link
`payment_failed_no_subscription` — every event WITHOUT a subscription_id):
- Entrypoint: same `/webhook/razorpay` → `recovery/webhook_pipeline.py::process_raw_event`
  → `recovery/revenue_orchestrator.py::orchestrate_revenue_event`, or the
  in-process promise-sweep scheduler (`recovery/scheduler.py`, always on
  unless `ENABLE_PROMISE_SWEEP_SCHEDULER=false`) for broken-promise events
- Model: **`unified_catboost_v1`** (`model/unified_model.py`), the single
  model shared across all five of these domains, loaded once via
  `get_live_unified_model()` and cached process-wide
- Policy: **`unified-ml-v1`** (`policy/revenue_recovery_policy.py::decide_for_revenue_risk_event`)
  — always consults BOTH a per-domain rule-based decider and the unified ML
  model; the rule decider is authoritative for eligibility/safety, ML's
  recommendation is used as the actual selection whenever the rule decider
  doesn't need to override it (see [§3c](#3c-unified-ml-generalization))
- Compliance: `policy/compliance_v2.py::evaluate_compliance_v2` (`compliance-v2`,
  ALLOWED/BLOCKED/HUMAN_REVIEW), same contact-hours gate as the legacy path
- LLM: identical `get_llm_client()` resolution as the subscription path —
  one provider, one config, both paths

**LEGACY / EXPERIMENTAL — reachable only from evaluation/training scripts and
tests, never from a live webhook:**
- `policy/decision_engine.py`'s own `decide_engine()` (POLICY_VERSION
  `policy-v3`) — policy-v3's single-abstention-threshold mechanism, kept only as
  the `original_fallback_policy` comparison baseline in
  `evaluation/evaluate_decision_engine_v4.py`
- `model/train_candidate_model.py`, `model/train.py`, `model/calibrate.py` —
  early model iterations, superseded by the latent-value regressor (Model B);
  kept for `evaluation/evaluate_models.py`'s historical comparison, never
  loaded by any live decision path
- `policy/recovery_policy.py::decide_candidate_aware` — the pre-policy-v3
  probability-based policy interface, superseded by
  `policy/decision_engine.py`'s value-native interface; exercised only by
  its own historical tests
- `evaluation/diagnose_original_fallback_effect.py`, `evaluation/evaluate_decision_engine.py`,
  `evaluation/evaluate_counterfactual_policy.py`, `evaluation/evaluate_policy.py`,
  `evaluation/evaluate_ranking_policy.py` — one-time or superseded diagnostic
  scripts from earlier days, kept as historical record and reproducibility
  artifacts, not part of any live or currently-recommended evaluation run
- `evaluation/reports/*.json` other than `decision_engine_v4_evaluation.json`
  and the unified-ML report — frozen historical SYNTHETIC BENCHMARK output,
  read-only, never recomputed by live code

## 5. End-to-end workflow

1. **Detect** — a Razorpay Subscriptions webhook fires on a failed charge.
2. **Verify + store** — HMAC-SHA256 signature checked against the raw
   request body; a duplicate `x-razorpay-event-id` is acknowledged (200) but
   not re-stored (`app/main.py`, `app/webhook_security.py`).
3. **Classify** — automatically, in the same request, a deterministic rule
   buckets `error_reason` into `retryable_soft` / `hard_decline` /
   `customer_cancelled` / `unmapped` (`classification/rules.py`, invoked via
   `recovery/webhook_pipeline.py`). `scripts/reprocess_raw_events.py` remains
   available for manual re-processing.
4. **Decide** — for `retryable_soft`: Model B scores 5 discrete candidate
   retry times, picks the highest net-value one that clears a decision-
   margin gate, with a rule-based fallback and a `NO_ACTION` safety net
   (`policy/decision_engine_v4.py`). Every other bucket → `NO_ACTION`,
   never guessed.
5. **Promise-to-pay override (optional)** — if the customer has already
   replied with a free-text promise, `recovery/promise_service.py` has
   parsed (LLM), validated (deterministic), and persisted it; a still-VALID
   promise's date is tried against compliance FIRST, and becomes the
   effective candidate if accepted — never bypassing compliance, only
   retiming an already-valid candidate (`recovery/orchestrator.py`).
6. **Compliance gate** — independently gates the payment action and the
   communication action (contact caps, opt-out/cancellation, required
   fields) before either is allowed to proceed (`policy/compliance.py`).
7. **Payment action** — recorded only (`retry_scheduled` / `blocked` /
   `no_action`) — no live Razorpay call is ever made.
8. **Communication (simulated)** — if allowed, outreach microcopy is
   generated by the LLM layer and logged as a structured "would have sent"
   record (`llm/service.py::generate_outreach_microcopy_and_log`). A
   `hard_decline` event gets a payment-method-update nudge specifically —
   never a false "we'll retry" message.
9. **Report** — `evaluation/*.py` scripts compute recovered-₹ / recovery
   rate / cost-per-recovery / etc. against 3 baselines on a held-out
   synthetic test split; the Streamlit dashboard visualizes both that frozen
   benchmark and a live sample run of the real orchestrator.

Every stage above is implemented by exactly one module, called by
`recovery/orchestrator.py` and reused unmodified by the CLI demo scripts,
the dashboard, and the test suite — there is no duplicated business logic
in the UI, demo scripts, or evaluation scripts (verified directly; see the
final audit report §3).

## 6. Technology stack

FastAPI + Uvicorn (webhook receiver), SQLAlchemy + SQLite (event store,
audit log — file at `data/recovery_agent.db`), scikit-learn (linear
baseline) + CatBoost (both the original subscription-only Model B and the
unified cross-domain model, §3c), an LLM provider abstraction
(`llm/client.py`) supporting `mock` (offline, default), `anthropic` (Claude
API), `gemini` (Google Gemini API), or **`ollama`** (a locally-running
`qwen3:14b`, no API key or external network call — the provider this
working copy runs live communication through; see §11) for the LLM jobs,
Streamlit + Plotly (dashboard), pytest (test suite), zrok (webhook tunnel —
Razorpay blacklists `ngrok.io`, see [§10](#10-razorpay-test-mode-setup)).
Exact pinned versions: `requirements.txt`.

## 7. Setup

```bash
git clone <your-repo-url> recovery-agent
cd recovery-agent

python3.12 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

cp .env.example .env
# edit .env — see §8 for what each value needs and whether it's required
```

`pip check` is clean against the pinned `requirements.txt` (verified in this
audit pass). Everything below works fully offline with no further
configuration — see [§9](#9-offline-mode).

## 8. Required human inputs

| Input | Where it's used | Status in this repo | Required for |
|---|---|---|---|
| `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` | `app/config.py` (loaded); optionally used by `recovery/live_feature_enrichment.py`'s Subscriptions-API call, ONLY when `LIVE_FEATURE_ENRICHMENT_ENABLED=true` (default `false`) | Configured in local `.env` (not committed); empty in `.env.example` | Required for real Test Mode subscriptions in the Dashboard (§10), and optionally for live feature enrichment (§4a). **NOT REQUIRED FOR OFFLINE TESTS** — `pytest tests/ -q` never reads these. |
| `RAZORPAY_WEBHOOK_SECRET` | HMAC verification (`app/webhook_security.py`) | **NOT REQUIRED FOR OFFLINE TESTS.** `.env.example` ships a clearly-labeled FAKE placeholder (`local_dev_placeholder_change_me`) so `cp .env.example .env` alone is enough for FastAPI to start locally — it behaves as an ordinary HMAC secret (real verification, no bypass), just not a real Razorpay-issued one. **REQUIRED FOR A REAL RAZORPAY TEST WEBHOOK**: replace it with the Dashboard-issued secret (§10) before pointing this server at genuine Razorpay traffic. | See left. |
| `LIVE_FEATURE_ENRICHMENT_ENABLED` / `LIVE_FEATURE_ENRICHMENT_TIMEOUT_SECONDS` | `recovery/live_feature_enrichment.py` | Defaults to `false` / `5` in `.env.example` | **OPTIONAL.** Safe to leave at the default for every workflow in this repo, tests included — see the feature-source table in §4a for why enabling it still never makes Model B run live on real traffic. |
| zrok token / tunnel | Exposes local FastAPI to a public HTTPS URL Razorpay's Dashboard will accept (`ngrok.io` is explicitly blacklisted by Razorpay) | Not stored in this repo — a per-developer, per-session manual step | Required only for a live webhook demo (§10). Not required for anything else. |
| `ANTHROPIC_API_KEY` | `llm/client.py::AnthropicLLMClient`, only reached when `LLM_PROVIDER=anthropic` | Empty in `.env.example`; `LLM_PROVIDER` defaults to `mock`, which never reads this value | Required only to see real Claude-generated output instead of the deterministic mock. Test suite, demos, and dashboard all default to mock and need no key. |
| `GEMINI_API_KEY` | `llm/client.py::GeminiLLMClient`, only reached when `LLM_PROVIDER=gemini` | Empty in `.env.example`; free-tier quota is small and can exhaust mid-demo, in which case the deterministic fallback is used and logged — never treated as a bug | Required only to see real Gemini-generated output. Test suite always forces `LLM_PROVIDER=mock` regardless of `.env` (`tests/conftest.py::_force_mock_llm_provider_by_default`), so it never affects test results either way. |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | `llm/client.py::OllamaLLMClient`, only reached when `LLM_PROVIDER=ollama` | Defaults to `http://localhost:11434` / `qwen3:14b` in `.env.example` — no API key of any kind | Required only to use a locally-running Ollama server instead of a remote provider. Requires `ollama serve` running and the model already pulled (`ollama pull qwen3:14b`); reachability/model-missing failures fall back to the same deterministic mock output as any other provider failure. Test suite always forces `LLM_PROVIDER=mock`, so it never makes a real local call either. |
| Trained model artifacts | `model/latent_target_artifacts/`, `model/artifacts/`, etc. | **Not committed** (`.gitignore` excludes all `model/*_artifacts/` and `evaluation/reports/`) — present only in this working copy because training/evaluation were already run here | **NOT REQUIRED FOR OFFLINE TESTS**: `pytest tests/ -q` trains its own small, deterministic, tmp-directory-only unified-model artifact automatically (`tests/conftest.py`'s session-scoped fixture) and every other model dependency in the test suite either injects a fake model or gracefully exercises the fallback path — no manual training step is needed for a fresh clone's tests to pass. **Still a manual step for the dashboard/demo to show real (non-fallback) model output**: run `python -m model.train`, `python -m model.train_candidate_model`, `python -m model.train_ranking_model`, `python -m model.train_latent_target_model`, `python -m model.train_unified_model`, then the matching `evaluation/evaluate_*.py` scripts. The orchestrator degrades gracefully if skipped either way (falls back to the rule-based/deterministic tier), but the "current model" evaluation numbers in §16 won't exist until these are run. |
| Generated synthetic dataset | `data/raw/*.csv`, `data/processed/*.csv` | Present and committed — the dataset itself, unlike the trained artifacts above, is tracked in git | Nothing further needed; regenerate via `python -m data.generate_synthetic_dataset` / `python -m data.generate_counterfactual_dataset` only if you want a fresh draw |

No secret value is printed anywhere in this document or in application logs
(verified — `tests/test_llm.py::TestNoSecretsInLogs`,
`tests/test_orchestrator.py::test_no_secrets_in_audit_trail`).

## 9. Offline mode

Everything except a live Razorpay webhook delivery runs with **zero network
calls**: `LLM_PROVIDER=mock` (the default) makes no Anthropic API call
(verified by a monkeypatched-socket test that raises if any socket connects
at all), and every demo/evaluation/dashboard script uses either the
committed synthetic dataset or a throwaway in-memory SQLite database. This
is the mode the full test suite runs in, and the mode `scripts/run_dashboard.sh`,
`scripts/run_end_to_end_demo.py`, and `scripts/run_llm_demo.py` all default
to.

## 10. Razorpay Test Mode setup

Only needed to demo a **real** webhook delivery — everything else in this
repo works without it.

1. Sign in at `dashboard.razorpay.com`, switch to **Test Mode**.
2. **API keys**: Account & Settings → API Keys → Generate Test Key → into
   `.env` as `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET`.
3. **Tunnel** (zrok, not ngrok — Razorpay's webhook docs explicitly
   blacklist `ngrok.io` and reject `localhost`):
   ```bash
   zrok enable <token from your zrok account>
   zrok share public localhost:8000
   ```
4. **Webhook**: Dashboard → Accounts & Settings → Webhooks → + Add New
   Webhook. URL = your zrok URL + `/webhook/razorpay`. Secret = a long
   random string → `RAZORPAY_WEBHOOK_SECRET` in `.env` (**not** the same as
   `RAZORPAY_KEY_SECRET`). Active events: at least `payment.failed`.
5. **Trigger `insufficient_fund`**: create a Test Mode Subscription, pay its
   first cycle with test card `4100 2800 0009 0000` (Visa), and — on the
   mock bank screen — actively choose **Failure**.
6. Watch the FastAPI server log for `Stored event_type=payment.failed ...
   error_reason=insufficient_fund` immediately followed by
   `Orchestration complete for event_id=... final_status=...` — classification
   and full orchestration now run automatically in the same request (§4). If
   you ever need to re-run it manually (e.g. after fixing a downstream bug),
   `./venv/bin/python -m scripts.reprocess_raw_events` is idempotent and safe
   to re-run.

## 11. LLM configuration

```bash
# .env
LLM_PROVIDER=mock          # mock (default, offline, no key) | anthropic | gemini | ollama
ANTHROPIC_API_KEY=         # only read when LLM_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-sonnet-5
GEMINI_API_KEY=            # only read when LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-3.6-flash
OLLAMA_BASE_URL=http://localhost:11434   # only used when LLM_PROVIDER=ollama; no API key
OLLAMA_MODEL=qwen3:14b                   # must already be pulled locally (`ollama pull qwen3:14b`)
```

Exactly **three** LLM jobs exist anywhere in this codebase
(`llm/service.py`): outreach microcopy generation, promise-to-pay reply
parsing, and batch-level report narration. Nothing else calls an LLM —
classification, candidate-time scoring, and compliance are all deterministic
or model-driven, never LLM-driven. Verified directly in this audit: an LLM
failure (unavailable, timeout, malformed JSON, schema-invalid JSON) never
changes `classification_bucket`, `selected_candidate_type`, or
`compliance_allowed` — the policy decision is persisted to the database
*before* any LLM call is made, and no code path reads an LLM result back
into a policy/compliance field (`recovery/orchestrator.py`, confirmed by
`tests/test_orchestrator.py::test_llm_failure_never_changes_selected_candidate_or_compliance`
and re-confirmed by direct execution in this audit — see the final report).

## 12. Running FastAPI

```bash
./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
curl http://127.0.0.1:8000/health   # -> {"status":"ok","env":"test"}
```

## 13. Running the dashboard

```bash
./scripts/run_dashboard.sh
# or directly:
./venv/bin/streamlit run ui/app.py
```

**Works standalone, with zero setup, on a completely fresh clone and an
empty/nonexistent database file — FastAPI does NOT need to have run first.**
(Final pre-submission correction: this used to crash every live-DB page with
`sqlite3.OperationalError: no such table: raw_events` on a brand-new
checkout, because schema creation only happened in `app/main.py`'s FastAPI
lifespan. `ui/app.py::main()` now calls `ui/data.py::ensure_schema_initialized()`
— which wraps the EXISTING `app/db.py::init_db()`, not a second schema
initializer — before any live-DB query runs. With no events yet, every page
shows a truthful empty state, never fabricated data. See
`tests/test_ui.py::TestFreshCloneDatabaseInitialization`.)

Runs fully offline. Verified in this audit: clean startup, HTTP 200 on the
root route, and every data-layer function (`ui/data.py`) exercised directly
— including missing-file, corrupt-JSON, and unknown-event-id paths — with
zero unhandled exceptions. The repo's own `tests/test_ui.py` additionally
drives every page and every interactive control through Streamlit's real
`AppTest` script-execution harness (not a mock).

## 14. Running the end-to-end demo

```bash
./venv/bin/python scripts/run_end_to_end_demo.py   # 5 scenarios: insufficient_fund normal recovery,
                                                     # hard decline + payment-method-update nudge,
                                                     # communication-blocked (opt-out), customer reply ->
                                                     # promise-to-pay -> retry override, LLM failure, and
                                                     # webhook ingestion -> automatic orchestration
./venv/bin/python scripts/run_llm_demo.py           # the 3 LLM jobs in isolation, including a forced failure
```

Both use a throwaway in-memory database per scenario and the real trained
model artifacts; neither touches `data/recovery_agent.db` nor makes a real
network call — scenario 5's "webhook" is a signed, in-process HTTP request
to this project's own FastAPI app (`fastapi.testclient.TestClient`), never a
call to Razorpay's servers. Both scripts bootstrap `sys.path` explicitly so
they also work under their documented direct invocation (`python scripts/foo.py`,
not just `-m`).

## 15. Testing

```bash
./venv/bin/python -m pytest tests/ -v
```

Self-contained on a genuinely fresh clone — **no manual training step is a
prerequisite.** `tests/conftest.py`'s session-scoped fixture trains one real,
small unified-model artifact into a pytest tmp directory automatically (never
into `model/artifacts/`, never overwriting a real committed-ignored
artifact); every other model dependency in the suite either injects a fake
model object (`policy/decision_engine.py`'s Model B tests) or exercises the
documented, tested fallback path directly.

**1019 tests collected, all passing** in this working copy (verified by
direct execution, twice, in the BUG-1/2/3/4 pre-submission audit fix pass;
also independently verified passing — 1016 passed, 3 skipped — from a
genuinely fresh clone with no pre-existing model artifacts or database, per
that audit's BUG-2 fix); the historical "432/432" figure below in §17
predates Track-03 entirely and is kept only as a point-in-time record —
always trust `pytest tests -q`'s own output over any number in this
document, including this one.
Coverage includes, per boundary: malformed webhook body, invalid/missing/
tampered signature, duplicate webhook delivery, missing required fields, an
unsupported event type or a missing subscription/payment entity, a
downstream classification/orchestration/DB/LLM failure after a webhook is
already stored, 12 model-failure modes (missing artifact, NaN/negative/
implausible predictions, prediction exceptions), 6 LLM failure modes per job
(provider unavailable, timeout, empty response, invalid JSON, schema-invalid
JSON, unexpected exception), every compliance rule independently, promise-to-pay
validation/persistence/idempotency/supersession and its orchestrator-level
override, hard-decline communication, LLM-vs-policy independence, and
no-secrets-in-logs, PLUS (added in the unified-ML pass) unified feature
construction, all-five-domain candidate generation, entity-split integrity,
no-target-leakage, artifact loading + missing-artifact fallback, the
ML-consulted-but-overridden policy boundary, and dashboard ML-status
correctness (`tests/test_unified_model.py`,
`tests/test_revenue_recovery_policy.py`, `tests/test_revenue_ui.py`).
`pip check` is clean.

## 16. Evaluation

All evaluation numbers are **SYNTHETIC COUNTERFACTUAL EVALUATION** — computed
against `data/raw/counterfactual_outcomes.csv`, a hand-designed simulation
this project's own generator authored. **None of it measures real Razorpay
recovery performance.** The dashboard and every report JSON label this
explicitly and consistently.

**Baselines** (`policy/baselines.py`) — **fixed for fidelity to the
specification in this pass; both deviations below are now closed:**
No Recovery (never acts, matches spec, unchanged). **Fixed Retry** now
schedules the specification's full "silent auto-retry once/day for 3 days,
same channel, then gives up" cadence — T+1 → T+2 → T+3 — instead of only
ever selecting T+1. T+1 and T+3 reuse this project's existing
`plus_1_day_morning` / `plus_3_days` candidates 1:1 (both already have real,
independently-simulated outcome rows in `data/raw/counterfactual_outcomes.csv`);
T+2 has no equivalent in that 5-candidate framework (never extended with a
6th "day+2" type, and the counterfactual dataset was never generated with
one — adding one now would mean regenerating the synthetic dataset with a
new outcome definition, which this fix was explicitly instructed not to do).
T+2 is still genuinely SCHEDULED (a real, distinct attempt — the retry-cost
economics below charge for it), but contributes zero incremental recovery
probability of its own — a deliberately conservative, disclosed
simplification, not an invented estimate. **Rule-Based** now also attaches
the specification's communication dimension — one deterministic WhatsApp
nudge (fixed template, sent alongside the payday-timed retry) plus one
follow-up 3 days later, then stop — while its own retry-timing decision
(payday proximity ≤2 days → `payday_window`, else → `plus_1_day_morning`)
is completely unchanged. Both fixes are purely ADDITIVE to
`fixed_retry_baseline` / `rule_based_baseline`'s returned dicts —
`selected_candidate_type` / `selected_candidate_datetime`, the only two
keys `policy/decision_engine.py` / `policy/decision_engine_v4.py`'s live
fallback tier ever reads from `rule_based_baseline`, are byte-identical to
before this fix; the operational policy-v4 decision engine was not touched.
See `tests/test_baseline_fidelity.py`.

**Current model** (the one the live orchestrator and dashboard actually
use): "Model B", a CatBoost regressor predicting `expected_recovery_value_latent`
directly (₹, not a probability), selected over five earlier model/objective
iterations (a plain calibrated classifier, a candidate-aware classifier, a
pairwise ranker, and Model A — a companion probability regressor)
specifically because it was **the only one of six trained models that beat
Fixed Retry on the noise-free latent economic ground truth**. The full
diagnostic trail — including two honestly-reported negative results (the
candidate-aware and pairwise ranking models scoring *worse than random*
top-1 accuracy) — is in `evaluation/evaluate_ranking_policy.py`'s and
`model/diagnose_ranking_failure.py`'s own output, and is the strongest
evidence in this repo of genuine, undodged empirical iteration.

**Headline result, test set (n=60 held-out events), from
`evaluation/reports/decision_engine_v4_evaluation.json`
(the authoritative report — the only one reflecting the actually-deployed
policy-v4 + Model B combination with its frozen production config):**

| Policy | Latent ₹ (noise-free) | vs. Fixed Retry | **Realized ₹ (stochastic)** | **vs. Fixed Retry (realized)** | Recovery rate |
|---|---|---|---|---|---|
| Fixed Retry (baseline) | 19,114.77 | +0.00 | 23,296.10 | +0.00 | 85.0% |
| Rule-Based (baseline) | 19,050.32 | −64.45 | 21,431.15 | −1,864.95 | 76.7% |
| Model B alone | 20,347.65 | **+1,232.88** | 21,278.18 | −2,017.92 | 73.3% |
| **Deployed policy (policy-v4) — SUPERSEDED, see §16b** | 19,032.25 | −82.51 | 19,997.23 | −3,298.87 | 70.0% |
| Oracle (upper bound) | 23,031.64 | +3,916.87 | 24,275.30 | +979.20 | 86.7% |

**This table is preserved as the historical record of the finding that
triggered the correction below — the deployed policy row is no longer
accurate.** [§16b](#16b-economic-correction-subscription-decision-policy)
identifies the exact mechanism responsible (a blind-swap fallback rule) and
corrects it; the deployed policy now scores identically to "Model B alone"
above (₹21,278.18 realized, −₹2,017.92 vs. Fixed Retry — still a loss, but
₹1,280.95 smaller, and the bootstrap CI on the ₹ delta no longer excludes
zero). Read §16b before citing this table's deployed-policy row anywhere.

Latent ₹ (noise-free) is unchanged by the baseline-fidelity fix below —
it's the model's own belief about a SINGLE selected candidate, and the
existing latent-value framework has no defined notion of a multi-attempt
sequence to extend it to (see the fix note below); only the realized/
statistical/economics columns, which score against actual simulated
outcomes, change.

**Read honestly, not favorably — the core finding of this evaluation:**
Model B beats Fixed Retry on the noise-free latent economic ground truth
(what it should achieve in expectation), but **on realized (stochastic) ₹
recovered — the metric closest to a real outcome — every trained-model
policy loses to the simple Fixed Retry baseline, including the policy
actually wired into `recovery/orchestrator.py` today** (−₹3,298.87, −15
points of recovery rate). This gap is LARGER than previously reported
(−₹1,856.87, −10 points) because Fixed Retry itself was previously
under-modeled as a single T+1 attempt rather than the specification's full
3-day campaign — see "Baseline fidelity fix" below. Correcting that made
Fixed Retry's own recovery rate rise from 80.0% to 85.0%, which widens,
not narrows, the deployed policy's shortfall — reported here exactly as
measured, with no attempt to tune the model back ahead of it (explicitly
out of scope for that fix). The deployed policy's validation-tuned fallback
logic, meant to guard against low-confidence predictions, trades away most
of Model B's raw latent-value edge without recovering it on the realized
draw. The specification's core requirement — "the agent must clear all
three baselines, not just the easiest one" — **is not currently met on
realized money for the deployed policy** (as of this diagnosis; **see
[§16b](#16b-economic-correction-subscription-decision-policy) — the exact
mechanism causing most of this gap, the blind-swap fallback rule, has since
been fixed on validation and the gap is now ₹2,017.92, not ₹3,298.87**).

**Baseline fidelity fix (`policy/baselines.py`, `evaluation/evaluate_decision_engine_v4.py::score_fixed_retry_sequence`):**
the table above previously scored Fixed Retry against ONLY its T+1
attempt's own outcome row — silently understating the specification's
"retry once/day for 3 days, then gives up" cadence as a single try. Fixed
Retry now schedules T+1 → T+2 → T+3 (see §16 baselines paragraph above for
exactly how T+2's missing outcome data is handled) and is scored as
recovered if EITHER T+1's or T+3's existing, already-simulated outcome row
recovers — using the SAME held-out test events and SAME counterfactual
machinery as every other policy in this table, never a new outcome
definition. Rule-Based's own recovery/₹ numbers are UNCHANGED by this fix
(only its communication metrics below are new) — its retry-timing decision
was never wrong, only its missing communication dimension was.

**Statistical significance (`evaluation/statistics.py`, wired into
`evaluate_decision_engine_v4.py`'s report as `statistical_tests`):** for the
headline comparison above — deployed policy vs Fixed Retry, on the SAME 60
paired held-out test events used everywhere else in this section —

- **McNemar's exact test** (binomial, two-sided, on the paired binary
  `realized_recovered` outcome — never applied to the continuous ₹ amounts,
  which is a statistically different question): of the 60 paired events,
  1 recovered under the deployed policy but not Fixed Retry (`b`), and 10
  recovered under Fixed Retry but not the deployed policy (`c`) — **p = 0.0117**,
  significant at p<0.05.
- **95% bootstrap CI** (percentile method, 10,000 resamples over paired
  events, seed=42) on the realized-₹ delta: **−₹3,298.87 [−₹6,605.61, −₹47.65]**
  — the interval no longer spans zero.

**Read together, honestly:** with the baselines corrected, both tests now
find the deployed policy's realized-₹ loss against Fixed Retry
**statistically significant** at this sample size (n=60) — a materially
different, more defensible conclusion than the previous pass's "not
significant" finding, which rested on an understated Fixed Retry baseline.
This still does not establish real-world production superiority of Fixed
Retry (see the synthetic-evaluation caveat throughout this section) — only
that, WITHIN this held-out synthetic slice, the gap is unlikely to be
sampling noise. **This McNemar/bootstrap result is for the pre-correction
config; [§16b](#16b-economic-correction-subscription-decision-policy) has
the corrected, current numbers (p=0.0391, CI now spans zero).** See
`tests/test_statistics.py` and `tests/test_baseline_fidelity.py` for the
methodology's own test coverage,
and §19 item 1 below.

**Contact / communication metrics (specification section 12, now wired
into the report as `contact_and_intervention_metrics` / `cost_per_recovery`):**
only Rule-Based has any communication modeled in this evaluation layer
(Fixed Retry is silent, per spec; the other policies' real LLM-generated
communication is the live orchestrator's own, unrelated code path).
Rule-Based: customer-contact rate 100.0% (every retryable_soft test event
gets the nudge), 2.0 average contacts per contacted subscription (nudge +
follow-up, always exactly 2), total contact cost ₹16.20 (120 messages ×
₹0.135), cost per recovery ₹0.3522. Unnecessary-intervention rate reuses
this codebase's own existing definition (`evaluation/evaluate_counterfactual_policy.py`:
a real action that did not result in recovery — the specification's literal
"would have recovered under No Recovery anyway" condition has no
counterfactual outcome row to test against in this dataset) — Fixed Retry
15.0%, Rule-Based 23.3%, deployed policy 26.67% (post-correction — see §16b; was 30.0% pre-correction).

**Fee economics (`policy/economics.py`, wired into the same report as
`economics`):** the specification's "report both raw merchant GMV and
Razorpay's own fee take ... as two separate numbers" is now implemented.
Fee assumption, verified against the specification rather than guessed: the
document states domestic card payments are priced at "roughly 2% + 18% GST"
and separately flags UPI's exact rate as an unresolved inconsistency in
Razorpay's own public materials ("do not state a single clean UPI take-rate
figure") — so this project applies the one verified card rate uniformly
(effective ≈2.36% of recovered GMV, gross, i.e. fee + GST-on-the-fee) and
documents that simplification rather than inventing a second, uncited UPI
number. Intervention cost for Fixed Retry now reflects the ACTUAL number of
attempts made per event (1 if recovered at T+1, up to 3 otherwise — average
1.4/event on this test set); Rule-Based's now includes its WhatsApp contact
cost on top of its one retry attempt. For the headline comparison, synthetic
test set:

| Policy | Recovered GMV | Intervention cost | Razorpay fee take (gross) | Net recovery value |
|---|---|---|---|---|
| Fixed Retry | ₹23,296.10 | ₹420.00 | ₹549.79 | ₹22,326.31 |
| Rule-Based | ₹21,431.15 | ₹316.20 | ₹505.78 | ₹20,609.17 |
| **Deployed policy (policy-v4), post-correction — see §16b** | ₹21,278.18 | ₹300.00 | ₹502.17 | ₹20,476.01 |

`net_recovery_value = recovered_gmv − intervention_cost − razorpay_fee_take`
— a new, REALIZED, report-level summary metric, distinct from
`policy/decision_engine.py`'s pre-existing `expected_net_value`/
`decision_margin` (which subtracts only `intervention_cost`, computed at
DECISION time before any fee modeling existed — both names are kept,
documented, never merged). See `tests/test_economics.py`.

Full metric definitions, every intermediate model's numbers (including two
architectures — the candidate-aware and pairwise ranking models — that
scored *below random* and are reported as honest negative results, not
hidden), and the full diagnostic history are in
`evaluation/reports/*.json` and the relevant `evaluate_*.py` scripts.

## 16b. Economic correction: subscription decision policy

Final pre-submission audit. Traced the exact mechanism responsible for the
₹3,298.87 gap reported in §16 above, fixed it using validation data only,
and re-ran the frozen held-out TEST split exactly once.

**Root cause (`policy/decision_engine_v4.py`, `evaluation/evaluate_decision_engine_v4.py::select_validation_configuration`):**
the original policy-v4 validation search chose its configuration
(`margin_threshold=5.0, fallback_mode=ALWAYS_FALLBACK_WHEN_BELOW_MARGIN`) by
maximizing total **latent** value — a smooth proxy
(`expected_recovery_value_latent = recovery_probability_latent × amount`).
Re-running that same 108-configuration search on the same 59-event
validation split, scored instead by total **realized** ₹ recovered, shows
the two metrics disagree — on validation itself, not merely on test: the
originally-chosen config scores ₹16,417.73 realized (a ₹2,105.48 *loss*
vs. Model-B-alone's ₹18,523.21), despite scoring ₹350.53 *higher* on the
latent proxy. Mechanistically: `ALWAYS_FALLBACK_WHEN_BELOW_MARGIN`
unconditionally discards Model B's own top pick and substitutes
Rule-Based's candidate whenever Model B's own top-2 margin is small —
**without ever checking whether the substitute is actually any good.**
`operational.average_fallback_advantage_rs = −7.59` in the original
(`decision_engine_v4_evaluation.json`) report already proved this: the
substituted candidate was, on average, ₹7.59 *worse* by Model B's own
valuation, every single time the fallback fired. On the held-out TEST set,
this cost exactly **₹1,280.95 out of the ₹3,298.87 gap** (`model_b_alone`
₹21,278.18 vs. the old `improved_fallback_policy` ₹19,997.23) — the
remainder of the gap (₹2,017.92) is a separate, structural finding (below),
not fixed by this correction.

**Stage decomposition (held-out TEST, n=60, `evaluation/reports/decision_engine_v4_evaluation.json`'s
`stage_decomposition`) — isolating exactly where value is lost:**

| Stage | ₹ recovered | Recovery rate | Cost | Net value | Diff vs. Fixed Retry |
|---|---|---|---|---|---|
| 1. Fixed Retry (baseline) | 23,296.10 | 85.0% | 420.00 | 22,876.10 | +0.00 |
| 2. Rule-Based (baseline) | 21,431.15 | 76.7% | 316.20 | 21,114.95 | −1,761.15 |
| 3. Model-only, no margin gate | 21,278.18 | 73.3% | 300.00 | 20,978.18 | −1,897.92 |
| 4. Model + margin gate + NO_ACTION fallback *(diagnostic)* | 9,712.64 | 40.0% | 175.00 | 9,537.64 | −13,338.46 |
| 5. Model + margin gate + OLD blind-swap fallback *(diagnostic — the rejected, pre-correction mechanism)* | 19,997.23 | 70.0% | 300.00 | 19,697.23 | −3,178.87 |
| 6. **Deployed policy, corrected** | **21,278.18** | **73.3%** | **300.00** | **20,978.18** | **−1,897.92** |
| 7. Oracle (upper bound) | 24,275.30 | 86.7% | 300.00 | 23,975.30 | +1,099.20 |

Guardrails (`policy/guardrails.py`: classification bucket, max retry
attempts, candidate-timing validity, duplicate-decision protection) are not
a separate row — they run first, identically, before Model B is ever
consulted, in every stage above, so they cannot be the source of the gap
(confirmed: they never differ between stages 3–6). Stage 4 shows abstaining
under ambiguity is far worse than either fallback strategy — the margin
gate fires often enough (37/59 validation events, 25/60 test events) that
refusing to act on all of them forfeits most of the achievable value. Stage
5 is the exact old mechanism being replaced; stage 6 is what ships now.

**The fix (validation-only, `evaluation/evaluate_decision_engine_v4.py::select_validation_configuration`):**
the search now scores every configuration by BOTH total realized ₹ AND
total latent ₹ on validation (both legitimate to use — neither ever touches
test), with realized ₹ as the primary key (ties broken by higher latent
value, then by preferring the structurally safest fallback mode — see
below — then fewer fallbacks/no-actions). Re-run on the SAME 59 validation
events across the SAME 108 configurations required at minimum (current
deployed / model-only / model + guardrails / revised fallback modes /
revised margin thresholds / combinations): **every configuration that ever
blind-swaps away from Model B's own top pick scores worse on realized ₹
than Model B alone** — the winner (tied across all no-blind-swap-equivalent
configs) is `margin_threshold=0.0, fallback_mode=KEEP_MODEL_UNLESS_RULE_HAS_CLEAR_ADVANTAGE,
fallback_advantage_threshold=0.0`. This mode is mathematically guaranteed
(see `policy/decision_engine_v4.py`'s STRUCTURAL FINDING) to never select a
worse candidate than Model B's own best pick, because Rule-Based's
candidate is always drawn from the same set Model B already scored — it was
chosen over the behaviorally-identical `margin_threshold=0` + any mode
specifically for this safety guarantee: if the candidate/cost architecture
ever changes such that the two are no longer equivalent, this mode can
still never do worse than Model B alone, unlike the old
`ALWAYS_FALLBACK_WHEN_BELOW_MARGIN`. No guardrail, compliance check,
attempt cap, or timing constraint was touched.

**Final held-out TEST result (run exactly once, after the config was frozen
from validation — never re-tuned against this number):**

| Policy | Realized ₹ recovered | vs. Fixed Retry |
|---|---|---|
| Fixed Retry | 23,296.10 | +0.00 |
| Rule-Based | 21,431.15 | −1,864.95 |
| Model B alone | 21,278.18 | −2,017.92 |
| **Deployed policy (corrected)** | **21,278.18** | **−2,017.92** |
| Oracle | 24,275.30 | +979.20 |

McNemar's exact test (paired binary outcome): p = **0.0391** (still
significant at p<0.05, down from p=0.0117). 95% bootstrap CI on the ₹
delta: **[−₹4,719.40, +₹800.02]** — this interval now **spans zero**,
unlike the pre-correction CI ([−₹6,605.61, −₹47.65]), meaning the ₹ loss is
no longer confidently distinguishable from zero at this sample size, even
though the binary recovered/not-recovered outcome still favors Fixed Retry
significantly.

**Verdict at this stage (single attempt per event) — the corrected policy
does NOT beat Fixed Retry.** The fix recovers ₹1,280.95 of the ₹3,298.87 gap
(39%) and narrows it from −14.2% to −8.7%, but a real gap remains. The
residual cause is structural, not a threshold to tune: **Fixed Retry gets
three scheduled attempts (T+1/T+2/T+3) per event; every model-based and
rule-based policy in this codebase makes exactly one.** `decide_engine_v4`
returned a single `selected_candidate_type` / `selected_candidate_datetime`
at this point, with no multi-attempt sequencing mechanism — see §16c below,
where this residual gap is addressed directly, superseding the verdict in
this paragraph.

Reproduce: `./venv/bin/python -m evaluation.evaluate_decision_engine_v4`.
See `tests/test_decision_engine_v4.py::test_default_config_values_are_the_validation_selected_ones`
and `test_default_config_never_blindly_swaps_away_from_models_own_best_pick`
for the regression tests pinning this correction.

## 16c. Multi-attempt persistence (closes the structural gap from §16b)

Final pre-submission audit, second pass. §16b's own honest verdict named the
exact residual cause: Fixed Retry gets three scheduled attempts per event,
every other policy here made exactly one. This section gives the deployed
policy the same persistence — reusing Model B's own value ranking instead of
Fixed Retry's fixed calendar cadence — wires it into BOTH the evaluation and
the live path, and reports the real re-run result.

**Mechanism (`policy/decision_engine_v4.py::build_retry_schedule_from_decision`):**
purely additive to `decide_engine_v4` — slot 1 is exactly the existing
margin/fallback-gated selection, unchanged. Slots 2–3 backfill with the
next-best DISTINCT candidate types from `decision.candidate_scores`, ranked
by Model B's OWN predicted net value (the same ranking already computed to
pick slot 1 — no new model call), capped at `guardrails.MAX_RETRY_ATTEMPTS`
(3). Evaluation scores this schedule with the SAME "stop at first recovered
attempt" function Fixed Retry's own T+1/T+2/T+3 schedule is scored with
(`evaluation/evaluate_decision_engine_v4.py::score_fixed_retry_sequence`) —
an apples-to-apples comparison, not a new cost or outcome model.

**Live wiring (`recovery/retry_sweep.py`, `app/main.py`'s lifespan, same
in-process asyncio pattern as the promise-sweep scheduler — no second
scheduler framework):** `decide_for_failure_event_engine_v4` now persists
the schedule on the `policy_decisions` row (`retry_schedule_json`,
`retry_schedule_datetimes_json`, `retry_schedule_next_index`, all
nullable/additive). `recovery/retry_sweep.py`'s periodic loop
(`ENABLE_RETRY_SWEEP_SCHEDULER`, default on, `RETRY_SWEEP_INTERVAL_SECONDS`,
default 300s) advances one schedule step at a time as each step's scheduled
time arrives, and stops early the moment `RecoveryOutcome` confirms recovery
— matching this project's existing binding rule that no live payment is ever
actually executed (every advance is an audited record, exactly like attempt
1). This is the SAME capability the evaluation number below credits, not an
evaluation-only fiction — see `tests/test_decision_engine_v4.py`'s
`build_retry_schedule_from_decision` tests and `tests/test_retry_sweep.py`.

**RE-CHECK-BEFORE-ACTING (found during self-review of this same fix, fixed
before shipping):** spreading real attempts across real elapsed time creates
a new failure mode a single-decision policy never had — a customer opting
out, or a subscription getting cancelled, BETWEEN attempt 1 and a later
scheduled attempt or deferred communication. `advance_one_retry_schedule` /
`fire_one_deferred_communication` now re-check durable state
(`PolicyDecision.customer_opted_out`, a new sticky durable field — this
project previously had NO durable opt-out record at all, only a request-
scoped flag invisible to a sweep running hours later; and the MOST RECENT
`classification_bucket` across every `policy_decisions` row for the same
subscription) before every advance, and PERMANENTLY suppress every remaining
step the moment either signal disqualifies the subscription — never just
skip one step and try again later. Mirrors `recovery/promise_sweep.py`'s own
re-check-before-acting shape. See `tests/test_retry_sweep.py::TestReCheckBeforeActing`.

**APPLES-TO-APPLES CAVEAT — read before the result table below.** The first
version of this fix gave the deployed policy and Fixed Retry up to 3
scheduled attempts each but left `oracle_policy` scored as a SINGLE pick —
so the deployed policy could (and briefly did, in an earlier draft of this
section) numerically beat "Oracle," which is nonsensical for a quantity
defined to be an upper bound: a multi-attempt policy gets more independent
rolls at a stochastic outcome than a one-shot "perfect" pick, regardless of
how good either one is. Caught and fixed before this was reported as a
result: `oracle_policy` is now ALSO scored with up to `MAX_RETRY_ATTEMPTS`
attempts, ranked by the SAME `latent_value` proxy it always used (never the
true realized outcome, which would make it a trivial, always-recovers
oracle) — see `tests/test_decision_engine_v4.py::test_oracle_realized_value_is_never_below_the_deployed_policy`,
which asserts this can never regress silently again.

**Final held-out TEST result (re-run once after wiring, same frozen
validation-selected config from §16b, same n=60 population, Oracle
comparison apples-to-apples per the caveat above):**

| Policy | Realized ₹ recovered | Recovery rate | vs. Fixed Retry |
|---|---|---|---|
| Fixed Retry | 23,296.10 | 85.0% | +0.00 |
| Rule-Based | 21,431.15 | 76.7% | −1,864.95 |
| Model B alone (single attempt, §16b) | 21,278.18 | 73.3% | −2,017.92 |
| **Deployed policy (multi-attempt)** | **25,421.34** | **90.0%** | **+2,125.24** |
| Oracle (multi-attempt upper bound) | 25,943.30 | 91.7% | +2,647.20 |

Economics (`policy/economics.py`, GMV/fee/net split): deployed policy net
recovery value ₹24,396.40 vs. Fixed Retry's ₹22,326.31 (+₹2,070.09) and
correctly below Oracle's ₹24,966.04, intervention cost ₹425.00 (vs. Fixed
Retry's ₹420.00 — genuinely comparable, since both are now costed by
attempts actually made, capped at 3, until first recovery).

**Unnecessary-intervention rate — the metric explicitly checked before
trusting this result** (an event where a real action was taken but the
WHOLE campaign still never recovered): deployed policy **10.0%**, down from
single-attempt Model B alone's 26.7% and BELOW Fixed Retry's own 15.0%. A
second/third model-ranked attempt did not trade recovery ₹ for more
wasted contact — it improved both simultaneously here.

McNemar's exact test (paired binary outcome, deployed policy vs. Fixed
Retry): b=3, c=0, p = **0.2500** — NOT significant at this sample size (only
3 discordant event pairs out of 60). 95% bootstrap CI on the ₹ delta:
**[−₹32.43, +₹4,797.25]** — positive point estimate, and the interval only
barely still touches zero at its lower bound.

**Honest verdict — directionally positive, not statistically confirmed at
n=60, and correctly still below the (now apples-to-apples) Oracle upper
bound.** The deployed policy now recovers MORE ₹ than Fixed Retry on this
held-out split, by a wide absolute margin (+₹2,125.24 / +9.1%), and does so
with a BETTER (lower) unnecessary-intervention rate, not a worse one. This
is a genuine reversal from §16b's verdict, achieved by removing the actual
structural cause named there, not by re-tuning a threshold. But at n=60 the
McNemar test does not reach significance (p=0.25) and the bootstrap CI still
touches zero — so lead with **"directionally positive, not statistically
confirmed at n=60"** when citing this result, not the bare +9.1% figure; a
larger held-out population would be needed to confirm significance.

Reproduce: `./venv/bin/python -m evaluation.evaluate_decision_engine_v4`
(same command as §16b — the multi-attempt schedule is now baked into
`evaluate_events_v4`'s own `improved_fallback_policy` AND `oracle_policy`
scoring, so a single run reflects both). All numbers in this section are
superseded by §16d's larger-dataset re-run below — kept here as the accurate
historical record of the 200-subscription state.

## 16d. Model B: larger synthetic training population (data only, no model/policy redesign)

Final pre-submission audit, fourth pass — scoped exclusively to `data/generate_synthetic_dataset.py::DEFAULT_N_SUBSCRIPTIONS`
(200 → 1500). Nothing else changed: same target (`expected_recovery_value_latent`),
same feature schema, same CatBoost hyperparameters (`model/train_latent_target_model.py::CATBOOST_REGRESSOR_PARAMS`,
untouched), same policy logic, same 60/20/20 subscription-level TRAIN/VALIDATION/TEST
split methodology, same seed (42), same generator/archetype/logit coefficients.

**Regenerated population:** 1,500 subscriptions (up from 200) → 2,344 failure
events (up from ~300) → 11,720 candidate-level counterfactual rows (5 per
event). Split: 900/300/300 subscriptions → 1,407/468/469 failure events
(train/validation/test). Class balance (`recovered_within_14d`): 67.4%
overall, 67.7% train, 67.7% validation, 66.1% test — closely matched across
splits, no distribution shift introduced by the resplit. Both raw regen
commands are the existing, unmodified generators:
`./venv/bin/python -m data.generate_synthetic_dataset` then
`./venv/bin/python -m data.generate_counterfactual_dataset` (the latter reads
`DEFAULT_N_SUBSCRIPTIONS` from the former, so both scale together
automatically). Model B retrained with the existing, unmodified
`./venv/bin/python -m model.train_latent_target_model` — same artifact path
(`model/latent_target_artifacts/value/`), same schema
`policy/decision_engine.py::_load_model_safely` already expects.

**Regression (TEST set, `evaluation/evaluate_latent_target_policy.py` Section A):**

| | 200 subs (old) | 1500 subs (new) |
|---|---|---|
| MAE | 94.49 | 162.89 |
| RMSE | 163.05 | 313.38 |
| R² | 0.8707 | 0.7977 |

**Regression got WORSE in absolute terms, not better — reported honestly, not
hidden.** MAE/RMSE roughly doubled and R² dropped. The target's own scale
widened only modestly (mean failed `amount` ₹607→₹625, std ₹798→₹843), which
does not fully explain a ~2× MAE increase — the more likely explanation is
that 1,500 subscriptions expose more genuine population heterogeneity
(broader archetype/plan-tier/city-tier mix) than the SAME fixed model
capacity (`depth=4`, `iterations=500`) fits as tightly as it could fit a
smaller, more homogeneous 200-subscription sample. This pass deliberately
did not retune hyperparameters to chase this back down (out of scope, see
constraints above) — this is exactly why §5 below does not stop at R².

**Candidate ranking (TEST set, ground truth = `expected_recovery_value_latent`,
`evaluation/evaluate_latent_target_policy.py` Section B, `model_b_value` row):**

| | 200 subs (old) | 1500 subs (new) |
|---|---|---|
| Top-1 accuracy | 28.3% | **33.5%** |
| MRR | 0.5425 | **0.5786** |
| Mean rank (of 5) | 2.55 | **2.39** |
| Avg regret (₹) | 45.88 | 73.78 |
| Avg regret as % of avg event value | 13.6% | 15.2% |

Ranking genuinely improved on every rank-based metric (top-1, MRR, mean
rank) — the model picks the objectively-best candidate more often and ranks
it higher when it doesn't. Average regret in absolute ₹ rose (again tracking
the wider value range), and as a fraction of typical event value it rose
slightly too (13.6%→15.2%) — reported alongside the improvement, not
omitted, per this section's own instruction not to stop at one metric.

**Economic evaluation — deployed policy UNCHANGED (`policy/decision_engine_v4.py`'s
frozen `margin_threshold=0.0, fallback_mode=keep_model_unless_rule_has_clear_advantage,
fallback_advantage_threshold=0.0`), only Model B retrained, `evaluation/evaluate_decision_engine_v4.py::evaluate_events_v4`
called directly against this exact frozen config (see caveat below for why
NOT the script's own `main()`):**

| Policy | Realized ₹ (old, n=60) | Realized ₹ (new, n=469) | Recovery rate (old→new) | Net value (old→new) | Unnecessary-intervention rate (old→new) |
|---|---|---|---|---|---|
| Fixed Retry | 23,296.10 | 272,197.42 | 85.0%→85.1% | 22,326.31→262,048.56 | 15.0%→14.9% |
| Rule-Based | 21,431.15 | 229,573.46 | 76.7%→68.4% | 20,609.17→221,683.90 | 23.3%→31.6% |
| Model B alone (single attempt) | 21,278.18 | 219,859.19 | 73.3%→70.2% | 20,476.01→212,325.51 | 26.7%→29.9% |
| **Deployed policy (multi-attempt)** | **25,421.34** | **285,067.48** | **90.0%→89.6%** | **24,396.40→274,944.89** | **10.0%→10.5%** |
| Oracle (multi-attempt upper bound) | 25,943.30 | 285,657.33 | 91.7%→90.6% | 24,966.04→275,815.82 | 8.3%→9.4% |

Delta vs. Fixed Retry (deployed policy): old **+₹2,125.24 / +9.1%**, new
**+₹12,870.06 / +4.7%** — a smaller PERCENTAGE lift (larger population, more
Fixed-Retry-favorable events included), but a real one, and see the
statistics below for why it is now far more trustworthy. Regret vs. Oracle
(deployed policy): old ₹521.96 total / ₹8.70 per event, new ₹589.85 total /
**₹1.26 per event** — the deployed policy now sits MUCH closer to its own
upper bound on a per-event basis.

**McNemar's exact test / bootstrap CI (deployed policy vs. Fixed Retry):**

| | 200 subs (old, n=60) | 1500 subs (new, n=469) |
|---|---|---|
| McNemar p-value | 0.2500 (NOT significant) | **0.0001 (highly significant)** |
| Discordant pairs (b, c) | 3, 0 | 25, 4 |
| Bootstrap 95% CI on ₹ delta | [−32.43, +4,797.25] (touches zero) | **[+3,946.96, +24,820.88] (fully positive)** |

**This is the real, substantive improvement from more data — not the raw ₹
delta, the STATISTICAL CONFIDENCE behind it.** At n=60 the deployed policy's
edge over Fixed Retry could not be distinguished from noise (§16c's own
honest verdict: "directionally positive, not statistically confirmed").
At n=469 the same underlying effect is now clearly significant on both
tests. The structural safety guarantee also held perfectly under the new
model: `fallback_count=0` (Model B's own top pick was used in 469/469 test
events, `n_decisions_changed_by_fallback_vs_model_b_alone=0`) — the
`keep_model_unless_rule_has_clear_advantage` mode's "can never do worse than
Model B alone" property (§16b's STRUCTURAL FINDING) is a mathematical
property of the mechanism, not data-dependent, and it reproduced exactly as
designed with a completely different trained model.

**CAVEAT — a real finding this pass surfaced but explicitly did NOT act on.**
`evaluate_decision_engine_v4.py::main()`'s own validation-only search
(unmodified, pre-existing code — see §16b) was re-run against the new
validation split as part of routine reproduction, and it picked a DIFFERENT
configuration than the currently-frozen one: `margin_threshold=100.0,
fallback_mode=always_fallback_when_below_margin` — literally the OLD,
previously-rejected "blind swap" mechanism §16b's own economic correction
existed to move away from (see that section's ECONOMIC-CORRECTION FINDING).
This happened because the retrained model shifted the validation-set
landscape enough that, on this specific validation split, blind-swapping to
Rule-Based happens to score highest by realized ₹ — a real, honestly
disclosed property of the new model+data, not a bug in the search. **This
pass deliberately did NOT follow that suggestion**: `policy/decision_engine_v4.py`'s
frozen defaults were never touched, and every number in the two tables above
uses the frozen config, evaluated directly (bypassing `main()`'s own
auto-selected config for exactly this reason) — "do not change the policy"
and "do not tune against TEST" both take precedence over chasing a better-
looking validation-search result. Whether the frozen config should be
re-validated on this larger population is a real open question, explicitly
left for a future, dedicated pass — never decided inside this data-only one.

**Honest verdict.** Regression metrics got measurably worse in absolute
terms; ranking metrics (top-1, MRR, mean rank) got measurably better;
per-event economic advantage over Fixed Retry narrowed slightly in ₹ terms
but tightened enormously in statistical confidence (not significant →
highly significant); regret vs. Oracle per event improved substantially.
Model B already beat Fixed Retry before this change (§16c) and still does
after it — what changed is that the finding is now backed by 469 held-out
events instead of 60, with a McNemar p-value of 0.0001 and a bootstrap CI
that no longer touches zero. This remains a **SYNTHETIC COUNTERFACTUAL
EVALUATION** regardless of population size — none of this measures real
Razorpay recovery performance, and Model B is not claimed to be
"production accurate" by any of these numbers.

Reproduce: `./venv/bin/python -m data.generate_synthetic_dataset && ./venv/bin/python -m data.generate_counterfactual_dataset && ./venv/bin/python -m model.train_latent_target_model && ./venv/bin/python -m model.train_candidate_model && ./venv/bin/python -m model.train_ranking_model && ./venv/bin/python -m evaluation.evaluate_latent_target_policy`
(`evaluate_latent_target_policy.py` compares Model B against the earlier
candidate-aware/ranking model generations too, so it needs both of their
artifacts on disk as well — verified end to end from a genuinely fresh
clone, see BUG-3 in the pre-submission audit report)
for regression/ranking; the economic table above requires calling
`evaluate_events_v4(test_df, model, FROZEN_CONFIG)` directly with the frozen
config (not `evaluate_decision_engine_v4.py`'s own `main()`) for the reason
stated in the caveat.

## 16a. Unified ML held-out evaluation

Separate from §16's Model-B/subscription-only counterfactual evaluation
above: `./venv/bin/python -m evaluation.evaluate_unified_model` backtests
the unified model (§3c) against a naive fixed-candidate baseline on its own
held-out **TEST** entities (never touched during training) — again entirely
**SYNTHETIC, SIMULATED** data; `recovered_amount` below means "the
simulated outcome for the candidate actually selected," not a real
Razorpay confirmation.

Data/target/tuning correction (final pre-submission audit, on top of the
original §3c work): the synthetic training-data generator was rewritten to
use logit-space, multi-feature nonlinear interactions (segment × urgency,
prior-failures × urgency with an opposing sign, failure-reason × urgency,
plus a domain-specific interaction per domain), the training population
raised from 240 to 900 entities/domain (12,600 total rows), and
hyperparameters re-selected via a 5-config grid, validation-only (never
touching test). **Held-out test ROC-AUC improved from 0.550 to 0.630**
(PR-AUC 0.597; validation AUC 0.626). Evaluation was also extended with a
REAL rule-baseline comparison (`policy/checkout_rules.py` /
`policy/receivables_rules.py` — the only two domains where a stateless
rule-baseline call is well-posed; mandate/promise/payment_failed's rule
modules are stateful and excluded, documented in
`evaluation/evaluate_unified_model.py`, not faked), oracle-based top-1
accuracy, NDCG, and regret.

| Domain | ML recovery rate | ML net ₹ (test split) | vs. random | vs. real rule baseline | vs. naive first-candidate | Top-1 acc. / NDCG |
|---|---|---|---|---|---|---|
| payment_failed (Payment Link) | 56.3% | ₹472,566 | tied | n/a (single candidate) | tied | n/a |
| checkout_abandoned | 51.9% | ₹457,175 | **+14.7%** (net +₹45,613) | **+15.1%** (net +₹114,851) | −1.5% (net −₹14,173) | 23.0% / 0.989 |
| mandate_failed | 49.6% | ₹387,444 | +6.3% | n/a (stateful, excluded) | +8.1% (net −₹10,859) | 46.7% / 0.985 |
| receivable_overdue | 58.5% | ₹553,326 | **+23.4%** (net +₹110,423) | **+17.9%** (net +₹110,615) | +2.6% (net +₹34,062) | 86.7% / 0.994 |
| promise_to_pay_broken | 54.1% | ₹455,652 | +9.0% | n/a (stateful, excluded) | −8.8% (net −₹45,580) | 54.1% / 0.983 |
| **Overall** | 56.6% | ₹2,326,163 | (mixed by domain) | (wins where computable) | −1.5% (net −₹36,550) | total regret vs. oracle ₹74,621 |

**Honest verdict: the corrected model clearly beats `random_candidate` and
the real rule-engine baseline everywhere that comparison is well-posed, but
remains roughly tied with (slightly behind) the naive `first_candidate`
baseline overall.** This is a genuine, held-out-TEST-verified improvement
over the original 0.550 AUC / marginal-lift model, not an unambiguous win —
reported exactly as measured. See §3c and §19 for the full methodology and
limitations.

## 17. Failure handling

Every boundary below was verified in this FIX pass — either by the
existing test suite (432/432 passing at the time — see the current count in
§15) or by direct execution against the running system:

| Boundary | Handling | Verified |
|---|---|---|
| Malformed webhook body | `400 malformed json body`, nothing stored | test + code review |
| Invalid/missing/tampered signature | `400 invalid signature`, nothing stored, constant-time comparison | test + code review |
| Duplicate webhook delivery | `200`, acknowledged, not re-stored (unique constraint + query-before-insert) | test + code review |
| Missing required webhook fields | `400`, rejected before storage | test |
| Unsupported webhook event type (e.g. `subscription.charged`) | Stored, orchestration skipped (`orchestration=skipped_unsupported_event_type`) | test |
| Webhook missing `payload.subscription` | Stored, orchestration skipped (`orchestration=skipped_missing_subscription_id`) | test |
| Webhook missing `payload.payment` | Stored gracefully, `error_reason=None`, no traceback | test |
| Classification/orchestration/DB failure after a webhook is already stored | Raw event stays stored, failure audited (`orchestration_failed_after_storage`), `200` returned (storage succeeded), safely reprocessable via `scripts/reprocess_raw_events.py` | test |
| LLM failure reached via a live webhook | Payment decision unaffected, deterministic fallback used, `orchestration=completed` | test |
| Model artifact missing/corrupt | Falls back to rule-based tier, never crashes, `decision_source` records which tier decided | test + code review |
| Malformed/implausible model prediction | Treated as malformed (NaN/negative/>2× amount), triggers fallback tier | test |
| Missing/empty CSV or evaluation report | Dashboard renders an explicit empty state, never raises | test + direct execution |
| Malformed report JSON | `load_report` returns `None`, dashboard shows empty state | direct execution (deliberately corrupted a report file in this audit) |
| LLM unavailable / timeout / empty response / invalid JSON / schema-invalid / unexpected exception | Deterministic, non-fabricating fallback per job; payment decision unaffected | test + direct execution (all 5 orchestrator demo scenarios re-run) |
| Missing `ANTHROPIC_API_KEY` with `LLM_PROVIDER=anthropic` | Logs a warning, falls back to the mock provider | test |
| Unmapped/unknown decline reason | `unmapped` bucket, routed to `NO_ACTION`, never guessed | test |
| Compliance block (opt-out, cancellation, contact cap) | Logged, halts that action specifically; payment/communication gated independently | test + direct execution |
| Invalid/expired/low-confidence/duplicate promise-to-pay reply | Never overrides retry timing; original policy candidate used unchanged | test |
| Promise-to-pay date outside the 14-day recovery horizon | Compliance rejects the promise's own timing; falls back to the original, already-valid candidate rather than blocking the payment | test |
| Unknown event ID requested in the UI | Returns `None`, empty state rendered, no traceback | direct execution |

No raw traceback is ever surfaced to a normal dashboard user in the pages
above; the one intentional exception is the "Run full test suite now"
developer control on the System/Demo page, which deliberately surfaces raw
pytest output as its designed function.

## 18. Security / auditability

- Webhook signatures are verified manually via `hmac.compare_digest`
  (constant-time) over the exact raw request body — never a re-serialized
  copy — per Razorpay's documented scheme.
- No secret (API key, webhook secret, raw auth header) is ever logged, ever
  written to `audit_log` / `llm_invocations`, or ever included in an
  exception message — `AnthropicLLMClient` deliberately raises `type(exc).__name__`
  only, never `str(exc)`, since an SDK exception's string form can embed
  request details. Verified by a dedicated test that forces a fake secret
  into an exception message and asserts it never surfaces anywhere.
- Every decision the system makes — including deciding to do nothing, a
  compliance block, or a duplicate/skipped action — writes an `audit_log`
  row with an explicit `actor` (`system` / `rule` / `classifier` / `policy`
  / `compliance` / `llm` / `orchestrator`). Nothing is silently discarded.
- Idempotency is enforced at the database layer for webhook events (unique
  constraint on `x-razorpay-event-id`) and at the application layer for
  classification and policy decisions (query-before-insert, keyed on
  `raw_event_id` / `event_id`).
- **Contact-hours gate** (`policy/contact_hours.py`, final pre-submission
  correction): the compliance gate can visibly *refuse* a communication
  action scheduled outside a configurable, timezone-aware window — default
  09:00–21:00 `Asia/Kolkata` (TRAI's own commercial-communication window; a
  project guardrail, not a claim of TRAI/DPDP/RBI regulatory compliance).
  RBI's Fair Practices Code — a stricter, lending-specific 08:00–19:00 window
  with real enforcement history — is deliberately NOT the default: this
  project is subscription/receivables recovery, not lending, so RBI's
  lending-specific code has no direct jurisdiction here and no claim is made
  that it does; both the start/end times are fully configurable via
  `CONTACT_HOURS_START`/`CONTACT_HOURS_END` if a deployment needs to match a
  stricter window. Checks the candidate's own SCHEDULED time
  (`selected_candidate_datetime`), never the current process clock; disabled
  entirely via `CONTACT_HOURS_ENABLED=false`. Wired into both
  `policy/compliance.py`'s and `policy/compliance_v2.py`'s communication gate
  (not the payment/retry gate — a backend retry API call does not itself
  contact a customer). See `tests/test_contact_hours.py` (before/inside/after
  window, timezone conversion, DST/boundary-crossing cases) and
  `tests/test_compliance.py::TestContactHoursGate` /
  `tests/test_compliance_v2.py::test_candidate_outside_contact_hours_blocks_communication_only`
  for the wiring proof. This exposed a genuine pre-existing scheduling
  defect: `next_month_end_after` (`data/generate_synthetic_dataset.py`, also
  used live by `policy/retry_candidates.py::generate_candidates`) hardcoded
  18:00 UTC = 23:30 IST — always outside any reasonable contact-hours
  window. Corrected to 14:00 UTC = 19:30 IST (still the latest-in-the-day of
  the 5 candidate times, preserving the "evening reminder" intent).
- **Defer, don't terminate** (`policy/contact_hours.py::next_contact_hours_start`,
  `recovery/retry_sweep.py`, final pre-submission audit): a communication
  blocked SPECIFICALLY by contact-hours (never opt-out/consent/duplicate —
  those are not timing problems) is no longer a dead end. Compliance now
  also returns `communication_deferred_until` — the next window's opening
  time — and the orchestrator records `final_status=COMMUNICATION_DEFERRED`
  instead of `COMMUNICATION_BLOCKED`. `recovery/retry_sweep.py`'s same
  periodic sweep (below) fires the deferred communication once that time
  arrives, using the exact same `llm/service.py` call path as an on-time
  communication — so a late-evening or overnight failure gets its nudge
  delayed a few hours, not lost outright. See
  `tests/test_contact_hours.py::TestNextContactHoursStart`,
  `tests/test_compliance.py`'s `test_contact_hours_block_sets_deferred_until_the_next_window`
  / `test_opt_out_block_never_sets_deferred_until`, and
  `tests/test_retry_sweep.py::TestFireOneDeferredCommunication`.
- **Multi-attempt persistence** (`policy/decision_engine_v4.py::build_retry_schedule_from_decision`,
  `recovery/retry_sweep.py`, final pre-submission audit): the deployed
  subscription policy can now make up to `guardrails.MAX_RETRY_ATTEMPTS` (3)
  scheduled attempts per event — the same persistence Fixed Retry always had
  — ranked by Model B's own predicted value rather than a fixed calendar
  cadence. `recovery/retry_sweep.py`'s periodic sweep
  (`ENABLE_RETRY_SWEEP_SCHEDULER`) advances one step at a time and stops
  early the moment a real `payment.captured` webhook confirms recovery; every
  advance is recorded/audited only, never a live Razorpay call — see §16c for
  the economic result this produced and §4 for the promise-to-pay sweep this
  reuses the exact same in-process loop pattern from. Before every scheduled
  advance, `_subscription_still_eligible` also re-checks durable opt-out
  (`PolicyDecision.customer_opted_out`) and the subscription's most recent
  classification across every `policy_decisions` row — an opt-out or
  cancellation recorded on a LATER event permanently suppresses every
  remaining attempt, never just skips one and retries later.

## 19. Known limitations

**Confirmed in this FIX pass — read before treating any claim above as
"fully wired":**

1. **The deployed policy (policy-v4) now recovers MORE realized ₹ than the
   Fixed Retry baseline on held-out TEST, but this is NOT yet a
   statistically confirmed result at this sample size** (§16b then §16c,
   final pre-submission audit, two passes): the original −₹3,298.87 / −14.2%
   gap (McNemar p=0.0117, bootstrap CI excluding zero) traced first to a
   blind-swap fallback mechanism (fixed in §16b, narrowing the gap to
   −₹2,017.92 / −8.7%), then to the genuinely structural remaining cause —
   Fixed Retry's three scheduled attempts per event vs. every other policy's
   one. §16c gives the deployed policy the same multi-attempt persistence
   (`policy/decision_engine_v4.py::build_retry_schedule_from_decision`,
   `recovery/retry_sweep.py`, ranked by Model B's own value predictions,
   capped at `guardrails.MAX_RETRY_ATTEMPTS`), wired into both evaluation and
   the live path, and reports **+₹2,125.24 / +9.1% vs. Fixed Retry** with a
   BETTER (lower, 10.0% vs. 15.0%) unnecessary-intervention rate — but
   McNemar's test on this n=60 population is p=0.2500 (not significant,
   only 3 discordant pairs) and the 95% bootstrap CI ([−₹32.43, +₹4,797.25])
   still barely touches zero. Reported as a real, mechanistically-explained,
   reproducible improvement — not oversold as statistically proven
   superiority. See §16c. Neither finding was chased or engineered — §16b's
   correction was scoped only to fallback-selection methodology using
   validation data exclusively; §16c's fix directly addresses the exact
   structural cause §16b itself named, with test run once after the change.
   **Self-review caught two follow-on issues before this was reported, both
   fixed in §16c**: (a) Oracle was still scored as a single pick after the
   deployed policy and Fixed Retry both got multi-attempt scoring, so it
   briefly (and incorrectly) scored BELOW the deployed policy in an earlier
   draft — fixed by extending Oracle to the same up-to-3-attempt scoring,
   ranked by the same `latent_value` proxy it always used; Oracle is now
   correctly the upper bound again (₹25,943.30 / 91.7%, above the deployed
   policy's ₹25,421.34 / 90.0%). (b) `recovery/retry_sweep.py` re-checks
   durable opt-out/cancellation state before every scheduled follow-up
   attempt or deferred communication, and permanently suppresses the rest of
   the schedule the moment either disqualifies the subscription — a real gap
   opened by spreading attempts across real elapsed time, closed before
   shipping, not left for a later pass.
2. **Razorpay's fee take is modeled at one uniform rate** (`policy/economics.py`,
   ≈2.36% of recovered GMV, gross — the specification's disclosed "2% + 18%
   GST" domestic card rate), applied to every recovered rupee regardless of
   payment instrument. The specification itself flags UPI's exact take-rate
   as an unresolved inconsistency in Razorpay's own public materials and
   instructs against stating one clean UPI figure — this project follows
   that instruction rather than inventing an uncited second rate, which
   means the fee figures in §16/the dashboard are a documented
   simplification for non-card instruments, not a precise per-instrument
   fee model.
3. **Fixed Retry's T+2 attempt has no simulated outcome of its own.**
   `policy/retry_candidates.py::CANDIDATE_TYPES` (and the counterfactual
   dataset generated against it) was never extended with a 6th "day+2"
   candidate — doing so now would mean regenerating the synthetic dataset
   with a new outcome definition, out of scope for the baseline-fidelity
   fix. T+2 is genuinely scheduled (and costed) but contributes zero
   incremental recovery probability in the evaluation's scoring — a
   conservative simplification, disclosed here and in
   `policy/baselines.py`'s own docstring, not an invented estimate.
4. **The deployed model was not built the way the specification's Model 1
   describes.** The specification calls for "a calibrated binary classifier
   predicting P(recover within 14 days | ...)". The project's first
   attempt at exactly that (the calibrated failure-time-only classifier)
   topped out at test ROC-AUC 0.566 — a weak result honestly diagnosed and
   reported, not hidden — and after two further iterations (the
   candidate-aware and pairwise ranking models) also failed to beat random
   guessing on within-event ranking, the project pivoted to a CatBoost
   regressor trained directly on the synthetic generator's own latent
   expected-value target, not the noisy observed outcome. This is
   disclosed thoroughly above as a deliberate, diagnosed pivot — but it
   means the deployed model's strong-looking fit (R²=0.7977 test /
   0.8459 validation on the current 1,500-subscription population — see
   §16d; this was R²≈0.87 on the earlier 200-subscription population,
   corrected here so this section never again disagrees with the
   evaluation section it's summarizing) partly reflects reconstructing the
   generator's own formula, and the specification's originally-scoped
   Model 1 architecture was never actually shipped as the production
   model.
5. **All evaluation numbers are synthetic.** Nothing in this repository has
   touched a real Razorpay production account, a real customer, or a real
   message-sending API. The dataset is archetype-generated with
   probabilistic (not deterministic) labels, by design.
6. **Small test set (n=60 events).** Every ₹/rate gap reported against a
   baseline in §16 without its own McNemar/bootstrap result (i.e. every
   comparison other than deployed-vs-Fixed-Retry) remains directional,
   synthetic-benchmark evidence, not a statistically confirmed result at
   this sample size — n=60 is still small in absolute terms even where a
   test does clear p<0.05.
7. **`consent_for_communication` is an unimplemented placeholder** in the
   compliance gate (defaults to allowed) — there is no real consent-tracking
   system in this project.
8. **Compliance checks here are project guardrails, not legal compliance.**
   Nothing claims to satisfy DPDP/TRAI/RBI or any other real regulatory
   regime — stated verbatim in `policy/compliance.py`'s own docstring.
9. **No live payment execution loop exists to observe a real promise
   fulfilled/broken outcome.** `PromiseToPay.status` therefore models only
   VALID/LOW_CONFIDENCE/INVALID_DATE/EXPIRED/SUPERSEDED — never a
   FULFILLED/BROKEN state — since this project makes no live Razorpay
   payment call to actually observe one against.
10. **The unified ML model (§3c) does not clearly beat a naive
    fixed-candidate baseline** on its own held-out synthetic test split —
    see §16a for the honest, mixed per-domain numbers. Do not cite this as
    an ML win.
11. **Every real, live Razorpay Test Mode Payment Link failure verified
    against this system so far returned a generic, classifier-unrecognized
    `error_reason` (`"payment_failed"`)** — real cards were used, but none
    triggered a specific reason like `insufficient_fund`. The unified
    model is genuinely CONSULTED for these (proven live, `ml_consulted=True`
    with a real score in the audit trail), but the eligibility gate
    correctly overrides it to `NO_ACTION`, so the "ML USED" (ML's own
    candidate is final) state has not yet been observed live on this
    specific domain with real data — only via the trained artifact
    directly and via the other 4 domains, which have.
12. **`LLM_PROVIDER=ollama` in the local `.env` is this session's working
    configuration**, not a hardcoded requirement — `mock`/`anthropic`/
    `gemini` remain fully supported (§11); the mock provider stays the
    default in `.env.example` and in every test.

## 20. Final feature status

| Feature (per the original specification) | Status |
|---|---|
| Webhook receiver, HMAC verification, idempotent storage | **DONE** — real Razorpay webhook mechanics, tested |
| Deterministic failure classification | **DONE** |
| Synthetic dataset (archetypes, probabilistic labels, ID-level split) | **DONE** |
| Calibrated recovery-likelihood / candidate-time model | **PARTIAL** — a value-regressor (Model B) is deployed and beats baselines on latent value, but loses to Fixed Retry on realized ₹; the spec's originally-scoped "calibrated binary classifier" was attempted (the calibrated failure-time-only classifier) but never exceeded weak (0.57) AUC and was superseded. See §16, §19. |
| Deterministic compliance gate | **DONE** |
| Payment action (recorded, never live) | **DONE** |
| LLM: outreach microcopy | **DONE**, wired into the live orchestrator, including a `hard_decline` payment-method-update nudge (FIX #3) |
| LLM: promise-to-pay parsing | **DONE** — parsed (LLM), validated (deterministic, `policy/promise_to_pay.py`), persisted (`PromiseToPay`), and capable of overriding the model's chosen retry timing through compliance in `recovery/orchestrator.py` (FIX #1). No live payment execution loop exists to observe a real fulfilled/broken outcome — see §19 item 9. |
| LLM: batch-level report explanation | **DONE**, used by the dashboard |
| 3 baselines (No Recovery, Fixed Retry, Rule-Based) | **DONE** — Fixed Retry now schedules the spec's full T+1/T+2/T+3 "silent, same-channel, then gives up" cadence, and Rule-Based now attaches the spec's WhatsApp-nudge + 3-day-follow-up communication dimension (`policy/baselines.py`, deterministic fixed templates, no LLM). See §16, §19 item 3 for the one disclosed simplification (T+2 has no outcome data of its own). |
| 7 evaluation metrics | **DONE** — recovery rate, ₹ recovered (split into merchant GMV / Razorpay fee take / net, `policy/economics.py`), incremental ₹, cost-per-recovery, unnecessary-intervention rate, and customer-contact rate (`evaluate_decision_engine_v4.py`'s `contact_and_intervention_metrics` / `cost_per_recovery`) are all computed on the authoritative report; McNemar's-test/bootstrap-CI significance on the headline deployed-vs-Fixed-Retry comparison (`evaluation/statistics.py`) is implemented and now finds the gap significant. See §16. |
| Audit trail | **DONE** |
| End-to-end automatic webhook→orchestration wiring | **DONE** (FIX #2) — a stored, verified webhook event continues automatically into classification + full orchestration in the same request; `scripts/reprocess_raw_events.py` remains available for manual re-processing of a failed/pre-fix event. |
| Streamlit dashboard | **DONE** — 8 pages, verified via `AppTest` and direct execution, including promise-to-pay and hard-decline-communication detail |
| Failure-mode demonstrations (Section 13 of the spec) | **DONE** — all 5 required scenarios (insufficient_fund recovery, hard decline + nudge, promise-to-pay override, LLM failure, webhook ingestion) plus a bonus opt-out scenario, demonstrated and tested |
| Unified ML model across all 5 revenue-risk domains (§3c) | **DONE** — real trained `CatBoostClassifier`, live-loaded via one cached function, real inference verified live for all 5 domains including a real Payment Link webhook; held-out test AUC 0.630 (up from 0.550), clearly beats random and the real rule baseline, roughly ties the naive first-candidate baseline (§16a, §19) — disclosed, not hidden |
| Payment Link / one-time-payment generalization (`subscription_id=NULL`) | **DONE** — no longer dead-ends; reaches classification, the unified model (always consulted), policy, and compliance. Verified against 3 real Razorpay Test Mode Payment Link failures |
| Dashboard ML-status distinction (USED / CONSULTED_OVERRIDDEN / FALLBACK) | **DONE** — `ui/data.py::get_live_revenue_pipeline_snapshot`, Overview's "Latest revenue-risk event" card |
| Contact-hours compliance gate | **DONE** (final pre-submission correction) — README previously claimed this without an implementation; now real, timezone-aware, configurable, enforced on the communication gate of both `policy/compliance.py` and `policy/compliance_v2.py`. See §16b, §18, `tests/test_contact_hours.py`. |
| Fresh-clone dashboard startup | **DONE** (final pre-submission correction) — `streamlit run ui/app.py` on a brand-new checkout with no FastAPI process ever run and an empty/nonexistent DB file now initializes schema itself (`ui/data.py::ensure_schema_initialized`, wrapping the existing `app/db.py::init_db()`) instead of crashing with "no such table". See §13, `tests/test_ui.py::TestFreshCloneDatabaseInitialization`. |
| Deployed subscription-policy economic gap vs. Fixed Retry | **FIXED, directionally — not yet statistically confirmed** (final pre-submission correction, two passes) — §16b's fallback-selection fix narrowed the gap from −₹3,298.87/−14.2% to −₹2,017.92/−8.7%; §16c's multi-attempt persistence fix then closed the remaining structural cause, producing +₹2,125.24/+9.1% vs. Fixed Retry with a BETTER unnecessary-intervention rate — but McNemar p=0.2500 (not significant at n=60) and the bootstrap CI still touches zero. See §16c. |
| Multi-attempt persistence (deployed subscription policy) | **DONE** (final pre-submission audit) — `policy/decision_engine_v4.py::build_retry_schedule_from_decision` + `recovery/retry_sweep.py`, up to `guardrails.MAX_RETRY_ATTEMPTS` (3) attempts per event, ranked by Model B's own value predictions; wired into both evaluation and the live path (`ENABLE_RETRY_SWEEP_SCHEDULER`). See §16c, §18. |
| Deferred (not lost) communication outside contact hours | **DONE** (final pre-submission audit) — `policy/contact_hours.py::next_contact_hours_start` + `recovery/retry_sweep.py`; a pure contact-hours block now returns `final_status=COMMUNICATION_DEFERRED` with a real next-window time, fired automatically once due, instead of a dead-end `COMMUNICATION_BLOCKED`. See §18. |
| Oracle scored apples-to-apples with multi-attempt policies | **DONE** (final pre-submission audit, caught in self-review) — `oracle_policy` is now ALSO scored with up to `guardrails.MAX_RETRY_ATTEMPTS` scheduled attempts (same ranking metric, `latent_value`, as before); previously Oracle stayed single-attempt after Fixed Retry and the deployed policy both got multi-attempt scoring, letting the deployed policy numerically (and incorrectly) beat its own upper bound. See §16c, `tests/test_decision_engine_v4.py::test_oracle_realized_value_is_never_below_the_deployed_policy`. |
| Re-check-before-acting for scheduled follow-up attempts | **DONE** (final pre-submission audit) — `recovery/retry_sweep.py::_subscription_still_eligible` re-checks durable opt-out (new `PolicyDecision.customer_opted_out`, sticky) and the most recent classification across every `policy_decisions` row for the subscription before every scheduled attempt/deferred communication, and permanently suppresses the rest of the sequence if either disqualifies it. See §16c, §19, `tests/test_retry_sweep.py::TestReCheckBeforeActing`. |

## 21. Manual setup remaining

To go beyond the offline demo (already fully functional with no setup
beyond §7):

- Generate real Razorpay Test Mode API keys and a webhook secret, and run
  a zrok tunnel, if you want to demonstrate a genuine live webhook delivery
  (§10). Not required for tests, demos, or the dashboard.
- Set `LLM_PROVIDER=anthropic`/`gemini` with a real API key, or
  `LLM_PROVIDER=ollama` with a local Ollama server running (no API key —
  see §11), if you want genuinely LLM-generated (rather than deterministic
  mock) outreach copy, promise-to-pay parsing, or report narration.
- McNemar's test, the bootstrap CI, Razorpay fee-take modeling, and the
  Fixed Retry/Rule-Based baseline-fidelity fix are all now implemented
  (§16, §19 items 2–3); a real per-instrument fee schedule and a genuinely
  simulated T+2 outcome (vs. this project's one uniform card-rate
  assumption and T+2's zero-incremental-probability treatment) remain
  scoped, understood simplifications, not unknowns.

---

Done By : Arun Vasanth Selwyn Sudhaker 
