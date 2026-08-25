# Adaptive Payment Recovery Agent

**Razorpay AI Buildathon — Track 3: AI Revenue Recovery**

An offline-verifiable prototype that replaces Razorpay Subscriptions' blind
fixed-interval retry (retry once/day for 3 days, regardless of *why* a charge
failed) with a per-failure decision: classify the failure reason
deterministically, score candidate retry times with a calibrated model,
enforce a deterministic compliance gate, generate outreach copy and parse
promise-to-pay replies with an LLM (exactly 3 jobs, never a decision-maker),
and report recovered-₹ against Razorpay's own real baseline on a held-out
synthetic batch.

Scope, per the original project specification
(`~/Downloads/razorpay-track3-project-specification.md`): **Razorpay
Subscriptions only, `insufficient_fund` decline reason only.** Everything
below this line is a single integrated system, not a day-by-day changelog —
the full build diary is preserved in the [Appendix](#appendix-day-by-day-development-log)
for anyone who wants the detailed, dated reasoning behind each decision.

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
baseline) + CatBoost (main candidate-time value model), Anthropic Claude API
(the 3 LLM jobs, via `anthropic` SDK — offline mock provider by default),
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
| `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` | `app/config.py` (loaded, currently unused by any request path — no live Razorpay API call exists anywhere in the codebase) | Configured in local `.env` (not committed); empty in `.env.example` | Only if you intend to create real Test Mode subscriptions in the Dashboard |
| `RAZORPAY_WEBHOOK_SECRET` | HMAC verification (`app/webhook_security.py`) | Configured in local `.env`; `.env.example` ships a placeholder value, must be replaced with the Dashboard-issued secret for real deliveries | Required to receive and verify a **real** Razorpay webhook. Not required for tests, demos, or the dashboard — all run fully offline against synthetic/mock data. |
| zrok token / tunnel | Exposes local FastAPI to a public HTTPS URL Razorpay's Dashboard will accept (`ngrok.io` is explicitly blacklisted by Razorpay) | Not stored in this repo — a per-developer, per-session manual step | Required only for a live webhook demo (§10). Not required for anything else. |
| `ANTHROPIC_API_KEY` | `llm/client.py::AnthropicLLMClient`, only reached when `LLM_PROVIDER=anthropic` | Empty in `.env.example`; `LLM_PROVIDER` defaults to `mock`, which never reads this value | Required only to see real Claude-generated output instead of the deterministic mock. Test suite, demos, and dashboard all default to mock and need no key. |
| Trained model artifacts | `model/latent_target_artifacts/`, `model/artifacts/`, etc. | **Not committed** (`.gitignore` excludes all `model/*_artifacts/` and `evaluation/reports/`) — present only in this working copy because training/evaluation were already run here | **Required manual step on a fresh clone**: run `model/train.py`, `model/train_candidate_model.py`, `model/train_ranking_model.py`, `model/train_latent_target_model.py`, then the matching `evaluation/evaluate_*.py` scripts, before the dashboard/demo will show real (non-fallback) model output. The orchestrator degrades gracefully if skipped (falls back to the rule-based tier), but the "current model" evaluation numbers in §16 won't exist until these are run. |
| Generated synthetic dataset | `data/raw/*.csv`, `data/processed/*.csv` | Present and committed — the dataset itself, unlike the trained artifacts above, is tracked in git | Nothing further needed; regenerate via `data/generate_synthetic_dataset.py` / `data/generate_counterfactual_dataset.py` only if you want a fresh draw |

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

Full troubleshooting table preserved in the
[Day 1 appendix section](#day-1).

## 11. LLM configuration

```bash
# .env
LLM_PROVIDER=mock          # default — offline, deterministic, no key needed
ANTHROPIC_API_KEY=         # only read when LLM_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-sonnet-5
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

**432 tests, 432 passing** (verified by direct execution in this FIX pass).
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
no-secrets-in-logs. `pip check` is clean.

## 16. Evaluation

All evaluation numbers are **SYNTHETIC COUNTERFACTUAL EVALUATION** — computed
against `data/raw/counterfactual_outcomes.csv`, a hand-designed simulation
this project's own generator authored. **None of it measures real Razorpay
recovery performance.** The dashboard and every report JSON label this
explicitly and consistently.

**Baselines** (`policy/baselines.py`) — **verified against the specification
in this audit pass, with two real deviations found:** No Recovery (never
acts, matches spec). Fixed Retry always selects the `plus_1_day_morning`
(T+1) candidate — the specification defines Fixed Retry as "Razorpay's real
policy: silent auto-retry once/day for 3 days"; the codebase implements a
single T+1 decision, never T+3 (`plus_3_days` exists as a candidate type
elsewhere in the system but this baseline never selects it). Rule-Based
implements only the timing half of its spec definition (payday proximity
≤2 days → `payday_window`, else → `plus_1_day_morning`) — the specification
additionally calls for "one WhatsApp nudge, one follow-up 3 days later";
no WhatsApp/SMS/voice channel exists anywhere in the codebase (`policy/costs.py`'s
`whatsapp_cost`/`sms_cost`/`voice_cost` are unused `Rs0` placeholders), so
this half of the baseline's definition was never built. Both deviations are
disclosed in the appendix's Day-5 section but without being framed as
divergence from the spec's literal baseline definitions — flagged here for
precision.

**Current model** (the one the live orchestrator and dashboard actually
use): "Model B", a CatBoost regressor predicting `expected_recovery_value_latent`
directly (₹, not a probability), selected in Day 8 over five earlier
model/objective iterations (Day 4's plain classifier, Day 6's candidate-aware
classifier, Day 7's pairwise ranker, and Model A — a companion probability
regressor) specifically because it was **the only one of six trained models
that beat Fixed Retry on the noise-free latent economic ground truth**. The
full diagnostic trail — including two honestly-reported negative results
(Day 6/7's ranking models scoring *worse than random* top-1 accuracy) — is
in the [Appendix](#day-6) and is the strongest evidence in this repo of
genuine, undodged empirical iteration.

**Headline result, test set (n=60 held-out events), from
`evaluation/reports/decision_engine_v4_evaluation.json`
(the authoritative report — the only one reflecting the actually-deployed
policy-v4 + Model B combination with its frozen production config):**

| Policy | Latent ₹ (noise-free) | vs. Fixed Retry | **Realized ₹ (stochastic)** | **vs. Fixed Retry (realized)** | Recovery rate |
|---|---|---|---|---|---|
| Fixed Retry (baseline) | 19,114.77 | +0.00 | 21,854.10 | +0.00 | 80.0% |
| Rule-Based (baseline) | 19,050.32 | −64.45 | 21,431.15 | −422.95 | 76.7% |
| Model B alone | 20,347.65 | **+1,232.88** | 21,278.18 | −575.92 | 73.3% |
| **Deployed policy (policy-v4)** | 19,032.25 | −82.51 | **19,997.23** | **−1,856.87** | 70.0% |
| Oracle (upper bound) | 23,031.64 | +3,916.87 | 24,275.30 | +2,421.20 | 86.7% |

**Read honestly, not favorably — the core finding of this evaluation:**
Model B beats Fixed Retry on the noise-free latent economic ground truth
(what it should achieve in expectation), but **on realized (stochastic) ₹
recovered — the metric closest to a real outcome — every trained-model
policy loses to the simple Fixed Retry baseline, including the policy
actually wired into `recovery/orchestrator.py` today** (−₹1,856.87, −10
points of recovery rate). The deployed policy's validation-tuned fallback
logic, meant to guard against low-confidence predictions, trades away most
of Model B's raw latent-value edge without recovering it on the realized
draw. The specification's core requirement — "the agent must clear all
three baselines, not just the easiest one" — **is not currently met on
realized money for the deployed policy.**

**Statistical significance (`evaluation/statistics.py`, wired into
`evaluate_decision_engine_v4.py`'s report as `statistical_tests`):** for the
headline comparison above — deployed policy vs Fixed Retry, on the SAME 60
paired held-out test events used everywhere else in this section —

- **McNemar's exact test** (binomial, two-sided, on the paired binary
  `realized_recovered` outcome — never applied to the continuous ₹ amounts,
  which is a statistically different question): of the 60 paired events,
  3 recovered under the deployed policy but not Fixed Retry (`b`), and 9
  recovered under Fixed Retry but not the deployed policy (`c`) — **p = 0.146**,
  not significant at p<0.05.
- **95% bootstrap CI** (percentile method, 10,000 resamples over paired
  events, seed=42) on the realized-₹ delta: **−₹1,856.87 [−₹4,878.20, +₹1,295.62]**
  — the interval spans zero, consistent with the McNemar result above.

**Read together, honestly:** neither test finds the deployed policy's
realized-₹ loss against Fixed Retry to be statistically distinguishable from
chance at this sample size (n=60). This does **not** mean the deployed
policy is "actually fine" — it means this evaluation cannot currently
distinguish a real regression from sampling noise, which is itself an
honest, disclosed limitation of a n=60 synthetic held-out slice, not a
result to lean on either way. See `tests/test_statistics.py` for the
methodology's own test coverage (toy contingency tables, edge cases,
reproducibility) and §19 item 2 below.

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
number. For the headline comparison, synthetic test set:

| Policy | Recovered GMV | Intervention cost | Razorpay fee take (gross) | Net recovery value |
|---|---|---|---|---|
| Fixed Retry | ₹21,854.10 | ₹300.00 | ₹515.76 | ₹21,038.34 |
| **Deployed policy (policy-v4)** | ₹19,997.23 | ₹300.00 | ₹471.93 | ₹19,225.30 |

`net_recovery_value = recovered_gmv − intervention_cost − razorpay_fee_take`
— a new, REALIZED, report-level summary metric, distinct from
`policy/decision_engine.py`'s pre-existing `expected_net_value`/
`decision_margin` (which subtracts only `intervention_cost`, computed at
DECISION time before any fee modeling existed — both names are kept,
documented, never merged). See `tests/test_economics.py`.

Full metric definitions, every intermediate model's numbers (including two
architectures — Day 6/7's ranking models — that scored *below random* and
are reported as honest negative results, not hidden), and the full
diagnostic history are in the Appendix (Days 4, 6–10) and in
`evaluation/reports/*.json`.

## 17. Failure handling

Every boundary below was verified in this FIX pass — either by the
existing test suite (432/432 passing) or by direct execution against the
running system:

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

## 19. Known limitations

**Confirmed in this FIX pass — read before treating any claim above as
"fully wired":**

1. **The currently-deployed policy (policy-v4) loses to the simple Fixed
   Retry baseline on realized ₹ recovered** (−₹1,856.87, −10 points of
   recovery rate, n=60 test set) — the metric closest to a real outcome.
   Model B alone beats Fixed Retry on the noise-free latent economic ground
   truth, but the validation-tuned fallback logic wrapped around it (added
   to guard against low-confidence predictions) trades that edge away
   without recovering it on the realized draw. The specification's core
   requirement — clearing all three baselines — is not currently met for
   the deployed policy on this metric. See §16.
2. **McNemar's test finds the deployed-vs-Fixed-Retry realized-₹ gap not
   statistically significant at this sample size** (exact binomial p=0.146,
   b=3, c=9, n=60 paired events; 95% bootstrap CI on the ₹ delta spans zero:
   [−₹4,878.20, +₹1,295.62]). This is a genuine result, not an unimplemented
   gap (`evaluation/statistics.py`, wired into `evaluate_decision_engine_v4.py`'s
   report as `statistical_tests`) — but it cuts both ways: it means this n=60
   synthetic evaluation cannot currently distinguish the observed loss from
   sampling noise either. See §16.
3. **Razorpay's fee take is modeled at one uniform rate** (`policy/economics.py`,
   ≈2.36% of recovered GMV, gross — the specification's disclosed "2% + 18%
   GST" domestic card rate), applied to every recovered rupee regardless of
   payment instrument. The specification itself flags UPI's exact take-rate
   as an unresolved inconsistency in Razorpay's own public materials and
   instructs against stating one clean UPI figure — this project follows
   that instruction rather than inventing an uncited second rate, which
   means the fee figures in §16/the dashboard are a documented
   simplification for non-card instruments, not a precise per-instrument
   fee model.
4. **The deployed model was not built the way the specification's Model 1
   describes.** The specification calls for "a calibrated binary classifier
   predicting P(recover within 14 days | ...)". The project's first
   attempt at exactly that (Day 4) topped out at test ROC-AUC 0.566 — a
   weak result honestly diagnosed and reported, not hidden — and after two
   further iterations (Day 6/7) also failed to beat random guessing on
   within-event ranking, the project pivoted (Day 8) to a CatBoost
   regressor trained directly on the synthetic generator's own latent
   expected-value target, not the noisy observed outcome. This is
   disclosed thoroughly in the appendix as a deliberate, diagnosed pivot —
   but it means the deployed model's strong-looking fit (R²≈0.87) partly
   reflects reconstructing the generator's own formula, and the
   specification's originally-scoped Model 1 architecture was never
   actually shipped as the production model.
5. **All evaluation numbers are synthetic.** Nothing in this repository has
   touched a real Razorpay production account, a real customer, or a real
   message-sending API. The dataset is archetype-generated with
   probabilistic (not deterministic) labels, by design.
6. **Small test set (n=60 events).** Every ₹/rate gap reported against a
   baseline in §16 is directional, synthetic-benchmark evidence, not a
   statistically confirmed result at this sample size.
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

## 20. Final feature status

| Feature (per the original specification) | Status |
|---|---|
| Webhook receiver, HMAC verification, idempotent storage | **DONE** — real Razorpay webhook mechanics, tested |
| Deterministic failure classification | **DONE** |
| Synthetic dataset (archetypes, probabilistic labels, ID-level split) | **DONE** |
| Calibrated recovery-likelihood / candidate-time model | **PARTIAL** — a value-regressor (Model B) is deployed and beats baselines on latent value, but loses to Fixed Retry on realized ₹; the spec's originally-scoped "calibrated binary classifier" was attempted (Day 4) but never exceeded weak (0.57) AUC and was superseded. See §16, §19. |
| Deterministic compliance gate | **DONE** |
| Payment action (recorded, never live) | **DONE** |
| LLM: outreach microcopy | **DONE**, wired into the live orchestrator, including a `hard_decline` payment-method-update nudge (FIX #3) |
| LLM: promise-to-pay parsing | **DONE** — parsed (LLM), validated (deterministic, `policy/promise_to_pay.py`), persisted (`PromiseToPay`), and capable of overriding the model's chosen retry timing through compliance in `recovery/orchestrator.py` (FIX #1). No live payment execution loop exists to observe a real fulfilled/broken outcome — see §19 item 9. |
| LLM: batch-level report explanation | **DONE**, used by the dashboard |
| 3 baselines (No Recovery, Fixed Retry, Rule-Based) | **PARTIAL** — implemented, but Fixed Retry simplifies Razorpay's T+3 policy to a single T+1 decision, and Rule-Based omits the spec's WhatsApp-nudge/3-day-followup behavior (no communication channel exists to implement it with). See §16. |
| 7 evaluation metrics | **PARTIAL** — recovery rate, ₹ recovered (now split into merchant GMV / Razorpay fee take / net, `policy/economics.py`), incremental ₹, cost-per-recovery, unnecessary-intervention rate, and customer-contact rate are computed; the required McNemar's-test/bootstrap-CI significance on the headline deployed-vs-Fixed-Retry comparison (`evaluation/statistics.py`) is now also implemented. See §16. |
| Audit trail | **DONE** |
| End-to-end automatic webhook→orchestration wiring | **DONE** (FIX #2) — a stored, verified webhook event continues automatically into classification + full orchestration in the same request; `scripts/reprocess_raw_events.py` remains available for manual re-processing of a failed/pre-fix event. |
| Streamlit dashboard | **DONE** — 7 pages, verified via `AppTest` and direct execution, including promise-to-pay and hard-decline-communication detail |
| Failure-mode demonstrations (Section 13 of the spec) | **DONE** — all 5 required scenarios (insufficient_fund recovery, hard decline + nudge, promise-to-pay override, LLM failure, webhook ingestion) plus a bonus opt-out scenario, demonstrated and tested |

## 21. Manual setup remaining

To go beyond the offline demo (already fully functional with no setup
beyond §7):

- Generate real Razorpay Test Mode API keys and a webhook secret, and run
  a zrok tunnel, if you want to demonstrate a genuine live webhook delivery
  (§10). Not required for tests, demos, or the dashboard.
- Set `LLM_PROVIDER=anthropic` and a real `ANTHROPIC_API_KEY` if you want
  genuinely LLM-generated (rather than deterministic mock) outreach copy,
  promise-to-pay parsing, or report narration.
- McNemar's test, the bootstrap CI, and Razorpay fee-take modeling are now
  implemented (§16, §19 items 2–3); a real per-instrument fee schedule (vs.
  this project's one uniform card-rate assumption) remains a scoped,
  understood simplification, not an unknown.

---
## Appendix: day-by-day development log

Everything above this line is the current, integrated product document. What
follows is the original, unedited day-by-day build diary — preserved in full
because it contains the actual reasoning, diagnosed failures, and honestly-
reported negative results (e.g. Day 6/7's ranking models scoring worse than
random) behind every decision summarized above. It is retained for anyone
who wants the detailed history; nothing in §1–21 above depends on reading
further. A small number of headline figures are repeated between the two
(e.g. §16's evaluation table) — where they differ, the section above is the
one confirmed current by this audit pass; a fact appearing only below is
historical context, not a live claim.

Note on Day 3: the dataset-generation day has no dedicated section below —
its full write-up (schema, archetype methodology, leakage-prevention
discipline, probabilistic label generation) lives in `data/README.md`
instead, referenced throughout the days below wherever "Day 3" appears.

---

# Adaptive Payment Recovery Agent — Day 1

Razorpay Test Mode Subscription → `insufficient_fund` failure → real Razorpay webhook →
zrok → FastAPI → HMAC verification → idempotency check → SQLite → stored structured event.

Day 1 only. No classification, ML, LLM, policy logic, or UI yet — see `app/models.py` for
where Day 2 plugs in.

## ⚠️ Stack correction from the original plan

The original stack lock said **ngrok**. Razorpay's current webhook documentation
(https://razorpay.com/docs/webhooks/validate-test/) explicitly blacklists `ngrok.io` as a
webhook URL domain and recommends **zrok** instead. This project uses **zrok**, not ngrok.

## Setup

```bash
git clone <your-repo-url> recovery-agent
cd recovery-agent

python3.12 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

cp .env.example .env
# now edit .env and fill in real values (see "Getting your Razorpay secrets" below)
```

## Run

```bash
./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Confirm it's up: `curl http://127.0.0.1:8000/health` → `{"status":"ok","env":"test"}`

## Test

```bash
./venv/bin/python -m pytest tests/ -v
```

11 tests, covering: valid signature, invalid signature, tampered body, missing signature,
empty secret, manual-vs-official-SDK signature cross-check, malformed JSON body, missing
`x-razorpay-event-id` header, duplicate event id, and successful DB insertion with full
field extraction.

## Inspect the database

```bash
./venv/bin/python -c "
import sqlite3
conn = sqlite3.connect('data/recovery_agent.db')
conn.row_factory = sqlite3.Row
for row in conn.execute('SELECT id, event_type, payment_id, subscription_id, error_reason, signature_verified FROM raw_events'):
    print(dict(row))
"
```

---

## Getting your Razorpay secrets

1. Sign in at https://dashboard.razorpay.com and switch to **Test Mode** (toggle, top left).
2. **API keys**: Account & Settings → API Keys → Generate Test Key. Copy the Key Id and Key
   Secret into `.env` as `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET`.
3. **Webhook secret**: created in the next section — it is a *different* value from the API
   key secret above. Do not reuse one for the other.

## Set up the tunnel (zrok, not ngrok)

Razorpay will only deliver webhooks to a public HTTPS URL — `localhost` is rejected outright,
and several common tunneling domains (including `ngrok.io`) are explicitly blacklisted.

```bash
# 1. Install zrok for your OS: https://docs.zrok.io/docs/getting-started
# 2. Create an account / get invited, then enable your local environment:
zrok enable <the token from your zrok account email/dashboard>

# 3. With the FastAPI server already running on port 8000, share it:
zrok share public localhost:8000
```

`zrok` prints a public HTTPS URL (e.g. `https://abcd1234.share.zrok.io`). That URL plus
`/webhook/razorpay` is what goes into the Dashboard in the next step. The share stays live
only while this command keeps running — leave the terminal open during testing.

## Configure the webhook in the Dashboard

Exact current steps (https://razorpay.com/docs/webhooks/setup-edit-payments/):

1. Log in to the Dashboard → **Accounts & Settings**.
2. Click **Webhooks** under **Website and app settings**.
3. Click **+ Add New Webhook**.
4. In the **Webhook Setup** pop-up:
   - **URL**: your zrok URL + `/webhook/razorpay`, e.g. `https://abcd1234.share.zrok.io/webhook/razorpay`
   - **Secret**: make one up (e.g. a long random string) — this is `RAZORPAY_WEBHOOK_SECRET`
     in your `.env`, not your API key secret.
   - **Alert Email**: your email, for failure notifications.
   - **Active Events**: check at least `payment.failed` and `subscription.charged` (add more
     later as Day 2+ needs them).
5. Click **Create Webhook**.
6. If prompted for an OTP while setting this up in Test Mode, Razorpay's documented test-mode
   OTP is **754081**.

## Create a Test Mode subscription and trigger `insufficient_fund`

1. Dashboard (Test Mode) → **Subscriptions** → create a **Plan** (any amount/interval), then
   create a **Subscription** against that plan.
2. Complete the subscription's first authorization charge using the checkout that opens, but
   use this specific test card instead of a random one:
   - **Card number**: `4100 2800 0009 0000` (Visa)
   - **CVV**: any random 3 digits
   - **Expiry**: any future date
3. On the mock bank success/failure screen Razorpay's test mode shows, you must **actively
   select "Failure"** — the test card number alone only maps to the `insufficient_fund`
   error reason once you choose failure on that screen.
4. This fires a `payment.failed` webhook (and, since it's tied to a subscription, the payload
   also carries a `payload.subscription.entity` block) to your configured URL.

## End-to-end test procedure

1. Terminal 1: `./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000`
2. Terminal 2: `zrok share public localhost:8000`
3. Configure the Dashboard webhook with the printed zrok URL (see above).
4. Trigger the `insufficient_fund` test-card failure on a Test Mode subscription (see above).
5. Watch Terminal 1 — you should see a `Stored event_type=payment.failed ... error_reason=insufficient_fund`
   log line within a few seconds.
6. Run the DB inspection command above and confirm a new row with `error_reason =
   insufficient_fund` and `signature_verified = 1`.
7. Trigger the same failure a second time (or use the Dashboard's webhook retry button) and
   confirm the row count does not increase — the duplicate is logged and acknowledged, not
   re-stored.

## Common errors and fixes

| Symptom | Cause | Fix |
|---|---|---|
| Dashboard rejects your webhook URL | Used `ngrok`/`localhost`/another blacklisted domain | Use the `zrok` public URL |
| `400 invalid signature` on every real webhook | Webhook secret mismatch, or you reused the API key secret | Use the exact secret entered in Dashboard → Webhooks, not `RAZORPAY_KEY_SECRET` |
| Signature check fails intermittently | Body got re-serialized/parsed before hashing (framework middleware, logging, etc.) | Verify against `await request.body()` raw bytes, before any `json.loads` |
| `RuntimeError: RAZORPAY_WEBHOOK_SECRET is not set` on startup | `.env` missing or not copied from `.env.example` | `cp .env.example .env` and fill it in |
| `ModuleNotFoundError: No module named 'pkg_resources'` | Python 3.12 venvs don't always ship `setuptools`, which `razorpay-python` needs | Already pinned in `requirements.txt`; if you hit this, `pip install setuptools` |
| Webhook never arrives | zrok share not running, or FastAPI server not running | Both must be running simultaneously in separate terminals during testing |
| `payment.failed` doesn't fire | Razorpay does not trigger `payment.failed` for failures during the very first authorization in some flows | Use a subscription that has already completed one successful cycle, then force a later charge to fail, if the first-charge webhook doesn't appear |

## Day-1 acceptance checklist

- [ ] `pytest tests/ -v` passes all 11 tests
- [ ] `/health` returns 200 with the running server
- [ ] A real Razorpay Test Mode `insufficient_fund` failure produces a webhook delivery
- [ ] The delivered webhook is stored in `raw_events` with correct `error_reason`, `payment_id`, `subscription_id`
- [ ] A duplicate delivery of the same event does not create a second row
- [ ] An intentionally-wrong signature is rejected with 400 and nothing is stored
- [ ] `.env` is not committed (`git status` shows it ignored)
- [ ] `failure_events` and `audit_log` tables exist (even though `failure_events` stays empty until Day 2)

---

# Day 2 — Deterministic Failure Classification

```
raw_events  →  classification/rules.py (deterministic)  →  failure_events  →  audit_log
```

Day 2 scope only: a rule-based classifier that maps a verified `error_reason`
string to one of four buckets, and a CLI to run it over rows already sitting
in `raw_events`. No ML/CatBoost, no Claude/LLM, no retry-time prediction, no
Streamlit, no compliance gate — those are later days.

## Supported classification rules (`rule_version = "v1"`)

Every mapping below is traceable to Razorpay's own documentation, fetched
2026-08-24 from Razorpay's official "list of possible error reasons"
spreadsheet (linked from
[razorpay.com/docs/payment-gateway/rainy-day/errors/error-reasons](https://razorpay.com/docs/payment-gateway/rainy-day/errors/error-reasons/))
and cross-checked against
[razorpay.com/docs/errors/payments/cards](https://razorpay.com/docs/errors/payments/cards/).
The exact justification for each rule is also in the code — see
`classification/rules.py`.

| `error_reason` | bucket | confidence | why |
|---|---|---|---|
| `insufficient_fund` | `retryable_soft` | 1.0 | Repo's Day-1 fixture / actual Razorpay webhook field spelling. Balance can change — same instrument may succeed later. |
| `insufficient_funds` | `retryable_soft` | 1.0 | Same condition; Razorpay's official spreadsheet spells it in the plural. |
| `bank_technical_error` | `retryable_soft` | 1.0 | Issuing bank technical/downtime issue (Razorpay docs) — transient. |
| `server_error` | `retryable_soft` | 1.0 | Technical error at Razorpay's own server (Razorpay docs) — transient. |
| `request_timed_out` | `retryable_soft` | 1.0 | Request timeout (Razorpay docs) — transient. |
| `payment_timed_out` | `retryable_soft` | 1.0 | Customer/gateway didn't complete in time (Razorpay docs) — transient. |
| `card_expired` | `hard_decline` | 1.0 | The card itself is expired (Razorpay docs) — permanent for this instrument. |
| `debit_instrument_blocked` | `hard_decline` | 1.0 | Card blocked by issuer or customer (Razorpay docs) — permanent. |
| `debit_instrument_inactive` | `hard_decline` | 1.0 | Card inactive/frozen (Razorpay docs) — permanent. |
| `payment_risk_check_failed` | `hard_decline` | 1.0 | Fraud/risk check decline at Razorpay/gateway/issuer (Razorpay docs). |
| `card_declined` | `hard_decline` | 1.0 | Explicit issuer decline after the bank's own checks (Razorpay docs). |
| `payment_cancelled` | `customer_cancelled` | 1.0 | Customer explicitly cancelled during authentication (Razorpay docs). |
| anything else — including a missing/empty `error_reason` | `unmapped` | 0.0 | Not in the verified table. |

**Unknown reasons are never guessed.** Any `error_reason` not in the table
above — even one that sounds plausible — comes back `unmapped` with
confidence `0.0`. Extending the table is a deliberate, reviewable edit to
`classification/rules.py`, never something inferred at runtime.

## Example: `insufficient_fund`

Input (relevant fields on a `raw_events` row):

```
error_code   = "BAD_REQUEST_ERROR"
error_reason = "insufficient_fund"
error_source = "customer"
error_step   = "payment_authorization"
```

Output (the `failure_events` row it produces):

```
classification_bucket     = "retryable_soft"
classification_confidence = 1.0
rule_version               = "v1"
```

Plus one `audit_log` row (`action="classified"`, `actor="rule"`) whose
`reason` column quotes the exact rule that matched.

## Classify existing raw_events (no Razorpay needed)

```bash
# Classify every raw_event that doesn't already have a failure_events row.
# Safe to re-run — already-classified rows are skipped, never duplicated.
./venv/bin/python -m scripts.classify_raw_events

# Classify (or report the existing classification of) one specific row.
./venv/bin/python -m scripts.classify_raw_events --raw-event-id 42
```

## Idempotency

Classification is keyed on `raw_event_id`. Classifying the same `raw_event`
twice — via the CLI or `classification.service.classify_raw_event` directly
— never creates a second `failure_events` row: the existing row is returned
unchanged, and a `classification_skipped_duplicate` audit_log entry is
written instead of a new classification. Deciding not to re-classify is
still a decision, per the audit_log convention `app/models.py` set in Day 1
("every decision the system makes — including deciding to do nothing —
goes here").

## Test

```bash
./venv/bin/python -m pytest tests/ -v
```

30 tests total: the original 11 Day-1 tests, unchanged, plus 19 new Day-2
tests — `tests/test_classification_rules.py` (pure classifier logic: each
bucket, unmapped, missing/malformed fields, confidence values) and
`tests/test_classification_service.py` (DB-backed: storage, idempotency,
audit log, batch classification via `classify_all_raw_events`).

## Day-2 acceptance checklist

- [ ] `pytest tests/ -v` passes all 30 tests
- [ ] `insufficient_fund` classifies as `retryable_soft` with confidence `1.0`
- [ ] A verified hard-decline reason (e.g. `card_expired`) classifies as `hard_decline`
- [ ] A verified cancellation reason (`payment_cancelled`) classifies as `customer_cancelled`
- [ ] An unverified/unknown `error_reason` classifies as `unmapped`, never guessed
- [ ] Classifying the same `raw_event` twice produces exactly one `failure_events` row
- [ ] Every classification decision — including a skipped duplicate — has a corresponding `audit_log` row
- [ ] `./venv/bin/python -m scripts.classify_raw_events` classifies existing `raw_events` without a Razorpay webhook

---

# Day 4 — Recovery-Likelihood Model

Day-3's synthetic dataset (see `data/README.md`) → preprocessing → a Logistic
Regression baseline and a CatBoost main model → probability calibration →
evaluation on the untouched test split. **No retry-time selection, recovery
policy, compliance gate, LLM, or UI is added today** — this is purely the
"how likely is this failure to recover" model those later stages will consume.

## Target

```
P(recovered_within_14d | information available at the moment of the failed payment)
```

A binary classifier producing a calibrated probability, not a hard decision.

## Feature inclusion / exclusion

Every column in `data/processed/*.csv` was reviewed explicitly — see
`model/preprocessing.py::EXCLUDED_COLUMNS` (also written to
`model/artifacts/feature_list.json` on every training run) for the
single source of truth both the code and `tests/test_model_pipeline.py`
check against.

**Used (17 features):**

| type | features |
|---|---|
| numeric | `day_of_month`, `days_to_nearest_payday_window`, `amount`, `prior_if_failure_count`, `prior_if_self_resolved_rate` (+ its missing-flag), `tenure_days`, `issuing_bank_downtime_flag`, `is_month_end_settlement_rush` |
| categorical | `plan_tier`, `primary_instrument`, `city_tier`, `bank_network_conditions`, `network_latency_bucket` |
| distractor (categorical, intentionally kept) | `app_version`, `device_build`, `ui_theme` |

**Excluded, with reason:**

| column | reason |
|---|---|
| `event_id`, `subscription_id` | identifiers, not predictive |
| `failure_timestamp`, `signup_date` | raw high-cardinality timestamps; their useful signal is already captured by `day_of_month` / `days_to_nearest_payday_window` / `tenure_days` — feeding raw dates into ~200 rows risks memorizing specific dates |
| `monthly_amount` | exact duplicate of `amount` in this dataset (failure amount == subscription's billed amount by construction) |
| `error_reason` | constant (`insufficient_fund` only, by Day-3 scope) — zero variance |
| `recovered_within_14d` | **the label**, used as `y` only |
| `recovered_at`, `recovered_via`, `final_amount_recovered` | post-outcome information — only knowable after the failure resolves; using them would leak the label |
| `archetype` | hidden, generation-only — must never be a model feature (also not present in `data/processed/*.csv` at all; Day 3 already drops it) |
| `split` | dataset-partition bookkeeping, not a real-world feature (also not present in `data/processed/*.csv`) |

`tenure_days` deserves a specific callout: `data/processed/*.csv` carries
`failure_events.tenure_days` ("as of this failure"), not
`subscriptions.tenure_days` ("as of the Day-3 extraction date") — Day 3
already resolved this leakage trap during dataset construction (see
`data/README.md`), Day 4 just consumes the leak-safe column.

## Preprocessing (`model/preprocessing.py`)

- **Missing `prior_if_self_resolved_rate`** (first-time failures have no
  prior history): `PriorSelfResolvedImputer` is fit on the training split
  only — it learns one scalar (train's mean of the non-missing values) —
  then adds an explicit `prior_if_self_resolved_rate_missing` flag and
  fills the gap with that train-only scalar. Validation and test are only
  ever `.transform()`-ed with it, never refit.
- **Categoricals**: one-hot encoded (`handle_unknown="ignore"`) for
  Logistic Regression; passed natively as strings to CatBoost via
  `cat_features=`.
- **Numeric scaling**: `StandardScaler`, fit on train only, for Logistic
  Regression (CatBoost is tree-based and doesn't need it).
- **Fit/transform discipline**: every `ColumnTransformer` / imputer is
  `.fit()` exactly once, on `train_df`, inside `model/train.py::fit_pipeline`
  — a function whose signature is literally `(train_df, val_df)`, with no
  parameter through which test data could reach it (verified in
  `tests/test_model_pipeline.py::test_fit_pipeline_signature_has_no_test_parameter`
  and `test_training_result_is_unaffected_by_unrelated_test_split_contents`).

## Baseline: Logistic Regression

`sklearn.linear_model.LogisticRegression`, `solver="lbfgs"`. Regularization
strength `C` is selected by validation ROC-AUC over `[0.001, 0.01, 0.1, 1.0,
10.0]` (legitimate validation-based model selection, not test-set tuning) —
this matters here: one-hot expanding 8 categorical columns (including the 3
distractors) produces **39 features from 194 training rows**, and sklearn's
default `C=1.0` measurably overfits at that ratio (train AUC 0.73 vs
validation AUC 0.42 at C=1.0, in the seed-42 run).

## CatBoost configuration

`CatBoostClassifier(cat_features=<the 5 real categoricals + 3 distractors>,
iterations=500, depth=4, learning_rate=0.05, loss_function="Logloss",
eval_metric="AUC", early_stopping_rounds=50, use_best_model=True,
random_seed=42)`. Chosen over Logistic Regression as the main model because
(a) the dataset is mostly categorical and CatBoost handles that natively,
without the one-hot dimensionality blow-up above, and (b) it can capture
the interaction effects Day 3's generator deliberately built in (e.g. the
payday-proximity effect is only visible *within* the `cash_strapped_cyclical`
archetype, not marginally — see `data/README.md`), which a linear model
structurally cannot. No deep learning was used, per Day-4 scope.

## Calibration (`model/calibrate.py`)

Both **sigmoid (Platt)** and **isotonic** calibrators are fit — via
`sklearn.frozen.FrozenEstimator` wrapping the already-train-fit CatBoost
model, so `CalibratedClassifierCV` cannot silently refit the base model —
using the **validation split only** (59 rows). The choice is not automatic:
`isotonic_is_defensible()` encodes the rule (isotonic fits an unconstrained
step function and needs enough points per step; this project's floor is
200 validation rows) and logs the result. At 59 validation rows, isotonic
is **not** considered defensible here — and empirically, fitting it on so
few points produced visibly degenerate output during development (repeated
identical probabilities, exact 0.0/1.0 extremes). **Sigmoid is the
recommended method.** Both calibrators are still saved and both are
evaluated on the test set below, for transparency.

## Evaluation protocol

- **Train** (194 rows): fits Logistic Regression and CatBoost.
- **Validation** (59 rows): CatBoost early stopping, Logistic Regression's
  `C` selection, both calibrators' fitting. Never used to fit the final
  reported test metrics.
- **Test** (60 rows): read by exactly one script,
  `evaluation/evaluate_models.py`, which loads already-trained artifacts
  and computes every metric below. No file in `model/` reads
  `data/processed/test.csv`.
- Two trivial baselines are evaluated alongside the real models:
  **majority-class** (always predicts the training set's majority class)
  and **train recovery-rate** (always predicts train's constant positive
  rate — the correct "no-skill" probabilistic baseline for Brier/LogLoss).
- Because train/validation/test are only 194/59/60 rows, ROC-AUC and PR-AUC
  on the test set are reported with a **1,000-resample bootstrap 95% CI**
  (`evaluation/evaluate_models.py::bootstrap_ci`), seeded for reproducibility.

## Results (seed 42, exact numbers reproduced by `model/train.py` + `evaluation/evaluate_models.py`)

**⚠️ Read "Limitations" below before interpreting these — they are reported
honestly, not tuned to look good, and they are weak.**

Class balance drifted across the fixed Day-3 split (unstratified,
subscription-level, only 200 subscriptions): train 62.4% recovered,
validation 74.6%, test **83.3%**.

| model | ROC-AUC (test) | PR-AUC (test) | Log Loss | Brier | Accuracy |
|---|---|---|---|---|---|
| trivial: majority-class | 0.500 | 0.833 | 2.303 | 0.167 | 0.833 |
| trivial: train recovery-rate | 0.500 | 0.833 | 0.556 | 0.183 | 0.833 |
| Logistic Regression | 0.402 | 0.799 | 0.567 | 0.188 | 0.833 |
| CatBoost (uncalibrated) | 0.566 | 0.873 | 0.661 | 0.234 | 0.833 |
| CatBoost (sigmoid-calibrated) | 0.566 | 0.873 | **0.464** | **0.143** | 0.833 |
| CatBoost (isotonic-calibrated) | 0.503 | 0.839 | 0.485 | 0.152 | 0.833 |

All models land on the same 0.833 accuracy because, at the default 0.5
threshold, every model's ranking is too weak on this 60-row test set to
flip any prediction away from the majority class — a clear symptom of the
low AUCs, not a meaningful result on its own (which is exactly why this
project's primary metrics are ROC-AUC / PR-AUC / Brier, not accuracy).
Bootstrap 95% CIs on test ROC-AUC are wide, as expected at n=60 (CatBoost
uncalibrated: mean 0.565, 95% CI [0.364, 0.781]; Logistic Regression: mean
0.397, 95% CI [0.204, 0.613] — 1,000 resamples, seed 42, exact figures in
`model/reports/metrics.json`) and straddle 0.5.

**Confusion matrices, the calibration reliability curve, and full
per-model metrics** are saved to `model/reports/` (`metrics.json`,
`metrics.csv`, `calibration_curve.png`,
`confusion_matrix_logistic_regression.png`,
`confusion_matrix_catboost_uncalibrated.png`).

## Feature importance (CatBoost)

| rank | feature | importance | distractor? |
|---|---|---|---|
| 1 | `prior_if_self_resolved_rate` | 27.99 | no |
| 2 | `ui_theme` | 20.85 | **yes** |
| 3 | `day_of_month` | 12.05 | no |
| 4 | `tenure_days` | 9.18 | no |
| 5 | `amount` | 8.62 | no |
| 6 | `network_latency_bucket` | 7.10 | no |
| 7 | `device_build` | 6.23 | **yes** |
| 8 | `primary_instrument` | 4.70 | no |
| 9 | `days_to_nearest_payday_window` | 3.29 | no |
| 14 | `app_version` | 0.00 | **yes** |

`prior_if_self_resolved_rate` ranking #1 matches the intended design (Day
3's historical-behavior proxy). But **`ui_theme` — a feature deliberately
built to carry zero signal — ranks #2**, ahead of `days_to_nearest_payday_window`
and every other domain-sensible feature except one. `device_build` is
mid-pack (#7); `app_version` correctly lands near the bottom (0
importance). This is not evidence that `ui_theme` matters — CatBoost
early-stopped after **3 boosting iterations** on only 194 training rows,
so importance here reflects a handful of greedy early splits on a small,
noisy sample, not a stable ranking. Distractors dominating like this on
such a small dataset is itself a useful, honest finding, reported per the
Day-4 brief rather than hidden. **No causal claim is made about any feature
here — high rank in a 3-iteration tree ensemble over 194 rows is weak
evidence at best.**

## Honest assessment: does CatBoost's complexity earn its keep here?

**Not clearly, on this run.** CatBoost (test ROC-AUC 0.566) ranks better
than Logistic Regression (0.402, *below* chance) and better than both
trivial baselines (0.500) — but 0.566 is itself a weak result, within a
wide bootstrap CI that straddles 0.5, on a 60-row test set with a
9-percentage-point train→test recovery-rate drift baked into the fixed
Day-3 split. Diagnostics run during development (5-fold CV on pooled
train+validation, i.e. not the fixed split) put mean AUC around 0.54 with
per-fold swings from 0.46 to 0.65 — consistent with a genuinely weak,
noise-dominated small-sample regime rather than a code defect (see "How
this was diagnosed" below). Neither model reaches the 0.75–0.85 AUC Day 3
described as a *design intent* for a well-fit model — that target was
never claimed as achieved, and it was not achieved here.

What CatBoost *does* earn on structural grounds, independent of this run's
AUC: native categorical handling avoids Logistic Regression's 39-column
one-hot blow-up over 194 rows, and it's the only one of the two models
capable of learning the interaction effects (e.g. archetype × payday
proximity) Day 3's generator actually encodes. **The clearest genuine win
this run produced is calibration, not the base model**: sigmoid-calibrating
CatBoost cut Brier score from 0.234 → 0.143 and Log Loss from 0.661 → 0.464
— better than *both* trivial baselines on both metrics — without changing
ranking at all (calibration is monotonic; ROC-AUC is unchanged by
construction). If this model were used downstream today, the honest
recommendation is: trust its calibrated probabilities more than its
ranking, and do not treat 0.566 AUC as a solved discrimination problem.

## How this was diagnosed (not "fixed")

Per the Day-4 brief, poor results were investigated, not tuned away or
hidden. Confirmed during development, all reproducible from
`data/processed/*.csv`:

1. **Index alignment is correct** — `X.index == y.index` holds; no
   row-shuffling bug.
2. **Every individual numeric feature has weak marginal correlation with
   the label** even pooled across all 313 events (`|r| < 0.12` for all of
   them) — consistent with Day 3's design (signal lives in interactions
   and noise dominates single features), not a symptom of broken feature
   construction.
3. **The fixed Day-3 split has real class-balance drift** (62%/75%/83%
   train/val/test) — an expected consequence of an unstratified,
   subscription-level 60/20/20 split over only 200 subscriptions, not
   something Day 4 is permitted to alter (`Use the Day-3 train/validation/test
   splits exactly as generated`).
4. **5-fold stratified CV on pooled train+validation** (bypassing this one
   split's specific noise) gives mean ROC-AUC ≈ 0.54, fold range
   0.46–0.65 — small, high-variance, but centered near what the fixed
   split shows, not wildly different. This corroborates "genuinely weak
   signal at this sample size" over "this one split is a fluke."
5. **Logistic Regression's default regularization overfits**
   (train AUC 0.73 → val AUC 0.42 at C=1.0); validation-based C selection
   is included in `model/train.py` as the one legitimate tuning step taken
   — it improved val AUC to 0.45, still weak, and no further tuning was
   applied in pursuit of a better number.

## Limitations

- **This model is not production-ready and makes no claim to be.** Its
  test-set discrimination (ROC-AUC ≈ 0.57 for CatBoost, worse for Logistic
  Regression) is weak and imprecisely estimated (wide bootstrap CI, n=60).
- **Trained and evaluated entirely on synthetic data** — see the
  disclaimer in `data/README.md`. Nothing here is evidence about real
  Razorpay customer recovery behavior.
- **Small, drifted split.** 194/59/60 rows with an unstratified,
  subscription-level split that happened to produce a 21-point
  train→test recovery-rate gap. A larger or re-stratified regeneration of
  Day 3's dataset would very likely change every number in this section.
- **CatBoost's early stopping at iteration 3** means the "main model" is
  effectively a handful of shallow trees — appropriate given how little
  training data exists, but it limits how much of the intended
  archetype-interaction signal it could realistically have captured.
- **Isotonic calibration is reported, not recommended**, precisely because
  59 validation rows is too few to fit one defensibly — see "Calibration"
  above.
- **No hyperparameter search beyond the one documented, validation-based
  Logistic Regression `C` grid.** CatBoost's depth/learning-rate/iterations
  were set to reasonable, conservative defaults for a dataset this size and
  were not tuned against validation or test performance, per the brief's
  instruction not to "silently tune until it looks good."
- **Retry-time selection, a recovery policy, and everything downstream of
  "how likely is this to recover" are explicitly out of scope for Day 4**
  and are not implemented.

## Reproduce

```bash
./venv/bin/python model/train.py            # fits on train, selects on validation, saves model/artifacts/
./venv/bin/python evaluation/evaluate_models.py   # loads artifacts, evaluates once on test, saves model/reports/
./venv/bin/python -m pytest tests/ -v       # Day 1 + 2 + 3 + 4, 66 tests
```

Every random draw goes through a seed fixed at 42 (`model/preprocessing.py::SEED`,
reused by CatBoost's `random_seed`, Logistic Regression's `random_state`,
and the bootstrap CI's `numpy.random.default_rng`) — rerunning both scripts
reproduces the exact numbers in this section.

## Day-4 acceptance checklist

- [x] Logistic Regression baseline trained
- [x] CatBoost trained
- [x] Proper train/validation/test separation (`fit_pipeline(train_df, val_df)` has no test parameter; test read only by `evaluation/evaluate_models.py`)
- [x] No leakage (archetype/split/post-outcome fields excluded and tested; historical feature imputed train-only; `tenure_days` leakage trap avoided)
- [x] Probability calibration evaluated (sigmoid + isotonic, both reported, sigmoid recommended with reasoning)
- [x] Test set untouched until final evaluation
- [x] Feature importance inspected
- [x] Distractor features checked — and found **not** clean (`ui_theme` ranks #2), reported honestly
- [x] Metrics saved (`model/reports/metrics.json`, `.csv`, plots)
- [x] Full test suite passes (66/66: 11 Day-1 + 19 Day-2 + 22 Day-3 + 14 Day-4)
- [x] No retry policy / LLM / compliance / UI added yet

---

# Day 5 — Recovery-Policy Layer

Failed payment → candidate retry windows (Day 3's five types) → per-candidate
scoring → deterministic policy → selected action (or `NO_ACTION`) → audit
log. **No LLM, no WhatsApp/voice, no Streamlit, no real Razorpay retries, no
promise-to-pay parsing** — this is the deterministic decision layer those
later stages will sit behind.

## The decision problem, and what the data can and can't support

The brief asks for
`P(recovery within 14 days | payment/customer context, candidate retry time)`.
Day 3's dataset records exactly **one** observed outcome
(`recovered_within_14d`) per failure event — not what would have happened
under each of the 5 candidate retry times. There is no genuine per-candidate
counterfactual label to train or evaluate against. Fabricating one (e.g.
"the observed outcome belongs to whichever candidate its timestamp is
closest to") would misrepresent correlation as causal evidence for an action
we never actually simulated.

So Day 5 does **not** retrain or repurpose the Day-4 model to condition on
candidate time (`Option A` in the brief) — Day-4's calibrated CatBoost model
(`model/preprocessing.py::FEATURE_COLUMNS`) has never seen a candidate
retry time as an input, and there's no defensible way to make it "aware" of
one without a label that doesn't exist. Instead this implements `Option B`:
a transparent, documented **policy heuristic** layered on top of the
unchanged Day-4 probability — see `policy/scoring.py` module docstring for
the full reasoning, and "Counterfactual-evaluation limitation" below.

## Candidate-time feature engineering (`policy/retry_candidates.py`)

Candidates reuse Day 3's exact payday/month-end calendar logic
(`data/generate_synthetic_dataset.py::next_payday_window_after` /
`next_month_end_after` / `days_to_nearest_payday_window`), with **fixed**
(not jittered) offsets so a policy decision is reproducible — the same
`failure_timestamp` always yields the same 5 candidates.

| Feature | Type | Meaning |
|---|---|---|
| `hours_from_failure` | candidate-action | hours between failure and this candidate |
| `candidate_day_of_month` | candidate-action | calendar day the retry would happen |
| `candidate_day_of_week` | candidate-action | weekday name |
| `candidate_is_payday_aligned` | candidate-action | within 1 day of a payday window |
| `candidate_is_month_end_aligned` | candidate-action | lands on/adjacent to month end |
| `candidate_days_to_payday` | candidate-action | distance to nearest payday window |
| `candidate_type` | candidate-action | one of the 5 fixed types |

These combine with the **failure-time/customer features already used by the
Day-4 model** (`model/preprocessing.py::FEATURE_COLUMNS` — `day_of_month`,
`days_to_nearest_payday_window`, `amount`, `prior_if_failure_count`,
`prior_if_self_resolved_rate`, `tenure_days`, `plan_tier`,
`primary_instrument`, `city_tier`, `bank_network_conditions`,
`network_latency_bucket`, plus the 3 deliberate distractors, plus
`issuing_bank_downtime_flag` / `is_month_end_settlement_rush`). Excluded,
same as Day 4 and for the same reasons: `recovered_within_14d`,
`recovered_at`, `recovered_via`, `final_amount_recovered`, `archetype`, and
anything else not knowable at the moment of failure.

## Candidate scoring (`policy/scoring.py`)

```
base_probability          = Day-4 calibrated CatBoost model's prediction
                             (failure-time/customer features ONLY — no
                             candidate-time information reaches the model)
heuristic_adjustment       = ±  (a documented POLICY RULE, not a learned effect)
                               - immediate retry: −0.05 (unlikely to self-resolve within the hour)
                               + payday proximity: up to +0.08, linearly decaying to 0 beyond 7 days out
predicted_recovery_probability = clip(base_probability + heuristic_adjustment, 0, 1)

expected_recovery_value       = predicted_recovery_probability × amount
expected_incremental_value    = expected_recovery_value − intervention_cost   (cost defaults to 0.0)
```

`predicted_recovery_probability` is an **estimate**, and
`expected_recovery_value` is an **expected value**, not a claim about actual
money recovered.

## Three baselines (`policy/baselines.py`)

1. **No Recovery** — never takes any action.
2. **Fixed Retry** — always `plus_1_day_morning` (subject to the same
   classification/validity gates as everything else).
3. **Rule-Based Retry** — `days_to_payday ≤ 2` → `payday_window`, otherwise
   → `plus_1_day_morning`; falls back to the other candidate if its first
   choice is invalid.

## AI-assisted policy (`policy/recovery_policy.py`)

For every event: generate candidates → score each → apply hard guardrails →
select the highest-`expected_incremental_value` allowed candidate, or
`NO_ACTION` if none are allowed. `decide()` is a **pure function** —
identical inputs always produce an identical `DecisionResult`
(`policy_version = "policy-v1"`). `decide_for_failure_event()` is the
DB-aware wrapper: it computes guardrail state (prior attempt count,
whether this `event_id` was already decided) from `policy_decisions`, then
persists a `policy_decisions` row plus an `audit_log` row for every call —
including `NO_ACTION`, blocked, and duplicate ones.

**Decision output schema:** `event_id`, `subscription_id`,
`selected_candidate_type`, `selected_candidate_datetime`,
`predicted_recovery_probability`, `expected_recovery_value`,
`expected_incremental_value`, `baseline_action` (what the rule-based
baseline would have picked, for comparison), `policy_version`,
`decision_reason`.

## Hard guardrails (`policy/guardrails.py`)

| Guardrail | Where enforced |
|---|---|
| Classification must be `retryable_soft` | `is_classification_allowed()` — `hard_decline` / `customer_cancelled` / `unmapped` → `NO_ACTION` |
| No action after a cancellation state | same check — `customer_cancelled` is itself a cancellation state |
| Candidate must be after failure | `validate_candidate()` |
| Candidate must be within a 14-day recovery horizon | `validate_candidate()` (`MAX_CANDIDATE_HORIZON_DAYS`) |
| Maximum retry attempts (3) | `decide()` / `decide_for_failure_event()`, counting prior non-`NO_ACTION` `policy_decisions` rows for the subscription |
| Duplicate-decision prevention | `decide_for_failure_event()` — a second call for the same `event_id` returns the existing row and logs `policy_decision_skipped_duplicate` instead of creating a new one |

All guardrails are deterministic code — no LLM.

## Audit log

Every `decide_for_failure_event()` call writes exactly one `audit_log` row
(`actor="policy"`), including `NO_ACTION`, blocked, and duplicate decisions
— nothing is silently discarded. The row's `reason` embeds the full
per-candidate `candidate_scores` (type, datetime, base probability,
heuristic adjustment, predicted probability, expected value, valid/invalid)
as JSON, so a decision can be fully reconstructed from the log alone.

## Counterfactual-evaluation limitation

**We can score candidate actions using the available model/policy
estimates, but we cannot claim causal recovery lift from candidate timing
until counterfactual outcomes are available.** `evaluation/evaluate_policy.py`
therefore keeps two things strictly separate:

- **(A) Predicted expected value** — model/heuristic score of each
  approach's chosen action. Deterministic, reproducible, not causal.
- **(B) Observed historical reference** — the test set's actual
  `recovered_within_14d` rate, reported once as an aggregate, **never**
  broken out by `candidate_type` or by which approach "would have" picked
  what. Reporting a fake "actual recovery rate by candidate" would imply we
  ran each policy against reality and observed its outcome — we did not.

Latest run (`data/processed/test.csv`, 60 rows, Day-4 sigmoid-calibrated model):

| Approach | Actions taken | Σ predicted expected recovery value | Mean predicted probability (acted events) | Candidate distribution |
|---|---|---|---|---|
| No Recovery | 0/60 | 0.00 | n/a | — |
| Fixed Retry | 60/60 | 21,397.86 | 0.7882 | `plus_1_day_morning`: 60 |
| Rule-Based | 60/60 | 22,141.28 | 0.8099 | `payday_window`: 35, `plus_1_day_morning`: 25 |
| AI-Assisted Policy | 60/60 | 22,141.28 | 0.8099 | `plus_1_day_morning`: 39, `payday_window`: 21 |

Observed historical reference (aggregate only): recovery rate 0.8333.

The AI policy's total predicted value ties Rule-Based here (never falls
below it, verified per-row) because with a single failure-time
`base_probability` shared across all 5 candidates for a given event, the
heuristic's only lever is payday proximity — which the rule-based baseline
already targets directly for its two candidate options. The AI policy
additionally considers `immediate`, `plus_3_days`, and `month_end_window`
per event and only ever picks one of those when it scores higher, which on
this run it never does. This is expected given a heuristic this simple, not
a ceiling on what a genuine candidate-aware model (Day 6+, if counterfactual
labels become available) could find.

## Policy sanity checks

| # | Scenario | Result |
|---|---|---|
| A | `insufficient_fund` (→ `retryable_soft`) near payday | considers and often selects `payday_window` |
| B | `hard_decline` | `NO_ACTION` |
| C | `customer_cancelled` | `NO_ACTION` |
| D | `unmapped` | `NO_ACTION` |
| E | `retryable_soft`, candidate before failure or beyond the 14-day horizon | that candidate is excluded by `validate_candidate()` |
| F | Same `event_id` decided twice | second call returns the identical existing row (`created=False`), logs `policy_decision_skipped_duplicate` |

## Reproduce

```bash
./venv/bin/python evaluation/evaluate_policy.py   # scores test.csv under all 4 approaches, saves evaluation/reports/
./venv/bin/python -m pytest tests/ -v             # Day 1 + 2 + 3 + 4 + 5
```

## Day-5 acceptance checklist

- [x] Candidate retry windows scored (all 5 types, deterministic)
- [x] Expected recovery value calculated (`probability × amount`, cost-adjustable)
- [x] No-action guardrails implemented (classification, horizon, ordering, max attempts, duplicates)
- [x] Fixed + rule-based baselines implemented
- [x] AI-assisted policy implemented (pure `decide()` + DB-aware wrapper)
- [x] Decisions are deterministic and auditable
- [x] Idempotency enforced (`decide_for_failure_event`)
- [x] No LLM / voice / WhatsApp / UI added
- [x] No fake counterfactual claims — predicted vs. observed kept strictly separate
- [x] Full test suite passes (105/105: 66 Day 1–4 + 39 Day 5)

---

# Day 6 — Counterfactual Outcomes + Candidate-Aware Policy

**"SYNTHETIC COUNTERFACTUAL EVALUATION"** — every number in this section
comes from `data/raw/counterfactual_outcomes.csv`, a hand-designed
simulation. **This does not measure real Razorpay recovery performance.**

## Why this was needed

Day 5 flagged an honest limitation: the dataset had one observed outcome
per failure event, not one per candidate retry time, so nothing could
honestly claim that *timing itself* causes higher recovery. Day 6 fixes
that at the data layer, not by reinterpreting the existing single outcome.

## How synthetic counterfactuals are generated

`data/generate_counterfactual_dataset.py` reuses Day 3's exact
subscriptions/failure_events/retry_candidates (same seed → byte-identical),
then layers a **second, independent** random stream
(`seed + 5000`, never touching Day 3's own stream) to draw a genuine
outcome for **every one of the 5 candidates per event** — 1,565 rows for
313 events in the committed dataset.

Each candidate's latent probability is:

```
logit = archetype_base_logit                                    # hidden, same as Day 3
      + payday_sensitivity[archetype]  × proximity_to_payday(candidate)     # NEW: candidate-specific
      + month_end_sensitivity[archetype] × is_month_end_aligned(candidate)  # NEW
      + immediate_penalty[archetype]     if candidate_type == "immediate"   # NEW
      + elapsed_time_bonus[archetype]    × elapsed_fraction(candidate)      # NEW
      + prior_if_self_resolved_rate + amount_penalty + exogenous penalties + tenure_bonus   # shared, same as Day 3
      + Normal(0, 0.9)
p = clip(sigmoid(logit), 0.02, 0.98)
```

Sensitivities are sized per archetype exactly as the brief specifies:
`cash_strapped_cyclical` reacts strongly to payday proximity (1.8) and is
hurt badly by retrying immediately (−0.9); `reliable` and `quiet_canceller`
barely react to timing at all (0.1–0.2); `chronic_struggler` sits in
between. **Physical constraint, not a modeling choice:** a candidate whose
own scheduled time already falls at/after `failure_timestamp + 14 days`
cannot be recorded as `recovered_within_14d`, no matter how high its latent
probability — there's no time left. This has real bite:
`payday_window`/`month_end_window` land beyond the 14-day horizon 38%/58%
of the time in the committed dataset (a genuine consequence of monthly
payday cycles vs. a 14-day recovery SLA, already anticipated by Day 5's own
`MAX_CANDIDATE_HORIZON_DAYS` guardrail).

## What's hidden from the model

`archetype` drives generation only — it is never joined into the
candidate-level table at all (structurally absent, not just excluded by
convention). Also excluded, same discipline as Day 4:
`recovery_probability_latent` (label-adjacent), `recovered_at`,
`recovered_via`, `amount_recovered` (post-outcome), `split`. See
`model/candidate_preprocessing.py::EXCLUDED_COLUMNS` for the full,
documented list.

## Candidate-time features

`candidate_type`, `hours_from_failure`, `candidate_day_of_month`,
`candidate_day_of_week`, `candidate_days_to_payday`,
`candidate_is_payday_aligned`, `candidate_is_month_end_aligned` — combined
with Day 4's unchanged failure-time/customer features. No new
`data/processed/*.csv` files: the candidate-level table (5 rows/event) is
built at load time by joining `counterfactual_outcomes.csv` +
`retry_candidates.csv` + `failure_events.csv` + `subscriptions.csv`
(`model/candidate_preprocessing.py::build_candidate_level_dataset`).

## Train / validation / test methodology

Same subscription-level split as Day 3 (60/20/20), inherited via
`subscription_id` — a subscription and **all 5×N of its candidate rows**
land in exactly one split, never crossing (`test_no_split_leakage_across_subscriptions`).
970 / 295 / 300 candidate rows in the committed dataset (194/59/60 events ×5).

## Candidate-aware model (`model/train_candidate_model.py`)

Structurally identical pipeline to Day 4 (imputer → LogReg → CatBoost →
sigmoid + isotonic calibration), trained on the candidate-level table
instead. Genuinely answers `P(recovered_within_14d | failure context,
candidate retry action)` — Day 4's model never saw candidate timing at all.

| Metric (sigmoid-calibrated CatBoost, test, n=300) | Value |
|---|---|
| ROC-AUC | 0.7999 (bootstrap 95% CI: 0.746–0.851) |
| PR-AUC | 0.8003 |
| Log loss | 0.4678 |
| Brier score | 0.1582 |
| Accuracy | 0.7700 |
| Precision / Recall / F1 | 0.7124 / 0.9881 / 0.8279 |

Isotonic calibration is now defensible (295 validation rows ≥ the 200-row
floor `model/calibrate.py` uses) — a change from Day 4, purely because 5×
the candidate-level rows exist per split.

## Oracle definition

`oracle_action` = the candidate with the highest **latent** counterfactual
probability, selected via the exact same guardrails as every other policy
(`policy/recovery_policy.py::decide_candidate_aware`, fed latent
probabilities instead of model predictions — the oracle is that function
composed with ground truth, not a separate implementation). It is an
**upper bound only, not a deployable policy** — nothing in a real pipeline
would ever know the latent probability.

## Ranking quality and a surprising finding

| Metric (n=60 test events) | Value |
|---|---|
| Top-1 candidate selection accuracy (AI vs. Oracle) | 16.7% |
| Top-2 candidate coverage | 53.3% |
| Avg. regret (expected value, ₹) | 83.84 |
| Policy/Oracle gap (probability scale) | 0.138 |
| **Mean within-event rank correlation** | **−0.149** |

**Surprising finding, reported honestly rather than tuned away:** the
candidate-aware model's pooled ROC-AUC (0.80) is good, but its top-1
accuracy (16.7%) is *worse than random* (20% for 5 candidates), and the
mean **within-event** Spearman correlation between predicted and latent
probability is slightly **negative** (−0.149). These are not
contradictory — pooled AUC measures discrimination across *all* test rows,
which mixes strong between-event variance (archetype, amount, prior
history — logit swings of several units) with much weaker within-event
variance (the candidate-timing effect, deliberately sized at 0.1–1.8 logit
units per the brief). A model can achieve a good pooled AUC almost entirely
by correctly separating "easy" events from "hard" ones, while still failing
to rank *which candidate* is best *within* an event — exactly what
happened here. The model (only 18 CatBoost iterations before early
stopping, 970 training rows) appears to have leaned heavily on
`hours_from_failure`'s strong, easy-to-learn threshold effect (via the
14-day horizon truncation) and under-learned the subtler payday-alignment
interaction, converging on "prefer `plus_3_days`" as a safe default
(60% of AI selections) rather than genuinely discriminating between
candidates per event.

**This was not fixed by tuning.** Per the brief's explicit instruction not
to tune blindly toward a nicer number, the counterfactual generator's
coefficients and the model's hyperparameters were left as originally
designed. A legitimate fix would need a ranking-aware objective (pairwise/
listwell loss, or per-event-normalized labels) rather than pooled log loss
— a Day-7+ candidate, not a same-day patch.

## Oracle vs. AI-Assisted Policy vs. baselines (money metrics, synthetic)

n=60 test events, ₹ = simulated amount, from realized counterfactual
outcomes for whichever candidate each policy selected:

| Policy | ₹ recovered | Recovery rate | Lift vs. Fixed Retry | Regret vs. Oracle (₹) | Unnecessary interventions |
|---|---|---|---|---|---|
| No Recovery | 0.00 | 0.0% | −80.0pp | 24,275.30 | 0 |
| Fixed Retry | 21,854.10 | 80.0% | +0.0pp | 2,421.20 | 12 |
| Rule-Based | 21,431.15 | 76.7% | −3.3pp | 2,844.15 | 14 |
| **AI-Assisted Policy** | **18,616.87** | **68.3%** | **−11.7pp** | **5,658.43** | **19** |
| Oracle Policy (upper bound) | 24,275.30 | 86.7% | +6.7pp | 0.00 | 8 |

**The AI-Assisted Policy underperforms the naive baselines in this run** —
a direct consequence of the ranking failure above, not a separate bug. The
Oracle confirms real headroom exists (86.7% vs. Fixed Retry's 80.0%
achievable rate) — the candidate-timing signal is genuinely there in the
data (§ sanity checks below) — but the current model doesn't capture it
well enough to act on. Shipping this model as the retry-timing policy would
be worse than simply always retrying `plus_1_day_morning`; that's the
honest conclusion this evaluation is built to be able to reach.

## Sanity checks — timing effects are real

From `data/generate_counterfactual_dataset.py`'s own summary (313 events,
1,565 counterfactual rows):

- **No single candidate dominates.** Oracle (unconstrained by guardrails)
  picks `month_end_window` most often, but only 38.3% of the time — every
  one of the 5 types wins somewhere.
- **`cash_strapped_cyclical` shows a materially larger latent-probability
  spread across its 5 candidates than `quiet_canceller`**
  (`test_payday_sensitive_archetype_shows_larger_candidate_spread_than_insensitive_archetype`)
  — the required archetype × timing interaction is present and testable,
  independent of noise draws.
- **Realized recovery rate by candidate type** (accounts for horizon
  truncation): `plus_3_days` 70.3%, `plus_1_day_morning` 68.4%, `immediate`
  60.1%, `payday_window` 46.7%, `month_end_window` 34.8% — payday/month-end
  have the *highest latent probability* but the *lowest realized rate*,
  because they're the two most likely to arrive too late to count. A real,
  reportable tension, not a bug.

## Policy integration

`policy/recovery_policy.py::decide_candidate_aware()` /
`decide_for_failure_event_candidate_aware()` are **new, additive**
functions — Day 5's `decide()` / `decide_for_failure_event()` are
byte-for-byte unchanged and all 39 of their tests still pass. Same
guardrails (classification gate, candidate validity, max 3 attempts),
same determinism, same audit-log/idempotency behavior — `policy_version`
distinguishes them (`policy-v1` heuristic vs. `policy-v2` candidate-aware),
and idempotency is enforced on `event_id` regardless of which version
decided it first (`test_idempotency_shared_across_policy_versions`).

## Reproduce

```bash
./venv/bin/python data/generate_counterfactual_dataset.py       # writes data/raw/counterfactual_outcomes.csv
./venv/bin/python model/train_candidate_model.py                # writes model/candidate_artifacts/
./venv/bin/python evaluation/evaluate_counterfactual_policy.py  # writes evaluation/reports/counterfactual_*
./venv/bin/python -m pytest tests/ -v                            # Day 1 + 2 + 3 + 4 + 5 + 6
```

## Limitations

- **Synthetic only, restated.** Every probability, coefficient, and
  archetype behavior here is an authored assumption, not fit to or derived
  from real data. See `data/README.md`'s disclaimer — it applies here too.
- **The candidate-aware model does not currently beat simple baselines.**
  See "Ranking quality and a surprising finding" above — reported, not
  hidden.
- **Regret and ranking metrics depend on the hand-designed latent
  mechanism**, not on any real recovery process — they measure "how well
  does the model recover the generator's own logic," not "how well would
  this work on real payments."
- **Small test set (60 events).** Ranking/regret numbers carry real sampling
  noise at this size; the realized-₹ money metrics additionally depend on
  actual Bernoulli draws, not just latent probabilities.
- **"Unnecessary interventions"** is defined pragmatically (a real action
  was taken but the realized outcome for that action was not recovered) —
  the dataset has no "what if we hadn't retried at all" counterfactual to
  compare against, only per-candidate outcomes.

## Day-6 acceptance checklist

- [x] Every failure has 5 counterfactual candidate outcomes
- [x] Candidate timing genuinely affects synthetic outcomes (archetype × timing interaction tested)
- [x] Candidate-aware CatBoost trained
- [x] Logistic baseline trained
- [x] No leakage (archetype structurally absent; latent probability, post-outcome, and split columns excluded and tested)
- [x] Oracle policy implemented (guardrail-respecting, reuses `decide_candidate_aware`)
- [x] AI policy evaluated against oracle (ranking accuracy, top-2 coverage, regret)
- [x] ₹ recovery comparison is now legitimate within the synthetic environment
- [x] Policy regret calculated (expected/latent-based AND realized/₹-based)
- [x] Existing guardrails/idempotency preserved (Day-5 tests untouched and passing)
- [x] Full test suite passes (149/149: 105 Day 1–5 + 44 Day 6)
- [x] Results clearly labeled synthetic throughout

---

# Day 7 — Within-Event Ranking

**"SYNTHETIC COUNTERFACTUAL EVALUATION"** — same disclaimer as Day 6: every
number below comes from `data/raw/counterfactual_outcomes.csv`, a
hand-designed simulation. **This does not measure real Razorpay recovery
performance.**

## Why Day 6's classification approach was insufficient

**High pooled AUC does not guarantee correct within-event retry selection.**
Day 6's candidate-aware CatBoost scored a respectable 0.80 pooled ROC-AUC,
but its top-1 candidate-selection accuracy was *below random* (16.7% vs.
20% chance) and its mean within-event rank correlation with the true latent
probability was **negative** (−0.149). Classification accuracy measures
"can the model tell easy events from hard events" — pooled across every
row. Ranking measures something structurally different: "for THIS event,
does the model get the order of its 5 candidates right." A model can ace
the first and fail the second, because between-event variance (archetype,
amount, prior history) can dwarf within-event variance (candidate timing)
in a pooled loss, letting a model "win" on AUC by explaining the former
alone.

## Diagnosis (performed before any new model code was written)

Reproducible via `model/diagnose_ranking_failure.py`. Findings:

1. **One feature dominates almost completely.** Day 6's CatBoost feature
   importance: `hours_from_failure` **93.3%**. Everything else — including
   the features actually meant to carry the causal timing signal —
   `candidate_days_to_payday` **0.03%**, `candidate_is_payday_aligned`
   **0.0%**, `candidate_is_month_end_aligned` **0.0%** — is crowded out
   almost entirely.
2. **That dominant feature is a feasibility proxy, not a preference
   signal.** `hours_from_failure` mostly captures whether a candidate's own
   scheduled time still leaves room to recover within the 14-day horizon
   (see Day 6: `payday_window`/`month_end_window` land beyond it 38%/58% of
   the time) — a real, legitimate, easy-to-fit effect that a shallow,
   early-stopped model (18 iterations) can exploit for most of its pooled
   AUC without ever learning the weaker payday-alignment interaction.
3. **The true signal exists but is weak.** Restricted to only
   horizon-valid candidates (removing the feasibility confound),
   `candidate_days_to_payday` alone correlates **+0.11** with latent
   probability, within-event — small but real, and a meaningful ceiling
   estimate against Day 6's measured **−0.149**.
4. **Event-level context also generally outranks candidate features.**
   Even setting `hours_from_failure` aside, `days_to_nearest_payday_window`
   (2.1%), `city_tier` (0.82%), `plan_tier` (0.50%) individually outrank
   `candidate_days_to_payday` (0.03%).
5. **Calibration is not a contributing factor.** Platt/sigmoid calibration
   is a monotonic transform — it provably cannot change within-group rank
   order. Verified empirically: identical within-event correlation
   (−0.149) before and after calibration.
6. **Class imbalance across candidate rows is not the driver either.**
   Recovery rates differ 35%–70% by `candidate_type` — that variation *is*
   the intended causal signal, not a data problem. Uniform-label groups
   (zero informative pairs — all 5 candidates share one label) affect only
   **62/313 events (19.8%)**, not enough to explain the magnitude of the
   failure on their own.
7. **The loss function optimizes the wrong thing.** Pointwise log loss +
   early stopping on *pooled* validation AUC never specifically rewards
   getting one event's 5 candidates in the right order — a model can score
   very well on both while getting every single event's internal ranking
   backwards, which is close to what happened.

## Classification vs. ranking

| | Day 6 (classification) | Day 7 (ranking) |
|---|---|---|
| Target | P(recovered) per row | Relative preference within a group |
| Loss | Pointwise log loss | Pairwise: "does A beat B?" |
| Unit of comparison | Every row, pooled | Only rows from the SAME event |
| Model selection | Pooled validation AUC | Pairwise validation AUC + validation top-1 (vs. realized label) |
| What a constant offset does | Changes every prediction | **Cancels out** in the difference — cannot "solve" ranking by fitting one global threshold |

## Approach selected

**Option A — pairwise ranking**, chosen over B (listwise) and C (raw
score-difference regression) as literally described in the brief: for
every event, for every (recovered, not-recovered) candidate pair, build a
training example whose feature vector is the **difference** of the two
candidates' (one-hot-encoded, scaled) feature vectors, and whose target is
"did the higher-recovery candidate rank first" (both directions included,
symmetric). CatBoost's specialized ranking-Pool API (`YetiRank`/`PairLogit`
with `group_id`) was considered and rejected in favor of building pairs
explicitly: more testable (`build_pairwise_dataset` has 4 dedicated unit
tests), fully transparent, and avoids a second, more opaque configuration
surface — "do not introduce a massive ranking framework." At inference
time, an event's 5 candidates are scored via **round-robin tournament**:
each candidate's score is its mean predicted "beats" probability against
the other 4 candidates from the *same* event (`score_candidates_for_event`)
— a natural [0, 1] score, structurally incapable of "cheating" by fitting
one global offset, since a constant added to every candidate's feature
vector cancels out in the subtraction.

## Feature set

Exactly the brief's section-5 list (`model/ranking_preprocessing.py`) — a
deliberate narrowing from Day 6's set: `app_version`/`device_build`/
`ui_theme` (proven non-predictive distractors) are dropped entirely, not
just left at zero importance, since a model already starved for
within-group signal shouldn't spend capacity on columns known to carry
none. Failure-time/customer context and candidate-action features are
otherwise identical to Day 6's; `archetype`, `recovery_probability_latent`,
`recovered_at`, `recovered_via`, `amount_recovered` remain fully excluded
(`model/ranking_preprocessing.py::EXCLUDED_COLUMNS`) — the pairwise
targets are built exclusively from `recovered_within_14d`, never from any
post-outcome or hidden field.

## Training setup

Same subscription-level 60/20/20 split as every prior day — a subscription
and all 5× of its candidate rows land in exactly one split
(`test_same_event_never_crosses_splits`). 970/295/300 candidate rows
(194/59/60 events). 1,648 pairwise training examples built from the 194
train events; 516 from the 59 validation events. CatBoost trained on the
pairwise-difference features (300 iterations, depth 4, early stopping on
*pairwise* validation AUC — best iteration 178, pairwise validation AUC
**0.8374**, validation top-1 accuracy vs. the realized label **0.76**).

## Evaluation metrics (test, n=60 events, held out, never used for tuning)

| Policy | Top-1 | Top-2 | MRR | NDCG@5 | Mean rank | Pairwise acc. | Avg. regret (₹) |
|---|---|---|---|---|---|---|---|
| Random candidate | 0.200 | 0.367 | 0.447 | 0.930 | 3.10 | 0.471 | 62.48 |
| Fixed Retry | 0.150 | 0.300 | 0.392 | 0.935 | 3.52 | 0.553 | 65.28 |
| Day-6 probability model | 0.083 | 0.250 | 0.345 | 0.929 | 3.68 | 0.456 | 76.05 |
| **Day-7 ranking model** | **0.100** | **0.267** | **0.356** | **0.920** | **3.65** | **0.432** | **79.40** |
| Oracle (upper bound) | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 | 1.000 | 0.00 |

Pooled classification metrics, for reference only (Day 7's model was never
optimized for this): Day-6 ROC-AUC 0.797 / Day-7 ROC-AUC 0.698 — Day-7's
pooled AUC is *lower*, exactly as expected for a model trained purely on
within-group comparisons rather than pooled discrimination.

## Ablation study (top-1 / NDCG@5 / mean rank vs. Oracle, test set)

| Ablation | Top-1 | NDCG@5 | Mean rank |
|---|---|---|---|
| A. Event context only | 0.150 | 0.935 | 3.52 |
| B. Candidate features only | 0.133 | 0.933 | 3.55 |
| C. Event + candidate, **pointwise** objective | 0.183 | 0.933 | 3.57 |
| D. Day-6 probability model (pointwise, distractors included) | 0.083 | 0.929 | 3.68 |
| E. Day-7 ranking model (pairwise objective) | 0.100 | 0.920 | 3.65 |

This isolates features from objective, as section 9 intended: **C beats D**
(dropping distractors and using a cleaner feature set alone helps a
pointwise model, 0.183 vs. 0.083), but **C also beats E** — the ranking
objective alone, on this dataset and at this scale, did not outperform a
correctly-featured pointwise model. Every trained variant (A–E) still
underperforms **Random** (0.200). The honest reading: the diagnosed
feature-dominance problem was real and partially addressed (see feature
importance below), but a second, independent problem — described next —
limits how much any of these approaches can close the gap.

## A second, deeper finding: the training target is a noisy, confounded proxy

Feature importance for the Day-7 pairwise model (transformed space):
`hours_from_failure` 41.4% + `candidate_day_of_month` 35.5% ≈ 77% combined
— still dominant, but `candidate_days_to_payday` rose to **3.1%** (~100×
Day 6's 0.03% share) and `candidate_is_payday_aligned` /
`candidate_is_month_end_aligned` moved from **exactly 0.0%** to **0.94%** /
**0.79%**. The pairwise objective *did* measurably redirect the model's
attention toward the intended causal features — a real, quantifiable,
honest improvement in that specific sense. It was not enough to overcome
the residual dominance of the horizon-feasibility proxy at this dataset
size, and — more fundamentally — **`recovered_within_14d` is a single
noisy Bernoulli draw, further confounded by horizon-truncation, of the true
latent preference it is supposed to proxy for**. A model can get
genuinely, measurably better at predicting that noisy/confounded target
(validation pairwise AUC 0.84, validation top-1 vs. realized label 0.76 —
both real, both improvements) without that improvement transferring to
agreement with the true latent ordering on held-out data, because the two
are not the same target. This is not a bug to patch; it is a property of
what `recovered_within_14d` actually measures.

## Money evaluation (synthetic, n=60 test events)

| Policy | ₹ recovered | Rate | Avg. ₹/event | Lift vs. Fixed Retry | Incremental ₹ vs. Fixed | Regret vs. Oracle (₹) |
|---|---|---|---|---|---|---|
| Random candidate | 18,687.89 | 65.0% | 311.46 | −15.0pp | −3,166.21 | 5,587.41 |
| Fixed Retry | 21,854.10 | 80.0% | 364.23 | +0.0pp | 0.00 | 2,421.20 |
| Day-6 probability model | 19,317.96 | 70.0% | 321.97 | −10.0pp | −2,536.14 | 4,957.34 |
| **Day-7 ranking model** | **19,742.65** | **70.0%** | **329.04** | **−10.0pp** | **−2,111.45** | **4,532.65** |
| Oracle (upper bound) | 24,275.30 | 86.7% | 404.59 | +6.7pp | +2,421.20 | 0.00 |

Day-7 recovers **₹424.69 more than Day-6** (a small, real, consistent
improvement — same recovery rate, higher amount because it more often
picks a higher-value event's better-scoring candidate) and has **lower
regret vs. Oracle** than Day-6 (₹4,532.65 vs. ₹4,957.34). It does **not**
beat Fixed Retry, and both trained models remain behind Random on pure
ranking accuracy on this 60-event test set.

## Sanity checks

Every one of the 313 events has exactly 5 candidates
(`test_ranking_groups_contain_exactly_five_candidates`); every policy
selects exactly one candidate (or `NO_ACTION`) per event, always a member
of the 5 known types; no policy other than the intentionally-fixed baseline
concentrates on one candidate type
(`random`/`day6`/`day7`/`oracle_no_single_candidate_dominates` all `True`
— `fixed_retry`'s 100% concentration on `plus_1_day_morning` is by
definition, not a red flag). Guardrails (classification gate, horizon
validity, max attempts) apply identically regardless of which score source
feeds `decide_candidate_aware`.

## Policy integration

**No new policy code.** `policy/recovery_policy.py::decide_candidate_aware`
/ `decide_for_failure_event_candidate_aware` — Day 6's functions, untouched
— accept any `candidate_type -> probability`-shaped dict; Day 7's
round-robin ranking scores are just a different (and, per the money
evaluation, modestly better) source for that dict. Every Day-5 guardrail
(`retryable_soft`-only, no duplicate action, cancellation/hard-decline
block action, invalid/beyond-horizon candidates blocked, deterministic
decisions, full audit trail) applies identically and is re-verified against
real ranking-model output in `tests/test_ranking_policy.py`, not just
against hand-crafted toy probabilities.

## Reproduce

```bash
./venv/bin/python model/diagnose_ranking_failure.py         # reproduces the diagnosis numbers above
./venv/bin/python model/train_ranking_model.py               # writes model/ranking_artifacts/
./venv/bin/python evaluation/evaluate_ranking_policy.py      # writes evaluation/reports/ranking_*
./venv/bin/python -m pytest tests/ -v                         # Day 1 + 2 + 3 + 4 + 5 + 6 + 7
```

## Limitations

- **Day 7 did not solve retry-time selection.** It measurably fixed the
  specific optimization mismatch it targeted (feature attention shifted
  toward the true causal signal; pairwise validation metrics genuinely
  improved; ₹ recovered and regret vs. Oracle both improved over Day 6) but
  did **not** beat Fixed Retry on money, and did **not** improve — and by
  a small amount, slightly worsened — agreement with the Oracle on
  held-out top-1 accuracy. This is reported as-is, not tuned toward a
  better number.
- **The deeper limitation is the target, not (only) the objective.**
  `recovered_within_14d` conflates true timing preference with horizon
  feasibility and single-draw Bernoulli noise. No amount of feature-set or
  loss-function refinement within Day 7's no-cheating constraints (no
  `recovery_probability_latent` as an input) can fully separate these.
- **Small test set (60 events).** At this scale, `~2` standard errors
  separate Day-6's/Day-7's top-1 accuracy from the 20% chance rate — a real
  effect, not pure noise, but the exact numbers would move with a larger
  regeneration.
- **Not tuned toward test.** Hyperparameters (300 iterations, depth 4,
  pairwise-AUC early stopping) were fixed before looking at test-set
  ranking/money numbers and were not revisited after seeing them.
- **Recommendations for Day 8+ (not implemented here, to avoid exactly the
  blind-tuning trap the brief warns against):** (a) regenerate with more
  subscriptions so within-event variance carries more statistical power;
  (b) reconsider whether `recovered_within_14d` should be measured relative
  to candidate time rather than failure time, reducing the
  horizon-truncation confound; (c) restrict pairwise training to
  horizon-valid candidate pairs only, isolating the feasibility split from
  the pairs the model is asked to learn from.

## Day-7 acceptance checklist

- [x] Root cause identified (feature dominance + objective mismatch, reproducible via `model/diagnose_ranking_failure.py`)
- [x] Ranking-specific model implemented (pairwise, `model/train_ranking_model.py`)
- [x] Candidate groups preserved (never split across train/val/test; pairs never cross events)
- [x] No leakage (archetype/latent/post-outcome fields excluded and tested; distractors dropped)
- [x] Ranking metrics implemented and tested (top-1, top-2, MRR, NDCG@5, mean rank, pairwise accuracy, regret)
- [x] Day-6 baseline retained (unchanged artifacts, unchanged code, still evaluated side-by-side)
- [x] Policy updated to consume the Day-7 model (via the existing, unmodified `decide_candidate_aware`)
- [x] Synthetic ₹ evaluation completed, clearly labeled synthetic throughout
- [x] Full test suite passes (188/188: 149 Day 1–6 + 39 Day 7)
- [x] **Results honestly reported — Day 7 did NOT solve retry-time selection.** It improved on Day 6 in several measurable, real ways (pairwise validation metrics, ₹ recovered, regret vs. Oracle) without closing the gap to Fixed Retry or to the Oracle's top-1 agreement. No result was tuned to look better than it is.

---

# Day 8 — Fixing the Target, Not Just the Model

**"SYNTHETIC COUNTERFACTUAL EVALUATION"** — same disclaimer as Day 6/7:
every number below comes from `data/raw/counterfactual_outcomes.csv`, a
hand-designed simulation. **This does not measure real Razorpay recovery
performance.** `recovery_probability_latent` and
`expected_recovery_value_latent` are **synthetic benchmark targets** this
project's own generator authored to make offline evaluation possible — see
"Why the latent target is synthetic-only" below. **The latent target exists
only because this is a synthetic benchmark. A production system would need
an equivalent observable target built from historical retries and their
outcomes.**

## Why Day 7 failed

Day 7 tried to fix Day 6's within-event ranking failure by changing the
**objective** (pointwise → pairwise) while keeping the same **target**
(`recovered_within_14d`). It measurably shifted feature attention toward
the right signal (`candidate_days_to_payday` importance rose ~100×) but did
not close the gap: top-1 accuracy against the Oracle stayed *below random*
(Random 20%, Day-6 8.3%, Day-7 10%), and Day-7 still recovered less than
Fixed Retry (₹19,742.65 vs. ₹21,854.10). Day 7's own conclusion was that
the deeper problem was the **target itself**:
`recovered_within_14d` is a single noisy Bernoulli draw, further confounded
by 14-day horizon truncation, of the true latent preference it's supposed
to proxy for. No amount of objective engineering on a noisy/confounded
target reliably recovers the true ordering. Day 8 tests that conclusion
directly by training against the generator's own latent ground truth
instead.

## Three concepts, explicitly separated (`model/latent_target_preprocessing.py`)

| | Column | What it is |
|---|---|---|
| **A. Observed outcome** | `recovered_within_14d` | A noisy Bernoulli **realization** — one coin-flip per (event, candidate), sampled from the latent probability and truncated to `False` beyond the 14-day horizon. What Day 6/7 trained against. |
| **B. Latent recovery probability** | `recovery_probability_latent` | The synthetic **ground truth** the generator computed *before* sampling (A) from it. Continuous, `[0.02, 0.98]`. |
| **C. Latent expected money value** | `expected_recovery_value_latent` = B × `amount` | The synthetic ground-truth **economic** objective — not "how likely," but "how many ₹ do we expect." Day 8's **primary** target (what a hackathon track cares about). |

These are never treated as interchangeable in code: each has its own
column, its own documented exclusion entry, and its own place in
`EXCLUDED_COLUMNS` so a model can never accidentally ingest one as a
feature.

## Why the latent target is synthetic-only

Training against `recovery_probability_latent` / `expected_recovery_value_latent`
is legitimate **here** because this project's own generator
(`data/generate_counterfactual_dataset.py`) authored them as a known,
self-created ground truth for benchmarking — exactly the same legitimacy
Day 6/7's Oracle upper bound already relied on. It is **not** a claim that
either column is observable, computable, or available in any real
deployment. **The latent target exists only because this is a synthetic
benchmark. A production system would need an equivalent observable target
built from historical retries/outcomes** — e.g. aggregated recovery rates
per candidate-timing bucket over enough real retries to be statistically
stable, which this project has never had (real retries are explicitly out
of scope through Day 8). Both latent columns are hard-excluded from
`FEATURE_COLUMNS` (`model/latent_target_preprocessing.py::EXCLUDED_COLUMNS`),
identically to how `archetype` has been excluded since Day 3.

## Target construction

```
expected_recovery_rate_latent  = recovery_probability_latent          # explicit alias (brief section 2)
expected_recovery_value_latent = recovery_probability_latent × amount  # PRIMARY target
```

Pure, deterministic, no fitting (`add_latent_targets`). Validated
(`validate_latent_targets`, and `model/train_latent_target_model.py::main()`
refuses to train if validation fails): probability in `[0,1]`; value never
negative; value never exceeds the original `amount`; the alias is exact.
Intervention cost is **not** added — the existing architecture
(`policy/scoring.py::score_candidate_with_model_probability`) already
supports a non-zero `intervention_cost` cleanly without any change, so
there was nothing to build.

## Feature set

**Unchanged from Day 7** — `model/latent_target_preprocessing.py` imports
`FEATURE_COLUMNS` directly from `model/ranking_preprocessing.py` rather
than redefining it, since the brief's section-4 list is identical to
Day 7's section-5 list. Distractors remain excluded; `archetype` remains
structurally absent (never joined into the candidate-level table at all).

## Two models, one controlled experiment (`model/train_latent_target_model.py`)

| | Model A | Model B |
|---|---|---|
| Target | `recovery_probability_latent` | `expected_recovery_value_latent` (**primary**) |
| Main model | CatBoostRegressor (RMSE loss) | CatBoostRegressor (RMSE loss) |
| Baseline | LinearRegression | LinearRegression |
| Calibration | **None** — see below | **None** |

**Per the brief section 9: no Day-4-style sigmoid/isotonic calibration is
applied.** That machinery calibrates a binary classifier's probability
output against a binary label; forcing it onto a continuous regression
target would be a category error. Where a probability-shaped value is
needed downstream (feeding Model B into the existing policy architecture,
which expects `predicted_recovery_probability`), it's obtained by a
simple, explicit, documented conversion — `predicted_value / amount`,
clipped to `[0,1]` — never by `CalibratedClassifierCV`.

## Group-aware evaluation

Same subscription-level 60/20/20 split as every prior day (970/295/300
candidate rows, 194/59/60 events) — train fits, validation selects
(regression RMSE), test evaluates once. Primary ranking ground truth is
`expected_recovery_value_latent` per the brief, for **every** policy
(including Model A's probability-based scores) — this is a deliberate,
verified choice: `amount` is constant across one event's 5 candidates, so
ranking by probability and ranking by probability×amount are
**mathematically identical within an event**
(`sanity_checks`: `latent_probability_and_value_argmax_always_agree_within_event`
— verified `True` on all 60 test events). Model A and Model B can still
diverge in practice, though, because they are two **different fitted
models** on two different target scales with different loss surfaces (RMSE
on a `[0,1]` value vs. RMSE on a ₹ value whose variance is dominated by
`amount`'s cross-event spread) — their predictions are not forced to be
proportional to each other.

## Model A vs. Model B — not decided on one metric (test, n=60 events)

| Metric | Random | Fixed Retry | Rule-Based | Day-6 | Day-7 | **Model A** | **Model B** | Oracle |
|---|---|---|---|---|---|---|---|---|
| Top-1 | 0.200 | 0.150 | 0.133 | 0.083 | 0.100 | 0.250 | **0.283** | 1.000 |
| Top-2 | 0.367 | 0.300 | 0.283 | 0.250 | 0.267 | **0.633** | 0.583 | 1.000 |
| MRR | 0.447 | 0.392 | 0.381 | 0.345 | 0.356 | 0.532 | **0.543** | 1.000 |
| NDCG@5 | 0.930 | 0.935 | 0.931 | 0.929 | 0.920 | **0.957** | 0.955 | 1.000 |
| Mean rank | 3.10 | 3.52 | 3.55 | 3.68 | 3.65 | **2.55** | **2.55** | 1.00 |
| Pairwise acc. | 0.471 | 0.553 | 0.569 | 0.456 | 0.432 | 0.618 | 0.618 | 1.000 |
| Avg. regret (₹) | 62.48 | 65.28 | 66.36 | 76.05 | 79.40 | 71.58 | **45.88** | 0.00 |

**Both models beat every prior policy — including Random — on every
ranking metric.** This is the headline finding: fixing the target, not the
model or the objective, was the correct diagnosis. Model A edges out on
pure ranking-order metrics (top-2, NDCG@5); Model B wins on regret (₹) —
the metric that weights *how costly* a ranking mistake is, not just
whether one occurred.

## Economic metrics settle it — Model B is selected

| | Random | Fixed Retry | Rule-Based | Day-6 | Day-7 | Model A | **Model B** | Oracle |
|---|---|---|---|---|---|---|---|---|
| Total latent value (₹) | 19,282.75 | 19,114.77 | 19,050.32 | 18,468.66 | 18,267.93 | 18,737.00 | **20,278.58** | 23,031.64 |
| vs. Fixed Retry (₹) | +167.98 | 0.00 | −64.45 | −646.11 | −846.84 | −377.77 | **+1,163.81** | +3,916.87 |
| Regret vs. Oracle (₹) | 3,748.89 | 3,916.87 | 3,981.32 | 4,562.98 | 4,763.71 | 4,294.64 | **2,753.06** | 0.00 |

**Model B is the only trained model — of six — that beats Fixed Retry on
the noise-free latent economic ground truth.** This directly answers the
brief's section-6 question ("which objective better matches 'revenue
recovery'"): predicting ₹ directly (Model B) matches the actual decision
objective better than predicting a probability and hoping the downstream
architecture's probability×amount arithmetic sorts candidates the same
way Model A's own fitting process happened to prioritize. **Model B is
selected as Day 8's primary model.**

## Realized (stochastic) counterfactual results — separated from the latent numbers

| | Random | Fixed Retry | Rule-Based | Day-6 | Day-7 | Model A | **Model B** | Oracle |
|---|---|---|---|---|---|---|---|---|
| ₹ recovered | 18,687.89 | 21,854.10 | 21,431.15 | 19,317.96 | 19,742.65 | 19,369.01 | **21,372.13** | 24,275.30 |
| Recovery rate | 65.0% | 80.0% | 76.7% | 70.0% | 70.0% | 70.0% | **75.0%** | 86.7% |
| Lift vs. Fixed Retry | −15.0pp | +0.0pp | −3.3pp | −10.0pp | −10.0pp | −10.0pp | **−5.0pp** | +6.7pp |

Model B is the best-performing trained model here too — but **still
technically trails Fixed Retry** on this single stochastic draw. Checked
explicitly, not glossed over: at n=60, the standard error of a rate
difference this size is ≈7.6pp; the observed 5.0pp gap is **0.66 standard
errors — not statistically distinguishable from zero at this sample
size.** The honest summary: **Model B beats Fixed Retry on the noise-free
latent economic ground truth (what it should achieve in expectation), and
is statistically indistinguishable from Fixed Retry on the one noisy
realized draw available for testing.** These are reported as two different
facts about two different things, not reconciled into one number.

## Sanity checks

Every one of the 313 events has exactly 5 candidates; target formula
(`expected_recovery_value_latent == recovery_probability_latent × amount`)
verified exactly on every row; probability bounded `[0,1]`; value never
negative, never exceeds `amount`; both latent columns absent from
`FEATURE_COLUMNS` and `archetype` structurally absent from the
candidate-level table; no event crosses a split; repeated training with
the same seed is deterministic (`tests/test_latent_target_model.py`).

## Important scientific check (brief section 11)

Before writing any of the above up as success, Day-8 model rankings were
compared directly against `expected_recovery_value_latent` — the actual
synthetic latent target, not a proxy. **The models do not rank badly** —
both clear the "worse than random" bar Day 6/7 failed to clear, by a wide
margin. There was no need to invoke the fallback finding ("observable
candidate features do not fully identify the synthetic latent preference")
because the result did not require it. What *does* remain, honestly: even
Model B's top-1 accuracy (28.3%) is far from Oracle's 100%, and its
regret vs. Oracle (₹45.88/event average) is real, not zero — observable
candidate features explain *some but not all* of the latent preference,
which is exactly what a noise term (`NOISE_STD = 0.9` on the generator's
logit scale, by design — see `data/README.md`) should produce.

## Policy integration

**No new policy code, again.** Model A's probability output feeds
`policy/recovery_policy.py::decide_candidate_aware` directly (unchanged
since Day 6). Model B's ₹ output is converted via `predicted_value /
amount` (clipped `[0,1]`) before the same call — documented in
`evaluation/evaluate_latent_target_policy.py::compute_all_scores`. Every
Day-5 guardrail (classification gate, horizon validity, max attempts, full
audit trail, idempotency) applies identically and is re-verified against
real Model-A/Model-B output in `tests/test_latent_target_policy.py`.

## Reproduce

```bash
./venv/bin/python model/train_latent_target_model.py            # writes model/latent_target_artifacts/{probability,value}/
./venv/bin/python evaluation/evaluate_latent_target_policy.py   # writes evaluation/reports/latent_target_*
./venv/bin/python -m pytest tests/ -v                             # Day 1 + 2 + ... + 8
```

## Limitations

- **Model B still loses to Oracle by a wide margin** (₹2,753.06 regret;
  28.3% top-1 vs. 100%). Fixing the target closed most, not all, of the
  gap — real headroom remains, most plausibly in feature richness (see
  Day 7's ablation: even a perfectly-clean feature set under a pointwise
  objective topped out around 18% top-1 against Day-7's ground truth).
- **`expected_recovery_value_latent` R² (0.87) is partly an artifact of
  `amount`'s cross-event variance**, not proof the model captured the
  candidate-timing signal well — `amount` alone is a strong, directly
  observable predictor of ₹ value (higher-amount subscriptions have
  higher-value outcomes almost by construction). The WITHIN-event ranking
  metrics above are the metrics that actually isolate candidate-timing
  discrimination; pooled R² is reported for completeness, not as the
  headline number.
- **The realized-money win is not yet statistically confirmed** at n=60
  test events (0.66 SE gap to Fixed Retry) — see above. A larger dataset
  regeneration would sharpen this.
- **Synthetic only, restated.** Every number in this section comes from a
  hand-designed simulation. It is not evidence about real Razorpay
  recovery behavior, and the latent target this day's entire improvement
  rests on has no real-world equivalent without new data collection.
- **Day 6 and Day 7 are retained, not deleted** — both remain trained,
  evaluated, and tested exactly as before, for the clean before/after
  comparison this day's finding depends on.

## Day-8 acceptance checklist

- [x] Latent probability target formalized (`recovery_probability_latent`, documented as concept B)
- [x] Latent expected-value target formalized (`expected_recovery_value_latent`, documented as concept C, primary target)
- [x] Targets kept out of model features (excluded + tested, both new columns)
- [x] Regression baseline trained (LinearRegression, both targets)
- [x] CatBoost regression trained (CatBoostRegressor, both targets)
- [x] Within-event ranking evaluated (top-1/top-2/MRR/NDCG@5/mean rank/pairwise/regret, ground truth = latent value)
- [x] Day-7 retained as baseline (unchanged code and artifacts, still evaluated side-by-side)
- [x] Synthetic ₹ evaluation completed, clearly labeled synthetic throughout
- [x] Realized vs. latent metrics separated (sections C and D never conflated; statistical significance of the realized gap explicitly checked)
- [x] No leakage (both latent columns + archetype + post-outcome fields excluded and tested)
- [x] Full test suite passes (218/218: 188 Day 1–7 + 30 Day 8)
- [x] **Results honestly reported.** Model B beats every other trained policy on the latent economic ground truth and is the first to statistically match Fixed Retry on realized money — but still trails Oracle by a wide margin, and the realized-money win over Fixed Retry is not yet statistically significant at this sample size. Neither overclaimed nor undersold.

---

# Day 9 — Production-Shaped Recovery Decision Engine

**"SYNTHETIC COUNTERFACTUAL EVALUATION"** — same disclaimer as Day 6/7/8:
every number below comes from `data/raw/counterfactual_outcomes.csv`, a
hand-designed simulation. **This does not measure real Razorpay recovery
performance.**

## Why no new ML model

Day 8 already found the right target — Model B beat every prior policy on
the latent economic ground truth. Day 9's job is to turn that prediction
into something that behaves like a real decision system: aware of cost, not
blindly confident, with a defined fallback when the model can't be trusted.
**None of that requires new model weights** — `model/latent_target_artifacts/value/`
(Day 8's Model B) is loaded and used exactly as trained. **No new model was
added; this checklist item is satisfied by omission, not by exception.**

## Value-native decisioning

`policy/decision_engine.py` is new (Day 6/7/8 all reused
`policy/recovery_policy.py::decide_candidate_aware` by converting a
predicted value into a probability-equivalent first). Day 9 does the
opposite deliberately — Model B's ₹ output is used **directly**, never
converted to a probability, because `decide_candidate_aware`'s
probability×amount interface doesn't fit cost-aware net-value scoring or a
rule-based fallback chain that doesn't produce a probability at all.

```
expected_net_value = predicted_recovery_value - intervention_cost
selected = argmax(expected_net_value) among valid, positive-net-value candidates
```

## Cost model (`policy/costs.py`)

One central, configurable dataclass — no cost is ever hard-coded in policy
logic:

```python
@dataclass(frozen=True)
class InterventionCosts:
    retry_cost: float = 5.0          # Rs -- SYNTHETIC PROJECT ASSUMPTION, not a Razorpay price
    whatsapp_cost: float = 0.0       # placeholder -- no WhatsApp channel implemented
    sms_cost: float = 0.0            # placeholder -- no SMS channel implemented
    voice_cost: float = 0.0          # placeholder -- no voice channel implemented
    operational_cost: float = 0.0    # placeholder -- fixed per-decision overhead, not yet modeled
```

All 5 candidate types are automated retries and share `retry_cost` +
`operational_cost` (`cost_for_candidate`). The other three fields exist so
a future communication channel can be priced without another schema
change — **nothing selects those channels yet; no LLM/WhatsApp/voice logic
exists in this project**, same as every prior day.

## Confidence, honestly labeled

**"Model confidence is represented only by a deterministic decision-margin
abstention rule; this is not calibrated probabilistic uncertainty."** The
engine never estimates a variance or a calibrated confidence interval — it
compares the best candidate's net value against the second-best's
(`decision_margin`) and abstains from trusting the model when that gap is
too small to trust, exactly as the brief specifies. Guardrails checked, in
order, before any model call: duplicate decision, classification bucket,
max retry attempts, candidate validity. Then: model output validity
(finite, non-negative, ≤2× `amount` — a implausible prediction is treated
as malformed, not as a confident outlier), decision margin, and finally
positive net value.

## Fallback chain

Three tiers, always producing a well-defined outcome, never an unhandled
exception:

1. **PRIMARY — `day8_model_b`**: used when the model loads, predicts
   without error, produces well-formed output for every candidate, and the
   margin between best and second-best clears the threshold.
2. **FALLBACK — `rule_based_fallback`**: triggered by model unavailability,
   a prediction exception, malformed output (NaN/negative/implausibly
   huge), missing required features, *or* an insufficient decision margin.
   When the model DID run successfully (low-margin case), its own
   prediction for whichever candidate Rule-Based picks is reused in the
   decision record — no wasted computation, no fabricated number. When the
   model genuinely failed, `predicted_recovery_value` is honestly `None`.
3. **`no_action`**: the final safety net — no valid candidate, no positive
   net value (checked in both the primary and fallback tiers), or the
   fallback's own pick is itself invalid.

`decision_source` is always one of these three strings, recorded on every
row — **never silent**.

## Decision object

`policy/decision_engine.py::Decision` — a frozen dataclass with every field
the brief lists (`event_id`, `subscription_id`, `classification_bucket`,
`selected_candidate_type`, `selected_candidate_datetime`,
`predicted_recovery_value`, `intervention_cost`, `expected_net_value`,
`runner_up_value`, `decision_margin`, `decision_source`, `model_version`,
`policy_version`, `decision_reason`, `created_at`), `.to_json()`-serializable.
`created_at` is excluded from equality (`compare=False`, same pattern
Day 5's `DecisionResult.candidate_scores` already established) so
`decide_engine()` stays provably deterministic — two calls with identical
inputs produce an identical `Decision`, timestamp aside.

Persistence extends `policy_decisions` additively (`app/models.py`) —
`classification_bucket`, `intervention_cost`, `runner_up_value`,
`decision_margin`, `decision_source`, `model_version` are new nullable
columns; `expected_recovery_value` / `expected_incremental_value` (Day 5)
are reused as-is for `predicted_recovery_value` / `expected_net_value` —
same formula, no duplicate columns. Every decision (including `NO_ACTION`,
fallback, and duplicate ones) still gets exactly one `audit_log` row, with
the full per-candidate score breakdown (including rejected/invalid
candidates) embedded as JSON — no secrets ever flow through this layer
(webhook secrets and API keys belong to Day 1's ingestion path, never
touched here).

## Validation-only threshold selection

Searched `{0, 10, 25, 50, 100, 150, 200, 250}` (Rs of decision margin) on
the 59 **validation** events only, scored by total **latent** net value
selected (the noise-free economic ground truth — legitimate for offline
threshold tuning, same reasoning Day 8 used for evaluation):

| Threshold (Rs) | Total latent value selected (validation) |
|---|---|
| 0 | 18,258.62 |
| **10** | **18,590.40 ← selected** |
| 25 | 17,997.99 |
| 50–250 | 17,997.99 (flat — fallback path fully saturates above Rs25) |

**Rs10 was frozen into `DEFAULT_ABSTENTION_THRESHOLD_RS` and the test split
was run exactly once, after freezing.** An earlier placeholder value
(Rs25, picked before the search was built) was corrected to match the
search's actual output before any test-set number was looked at — the
search result was used, not adjusted to match a guess.

## An honest surprise: the frozen threshold underperformed on test

Evaluated on all 60 test events, with the threshold frozen:

| | Latent economic (Rs) | Realized (Rs, rate) |
|---|---|---|
| Fixed Retry | 19,114.77 | 21,854.10 (80.0%) |
| Rule-Based | 19,050.32 | 21,431.15 (76.7%) |
| Day-8 Model B, no abstention | **20,347.65** | 21,278.18 (73.3%) |
| **Day-9 decision engine (Rs10 threshold)** | 18,748.76 | 20,148.47 (71.7%) |
| Oracle | 23,031.64 | 24,275.30 (86.7%) |

**The abstention/fallback mechanism made test-set performance worse, not
better**, compared to trusting Model B's raw argmax with no abstention at
all. Verified, not just suspected: of the 60 test events, 41 (68.3%)
triggered fallback under the Rs10 threshold. On exactly those 41 events,
Model B's own (untrusted) argmax choice would have totaled **Rs13,795.84**
in latent value — Rule-Based, which the engine used instead, delivered
only **Rs12,196.95** — a **Rs1,598.89 opportunity cost** from abstaining.
Cross-checked directly against both baselines (`fallback` selections match
Rule-Based's own picks exactly on those 41 events; `non-fallback`
selections match Model B's raw argmax exactly on the other 19) — this is
not a bug in the fallback wiring, it is a real property of this specific
validation→test split. Most plausibly: 59 validation events is too few for
a margin threshold to generalize reliably, and a small predicted margin
did not turn out to reliably predict low true regret at this sample size —
in the same spirit as Day 7's finding that a model's own confidence signal
doesn't automatically track ranking correctness. **This was not re-tuned
after seeing it.** The threshold search was run once, correctly, on
validation only, and its result is reported as-is.

## Example decisions

| Scenario | Selected | Source | Net value (Rs) | Margin (Rs) |
|---|---|---|---|---|
| `retryable_soft`, low margin | `payday_window` | `rule_based_fallback` | 710.60 | — (fallback) |
| `retryable_soft`, high margin | `payday_window` | `day8_model_b` | 964.08 | 14.53 |
| `hard_decline` | `NO_ACTION` | `no_action` | — | — |
| `customer_cancelled` | `NO_ACTION` | `no_action` | — | — |
| `unmapped` | `NO_ACTION` | `no_action` | — | — |

## Operational metrics (test, n=60, Rs10 threshold)

- Actions selected: 60/60 (0 `NO_ACTION` on this test set)
- Fallback count: 41 (68.3% of decisions)
- Average decision margin (non-fallback decisions): Rs22.13 (n=19)
- `decision_source` distribution: `rule_based_fallback` 41, `day8_model_b` 19

## Reproduce

```bash
./venv/bin/python evaluation/evaluate_decision_engine.py   # runs the validation threshold search, then the frozen test evaluation, once
./venv/bin/python -m pytest tests/ -v                        # Day 1 + 2 + ... + 9
```

## Limitations

- **The abstention mechanism did not improve outcomes on this test set** —
  see above. It is retained because it is the mechanism the brief asked
  for and the failure-mode/fail-closed behavior it provides (never
  crashing, always a well-defined `decision_source`) is valuable
  independent of whether the specific threshold helped here; a larger
  validation set is the most likely fix, not a different threshold-search
  heuristic.
- **`retry_cost = Rs5` and all other cost fields are illustrative project
  assumptions**, not sourced from any real payment-gateway or messaging
  vendor pricing.
- **Still entirely synthetic.** Every number in this section depends on
  `data/raw/counterfactual_outcomes.csv`'s hand-designed generator; nothing
  here is evidence about real Razorpay recovery behavior.
- **No communication channels exist yet.** `whatsapp_cost` / `sms_cost` /
  `voice_cost` are configured but unused — no LLM, WhatsApp, SMS, or voice
  logic exists in this project through Day 9.
- **Day 6/7/8 are retained, not replaced** — `policy/recovery_policy.py`
  and its `decide_candidate_aware` path are untouched; Day 9 adds a
  parallel, more production-shaped engine rather than modifying the
  existing one.

## Day-9 acceptance checklist

- [x] Day-8 Model B used (loaded from `model/latent_target_artifacts/value/`, not retrained)
- [x] Cost-aware decisioning (`policy/costs.py`, one central config, never hard-coded)
- [x] Net-value scoring (`expected_net_value = predicted_recovery_value - intervention_cost`, value-native, no probability conversion)
- [x] Deterministic abstention (decision-margin rule, explicitly labeled as not calibrated uncertainty)
- [x] Validation-only threshold selection (searched `{0,10,25,50,100,150,200,250}` on 59 validation events; Rs10 frozen; test run once)
- [x] Rule-based fallback (3-tier chain: model → rule-based → `NO_ACTION`, never silent)
- [x] Fail-closed behavior (12 failure modes tested: missing model file, NaN/negative/huge predictions, prediction exceptions, insufficient features, empty/all-invalid candidates, non-retryable classification, low margin, max attempts, duplicate event — none crash, all resolve to a well-defined `decision_source`)
- [x] Full audit trail (every decision, including `NO_ACTION` and fallback, gets one `audit_log` row with full candidate-score JSON; no secrets logged)
- [x] Idempotency preserved (same event_id/subscription pattern as Day 5–8, tested)
- [x] Existing guardrails preserved (all of Day 5's, none removed, all re-tested against the new engine)
- [x] Latent and realized metrics separated (never conflated in the evaluation report or this section)
- [x] Full test suite passes (250/250: 218 Day 1–8 + 32 Day 9)
- [x] No new ML model added (Day-8 Model B reused as-is throughout)

# Day 10 — Improved Fallback/Abstention Policy (policy-v4)

**"Thresholds and fallback parameters were selected using validation data only."**

Day 9 shipped a working three-tier decision engine, but honestly reported a
surprise: its abstention/fallback mechanism made test-set outcomes *worse*
than trusting Day-8 Model B directly (Rs18,748.76 vs Rs20,347.65 latent
value — see the Day-9 section above). Day 10's job is narrow: **diagnose
that regression and fix the fallback/abstention *logic*, without touching
Model B, without adding a new model, and without deleting Day 9.**
`policy/decision_engine.py` (policy-v3) is completely unmodified;
`policy/decision_engine_v4.py` is a new, separate module (policy-v4).

## 1. Diagnosing Day 9 (`evaluation/diagnose_day9_fallback.py`)

Reproducing Day-9's own frozen Rs10 threshold on **both** splits (not just
test) revealed the root problem immediately:

| Split | Model-direct | Fallback | Latent value — Model B alone | Latent value — after fallback | Value lost to fallback | Verdict |
|---|---|---|---|---|---|---|
| Validation (n=59) | 9 | 50 | Rs18,258.62 | Rs18,590.40 | **−Rs331.78** | fallback **HELPS** |
| Test (n=60) | 19 | 41 | Rs20,347.65 | Rs18,748.76 | **+Rs1,598.89** | fallback **HURTS** |

Day 9's mechanism looked beneficial on validation (which is exactly why its
own search picked a nonzero threshold) and then flipped sign on test — a
59-event validation set is simply too small for a margin-only heuristic to
generalize. On the fallback-triggered test events specifically, Model B's
own pick was already the Oracle's pick 14/41 times, vs. Rule-Based's pick
only 10/41 times — Model B was, if anything, the better source to trust
more, not less, exactly when Day 9's margin rule handed control away from
it.

## 2–3. Validation-only search + the evidence-based fallback rule

Two knobs, searched together on validation only (test never touched):

- **`margin_threshold`** ∈ `{0,5,10,15,20,25,50,75,100}` — same Day-9
  concept (gap between Model B's own top-2 *net* values); still only gates
  *whether* to reconsider Model B's pick, not what to do next.
- **`fallback_mode`** — what happens once that gate fires:
  1. `always_fallback_when_below_margin` — Day-9's original rule, reimplemented for a fair comparison.
  2. `no_action_when_below_margin` — abstain entirely rather than guess.
  3. `keep_model_when_better_than_rule` — brief section 3's comparison: keep Model B unless Rule-Based's *own Model-B-predicted value* beats it, by any amount.
  4. `keep_model_unless_rule_has_clear_advantage` — same comparison, but Rule-Based must clear `fallback_advantage_threshold` (also searched over the same 9-value set) to win.

**Honest structural finding, not a bug** (verified by direct cross-check,
`tests/test_decision_engine_v4.py::test_evidence_based_modes_never_beat_models_own_global_best`):
modes 3 and 4 scored **exactly** Rs18,258.62 — zero fallbacks — across
**all 90** margin×advantage combinations tested for them. This is
mathematically guaranteed, not coincidental: Rule-Based always picks from
the same ≤5 candidate types Model B already scores in full, so Model B's
own top pick (the argmax over that entire set) can never have a *lower*
net value than Model B's own estimate for Rule-Based's candidate — there
is nothing for a "clear advantage" to ever detect. Both modes collapse to
"Model B alone" in this architecture. This was implemented exactly as the
brief specified (section 3's formula, verbatim); the collapse is a
property of comparing "the best of N" against "one specific member of that
same N," not an implementation error, and it is reported here rather than
silently patched to look more novel.

**108 configurations searched. Winner** (top 5 by total validation latent value):

| margin_threshold | fallback_mode | advantage_threshold | Total latent value (validation) | n_fallback |
|---|---|---|---|---|
| **Rs5** | **always_fallback_when_below_margin** | **—** | **Rs18,609.15** | **37** |
| Rs10 | always_fallback_when_below_margin | — | Rs18,590.40 (Day-9's own frozen config) | 50 |
| Rs0 | *(any mode — gate never fires)* | — | Rs18,258.62 (= Model B alone) | 0 |
| Rs15 | always_fallback_when_below_margin | — | Rs18,093.38 | 54 |
| Rs20–100 | always_fallback_when_below_margin | — | Rs17,997.99 (flat) | 57–58 |

**Frozen: `margin_threshold = Rs5`, `fallback_mode = always_fallback_when_below_margin`, `fallback_advantage_threshold = 0`** (unused by this mode; kept well-defined rather than stale). Given the structural finding above, the search correctly picked the best of the two modes that behave non-trivially, with a much narrower margin gate than Day-9's Rs10 (fewer, more selective fallbacks: 37 vs 50 on validation).

## 5. Costs

Unchanged from Day 9 — `policy/costs.py`'s `DEFAULT_COSTS` (`retry_cost=Rs5`, all other channel costs `Rs0` placeholders, still synthetic, still not Razorpay prices). `predicted_net_value = predicted_recovery_value − intervention_cost`, same formula throughout.

## 6. Test-set evaluation (frozen config, run once)

| Policy | Latent total (Rs) | vs. Fixed Retry | Regret vs. Oracle | Realized total (Rs) | Recovery rate |
|---|---|---|---|---|---|
| Fixed Retry | 19,114.77 | +0.00 | 3,916.87 | 21,854.10 | 80.0% |
| Rule-Based | 19,050.32 | −64.45 | 3,981.32 | 21,431.15 | 76.7% |
| **Day-8 Model B alone** | **20,347.65** | **+1,232.88** | **2,683.99** | 21,278.18 | 73.3% |
| Day-9 original fallback | 18,748.76 | −366.01 | 4,282.88 | 20,148.47 | 71.7% |
| **Day-10 improved fallback** | **19,032.25** | −82.51 | 3,999.39 | 19,997.23 | 70.0% |
| Oracle | 23,031.64 | +3,916.87 | 0.00 | 24,275.30 | 86.7% |

## The honest bottom line

Day 10 **is a real, validation-only-selected improvement over Day 9**:
+Rs283.49 latent value vs. Day-9's original fallback (Rs19,032.25 vs.
Rs18,748.76), and its regret-vs-Oracle shrank from Rs4,282.88 to
Rs3,999.39. It did this narrowly and honestly — a tighter margin gate,
searched only on validation, chosen because it scored highest there, not
because it was hand-picked to look better on test.

**But it still does not beat trusting Model B directly** on either latent
(Rs19,032.25 vs. Rs20,347.65) or realized (Rs19,997.23 vs. Rs21,278.18)
metrics on this 60-event test set. The "smarter," evidence-based modes
that were supposed to fix this (section 3's rule) turned out to be
structurally inert given this architecture (see above) — so what actually
won was still a member of the "blind fallback" family, just tuned
narrower. On a test set this small (n=60), none of these latent-value
gaps should be read as statistically significant; they are directional,
synthetic-benchmark evidence, not a proof that any one policy dominates.

## Operational metrics (test, n=60, frozen config)

- Model B direct selections: 35 (58.3%)
- Fallback count: 25 (41.7%) — down from Day-9's 68.3%
- No-action count: 0
- Average decision margin (all decisions): Rs9.75
- Decisions changed by fallback vs. Model B alone: 20/60
- `decision_source` distribution: `day8_model_b` 35, `rule_based_fallback` 25

## Decision trace (sample, `evaluation/reports/decision_engine_v4_test_set.csv` has all 60)

| Event | Model B best | Model B value | Rule pick | Rule's Model-B value | Diff | Threshold | Final | Source |
|---|---|---|---|---|---|---|---|---|
| evt_SYN000008 | plus_3_days | Rs126.31 | plus_1_day_morning | Rs109.52 | −16.79 | Rs5 | plus_3_days | day8_model_b |
| evt_SYN000019 | plus_3_days | Rs228.38 | payday_window | Rs221.48 | −6.90 | Rs5 | payday_window | rule_based_fallback |
| evt_SYN000020 | month_end_window | Rs140.94 | payday_window | Rs139.59 | −1.35 | Rs5 | payday_window | rule_based_fallback |
| evt_SYN000051 | month_end_window | Rs286.02 | payday_window | Rs285.51 | −0.51 | Rs5 | payday_window | rule_based_fallback |
| evt_SYN000056 | plus_3_days | Rs128.78 | plus_1_day_morning | Rs119.43 | −9.35 | Rs5 | plus_3_days | day8_model_b |

## Config versioning (policy-v4)

Every `PolicyDecision` row and audit-log entry now records `policy_version`,
`model_version`, `margin_threshold_used`, `fallback_advantage_threshold`,
and `fallback_strategy` — three new nullable/additive columns on
`policy_decisions` (`app/models.py`), always `NULL` on every Day 5–9 row.
Day-9 semantics are completely unchanged: `policy/decision_engine.py` was
not edited except to add three new always-`None`-for-v3 optional fields to
the shared `Decision` dataclass (verified by the full, unmodified Day-9
test suite still passing bit-for-bit).

## Reproduce

```bash
./venv/bin/python evaluation/diagnose_day9_fallback.py      # section 1: reproduces the Day-9 regression on both splits
./venv/bin/python evaluation/evaluate_decision_engine_v4.py # sections 2-7: validation search + frozen test evaluation + decision trace
./venv/bin/python -m pytest tests/ -v                        # Day 1 + 2 + ... + 10
```

## Limitations

- **Day-10 still does not beat Model B alone on this test set** — see "The
  honest bottom line" above. It is a genuine, validation-only improvement
  over Day 9, not a solved problem.
- **The evidence-based modes (3, 4) are structurally inert** given this
  system's architecture (Model B scores every valid candidate, so its own
  best pick can never be beaten by its own estimate of Rule-Based's pick).
  A different architecture — e.g., a model that scores *only* a pairwise
  comparison, or a genuinely separate confidence signal — would be needed
  for that specific mechanism to ever fire. This is left as a real
  limitation, not patched around.
- **n=60 test events is too small for statistical significance** on any of
  the latent-value gaps reported above; all figures are directional,
  synthetic-benchmark evidence only.
- **Still entirely synthetic** — every number depends on
  `data/raw/counterfactual_outcomes.csv`'s hand-designed generator.
- **Day 9 is retained, not replaced** — `policy/decision_engine.py`
  (policy-v3) is byte-for-byte unmodified in its decision logic; Day 10
  adds a parallel policy-v4 engine.

## Day-10 acceptance checklist

- [x] Day-9 failure diagnosed (fallback helped on validation, hurt on test — root-caused to Model B being right more often than Rule-Based on exactly the ambiguous-margin events, and a validation set too small to generalize a margin-only heuristic)
- [x] No new ML model (Day-8 Model B reused as-is; only the fallback/abstention logic around it changed)
- [x] Validation-only parameter search (108 configurations: 9 margin thresholds × 4 fallback modes, plus 9 advantage thresholds for mode 4; test set never touched during selection)
- [x] Improved fallback logic (+Rs283.49 latent value vs. Day-9's original fallback on the frozen test set — honestly reported as an improvement over Day 9, not a solved problem, since it still trails Model B alone)
- [x] Policy version `policy-v4` recorded on every decision, alongside `margin_threshold_used` / `fallback_advantage_threshold` / `fallback_strategy`
- [x] Model-vs-rule comparison implemented (`fallback_advantage()` in `policy/decision_engine_v4.py`, unit-tested independently of the full flow since the full flow can never exercise "rule wins" — see structural finding)
- [x] Existing guardrails preserved (`retryable_soft` gate, max attempts, duplicate prevention, candidate horizon — all re-tested against policy-v4)
- [x] Full auditability (every decision — including `NO_ACTION` and fallback — gets an audit_log row with `decision_source` / `margin_threshold` / `fallback_advantage_threshold` / `fallback_strategy`; no secrets logged)
- [x] Test-set evaluated only after freezing configuration (search results frozen into `policy/decision_engine_v4.py`'s `DEFAULT_*` constants before Phase 2 ran; a `frozen_matches` check in `evaluation/evaluate_decision_engine_v4.py` verifies this on every run)
- [x] Full test suite passes (280/280: 250 Day 1–9 + 30 Day 10)
- [x] Results honestly reported (Day-10 improves on Day-9 but does not beat Model B alone; the "smarter" evidence-based modes were found to be structurally inert and this is reported prominently, not hidden)

# Day 11 — LLM-Assisted Recovery Communication Layer

**"The LLM is not the payment decision-maker."**

Everything through Day 10 is deterministic or model-driven: classification
(Day 2), Model B's value prediction (Day 8), and the cost-aware
decision-margin/fallback engine (Days 9–10) all run without ever calling an
LLM. Day 11 adds language tasks **strictly downstream** of that already-final
decision:

```
failure_event → classification → Day-8 Model B / policy-v4 → selected recovery action
                                                                        ↓
                                                          LLM communication task
                                                                        ↓
                                                       structured communication output
                                                                        ↓
                                                                   audit log
```

The LLM never sees an open decision to make — `policy/decision_engine_v4.py`
(Day 10, completely unmodified today) has already picked
`selected_candidate_type` / `NO_ACTION` by the time any LLM function is
called. None of the three job functions in `llm/service.py` accept the
policy layer as an input they could influence — they take its *output* as
plain, already-final data (`will_retry: bool`, `retry_window_description: str
| None`). `tests/test_llm.py::TestPolicyIndependenceFromLLM` proves this by
computing the same decision twice, with an all-jobs LLM outage forced in
between, and asserting the two decisions are bit-for-bit identical.

## The three LLM jobs (verbatim from the project specification)

1. **Generate outreach microcopy** per (failure bucket × customer segment ×
   language preference), including a **Hinglish variant as a prompt
   parameter**, not new infrastructure — `language` is just one more value
   (`"en"` / `"hi"` / `"hinglish"`) passed into the same function and the
   same prompt template; there is no separate Hinglish code path.
2. **Parse a free-text customer reply** (e.g. *"I'll pay Friday when salary
   comes"*) into a structured promise-to-pay object: `{date, confidence,
   channel}`.
3. **Generate the plain-English batch-level explanation** in the final
   report.

These are the only three jobs. No LLM function classifies a failure,
selects retry timing, makes a compliance decision, overrides policy, or
decides whether a payment should be retried — all five remain exactly
where Days 2–10 put them.

## Architecture (`llm/`)

```
llm/
    client.py    # MockLLMClient (offline, default) + AnthropicLLMClient, selected by LLM_PROVIDER
    prompts.py   # versioned system/user prompts per job + the mock response router
    schemas.py   # Pydantic output schemas + the LLMResult envelope
    service.py   # orchestration: build prompt → call client → validate → fallback → audit
```

No LangChain, no LangGraph, no agent framework, no autonomous tool use — a
provider interface with one method (`complete(system_prompt, user_prompt)
-> str`) and a validate/fallback wrapper around it.

## Structured outputs

Every job returns an `LLMResult` (`llm/schemas.py`):

```python
class LLMResult(BaseModel):
    task_name: Literal["outreach_microcopy", "promise_to_pay_parse", "batch_explanation"]
    model_name: str
    prompt_version: str
    provider: Literal["mock", "anthropic"]
    success: bool
    structured_result: dict | None   # validated schema's .model_dump(), or the deterministic fallback
    error_type: str | None
    created_at: datetime
```

Job-specific schemas (`extra="forbid"` on all three — an unexpected extra
field is a validation failure, not silently ignored):

| Job | Schema | Allowed values |
|---|---|---|
| 1 | `OutreachMicrocopyOutput{message_text, language, failure_bucket, customer_segment}` | `language ∈ {en, hi, hinglish}` |
| 2 | `PromiseToPayOutput{date, confidence, channel}` | `date`: ISO-8601 or `null`; `confidence ∈ [0.0, 1.0]`; `channel ∈ {credit_card, debit_card, upi_autopay, netbanking, unspecified}` (reuses the project's existing `INSTRUMENTS` vocabulary from `data/generate_synthetic_dataset.py`, not a new one) |
| 3 | `BatchExplanationOutput{explanation_text}` | 1–3000 chars |

## Prompt versioning

`llm/prompts.py` defines one frozen version constant per job
(`outreach_microcopy_v1`, `promise_to_pay_parse_v1`, `batch_explanation_v1`).
Every system prompt states, verbatim, the same hard-rule block: don't invent
payment facts, don't invent customer information, only use supplied
context, never claim a payment succeeded/failed/retried unless the data
says so, never make a retryability/timing/compliance decision, return
structured JSON only. A prompt's wording is never edited in place —
changing it means bumping the version constant, so an old `LLMResult`'s
recorded `prompt_version` always identifies exactly the text that produced it.

## Mock provider (offline development, brief section 6)

`LLM_PROVIDER=mock` (the default, and the only provider the test suite and
`scripts/run_llm_demo.py` use) makes **zero network calls** — verified
directly by `tests/test_llm.py::test_mock_provider_makes_no_network_call`,
which monkeypatches `socket.create_connection` / `socket.socket.connect` to
raise if called at all, then confirms a mock-provider call still succeeds.
`llm/prompts.py::mock_response_for_prompt` deterministically routes on a
marker embedded in the user prompt and returns schema-valid JSON derived
from the actual input (e.g. it resolves "Friday" against the supplied
`today` date, and extracts a mentioned payment channel) — same input always
produces the same output.

## Failure handling (brief section 5)

`llm/service.py::_invoke` is the single chokepoint every job's call goes
through. It catches, and turns into a `success=False` `LLMResult` carrying
a deterministic fallback (never an exception escaping to the caller):

| Failure mode | `error_type` | Fallback |
|---|---|---|
| Provider unavailable / misconfigured | `provider_error:...` | job-specific safe default |
| Timeout / SDK exception | `provider_error:...` | job-specific safe default |
| Empty response | `provider_error:empty_response` | job-specific safe default |
| Invalid JSON | `invalid_json` | job-specific safe default |
| Schema-invalid JSON | `schema_validation_error` | job-specific safe default |
| Any other unexpected exception | `unexpected_error:<ExceptionType>` | job-specific safe default |

Fallbacks never fabricate: Job 1 falls back to a short, neutral,
already-fact-checked-against-`will_retry` message (never claims a retry
that isn't actually happening); Job 2 falls back to `{date: null,
confidence: 0.0, channel: "unspecified"}` (never guesses); Job 3 falls back
to a short note pointing at the raw report data instead of fabricated prose.

## Audit logging (brief section 8)

Every invocation writes **two** rows, mirroring the pattern every prior day
established: a dedicated `llm_invocations` row (`app/models.py`, new
table — `event_id`/`batch_id`, `task_name`, `model_name`, `prompt_version`,
`provider`, `success`, `structured_output`, `error_type`, `created_at`) and
an `audit_log` row (`actor="llm"` — a value the `AuditLog.actor` docstring
had already reserved since Day 1). Neither ever stores an API key, webhook
secret, or raw auth header — verified by
`tests/test_llm.py::TestNoSecretsInLogs`, which forces a fake Anthropic SDK
exception whose message embeds a fake secret and asserts the secret never
appears in logs, the audit row, or the invocation row.

## Context boundaries (brief section 7)

Jobs 1 and 2 take only individually-named parameters (`failure_bucket`,
`customer_segment`, `language`, `will_retry`, `retry_window_description`,
`amount_rupees`, `customer_reply_text`) — never a raw context dict or
dataframe row — so a hidden synthetic field (`archetype`,
`recovery_probability_latent`, `expected_recovery_value_latent`, or a
counterfactual-outcome column) cannot reach them; there is no dict for one
to hide inside (`tests/test_llm.py` asserts this structurally via
`inspect.signature`). Job 3's `report_summary` is an
already-computed, already-labeled-`SYNTHETIC COUNTERFACTUAL EVALUATION`
aggregate (e.g. Day-10's `decision_engine_v4_evaluation.json`) — a
different category from the per-event hidden fields, and still passed
through `sanitize_context()` as defense-in-depth, which recursively strips
`FORBIDDEN_CONTEXT_KEYS` and logs if anything was stripped.

## Demo

```bash
LLM_PROVIDER=mock ./venv/bin/python scripts/run_llm_demo.py
```

Runs a synthetic failure event through classification → Day-8 Model B /
policy-v4 → all 3 LLM jobs (including a forced-failure example for Job 2)
→ prints every structured output, which provider mode was used, and every
`llm_invocations`/`audit_log` row written, using a throwaway in-memory
database (never touches `data/recovery_agent.db`).

## Environment

```bash
# .env.example (Day 11 additions)
LLM_PROVIDER=mock
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-5
```

`ANTHROPIC_API_KEY` stays optional — `LLM_PROVIDER=mock` (the default) never
reads it. Setting `LLM_PROVIDER=anthropic` without a key logs a warning and
safely falls back to the mock provider rather than crashing.

## Reproduce

```bash
LLM_PROVIDER=mock ./venv/bin/python scripts/run_llm_demo.py
./venv/bin/python -m pytest tests/ -v   # Day 1 + 2 + ... + 11
```

## Limitations

- **Mock-mode outreach copy is templated, not generative** — realistic
  enough to validate the schema/pipeline end-to-end, but not representative
  of real Claude output quality. `LLM_PROVIDER=anthropic` is needed to see
  genuinely generated microcopy.
- **No real customer replies exist in this project** — Job 2 is exercised
  with hand-written example text (this README, the demo script, and
  `tests/test_llm.py`); there is no live WhatsApp/SMS inbound channel.
- **`language`/`customer_segment` are not persisted customer attributes** —
  no Day 1–10 table has a language-preference or segment column; both are
  passed as plain function arguments, matching the brief's explicit
  "Hinglish as a prompt parameter, not new infrastructure" instruction.
- **Job 3's fallback is intentionally terse** — it never attempts to
  paraphrase report numbers itself (that would risk fabricating a claim
  about synthetic data being real), so a failed Job-3 call produces a
  pointer to the raw JSON, not a prose substitute.
- **Still entirely synthetic downstream of policy** — nothing in Day 11
  changes any Day 1–10 number; this section only adds a communication layer
  on top of decisions Days 2–10 already made.

## Day-11 acceptance checklist

- [x] Exact three LLM jobs from project specification implemented (outreach microcopy, promise-to-pay parsing, batch-level explanation — no others)
- [x] Structured outputs (Pydantic, `extra="forbid"`, validation failure handled explicitly as a distinct `error_type`)
- [x] Versioned prompts (`outreach_microcopy_v1` / `promise_to_pay_parse_v1` / `batch_explanation_v1`, frozen, never edited in place)
- [x] Anthropic integration isolated (`llm/client.py::AnthropicLLMClient`, lazy `import anthropic`, only place the SDK is touched)
- [x] Mock provider works without API key (`LLM_PROVIDER=mock` default; zero network calls, verified by a monkeypatched-socket test)
- [x] LLM failures fail safely (6 distinct failure modes tested per job — provider unavailable, timeout/SDK error, empty response, invalid JSON, schema-invalid JSON, unexpected exception — all resolve to a deterministic, non-fabricating fallback)
- [x] No policy decisions made by LLM (`decide_engine_v4` has no LLM-related parameter; a dedicated test proves an all-jobs LLM outage produces a bit-for-bit identical policy decision)
- [x] Synthetic hidden variables excluded (Jobs 1/2 structurally can't receive them — no dict parameter exists to carry one; Job 3's aggregate report is defense-in-depth sanitized)
- [x] Audit logging implemented (`llm_invocations` table + `audit_log` rows, actor="llm", every invocation including failures; no secrets, verified by a forced-secret-leak test)
- [x] Full test suite passes (320/320: 280 Day 1–10 + 40 Day 11)
- [x] Demo works offline in mock mode (`scripts/run_llm_demo.py`, no `ANTHROPIC_API_KEY` required)

# Day 12 — Compliance + End-to-End Orchestration

**"Compliance checks in this prototype are deterministic project guardrails. They are not presented as a complete legal/regulatory compliance implementation."**

Day 12 adds no new decision-making — no new ML model, no change to Day-8
Model B, no change to Day-10's policy objective, no change to what the LLM
is allowed to do. It wires everything built so far into one deterministic
flow, with a new gate inserted between "what should we do" and "do it":

```
failure_event → classification → policy-v4 decision → COMPLIANCE GATE → payment action → LLM communication → audit trail
```

## Architecture

```
policy/compliance.py      # the compliance gate (new)
recovery/
    __init__.py
    schemas.py             # RecoveryExecutionResult + FinalStatus precedence
    orchestrator.py         # orchestrate_recovery() -- the single entry point
```

`policy/compliance.py` sits alongside `policy/guardrails.py` (Day 5)
deliberately — it's the same kind of thing (deterministic, explainable, no
ML), just a second, independent checkpoint that re-validates an
already-made decision before it's allowed to execute, rather than
protecting policy's own candidate selection the way `guardrails.py` does.

## The compliance gate

```python
@dataclass(frozen=True)
class ComplianceResult:
    payment_action_allowed: bool
    payment_reason: str
    communication_action_allowed: bool
    communication_reason: str
    rule_version: str          # "compliance-v1"
    # .allowed / .reason alias payment_action_allowed / payment_reason --
    # the brief's minimal {allowed, reason, rule_version} shape
```

Rules evaluated (brief section 2's minimum list — every one is either a
guardrail this project already enforces elsewhere and reuses here, or a
new, clearly-labeled **project** guardrail, never an invented legal
requirement):

| Rule | Source | Gates |
|---|---|---|
| Classification is `retryable_soft` | reused: `policy/guardrails.py::is_classification_allowed` | payment |
| Action is not `NO_ACTION` | new, project-defined | payment |
| Candidate retry time is valid (after failure, within horizon) | reused constant: `MAX_CANDIDATE_HORIZON_DAYS` | payment |
| Retry count not exhausted | reused: `MAX_RETRY_ATTEMPTS` | payment |
| Duplicate action blocked | new — mirrors Day 5–10's `event_id` idempotency, applied independently at this layer | payment (and separately, communication) |
| Customer cancellation/opt-out blocks outreach | new — `customer_cancelled` bucket is an automatic opt-out signal; an explicit `customer_opted_out` flag also exists, independent of classification | communication |
| Consent for communication exists | new, project **placeholder** (no real consent-tracking system exists yet — defaults to allowed, documented as such) | communication |
| Required fields present | new | both |

## Action separation

Payment and communication are gated **independently** (brief section 3).
The canonical example — a `retryable_soft` customer who separately opted
out of messaging:

```
payment_action_allowed      = True   (payment_reason: "payment_action_allowed: all compliance checks passed")
communication_action_allowed = False  (communication_reason: "customer_opted_out_or_cancelled: outreach blocked")
```

Payment proceeds; communication does not. The reverse is equally
representable (e.g. max-attempts-exhausted blocks payment while
communication — "please contact support" — could still be allowed).

## Orchestrator flow (`recovery/orchestrator.py::orchestrate_recovery`)

1. **Classify** — `classification/rules.py::classify` (Day 2, reused as-is).
2. **Policy decision** — `policy/decision_engine_v4.py::decide_for_failure_event_engine_v4` (Day 10, reused as-is, already idempotent and self-auditing).
3. **Compliance gate** — `policy/compliance.py::evaluate_compliance` (new).
4. **Payment action** — recorded (`retry_scheduled` / `blocked` / `no_action`) — **no live Razorpay call is ever made**, this day or any prior one.
5. **LLM communication** — `llm/service.py::generate_outreach_microcopy_and_log` (Day 11, reused as-is) — called **only if** compliance allows it.
6. **Final status + audit** — one `RecoveryExecutionResult`, five `audit_log` rows.

The orchestrator contains no classification, scoring, compliance, or
prompt logic of its own — it only sequences calls to modules that already
exist and assembles their outputs.

## Structured final result

```python
@dataclass(frozen=True)
class RecoveryExecutionResult:
    event_id: int
    subscription_id: str
    classification_bucket: str
    policy_version: str
    selected_candidate_type: str
    selected_candidate_datetime: datetime | None
    compliance_allowed: bool
    compliance_reason: str
    payment_action: Literal["retry_scheduled", "blocked", "no_action"]
    communication_action: Literal["sent", "fallback_used", "blocked", "skipped"]
    llm_task_name: str | None
    llm_success: bool | None
    final_status: FinalStatus   # 7 values, deterministic precedence -- see recovery/schemas.py
    created_at: datetime
```

`final_status` precedence (most specific/urgent first — full detail is
still preserved in the other fields regardless of which single status
wins): `NO_ACTION` → `RETRY_BLOCKED` → `COMMUNICATION_BLOCKED` →
`POLICY_FALLBACK` → `LLM_FALLBACK` → `COMMUNICATION_ALLOWED` →
`RETRY_ALLOWED`. A BLOCKED outcome always outranks a "proceeded via
fallback" flag — see `recovery/schemas.py`'s docstring for the full
reasoning, including a real bug this precedence caught during development
(a duplicate-event retry that was correctly blocked by compliance was
initially still reporting `POLICY_FALLBACK` instead of `RETRY_BLOCKED`,
because the original ordering let the fallback flag outrank the block —
fixed before any test was written to assert the wrong behavior).

## LLM placement — and what happens when it fails

The LLM call is the **last** step, gated by compliance, and only ever
writes to `communication_action` / `llm_task_name` / `llm_success`. It
never runs before the policy decision is persisted (`policy_row` exists in
the database before `generate_outreach_microcopy_and_log` is ever called),
and nothing reads an LLM result back into `payment_action`,
`selected_candidate_type`, or `compliance_allowed`.

On LLM failure (unavailable, timeout, malformed/schema-invalid JSON — any
of Day 11's failure modes): the payment decision is **completely
unaffected**, `communication_action` becomes `"fallback_used"`, and
`llm_success=False` is recorded. `tests/test_orchestrator.py`'s scenarios
I and J prove this directly — the same event processed with a working vs.
a broken LLM client produces identical `selected_candidate_type` /
`compliance_allowed` / `payment_action`.

## Audit trail

Every orchestration call writes exactly 5 `audit_log` rows (one per
stage), with explicit actor values:

| Actor | Action | Written by |
|---|---|---|
| `classifier` | `orchestrator_classification` | orchestrator |
| `policy` | `policy_decision_made` / `policy_no_action` | Day-9/10's own persistence code, reused |
| `compliance` | `orchestrator_compliance` | orchestrator |
| `llm` | `llm_outreach_microcopy_succeeded` / `..._failed_used_fallback` | Day-11's own persistence code, reused (skipped if communication wasn't attempted) |
| `orchestrator` | `orchestrator_final_status` | orchestrator |

Never stores an API key, webhook secret, or raw auth header — verified by
`tests/test_orchestrator.py::test_no_secrets_in_audit_trail`.

## Demo

```bash
./venv/bin/python scripts/run_end_to_end_demo.py
```

Runs 4 scenarios against the real Day-10 trained model, printing the full
`EVENT → CLASSIFICATION → POLICY DECISION → COMPLIANCE → PAYMENT ACTION →
LLM COMMUNICATION → FINAL RESULT → AUDIT TRAIL` pipeline for each: (1) a
successful recovery flow, (2) a blocked flow (`hard_decline`), (2b) payment
allowed with communication blocked (opt-out — the section-3 example made
concrete), and (3) an LLM failure flow (forced provider outage) showing
`final_status=LLM_FALLBACK` with the payment decision fully intact. Uses a
throwaway in-memory database; never touches `data/recovery_agent.db`, and
no real payment retry or message send is ever attempted.

## Reproduce

```bash
./venv/bin/python scripts/run_end_to_end_demo.py
./venv/bin/python -m pytest tests/ -v   # Day 1 + 2 + ... + 12
```

## Limitations

- **Compliance rules here are project guardrails, not legal compliance** —
  see the required wording at the top of this section. Nothing claims to
  satisfy DPDP/TRAI/RBI or any other real regulatory regime.
- **`consent_for_communication` is a placeholder** — this project has no
  real consent-tracking system; the field defaults to allowed and exists
  so a future day can wire in a real one without changing the gate's shape.
- **No live payment or communication integration** — `payment_action` is
  a recorded intent, never an actual Razorpay retry call; communication is
  the mock (or, if configured, real Anthropic) LLM only — no live
  WhatsApp/SMS/voice channel exists.
- **Some compliance checks are structurally unreachable via the full
  orchestrator flow today** (candidate-time validity, max-attempts) because
  policy's own guardrails already prevent those states from reaching
  compliance — they're real, tested rules (directly, at the compliance
  layer — see `tests/test_compliance.py`), kept as defense-in-depth for a
  future policy engine that might not enforce them itself.
- **Still entirely synthetic downstream of policy** — nothing in Day 12
  changes any Day 1–11 number; this section only adds a compliance gate
  and orchestration layer on top of decisions already made.

## Day-12 acceptance checklist

- [x] Deterministic compliance gate (`policy/compliance.py`, no ML, no LLM, explainable structured result)
- [x] Payment and communication actions separated (independently gated; opt-out example demonstrates payment-allowed/communication-blocked)
- [x] Orchestrator implemented (`recovery/orchestrator.py`, no duplicated decision logic — every stage reuses an existing module)
- [x] Day-10 policy reused (`decide_for_failure_event_engine_v4`, unmodified)
- [x] Day-11 LLM reused (`generate_outreach_microcopy_and_log`, unmodified)
- [x] LLM cannot override policy (policy persisted before any LLM call; a dedicated test asserts this ordering; failure-mode tests I/J prove payment decisions are unaffected)
- [x] Failures handled safely (11-scenario failure matrix A–K, all tested)
- [x] Full audit trail (5 actor-labeled rows per call; no secrets, verified)
- [x] End-to-end offline demo (`scripts/run_end_to_end_demo.py`, real trained model, mock LLM, no network)
- [x] Full test suite passes (358/358: 320 Day 1–11 + 38 Day 12)
- [x] No new ML model
- [x] No live payment/communication integration yet

# Day 13 — Dashboard + Final Demo

**"No live payment retry or customer messaging is executed by this prototype."**

Day 13 is UI polish and presentation only — no new decision logic. It adds
one thing: a dashboard that sits **on top of** everything Days 1–12 already
built, reusing every layer exactly as it was, with a professional,
original, Razorpay-inspired merchant-console visual language.

## UI architecture

```
ui/
    __init__.py
    styles.py        # color tokens + injected CSS (light fintech SaaS: white cards, charcoal/navy text, restrained indigo accent)
    data.py           # data loaders -- SYNTHETIC BENCHMARK (frozen evaluation/reports/*.json) vs. OPERATIONAL DEMO DATA (live orchestrator run)
    components.py     # reusable render helpers: KPI cards, status badges, timeline steps, empty states
    app.py             # the Streamlit entry point -- 7-page sidebar-navigated app
scripts/
    run_dashboard.sh   # ./scripts/run_dashboard.sh
```

No page re-implements classification, scoring, compliance, or LLM logic.
Every page either (a) reads a frozen evaluation report Days 6–10 already
wrote, or (b) calls `recovery/orchestrator.py::orchestrate_recovery`
(Day 12, byte-for-byte unmodified) to populate a throwaway in-memory
database with real orchestration output. Nothing in `ui/` imports or
duplicates any decision-making code from `policy/`, `classification/`, or
`llm/` beyond calling their existing public functions.

## Start the dashboard

```bash
./scripts/run_dashboard.sh
# or directly:
./venv/bin/streamlit run ui/app.py
```

Runs entirely offline: `LLM_PROVIDER=mock` (the project default — see
`.env.example`), no `ANTHROPIC_API_KEY` needed, no live Razorpay call ever
attempted. Verified end-to-end this session: the server starts cleanly
(`streamlit run` → `HTTP 200` on every route), and every one of the 7 pages
plus every interactive control (row selection, the promise-to-pay parser,
and all 4 demo scenarios) was executed via
`streamlit.testing.v1.AppTest` — Streamlit's real script-execution test
harness, not a mock — with zero exceptions.

## Pages

| Page | What it shows | Source |
|---|---|---|
| **Overview** | 6 KPI cards (each individually tagged Synthetic benchmark / Operational demo data — never mixed) + the latent-vs-realized money-recovery chart pair | both |
| **Recovery Queue** | Merchant-style table of every orchestrated event; select a row to open its full decision timeline + retry-candidate scoring inline | operational |
| **Payment Events** | Filterable browser over the raw generated `failure_events.csv` / `subscriptions.csv` dataset | synthetic |
| **Analytics** | 7 charts: recovery value & rate by policy, candidate selection distribution, classification distribution, compliance outcomes, LLM success/fallback rate, audit volume by actor | mixed, each tagged |
| **Communications** | The 3 Day-11 LLM jobs: generated outreach microcopy (English/Hindi/Hinglish), an interactive promise-to-pay parser, and the batch-level plain-English report explanation | operational + synthetic |
| **Audit Log** | Filterable (actor / status) table of every `audit_log` row the demo run produced | operational |
| **System / Demo** | Live environment/version panel (LLM provider, `policy-v4`, `compliance-v1`, Day-8 Model B) + dynamic test count + the interactive 4-scenario demo runner | operational |

## Data sources

- **Synthetic benchmark**: `data/raw/*.csv` (raw generated dataset) and
  `evaluation/reports/*.json` (Days 6–10's frozen offline evaluations) —
  read-only, never recomputed by the UI.
- **Operational demo data**: a live call to `recovery/orchestrator.py`
  over a 60-event sample of the real generated `failure_events.csv`
  (joined with `subscriptions.csv` for plan tier / instrument / city
  tier), using the real trained Day-8 Model B artifact and the Day-11 mock
  LLM. Cached per dashboard session (`st.cache_resource`) so navigating
  pages doesn't re-run it.
- Every KPI, chart, and table is explicitly tagged with which of the two
  it is — see the "Synthetic benchmark" / "Operational demo data" pill on
  every section.

## Demo scenarios

The System / Demo page's "Run recovery" control calls
`recovery/orchestrator.py::orchestrate_recovery` directly (no
reimplemented logic) for 4 predefined scenarios: a recoverable
insufficient-fund payment, a hard decline, a customer opt-out (payment
allowed, communication blocked), and a forced LLM outage (payment
unaffected, deterministic fallback). Each prints the full
Classification → Policy → Compliance → Communication → Final result
pipeline.

## Offline / mock mode

The dashboard never requires network access. `ui/data.py::build_demo_database`
falls back to policy's own model-unavailable path (still a valid, audited
`NO_ACTION`/fallback outcome — never a crash) if the Day-8 model artifact
is missing, and every loader in `ui/data.py` returns `None` / an empty
`DataFrame` — never raises — for a missing evaluation report or empty
database, rendering a clear "No … available" empty state instead.

## Synthetic benchmark labeling

Every chart/table sourced from `evaluation/reports/*.json` carries a
visible "Synthetic benchmark" tag and, on the money-recovery charts, the
same latent-vs-realized separation and caveat established in the Day 9
README section — nothing here is presented as measured Razorpay production
performance.

## A dependency note (environment, not logic)

Getting Streamlit into this environment required care: recent Streamlit
releases (1.57+) unconditionally import an optional starlette-based ASGI
server at package-import time, needing `starlette>=0.46` — directly
incompatible with `fastapi==0.115.6`'s `starlette<0.42.0` pin from Day 1.
`streamlit==1.56.0` is the newest release where that integration is still
an opt-in extra; it was verified compatible (`pip check` clean, full Day
1–12 test suite unaffected) before being pinned in `requirements.txt`. No
Day 1–12 dependency was upgraded or changed to make this work.

## Reproduce

```bash
./scripts/run_dashboard.sh
./venv/bin/python -m pytest tests/ -v   # Day 1 + 2 + ... + 13
```

## Limitations

- **The Recovery Queue's "real data" is entirely `insufficient_fund`** —
  every row in `data/raw/failure_events.csv` has that error_reason (the
  synthetic generator's own scope, Day 3), so hard_decline/
  customer_cancelled/unmapped variety is only visible via the Interactive
  Demo scenarios, not the live queue. This is honest, not a bug — the
  brief says "do not invent fake dashboard numbers," and the real
  generated data simply doesn't contain those cases.
- **The dynamic test count is a static scan of `def test_*` functions**
  (336), not a live pytest collection count — parametrized tests expand to
  more cases at collection time (358 actually run). The "Run full test
  suite now" button on System / Demo runs the real suite on demand for an
  exact, live number.
- **The retry-candidate breakdown re-runs `decide_engine_v4` read-only**
  for visualization — the exact same computation that produced the live
  decision, not a new one, but a second call rather than a cached value.
- **Still no live integration** — no real Razorpay retry, no real
  WhatsApp/SMS/voice send. This dashboard visualizes decisions the system
  already makes deterministically; it does not execute them.

## Final MVP

```
Razorpay Test Mode webhook
    → Classification              (Day 2, deterministic)
    → Recovery Model / Policy     (Day 8 Model B + Day 10 policy-v4)
    → Compliance                  (Day 12, deterministic project guardrails)
    → Recovery Action             (recorded, never executed live)
    → LLM Communication           (Day 11, 3 jobs, downstream of policy — never decides)
    → Audit Trail                 (every stage, every actor, no secrets)
    → Dashboard                   (Day 13, read-only visualization on top)
```

**No live payment retry or customer messaging is executed by this
prototype.** Every number that looks like a real outcome is explicitly
labeled synthetic-benchmark or operational-demo-data; nothing here has
ever touched a real Razorpay production account, a real customer, or a
real message-sending API.

## Day-13 acceptance checklist

- [x] Razorpay-inspired professional merchant-dashboard UI (light fintech SaaS, white cards, charcoal/navy text, restrained indigo accent, subtle borders, compact tables)
- [x] Original design, not a clone (no Razorpay logos, layouts, or proprietary assets reproduced)
- [x] Overview dashboard (6 individually-tagged KPIs + latent/realized money-recovery chart pair)
- [x] Recovery Queue (merchant-style table, selectable rows)
- [x] Event detail/timeline (8-step timeline, reasoning fields only — nothing invented)
- [x] Candidate scoring view (all 5 candidates, predicted value/cost/net, model/rule/oracle picks, synthetic-benchmark tagged)
- [x] Analytics (7 charts, each source-tagged)
- [x] Communications (all 3 Day-11 jobs; English/Hindi/Hinglish shown)
- [x] Audit Log (filterable by actor/status, no secrets — verified)
- [x] Demo scenario runner (4 scenarios, calls `recovery/orchestrator.py` directly)
- [x] Uses existing orchestrator/policy/LLM layers (zero reimplemented decision logic anywhere in `ui/`)
- [x] Synthetic results clearly labeled (every chart/table/KPI individually tagged)
- [x] Offline/mock mode (verified: no network calls, `LLM_PROVIDER=mock`)
- [x] No live payment actions (payment_action is always a recorded intent)
- [x] Dashboard launches successfully (verified: clean `streamlit run` startup, `HTTP 200`, all 7 pages + all interactive controls exception-free under `AppTest`)
- [x] Full test suite passes (393/393: 358 Day 1–12 + 35 Day 13)
- [x] README updated
