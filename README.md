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
