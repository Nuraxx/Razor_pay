"""
Live feature-enrichment boundary (BUG-4, pre-submission audit fix).

Model B (`policy/decision_engine.py::EVENT_FEATURE_KEYS`) was originally
trained on a synthetic dataset that hands it 12 per-event features. A real
Razorpay `payment.failed` webhook does not carry those 12 fields -- most of
them (`bank_network_conditions`, `issuing_bank_downtime_flag`,
`network_latency_bucket`, `is_month_end_settlement_rush`, `plan_tier`,
`city_tier`) are, on inspection, simulation-only constructs with no live
counterpart Razorpay (or any other integration this project actually has)
exposes -- see FEATURE_SOURCES below for the full, verified breakdown.

This module is the boundary between an already-stored, already-verified raw
webhook event (plus this project's own durable event history) and whatever
SUBSET of those 12 features can be honestly, authoritatively assembled live.
It never fabricates a value for a feature it cannot source, never copies a
synthetic-dataset value into a live feature vector, and never calls the
random data-generating functions in data/generate_synthetic_dataset.py (see
tests/test_live_feature_enrichment.py's leakage-guard tests). The one pure,
deterministic calendar-math helper it DOES reuse from that module
(`days_to_nearest_payday_window`) draws no randomness and touches no
simulated per-customer data -- it is the exact same helper
policy/retry_candidates.py already uses on the live path today to schedule
real retry timing, so a live event's "payday window" and a live candidate's
"payday window" mean the same calendar-math thing.

FEATURE_SOURCES -- one of five values per key, verified against this
project's actual code/integrations (never asserted without a concrete
source):

    WEBHOOK_NATIVE                 -- present directly in the Razorpay
                                       webhook payload this project already
                                       stores.
    DERIVED_FROM_AUTHORITATIVE_DATA -- computed deterministically from data
                                       this project already durably stores
                                       (the failure timestamp, or this
                                       project's own prior RawEvent history
                                       for the subscription) -- no external
                                       call, no randomness.
    RAZORPAY_API_ENRICHED          -- obtainable via an authenticated call
                                       to Razorpay's own REST API, using the
                                       existing RAZORPAY_KEY_ID/
                                       RAZORPAY_KEY_SECRET credentials.
                                       Optional (LIVE_FEATURE_ENRICHMENT_ENABLED),
                                       off by default, and never crashes the
                                       webhook on failure.
    MERCHANT_PROFILE               -- would require a merchant-supplied
                                       catalog (e.g. plan_id -> business
                                       tier) this project does not have any
                                       integration for today. No such
                                       catalog exists in this codebase, so
                                       these stay UNAVAILABLE in practice --
                                       listed separately from UNAVAILABLE
                                       only because a real path to
                                       availability plausibly exists if a
                                       merchant profile were ever added.
    UNAVAILABLE                    -- no real, verifiable source exists at
                                       all (live, via any API, or via a
                                       hypothetical merchant profile) --
                                       these are simulation-only constructs
                                       in the synthetic dataset generator.

    day_of_month                    -> DERIVED_FROM_AUTHORITATIVE_DATA
    days_to_nearest_payday_window   -> DERIVED_FROM_AUTHORITATIVE_DATA
    prior_if_failure_count          -> DERIVED_FROM_AUTHORITATIVE_DATA
    primary_instrument              -> WEBHOOK_NATIVE
    tenure_days                     -> RAZORPAY_API_ENRICHED (optional, best-effort)
    plan_tier                       -> MERCHANT_PROFILE (not integrated -- unavailable today)
    city_tier                       -> MERCHANT_PROFILE (not integrated -- unavailable today)
    prior_if_self_resolved_rate     -> UNAVAILABLE (see note below)
    is_month_end_settlement_rush    -> UNAVAILABLE (see note below)
    bank_network_conditions         -> UNAVAILABLE
    issuing_bank_downtime_flag      -> UNAVAILABLE
    network_latency_bucket          -> UNAVAILABLE

`prior_if_self_resolved_rate` note: the synthetic generator's version is a
per-entity draw from a hidden "archetype" that this project has no live
analog for. This system also never executes a real retry itself (every
payment_action is "recorded only, never executed" -- see
recovery/orchestrator.py) -- so it has no reliable way to attribute a later
RECOVERED outcome to "the customer/bank resolved it on their own" vs. "our
own communication nudge happened to precede it," making even an inferred
proxy dishonest to assert with confidence. Left UNAVAILABLE rather than
guessed at.

`is_month_end_settlement_rush` note: in the generator, this is a per-event
COIN FLIP (`bool(rng.random() < rush_p)`) only WEIGHTED by day-of-month, not
a deterministic fact of the date -- computing "is it near month end" on the
live path and treating that as this feature would silently reintroduce the
generator's own randomness by another name. Left UNAVAILABLE.

COMPLETENESS: because 5-7 of the 12 keys are always UNAVAILABLE today, a
genuine live subscription webhook can NEVER produce a complete Model B
feature vector, with or without LIVE_FEATURE_ENRICHMENT_ENABLED -- Model B
is architecturally never invoked for real live traffic, and the rule-based
policy tier (policy/baselines.py::rule_based_baseline, via
policy/decision_engine_v4.py's fallback chain) is what actually decides
every live subscription recovery today. This is intentional and honestly
documented (README §4a), not a bug to work around by inventing values.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy.orm import Session

from app.config import settings
from app.logging_config import log
from app.models import RawEvent
from data.generate_synthetic_dataset import days_to_nearest_payday_window
from policy.decision_engine import EVENT_FEATURE_KEYS

FEATURE_SOURCES: dict[str, str] = {
    "day_of_month": "DERIVED_FROM_AUTHORITATIVE_DATA",
    "days_to_nearest_payday_window": "DERIVED_FROM_AUTHORITATIVE_DATA",
    "prior_if_failure_count": "DERIVED_FROM_AUTHORITATIVE_DATA",
    "primary_instrument": "WEBHOOK_NATIVE",
    "tenure_days": "RAZORPAY_API_ENRICHED",
    "plan_tier": "MERCHANT_PROFILE",
    "city_tier": "MERCHANT_PROFILE",
    "prior_if_self_resolved_rate": "UNAVAILABLE",
    "is_month_end_settlement_rush": "UNAVAILABLE",
    "bank_network_conditions": "UNAVAILABLE",
    "issuing_bank_downtime_flag": "UNAVAILABLE",
    "network_latency_bucket": "UNAVAILABLE",
}
assert set(FEATURE_SOURCES) == set(EVENT_FEATURE_KEYS), "FEATURE_SOURCES must classify exactly Model B's 12 required keys"

RAZORPAY_API_BASE_URL = "https://api.razorpay.com/v1"


@dataclass(frozen=True)
class LiveFeatureResult:
    """Result of one build_live_features() call. `features` contains ONLY
    keys that were genuinely, honestly obtained -- never a placeholder or
    fabricated value for a missing one. `complete` is True only when every
    one of Model B's 12 required keys is present."""

    features: dict[str, Any] = field(default_factory=dict)
    missing_features: list[str] = field(default_factory=list)
    complete: bool = False
    enrichment_attempted: bool = False
    enrichment_succeeded: bool = False
    enrichment_error: str | None = None  # normalized failure category only -- never a raw exception string (may embed request/response detail)


# ---------------------------------------------------------------------------
# DERIVED_FROM_AUTHORITATIVE_DATA -- pure calendar math + this project's own
# durable event history. No network call, no randomness.
# ---------------------------------------------------------------------------

def _count_prior_failures(db: Session, subscription_id: str, raw_event_id: int) -> int:
    """Prior `payment.failed` deliveries for this subscription, per this
    project's own already-stored, already-HMAC-verified RawEvent history.
    Ordered by insertion (id) rather than a timestamp field to sidestep
    naive/aware datetime comparison entirely -- insertion order already IS
    processing order for this project's single-webhook-handler design."""
    return (
        db.query(RawEvent)
        .filter(
            RawEvent.subscription_id == subscription_id,
            RawEvent.event_type == "payment.failed",
            RawEvent.id < raw_event_id,
        )
        .count()
    )


def _derive_authoritative_features(db: Session, subscription_id: str, raw_event_id: int, failure_timestamp: datetime) -> dict[str, Any]:
    return {
        "day_of_month": failure_timestamp.day,
        "days_to_nearest_payday_window": days_to_nearest_payday_window(failure_timestamp),
        "prior_if_failure_count": _count_prior_failures(db, subscription_id, raw_event_id),
    }


# ---------------------------------------------------------------------------
# WEBHOOK_NATIVE -- already present in the stored raw payload.
# ---------------------------------------------------------------------------

def _extract_primary_instrument(raw_event: RawEvent) -> str | None:
    """`payload.payment.entity.method` (e.g. "card"/"upi"/"netbanking"/
    "wallet"/"emandate") -- present on every genuine Razorpay payment.failed
    delivery, but not persisted as its own RawEvent column (see
    app/models.py), so this re-reads the already-stored, already-verified
    raw JSON. Tolerates malformed/missing data by returning None (never
    raises) -- raw_payload is always well-formed JSON in practice (app/main.py
    parses it before ever constructing a RawEvent), but this is still a
    boundary the rest of this module treats as untrusted input."""
    try:
        payload = json.loads(raw_event.raw_payload)
        method = (((payload.get("payload") or {}).get("payment") or {}).get("entity") or {}).get("method")
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None
    return str(method) if method else None


# ---------------------------------------------------------------------------
# RAZORPAY_API_ENRICHED -- optional, best-effort, bounded-timeout call to
# Razorpay's own Subscriptions API. Gated by LIVE_FEATURE_ENRICHMENT_ENABLED
# (default off) and by the existing RAZORPAY_KEY_ID/RAZORPAY_KEY_SECRET
# actually being configured. Every failure mode degrades to "not enriched"
# -- this function never raises.
# ---------------------------------------------------------------------------

# Injectable seam for tests -- a callable(url, auth, timeout) -> httpx.Response-like
# object, defaulting to a real httpx.Client.get call. Never used to fabricate
# data; only to avoid a real network call in tests (see
# tests/test_live_feature_enrichment.py, which uses httpx.MockTransport).
SubscriptionFetcher = Callable[[str], dict[str, Any] | None]


def _default_fetch_subscription(subscription_id: str, *, http_client: Any | None = None) -> dict[str, Any] | None:
    import httpx

    if not settings.RAZORPAY_KEY_ID or not settings.RAZORPAY_KEY_SECRET:
        return None

    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=settings.LIVE_FEATURE_ENRICHMENT_TIMEOUT_SECONDS)
    url = f"{RAZORPAY_API_BASE_URL}/subscriptions/{subscription_id}"
    try:
        response = client.get(url, auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET))
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            log.warning("live_feature_enrichment: unexpected response type (%s) fetching subscription", type(data).__name__)
            return None
        return data
    except httpx.TimeoutException:
        log.warning("live_feature_enrichment: timeout fetching subscription (subscription_id redacted from log)")
        return None
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in (401, 403):
            log.warning("live_feature_enrichment: authentication failed (HTTP %s) fetching subscription", status)
        elif status == 404:
            log.warning("live_feature_enrichment: subscription not found (HTTP 404)")
        elif status == 429:
            log.warning("live_feature_enrichment: rate limited (HTTP 429) fetching subscription")
        else:
            log.warning("live_feature_enrichment: HTTP %s fetching subscription", status)
        return None
    except httpx.ConnectError:
        log.warning("live_feature_enrichment: network/DNS error fetching subscription")
        return None
    except httpx.HTTPError as exc:
        log.warning("live_feature_enrichment: request error fetching subscription (%s)", type(exc).__name__)
        return None
    except (ValueError, TypeError) as exc:  # malformed JSON body, or an unexpected value type inside it
        log.warning("live_feature_enrichment: malformed response fetching subscription (%s)", type(exc).__name__)
        return None
    finally:
        if owns_client:
            client.close()


def _extract_tenure_days(subscription_data: dict[str, Any], failure_timestamp: datetime) -> int | None:
    """Razorpay's Subscription entity carries `start_at` (unix timestamp,
    documented field) -- tenure is the whole days elapsed since then, as of
    this failure. Returns None (never raises, never guesses) if the field
    is missing or not a plausible unix timestamp."""
    start_at = subscription_data.get("start_at")
    if start_at is None:
        start_at = subscription_data.get("created_at")
    if not isinstance(start_at, (int, float)) or isinstance(start_at, bool):
        return None
    try:
        subscription_start = datetime.fromtimestamp(start_at, tz=timezone.utc).replace(tzinfo=None)
    except (ValueError, OSError, OverflowError):
        return None
    tenure = (failure_timestamp - subscription_start).days
    return max(0, tenure)


def _fetch_razorpay_enrichment(
    subscription_id: str, failure_timestamp: datetime, *, http_client: Any | None = None,
) -> tuple[dict[str, Any], bool, str | None]:
    """Returns (features, succeeded, error_category). Never raises -- every
    failure mode is caught in _default_fetch_subscription and reported here
    only as a normalized, secret-free category string."""
    subscription_data = _default_fetch_subscription(subscription_id, http_client=http_client)
    if subscription_data is None:
        return {}, False, "enrichment_unavailable_or_failed"

    tenure_days = _extract_tenure_days(subscription_data, failure_timestamp)
    if tenure_days is None:
        return {}, False, "subscription_response_missing_start_at"

    return {"tenure_days": tenure_days}, True, None


# ---------------------------------------------------------------------------
# Boundary entrypoint
# ---------------------------------------------------------------------------

def build_live_features(
    db: Session,
    raw_event: RawEvent,
    subscription_id: str,
    failure_timestamp: datetime,
    *,
    http_client: Any | None = None,
) -> LiveFeatureResult:
    """Assembles whatever subset of Model B's 12 required features can be
    honestly sourced for this live event. Never fabricates a value for a
    feature it cannot source (see FEATURE_SOURCES/module docstring) -- the
    caller (recovery/webhook_pipeline.py) passes `features` straight through
    as `failure_context`; policy/decision_engine.py's own existing
    missing-key check (`_predict_recovery_values`, unmodified by this fix)
    already fails closed to the rule-based fallback tier whenever any key is
    absent, and records exactly which keys were missing in the decision's
    audit trail (`decision_reason`) -- this function does not duplicate that
    logic, only supplies its input honestly.
    """
    features: dict[str, Any] = {}
    features.update(_derive_authoritative_features(db, subscription_id, raw_event.id, failure_timestamp))

    primary_instrument = _extract_primary_instrument(raw_event)
    if primary_instrument is not None:
        features["primary_instrument"] = primary_instrument

    enrichment_attempted = False
    enrichment_succeeded = False
    enrichment_error: str | None = None
    if settings.LIVE_FEATURE_ENRICHMENT_ENABLED:
        enrichment_attempted = True
        enriched, enrichment_succeeded, enrichment_error = _fetch_razorpay_enrichment(
            subscription_id, failure_timestamp, http_client=http_client,
        )
        features.update(enriched)

    missing = [key for key in EVENT_FEATURE_KEYS if key not in features]
    return LiveFeatureResult(
        features=features,
        missing_features=missing,
        complete=not missing,
        enrichment_attempted=enrichment_attempted,
        enrichment_succeeded=enrichment_succeeded,
        enrichment_error=enrichment_error,
    )
