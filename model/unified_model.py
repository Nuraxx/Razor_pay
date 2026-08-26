"""
Unified ML model shared across all five revenue-risk domains --
payment_failed (one-time/no-subscription only; see EVENT-TYPE ALIASING
below), checkout_abandoned, mandate_failed, receivable_overdue, and
promise_to_pay_broken.

TARGET: this model predicts recovery likelihood/value for a single
(event, candidate intervention) pair -- NOT a policy decision, NOT a
compliance decision, NOT an LLM output, and NOT a payment confirmation. The
label (`recovered`) is a synthetic simulated outcome generated independently
of any downstream policy/compliance/LLM logic (see `_make_training_data`);
nothing about how policy or compliance later act on a prediction feeds back
into training.

MODEL CLASS: a real, fitted `catboost.CatBoostClassifier` (see
`_fit_catboost_model`) -- not a heuristic and not a stand-in. CatBoost was
chosen deliberately over the imputed/one-hot sklearn pipeline this module
used to have: this schema has genuine, deliberate missingness (a
domain-specific feature like `invoice_amount` is structurally absent for a
`checkout_abandoned` event, not "missing at random"), and CatBoost's native
NaN handling for numeric features plus native categorical support means that
missingness is preserved as a real signal to the model instead of being
silently imputed into a fabricated value (Phase 3's "do not silently treat
missing information as a meaningful zero").

EVENT-TYPE ALIASING: the live dispatcher (policy/revenue_recovery_policy.py)
uses the storage-level event_type `payment_failed_no_subscription` for a
Payment Link / one-time-payment failure (distinct from a genuine
subscription payment_failed, which never reaches this model at all -- that
domain keeps using the separate, pre-existing Model B system in
policy/decision_engine_v4.py, unmodified). This module's own feature/
candidate schema was defined around the plain name `payment_failed`
(matching CANDIDATE_SPACE / SUPPORTED_EVENT_TYPES below); `normalize_event`
aliases `payment_failed_no_subscription` -> `payment_failed` for feature
construction ONLY, so a Payment Link event's amount/failure_reason/etc.
still lands in the correct schema slot instead of silently producing zero
valid candidates.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from app.logging_config import log

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "model" / "reports"

SUPPORTED_EVENT_TYPES = (
    "payment_failed",
    "checkout_abandoned",
    "mandate_failed",
    "receivable_overdue",
    "promise_to_pay_broken",
)

# Storage-level event_type strings that map onto one of SUPPORTED_EVENT_TYPES
# for ML feature/candidate purposes only -- see module docstring.
ML_EVENT_TYPE_ALIASES = {
    "payment_failed_no_subscription": "payment_failed",
}

CANDIDATE_SPACE = {
    # "payment_failed" here means the one-time-payment/Payment-Link domain
    # ONLY (payment_failed_no_subscription, see ML_EVENT_TYPE_ALIASES) --
    # genuine subscription payment_failed never reaches this model at all
    # (it keeps using the separate, pre-existing Model B system). The single
    # candidate below is deliberately the SAME name/action
    # policy/one_time_payment_rules.py already uses
    # (CANDIDATE_PAYMENT_LINK_REMINDER) -- Razorpay never silently
    # auto-retries a Payment Link, so "retry_1_day"/"retry_3_days" style
    # candidates (a real subscription capability) would be a false claim
    # here; see that module's docstring for the full reasoning. ML still
    # has a genuine role: estimating whether this candidate's recovery
    # probability/value clears the bar for the rule-based eligibility gate
    # to consult it at all (decide_for_revenue_risk_event never calls this
    # domain's ML path for an ineligible bucket in the first place).
    "payment_failed": ["payment_link_reminder"],
    # All of the below are the SAME candidate_type strings each domain's own
    # rule module (policy/checkout_rules.py, policy/mandate_rules.py,
    # policy/receivables_rules.py, policy/promise_broken_rules.py) and
    # recovery/revenue_orchestrator.py's _WINDOW_DESCRIPTIONS already use --
    # not a parallel, ML-only vocabulary. "wait"/NO_ACTION (checkout) and
    # "human_handoff" (receivables, always requires_human_review=True) are
    # deliberately excluded: those mean "nothing eligible to recommend right
    # now" / "must escalate to a human", which stay rule-authoritative (see
    # decide_for_revenue_risk_event's _ML_SKIP_CANDIDATES gate) rather than
    # something ML gets to pick among.
    "checkout_abandoned": ["reminder", "payment_link_reminder", "retry_checkout", "alternate_payment_method"],
    "mandate_failed": ["attempt_1", "attempt_2", "final_attempt"],
    "receivable_overdue": ["friendly_reminder", "payment_request", "promise_to_pay_request", "escalation"],
    "promise_to_pay_broken": ["urgent_reminder", "final_notice"],
}

SHARED_FEATURES = [
    "event_type",
    "amount",
    "currency",
    "failure_reason",
    "failure_code",
    "payment_method",
    "attempt_count",
    "prior_failure_count",
    "prior_recovery_rate",
    "customer_tenure",
    "customer_segment",
    "days_to_payday",
    "days_since_last_activity",
]

DOMAIN_SPECIFIC_FEATURES = {
    "payment_failed": ["subscription_age_days", "days_to_subscription_renewal"],
    "checkout_abandoned": ["checkout_age_minutes", "cart_value"],
    "mandate_failed": ["mandate_attempt_number"],
    "receivable_overdue": ["days_overdue", "invoice_amount", "invoice_age_days"],
    "promise_to_pay_broken": ["promise_age_days", "promise_confidence"],
}

FEATURE_COLUMNS = list(dict.fromkeys(SHARED_FEATURES + [f for features in DOMAIN_SPECIFIC_FEATURES.values() for f in features]))

NUMERIC_FEATURES = [
    "amount",
    "attempt_count",
    "prior_failure_count",
    "prior_recovery_rate",
    "customer_tenure",
    "days_to_payday",
    "days_since_last_activity",
    "subscription_age_days",
    "days_to_subscription_renewal",
    "checkout_age_minutes",
    "cart_value",
    "mandate_attempt_number",
    "days_overdue",
    "invoice_amount",
    "invoice_age_days",
    "promise_age_days",
    "promise_confidence",
]

CATEGORICAL_FEATURES = [
    "event_type",
    "currency",
    "failure_reason",
    "failure_code",
    "payment_method",
    "customer_segment",
]

MODEL_VERSION = "unified_catboost_v1"
UNIFIED_MODEL_PATH = ARTIFACTS_DIR / "unified_model.joblib"
TRAINING_REPORT_PATH = REPORTS_DIR / "unified_model_training_report.json"


class UnifiedModelUnavailable(RuntimeError):
    """Raised when no trained unified-model artifact exists on disk. Callers
    on the live path must catch this and fall back to the deterministic
    rule-based deciders -- see `get_live_unified_model` below."""


def _canon(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    return value


def normalize_event(event: Mapping[str, Any] | pd.Series | Any) -> dict[str, Any]:
    if hasattr(event, "to_dict"):
        event = event.to_dict()
    if not isinstance(event, Mapping):
        raise TypeError("event must be a mapping-like object")

    event_type = str(_canon(event.get("event_type"), "unknown")).strip()
    event_type = ML_EVENT_TYPE_ALIASES.get(event_type, event_type)
    normalized = {
        "event_type": event_type,
        "amount": float(_canon(event.get("amount"), 0.0) or 0.0),
        "currency": str(_canon(event.get("currency"), "INR") or "INR"),
        "failure_reason": _canon(event.get("failure_reason"), "unknown"),
        "failure_code": _canon(event.get("failure_code"), "unknown"),
        "payment_method": _canon(event.get("payment_method"), "unknown"),
        "attempt_count": int(_canon(event.get("attempt_count"), 0) or 0),
        "prior_failure_count": int(_canon(event.get("prior_failure_count"), 0) or 0),
        "prior_recovery_rate": float(_canon(event.get("prior_recovery_rate"), 0.0) or 0.0),
        "customer_tenure": float(_canon(event.get("customer_tenure"), 0.0) or 0.0),
        "customer_segment": _canon(event.get("customer_segment"), "unknown"),
        "days_to_payday": float(_canon(event.get("days_to_payday"), 30.0) or 30.0),
        "days_since_last_activity": float(_canon(event.get("days_since_last_activity"), 7.0) or 7.0),
    }

    for domain_name, feature_list in DOMAIN_SPECIFIC_FEATURES.items():
        for feature in feature_list:
            normalized[feature] = _canon(event.get(feature), None)
            if normalized[feature] is None:
                normalized[feature] = np.nan
    return normalized


def generate_valid_candidates(event_type: str) -> list[str]:
    event_type = str(event_type or "").strip()
    event_type = ML_EVENT_TYPE_ALIASES.get(event_type, event_type)
    if event_type not in CANDIDATE_SPACE:
        return []
    return list(CANDIDATE_SPACE[event_type])


def _build_candidate_context(candidate_type: str | None) -> dict[str, Any]:
    return {
        "candidate_type": candidate_type,
        "candidate_is_payday": bool(candidate_type and "payday" in candidate_type),
        "candidate_is_urgent": bool(candidate_type and ("final" in candidate_type or "escalation" in candidate_type or "human_handoff" in candidate_type)),
        "candidate_is_reminder": bool(candidate_type and "reminder" in candidate_type),
    }


def build_unified_feature_vector(event: Mapping[str, Any] | pd.Series, candidate_type: str | None = None) -> pd.DataFrame:
    base = normalize_event(event)
    base.update(_build_candidate_context(candidate_type))
    columns = list(dict.fromkeys(SHARED_FEATURES + [f for features in DOMAIN_SPECIFIC_FEATURES.values() for f in features] + ["candidate_type", "candidate_is_payday", "candidate_is_urgent", "candidate_is_reminder"]))
    record = {col: base.get(col, np.nan) for col in columns}
    return pd.DataFrame([record], columns=columns)


# ---------------------------------------------------------------------------
# Synthetic unified dataset
# ---------------------------------------------------------------------------

ENTITIES_PER_DOMAIN = 900  # up from 240 -- a substantially larger population,
# entity-level split still strictly enforced (see _entity_level_split).

# Segment x candidate-urgency interaction (used only by the synthetic
# generator below): vip/loyal customers respond better to a gentle
# candidate and worse to an urgent/escalation one; at_risk customers are the
# reverse (+1 = "likes urgency", -1 = "dislikes urgency"). Without an
# interaction like this, the entity's own features never change WHICH
# candidate is best for it (only the candidate's fixed identity does, which
# any fixed rule could hardcode with no ML at all) -- see
# model/reports/unified_model_evaluation_report.json's "note" for why this
# matters for the held-out evaluation's baseline comparison.
_SEGMENT_URGENCY_PREFERENCE = {"vip": -1.0, "loyal": -0.5, "new": 0.0, "at_risk": 1.0}

# Some failure reasons genuinely respond worse to an urgent/escalation-style
# candidate than others (a "dispute" customer is already adversarial; an
# urgent nudge is more likely to backfire than for a plain "timeout").
_FAILURE_REASON_URGENCY_PREFERENCE = {
    "insufficient_fund": 0.3, "card_declined": 0.1, "timeout": 0.4, "dispute": -0.8,
    "promise_broken": 0.2, "mandate_failure": 0.0, "payment_failure": 0.0, "unknown": -0.2,
}

_SEGMENT_BASE_RATE = {"vip": 0.20, "loyal": 0.10, "new": 0.0, "at_risk": -0.20}


def _candidate_urgency_level(candidate: str) -> int:
    """0=gentle (a reminder), 2=urgent (escalation/final/human_handoff), 1=everything else (a retry/alternate-method/request-style action)."""
    if "final" in candidate or "escalation" in candidate or "urgent" in candidate or "human_handoff" in candidate:
        return 2
    if "reminder" in candidate:
        return 0
    return 1


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _make_training_data(seed: int = 7) -> pd.DataFrame:
    """One row per (entity, candidate) pair, across all 5 domains. Each
    entity is uniquely identified by `entity_key` = f"{event_type}:{local_id}"
    -- distinct entities never share features (see _entity_level_split,
    which groups by this key so no entity's rows can straddle two splits).

    Generative model, in LOGIT space (so every effect below composes
    nonlinearly through the final sigmoid, not just additively on the
    probability scale): an ENTITY-level base logit (amount, recency,
    payday proximity, prior recovery rate, attempt fatigue, segment,
    failure_reason, plus each domain's own specific feature -- checkout_age,
    days_overdue, mandate_attempt_number, promise_confidence) that is
    IDENTICAL across every candidate for that entity, plus a CANDIDATE-level
    interaction term that genuinely changes which candidate is best for
    THAT entity (segment x urgency, prior_failure_count x urgency,
    attempt_count x urgency with an OPPOSING sign -- a real non-monotonic
    combination -- failure_reason x urgency, and each domain's own
    candidate-specific interaction). `_true_probability` (the column) is
    the DETERMINISTIC, feature-driven mean probability -- i.e. the "oracle"
    ceiling any feature-based model could achieve; the actually-observed
    `recovered` label is a Bernoulli draw against that mean PLUS a small
    idiosyncratic per-draw noise term representing genuinely unobservable
    factors no feature could ever predict. Neither `_true_probability` nor
    the noise term is ever exposed to the model as a feature -- see
    FEATURE_COLUMNS, which never includes it."""
    rng = np.random.default_rng(seed)
    rows = []
    for event_type in SUPPORTED_EVENT_TYPES:
        for local_id in range(ENTITIES_PER_DOMAIN):
            entity_key = f"{event_type}:{local_id}"
            amount = float(rng.integers(200, 12000))
            attempt_count = int(rng.integers(1, 5))
            prior_failure_count = int(rng.integers(0, 4))
            prior_recovery_rate = float(rng.random() * 0.8)
            customer_segment = rng.choice(["new", "loyal", "vip", "at_risk"])
            failure_reason = rng.choice(["insufficient_fund", "card_declined", "timeout", "dispute", "promise_broken", "mandate_failure", "payment_failure", "unknown"])
            days_to_payday = float(rng.integers(1, 30))
            days_since_last_activity = float(rng.integers(1, 20))
            checkout_age_minutes = float(rng.integers(15, 240)) if event_type == "checkout_abandoned" else np.nan
            mandate_attempt_number = float(rng.integers(1, 4)) if event_type == "mandate_failed" else np.nan
            days_overdue = float(rng.integers(1, 30)) if event_type == "receivable_overdue" else np.nan
            promise_confidence = float(rng.random()) if event_type == "promise_to_pay_broken" else np.nan

            est = {
                "event_type": event_type,
                "amount": amount,
                "currency": "INR",
                "failure_reason": failure_reason,
                "failure_code": rng.choice(["BAD_REQUEST_ERROR", "AUTH_ERROR", "GATEWAY_ERROR", "THROTTLED", "UNKNOWN"]),
                "payment_method": rng.choice(["card", "upi", "netbanking", "wallet", "emandate"]),
                "attempt_count": attempt_count,
                "prior_failure_count": prior_failure_count,
                "prior_recovery_rate": prior_recovery_rate,
                "customer_tenure": float(rng.integers(30, 730)),
                "customer_segment": customer_segment,
                "days_to_payday": days_to_payday,
                "days_since_last_activity": days_since_last_activity,
                "subscription_age_days": float(rng.integers(10, 400)) if event_type == "payment_failed" else np.nan,
                "days_to_subscription_renewal": float(rng.integers(1, 30)) if event_type == "payment_failed" else np.nan,
                "checkout_age_minutes": checkout_age_minutes,
                "cart_value": float(rng.integers(200, 10000)) if event_type == "checkout_abandoned" else np.nan,
                "mandate_attempt_number": mandate_attempt_number,
                "days_overdue": days_overdue,
                "invoice_amount": float(rng.integers(200, 20000)) if event_type == "receivable_overdue" else np.nan,
                "invoice_age_days": float(rng.integers(1, 90)) if event_type == "receivable_overdue" else np.nan,
                "promise_age_days": float(rng.integers(1, 30)) if event_type == "promise_to_pay_broken" else np.nan,
                "promise_confidence": promise_confidence,
                # Distractor features: carried through the row but never read
                # by normalize_event/FEATURE_COLUMNS, so they cannot affect
                # training or inference -- present only so a downstream
                # feature-importance audit has something irrelevant to check.
                "distractor_ui_theme": rng.choice(["light", "dark"]),
                "distractor_app_build": int(rng.integers(1000, 9999)),
            }

            # --- Entity-level base logit -- identical across every candidate
            # for this entity (this is what makes it a genuine "how
            # recoverable is this entity at all" signal, not a candidate
            # ranking signal). Deliberately nonlinear: tanh/exp saturating
            # terms, not a plain linear sum.
            base_logit = (
                -0.4
                + 1.1 * prior_recovery_rate                                    # strongest single signal: has this customer recovered before?
                + 0.45 * np.tanh((amount - 3000.0) / 3000.0)                    # higher amount -> more invested -> more recoverable, saturating
                - 0.10 * attempt_count                                          # fatigue: more prior attempts, lower overall receptivity
                - 0.18 * np.log1p(days_since_last_activity) / np.log1p(20.0)    # staler activity -> harder to recover
                + 0.30 * np.exp(-days_to_payday / 8.0)                          # sharply higher right around payday, decays fast
                + _SEGMENT_BASE_RATE[customer_segment]
                + _FAILURE_REASON_URGENCY_PREFERENCE[failure_reason] * 0.15     # reason itself shifts overall recoverability a little, independent of urgency interaction below
            )
            if event_type == "checkout_abandoned":
                base_logit += -0.35 * np.tanh(checkout_age_minutes / 90.0)  # a checkout abandoned longer ago is structurally harder to recover
            elif event_type == "mandate_failed":
                base_logit += -0.30 * np.tanh((mandate_attempt_number - 1.0) / 2.5)  # saturating fatigue across the retry sequence
            elif event_type == "receivable_overdue":
                base_logit += -0.40 * np.tanh(days_overdue / 25.0)  # longer overdue -> structurally harder, saturating
            elif event_type == "promise_to_pay_broken":
                base_logit += 0.5 * (promise_confidence - 0.5)  # a broken promise that was originally high-confidence is still more recoverable than a low-confidence one

            for candidate in generate_valid_candidates(event_type):
                centered_urgency = _candidate_urgency_level(candidate) - 1  # gentle=-1, moderate=0, urgent=+1
                interaction = (
                    0.55 * _SEGMENT_URGENCY_PREFERENCE[customer_segment] * centered_urgency
                    + 0.18 * prior_failure_count * centered_urgency          # more prior failures -> urgency helps more
                    - 0.14 * attempt_count * centered_urgency                # BUT already many attempts -> urgency causes fatigue/annoyance instead (a real non-monotonic combination with the term above)
                    + 0.35 * _FAILURE_REASON_URGENCY_PREFERENCE[failure_reason] * centered_urgency
                )
                if event_type == "checkout_abandoned":
                    # Fresh checkouts respond better to a gentle nudge; older
                    # ones need a more structural action (retry/alternate
                    # method) -- a genuine feature x candidate interaction.
                    interaction += 0.40 * np.tanh(checkout_age_minutes / 60.0 - 1.2) * centered_urgency
                elif event_type == "mandate_failed":
                    # "final_attempt" carries a real, non-monotonic "last
                    # chance" framing bonus beyond plain urgency, but ONLY
                    # once the sequence has actually progressed.
                    if candidate == "final_attempt" and mandate_attempt_number >= 2:
                        interaction += 0.30
                elif event_type == "receivable_overdue":
                    interaction += 0.35 * np.tanh(days_overdue / 18.0 - 1.0) * centered_urgency
                elif event_type == "promise_to_pay_broken":
                    # urgency_level collapses to the same bucket for both of
                    # this domain's candidates (both "urgent"/"final"), so
                    # urgency-based interactions above cannot distinguish
                    # them -- promise_confidence directly does: a
                    # higher-confidence broken promise responds better to a
                    # still-trusting reminder; a low-confidence one needs
                    # the more formal final notice.
                    interaction += 0.6 * (promise_confidence - 0.5) * (1.0 if candidate == "urgent_reminder" else -1.0)

                mean_prob = float(np.clip(_sigmoid(base_logit + interaction), 0.01, 0.99))
                # Small idiosyncratic per-draw noise -- genuinely unobservable
                # factors no feature could ever predict. `mean_prob` (NOT
                # this noisy version) is what "oracle"/regret evaluation
                # compares against, since a perfect feature-based model
                # could never predict pure per-draw noise either.
                noisy_prob = float(np.clip(mean_prob + rng.normal(0.0, 0.04), 0.01, 0.99))
                recovered = bool(rng.random() < noisy_prob)
                rows.append({
                    **est,
                    "candidate_type": candidate,
                    "recovered": recovered,
                    "entity_key": entity_key,
                    "_true_probability": mean_prob,
                })
    return pd.DataFrame(rows)


def _entity_level_split(
    df: pd.DataFrame, seed: int = 13, train_frac: float = 0.70, val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits by `entity_key`, independently within EACH event_type, so every
    domain is represented in train/val/test and no entity's rows straddle
    two splits. (A naive global shuffle-by-row-index would still be safe
    against leakage but risks under-representing a domain in a given split
    by chance; splitting per-domain avoids that entirely.)"""
    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []
    for event_type in SUPPORTED_EVENT_TYPES:
        domain_df = df[df["event_type"] == event_type]
        entities = np.array(sorted(domain_df["entity_key"].unique()), dtype=object)
        rng.shuffle(entities)
        n = len(entities)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        train_entities = set(entities[:n_train])
        val_entities = set(entities[n_train:n_train + n_val])
        test_entities = set(entities[n_train + n_val:])
        train_parts.append(domain_df[domain_df["entity_key"].isin(train_entities)])
        val_parts.append(domain_df[domain_df["entity_key"].isin(val_entities)])
        test_parts.append(domain_df[domain_df["entity_key"].isin(test_entities)])
    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    train_keys = set(train_df["entity_key"])
    val_keys = set(val_df["entity_key"])
    test_keys = set(test_df["entity_key"])
    assert not (train_keys & val_keys), "entity leakage between train and val"
    assert not (train_keys & test_keys), "entity leakage between train and test"
    assert not (val_keys & test_keys), "entity leakage between val and test"

    return train_df, val_df, test_df


_CATBOOST_FIXED_PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=13,
    early_stopping_rounds=40,
    use_best_model=True,
    verbose=False,
)

# Small hyperparameter grid searched on VALIDATION AUC only (see
# _select_catboost_hyperparameters) -- TEST is never touched during this
# selection. `CATBOOST_PARAMS` is kept as a module-level name (the grid's
# first/default entry) purely so anything that imported it directly for
# inspection/logging still finds a sensible value; the actual fitted model
# always uses whichever grid entry validation preferred.
_CATBOOST_HYPERPARAM_GRID = [
    dict(iterations=400, depth=5, learning_rate=0.06, l2_leaf_reg=3.0),
    dict(iterations=400, depth=4, learning_rate=0.05, l2_leaf_reg=3.0),
    dict(iterations=600, depth=6, learning_rate=0.04, l2_leaf_reg=5.0),
    dict(iterations=300, depth=4, learning_rate=0.08, l2_leaf_reg=1.0),
    dict(iterations=500, depth=6, learning_rate=0.03, l2_leaf_reg=7.0),
]
CATBOOST_PARAMS = {**_CATBOOST_FIXED_PARAMS, **_CATBOOST_HYPERPARAM_GRID[0]}


def _model_feature_columns() -> list[str]:
    return FEATURE_COLUMNS + ["candidate_type"]


def _model_cat_features() -> list[str]:
    return [c for c in CATEGORICAL_FEATURES + ["candidate_type"] if c in _model_feature_columns()]


def _prepare_catboost_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[_model_feature_columns()].copy()
    for c in _model_cat_features():
        X[c] = X[c].astype(str)
    return X


def _select_catboost_hyperparameters(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, cat_features: list[str],
) -> tuple[CatBoostClassifier, dict[str, Any], list[dict[str, Any]]]:
    """Fits every entry in _CATBOOST_HYPERPARAM_GRID on TRAIN, scores each on
    VALIDATION AUC only, and returns the best-scoring already-fitted model.
    TEST is never referenced here -- this function doesn't even receive it."""
    results = []
    best_model: CatBoostClassifier | None = None
    best_params: dict[str, Any] | None = None
    best_val_auc = -1.0
    for params in _CATBOOST_HYPERPARAM_GRID:
        model = CatBoostClassifier(**_CATBOOST_FIXED_PARAMS, **params)
        model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val))
        val_proba = model.predict_proba(X_val)[:, 1]
        val_auc = float(roc_auc_score(y_val, val_proba))
        results.append({**params, "validation_auc": val_auc})
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model
            best_params = params
    return best_model, best_params, results


def _fit_catboost_model(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict[str, Any]:
    feature_cols = _model_feature_columns()
    cat_features = _model_cat_features()
    X_train = _prepare_catboost_frame(train_df)
    y_train = train_df["recovered"].astype(int)
    X_val = _prepare_catboost_frame(val_df)
    y_val = val_df["recovered"].astype(int)

    model, chosen_params, grid_results = _select_catboost_hyperparameters(X_train, y_train, X_val, y_val, cat_features)

    return {
        "model_version": MODEL_VERSION,
        "model": model,
        "feature_columns": feature_cols,
        "cat_features": cat_features,
        "candidate_space": dict(CANDIDATE_SPACE),
        "event_types": list(SUPPORTED_EVENT_TYPES),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "chosen_hyperparameters": chosen_params,
        "hyperparameter_search_results": grid_results,
    }


def _score_split(fitted: dict[str, Any], df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"n": 0, "auc": None, "pr_auc": None, "log_loss": None, "brier": None}
    X = _prepare_catboost_frame(df)
    y = df["recovered"].astype(int)
    proba = fitted["model"].predict_proba(X)[:, 1]
    metrics: dict[str, float] = {"n": int(len(df))}
    try:
        metrics["auc"] = float(roc_auc_score(y, proba)) if y.nunique() > 1 else None
    except ValueError:
        metrics["auc"] = None
    try:
        metrics["pr_auc"] = float(average_precision_score(y, proba)) if y.nunique() > 1 else None
    except ValueError:
        metrics["pr_auc"] = None
    try:
        metrics["log_loss"] = float(log_loss(y, proba, labels=[0, 1]))
    except ValueError:
        metrics["log_loss"] = None
    try:
        metrics["brier"] = float(brier_score_loss(y, proba))
    except ValueError:
        metrics["brier"] = None
    return metrics


def _model_path() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return UNIFIED_MODEL_PATH


def train_unified_model(
    train_df: pd.DataFrame | None = None,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    *,
    write_report: bool = True,
) -> dict[str, Any]:
    """Trains on TRAIN only; VALIDATION is used for CatBoost early stopping
    (see CATBOOST_PARAMS' early_stopping_rounds/use_best_model). TEST metrics
    are computed only for reporting AFTER fitting is complete -- test rows
    are never used to fit the model or to choose CATBOOST_PARAMS."""
    full_dataset = None
    if train_df is None:
        full_dataset = _make_training_data()
        train_df, val_df, test_df = _entity_level_split(full_dataset)
    elif val_df is None or test_df is None:
        raise ValueError("val_df and test_df must be supplied together with train_df")

    fitted = _fit_catboost_model(train_df, val_df)

    metrics = {
        "train": _score_split(fitted, train_df),
        "validation": _score_split(fitted, val_df),
        "test": _score_split(fitted, test_df),
    }
    fitted["metrics"] = metrics

    # Per-domain TEST metrics (Phase-11 requirement: report by domain, not
    # just pooled across all 5) -- payment_failed has only 1 candidate, so
    # its AUC/PR-AUC are not a meaningful ranking signal (see the report's
    # own note); still computed for completeness.
    metrics_by_domain = {
        et: _score_split(fitted, test_df[test_df["event_type"] == et]) for et in SUPPORTED_EVENT_TYPES
    }
    fitted["metrics_by_domain_test"] = metrics_by_domain

    dataset_stats = {
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_entities": int(train_df["entity_key"].nunique()) if "entity_key" in train_df else None,
        "validation_entities": int(val_df["entity_key"].nunique()) if "entity_key" in val_df else None,
        "test_entities": int(test_df["entity_key"].nunique()) if "entity_key" in test_df else None,
        "events_per_domain": {
            et: int((pd.concat([train_df, val_df, test_df])["event_type"] == et).sum())
            for et in SUPPORTED_EVENT_TYPES
        } if full_dataset is not None else None,
        "class_balance_recovered_rate": float(pd.concat([train_df, val_df, test_df])["recovered"].mean()),
        "feature_count": len(fitted["feature_columns"]),
    }
    fitted["dataset_stats"] = dataset_stats

    joblib.dump(fitted, _model_path())

    if write_report:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report = {
            "model_version": fitted["model_version"],
            "trained_at": fitted["trained_at"],
            "artifact_path": str(UNIFIED_MODEL_PATH),
            "feature_columns": fitted["feature_columns"],
            "cat_features": fitted["cat_features"],
            "candidate_space": fitted["candidate_space"],
            "event_types": fitted["event_types"],
            "chosen_hyperparameters": fitted["chosen_hyperparameters"],
            "hyperparameter_search_results": fitted["hyperparameter_search_results"],
            "dataset_stats": dataset_stats,
            "metrics": metrics,
            "metrics_by_domain_test": metrics_by_domain,
        }
        TRAINING_REPORT_PATH.write_text(json.dumps(report, indent=2))

    return fitted


def load_unified_model() -> dict[str, Any]:
    """Loads the persisted artifact. Raises UnifiedModelUnavailable if it
    does not exist OR fails to deserialize (corrupt/truncated file, wrong
    format, etc.) -- deliberately does NOT silently train a fresh model on
    the live path (that would be nondeterministic and mask genuine
    unavailability); run model/train_unified_model.py explicitly instead.
    A corrupt artifact must degrade the same way a missing one does (caught
    by get_live_unified_model() below), never crash the caller with a raw
    joblib/pickle exception."""
    artifact = _model_path()
    if not artifact.exists():
        raise UnifiedModelUnavailable(
            f"Unified model artifact not found at {artifact}. Run `./venv/bin/python model/train_unified_model.py` first."
        )
    try:
        loaded = joblib.load(artifact)
    except Exception as exc:
        raise UnifiedModelUnavailable(f"Unified model artifact at {artifact} could not be deserialized ({type(exc).__name__}: {exc}).") from exc
    if not isinstance(loaded, dict) or "model" not in loaded or "feature_columns" not in loaded:
        raise UnifiedModelUnavailable(f"Unified model artifact at {artifact} does not have the expected shape (missing 'model'/'feature_columns').")
    return loaded


# ---------------------------------------------------------------------------
# Live-process cached loader -- THE single entrypoint app/main.py,
# recovery/scheduler.py, and recovery/demo_generator.py use to obtain the
# unified model for live inference. Cached so a request/sweep-cycle never
# re-pays deserialization cost; loaded (and logged) at most once per process.
# ---------------------------------------------------------------------------
_LIVE_MODEL_CACHE: dict[str, Any] | None = None
_LIVE_MODEL_LOAD_ATTEMPTED = False


def get_live_unified_model() -> dict[str, Any] | None:
    global _LIVE_MODEL_CACHE, _LIVE_MODEL_LOAD_ATTEMPTED
    if not _LIVE_MODEL_LOAD_ATTEMPTED:
        _LIVE_MODEL_LOAD_ATTEMPTED = True
        try:
            _LIVE_MODEL_CACHE = load_unified_model()
            log.info(
                "Unified ML model loaded: model=%s artifact=%s",
                _LIVE_MODEL_CACHE.get("model_version", MODEL_VERSION), UNIFIED_MODEL_PATH.name,
            )
        except UnifiedModelUnavailable as exc:
            log.warning(
                "Unified ML model unavailable -- revenue-risk domains will use the deterministic rule-based "
                "fallback until retrained (%s)", exc,
            )
            _LIVE_MODEL_CACHE = None
    return _LIVE_MODEL_CACHE


def reset_live_unified_model_cache() -> None:
    """Test-only: clears the process-wide cache so a test can simulate the
    artifact being unavailable (Phase 16) or force a fresh load."""
    global _LIVE_MODEL_CACHE, _LIVE_MODEL_LOAD_ATTEMPTED
    _LIVE_MODEL_CACHE = None
    _LIVE_MODEL_LOAD_ATTEMPTED = False


def predict_unified_probability(event: Mapping[str, Any] | pd.Series, candidate_type: str, model: dict[str, Any] | None = None) -> float:
    model = model or load_unified_model()
    feature_df = build_unified_feature_vector(event, candidate_type)
    X = feature_df[[c for c in model["feature_columns"] if c != "candidate_type"]].copy()
    X["candidate_type"] = candidate_type
    X = X[model["feature_columns"]]
    for c in model.get("cat_features", []):
        X[c] = X[c].astype(str)
    prob = model["model"].predict_proba(X)[0, 1]
    return float(np.clip(prob, 0.0, 1.0))


def score_event_candidates(event: Mapping[str, Any] | pd.Series, model: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    normalized = normalize_event(event)
    event_type = str(normalized.get("event_type", "unknown"))
    scoring_model = model or load_unified_model()
    amount = float(normalized.get("amount", 0.0) or 0.0)
    scores = []
    for candidate in generate_valid_candidates(event_type):
        prob = predict_unified_probability(event, candidate, scoring_model)
        value = prob * amount
        scores.append({
            "candidate_type": candidate,
            "predicted_recovery_probability": prob,
            "predicted_recovery_value": value,
            "expected_incremental_value": value,
            "model_version": scoring_model.get("model_version", MODEL_VERSION),
        })
    return scores


def select_best_candidate(event: Mapping[str, Any] | pd.Series, model: dict[str, Any] | None = None) -> dict[str, Any]:
    scores = score_event_candidates(event, model)
    if not scores:
        return {"candidate_type": "NO_ACTION", "predicted_recovery_probability": 0.0, "predicted_recovery_value": 0.0, "expected_incremental_value": 0.0}
    return max(scores, key=lambda item: item["predicted_recovery_value"])


def main() -> None:
    fitted = train_unified_model()
    print(f"Unified model trained: {fitted['model_version']}")
    print(f"Artifact: {UNIFIED_MODEL_PATH}")
    print(f"Chosen hyperparameters (selected on VALIDATION AUC only): {json.dumps(fitted['chosen_hyperparameters'], indent=2)}")
    print(f"Hyperparameter search results: {json.dumps(fitted['hyperparameter_search_results'], indent=2)}")
    print(f"Dataset stats: {json.dumps(fitted['dataset_stats'], indent=2)}")
    print(f"Metrics: {json.dumps(fitted['metrics'], indent=2)}")
    print(f"Metrics by domain (TEST): {json.dumps(fitted['metrics_by_domain_test'], indent=2)}")
    print(f"Full report written to: {TRAINING_REPORT_PATH}")


if __name__ == "__main__":
    main()
