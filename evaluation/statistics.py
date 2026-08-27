"""
FIX pass: statistical significance for the baseline comparison already
computed by `evaluation/evaluate_decision_engine_v4.py` -- no new evaluation
population, no new outcome definition. Both functions below operate on the
SAME per-event `events` DataFrame that script already builds from the SAME
held-out test split (`model/latent_target_preprocessing.py::split_candidate_dataset`),
using the SAME two columns that script already computes per policy:
`{policy}__realized_recovered` (binary) and `{policy}__realized_amount_recovered`
(₹, continuous).

Per the original specification
(~/Downloads/razorpay-track3-project-specification.md, "Evaluation metrics
and formulas"): "Because these are paired outcomes on the same events, use
McNemar's test (not a two-proportion z-test) for significance" for recovery
lift, and "report a bootstrap confidence interval on any model-vs-baseline
delta, not a bare point estimate."

McNemar's test is applied ONLY to the paired BINARY `realized_recovered`
outcome -- never to the continuous ₹ amounts, which is what the bootstrap CI
below is for instead. Applying McNemar to a continuous quantity would not be
statistically meaningful (it is a test for paired nominal/binary data), so
this module deliberately keeps the two apart rather than offering one
McNemar-shaped function that could be misapplied to either.

Every number produced here is still part of the SYNTHETIC COUNTERFACTUAL
EVALUATION this whole project's evaluation layer already is -- see this
module's callers for that label. Neither function below establishes real-world
production superiority; they quantify uncertainty in this synthetic,
held-out result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy.stats import binomtest, chi2


@dataclass(frozen=True)
class McNemarResult:
    """`policy_a` is conventionally the policy being evaluated (e.g. the
    deployed policy), `policy_b` the baseline it's compared against (e.g.
    Fixed Retry) -- purely a labeling convention, the test itself is
    symmetric in a/b (b and c simply swap)."""

    policy_a: str
    policy_b: str
    n_paired_events: int
    both_recovered: int  # a: recovered under both policies
    only_a_recovered: int  # b: recovered under policy_a only
    only_b_recovered: int  # c: recovered under policy_b only
    neither_recovered: int  # d: recovered under neither
    statistic: float | None  # continuity-corrected chi-square statistic; None for the exact (binomial) method
    p_value: float
    method: str  # "exact_binomial" | "chi_square_continuity_corrected" | "no_discordant_pairs"
    exact: bool

    def to_dict(self) -> dict:
        return asdict(self)


def mcnemar_test(
    outcomes_a: Sequence[bool],
    outcomes_b: Sequence[bool],
    *,
    policy_a: str,
    policy_b: str,
    exact: bool = True,
) -> McNemarResult:
    """
    Paired McNemar's test on two same-length sequences of booleans -- the
    SAME event, in the SAME order, under two different policies (the caller
    is responsible for that alignment; this function only counts).

    `exact=True` (default, and what the rest of this project reports):
    the exact binomial test on the discordant pairs (`scipy.stats.binomtest`,
    two-sided, p=0.5) -- the standard "exact McNemar" method, appropriate at
    any sample size and exactly correct rather than an approximation.

    `exact=False`: the classic continuity-corrected chi-square approximation
    -- `(|b-c|-1)^2 / (b+c) ~ chi2(1)` -- provided because it is the more
    commonly cited textbook form, not because it is preferred here.

    When b + c == 0 (the two policies agree on every single paired event),
    there are no discordant pairs for McNemar's test to use at all -- this
    is reported explicitly (`method="no_discordant_pairs"`, p_value=1.0)
    rather than silently calling into scipy with n=0.
    """
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError(f"outcomes_a and outcomes_b must be the same length (paired events): {len(outcomes_a)} != {len(outcomes_b)}")
    if len(outcomes_a) == 0:
        raise ValueError("mcnemar_test requires at least one paired event")

    a = b = c = d = 0
    for oa, ob in zip(outcomes_a, outcomes_b):
        oa, ob = bool(oa), bool(ob)
        if oa and ob:
            a += 1
        elif oa and not ob:
            b += 1
        elif not oa and ob:
            c += 1
        else:
            d += 1

    if b + c == 0:
        return McNemarResult(
            policy_a=policy_a, policy_b=policy_b, n_paired_events=len(outcomes_a),
            both_recovered=a, only_a_recovered=b, only_b_recovered=c, neither_recovered=d,
            statistic=None, p_value=1.0, method="no_discordant_pairs", exact=exact,
        )

    if exact:
        p_value = binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue
        return McNemarResult(
            policy_a=policy_a, policy_b=policy_b, n_paired_events=len(outcomes_a),
            both_recovered=a, only_a_recovered=b, only_b_recovered=c, neither_recovered=d,
            statistic=None, p_value=float(p_value), method="exact_binomial", exact=True,
        )

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(chi2.sf(statistic, df=1))
    return McNemarResult(
        policy_a=policy_a, policy_b=policy_b, n_paired_events=len(outcomes_a),
        both_recovered=a, only_a_recovered=b, only_b_recovered=c, neither_recovered=d,
        statistic=float(statistic), p_value=p_value, method="chi_square_continuity_corrected", exact=False,
    )


@dataclass(frozen=True)
class BootstrapCIResult:
    metric: str
    policy_a: str  # conventionally the policy being evaluated (e.g. the deployed policy)
    policy_b: str  # conventionally the baseline (e.g. Fixed Retry)
    point_estimate: float  # sum(values_a) - sum(values_b) on the ACTUAL (non-resampled) test set -- sign convention: positive means policy_a > policy_b
    method: str
    n_resamples: int
    seed: int
    confidence_level: float
    lower_bound: float
    upper_bound: float
    n_events: int

    def to_dict(self) -> dict:
        return asdict(self)


def bootstrap_delta_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    policy_a: str,
    policy_b: str,
    metric_name: str,
    n_resamples: int = 10000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> BootstrapCIResult:
    """
    Percentile bootstrap CI (Efron) on `sum(values_a) - sum(values_b)`.

    Resamples EVENTS (not the two value sequences independently) with
    replacement -- each bootstrap draw picks a set of event indices and pulls
    both policies' values for those same events, preserving the pairing
    (the whole reason `values_a[i]`/`values_b[i]` are the same event under
    two policies, not two independent samples).

    Deterministic: uses a seeded `numpy.random.default_rng(seed)` local
    generator (never the global numpy random state), so the same inputs +
    seed always reproduce the identical bounds.

    Zero-variance case: if `values_a[i] - values_b[i]` is the same constant
    for every event (most simply, all-zero), every resample produces that
    same constant sum-delta, so lower_bound == upper_bound == point_estimate
    -- a genuinely zero-width interval, not a bug.
    """
    if len(values_a) != len(values_b):
        raise ValueError(f"values_a and values_b must be the same length (paired events): {len(values_a)} != {len(values_b)}")
    n = len(values_a)
    if n == 0:
        raise ValueError("bootstrap_delta_ci requires at least one paired event")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1): {confidence_level}")

    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    point_estimate = float(values_a.sum() - values_b.sum())

    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(0, n, size=(n_resamples, n))
    resampled_deltas = values_a[resample_indices].sum(axis=1) - values_b[resample_indices].sum(axis=1)

    alpha = 1.0 - confidence_level
    lower_bound = float(np.percentile(resampled_deltas, 100 * (alpha / 2)))
    upper_bound = float(np.percentile(resampled_deltas, 100 * (1 - alpha / 2)))

    return BootstrapCIResult(
        metric=metric_name, policy_a=policy_a, policy_b=policy_b, point_estimate=point_estimate,
        method="percentile_bootstrap_paired_event_resampling", n_resamples=n_resamples, seed=seed,
        confidence_level=confidence_level, lower_bound=lower_bound, upper_bound=upper_bound, n_events=n,
    )
