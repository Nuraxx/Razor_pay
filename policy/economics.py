"""
FIX pass: separates three genuinely distinct quantities the original
specification requires reported separately, never blended into one number
(~/Downloads/razorpay-track3-project-specification.md, "₹ recovered": "Report
both raw merchant GMV and Razorpay's own fee take (× the disclosed ~2%+GST
card rate) as two separate numbers."):

    1. recovered_gmv        -- the merchant's recovered amount (what the
                                rest of this codebase already calls
                                `total_recovered_rs` / `realized_amount_recovered`
                                -- see the module-mapping note below).
    2. intervention_cost    -- this project's OWN existing cost model
                                (policy/costs.py, unchanged), summed over
                                every real (non-NO_ACTION) action a policy
                                selected.
    3. razorpay_fee_take    -- Razorpay's disclosed transaction fee, newly
                                modeled here for the first time in this
                                project (no prior module computed it).

FEE ASSUMPTION -- verified against the specification, not silently guessed:
the specification (line 24) states "Razorpay's own blog confirms domestic
card payments are priced at roughly 2% + 18% GST." The same document (line
25) explicitly flags UPI/RuPay's exact rate as an unresolved inconsistency
in Razorpay's own public materials and instructs: "do not state a single
clean UPI take-rate figure." This module therefore applies the one verified,
citable CARD rate uniformly to every recovered rupee regardless of payment
instrument -- a documented simplification, not a second, uncited number
invented for UPI.

FEE CONVENTION (explicit, per the spec's "2% + 18% GST" wording -- GST in
India is charged ON the service fee, not on the transaction GMV itself):

    base_fee          = recovered_amount * BASE_FEE_RATE            (2% of GMV)
    gst_on_fee        = base_fee * GST_RATE                          (18% GST on the fee itself)
    razorpay_fee_take = base_fee + gst_on_fee                        (effective ~2.36% of GMV)

`razorpay_fee_take` as computed here is the GROSS amount the merchant is
billed for that transaction (fee + GST on the fee) -- i.e. what actually
leaves the merchant's settlement, not Razorpay's post-GST-remittance net
kept revenue (the GST portion is collected on Razorpay's behalf and remitted
to the government, not retained -- the specification does not ask for that
further split, so this module does not attempt it either).

TERMINOLOGY MAPPING (per the FIX pass instruction to preserve existing
names, not rename them): `policy/decision_engine.py`'s per-candidate
`expected_net_value` / `decision_margin` (recovered value minus
`intervention_cost` ONLY, computed at DECISION time, before any outcome is
known) is UNCHANGED and NOT touched by this module. `net_recovery_value`
below is a DIFFERENT, new, REALIZED (post-outcome) summary metric --
recovered GMV minus BOTH intervention cost AND Razorpay's fee take -- and is
never used anywhere a decision is made, only in the evaluation report/
dashboard, alongside the existing (unrenamed) `total_recovered_rs` /
`realized_amount_recovered` / `expected_net_value` fields.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

BASE_FEE_RATE = 0.02  # ~2%, per the specification's disclosed domestic card rate
GST_RATE = 0.18  # 18% GST on the fee itself, per the specification
EFFECTIVE_FEE_RATE = BASE_FEE_RATE * (1 + GST_RATE)  # ~2.36% of recovered GMV, gross (fee + GST on the fee)


@dataclass(frozen=True)
class RecoveryEconomics:
    recovered_gmv: float  # merchant's recovered amount -- see module docstring's terminology mapping
    intervention_cost: float  # this project's existing cost model (policy/costs.py), unchanged
    razorpay_fee_take: float  # GROSS, including GST -- see module docstring
    net_recovery_value: float  # recovered_gmv - intervention_cost - razorpay_fee_take -- see module docstring; distinct from policy/decision_engine.py's expected_net_value

    def to_dict(self) -> dict:
        return asdict(self)


def compute_recovery_economics(recovered_gmv: float, intervention_cost: float) -> RecoveryEconomics:
    """
    Pure, deterministic. `recovered_gmv` and `intervention_cost` are the
    project's EXISTING, already-computed quantities (e.g.
    `evaluation/evaluate_decision_engine_v4.py`'s
    `total_recovered_rs` / summed `policy/costs.py::cost_for_candidate`) --
    this function only adds the fee-take split on top; it never recomputes
    or renames either input, and never mixes recovered GMV with fee revenue
    in a single blended field.
    """
    razorpay_fee_take = round(recovered_gmv * EFFECTIVE_FEE_RATE, 2)
    net_recovery_value = round(recovered_gmv - intervention_cost - razorpay_fee_take, 2)
    return RecoveryEconomics(
        recovered_gmv=round(recovered_gmv, 2),
        intervention_cost=round(intervention_cost, 2),
        razorpay_fee_take=razorpay_fee_take,
        net_recovery_value=net_recovery_value,
    )
