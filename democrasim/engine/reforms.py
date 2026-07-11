"""The two platforms: actual encoded reforms with opposite incidence.

Labels are generic by design — Policy A and Policy B, never real candidates
or parties. Both are federal individual income tax changes for 2026 onward
(explicit start AND end dates, per PolicyEngine reform semantics), chosen to
concentrate their gross benefits at opposite ends of the income
distribution and calibrated to roughly matched budgetary cost so that
neither is simply "the bigger giveaway":

- **Policy A** raises the Child Tax Credit base amount. Gross benefits go to
  households with children, concentrated in the lower and middle of the
  distribution (the refundability phase-in limits how much reaches the very
  bottom — that is measured, not assumed).
- **Policy B** cuts the top two marginal income tax rates. Gross benefits go
  almost entirely to the highest-income households.

The exact parameter values were calibrated against the engine so the two
policies' total household dollars are within a few percent of each other;
totals are recorded in the artifact metadata at build time.
"""

PERIOD = "2026-01-01.2100-12-31"
YEAR = 2026

POLICIES: list[dict] = [
    {
        "label": "Policy A",
        "column": "delta_policy_a",
        "description": (
            "Raise the federal Child Tax Credit base amount from $2,200 to "
            "$3,200 per child, from 2026 onward."
        ),
        "reform": {
            "gov.irs.credits.ctc.amount.base[0].amount": {PERIOD: 3_200},
        },
    },
    {
        "label": "Policy B",
        "column": "delta_policy_b",
        "description": (
            "Cap the top federal marginal income tax rate at 34% (the 35% "
            "and 37% brackets both drop to 34%), from 2026 onward."
        ),
        "reform": {
            "gov.irs.income.bracket.rates.7": {PERIOD: 0.34},
            "gov.irs.income.bracket.rates.6": {PERIOD: 0.34},
        },
    },
]
