"""Synthetic comparator electorates — clearly labeled, never real data.

These exist to answer one question in the findings: *what did moving from
invented to measured impact distributions change?* Every electorate built
here carries a ``TOY`` source string and must be presented as synthetic.

Two comparators:

- :func:`moment_matched_toy` — a jointly Gaussian world matched to the
  measured electorate's weighted means and covariances of
  ``(impact_A, impact_B, log income)``. Matching the income covariance
  matters: welfare metrics read the *incidence* of a policy (who gains,
  ordered by income), so a toy that matched impact moments alone would
  quietly destroy the welfare signal along with the shape. With all first
  and second moments held, differences from the measured results are
  attributable to distribution shape — skew, tails, and the mass of voters
  with no stake.
- :func:`homogeneous_toy` — the old model's implicit world: every voter has
  the same stake in the outcome, so heterogeneity comes only from perception
  noise. This is what "candidate A's economic value is 1.0, B's is 0.8"
  amounted to.
"""

import numpy as np

from democrasim.electorate import Electorate


def moment_matched_toy(
    electorate: Electorate,
    rng: np.random.Generator,
    *,
    n: int | None = None,
) -> Electorate:
    """Gaussian world matched to the measured first and second moments.

    ``(δ_1, …, δ_p, ln max(income, $1))`` is drawn from a multivariate
    normal whose mean vector and covariance matrix equal the weighted
    moments of the source electorate; income is then exponentiated (so its
    marginal is lognormal and its rank-correlation with impacts is
    preserved). Weights are uniform and every voter is their household's
    only adult, as the old model implicitly assumed.
    """
    n = electorate.n_voters if n is None else n
    log_income = np.log(np.maximum(electorate.base_income, 1.0))
    joint = np.column_stack([electorate.deltas, log_income])
    mean = np.average(joint, axis=0, weights=electorate.weights)
    cov = np.cov(joint.T, aweights=electorate.weights, ddof=0)
    draws = rng.multivariate_normal(mean, cov, size=n)
    return Electorate(
        deltas=draws[:, :-1],
        weights=np.ones(n),
        base_income=np.exp(draws[:, -1]),
        hh_adults=np.ones(n),
        policy_labels=electorate.policy_labels,
        source=(
            "TOY moment-matched Gaussian (impact/log-income mean+cov "
            f"matched to: {electorate.source})"
        ),
    )


def homogeneous_toy(
    *,
    margin: float,
    n: int = 100_000,
    base_income: float = 70_000.0,
    labels: tuple[str, str] = ("Policy A", "Policy B"),
) -> Electorate:
    """The old model's world: every voter has an identical stake.

    Policy A is truly worth ``+margin/2`` dollars a year to every household
    and Policy B ``−margin/2``, so A is unanimously better and the only
    obstacle to electing it is perception. There is no heterogeneity to
    average over — accuracy hits every voter identically, which is what
    makes this world's threshold behavior so sharp.
    """
    if margin <= 0:
        raise ValueError("margin must be positive")
    deltas = np.column_stack([np.full(n, margin / 2.0), np.full(n, -margin / 2.0)])
    return Electorate(
        deltas=deltas,
        weights=np.ones(n),
        base_income=np.full(n, base_income),
        hh_adults=np.ones(n),
        policy_labels=labels,
        source=f"TOY homogeneous stakes (every voter ±${margin / 2:,.0f})",
    )
