"""Synthetic comparator electorates — clearly labeled, never real data.

These exist to answer one question in the findings: *what did moving from
invented to measured impact distributions change?* Every electorate built
here carries a ``TOY`` source string and must be presented as synthetic.

Two comparators:

- :func:`moment_matched_toy` — multivariate-Gaussian impacts matched to the
  measured electorate's weighted mean vector and covariance matrix. Any
  difference between its results and the measured results is attributable
  purely to distribution *shape* (skew, tails, the mass of voters with no
  stake) — the moments are identical by construction.
- :func:`homogeneous_toy` — the old model's implicit world: every voter has
  the same stake in the outcome, so heterogeneity comes only from perception
  noise. This is what "candidate A's economic value is 1.0, B's is 0.8"
  amounted to.
"""

import numpy as np

from democrasim.electorate import Electorate, FloatArray


def _matched_lognormal_income(
    base_income: FloatArray,
    weights: FloatArray,
    n: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Lognormal incomes matched to the weighted mean and sd of the source.

    Incomes below $1 are floored before matching — the lognormal is a
    smooth stand-in for the toy world, not a model of the real left tail.
    """
    y = np.maximum(base_income, 1.0)
    mean = float(np.average(y, weights=weights))
    var = float(np.average((y - mean) ** 2, weights=weights))
    sigma2 = np.log(1.0 + var / mean**2)
    mu = np.log(mean) - sigma2 / 2.0
    return rng.lognormal(mu, np.sqrt(sigma2), size=n)


def moment_matched_toy(
    electorate: Electorate,
    rng: np.random.Generator,
    *,
    n: int | None = None,
) -> Electorate:
    """Gaussian world with the measured electorate's first two moments.

    Impacts are drawn iid from a multivariate normal whose mean vector and
    covariance matrix equal the weighted moments of ``electorate.deltas``.
    Weights are uniform and every voter is their household's only adult, as
    the old model implicitly assumed.
    """
    n = electorate.n_voters if n is None else n
    mean = np.average(electorate.deltas, axis=0, weights=electorate.weights)
    cov = np.cov(electorate.deltas.T, aweights=electorate.weights, ddof=0)
    deltas = rng.multivariate_normal(mean, cov, size=n)
    return Electorate(
        deltas=deltas,
        weights=np.ones(n),
        base_income=_matched_lognormal_income(
            electorate.base_income, electorate.weights, n, rng
        ),
        hh_adults=np.ones(n),
        policy_labels=electorate.policy_labels,
        source=(
            f"TOY moment-matched Gaussian (mean/cov matched to: {electorate.source})"
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
