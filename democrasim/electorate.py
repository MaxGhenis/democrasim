"""The electorate: who votes, and what each policy would truly do to them.

An :class:`Electorate` is a table of voting-age adults. Each row carries the
*measured* per-household impact of every policy on the ballot — the annual
change in household net income, in dollars, as computed by a tax-benefit
microsimulation engine — plus the survey weight (how many real adults the row
represents) and enough household context to aggregate welfare without double
counting.

Two aggregation conventions matter throughout:

- **Voting** happens at the adult level. Every adult perceives and votes on
  their *household's* full dollar impact (both spouses see the same number).
- **Welfare** counts each household's dollars once. Because all adults in a
  household share one impact, welfare functionals divide each row's
  contribution by ``hh_adults`` so that summing over adult rows reproduces
  household-level totals exactly.
"""

from dataclasses import dataclass, field
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]

#: Dollar threshold under which an impact counts as "unaffected" in summaries.
AFFECTED_EPSILON = 1.0


def weighted_quantile(
    values: FloatArray, weights: FloatArray, quantiles: FloatArray | list[float]
) -> FloatArray:
    """Weighted quantiles of ``values`` (interpolated, inclusive definition)."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum = np.cumsum(weights) - 0.5 * weights
    cum /= weights.sum()
    return np.interp(np.asarray(quantiles, dtype=np.float64), cum, values)


@dataclass(frozen=True)
class Electorate:
    """A population of voting-age adults with measured policy impacts.

    Args:
        deltas: ``(n_voters, n_policies)`` — annual change in each voter's
            *household* net income under each policy, in dollars, relative to
            current law.
        weights: ``(n_voters,)`` — survey weight; the number of real adults
            each row represents. Uniform (all ones) for sampled or synthetic
            electorates.
        base_income: ``(n_voters,)`` — the voter's household net income under
            current law, in dollars. Used by inequality-averse welfare
            functionals and proportional financing.
        hh_adults: ``(n_voters,)`` — number of voting-age adults in the
            voter's household. Welfare functionals divide by this so each
            household's dollars are counted once.
        policy_labels: One generic label per policy column ("Policy A", ...).
            Never real candidates or parties.
        source: Provenance string describing where the impacts came from.
            Synthetic electorates must say so loudly (e.g. ``"TOY ..."``).
        demographics: Optional per-voter attributes (age, state, household
            size, ...) used by demographic-varying perception models and for
            reporting. Row-aligned with the arrays.
    """

    deltas: FloatArray
    weights: FloatArray
    base_income: FloatArray
    hh_adults: FloatArray
    policy_labels: tuple[str, ...]
    source: str
    demographics: pd.DataFrame | None = field(default=None)

    def __post_init__(self) -> None:
        deltas = np.atleast_2d(np.asarray(self.deltas, dtype=np.float64)).copy()
        weights = np.asarray(self.weights, dtype=np.float64).copy()
        base_income = np.asarray(self.base_income, dtype=np.float64).copy()
        hh_adults = np.asarray(self.hh_adults, dtype=np.float64).copy()

        n, p = deltas.shape
        if p < 2:
            raise ValueError("an election needs at least two policies")
        for name, arr in (
            ("weights", weights),
            ("base_income", base_income),
            ("hh_adults", hh_adults),
        ):
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
        if not np.all(np.isfinite(deltas)):
            raise ValueError("deltas must be finite")
        if not np.all(np.isfinite(base_income)):
            raise ValueError("base_income must be finite")
        if not np.all(weights > 0):
            raise ValueError("weights must be strictly positive")
        if not np.all(hh_adults >= 1):
            raise ValueError("hh_adults must be >= 1 for every voter")
        if len(self.policy_labels) != p:
            raise ValueError(
                f"got {len(self.policy_labels)} policy labels for {p} policies"
            )
        if self.demographics is not None and len(self.demographics) != n:
            raise ValueError("demographics must be row-aligned with the arrays")

        for arr in (deltas, weights, base_income, hh_adults):
            arr.setflags(write=False)
        object.__setattr__(self, "deltas", deltas)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "base_income", base_income)
        object.__setattr__(self, "hh_adults", hh_adults)
        object.__setattr__(self, "policy_labels", tuple(self.policy_labels))
        if self.demographics is not None:
            object.__setattr__(
                self, "demographics", self.demographics.reset_index(drop=True)
            )

    # ------------------------------------------------------------------ shape
    @property
    def n_voters(self) -> int:
        return self.deltas.shape[0]

    @property
    def n_policies(self) -> int:
        return self.deltas.shape[1]

    @property
    def population(self) -> float:
        """Total adult population represented (sum of weights)."""
        return float(self.weights.sum())

    @property
    def margins(self) -> FloatArray:
        """Per-voter true margin between the first two policies (δ₀ − δ₁)."""
        return self.deltas[:, 0] - self.deltas[:, 1]

    # -------------------------------------------------------------- operations
    def sample(self, n: int, rng: np.random.Generator) -> Self:
        """Draw a finite electorate of ``n`` adults, probability ∝ weight.

        Sampling is with replacement; the sampled electorate has uniform
        weights (one adult, one vote).
        """
        if n < 1:
            raise ValueError("sample size must be >= 1")
        probabilities = self.weights / self.weights.sum()
        idx = rng.choice(self.n_voters, size=n, replace=True, p=probabilities)
        return self.take(idx, uniform_weights=True)

    def take(self, idx: IntArray, *, uniform_weights: bool = False) -> Self:
        """Return the electorate restricted to rows ``idx`` (in order)."""
        return Electorate(
            deltas=self.deltas[idx],
            weights=np.ones(len(idx)) if uniform_weights else self.weights[idx],
            base_income=self.base_income[idx],
            hh_adults=self.hh_adults[idx],
            policy_labels=self.policy_labels,
            source=self.source,
            demographics=(
                None
                if self.demographics is None
                else self.demographics.iloc[idx].reset_index(drop=True)
            ),
        )

    def with_deltas(self, deltas: FloatArray, *, note: str = "") -> Self:
        """Return a copy with replaced impact matrix (e.g. after financing)."""
        return Electorate(
            deltas=deltas,
            weights=self.weights,
            base_income=self.base_income,
            hh_adults=self.hh_adults,
            policy_labels=self.policy_labels,
            source=self.source + (f" | {note}" if note else ""),
            demographics=self.demographics,
        )

    # ---------------------------------------------------------------- reporting
    def household_dollars(self) -> FloatArray:
        """Total dollars per policy, counting each household once. ``(p,)``"""
        share = self.weights / self.hh_adults
        return share @ self.deltas

    def impact_summary(self, epsilon: float = AFFECTED_EPSILON) -> pd.DataFrame:
        """Weighted description of each policy's impact distribution."""
        rows = []
        w = self.weights
        for j, label in enumerate(self.policy_labels):
            d = self.deltas[:, j]
            gaining = w[d > epsilon].sum() / w.sum()
            losing = w[d < -epsilon].sum() / w.sum()
            q10, q50, q90 = weighted_quantile(d, w, [0.10, 0.50, 0.90])
            rows.append(
                {
                    "policy": label,
                    "total_household_dollars_bn": self.household_dollars()[j] / 1e9,
                    "share_gaining": gaining,
                    "share_losing": losing,
                    "share_unaffected": 1.0 - gaining - losing,
                    "mean_delta": float(np.average(d, weights=w)),
                    "p10_delta": q10,
                    "median_delta": q50,
                    "p90_delta": q90,
                }
            )
        return pd.DataFrame(rows)
