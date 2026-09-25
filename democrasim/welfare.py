"""Welfare functionals and financing: ranking policies by measured impacts.

A welfare metric maps an electorate to one scalar per policy — the social
value of enacting that policy relative to current law. All metrics count
each household's dollars once (rows are adults; contributions are divided by
``hh_adults``), so utilitarian welfare of a policy equals the engine-computed
total household dollars it distributes.

Financing closes the budget. The impact deltas produced by a tax-benefit
engine are *gross*: a tax cut shows only winners because the engine does not
distribute the deficit. :func:`apply_financing` subtracts each policy's total
cost back out of household incomes under an explicit, labeled rule, making
every policy budget-neutral by construction. With balanced budgets,
utilitarian welfare is ~0 for every policy and the welfare ranking is purely
distributional — which is where inequality-averse metrics earn their keep.
"""

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from democrasim.electorate import Electorate, FloatArray

type FinancingMode = Literal["none", "per_capita", "proportional"]


@runtime_checkable
class WelfareMetric(Protocol):
    def per_policy(self, electorate: Electorate) -> FloatArray:
        """Welfare change of each policy vs current law. ``(n_policies,)``"""
        ...

    def dollar_equivalent(self, electorate: Electorate) -> FloatArray:
        """Each policy's societal value in dollars per household per year.

        The change in equally-distributed-equivalent (EDE) household income:
        the equal income that would yield the same welfare as the actual
        distribution. A strictly increasing transform of ``per_policy``, so
        the two rank policies identically — but dollars are commensurable
        with household stakes, which is what lets voters and candidates mix
        selfish and societal motives on one scale. ``(n_policies,)``
        """
        ...


def _welfare_is_degenerate(values: FloatArray) -> bool:
    """Whether the top two welfare values fail the relative-distinctness rule."""
    ordered = np.sort(values)
    spread = float(ordered[-1] - ordered[-2])
    scale = float(np.max(np.abs(values)))
    return spread <= 1e-9 * max(scale, 1e-12)


def welfare_optimal(
    metric: WelfareMetric,
    electorate: Electorate,
    *,
    require_distinct: bool = False,
) -> int:
    """Index of the policy the metric ranks highest.

    If ``require_distinct`` is true, reject a ranking whose top-two spread is
    no more than ``1e-9`` times the scale of the policy welfare values.
    """
    values = metric.per_policy(electorate)
    if require_distinct and _welfare_is_degenerate(values):
        raise ValueError(
            "degenerate welfare ranking: the top two policies are not distinct"
        )
    return int(np.argmax(values))


@dataclass(frozen=True)
class Utilitarian:
    """Total dollars, one per household: Σ weight·delta / hh_adults."""

    def per_policy(self, electorate: Electorate) -> FloatArray:
        return electorate.household_dollars()

    def dollar_equivalent(self, electorate: Electorate) -> FloatArray:
        """Mean household dollars: the utilitarian EDE change is the mean."""
        households = float((electorate.weights / electorate.hh_adults).sum())
        return electorate.household_dollars() / households


@dataclass(frozen=True)
class Isoelastic:
    """Isoelastic (CRRA) social welfare over household net incomes.

    Welfare change of a policy is ``Σ (weight/hh_adults)·[u(y₁) − u(y₀)]``
    with ``u(y) = y^(1−η)/(1−η)`` (``ln y`` at η=1). η=0 yields
    floor-censored dollars, equal to utilitarian dollars only when baseline
    and post-delta incomes stay above the floor. Higher η weights gains to
    poorer households more.

    Real microdata contains zero and negative net incomes, where CRRA
    utility is undefined, so incomes are floored: ``y₀ = max(y, floor)``
    and ``y₁ = max(y₀ + δ, floor)``. Flooring *before* the delta applies
    means gains to deep-negative-income households count (at the floor's
    high marginal utility) rather than silently vanishing, and losses
    cannot push utility-relevant income below the floor. The floor is an
    explicit modeling choice, not a data-cleaning step; results at high η
    can be sensitive to it.
    """

    eta: float = 1.0
    income_floor: float = 1_000.0

    def __post_init__(self) -> None:
        if self.eta < 0:
            raise ValueError("eta must be >= 0")
        if self.income_floor <= 0:
            raise ValueError("income_floor must be positive")

    def _u(self, floored_income: FloatArray) -> FloatArray:
        if self.eta == 1.0:
            return np.log(floored_income)
        return floored_income ** (1.0 - self.eta) / (1.0 - self.eta)

    def per_policy(self, electorate: Electorate) -> FloatArray:
        share = electorate.weights / electorate.hh_adults
        floored = np.maximum(electorate.base_income, self.income_floor)
        base_utility = self._u(floored)
        out = np.empty(electorate.n_policies)
        for j in range(electorate.n_policies):
            reformed = np.maximum(floored + electorate.deltas[:, j], self.income_floor)
            out[j] = float(share @ (self._u(reformed) - base_utility))
        return out

    def _ede(self, floored_income: FloatArray, shares: FloatArray) -> float:
        """Equally-distributed-equivalent income: u⁻¹(mean utility)."""
        if self.eta == 1.0:
            return float(np.exp(shares @ np.log(floored_income)))
        mean_power = float(shares @ floored_income ** (1.0 - self.eta))
        return mean_power ** (1.0 / (1.0 - self.eta))

    def dollar_equivalent(self, electorate: Electorate) -> FloatArray:
        """ΔEDE per policy, in dollars per household per year.

        Uses the same income floor as :meth:`per_policy`; because the EDE is
        a strictly increasing transform of mean utility, the dollar values
        rank policies exactly as the welfare values do.
        """
        share = electorate.weights / electorate.hh_adults
        shares = share / share.sum()
        floored = np.maximum(electorate.base_income, self.income_floor)
        base = self._ede(floored, shares)
        out = np.empty(electorate.n_policies)
        for j in range(electorate.n_policies):
            reformed = np.maximum(floored + electorate.deltas[:, j], self.income_floor)
            out[j] = self._ede(reformed, shares) - base
        return out


def apply_financing(electorate: Electorate, mode: FinancingMode) -> Electorate:
    """Return an electorate whose deltas are net of financing each policy.

    Modes:
        none: Gross engine deltas, unchanged (deficits are invisible).
        per_capita: Every adult bears an equal share of each policy's total
            cost (a household of two adults bears two shares). A lump-sum
            levy — regressive relative to income.
        proportional: Each household bears a share of the cost proportional
            to its (nonnegative) baseline net income — a flat levy on income.

    Under both financing rules each policy's utilitarian total is zero by
    construction (up to float precision): the same dollars distributed are
    collected back.
    """
    if mode == "none":
        return electorate
    cost = electorate.household_dollars()  # (p,) dollars distributed
    if mode == "per_capita":
        levy_per_adult = cost / electorate.population
        # Household burden = adults in household × per-adult levy.
        burden = np.outer(electorate.hh_adults, levy_per_adult)
    elif mode == "proportional":
        income = np.maximum(electorate.base_income, 0.0)
        income_mass = float((electorate.weights / electorate.hh_adults) @ income)
        if income_mass <= 0:
            raise ValueError("no positive base income to finance against")
        rates = cost / income_mass
        burden = np.outer(income, rates)
    else:
        raise ValueError(f"unknown financing mode: {mode!r}")
    return electorate.with_deltas(electorate.deltas - burden, note=f"financing={mode}")
