"""A one-dimensional redistribution axis, with preferences over outcomes.

Everywhere else in this package a voter's preferences live over *policies*
and the model asks whether elections rank two of them correctly. This
module models the other half of the problem: a continuous policy dial, and
actors who care about the *outcome* it produces and hold beliefs about how
the dial moves that outcome.

**The dial.** Position ``t`` in [0, 1] raises every federal income tax
bracket rate by ``t x 10`` points and returns the revenue as an equal
per-adult transfer — the linear-tax-plus-demogrant axis, priced on
measured household incidence. Gross incidence is interpolated linearly in
``t`` from one engine-computed endpoint, an approximation measured at the
midpoint (``docs/results/axis_linearity_validation.json``). The *outcome*
is not linear: inequality falls along the axis at a decreasing rate,
because the transfer is flat while the tax base is concentrated.

**Preferences over outcomes.** An :class:`OutcomeType` has an ideal Gini
``target_gini``, a price ``dollars_per_gini_point`` on missing it, and a
selfish weight on their own household's dollars — so utility stays in
dollars, the same scale the rest of the package uses::

    U_i(t) = s_i * own_i(t)  -  (1 - s_i) * k_i * 100 * |G_hat_i(t) - g*_i|

Single-peaked in ``t`` whenever the outcome moves monotonically, so ideal
points are interior and heterogeneous without importing an efficiency cost
the engine does not model.

**Beliefs about the mapping.** ``belief_slope`` is the type's estimate of
how far policy moves the outcome, as a multiple of the truth::

    G_hat(t) = G(0) + lambda * (G(t) - G(0))

``lambda = 1`` is correct, ``lambda < 1`` underestimates the policy's
reach, ``lambda = 0`` believes it does nothing. Own-stake beliefs work the
same way: a voter's perceived own stake is ``t * (own_i + eps_i)`` with one
draw ``eps_i ~ N(bias, sigma^2)`` per voter — a misperceived *gradient*
rather than independent noise per position, so a voter's perceived utility
curve stays coherent across the continuum and the noise between two
positions scales with how far apart they are.

Both are labeled assumptions, and both are what a perception survey
elicits: "if rates rose ten points and the money came back as a check,
what would happen to inequality, and to you?"
"""

from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from democrasim.electorate import Electorate, FloatArray
from democrasim.perception import _phi
from democrasim.strategic import _multinomial_win
from democrasim.welfare import Isoelastic, WelfareMetric, apply_financing

#: Committed artifact carrying the axis endpoint.
AXIS_ARTIFACT_STEM = "us_2026_redistribution"


def weighted_gini(values: FloatArray, weights: FloatArray) -> float:
    """Weighted Gini coefficient of ``values``.

    Uses the covariance-free cumulative form, exact for weighted samples.
    Callers are responsible for flooring non-positive values; Gini is
    ill-behaved when the total is not positive.
    """
    order = np.argsort(values)
    v = np.asarray(values, dtype=np.float64)[order]
    w = np.asarray(weights, dtype=np.float64)[order]
    cumulative_weight = np.cumsum(w)
    cumulative_value = np.cumsum(v * w)
    total_weight = cumulative_weight[-1]
    total_value = cumulative_value[-1]
    if total_value <= 0 or total_weight <= 0:
        raise ValueError("weighted Gini needs positive total value and weight")
    previous = np.concatenate([[0.0], cumulative_value[:-1]])
    return float(
        1.0 - (w * (previous + cumulative_value)).sum() / (total_value * total_weight)
    )


@dataclass(frozen=True)
class RedistributionAxis:
    """The measured tax-and-transfer dial, plus the outcomes along it.

    Rows are voting-age adults, as everywhere else; ``gross_full`` is the
    adult's *household* gross net-income change at ``t = 1``, negative for
    households that pay more than the transfer returns.
    """

    gross_full: FloatArray
    weights: FloatArray
    base_income: FloatArray
    hh_adults: FloatArray
    income_floor: float = 1_000.0
    grid_points: int = 41
    source: str = "measured redistribution axis"

    @classmethod
    def from_artifact(
        cls, path: Path | str | None = None, **kwargs
    ) -> RedistributionAxis:
        """Load the committed axis artifact."""
        from democrasim.data import _data_dir

        path = Path(path) if path else _data_dir() / f"{AXIS_ARTIFACT_STEM}.parquet"
        frame = pd.read_parquet(path)
        return cls(
            gross_full=frame["delta_axis_full"].to_numpy(dtype=np.float64),
            weights=frame["weight"].to_numpy(dtype=np.float64),
            base_income=frame["base_income"].to_numpy(dtype=np.float64),
            hh_adults=frame["hh_adults"].to_numpy(dtype=np.float64),
            source=f"{AXIS_ARTIFACT_STEM} (engine-computed, financed per adult)",
            **kwargs,
        )

    @cached_property
    def net_full(self) -> FloatArray:
        """Per-adult net stake at ``t = 1``, after returning the revenue.

        Financing runs through the same ``apply_financing`` path as the rest
        of the package; because this policy *raises* revenue its per-adult
        levy is negative, which is the transfer.
        """
        gross = Electorate(
            deltas=np.column_stack([self.gross_full, np.zeros_like(self.gross_full)]),
            weights=self.weights,
            base_income=self.base_income,
            hh_adults=self.hh_adults,
            policy_labels=("axis endpoint", "status quo"),
            source=self.source,
        )
        return apply_financing(gross, "per_capita").deltas[:, 0].copy()

    @property
    def positions(self) -> FloatArray:
        return np.linspace(0.0, 1.0, self.grid_points)

    def net_deltas(self, t: float) -> FloatArray:
        """Per-adult net stake at position ``t`` (exactly linear in t)."""
        return t * self.net_full

    def incomes(self, t: float) -> FloatArray:
        """Household net income at ``t``, floored for the outcome metrics."""
        return np.maximum(self.base_income + self.net_deltas(t), self.income_floor)

    @cached_property
    def household_weights(self) -> FloatArray:
        """Adult-row weights that count each household once."""
        return self.weights / self.hh_adults

    def gini(self, t: float) -> float:
        """Gini of household net income at ``t``, counting households once."""
        return weighted_gini(self.incomes(t), self.household_weights)

    def ede(self, t: float, metric: WelfareMetric | None = None) -> float:
        """Change in equally-distributed-equivalent income at ``t``, dollars."""
        metric = metric if metric is not None else Isoelastic()
        electorate = Electorate(
            deltas=np.column_stack([self.net_deltas(t), np.zeros_like(self.net_full)]),
            weights=self.weights,
            base_income=self.base_income,
            hh_adults=self.hh_adults,
            policy_labels=("position", "status quo"),
            source=self.source,
        )
        return float(metric.dollar_equivalent(electorate)[0])

    @cached_property
    def gini_curve(self) -> FloatArray:
        """Gini at every grid position."""
        return np.array([self.gini(float(t)) for t in self.positions])

    def ede_curve(self, metric: WelfareMetric | None = None) -> FloatArray:
        return np.array([self.ede(float(t), metric) for t in self.positions])

    def own_curve(self, household_index: int) -> FloatArray:
        """One household's net stake at every grid position."""
        return self.positions * float(self.net_full[household_index])


@dataclass(frozen=True)
class OutcomeType:
    """An actor with an ideal outcome, a price on it, and a belief.

    Args:
        share: Population share (voter mixtures sum to 1).
        selfish_weight: ``s`` — weight on own household dollars; the rest
            goes to the outcome term.
        target_gini: ``g*`` — the Gini this actor wants. Values at or below
            the axis's reachable minimum make the preference monotone
            (always more redistribution); interior targets are single-peaked.
        dollars_per_gini_point: ``k`` — dollars per Gini *point* (0.01) of
            distance from the target. The exchange rate between the two
            motives.
        belief_slope: ``lambda`` — believed policy effect on the outcome as
            a multiple of the truth. 1 is correct; 0 believes policy does
            nothing.
        own_bias: Perceived own stake gradient shift, dollars at ``t = 1``.
        own_noise_sd: Standard deviation of the per-voter own-gradient
            error, dollars at ``t = 1``.
        household_index: Row whose own stake this actor holds. Required for
            any actor with ``selfish_weight > 0``.
    """

    share: float = 1.0
    selfish_weight: float = 0.0
    target_gini: float = 0.0
    dollars_per_gini_point: float = 500.0
    belief_slope: float = 1.0
    own_bias: float = 0.0
    own_noise_sd: float = 0.0
    household_index: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.share <= 1.0:
            raise ValueError("share must be in (0, 1]")
        if not 0.0 <= self.selfish_weight <= 1.0:
            raise ValueError("selfish_weight must be in [0, 1]")
        if not 0.0 <= self.target_gini <= 1.0:
            raise ValueError("target_gini must be in [0, 1]")
        if self.dollars_per_gini_point < 0:
            raise ValueError("dollars_per_gini_point must be >= 0")
        if self.own_noise_sd < 0:
            raise ValueError("own_noise_sd must be >= 0")
        if self.selfish_weight > 0 and self.household_index is None:
            raise ValueError("a selfish actor needs a household_index")

    def perceived_gini(self, axis: RedistributionAxis) -> FloatArray:
        """Believed Gini at every grid position."""
        truth = axis.gini_curve
        return truth[0] + self.belief_slope * (truth - truth[0])

    def outcome_value(self, axis: RedistributionAxis) -> FloatArray:
        """Believed outcome term at every position, in dollars (negative)."""
        distance = np.abs(self.perceived_gini(axis) - self.target_gini)
        return -self.dollars_per_gini_point * 100.0 * distance

    def true_outcome_value(self, axis: RedistributionAxis) -> FloatArray:
        """Outcome term the actor would report if their belief were correct."""
        return replace(self, belief_slope=1.0).outcome_value(axis)

    def ideal_position(self, axis: RedistributionAxis) -> float:
        """The position this type most wants, given beliefs and own stake."""
        return float(axis.positions[int(np.argmax(self.utility_curve(axis)))])

    def utility_curve(self, axis: RedistributionAxis) -> FloatArray:
        """Expected utility of each position, in dollars (no noise)."""
        outcome = (1.0 - self.selfish_weight) * self.outcome_value(axis)
        if self.selfish_weight == 0:
            return outcome
        own = axis.positions * (
            float(axis.net_full[self.household_index]) + self.own_bias
        )
        return self.selfish_weight * own + outcome


#: The reference voter: sociotropic, correct beliefs, wants zero inequality.
EGALITARIAN = OutcomeType()


def _vote_shares(
    axis: RedistributionAxis,
    types: tuple[OutcomeType, ...],
    left: int,
    right: int,
) -> tuple[float, float]:
    """Expected (prefer-left, indifferent) shares between two positions.

    Each voter's utility difference is normal — the own-stake gradient
    error is the only random term and it enters linearly — so the mixture
    closes in the same probit form the rest of the package uses.
    """
    positions = axis.positions
    gap = float(positions[left] - positions[right])
    share_weights = axis.weights / axis.weights.sum()
    prefer, indifferent = 0.0, 0.0
    for voter_type in types:
        outcome = voter_type.outcome_value(axis)
        societal = (1.0 - voter_type.selfish_weight) * float(
            outcome[left] - outcome[right]
        )
        s = voter_type.selfish_weight
        if s > 0:
            mean = s * gap * (axis.net_full + voter_type.own_bias) + societal
            sd = abs(s * gap) * voter_type.own_noise_sd
        else:
            mean = np.full(axis.weights.shape, societal)
            sd = 0.0
        if sd > 0:
            prefer += voter_type.share * float(share_weights @ _phi(mean / sd))
        else:
            prefer += voter_type.share * float(share_weights @ (mean > 0))
            indifferent += voter_type.share * float(share_weights @ (mean == 0))
    return prefer, indifferent


def win_probability_table(
    axis: RedistributionAxis,
    types: tuple[OutcomeType, ...],
    *,
    n_voters: int | None = 10_001,
) -> FloatArray:
    """P(the row position beats the column position), all pairs."""
    n = len(axis.positions)
    prefer = np.empty((n, n))
    indifferent = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            p, q = _vote_shares(axis, types, i, j)
            prefer[i, j], indifferent[i, j] = p, q
            prefer[j, i], indifferent[j, i] = 1.0 - p - q, q
    return _multinomial_win(prefer, indifferent, n_voters)


def median_ideal_position(
    axis: RedistributionAxis, types: tuple[OutcomeType, ...]
) -> float:
    """Weighted median of voters' (belief-mediated) ideal positions."""
    ideals, weights = [], []
    total = axis.weights.sum()
    for voter_type in types:
        if voter_type.selfish_weight == 0:
            ideals.append(voter_type.ideal_position(axis))
            weights.append(voter_type.share)
            continue
        # Selfish voters differ household by household: their ideal is a
        # corner set by the sign of their own net stake plus the outcome
        # term, so evaluate the full curve per household.
        outcome = (1.0 - voter_type.selfish_weight) * voter_type.outcome_value(axis)
        own = (
            np.outer(axis.net_full + voter_type.own_bias, axis.positions)
            * voter_type.selfish_weight
        )
        best = axis.positions[np.argmax(own + outcome[None, :], axis=1)]
        ideals.extend(best.tolist())
        weights.extend((voter_type.share * axis.weights / total).tolist())
    ideals = np.asarray(ideals)
    weights = np.asarray(weights)
    order = np.argsort(ideals)
    cumulative = np.cumsum(weights[order])
    return float(ideals[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


@dataclass(frozen=True)
class AxisCandidate:
    """A candidate choosing a position on the axis.

    Candidates hold the same preference family as voters
    (:class:`OutcomeType`), evaluated with correct beliefs unless the type
    says otherwise, plus an optional office rent in dollars.
    """

    label: str
    preferences: OutcomeType = field(default_factory=OutcomeType)
    office_rent: float = 0.0

    def utility_curve(self, axis: RedistributionAxis) -> FloatArray:
        return self.preferences.utility_curve(axis)


@dataclass(frozen=True)
class AxisEquilibrium:
    """A pure-strategy Nash profile of the one-dimensional position game."""

    position_1: float
    position_2: float
    p_win_1: float
    enacted: float


def axis_equilibria(
    axis: RedistributionAxis,
    candidate_1: AxisCandidate,
    candidate_2: AxisCandidate,
    *,
    types: tuple[OutcomeType, ...] = (EGALITARIAN,),
    n_voters: int | None = 10_001,
    win_table: FloatArray | None = None,
) -> list[AxisEquilibrium]:
    """Every pure Nash profile of the position game, by exhaustive check."""
    p_win = (
        win_probability_table(axis, types, n_voters=n_voters)
        if win_table is None
        else win_table
    )
    utility_1 = candidate_1.utility_curve(axis)
    utility_2 = candidate_2.utility_curve(axis)
    payoff_1 = (
        p_win * (utility_1[:, None] + candidate_1.office_rent)
        + (1.0 - p_win) * utility_1[None, :]
    )
    p_win_2 = 1.0 - p_win.T
    payoff_2 = (
        p_win_2 * (utility_2[:, None] + candidate_2.office_rent)
        + (1.0 - p_win_2) * utility_2[None, :]
    )
    tolerance = 1e-9
    best_1 = payoff_1 >= payoff_1.max(axis=0)[None, :] - tolerance
    best_2 = payoff_2 >= payoff_2.max(axis=0)[None, :] - tolerance
    positions = axis.positions
    equilibria = []
    for i, j in zip(*np.nonzero(best_1 & best_2.T), strict=True):
        p = float(p_win[i, j])
        equilibria.append(
            AxisEquilibrium(
                position_1=float(positions[i]),
                position_2=float(positions[j]),
                p_win_1=p,
                enacted=p * float(positions[i]) + (1 - p) * float(positions[j]),
            )
        )
    return equilibria


def demanded_position(
    axis: RedistributionAxis, target_gini: float, belief_slope: float
) -> float:
    """The position a sociotropic type demands, in closed form.

    A type believing the policy moves the outcome ``lambda`` times as far
    as it does needs the *true* Gini to reach
    ``G(0) + (g* - G(0)) / lambda`` before their belief reports the target
    — so attenuated beliefs (``lambda < 1``) demand more policy, saturating
    at the end of the axis.
    """
    if belief_slope <= 0:
        return 0.0
    truth = axis.gini_curve
    required = truth[0] + (target_gini - truth[0]) / belief_slope
    return float(axis.positions[int(np.argmin(np.abs(truth - required)))])
