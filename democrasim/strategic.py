"""Endogenous platforms: candidates pick positions to maximize objectives.

The fixed-platform model asks how elections grade two given policies. This
module closes the loop: two candidates each choose a position in a shared
policy space, knowing the opponent's choice, and the electorate votes
through the same perception model as everywhere else. Equilibria are
computed exactly (pure Nash by exhaustive enumeration on the grid), which
the measured data makes cheap:

- **Policy space.** Position ``(α, β)`` produces gross household deltas
  ``α·Δ_A + β·Δ_B`` — intensities on the two committed incidence vectors
  (α scales the CTC increment, β the top-rate cap depth). Under
  per-capita financing, net stakes are *exactly* linear in ``(α, β)``, so
  every position's net delta vector is a linear combination of two stored
  columns and no new engine runs are needed. Within-family linearity is a
  measured approximation (``docs/results/interpolation_validation.json``);
  the grid stays inside the validated ``[0, 1]`` range by default. The
  status quo ``(0, 0)`` is in the space.
- **Candidates.** Each candidate's objective mixes a selfish component
  (their own household's net delta — candidates are rows of the
  electorate) and a societal component (a welfare metric with their own
  inequality aversion), plus an optional office rent. Components are
  min–max normalized over the grid so dollars and welfare units mix on a
  common [0, 1] scale; the weights are the interpretable knobs.
- **Election.** The probability candidate 1 wins depends only on the
  *difference* of positions, through the same probit-and-normal-
  approximation machinery as :func:`democrasim.analytic_plurality_curve`.
  A (2G−1)² difference table therefore prices every one of the G⁴ profile
  comparisons, and exact best-response matrices follow.

Everything here inherits the fixed-platform model's labeled assumptions
(perception, financing, electorate size) and adds one more: the candidate
objective specification.
"""

from dataclasses import dataclass, field
from math import erf, sqrt

import numpy as np

from democrasim.electorate import Electorate, FloatArray
from democrasim.welfare import Isoelastic, WelfareMetric, apply_financing

_ERF = np.vectorize(erf)


def _phi(z: FloatArray) -> FloatArray:
    """Standard normal CDF, vectorized (mirrors democrasim.perception)."""
    return 0.5 * (1.0 + _ERF(np.asarray(z, dtype=np.float64) / sqrt(2.0)))


def win_probability(
    margins: FloatArray,
    weights: FloatArray,
    sigma: float,
    n_voters: int | None,
) -> float:
    """P(the first option wins) for per-voter margins, plurality voting.

    Same math as :func:`democrasim.analytic_plurality_curve`: per-voter
    probit vote probabilities, multinomial normal approximation at finite
    ``n_voters``, step function in the large-population limit. Exact ties
    in expected shares (including both candidates at the same position)
    return 0.5.
    """
    total = weights.sum()
    if sigma > 0:
        p_first_each = _phi(margins / (sigma * sqrt(2.0)))
        p_first = float(weights @ p_first_each) / total
        p_abstain = 0.0
    else:
        p_first = float(weights[margins > 0].sum()) / total
        p_abstain = float(weights[margins == 0].sum()) / total
    p_second = 1.0 - p_first - p_abstain
    gap = p_first - p_second
    if p_first + p_second == 0:
        return 0.5  # everyone abstains; identical enactments either way
    if n_voters is None:
        return 1.0 if gap > 0 else (0.5 if gap == 0 else 0.0)
    variance = n_voters * (p_first + p_second - gap**2)
    if variance <= 0:
        return 1.0 if gap > 0 else (0.5 if gap == 0 else 0.0)
    return float(0.5 * (1.0 + erf(n_voters * gap / sqrt(2.0 * variance))))


@dataclass(frozen=True)
class PolicySpace:
    """Shared 2D policy space spanned by the two measured incidence vectors."""

    net_a: FloatArray  # financed net deltas of the full Policy A, per adult
    net_b: FloatArray
    weights: FloatArray
    base_income: FloatArray
    hh_adults: FloatArray
    alphas: FloatArray  # grid of Policy-A intensities
    betas: FloatArray  # grid of Policy-B intensities
    # Cent-quantized (net_a, net_b) point cloud used for win-probability
    # tables; per-voter margin error is bounded by $0.01 per unit intensity.
    comp_a: FloatArray = field(repr=False, default=None)  # type: ignore[assignment]
    comp_b: FloatArray = field(repr=False, default=None)  # type: ignore[assignment]
    comp_weights: FloatArray = field(repr=False, default=None)  # type: ignore[assignment]

    @classmethod
    def from_gross_electorate(
        cls,
        gross: Electorate,
        *,
        grid_points: int = 21,
        max_intensity: float = 1.0,
    ) -> PolicySpace:
        """Build the space from the committed gross electorate.

        Per-capita financing is applied to the two endpoint policies once;
        linearity of the levy in the deltas makes every interior position's
        net stakes an exact linear combination of the two financed columns.
        """
        financed = apply_financing(gross, "per_capita")
        grid = np.linspace(0.0, max_intensity, grid_points)
        net_a = financed.deltas[:, 0].copy()
        net_b = financed.deltas[:, 1].copy()

        def quantize(values: FloatArray) -> FloatArray:
            return np.round(values, 2)  # cents everywhere: error <= $0.005

        pairs = np.column_stack([quantize(net_a), quantize(net_b)])
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        mass = np.bincount(inverse, weights=financed.weights)
        return cls(
            net_a=net_a,
            net_b=net_b,
            weights=financed.weights.copy(),
            base_income=financed.base_income.copy(),
            hh_adults=financed.hh_adults.copy(),
            alphas=grid,
            betas=grid.copy(),
            comp_a=unique[:, 0].copy(),
            comp_b=unique[:, 1].copy(),
            comp_weights=mass,
        )

    @property
    def positions(self) -> list[tuple[float, float]]:
        """All grid positions, row-major over (α, β)."""
        return [(float(a), float(b)) for a in self.alphas for b in self.betas]

    def position_index(self, alpha: float, beta: float) -> int:
        i = int(np.argmin(np.abs(self.alphas - alpha)))
        j = int(np.argmin(np.abs(self.betas - beta)))
        return i * len(self.betas) + j

    def net_deltas(self, alpha: float, beta: float) -> FloatArray:
        """Per-adult net stakes of position (α, β). Exactly linear."""
        return alpha * self.net_a + beta * self.net_b

    def societal_welfare_grid(self, metric: WelfareMetric) -> FloatArray:
        """metric's welfare change for every grid position (vs status quo)."""
        values = np.empty(len(self.alphas) * len(self.betas))
        zeros = np.zeros_like(self.net_a)
        for index, (alpha, beta) in enumerate(self.positions):
            electorate = Electorate(
                deltas=np.column_stack([self.net_deltas(alpha, beta), zeros]),
                weights=self.weights,
                base_income=self.base_income,
                hh_adults=self.hh_adults,
                policy_labels=("position", "status quo"),
                source="strategic policy space (financed, interpolated)",
            )
            values[index] = metric.per_policy(electorate)[0]
        return values

    def welfare_optimal_position(self, metric: WelfareMetric) -> tuple[float, float]:
        grid = self.societal_welfare_grid(metric)
        return self.positions[int(np.argmax(grid))]


@dataclass(frozen=True)
class Candidate:
    """A policy-motivated candidate with a selfish and a societal component.

    The objective over enacted positions is
    ``selfish_weight · Self̃(p) + (1 − selfish_weight) · Soc̃(p)``, where
    Self is the candidate's own household net delta at ``p``, Soc is
    ``societal.per_policy`` welfare, and tildes denote min–max
    normalization over the policy grid (dollars and welfare units are not
    commensurable; the normalized mix makes ``selfish_weight`` the
    interpretable knob). ``office_rent`` adds a fixed win bonus in the
    same normalized units — set it large with ``selfish_weight = 0`` and a
    flat societal metric to recover pure office-seeking.
    """

    label: str
    household_index: int
    selfish_weight: float = 0.5
    societal: WelfareMetric = field(default_factory=Isoelastic)
    office_rent: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.selfish_weight <= 1.0:
            raise ValueError("selfish_weight must be in [0, 1]")


def _normalize(values: FloatArray) -> FloatArray:
    spread = values.max() - values.min()
    if spread <= 0:
        return np.zeros_like(values)
    return (values - values.min()) / spread


def objective_grid(space: PolicySpace, candidate: Candidate) -> FloatArray:
    """The candidate's (normalized) utility of each enacted grid position."""
    row = candidate.household_index
    selfish = np.array(
        [
            alpha * space.net_a[row] + beta * space.net_b[row]
            for alpha, beta in space.positions
        ]
    )
    societal = space.societal_welfare_grid(candidate.societal)
    return candidate.selfish_weight * _normalize(selfish) + (
        1.0 - candidate.selfish_weight
    ) * _normalize(societal)


@dataclass(frozen=True)
class Equilibrium:
    """A pure-strategy Nash profile of the position game."""

    position_1: tuple[float, float]
    position_2: tuple[float, float]
    p_win_1: float
    utility_1: float
    utility_2: float


def _win_probability_table(
    space: PolicySpace, sigma: float, n_voters: int | None
) -> FloatArray:
    """P(candidate 1 wins) for every position *difference*.

    Margins between positions p₁=(a₁,b₁) and p₂=(a₂,b₂) are
    ``(a₁−a₂)·net_a + (b₁−b₂)·net_b`` — a function of the difference only —
    so one (2G_α−1)×(2G_β−1) table prices every profile.
    """
    n_alpha, n_beta = len(space.alphas), len(space.betas)
    step_a = space.alphas[1] - space.alphas[0] if n_alpha > 1 else 1.0
    step_b = space.betas[1] - space.betas[0] if n_beta > 1 else 1.0
    if space.comp_a is not None:
        cloud_a, cloud_b, cloud_w = space.comp_a, space.comp_b, space.comp_weights
    else:
        cloud_a, cloud_b, cloud_w = space.net_a, space.net_b, space.weights
    table = np.empty((2 * n_alpha - 1, 2 * n_beta - 1))
    for da in range(-(n_alpha - 1), n_alpha):
        for db in range(-(n_beta - 1), n_beta):
            margins = (da * step_a) * cloud_a + (db * step_b) * cloud_b
            table[da + n_alpha - 1, db + n_beta - 1] = win_probability(
                margins, cloud_w, sigma, n_voters
            )
    return table


def _payoff_matrices(
    space: PolicySpace,
    candidate_1: Candidate,
    candidate_2: Candidate,
    sigma: float,
    n_voters: int | None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Expected utilities and P(candidate 1 wins), all G x G matrices."""
    n_alpha, n_beta = len(space.alphas), len(space.betas)
    n_positions = n_alpha * n_beta
    win_table = _win_probability_table(space, sigma, n_voters)
    utility_1 = objective_grid(space, candidate_1)
    utility_2 = objective_grid(space, candidate_2)

    ai, bi = np.divmod(np.arange(n_positions), n_beta)
    # p_win[i, j] = P(candidate 1 wins with position i against position j)
    da = ai[:, None] - ai[None, :] + (n_alpha - 1)
    db = bi[:, None] - bi[None, :] + (n_beta - 1)
    p_win = win_table[da, db]

    payoff_1 = (
        p_win * (utility_1[:, None] + candidate_1.office_rent)
        + (1.0 - p_win) * utility_1[None, :]
    )
    # Candidate 2's win probability against position j of candidate 1 is
    # 1 − p_win[j, i] with roles transposed.
    p_win_2 = 1.0 - p_win.T
    payoff_2 = (
        p_win_2 * (utility_2[:, None] + candidate_2.office_rent)
        + (1.0 - p_win_2) * utility_2[None, :]
    )
    return payoff_1, payoff_2, p_win


def pure_nash_equilibria(
    space: PolicySpace,
    candidate_1: Candidate,
    candidate_2: Candidate,
    *,
    sigma: float,
    n_voters: int | None = 10_001,
) -> list[Equilibrium]:
    """Every pure-strategy Nash profile, by exhaustive verification."""
    payoff_1, payoff_2, p_win = _payoff_matrices(
        space, candidate_1, candidate_2, sigma, n_voters
    )
    best_1 = payoff_1.max(axis=0)  # best reply value against each opponent j
    best_2 = payoff_2.max(axis=0)
    positions = space.positions

    equilibria = []
    tol = 1e-12
    is_best_1 = payoff_1 >= best_1[None, :] - tol
    is_best_2 = payoff_2 >= best_2[None, :] - tol
    mutual = is_best_1 & is_best_2.T  # [i, j]: i best vs j AND j best vs i
    for i, j in zip(*np.nonzero(mutual), strict=True):
        equilibria.append(
            Equilibrium(
                position_1=positions[int(i)],
                position_2=positions[int(j)],
                p_win_1=float(p_win[i, j]),
                utility_1=float(payoff_1[i, j]),
                utility_2=float(payoff_2[j, i]),
            )
        )
    return equilibria


def iterated_best_response(
    space: PolicySpace,
    candidate_1: Candidate,
    candidate_2: Candidate,
    *,
    sigma: float,
    n_voters: int | None = 10_001,
    start_1: tuple[float, float] = (0.0, 0.0),
    start_2: tuple[float, float] = (0.0, 0.0),
    max_iterations: int = 200,
) -> dict:
    """Best-response dynamics from a starting profile (path + convergence).

    Complements :func:`pure_nash_equilibria` (which is exhaustive): the
    path shows which equilibrium dynamics select, or exposes a cycle when
    no pure equilibrium attracts.
    """
    payoff_1, payoff_2, _ = _payoff_matrices(
        space, candidate_1, candidate_2, sigma, n_voters
    )
    i = space.position_index(*start_1)
    j = space.position_index(*start_2)
    path = [(space.positions[i], space.positions[j])]
    seen = {(i, j)}
    for _ in range(max_iterations):
        i_new = int(np.argmax(payoff_1[:, j]))
        j_new = int(np.argmax(payoff_2[:, i_new]))
        path.append((space.positions[i_new], space.positions[j_new]))
        if (i_new, j_new) == (i, j):
            return {"converged": True, "cycle": False, "path": path}
        i, j = i_new, j_new
        if (i, j) in seen:
            return {"converged": False, "cycle": True, "path": path}
        seen.add((i, j))
    return {"converged": False, "cycle": False, "path": path}
