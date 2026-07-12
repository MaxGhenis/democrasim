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
  electorate) and a societal component (the position's change in
  equally-distributed-equivalent household income under the candidate's
  own welfare lens), plus an optional office rent. Every term is dollars
  per year, so the selfish weight trades own dollars against
  society-wide dollars per household with no normalization.
- **Voters.** The electorate defaults to the baseline self-interested
  voter, but any mixture of :class:`~democrasim.preferences.VoterType` s
  — heterogeneous selfish weights, inequality aversions, and information
  quality — slots into the same tables via ``voter_types``. Every actor
  in the game, candidate or voter, draws its utility from one dollar
  family.
- **Election.** For self-interested voters the probability candidate 1
  wins depends only on the *difference* of positions, through the same
  probit-and-normal-approximation machinery as
  :func:`democrasim.analytic_plurality_curve`, so a (2G−1)² difference
  table prices every one of the G⁴ profile comparisons. Sociotropic
  components add a position-pair-specific term (societal value is not a
  function of the difference); those tables cost G⁴ evaluations, exact
  either way.

Everything here inherits the fixed-platform model's labeled assumptions
(perception, financing, electorate size) and adds two more: the candidate
objective specification and, optionally, the voter-preference mixture.
"""

from dataclasses import dataclass, field
from math import sqrt

import numpy as np

from democrasim.electorate import Electorate, FloatArray
from democrasim.perception import _phi
from democrasim.preferences import VoterType
from democrasim.welfare import Isoelastic, WelfareMetric, apply_financing

#: The baseline electorate: everyone votes pure perceived self-interest.
_SELF_INTERESTED = (VoterType(share=1.0, selfish_weight=1.0),)


def _multinomial_win(
    p_first: FloatArray, p_abstain: FloatArray, n_voters: int | None
) -> FloatArray:
    """P(the first option wins) from expected vote shares, elementwise.

    Normal approximation to the vote-count difference at finite
    ``n_voters``; a step function of the expected-share gap in the
    large-population limit. All-abstain cells (including both candidates
    at the same position) are exact ties and return 0.5.
    """
    p_first = np.asarray(p_first, dtype=np.float64)
    p_abstain = np.asarray(p_abstain, dtype=np.float64)
    active = 1.0 - p_abstain
    gap = 2.0 * p_first - active  # p_first − p_second
    step = np.where(gap > 0, 1.0, np.where(gap < 0, 0.0, 0.5))
    if n_voters is None:
        out = step
    else:
        variance = n_voters * (active - gap**2)
        positive = variance > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(positive, n_voters * gap / np.sqrt(variance), 0.0)
        out = np.where(positive, _phi(z), step)
    return np.where(active == 0, 0.5, out)


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
        p_first = float(weights @ _phi(margins / (sigma * sqrt(2.0)))) / total
        p_abstain = 0.0
    else:
        p_first = float(weights[margins > 0].sum()) / total
        p_abstain = float(weights[margins == 0].sum()) / total
    return float(_multinomial_win(np.float64(p_first), np.float64(p_abstain), n_voters))


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

    def _position_electorate(self, alpha: float, beta: float) -> Electorate:
        zeros = np.zeros_like(self.net_a)
        return Electorate(
            deltas=np.column_stack([self.net_deltas(alpha, beta), zeros]),
            weights=self.weights,
            base_income=self.base_income,
            hh_adults=self.hh_adults,
            policy_labels=("position", "status quo"),
            source="strategic policy space (financed, interpolated)",
        )

    def societal_welfare_grid(self, metric: WelfareMetric) -> FloatArray:
        """metric's welfare change for every grid position (vs status quo)."""
        return np.array(
            [
                metric.per_policy(self._position_electorate(alpha, beta))[0]
                for alpha, beta in self.positions
            ]
        )

    def societal_dollar_grid(self, metric: WelfareMetric) -> FloatArray:
        """Each position's societal value in dollars per household per year.

        The change in equally-distributed-equivalent household income vs
        the status quo (``metric.dollar_equivalent``); ranks positions
        exactly as :meth:`societal_welfare_grid` does, on a scale
        commensurable with household stakes.
        """
        return np.array(
            [
                metric.dollar_equivalent(self._position_electorate(alpha, beta))[0]
                for alpha, beta in self.positions
            ]
        )

    def welfare_optimal_position(self, metric: WelfareMetric) -> tuple[float, float]:
        grid = self.societal_welfare_grid(metric)
        return self.positions[int(np.argmax(grid))]


@dataclass(frozen=True)
class Candidate:
    """A policy-motivated candidate with a selfish and a societal component.

    The objective over enacted positions, in dollars per year, is
    ``selfish_weight · Self(p) + (1 − selfish_weight) · Soc(p)``, where
    Self is the candidate's own household net delta at ``p`` (candidates
    are rows of the electorate) and Soc is the position's change in
    equally-distributed-equivalent household income under ``societal`` —
    the candidate's own welfare lens, e.g. ``Isoelastic(eta)``. Both
    components are dollars, so at ``selfish_weight = 0.5`` the candidate
    trades a dollar of their own for a dollar of society-wide EDE income
    per household. ``office_rent`` adds a fixed dollar win bonus — set it
    large with ``selfish_weight = 0`` and a flat societal metric to
    recover pure office-seeking.
    """

    label: str
    household_index: int
    selfish_weight: float = 0.5
    societal: WelfareMetric = field(default_factory=Isoelastic)
    office_rent: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.selfish_weight <= 1.0:
            raise ValueError("selfish_weight must be in [0, 1]")


def objective_grid(space: PolicySpace, candidate: Candidate) -> FloatArray:
    """The candidate's utility of each enacted grid position, in dollars."""
    row = candidate.household_index
    selfish = np.array(
        [
            alpha * space.net_a[row] + beta * space.net_b[row]
            for alpha, beta in space.positions
        ]
    )
    societal = space.societal_dollar_grid(candidate.societal)
    return (
        candidate.selfish_weight * selfish + (1.0 - candidate.selfish_weight) * societal
    )


@dataclass(frozen=True)
class Equilibrium:
    """A pure-strategy Nash profile of the position game."""

    position_1: tuple[float, float]
    position_2: tuple[float, float]
    p_win_1: float
    utility_1: float
    utility_2: float


def _cloud(space: PolicySpace) -> tuple[FloatArray, FloatArray, FloatArray]:
    if space.comp_a is not None:
        return space.comp_a, space.comp_b, space.comp_weights
    return space.net_a, space.net_b, space.weights


def _difference_indices(space: PolicySpace) -> tuple[np.ndarray, np.ndarray]:
    """Index maps from position pairs (i, j) into difference tables."""
    n_alpha, n_beta = len(space.alphas), len(space.betas)
    ai, bi = np.divmod(np.arange(n_alpha * n_beta), n_beta)
    da = ai[:, None] - ai[None, :] + (n_alpha - 1)
    db = bi[:, None] - bi[None, :] + (n_beta - 1)
    return da, db


def _selfish_share_tables(
    space: PolicySpace, sigma_own: float
) -> tuple[FloatArray, FloatArray]:
    """Expected (vote-for-1, abstain) shares of a pure-selfish type.

    Own margins between positions p₁=(a₁,b₁) and p₂=(a₂,b₂) are
    ``(a₁−a₂)·net_a + (b₁−b₂)·net_b`` — a function of the difference only —
    so one (2G_α−1)×(2G_β−1) table prices every profile.
    """
    n_alpha, n_beta = len(space.alphas), len(space.betas)
    step_a = space.alphas[1] - space.alphas[0] if n_alpha > 1 else 1.0
    step_b = space.betas[1] - space.betas[0] if n_beta > 1 else 1.0
    cloud_a, cloud_b, cloud_w = _cloud(space)
    shares = cloud_w / cloud_w.sum()
    first = np.empty((2 * n_alpha - 1, 2 * n_beta - 1))
    abstain = np.zeros_like(first)
    for da in range(-(n_alpha - 1), n_alpha):
        for db in range(-(n_beta - 1), n_beta):
            margins = (da * step_a) * cloud_a + (db * step_b) * cloud_b
            if sigma_own > 0:
                value = float(shares @ _phi(margins / (sigma_own * sqrt(2.0))))
            else:
                value = float(shares[margins > 0].sum())
                abstain[da + n_alpha - 1, db + n_beta - 1] = float(
                    shares[margins == 0].sum()
                )
            first[da + n_alpha - 1, db + n_beta - 1] = value
    return first, abstain


def _vote_share_matrices(
    space: PolicySpace, sigma: float, voter_types: tuple[VoterType, ...]
) -> tuple[FloatArray, FloatArray]:
    """Expected (vote-for-1, abstain) shares for every position pair.

    Mixes the :class:`VoterType` s exactly (shares are weights, not draws).
    Pure-selfish types ride the difference table; purely sociotropic types
    compare each pair's societal dollar values directly; interior types
    need the full pair-by-pair computation over the compressed stake cloud
    — exact in all three cases. Only scalar societal bias is supported
    here, and a scalar shifts every position equally, so it cancels from
    every comparison.
    """
    n_positions = len(space.alphas) * len(space.betas)
    da_idx, db_idx = _difference_indices(space)
    p_first = np.zeros((n_positions, n_positions))
    p_abstain = np.zeros((n_positions, n_positions))
    ede_grids: dict[float, FloatArray] = {}
    for voter_type in voter_types:
        if isinstance(voter_type.societal_bias, tuple):
            raise ValueError(
                "the position game supports only scalar societal bias "
                "(policies are grid positions, not a fixed pair) — and a "
                "scalar cancels from every pairwise comparison"
            )
        sigma_own = (
            voter_type.own_noise_sd if voter_type.own_noise_sd is not None else sigma
        )
        s = voter_type.selfish_weight
        sd = sqrt(2.0) * sqrt(
            s**2 * sigma_own**2 + (1.0 - s) ** 2 * voter_type.societal_noise_sd**2
        )
        if s == 1.0:
            first_d, abstain_d = _selfish_share_tables(space, sigma_own)
            p_first += voter_type.share * first_d[da_idx, db_idx]
            p_abstain += voter_type.share * abstain_d[da_idx, db_idx]
            continue
        if voter_type.eta not in ede_grids:
            ede_grids[voter_type.eta] = space.societal_dollar_grid(
                Isoelastic(eta=voter_type.eta)
            )
        ede = ede_grids[voter_type.eta]
        delta_ede = ede[:, None] - ede[None, :]
        if s == 0.0:
            if sd > 0:
                p_first += voter_type.share * _phi(delta_ede / sd)
            else:
                p_first += voter_type.share * (delta_ede > 0)
                p_abstain += voter_type.share * (delta_ede == 0)
            continue
        cloud_a, cloud_b, cloud_w = _cloud(space)
        shares = cloud_w / cloud_w.sum()
        alphas_flat = np.repeat(space.alphas, len(space.betas))
        betas_flat = np.tile(space.betas, len(space.alphas))
        first_k = np.empty((n_positions, n_positions))
        abstain_k = np.zeros_like(first_k)
        for i in range(n_positions):
            margins = (alphas_flat[i] - alphas_flat)[:, None] * cloud_a[None, :] + (
                betas_flat[i] - betas_flat
            )[:, None] * cloud_b[None, :]
            mu = s * margins + (1.0 - s) * delta_ede[i][:, None]
            if sd > 0:
                first_k[i] = _phi(mu / sd) @ shares
            else:
                first_k[i] = (mu > 0) @ shares
                abstain_k[i] = (mu == 0) @ shares
        p_first += voter_type.share * first_k
        p_abstain += voter_type.share * abstain_k
    return p_first, p_abstain


def _payoff_matrices(
    space: PolicySpace,
    candidate_1: Candidate,
    candidate_2: Candidate,
    sigma: float,
    n_voters: int | None,
    voter_types: tuple[VoterType, ...] = _SELF_INTERESTED,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Expected utilities and P(candidate 1 wins), all G x G matrices."""
    utility_1 = objective_grid(space, candidate_1)
    utility_2 = objective_grid(space, candidate_2)

    # p_win[i, j] = P(candidate 1 wins with position i against position j)
    p_first, p_abstain = _vote_share_matrices(space, sigma, voter_types)
    p_win = _multinomial_win(p_first, p_abstain, n_voters)

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
    voter_types: tuple[VoterType, ...] = _SELF_INTERESTED,
) -> list[Equilibrium]:
    """Every pure-strategy Nash profile, by exhaustive verification.

    ``voter_types`` sets the electorate's preference mixture; the default
    is the baseline pure-self-interest voter. ``sigma`` is the shared
    own-stake perception noise (types can override it per
    ``VoterType.own_noise_sd``).
    """
    payoff_1, payoff_2, p_win = _payoff_matrices(
        space, candidate_1, candidate_2, sigma, n_voters, voter_types
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
    voter_types: tuple[VoterType, ...] = _SELF_INTERESTED,
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
        space, candidate_1, candidate_2, sigma, n_voters, voter_types
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
