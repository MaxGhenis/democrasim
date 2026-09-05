"""Heterogeneous preferences: voters who mix self-interest with societal views.

The baseline model votes pure perceived self-interest. This module gives
every voter their own welfare function on one dollar scale::

    utility = s · (perceived own household delta)
            + (1 − s) · (perceived societal value)

where ``s`` is the voter's selfish weight and the societal value is the
policy's change in equally-distributed-equivalent (EDE) household income
under the voter's own inequality aversion η — dollars in both terms, so the
mix needs no normalization. ``s = 1`` recovers the baseline exactly;
``s = 0`` is a purely sociotropic voter; interior ``s`` trades own dollars
against society-wide dollars-per-household one for one.

Both components are *perceived*: the own component passes through the same
:class:`~democrasim.perception.LinearGaussianPerception` as everywhere else,
and the societal component carries its own noise and bias. A population is a
mixture of :class:`VoterType` s — shares, selfish weights, inequality
aversions, and information quality can all differ by type.

:class:`MixedMotivePerception` satisfies the ``PerceptionModel`` protocol, so
every voting rule, election harness, and sweep runs unchanged on mixed-motive
electorates; the perceived matrix simply means "perceived utility" instead of
"perceived own delta". Preferences join perception as labeled assumptions —
nothing here is measured.
"""

from dataclasses import dataclass, field, replace
from math import sqrt

import numpy as np

from democrasim.electorate import Electorate, FloatArray
from democrasim.perception import LinearGaussianPerception, _phi
from democrasim.welfare import Isoelastic


def _bias_vector(bias: float | tuple[float, ...], n_policies: int) -> FloatArray:
    out = np.asarray(bias, dtype=np.float64)
    if out.ndim == 0:
        return np.full(n_policies, float(out))
    if out.shape != (n_policies,):
        raise ValueError(f"bias has {out.shape[0]} entries for {n_policies} policies")
    return out


@dataclass(frozen=True)
class VoterType:
    """One preference-and-information type in a mixture electorate.

    Args:
        share: Population share of this type (shares sum to 1 across a
            mixture).
        selfish_weight: ``s`` — the dollar weight on the voter's own
            household delta; ``1 − s`` goes to the societal value. 1 is the
            baseline self-interested voter; 0 is purely sociotropic.
        eta: Inequality aversion of the voter's *own* societal view (the η
            of the :class:`~democrasim.welfare.Isoelastic` EDE they care
            about). Voters can disagree about values, not just facts.
        societal_noise_sd: Perception noise on the societal value, in
            dollars per household per year. 0 means the type knows each
            policy's true EDE change.
        societal_bias: Systematic misperception of the societal value;
            scalar or one value per policy, like perception bias.
        own_noise_sd: Optional override of the shared own-perception noise
            for this type (an informed or uninformed minority). ``None``
            uses the mixture's shared own model.
    """

    share: float
    selfish_weight: float = 1.0
    eta: float = 1.0
    societal_noise_sd: float = 0.0
    societal_bias: float | tuple[float, ...] = 0.0
    own_noise_sd: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.share <= 1.0:
            raise ValueError("share must be in (0, 1]")
        if not 0.0 <= self.selfish_weight <= 1.0:
            raise ValueError("selfish_weight must be in [0, 1]")
        if self.societal_noise_sd < 0:
            raise ValueError("societal_noise_sd must be >= 0")
        if self.own_noise_sd is not None and self.own_noise_sd < 0:
            raise ValueError("own_noise_sd must be >= 0")


@dataclass(frozen=True)
class MixedMotivePerception:
    """A mixture of :class:`VoterType` s over a shared own-stake perception.

    Each voter draws a type (probability = share), perceives their own
    household delta through ``own`` (with the type's ``own_noise_sd``
    override, if any), perceives each policy's societal EDE change with the
    type's societal noise and bias, and mixes the two dollar quantities with
    the type's selfish weight. The societal value is computed on the
    electorate being perceived — a sampled electorate's voters judge the
    society that is actually voting.

    ``income_floor`` is passed to the Isoelastic EDE (same convention as the
    welfare metrics).
    """

    own: LinearGaussianPerception = field(default_factory=LinearGaussianPerception)
    types: tuple[VoterType, ...] = (VoterType(share=1.0),)
    income_floor: float = 1_000.0

    def __post_init__(self) -> None:
        if not self.types:
            raise ValueError("at least one voter type is required")
        total = sum(t.share for t in self.types)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"type shares must sum to 1, got {total}")

    @classmethod
    def homogeneous(
        cls,
        *,
        own: LinearGaussianPerception | None = None,
        selfish_weight: float = 1.0,
        eta: float = 1.0,
        societal_noise_sd: float = 0.0,
        societal_bias: float | tuple[float, ...] = 0.0,
        income_floor: float = 1_000.0,
    ) -> MixedMotivePerception:
        """One-type population: every voter shares the same mixed motive."""
        return cls(
            own=own if own is not None else LinearGaussianPerception(),
            types=(
                VoterType(
                    share=1.0,
                    selfish_weight=selfish_weight,
                    eta=eta,
                    societal_noise_sd=societal_noise_sd,
                    societal_bias=societal_bias,
                ),
            ),
            income_floor=income_floor,
        )

    def societal_values(self, electorate: Electorate) -> dict[float, FloatArray]:
        """True ΔEDE per policy for each distinct η among the types."""
        return {
            eta: Isoelastic(eta=eta, income_floor=self.income_floor).dollar_equivalent(
                electorate
            )
            for eta in sorted({t.eta for t in self.types})
        }

    def _own_model(self, voter_type: VoterType) -> LinearGaussianPerception:
        if voter_type.own_noise_sd is None:
            return self.own
        return replace(self.own, noise_sd=voter_type.own_noise_sd)

    def perceive(self, electorate: Electorate, rng: np.random.Generator) -> FloatArray:
        shares = np.array([t.share for t in self.types])
        assignment = rng.choice(len(self.types), size=electorate.n_voters, p=shares)
        societal_true = self.societal_values(electorate)
        utilities = np.empty_like(electorate.deltas)
        for k, voter_type in enumerate(self.types):
            idx = np.flatnonzero(assignment == k)
            if len(idx) == 0:
                continue
            own = self._own_model(voter_type).perceive(electorate.take(idx), rng)
            societal = societal_true[voter_type.eta] + _bias_vector(
                voter_type.societal_bias, electorate.n_policies
            )
            societal = np.broadcast_to(societal, own.shape)
            if voter_type.societal_noise_sd > 0:
                societal = societal + rng.normal(
                    0.0, voter_type.societal_noise_sd, size=own.shape
                )
            s = voter_type.selfish_weight
            utilities[idx] = s * own + (1.0 - s) * societal
        return utilities

    def vote_probabilities(
        self, electorate: Electorate
    ) -> tuple[FloatArray, FloatArray]:
        """Exact per-voter P(prefer policy 0) and P(indifferent), two policies.

        The mixture is exact here — no type assignment is drawn. For voter
        ``i`` of type ``t`` the utility margin is normal with mean
        ``s·(k·mᵢ + Δb_own) + (1−s)·(ΔEDE_η + Δb_soc)`` and standard
        deviation ``√2·√(s²σ_own² + (1−s)²σ_soc²)``; degenerate types vote
        the sign of the mean and are indifferent at zero.
        """
        if electorate.n_policies != 2:
            raise ValueError("vote probabilities are defined for two policies")
        margins = electorate.margins
        societal_true = self.societal_values(electorate)
        p_first = np.zeros(electorate.n_voters)
        p_indifferent = np.zeros(electorate.n_voters)
        for voter_type in self.types:
            own = self._own_model(voter_type)
            own_bias = _bias_vector(own.bias, 2)
            soc_bias = _bias_vector(voter_type.societal_bias, 2)
            societal = societal_true[voter_type.eta]
            s = voter_type.selfish_weight
            mean = s * (own.attenuation * margins + own_bias[0] - own_bias[1]) + (
                1.0 - s
            ) * (societal[0] - societal[1] + soc_bias[0] - soc_bias[1])
            sd = sqrt(
                s**2 * own.noise_sd**2
                + (1.0 - s) ** 2 * voter_type.societal_noise_sd**2
            ) * sqrt(2.0)
            if sd > 0:
                p_first += voter_type.share * _phi(mean / sd)
            else:
                p_first += voter_type.share * (mean > 0)
                p_indifferent += voter_type.share * (mean == 0)
        return p_first, p_indifferent
