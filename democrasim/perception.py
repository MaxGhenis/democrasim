"""Perception: the one labeled assumption in the model.

Everything else in Democrasim is measured — the households are survey
records, the policy impacts come from a tax-benefit engine. What nobody has
measured yet is *how accurately voters perceive their own stakes*. This
module quarantines that assumption behind a small interface so that a
parametric guess can be swapped for an empirically estimated misperception
distribution (from a planned survey) without touching anything downstream.

The workhorse is :class:`LinearGaussianPerception`::

    perceived = attenuation * true + bias + noise,  noise ~ N(0, noise_sd²)

which is exactly the model a survey regression of perceived on true impacts
would estimate (slope = attenuation, intercept = bias, residual sd = noise).

Accuracy is reported on a survey-comparable scale: the probability that a
voter correctly ranks the two policies for their own household
(:func:`ranking_accuracy`), which runs from 0.5 (coin flip) to 1.0 (perfect).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import erf, sqrt
from typing import Protocol, runtime_checkable

import numpy as np

from democrasim.electorate import Electorate, FloatArray


@runtime_checkable
class PerceptionModel(Protocol):
    """Maps true household impacts to what each voter believes they are."""

    def perceive(self, electorate: Electorate, rng: np.random.Generator) -> FloatArray:
        """Return an ``(n_voters, n_policies)`` array of perceived impacts."""
        ...


def _phi(z: FloatArray) -> FloatArray:
    """Standard normal CDF, vectorized, without a scipy dependency."""
    if z.size == 0:
        return np.empty(z.shape, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))


@dataclass(frozen=True)
class LinearGaussianPerception:
    """``perceived = attenuation * true + bias + N(0, noise_sd²)``.

    Noise is drawn iid per (voter, policy), so adults in the same household
    perceive the same household delta with independent errors;
    household-correlated errors are not modeled.

    Args:
        noise_sd: Standard deviation of idiosyncratic perception error, in
            dollars per year. 0 means voters know their exact impact.
        bias: Systematic misperception added to every voter's view of each
            policy, in dollars. A scalar applies to all policies; a sequence
            gives one value per policy (e.g. ``(0, 500)`` makes everyone see
            the second policy as $500/year better than it is).
        attenuation: Multiplicative compression of true stakes. 1 is
            faithful; values in (0, 1) mean voters under-react to large
            stakes; 0 means perceived stakes carry no signal at all.
    """

    noise_sd: float = 0.0
    bias: float | tuple[float, ...] = 0.0
    attenuation: float = 1.0

    def __post_init__(self) -> None:
        if self.noise_sd < 0:
            raise ValueError("noise_sd must be >= 0")
        if isinstance(self.bias, tuple) and not all(np.isfinite(b) for b in self.bias):
            raise ValueError("bias must be finite")

    def _bias_vector(self, n_policies: int) -> FloatArray:
        bias = np.asarray(self.bias, dtype=np.float64)
        if bias.ndim == 0:
            return np.full(n_policies, float(bias))
        if bias.shape != (n_policies,):
            raise ValueError(
                f"bias has {bias.shape[0]} entries for {n_policies} policies"
            )
        return bias

    def perceive(self, electorate: Electorate, rng: np.random.Generator) -> FloatArray:
        bias = self._bias_vector(electorate.n_policies)
        perceived = self.attenuation * electorate.deltas + bias
        if self.noise_sd > 0:
            perceived = perceived + rng.normal(
                0.0, self.noise_sd, size=electorate.deltas.shape
            )
        return perceived

    def analytic_ranking_accuracy(self, electorate: Electorate) -> FloatArray:
        """Per-voter P(correctly rank policy 0 vs policy 1); NaN if margin 0.

        For perceived margin ``M = k·m + (b₀ − b₁) + ε`` with
        ``ε ~ N(0, 2·noise_sd²)``, the probability of matching the true
        ranking is ``Φ(sign(m)·(k·m + b₀ − b₁) / (noise_sd·√2))``.
        """
        if electorate.n_policies != 2:
            raise ValueError("ranking accuracy is defined for two policies")
        bias = self._bias_vector(2)
        m = electorate.margins
        shifted = self.attenuation * m + (bias[0] - bias[1])
        signed = np.sign(m) * shifted
        out = np.full(m.shape, np.nan)
        nonzero = m != 0
        if self.noise_sd > 0:
            out[nonzero] = _phi(signed[nonzero] / (self.noise_sd * sqrt(2.0)))
        else:
            out[nonzero] = np.where(
                signed[nonzero] > 0, 1.0, np.where(signed[nonzero] < 0, 0.0, 0.5)
            )
        return out


@dataclass(frozen=True)
class GroupedPerception:
    """Demographic-varying perception: a different model per subgroup.

    Args:
        column: Name of the column in ``electorate.demographics`` to split on.
        models: Mapping from column value to the perception model applied to
            those voters.
        default: Model for voters whose value is not in ``models``.
    """

    column: str
    models: Mapping[object, PerceptionModel]
    default: PerceptionModel

    def _group_indices(
        self, electorate: Electorate
    ) -> list[tuple[PerceptionModel, np.ndarray]]:
        if electorate.demographics is None:
            raise ValueError("GroupedPerception needs electorate.demographics")
        if self.column not in electorate.demographics:
            raise ValueError(f"demographics has no column {self.column!r}")
        values = electorate.demographics[self.column].to_numpy()
        assigned = np.zeros(len(values), dtype=bool)
        groups: list[tuple[PerceptionModel, np.ndarray]] = []
        for value, model in self.models.items():
            mask = values == value
            groups.append((model, np.flatnonzero(mask)))
            assigned |= mask
        groups.append((self.default, np.flatnonzero(~assigned)))
        return groups

    def perceive(self, electorate: Electorate, rng: np.random.Generator) -> FloatArray:
        perceived = np.empty_like(electorate.deltas)
        for model, idx in self._group_indices(electorate):
            if len(idx) == 0:
                continue
            perceived[idx] = model.perceive(electorate.take(idx), rng)
        return perceived

    def analytic_ranking_accuracy(self, electorate: Electorate) -> FloatArray:
        out = np.full(electorate.n_voters, np.nan)
        for model, idx in self._group_indices(electorate):
            if len(idx) == 0:
                continue
            accuracy = getattr(model, "analytic_ranking_accuracy", None)
            if accuracy is None:
                raise TypeError(
                    f"{type(model).__name__} has no analytic ranking accuracy; "
                    "use ranking_accuracy(..., n_draws=...) instead"
                )
            out[idx] = accuracy(electorate.take(idx))
        return out


#: Voters who know their household's exact impact under every policy.
PERFECT_PERCEPTION = LinearGaussianPerception(noise_sd=0.0)


def ranking_accuracy(
    model: PerceptionModel,
    electorate: Electorate,
    *,
    rng: np.random.Generator | None = None,
    n_draws: int = 0,
) -> float:
    """Population mean probability of correctly ranking the two policies.

    Weighted over voters with a nonzero true margin (voters whose household
    is identically affected by both policies have no ranking to get right).
    Uses the model's analytic formula when available; otherwise estimates by
    Monte Carlo with ``n_draws`` perception draws (requires ``rng``).
    """
    if electorate.n_policies != 2:
        raise ValueError("ranking accuracy is defined for two policies")
    analytic = getattr(model, "analytic_ranking_accuracy", None)
    if analytic is not None:
        per_voter = analytic(electorate)
    else:
        if rng is None or n_draws < 1:
            raise ValueError(
                "model has no analytic ranking accuracy; pass rng and n_draws"
            )
        m = electorate.margins
        correct = np.zeros(electorate.n_voters)
        for _ in range(n_draws):
            perceived = model.perceive(electorate, rng)
            pm = perceived[:, 0] - perceived[:, 1]
            correct += np.where(
                np.sign(pm) == np.sign(m), 1.0, np.where(pm == 0, 0.5, 0.0)
            )
        per_voter = np.where(m != 0, correct / n_draws, np.nan)

    valid = ~np.isnan(per_voter)
    if not valid.any():
        raise ValueError("no voter has a nonzero margin between the policies")
    weights = electorate.weights[valid]
    return float(np.average(per_voter[valid], weights=weights))
