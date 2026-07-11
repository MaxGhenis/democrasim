"""Experiment harness: the sweeps behind the headline results.

Three questions, inherited from the toy-era model and re-asked on measured
impact distributions:

1. :func:`accuracy_sweep` — as voters' perception accuracy rises, how often
   does the election select the welfare-optimal policy?
2. :func:`find_threshold` — how accurate do voters need to be before
   elections track welfare reliably?
3. :func:`bias_sweep` — how many dollars of systematic misperception does it
   take to flip the outcome?

Accuracy is reported on the ranking scale (probability a voter correctly
ranks the two policies for their own household). Caution when reading
sweeps on that axis: it is a property of the perception model *and* the
electorate's margin distribution, not a sufficient statistic for election
outcomes — a systematic bias toward the majority's true preference *raises*
mean ranking accuracy while destroying welfare tracking. Noise σ is the
primitive; accuracy is a derived coordinate.

Monte Carlo conventions: every grid point draws from its own RNG substream
(position-keyed off the sweep seed), so editing one grid point never
perturbs another. Tracking probabilities carry 95% Wilson intervals —
unlike the Wald standard error, these stay honest at p̂ ∈ {0, 1}.

:func:`analytic_plurality_curve` computes the same tracking probability in
closed form for plurality + linear-Gaussian perception at *any* electorate
size, including the large-population limit — the Monte Carlo results are
n-specific, and thresholds scale with √n (Condorcet jury logic), so no
single n's curve should be read as "the" answer.
"""

from collections.abc import Sequence
from dataclasses import replace
from math import erf, sqrt

import numpy as np
import pandas as pd

from democrasim.election import ElectionResult, ElectionSpec, run_election
from democrasim.electorate import Electorate
from democrasim.perception import LinearGaussianPerception, ranking_accuracy
from democrasim.voting import NO_WINNER
from democrasim.welfare import Isoelastic, WelfareMetric

#: z for the 95% Wilson score interval used across sweeps.
_WILSON_Z = 1.959963984540054


def _wilson_interval(p: float, n: int) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    z2 = _WILSON_Z**2
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = _WILSON_Z * sqrt(p * (1 - p) / n + z2 / (4 * n**2)) / (1 + z2 / n)
    return (max(0.0, center - half), min(1.0, center + half))


def _substream(seed: int, index: int) -> np.random.Generator:
    """Independent, position-keyed RNG substream for grid point ``index``."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(index,)))


def run_elections(
    electorate: Electorate,
    spec: ElectionSpec,
    n_elections: int,
    rng: np.random.Generator,
) -> list[ElectionResult]:
    """Run ``n_elections`` independent elections under one spec."""
    welfare_by_policy = spec.welfare.per_policy(electorate)
    return [
        run_election(electorate, spec, rng, welfare_by_policy=welfare_by_policy)
        for _ in range(n_elections)
    ]


def summarize_elections(results: Sequence[ElectionResult]) -> dict[str, float]:
    """Aggregate repeated elections into the quantities the sweeps report.

    ``p_tracked_lo``/``p_tracked_hi`` are the 95% Wilson interval — at
    0/240 or 240/240 the Wald standard error (also reported, for
    continuity) collapses to zero, which is exactly where honest
    uncertainty matters most.
    """
    n = len(results)
    tracked = np.array([r.tracked for r in results], dtype=float)
    p_tracked = float(tracked.mean())
    lo, hi = _wilson_interval(p_tracked, n)
    n_policies = len(results[0].welfare_by_policy)
    out: dict[str, float] = {
        "p_tracked": p_tracked,
        "p_tracked_lo": lo,
        "p_tracked_hi": hi,
        "p_tracked_se": float(np.sqrt(p_tracked * (1 - p_tracked) / n)),
        "mean_regret": float(np.mean([r.regret for r in results])),
        "mean_turnout": float(np.mean([r.tally.turnout for r in results])),
        "p_no_winner": float(np.mean([r.winner == NO_WINNER for r in results])),
    }
    for j in range(n_policies):
        out[f"p_win_{j}"] = float(np.mean([r.winner == j for r in results]))
    return out


def accuracy_sweep(
    electorate: Electorate,
    noise_sds: Sequence[float],
    *,
    spec: ElectionSpec | None = None,
    bias: float | tuple[float, ...] = 0.0,
    attenuation: float = 1.0,
    n_elections: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Sweep perception noise; report welfare tracking per noise level.

    ``spec`` provides the rule/welfare/electorate-size configuration; its
    perception model is replaced at every grid point by
    ``LinearGaussianPerception(noise_sd, bias, attenuation)``.
    """
    base = (
        spec
        if spec is not None
        else ElectionSpec(perception=LinearGaussianPerception())
    )
    rows = []
    for i, noise_sd in enumerate(noise_sds):
        perception = LinearGaussianPerception(
            noise_sd=float(noise_sd), bias=bias, attenuation=attenuation
        )
        results = run_elections(
            electorate,
            replace(base, perception=perception),
            n_elections,
            _substream(seed, i),
        )
        rows.append(
            {
                "noise_sd": float(noise_sd),
                "mean_ranking_accuracy": ranking_accuracy(perception, electorate),
                **summarize_elections(results),
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["policy_labels"] = electorate.policy_labels
    df.attrs["source"] = electorate.source
    return df


def bias_sweep(
    electorate: Electorate,
    bias_dollars: Sequence[float],
    *,
    toward: int = 1,
    noise_sd: float = 1_000.0,
    attenuation: float = 1.0,
    spec: ElectionSpec | None = None,
    n_elections: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Sweep a systematic bias favoring one policy, at fixed noise.

    Each grid point adds ``b`` perceived dollars per year to policy
    ``toward`` for every voter — the "everyone thinks this policy is worth
    $b more to them than it is" scenario.
    """
    if not 0 <= toward < electorate.n_policies:
        raise ValueError(
            f"toward={toward} is not a policy index of this electorate "
            f"(n_policies={electorate.n_policies})"
        )
    base = (
        spec
        if spec is not None
        else ElectionSpec(perception=LinearGaussianPerception())
    )
    rows = []
    for i, b in enumerate(bias_dollars):
        bias = tuple(
            float(b) if j == toward else 0.0 for j in range(electorate.n_policies)
        )
        perception = LinearGaussianPerception(
            noise_sd=noise_sd, bias=bias, attenuation=attenuation
        )
        results = run_elections(
            electorate,
            replace(base, perception=perception),
            n_elections,
            _substream(seed, i),
        )
        rows.append(
            {
                "bias_dollars": float(b),
                "mean_ranking_accuracy": ranking_accuracy(perception, electorate),
                **summarize_elections(results),
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["policy_labels"] = electorate.policy_labels
    df.attrs["source"] = electorate.source
    df.attrs["toward"] = toward
    df.attrs["noise_sd"] = noise_sd
    return df


def analytic_plurality_curve(
    electorate: Electorate,
    noise_sds: Sequence[float],
    *,
    n_voters: int | None = 10_001,
    bias: float | tuple[float, ...] = 0.0,
    attenuation: float = 1.0,
    welfare: WelfareMetric | None = None,
) -> pd.DataFrame:
    """Closed-form welfare tracking for plurality + linear-Gaussian voters.

    For a two-policy electorate under ``LinearGaussianPerception`` and
    ``Plurality()`` (zero abstention threshold), each sampled voter votes
    for policy 0 with probability ``Φ((k·mᵢ + b₀−b₁)/(σ√2))`` — so the
    weighted electorate reduces to one multinomial and the probability that
    the welfare-preferred policy wins has a closed form (normal
    approximation to the vote-count difference; exact in the limits).

    ``n_voters=None`` gives the infinite-electorate limit, where the winner
    is deterministic in the expected vote shares. Use this to see how the
    Monte Carlo results — which are specific to their electorate size —
    scale: tracking thresholds tighten toward 0.5 accuracy as n grows
    (jury-theorem √n logic), while expected-share crossings are
    n-invariant.
    """
    if electorate.n_policies != 2:
        raise ValueError("the analytic curve is defined for two policies")
    metric = welfare if welfare is not None else Isoelastic()
    welfare_by_policy = metric.per_policy(electorate)
    optimal = int(np.argmax(welfare_by_policy))

    weights = electorate.weights / electorate.weights.sum()
    margins = electorate.margins
    # Ranking accuracy is undefined when no voter has a stake in the choice.
    has_margins = bool(np.any(margins != 0))
    rows = []
    for noise_sd in noise_sds:
        model = LinearGaussianPerception(
            noise_sd=float(noise_sd), bias=bias, attenuation=attenuation
        )
        bias_vector = model._bias_vector(2)
        shifted = attenuation * margins + (bias_vector[0] - bias_vector[1])
        if noise_sd > 0:
            z = shifted / (noise_sd * sqrt(2.0))
            p_first = 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))
            p_abstain_each = np.zeros_like(p_first)
        else:
            p_first = np.where(shifted > 0, 1.0, np.where(shifted < 0, 0.0, 0.0))
            p_abstain_each = np.where(shifted == 0, 1.0, 0.0)

        share_first = float(weights @ p_first)
        share_abstain = float(weights @ p_abstain_each)
        share_second = 1.0 - share_first - share_abstain
        p_opt = share_first if optimal == 0 else share_second
        p_other = share_second if optimal == 0 else share_first

        gap = p_opt - p_other
        if n_voters is None:
            if p_opt + p_other == 0:
                p_tracked = 0.0  # everyone abstains; status quo persists
            else:
                p_tracked = 1.0 if gap > 0 else (0.5 if gap == 0 else 0.0)
        else:
            variance = n_voters * (p_opt + p_other - gap**2)
            if p_opt + p_other == 0:
                p_tracked = 0.0
            elif variance <= 0:
                p_tracked = 1.0 if gap > 0 else (0.5 if gap == 0 else 0.0)
            else:
                p_tracked = 0.5 * (1.0 + erf(n_voters * gap / sqrt(2.0 * variance)))
        rows.append(
            {
                "noise_sd": float(noise_sd),
                "mean_ranking_accuracy": (
                    ranking_accuracy(model, electorate) if has_margins else float("nan")
                ),
                "p_tracked": float(p_tracked),
                "expected_share_optimal": p_opt,
                "expected_share_other": p_other,
                "expected_share_abstain": share_abstain,
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["policy_labels"] = electorate.policy_labels
    df.attrs["source"] = electorate.source
    df.attrs["n_voters"] = n_voters
    df.attrs["analytic"] = True
    return df


def find_threshold(
    x: Sequence[float],
    y: Sequence[float],
    target: float = 0.9,
) -> float:
    """Smallest ``x`` at which ``y`` first reaches ``target``, interpolated.

    Points are sorted by ``x``; the crossing is linearly interpolated between
    the last point below and the first point at-or-above the target. Returns
    ``nan`` if ``y`` never reaches the target.
    """
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    reached = np.flatnonzero(ys >= target)
    if len(reached) == 0:
        return float("nan")
    i = reached[0]
    if i == 0:
        return float(xs[0])
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
