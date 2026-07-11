"""Experiment harness: the sweeps behind the headline results.

Three questions, inherited from the toy-era model and re-asked on measured
impact distributions:

1. :func:`accuracy_sweep` — as voters' perception accuracy rises, how often
   does the election select the welfare-optimal policy?
2. :func:`find_threshold` — how accurate do voters need to be before
   elections track welfare reliably?
3. :func:`bias_sweep` — how many dollars of systematic misperception does it
   take to flip the outcome?

Accuracy is reported on the survey-comparable ranking scale (probability a
voter correctly ranks the two policies for their own household), so sweeps
over different electorates — measured, moment-matched toy, homogeneous toy —
share an x-axis.
"""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from democrasim.election import ElectionResult, ElectionSpec, run_election
from democrasim.electorate import Electorate
from democrasim.perception import LinearGaussianPerception, ranking_accuracy
from democrasim.voting import NO_WINNER


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
    """Aggregate repeated elections into the quantities the sweeps report."""
    n = len(results)
    tracked = np.array([r.tracked for r in results], dtype=float)
    p_tracked = float(tracked.mean())
    n_policies = len(results[0].welfare_by_policy)
    out: dict[str, float] = {
        "p_tracked": p_tracked,
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
    rng = np.random.default_rng(seed)
    rows = []
    for noise_sd in noise_sds:
        perception = LinearGaussianPerception(
            noise_sd=float(noise_sd), bias=bias, attenuation=attenuation
        )
        results = run_elections(
            electorate, replace(base, perception=perception), n_elections, rng
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
    base = (
        spec
        if spec is not None
        else ElectionSpec(perception=LinearGaussianPerception())
    )
    rng = np.random.default_rng(seed)
    rows = []
    for b in bias_dollars:
        bias = tuple(
            float(b) if j == toward else 0.0 for j in range(electorate.n_policies)
        )
        perception = LinearGaussianPerception(
            noise_sd=noise_sd, bias=bias, attenuation=attenuation
        )
        results = run_elections(
            electorate, replace(base, perception=perception), n_elections, rng
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
