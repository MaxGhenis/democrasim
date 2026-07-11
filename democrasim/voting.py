"""Voting rules: how perceived stakes aggregate into a winner.

Every rule consumes the same inputs — an ``(n_voters, n_policies)`` matrix of
*perceived* household impacts and a vote-mass weight per voter — so rules are
interchangeable in any experiment. Plurality is the headline rule; approval
and instant-runoff exist as the comparison surface for mechanism questions.

Conventions:

- Perceived impacts are relative to current law, so 0 is the status quo.
- Exact preference and winner ties are resolved toward the lowest policy
  index, deterministically. Instant-runoff elimination ties instead eliminate
  the lowest policy index. Under any perception model with continuous noise,
  ties have probability zero; the convention only matters in noiseless edge
  cases.
- Under plurality, a voter who perceives an exact tie between their best
  options abstains (strict ``>`` on the gap). This makes "no stake, no noise,
  no vote" the default rather than silently breaking ties toward one policy.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from democrasim.electorate import FloatArray

#: Sentinel winner index when no ballots are cast.
NO_WINNER = -1


@dataclass(frozen=True)
class Tally:
    """Outcome of one aggregation.

    Args:
        winner: Index of the winning policy, or :data:`NO_WINNER` if no
            ballots were cast.
        shares: Per-policy support as a share of total vote mass. For
            plurality and instant-runoff these are (first-round) vote shares;
            for approval they are approval rates and need not sum to 1.
        turnout: Share of total vote mass that cast a non-empty ballot.
    """

    winner: int
    shares: FloatArray
    turnout: float


@runtime_checkable
class VotingRule(Protocol):
    def tally(self, perceived: FloatArray, weights: FloatArray) -> Tally: ...


def _winner(counts: FloatArray) -> int:
    if counts.sum() <= 0:
        return NO_WINNER
    return int(np.argmax(counts))


@dataclass(frozen=True)
class Plurality:
    """One vote for the policy perceived best; most vote mass wins.

    Args:
        abstain_below: A voter abstains unless their perceived best policy
            beats the runner-up by strictly more than this many dollars.
            The default 0 means only exact perceived indifference abstains.
    """

    abstain_below: float = 0.0

    def tally(self, perceived: FloatArray, weights: FloatArray) -> Tally:
        n, p = perceived.shape
        order = np.argsort(perceived, axis=1)
        best = order[:, -1]
        gap = perceived[np.arange(n), best] - perceived[np.arange(n), order[:, -2]]
        votes = gap > self.abstain_below
        counts = np.bincount(best[votes], weights=weights[votes], minlength=p)
        total_mass = weights.sum()
        return Tally(
            winner=_winner(counts),
            shares=counts / total_mass,
            turnout=float(weights[votes].sum() / total_mass),
        )


@dataclass(frozen=True)
class Approval:
    """Approve every policy perceived better than the status quo.

    Args:
        threshold: Approval requires a perceived gain strictly above this
            many dollars per year (default 0: any perceived gain).
    """

    threshold: float = 0.0

    def tally(self, perceived: FloatArray, weights: FloatArray) -> Tally:
        approvals = perceived > self.threshold
        counts = approvals.T @ weights
        total_mass = weights.sum()
        return Tally(
            winner=_winner(counts),
            shares=counts / total_mass,
            turnout=float(weights[approvals.any(axis=1)].sum() / total_mass),
        )


@dataclass(frozen=True)
class InstantRunoff:
    """Full ranking by perceived impact; eliminate the weakest until majority.

    With two policies, instant runoff coincides with unthresholded plurality
    only in the absence of exact indifference. By default, a ballot whose
    perceived values are all equal force-votes for policy 0, preserving the
    historical behavior. Elimination ties eliminate the lowest policy index.

    Args:
        indifferent_abstain: Exclude ballots whose perceived values are all
            exactly equal from every round and from turnout. The default
            ``False`` retains force-voting for policy 0.
    """

    indifferent_abstain: bool = False

    def tally(self, perceived: FloatArray, weights: FloatArray) -> Tally:
        _, p = perceived.shape
        # rankings[i] = policy indices from most to least preferred.
        rankings = np.argsort(-perceived, axis=1, kind="stable")
        total_mass = weights.sum()
        if self.indifferent_abstain:
            participating = ~np.all(perceived == perceived[:, :1], axis=1)
            rankings = rankings[participating]
            ballot_weights = weights[participating]
        else:
            ballot_weights = weights
        participating_mass = ballot_weights.sum()
        if participating_mass <= 0:
            return Tally(
                winner=NO_WINNER,
                shares=np.zeros(p, dtype=np.float64),
                turnout=0.0,
            )

        alive = np.ones(p, dtype=bool)
        first_round_shares: FloatArray | None = None

        while True:
            # Each ballot counts for its highest-ranked live policy.
            # (n_ballots, p) — which ranked entries are live.
            live_mask = alive[rankings]
            top_pos = np.argmax(live_mask, axis=1)
            top_choice = rankings[np.arange(len(rankings)), top_pos]
            counts = np.bincount(top_choice, weights=ballot_weights, minlength=p)
            if first_round_shares is None:
                first_round_shares = counts / total_mass
            if counts.max() > participating_mass / 2 or alive.sum() <= 2:
                winner = int(np.argmax(np.where(alive, counts, -np.inf)))
                return Tally(
                    winner=winner,
                    shares=first_round_shares,
                    turnout=float(participating_mass / total_mass),
                )
            live_counts = np.where(alive, counts, np.inf)
            alive[int(np.argmin(live_counts))] = False
