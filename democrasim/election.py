"""One election: sample voters, perceive stakes, vote, compare to welfare.

The core loop of the whole package:

1. The *truth*: a welfare metric ranks the policies on the full weighted
   electorate. This is the benchmark the election is graded against.
2. The *election*: a finite electorate is drawn (probability proportional to
   survey weight), each voter perceives their household's stakes through the
   perception model, and a voting rule aggregates ballots.
3. The *grade*: did the elected policy match the welfare-optimal one, and how
   much welfare was left on the table (regret)?

If no ballots are cast (possible under plurality when every sampled voter is
exactly indifferent), the status quo persists: realized welfare is 0.
"""

from dataclasses import dataclass, field

import numpy as np

from democrasim.electorate import Electorate, FloatArray
from democrasim.perception import PerceptionModel
from democrasim.voting import NO_WINNER, Plurality, Tally, VotingRule
from democrasim.welfare import Isoelastic, WelfareMetric


@dataclass(frozen=True)
class ElectionSpec:
    """Everything about an election except the electorate and the dice.

    Args:
        perception: How voters map true household impacts to beliefs.
        rule: How ballots aggregate into a winner.
        welfare: The metric defining which policy *should* win.
        n_voters: Size of the sampled electorate. ``None`` runs the election
            on the full weighted population (no sampling noise; perception
            noise is drawn once per row, which understates averaging within
            weight cells — prefer sampling for headline numbers).
    """

    perception: PerceptionModel
    rule: VotingRule = field(default_factory=Plurality)
    welfare: WelfareMetric = field(default_factory=Isoelastic)
    n_voters: int | None = 10_001


@dataclass(frozen=True)
class ElectionResult:
    """Outcome of one simulated election, graded against welfare.

    Attributes:
        winner: Winning policy index, or ``NO_WINNER`` (status quo).
        welfare_optimal: Policy index the welfare metric ranks highest.
        tracked: Whether the election selected the welfare-optimal policy.
        regret: Welfare of the optimal policy minus realized welfare
            (0 when tracked; welfare units of the chosen metric).
        welfare_by_policy: The metric's value per policy on the full
            electorate.
        tally: The vote-level outcome (shares, turnout).
    """

    winner: int
    welfare_optimal: int
    tracked: bool
    regret: float
    welfare_by_policy: FloatArray
    tally: Tally


def run_election(
    electorate: Electorate,
    spec: ElectionSpec,
    rng: np.random.Generator,
    *,
    welfare_by_policy: FloatArray | None = None,
) -> ElectionResult:
    """Simulate one election on ``electorate`` under ``spec``.

    ``welfare_by_policy`` may be precomputed (it depends only on the full
    electorate and metric, not on the dice) to avoid recomputing it across
    repeated elections.
    """
    if welfare_by_policy is None:
        welfare_by_policy = spec.welfare.per_policy(electorate)
    optimal = int(np.argmax(welfare_by_policy))

    voters = (
        electorate.sample(spec.n_voters, rng)
        if spec.n_voters is not None
        else electorate
    )
    perceived = spec.perception.perceive(voters, rng)
    tally = spec.rule.tally(perceived, voters.weights)

    realized = 0.0 if tally.winner == NO_WINNER else welfare_by_policy[tally.winner]
    return ElectionResult(
        winner=tally.winner,
        welfare_optimal=optimal,
        tracked=tally.winner == optimal,
        regret=float(welfare_by_policy[optimal] - realized),
        welfare_by_policy=welfare_by_policy,
        tally=tally,
    )
